"""
AIVIS 시스템으로 동영상 처리 및 녹화
- PPE: 안전모, 안전조끼
- 넘어짐/쓰러짐 감지
- 얼굴 인식
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))

import cv2
import numpy as np
import time
from ultralytics import YOLO
import faiss
from tqdm import tqdm
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# 모델 경로
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
FACE_DIR = os.path.join(os.path.dirname(__file__), 'face', 'data')

# 한글 폰트 로드
def get_korean_font(size=20):
    """한글 폰트 로드"""
    font_paths = [
        "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
        "C:/Windows/Fonts/NanumGothic.ttf", # 나눔고딕
        "C:/Windows/Fonts/gulim.ttc",       # 굴림
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def draw_text_korean(img, text, pos, color, font_size=22, bg_color=None):
    """한글 텍스트 그리기"""
    # OpenCV -> PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_korean_font(font_size)
    
    x, y = pos
    
    # 텍스트 크기 계산
    bbox = draw.textbbox((x, y), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # 배경
    if bg_color:
        padding = 6
        draw.rectangle(
            [x - padding, y - padding, x + tw + padding, y + th + padding],
            fill=bg_color
        )
    
    # 텍스트
    draw.text((x, y), text, font=font, fill=color)
    
    # PIL -> OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    input_video = r"C:\Users\ihan\Desktop\aivis-project-mac-windows\동영상.mp4"
    output_video = r"C:\Users\ihan\Desktop\aivis-project-mac-windows\동영상_processed.mp4"
    
    print("=" * 60)
    print("AIVIS 동영상 처리 시작")
    print("=" * 60)
    
    # 동영상 열기
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ 동영상을 열 수 없습니다: {input_video}")
        return
    
    # 동영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 입력 동영상: {input_video}")
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {total_frames}")
    
    # 출력 동영상 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 모델 로드
    print("\n모델 로딩 중...")
    
    # PPE 모델
    violation_model = YOLO(os.path.join(MODEL_DIR, 'Yolo11n_PPE1.engine'), task='detect')
    print("✅ PPE 모델 로드 완료")
    
    # Pose 모델
    pose_model = YOLO(os.path.join(MODEL_DIR, 'yolo11n-pose.engine'), task='pose')
    print("✅ Pose 모델 로드 완료")
    
    # Fall 모델
    fall_model = YOLO(os.path.join(MODEL_DIR, 'yolov11n-fall.engine'), task='detect')
    print("✅ Fall 모델 로드 완료")
    
    # Face 모델
    face_model = YOLO(os.path.join(MODEL_DIR, 'yolov8n-face.engine'), task='pose')
    print("✅ Face 모델 로드 완료")
    
    # AdaFace
    from fast_face_recognizer import FastIndustrialRecognizer
    fast_rec = FastIndustrialRecognizer(
        model_path=os.path.join(MODEL_DIR, 'adaface_ir50_ms1mv2.engine'),
        ctx_id=0,
        use_adaface=True
    )
    print("✅ AdaFace 로드 완료")
    
    # FAISS
    index = faiss.read_index(os.path.join(FACE_DIR, 'face_index.faiss'))
    labels = np.load(os.path.join(FACE_DIR, 'face_index.faiss.labels.npy'), allow_pickle=True)
    print(f"✅ FAISS 로드 완료 ({index.ntotal}개 임베딩)")
    
    # 색상 정의 (RGB for PIL)
    COLOR_SAFE = (0, 200, 0)       # 녹색 - 안전
    COLOR_VIOLATION = (255, 0, 0)  # 빨강 - 위반
    COLOR_WARNING = (255, 165, 0)  # 주황 - 경고/Unknown
    COLOR_FALL = (255, 0, 0)       # 빨강 - 넘어짐
    
    # BGR for OpenCV
    BGR_SAFE = (0, 200, 0)
    BGR_VIOLATION = (0, 0, 255)
    BGR_WARNING = (0, 165, 255)
    BGR_FALL = (0, 0, 255)
    
    # PPE 클래스 매핑
    # 0: Hardhat, 1: Mask, 2: NO-Hardhat, 3: NO-Mask, 4: NO-Safety Vest, 5: Person, 6: Safety Cone, 7: Safety Vest
    
    print("\n처리 시작...")
    frame_count = 0
    fps_times = deque(maxlen=30)
    last_time = time.time()
    
    with tqdm(total=total_frames, desc="처리 중", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            fps_times.append(current_time - last_time)
            last_time = current_time
            
            result_frame = frame.copy()
            
            # 감지된 정보 저장
            all_detections = []  # (box, class_id, conf)
            person_boxes = []
            detected_persons = []  # 최종 렌더링용
            
            # === PPE 감지 ===
            try:
                ppe_results = violation_model(frame, conf=0.3, verbose=False)
                if ppe_results and len(ppe_results) > 0:
                    for result in ppe_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                all_detections.append((x1, y1, x2, y2, cls_id, conf))
                                
                                if cls_id == 5:  # Person
                                    person_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                pass
            
            # 각 Person에 대해 PPE 상태 분석
            for px1, py1, px2, py2 in person_boxes:
                person_info = {
                    'box': (px1, py1, px2, py2),
                    'has_hardhat': False,
                    'has_vest': False,
                    'no_hardhat': False,
                    'no_vest': False,
                    'is_fall': False,
                    'name': None,
                    'similarity': 0.0
                }
                
                # PPE 상태 확인
                for x1, y1, x2, y2, cls_id, conf in all_detections:
                    # 박스가 Person 안에 있는지 확인
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if px1 <= cx <= px2 and py1 <= cy <= py2:
                        if cls_id == 0:  # Hardhat
                            person_info['has_hardhat'] = True
                        elif cls_id == 7:  # Safety Vest
                            person_info['has_vest'] = True
                        elif cls_id == 2:  # NO-Hardhat
                            person_info['no_hardhat'] = True
                        elif cls_id == 4:  # NO-Safety Vest
                            person_info['no_vest'] = True
                
                detected_persons.append(person_info)
            
            # === 넘어짐 감지 (Fall 모델) ===
            try:
                fall_results = fall_model(frame, conf=0.5, verbose=False)
                if fall_results and len(fall_results) > 0:
                    for result in fall_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                w, h = x2 - x1, y2 - y1
                                
                                if w > h * 0.8:  # 넘어진 자세
                                    # 기존 person과 매칭
                                    matched = False
                                    for p in detected_persons:
                                        px1, py1, px2, py2 = p['box']
                                        # IoU 계산
                                        ix1, iy1 = max(x1, px1), max(y1, py1)
                                        ix2, iy2 = min(x2, px2), min(y2, py2)
                                        if ix1 < ix2 and iy1 < iy2:
                                            p['is_fall'] = True
                                            matched = True
                                            break
                                    
                                    if not matched:
                                        detected_persons.append({
                                            'box': (x1, y1, x2, y2),
                                            'has_hardhat': False,
                                            'has_vest': False,
                                            'no_hardhat': False,
                                            'no_vest': False,
                                            'is_fall': True,
                                            'name': None,
                                            'similarity': 0.0
                                        })
            except Exception as e:
                pass
            
            # === Pose 기반 넘어짐 감지 ===
            try:
                pose_results = pose_model(frame, conf=0.5, verbose=False)
                if pose_results and len(pose_results) > 0:
                    for result in pose_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                w, h = x2 - x1, y2 - y1
                                
                                if w > h * 1.3:  # 넘어진 자세
                                    for p in detected_persons:
                                        px1, py1, px2, py2 = p['box']
                                        ix1, iy1 = max(x1, px1), max(y1, py1)
                                        ix2, iy2 = min(x2, px2), min(y2, py2)
                                        if ix1 < ix2 and iy1 < iy2:
                                            p['is_fall'] = True
                                            break
            except Exception as e:
                pass
            
            # === 얼굴 감지 및 인식 ===
            try:
                face_results = face_model(frame, conf=0.4, verbose=False)  # conf 상향
                if face_results and len(face_results) > 0 and face_results[0].boxes is not None:
                    for i, box in enumerate(face_results[0].boxes):
                        fx1, fy1, fx2, fy2 = map(int, box.xyxy[0].tolist())
                        face_w, face_h = fx2 - fx1, fy2 - fy1
                        
                        # 얼굴 크기 필터링 (너무 작거나 큰 얼굴 제외)
                        if face_w < 40 or face_h < 40:
                            continue
                        if face_w > 300 or face_h > 400:
                            continue
                        
                        person_name = None
                        similarity = 0.0
                        
                        if face_results[0].keypoints is not None and len(face_results[0].keypoints.xy) > i:
                            kps = face_results[0].keypoints.xy[i][:5].cpu().numpy()
                            
                            # 키포인트 유효성 검사
                            valid_kps = np.sum(kps > 0) >= 8  # 최소 4개 포인트
                            if not valid_kps:
                                continue
                            
                            result_emb = fast_rec.get_embedding_fast(frame, kps)
                            if result_emb is not None:
                                emb, _ = result_emb
                                
                                query = emb.reshape(1, -1).astype('float32')
                                faiss.normalize_L2(query)
                                D, I = index.search(query, 5)  # Top-5로 확장
                                
                                best_sim = D[0][0]
                                best_idx = I[0][0]
                                second_sim = D[0][1] if len(D[0]) > 1 else 0
                                third_sim = D[0][2] if len(D[0]) > 2 else 0
                                
                                # 같은 사람 투표 (Top-5 중 동일 인물 수)
                                best_name = labels[best_idx]
                                vote_count = sum(1 for idx in I[0] if labels[idx] == best_name)
                                
                                # 강화된 매칭 조건
                                sim_gap = best_sim - second_sim
                                
                                # 조건 1: 높은 유사도 (0.75 이상)
                                if best_sim >= 0.75:
                                    person_name = best_name
                                    similarity = best_sim
                                # 조건 2: 중간 유사도 + 충분한 차이 + 투표
                                elif best_sim >= 0.65 and sim_gap >= 0.05 and vote_count >= 3:
                                    person_name = best_name
                                    similarity = best_sim
                                # 조건 3: 유사도 + 큰 차이
                                elif best_sim >= 0.60 and sim_gap >= 0.08:
                                    person_name = best_name
                                    similarity = best_sim
                        
                        # Person과 매칭
                        for p in detected_persons:
                            px1, py1, px2, py2 = p['box']
                            if fx1 >= px1 - 30 and fy1 >= py1 - 30 and fx2 <= px2 + 30:
                                if person_name:
                                    p['name'] = person_name
                                    p['similarity'] = similarity
                                break
            except Exception as e:
                pass
            
            # === 결과 렌더링 ===
            for person in detected_persons:
                x1, y1, x2, y2 = person['box']
                name = person['name']
                is_fall = person['is_fall']
                has_hardhat = person['has_hardhat']
                has_vest = person['has_vest']
                no_hardhat = person['no_hardhat']
                no_vest = person['no_vest']
                
                # 위반사항 체크
                violations = []
                if is_fall:
                    violations.append("넘어짐")
                if no_hardhat or (not has_hardhat):
                    violations.append("안전모 미착용")
                if no_vest or (not has_vest):
                    violations.append("안전조끼 미착용")
                
                # 모든 PPE 착용 시 안전
                is_safe = has_hardhat and has_vest and not is_fall
                
                # 색상 결정
                if is_fall:
                    box_color = BGR_FALL
                    text_color = COLOR_FALL
                elif violations:
                    box_color = BGR_VIOLATION
                    text_color = COLOR_VIOLATION
                elif is_safe:
                    box_color = BGR_SAFE
                    text_color = COLOR_SAFE
                else:
                    box_color = BGR_WARNING
                    text_color = COLOR_WARNING
                
                # 바운딩 박스
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), box_color, 3)
                
                # 라벨 생성
                if name:
                    if is_safe:
                        label = f"{name}: 안전"
                    elif violations:
                        label = f"{name}: {', '.join(violations)}"
                    else:
                        label = f"{name}"
                else:
                    if violations:
                        label = f"미확인: {', '.join(violations)}"
                    else:
                        label = "미확인"
                
                # 라벨 배경색 (BGR -> RGB)
                if is_fall:
                    bg_color = (255, 0, 0)
                elif violations:
                    bg_color = (255, 0, 0)
                elif is_safe:
                    bg_color = (0, 180, 0)
                else:
                    bg_color = (255, 140, 0)
                
                # 한글 텍스트 그리기
                result_frame = draw_text_korean(
                    result_frame, label, (x1, y1 - 30), 
                    (255, 255, 255), font_size=20, bg_color=bg_color
                )
            
            # === FPS 표시 (좌측 상단) ===
            if len(fps_times) > 0:
                avg_time = sum(fps_times) / len(fps_times)
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                fps_text = f"FPS: {current_fps:.1f}"
                result_frame = draw_text_korean(
                    result_frame, fps_text, (15, 15),
                    (0, 255, 0), font_size=24, bg_color=(0, 0, 0)
                )
            
            # === AIVIS 로고 (우측 하단) ===
            result_frame = draw_text_korean(
                result_frame, "AIVIS Safety System", (width - 250, height - 35),
                (0, 255, 0), font_size=20, bg_color=(0, 0, 0)
            )
            
            # 출력 동영상에 쓰기
            out.write(result_frame)
            pbar.update(1)
    
    # 정리
    cap.release()
    out.release()
    
    print("\n" + "=" * 60)
    print(f"✅ 처리 완료!")
    print(f"📹 출력 동영상: {output_video}")
    print(f"   처리된 프레임: {frame_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()