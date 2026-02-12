import os
import sys
import cv2
import numpy as np
import torch
import logging
from pathlib import Path
import shutil
from typing import Tuple, Optional
import warnings

# ONNX Runtime 경고 무시 (shape 정보 병합 경고는 기능에 영향 없음)
warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src', 'backend'))

try:
    import config
    # core와 fast_face_recognizer는 import하지 않음 (insightface 의존성 문제 방지)
    from ultralytics import YOLO
    # onnxruntime은 나중에 직접 import (conda 환경 호환성)
except ImportError as e:
    print(f"❌ 필수 모듈 임포트 실패: {e}")
    print("프로젝트 루트에서 실행해주세요.")
    sys.exit(1)

# 로깅 설정 (ONNX Runtime 경고 필터링)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# ONNX Runtime 로깅 레벨 조정
logging.getLogger('onnxruntime').setLevel(logging.ERROR)  # WARNING 이상만 표시

def main():
    print("="*60)
    print("🔄 얼굴 데이터베이스 재구축 (YOLOv8-Face + AdaFace)")
    print("="*60)

    # 1. 모델 로드
    print("1. 모델 로딩 중...")
    
    # YOLO Face 모델 (ONNX 우선)
    yolo_path = config.Paths.YOLO_FACE_MODEL
    onnx_path = os.path.splitext(yolo_path)[0] + ".onnx"
    
    yolo_face = None
    if os.path.exists(onnx_path):
        try:
            print(f"   - YOLO Face ONNX 모델 로딩: {onnx_path}")
            # ONNX 모델은 task="pose"로 변환되어 랜드마크 제공
            yolo_face = YOLO(onnx_path, task="pose")
            print(f"   ✅ YOLO Face ONNX 모델 로드 완료 (task=pose, 랜드마크 지원)")
        except Exception as e:
            print(f"   ⚠️ ONNX 모델 로드 실패, PyTorch로 대체: {e}")
            if os.path.exists(yolo_path):
                print(f"   - YOLO Face PyTorch 모델 로딩: {yolo_path}")
                # PyTorch 모델도 pose 구조이므로 task="pose" 사용
                yolo_face = YOLO(yolo_path, task="pose")
                print(f"   ✅ YOLO Face PyTorch 모델 로드 완료 (task=pose, 랜드마크 지원)")
            else:
                print(f"❌ YOLO Face 모델 없음: {yolo_path}")
                return
    elif os.path.exists(yolo_path):
        print(f"   - YOLO Face PyTorch 모델 로딩: {yolo_path}")
        # PyTorch 모델도 pose 구조이므로 task="pose" 사용
        yolo_face = YOLO(yolo_path, task="pose")
        print(f"   ✅ YOLO Face PyTorch 모델 로드 완료 (task=pose, 랜드마크 지원)")
    else:
        print(f"❌ YOLO Face 모델 없음: {yolo_path}")
        return
    
    # AdaFace 모델 (onnxruntime 직접 사용)
    adaface_path = config.Paths.ADAFACE_MODEL
    if not os.path.exists(adaface_path):
        print(f"❌ AdaFace 모델 없음: {adaface_path}")
        return

    # ONNX Runtime 세션 생성 (insightface 없이 직접 사용)
    # onnxruntime 모듈 확인 및 import (conda 환경 호환성)
    try:
        # 직접 onnxruntime import 시도
        import onnxruntime
        from onnxruntime import InferenceSession
        print(f"   ✅ onnxruntime 모듈 로드 완료 (버전: {getattr(onnxruntime, '__version__', 'N/A')})")
    except ImportError as e:
        print(f"❌ onnxruntime 모듈을 import할 수 없습니다: {e}")
        print(f"   재설치 방법:")
        print(f"   1. pip uninstall onnxruntime onnxruntime-gpu -y")
        print(f"   2. pip install onnxruntime-gpu")
        print(f"   또는 conda 환경인 경우:")
        print(f"   conda install -c conda-forge onnxruntime-gpu")
        return
    except Exception as e:
        print(f"❌ onnxruntime 모듈 로드 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return
    
    gpu_id = 0 if torch.cuda.is_available() else -1
    providers = []
    if gpu_id >= 0:
        try:
            providers.append(('CUDAExecutionProvider', {'device_id': gpu_id}))
        except:
            pass
    providers.append('CPUExecutionProvider')
    
    try:
        adaface_session = InferenceSession(adaface_path, providers=providers)
        adaface_input_name = adaface_session.get_inputs()[0].name
        adaface_output_name = adaface_session.get_outputs()[0].name
        print(f"   ✅ AdaFace ONNX 모델 로드 완료 (GPU: {gpu_id})")
    except Exception as e:
        print(f"❌ AdaFace 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. FAISS 초기화
    index_path = Path(config.Paths.FAISS_INDEX)
    labels_path = Path(config.Paths.FAISS_LABELS)
    
    if index_path.exists():
        backup_path = index_path.with_suffix('.faiss.bak')
        shutil.copy(index_path, backup_path)
        print(f"ℹ️ 기존 인덱스 백업됨")
    
    if labels_path.exists():
        backup_path = labels_path.with_suffix('.npy.bak')
        shutil.copy(labels_path, backup_path)
        print(f"ℹ️ 기존 레이블 백업됨")

    import faiss
    index = faiss.IndexFlatIP(512)
    labels = []
    
    # 3. 이미지 처리
    project_root = Path(config.BASE_DIR).parent.parent
    images_dir = project_root / 'face' / 'data' / 'images'
    
    if not images_dir.exists():
        images_dir = project_root / 'images'
        
    if not images_dir.exists():
        print(f"❌ 이미지 폴더 없음")
        return

    print(f"2. 이미지 처리 시작 (폴더: {images_dir})")
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(list(images_dir.rglob(ext)))
    
    print(f"ℹ️ 처리 대상 이미지 파일 수: {len(image_files)}")
    
    success_count = 0
    person_count = 0
    
    # 얼굴 정렬 및 임베딩 추출 함수 (insightface 없이 직접 구현)
    def align_face_simple(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """간단한 얼굴 정렬 (bbox 기반)"""
        try:
            # 랜드마크 중심과 크기 계산
            center_x = np.mean(kps[:, 0])
            center_y = np.mean(kps[:, 1])
            
            # 얼굴 크기 추정 (눈 사이 거리 기반)
            if len(kps) >= 2:
                eye_distance = np.linalg.norm(kps[1] - kps[0])
                face_size = int(eye_distance * 2.5)
            else:
                # 랜드마크가 부족하면 bbox 기반
                x_coords = kps[:, 0]
                y_coords = kps[:, 1]
                face_size = int((np.max(x_coords) - np.min(x_coords) + np.max(y_coords) - np.min(y_coords)) / 2)
            
            # 크롭 영역 계산
            x1 = max(0, int(center_x - face_size // 2))
            y1 = max(0, int(center_y - face_size // 2))
            x2 = min(frame.shape[1], int(center_x + face_size // 2))
            y2 = min(frame.shape[0], int(center_y + face_size // 2))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # 얼굴 크롭
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            
            # 112x112로 리사이즈 (AdaFace 입력 크기)
            aligned_face = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            return aligned_face
        except Exception as e:
            return None
    
    def get_embedding_from_onnx(frame: np.ndarray, kps: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """얼굴 정렬 및 임베딩 추출 (onnxruntime 직접 사용)"""
        try:
            # 얼굴 정렬
            aligned_face = align_face_simple(frame, kps)
            if aligned_face is None or aligned_face.size == 0:
                return None, None
            
            # 화질 개선: CLAHE
            try:
                lab = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                aligned_face = cv2.merge([l_channel, a, b])
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_LAB2BGR)
            except:
                pass
            
            # AdaFace 전처리
            np_img = aligned_face.astype(np.float32) / 255.0
            np_img = (np_img - 0.5) / 0.5  # [-1, 1] 정규화
            tensor = np_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
            
            # ONNX 추론
            outputs = adaface_session.run([adaface_output_name], {adaface_input_name: tensor})
            embedding = outputs[0]
            
            # Flatten
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # 정규화 (L2 norm)
            norm_val = np.linalg.norm(embedding)
            if norm_val > 0:
                embedding = embedding / norm_val
            else:
                return None, None
            
            return embedding, aligned_face
        except Exception as e:
            return None, None
    
    # 증강 함수 정의 (CCTV 환경 시뮬레이션)
    def apply_gaussian_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """가우시안 블러 (CCTV 모션 블러 시뮬레이션)"""
        if img is None or img.size == 0:
            return img
        try:
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        except Exception:
            return img
    
    def apply_downscale_upscale(img: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
        """저해상도 다운스케일 후 업스케일 (CCTV 저해상도 시뮬레이션)"""
        if img is None or img.size == 0:
            return img
        try:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            if new_h <= 0 or new_w <= 0: return img
            
            # 다운스케일
            downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # 업스케일 (원래 크기로 복원, 화질 손실 발생)
            upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
            return upscaled
        except Exception:
            return img
    
    def add_gaussian_noise(img: np.ndarray, mean: float = 0, std: float = 10) -> np.ndarray:
        """가우시안 노이즈 추가 (CCTV 압축 노이즈 시뮬레이션)"""
        if img is None or img.size == 0:
            return img
        try:
            noise = np.random.normal(mean, std, img.shape).astype(np.float32)
            noisy = img.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        except Exception:
            return img
    
    def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
        """밝기 조정 (factor < 1: 어둡게, CCTV 어두운 조명 시뮬레이션)"""
        if img is None or img.size == 0:
            return img
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception:
            return img
    
    def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
        """대비 조정 (factor < 1: 대비 감소, CCTV 화질 저하 시뮬레이션)"""
        if img is None or img.size == 0:
            return img
        try:
            return cv2.convertScaleAbs(img, alpha=factor, beta=0)
        except Exception:
            return img
    
    def apply_jpeg_compression(img: np.ndarray, quality: int = 60) -> np.ndarray:
        """JPEG 압축 시뮬레이션 (CCTV 압축 아티팩트)"""
        if img is None or img.size == 0:
            return img
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            if result:
                return cv2.imdecode(encimg, 1)
            return img
        except Exception:
            return img
    
    def reduce_saturation(img: np.ndarray, factor: float = 0.7) -> np.ndarray:
        """채도 감소 (CCTV 색상 저하 시뮬레이션)"""
        if img is None or img.size == 0:
            return img
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)  # 채도 감소
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception:
            return img
    
    def apply_strong_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """강한 가우시안 블러 (CCTV 모션 블러 강화)"""
        if img is None or img.size == 0:
            return img
        try:
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        except Exception:
            return img

    for img_path in image_files:
        try:
            # ID/이름 추출
            filename = img_path.stem
            parts = filename.split('_')
            parent_name = img_path.parent.name
            
            if parent_name != images_dir.name:
                person_id = parent_name
                name = parent_name
            else:
                if len(parts) >= 2:
                    person_id = parts[0]
                    name = parts[1]
                else:
                    person_id = filename
                    name = filename
                
            print(f"   Processing: {img_path.name} (ID: {person_id})", end=" -> ")
            
            # 이미지 로드
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"❌ 이미지 로드 실패")
                continue
            
            # YOLO 추론 (keypoints 포함, NMS timeout 증가)
            # ONNX 모델은 task='pose'로 변환되어 랜드마크 제공
            # 마스크 착용 이미지도 감지하기 위해 confidence threshold를 더 낮춤
            results = yolo_face(
                frame, 
                verbose=False,
                task='pose',  # pose task로 랜드마크 유지 (ONNX 변환 시 task="pose" 사용)
                conf=0.1,  # confidence threshold 더 낮춤 (마스크 착용 얼굴도 감지)
                iou=0.5,  # NMS IoU
                max_det=10,  # 최대 감지 수
                imgsz=832  # ONNX 모델 해상도
            )
            
            if not results:
                print(f"⚠️ 결과 없음")
                continue
            
            result = results[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                # 마스크 착용 이미지의 경우 얼굴 감지가 어려울 수 있음
                # 매우 낮은 threshold로 재시도
                results_retry = yolo_face(
                    frame,
                    verbose=False,
                    task='pose',
                    conf=0.05,  # 매우 낮은 threshold로 재시도 (마스크 착용 얼굴)
                    iou=0.5,
                    max_det=10,
                    imgsz=832
                )
                if results_retry and len(results_retry) > 0:
                    result_retry = results_retry[0]
                    if result_retry.boxes is not None and len(result_retry.boxes) > 0:
                        result = result_retry
                        print(f"   ✅ 낮은 threshold 재시도 성공")
                    else:
                        print(f"⚠️ 얼굴 없음 (마스크 착용 또는 얼굴이 가려진 이미지)")
                        continue
                else:
                    print(f"⚠️ 얼굴 없음 (마스크 착용 또는 얼굴이 가려진 이미지)")
                    continue
            
            # keypoints 확인
            if result.keypoints is None:
                print(f"⚠️ 키포인트 없음 (모델이 keypoints를 지원하지 않을 수 있음)")
                # keypoints 없이도 진행 (bbox만 사용)
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                if len(boxes_xyxy) > 0:
                    # 가장 큰 얼굴 사용
                    best_box = boxes_xyxy[0]
                    max_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                    for i, box in enumerate(boxes_xyxy[1:], 1):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        if area > max_area:
                            max_area = area
                            best_box = box
                            best_idx = i
                    
                    # bbox에서 랜드마크 추정 (간단한 방법)
                    x1, y1, x2, y2 = best_box
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    width, height = x2 - x1, y2 - y1
                    # 간단한 랜드마크 추정 (5개 포인트)
                    kps = np.array([
                        [center_x - width * 0.2, center_y - height * 0.1],  # left_eye
                        [center_x + width * 0.2, center_y - height * 0.1],  # right_eye
                        [center_x, center_y + height * 0.1],  # nose
                        [center_x - width * 0.15, center_y + height * 0.3],  # left_mouth
                        [center_x + width * 0.15, center_y + height * 0.3],  # right_mouth
                    ], dtype=np.float32)
                    print(f"⚠️ 키포인트 추정 사용 (bbox 기반)")
                else:
                    continue
            else:
                # 가장 큰 얼굴 찾기
                best_idx = -1
                max_area = 0
                
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes_xyxy):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > max_area:
                        max_area = area
                        best_idx = i
                
                if best_idx == -1:
                    continue
                    
                # 랜드마크 추출
                kps = None
                # (N, 5, 2) 형태
                all_kps = result.keypoints.xy.cpu().numpy()
                if len(all_kps) > best_idx:
                    kps = all_kps[best_idx]
                
                if kps is None or len(kps) < 5:
                    print(f"⚠️ 랜드마크 부족 (bbox 기반 추정 사용)")
                    # bbox에서 랜드마크 추정
                    box = boxes_xyxy[best_idx]
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    width, height = x2 - x1, y2 - y1
                    kps = np.array([
                        [center_x - width * 0.2, center_y - height * 0.1],
                        [center_x + width * 0.2, center_y - height * 0.1],
                        [center_x, center_y + height * 0.1],
                        [center_x - width * 0.15, center_y + height * 0.3],
                        [center_x + width * 0.15, center_y + height * 0.3],
                    ], dtype=np.float32)

            # --- TTA (Test Time Augmentation) 적용 ---
            # CCTV 저화질 환경 시뮬레이션: 고화질 원본 이미지를 저화질로 변환하여 실시간 환경과 매칭
            embeddings_to_add = []
            
            h, w = frame.shape[:2]
            
            # 1. 원본 이미지
            emb_orig, _ = get_embedding_from_onnx(frame, kps)
            if emb_orig is not None:
                embeddings_to_add.append(emb_orig)
            
            # 2. 좌우 반전
            frame_flip = cv2.flip(frame, 1)
            kps_flip = kps.copy()
            kps_flip[:, 0] = w - kps_flip[:, 0]
            kps_flip[[0, 1]] = kps_flip[[1, 0]]  # left_eye <-> right_eye
            kps_flip[[3, 4]] = kps_flip[[4, 3]]  # left_mouth <-> right_mouth
            emb_flip, _ = get_embedding_from_onnx(frame_flip, kps_flip)
            if emb_flip is not None:
                embeddings_to_add.append(emb_flip)
            
            # 3. 가우시안 블러 (CCTV 모션 블러 시뮬레이션)
            frame_blur = apply_gaussian_blur(frame, kernel_size=3)
            emb_blur, _ = get_embedding_from_onnx(frame_blur, kps)
            if emb_blur is not None:
                embeddings_to_add.append(emb_blur)
            
            # 4. 저해상도 다운스케일 후 업스케일 (CCTV 저해상도 시뮬레이션)
            frame_lowres = apply_downscale_upscale(frame, scale_factor=0.6)
            emb_lowres, _ = get_embedding_from_onnx(frame_lowres, kps)
            if emb_lowres is not None:
                embeddings_to_add.append(emb_lowres)
            
            # 5. 노이즈 추가 (CCTV 압축 노이즈 시뮬레이션)
            frame_noise = add_gaussian_noise(frame, mean=0, std=8)
            emb_noise, _ = get_embedding_from_onnx(frame_noise, kps)
            if emb_noise is not None:
                embeddings_to_add.append(emb_noise)
            
            # 6. 어두운 조명 (밝기 감소, CCTV 어두운 환경 시뮬레이션)
            frame_dark = adjust_brightness(frame, factor=0.7)
            emb_dark, _ = get_embedding_from_onnx(frame_dark, kps)
            if emb_dark is not None:
                embeddings_to_add.append(emb_dark)
            
            # 7. 대비 감소 (CCTV 화질 저하 시뮬레이션)
            frame_low_contrast = adjust_contrast(frame, factor=0.8)
            emb_low_contrast, _ = get_embedding_from_onnx(frame_low_contrast, kps)
            if emb_low_contrast is not None:
                embeddings_to_add.append(emb_low_contrast)
            
            # 8. 복합 저화질 (블러 + 저해상도 + 어두움)
            frame_composite = apply_gaussian_blur(frame, kernel_size=3)
            frame_composite = apply_downscale_upscale(frame_composite, scale_factor=0.7)
            frame_composite = adjust_brightness(frame_composite, factor=0.8)
            emb_composite, _ = get_embedding_from_onnx(frame_composite, kps)
            if emb_composite is not None:
                embeddings_to_add.append(emb_composite)
            
            # 9. JPEG 압축 시뮬레이션 (CCTV 압축 아티팩트)
            frame_jpeg = apply_jpeg_compression(frame, quality=55)
            emb_jpeg, _ = get_embedding_from_onnx(frame_jpeg, kps)
            if emb_jpeg is not None:
                embeddings_to_add.append(emb_jpeg)
            
            # 10. 채도 감소 (CCTV 색상 저하)
            frame_desat = reduce_saturation(frame, factor=0.6)
            emb_desat, _ = get_embedding_from_onnx(frame_desat, kps)
            if emb_desat is not None:
                embeddings_to_add.append(emb_desat)
            
            # 11. 강한 블러 (CCTV 모션 블러 강화)
            frame_strong_blur = apply_strong_blur(frame, kernel_size=5)
            emb_strong_blur, _ = get_embedding_from_onnx(frame_strong_blur, kps)
            if emb_strong_blur is not None:
                embeddings_to_add.append(emb_strong_blur)
            
            # 12. 극단적 저화질 (모든 효과 복합)
            frame_extreme = apply_strong_blur(frame, kernel_size=5)
            frame_extreme = apply_downscale_upscale(frame_extreme, scale_factor=0.5)
            frame_extreme = adjust_brightness(frame_extreme, factor=0.7)
            frame_extreme = reduce_saturation(frame_extreme, factor=0.6)
            frame_extreme = apply_jpeg_compression(frame_extreme, quality=50)
            frame_extreme = add_gaussian_noise(frame_extreme, mean=0, std=10)
            emb_extreme, _ = get_embedding_from_onnx(frame_extreme, kps)
            if emb_extreme is not None:
                embeddings_to_add.append(emb_extreme)
            
            # DB 추가
            if embeddings_to_add:
                for emb in embeddings_to_add:
                    index.add(np.array([emb], dtype=np.float32))
                    labels.append({'id': person_id, 'name': name})
                success_count += len(embeddings_to_add)
                person_count += 1
                print(f"✅ 등록 성공 ({len(embeddings_to_add)}개 벡터)")
            else:
                print(f"⚠️ 임베딩 추출 실패")
                
        except Exception as e:
            print(f"❌ 오류: {e}")

    # 4. 저장
    print("="*60)
    if success_count > 0:
        faiss.write_index(index, str(index_path))
        np.save(str(labels_path), labels)
        print(f"🎉 완료! 총 {person_count}명 ({success_count}개 벡터) 등록됨.")
    else:
        print("⚠️ 등록된 얼굴이 없습니다.")

if __name__ == "__main__":
    main()
