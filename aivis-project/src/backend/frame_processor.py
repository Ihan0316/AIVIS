# frame_processor.py - 프레임 처리 로직
"""
단일 프레임 처리 모듈
AI 모델을 사용하여 프레임을 처리하고 결과를 반환합니다.
"""
import logging
import time
from typing import Dict, Tuple, Any, List
from concurrent.futures import as_completed, TimeoutError as FuturesTimeoutError

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Keypoints

import utils
import config
from utils import find_best_match_faiss, draw_modern_bbox, draw_fast_bbox, calculate_iou_batch
from exceptions import (
    FaceRecognitionError
)
import state
from state import (
    safety_system_lock,
    frame_stats,
    frame_stats_lock,
    yolo_executor,
    face_recognition_executor,
    recent_identity_cache,
    embedding_buffers,
    EMBEDDING_BUFFER_SIZE,
    EMBEDDING_BUFFER_MIN_SIZE,
    MAX_EMBEDDING_BUFFERS_PER_CAM,
    fall_start_times,
    FALL_DURATION_THRESHOLD,
    centroid_cache,
    face_bbox_cache,
    last_render_cache,
    last_face_detection_frame,
    face_detection_lock,
    face_recognition_cooldown_ts
)
from ai_processors import (
    _process_ppe_detection,
    _process_face_recognition,
    _process_dangerous_behavior
)

# 마지막 렌더링된 프레임 캐시 (스킵 프레임에서 바운딩 박스 유지용)
_last_rendered_frames = {}  # {cam_id: (frame_bytes, result_dict)}

def render_frame_results(
    frame: np.ndarray,
    recognized_faces: List[Dict],
    violations: List[Dict],
    cam_id: int,
    orig_w: int,
    orig_h: int
) -> np.ndarray:
    """
    프레임에 AI 결과를 렌더링합니다.
    ⭐ PPE 모델의 person detection 결과(recognized_faces)를 기준으로 박스 표시
    recognized_faces에 이미 모든 정보(person_box, 이름, PPE 위반)가 포함되어 있음
    """
    processed_frame = frame.copy()
    
    # ⭐ recognized_faces만 사용 (PPE 모델의 person detection 기준)
    # recognized_faces의 box는 person_box입니다 (person_data_list에서 생성됨)
    all_boxes = []
    box_to_info = {}  # box_tuple -> (name, ppe_violations, is_violation)
    
    # 디버깅: recognized_faces 입력 확인
    if len(recognized_faces) > 1:
        logging.debug(f"[CAM-{cam_id}] render_frame_results 입력: recognized_faces={len(recognized_faces)}개, 박스 좌표={[face.get('box') for face in recognized_faces]}")
    
    # recognized_faces 처리: 모든 사람(안전한 사람 포함) 박스 표시
    # ⭐ 먼저 IoU 기반으로 중복 제거하여 중복된 박스가 all_boxes에 추가되지 않도록 함
    for face in recognized_faces:
        box = face.get("box") or face.get("bbox") or face.get("person_box")
        if box and len(box) == 4:
            box_tuple = tuple(map(int, box))
            name = face.get("name", "Unknown")
            ppe_violations = face.get("ppe_violations", [])
            # 마스크 제외
            filtered_ppe = [v for v in ppe_violations if v != "마스크"]
            is_violation = face.get("isViolation", False) or len(filtered_ppe) > 0
            
            # ⭐ IoU 기반 중복 체크: 이미 추가된 박스와 중복되면 병합
            is_duplicate = False
            for existing_box_tuple in all_boxes:
                iou = utils.calculate_iou(box_tuple, existing_box_tuple)
                if iou > 0.3:  # IoU 0.3 이상이면 같은 사람으로 간주하여 병합
                    is_duplicate = True
                    # 정보 병합: 더 정확한 정보 유지
                    existing_name, existing_ppe, existing_violation = box_to_info[existing_box_tuple]
                    # 이름은 "Unknown"이 아닌 것을 우선 사용
                    merged_name = name if name != "Unknown" else existing_name
                    # 위반 정보 병합
                    merged_ppe = list(set(existing_ppe + filtered_ppe))
                    merged_violation = existing_violation or is_violation
                    # 더 큰 박스로 업데이트
                    box_area = (box_tuple[2] - box_tuple[0]) * (box_tuple[3] - box_tuple[1])
                    existing_area = (existing_box_tuple[2] - existing_box_tuple[0]) * (existing_box_tuple[3] - existing_box_tuple[1])
                    if box_area > existing_area:
                        # 더 큰 박스로 교체
                        all_boxes.remove(existing_box_tuple)
                        all_boxes.append(box_tuple)
                        box_to_info[box_tuple] = (merged_name, merged_ppe, merged_violation)
                        if existing_box_tuple in box_to_info:
                            del box_to_info[existing_box_tuple]
                    else:
                        # 기존 박스 유지, 정보만 업데이트
                        box_to_info[existing_box_tuple] = (merged_name, merged_ppe, merged_violation)
                    break
            
            if not is_duplicate:
                all_boxes.append(box_tuple)
                box_to_info[box_tuple] = (name, filtered_ppe, is_violation)
    
    # ⚠️ violations 처리 제거됨: recognized_faces에 이미 모든 정보가 포함되어 있으므로 불필요
    
    # 디버깅: all_boxes 확인
    if len(all_boxes) > 1:
        logging.debug(f"[CAM-{cam_id}] render_frame_results all_boxes: {len(all_boxes)}개, 좌표={all_boxes}")
    
    # final_boxes는 all_boxes와 동일 (이미 IoU 기반 중복 제거 완료)
    final_boxes = all_boxes.copy()
    
    # 디버깅: 최종 박스 개수 확인
    if len(final_boxes) > 1:
        logging.warning(f"[CAM-{cam_id}] render_frame_results 최종 박스: {len(final_boxes)}개, 좌표={final_boxes}")
    
    # ⚠️ 좌표 스무딩 비활성화: 박스 크기 불안정 문제로 인해 비활성화
    # 박스가 나타났다 사라졌다 하는 문제는 좌표 스무딩 로직에서 이전 프레임과 매칭이 안 될 때 발생
    # 박스 크기가 person에 맞지 않는 문제와 함께 발생하므로, 좌표 스무딩은 비활성화하고
    # person detection 단계에서 박스 크기를 안정화하는 것이 더 중요
    # 좌표 스무딩 로직 비활성화 (박스 안정화를 위해)
    if False and cam_id in _last_rendered_frames:  # 비활성화됨
        _, last_result = _last_rendered_frames[cam_id]
        last_faces = last_result.get("recognized_faces", [])
        
        # 이전 프레임의 박스와 매칭하여 좌표 스무딩 (현재 프레임에 있는 박스만)
        for last_face in last_faces:
            last_box = last_face.get("box") or last_face.get("bbox")
            if not last_box or len(last_box) != 4:
                continue
            last_box_tuple = tuple(map(int, last_box))
            
            # 현재 박스와 매칭 (현재 프레임에 있는 박스만 스무딩)
            for i, current_box_tuple in enumerate(final_boxes):
                iou = utils.calculate_iou(last_box_tuple, current_box_tuple)
                if iou > 0.3:  # IoU 임계값 완화: 0.5 -> 0.3 (더 많은 박스 매칭)
                    # 좌표 스무딩 (95% 현재, 5% 이전) - 정확도 최우선, 미세한 떨림만 방지
                    smoothed_box = (
                        int(current_box_tuple[0] * 0.95 + last_box_tuple[0] * 0.05),
                        int(current_box_tuple[1] * 0.95 + last_box_tuple[1] * 0.05),
                        int(current_box_tuple[2] * 0.95 + last_box_tuple[2] * 0.05),
                        int(current_box_tuple[3] * 0.95 + last_box_tuple[3] * 0.05)
                    )
                    final_boxes[i] = smoothed_box
                    box_to_info[smoothed_box] = box_to_info.pop(current_box_tuple, box_to_info[current_box_tuple])
                    break
    
    # 렌더링
    renderer = utils.TextRenderer(frame.shape)
    for box_tuple in final_boxes:
        x1, y1, x2, y2 = box_tuple
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        if x2 > x1 and y2 > y1:
            name, ppe_violations, is_violation = box_to_info.get(box_tuple, ("Unknown", [], False))
            
            # 박스 크기 검증: 너무 작거나 이상한 박스는 제외
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            frame_area = orig_w * orig_h
            box_ratio = box_area / frame_area if frame_area > 0 else 0
            
            # 박스가 프레임의 0.5% 미만이면 제외 (너무 작은 잘못된 감지)
            # 모든 사람 박스를 표시하므로 임계값을 낮춤 (2% -> 0.5%)
            if box_ratio < 0.005:
                logging.debug(f"[CAM-{cam_id}] 너무 작은 박스 제외: 박스={box_tuple}, 비율={box_ratio:.3f}")
                continue
            
            # 모든 사람에게 박스 표시 (위반 여부 관계없이)
            # 마스크 제외한 위반만 체크
            filtered_violations = [v for v in ppe_violations if v != "마스크"]
            
            # 디버깅: 위반 정보 로깅
            if len(filtered_violations) > 0:
                logging.debug(f"[CAM-{cam_id}] 박스 색상 결정: 위반={filtered_violations}, 이름={name}, 박스={box_tuple}")
            
            # 색상 결정: 넘어짐 감지는 비활성화되어 있으므로 주황색만 사용
            if len(filtered_violations) > 0:
                # PPE 위반 또는 기타 위반 (넘어짐 제외, 이미 비활성화됨)
                unified_color = (0, 140, 255)  # 주황색 (PPE 위반)
                alpha = 0.2
            else:
                unified_color = (50, 255, 50)  # 초록색 (안전!)
                alpha = 0.15
            
            draw_modern_bbox(processed_frame, x1, y1, x2, y2, unified_color, thickness=3, corner_length=35, alpha=alpha)
            
            if name != "Unknown" or is_violation or len(ppe_violations) == 0:
                display_name = name if name != "Unknown" else "알 수 없음"
                violation_str = ""
                if ppe_violations:
                    # 마스크 제외
                    filtered_violations = [v for v in ppe_violations if v != "마스크"]
                    if not filtered_violations:
                        # 마스크만 있었으면 안전으로 처리
                        violation_str = "안전"
                    elif "넘어짐" in filtered_violations:
                        other_violations = [v for v in filtered_violations if v != "넘어짐"]
                        if other_violations:
                            violation_str = f"넘어짐! {', '.join(other_violations)} 미착용"
                        else:
                            violation_str = "넘어짐!"
                    else:
                        violation_str = f"{', '.join(filtered_violations)} 미착용"
                else:
                    # 위반 없으면 안전
                    violation_str = "안전"
                status_text = f"{display_name}: {violation_str}"
                # 디버깅: 실제 표시되는 텍스트 확인
                if name != "Unknown":
                    logging.debug(f"[CAM-{cam_id}] 라벨 표시: {status_text}")
                renderer.add_text(status_text, (x1, y1 - 10), unified_color)
    
    # 프론트엔드용 통합 박스 정보 생성 (render_frame_results 결과를 프론트엔드에 전달)
    unified_boxes = []
    for box_tuple in final_boxes:
        x1, y1, x2, y2 = box_tuple
        name, ppe_violations, is_violation = box_to_info.get(box_tuple, ("Unknown", [], False))
        # 마스크 제외
        filtered_violations = [v for v in ppe_violations if v != "마스크"]
        unified_boxes.append({
            "box": [x1, y1, x2, y2],
            "name": name,
            "ppe_violations": filtered_violations,
            "is_violation": is_violation
        })
    
    return renderer.render_on(processed_frame), unified_boxes

def process_single_frame(
    frame_bytes: bytes,
    cam_id: int
) -> Tuple[bytes, Dict[str, Any]]:
    """
    단일 프레임을 처리하고 결과를 반환합니다.
    
    Note: yolo_executor와 face_recognition_executor는 모듈 레벨에서 import되지만,
    함수 내에서 재할당되므로 global 선언이 필요합니다.
    
    Args:
        frame_bytes: 프레임 이미지 바이트 데이터
        cam_id: 카메라 ID
        
    Returns:
        Tuple[bytes, Dict[str, Any]]: 처리된 프레임 바이트와 결과 딕셔너리
    """
    # global 선언: executor 변수들이 함수 내에서 재할당될 수 있음
    global yolo_executor, face_recognition_executor
    
    # 성능 측정용 딕셔너리
    perf_timings = {
        'total': 0.0,
        'decode': 0.0,
        'resize': 0.0,
        'yolo_violation': 0.0,
        'yolo_pose': 0.0,
        'parse_results': 0.0,
        'face_recognition': 0.0,
        'rendering': 0.0,
        'encoding': 0.0
    }
    
    total_start = time.time()
    
    # SafetySystem 초기화 확인 및 에러 처리 개선
    # 전역 변수 안전하게 읽기 (멀티스레드 환경 대비 - 락 사용)
    # state 모듈을 직접 import하여 최신 값을 읽도록 수정
    with safety_system_lock:
        safety_system = state.safety_system_instance
    
    if safety_system is None:
        logging.warning(f"[CAM-{cam_id}] SafetySystem이 초기화되지 않았습니다. 초기화 완료 대기 중... (에러 프레임 반환)")
        # 에러 프레임 생성
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "System Initializing...", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        return buffer.tobytes(), {"error": "System not ready", "recognized_faces": [], "violations": []}
    
    # SafetySystem이 준비되었는지 확인
    if safety_system.violation_model is None or safety_system.pose_model is None:
        logging.warning(f"[CAM-{cam_id}] 필수 모델이 아직 준비되지 않았습니다. 에러 프레임 반환.")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Models Loading...", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        return buffer.tobytes(), {"error": "Models not ready", "recognized_faces": [], "violations": []}
    

    # 함수 내에서 orig_h, orig_w 기본값 설정 (오류 방지)
    orig_h, orig_w = 480, 640
    frame = None # 오류 발생 시 사용하기 위해 초기화

    try:
        # 1. 바이트를 이미지로 디코딩
        decode_start = time.time()
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        perf_timings['decode'] = (time.time() - decode_start) * 1000  # ms
        if frame is None:
            logging.warning(f"프레임 디코딩 실패 (CAM-{cam_id})")
            empty_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            return buffer.tobytes(), {}
        orig_h, orig_w = frame.shape[:2]
        
        # 프레임 유효성 검사: 크기가 너무 작거나 비어있는지 확인
        if orig_h < 100 or orig_w < 100:
            logging.warning(f"프레임 크기가 너무 작음: {orig_w}x{orig_h} (CAM-{cam_id})")
            empty_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            return buffer.tobytes(), {}
        
        # 검은 프레임 체크 (캠이 꺼진 상태): 평균 밝기가 매우 낮으면 AI 처리만 건너뜀
        # 기준: BGR 평균이 2.0 미만이면 거의 완전히 검은 프레임으로 간주
        frame_mean = np.mean(frame)
        if frame_mean < 2.0:  # 평균 밝기가 2 미만이면 검은 프레임으로 간주
            # 검은 프레임도 스트림에는 표시하되, 위반 감지는 하지 않음 (품질 최적화: 100 → 85)
            # 복사 최적화: 인코딩만 필요하므로 원본 프레임 직접 사용
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes(), {"recognized_faces": [], "violations": [], "violation_count": 0, "performance": {}}

        # 2. 모델 입력 크기에 맞게 리사이즈 (640x480 최적화, 성능 개선)
        # 모델 입력 해상도 (기본값: 640x480)
        target_width = config.SystemConfig.MODEL_INPUT_WIDTH
        target_height = config.SystemConfig.MODEL_INPUT_HEIGHT
        
        # 업스케일링 방지: 원본이 목표보다 작으면 다운스케일링만 수행
        # 성능 최적화: 업스케일링은 품질 저하와 성능 저하를 초래하므로 방지
        if orig_w > target_width or orig_h > target_height:
            # 다운스케일링만 수행 (업스케일링 방지)
            resize_start = time.time()
            
            # 원본 비율 계산
            orig_ratio = orig_w / orig_h
            target_ratio = target_width / target_height
            
            # 목표 해상도에 맞게 리사이즈 (비율 유지하면서 fit, 품질 최적화)
            if orig_ratio > target_ratio:
                # 원본이 더 넓은 경우: 너비 기준
                new_w = target_width
                new_h = int(target_width / orig_ratio)
            else:
                # 원본이 더 높은 경우: 높이 기준
                new_h = target_height
                new_w = int(target_height * orig_ratio)
            
            # 다운스케일링: INTER_AREA 사용 (품질 향상)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif orig_w == target_width and orig_h == target_height:
            # 이미 목표 해상도인 경우: 그대로 사용
            resized_frame = frame.copy()
        else:
            # 원본이 목표보다 작은 경우: 업스케일링 방지, 원본 그대로 사용
            resized_frame = frame.copy()
        
        resize_start = time.time() if 'resize_start' not in locals() else resize_start
        
        # 바운딩 박스 좌표 변환을 위한 스케일 계산 (리사이즈된 크기 기준)
        # 정확한 스케일 계산: 원본 크기 / 리사이즈된 크기
        resized_w = resized_frame.shape[1]
        resized_h = resized_frame.shape[0]
        w_scale = orig_w / resized_w
        h_scale = orig_h / resized_h
        
        perf_timings['resize'] = (time.time() - resize_start) * 1000  # ms

        # 3. 처리된 프레임 생성 (원본 프레임 복사)
        # 최적화: 렌더링에만 사용되므로 view 사용 고려, 하지만 안정성을 위해 복사 유지
        processed_frame = frame.copy()
        renderer = utils.TextRenderer(frame.shape)

        # 4. 모든 모델을 병렬로 실행 (개별 실행 후 결과만 합치기)
        model_start = time.time()
        

        # SafetySystem은 이미 위에서 락으로 읽었으므로 재사용 (1013번 줄 제거)
        # safety_system 변수는 이미 929번 줄에서 설정됨
        
        # GPU 최적화 설정 (half precision, 배치 처리 등)
        # MPS는 half precision을 지원하지 않으므로 CUDA만 사용
        base_half_precision = config.SystemConfig.ENABLE_HALF_PRECISION and 'cuda' in str(safety_system.device)
        
        # MPS 최적화: 메모리 정리 (주기적으로 가비지 컬렉션)
        # cam_id별 상태 관리 사용 (함수 속성 대신)
        frame_state = state.get_frame_processing_state(cam_id)
        if 'mps' in str(safety_system.device):
            import gc
            # 50프레임마다 가비지 컬렉션 (100 -> 50, MPS 환경 메모리 관리 개선)
            frame_state['frame_count'] = frame_state.get('frame_count', 0) + 1
            if frame_state['frame_count'] % 50 == 0:
                gc.collect()
                # MPS 메모리 동기화 (통합 메모리 최적화)
                # torch는 모듈 레벨에서 import되었으므로 안전하게 사용 가능
                try:
                    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                except (AttributeError, NameError):
                    pass  # MPS가 없거나 empty_cache가 없는 경우 무시
        
        # YOLO 모델 입력 크기 최적화 (속도 향상)
        # ONNX 모델은 고정된 입력 크기(832x832)를 사용하므로 명시적으로 설정
        # ONNX 모델인지 확인 (모델 파일 경로 확인)
        import os
        violation_is_onnx = False
        pose_is_onnx = False
        
        # 모델 경로 확인 (여러 방법 시도)
        try:
            # 방법 1: ckpt_path 속성 확인
            if hasattr(safety_system.violation_model, 'ckpt_path'):
                violation_model_path = str(safety_system.violation_model.ckpt_path)
                violation_is_onnx = violation_model_path.endswith('.onnx')
            # 방법 2: 모델 파일 경로 직접 확인
            if not violation_is_onnx:
                violation_model_path = config.Paths.YOLO_VIOLATION_MODEL
                onnx_path = os.path.splitext(violation_model_path)[0] + ".onnx"
                violation_is_onnx = os.path.exists(onnx_path)
        except:
            pass
        
        try:
            if hasattr(safety_system.pose_model, 'ckpt_path'):
                pose_model_path = str(safety_system.pose_model.ckpt_path)
                pose_is_onnx = pose_model_path.endswith('.onnx')
            if not pose_is_onnx:
                pose_model_path = config.Paths.YOLO_POSE_MODEL
                onnx_path = os.path.splitext(pose_model_path)[0] + ".onnx"
                pose_is_onnx = os.path.exists(onnx_path)
        except:
            pass
        
        # ONNX 모델인 경우 변환 시 사용한 해상도 사용 (config.ONNX_MODEL_SIZE)
        if violation_is_onnx or pose_is_onnx:
            model_imgsz = config.SystemConfig.ONNX_MODEL_SIZE  # ONNX 모델 변환 시 사용한 해상도
        else:
            max_input_size = max(config.SystemConfig.MODEL_INPUT_WIDTH, config.SystemConfig.MODEL_INPUT_HEIGHT)
            model_imgsz = max_input_size  # 정수로 전달하면 정사각형으로 처리
        
        violation_kwargs = {'conf': config.Thresholds.YOLO_CONFIDENCE}
        pose_kwargs = {'conf': config.Thresholds.YOLO_CONFIDENCE}
        # TensorRT 최적화: iou와 max_det 설정 (TensorRT는 더 빠르게 처리 가능)
        # 5명까지 감지되도록 NMS IoU 완화 (0.5 -> 0.6)
        violation_kwargs.update({
            'verbose': False,
            'iou': 0.6,  # NMS IoU 임계값 (5명 감지 확장: 0.5 -> 0.6)
            'max_det': 100,  # 최대 감지 수 (TensorRT 최적화: 50 -> 100, 더 많은 객체 감지 가능)
        })
        pose_kwargs.update({
            'verbose': False,
            'iou': 0.6,  # NMS IoU 임계값 (5명 감지 확장: 0.5 -> 0.6)
            'max_det': 100,  # 최대 감지 수 (TensorRT 최적화)
        })
        
        if not safety_system.violation_uses_trt:
            violation_kwargs.update({
                'device': safety_system.device,
                'half': base_half_precision,
                'imgsz': model_imgsz,  # ONNX 모델인 경우 832, 그렇지 않으면 config 값
            })
        if not safety_system.pose_uses_trt:
            pose_kwargs.update({
                'device': safety_system.device,
                'half': base_half_precision,
                'imgsz': model_imgsz,  # ONNX 모델인 경우 832, 그렇지 않으면 config 값
            })
        
        # 얼굴 인식 모델 및 DB 가져오기 (병렬 실행 준비)
        face_model = safety_system.face_model
        face_analyzer = safety_system.face_analyzer
        face_database = safety_system.face_database
        
        # 얼굴 탐지 간격 체크 (병렬 실행 전에 확인)
        should_detect_faces_global = True
        with face_detection_lock:
            current_frame = frame_stats.get(cam_id, {}).get('frame_count', 0)
            last_frame = last_face_detection_frame.get(cam_id, -config.Thresholds.FACE_DETECTION_INTERVAL)
            if current_frame - last_frame < config.Thresholds.FACE_DETECTION_INTERVAL:
                should_detect_faces_global = False
        
        # 모든 모델을 병렬로 실행 (개별 실행 후 결과만 합치기)
        # GPU 메모리 정리는 필요시에만 (매 프레임은 오버헤드)
        # 50 프레임마다 한 번씩 정리 (100 -> 50, 메모리 관리 개선)
        # torch는 이미 모듈 레벨에서 import되었으므로 재import 불필요
        # cam_id별 상태 관리 사용
        frame_state = state.get_frame_processing_state(cam_id)
        frame_count = frame_state.get('frame_count', 0)
        if 'cuda' in str(safety_system.device) and frame_count % 50 == 0:
            # 모든 GPU의 메모리 정리
            try:
                for gpu_id in range(torch.cuda.device_count()):
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.debug(f"[CAM-{cam_id}] GPU 메모리 정리 중 오류 (무시): {e}")
        
        # GPU 최고 성능 설정 (멀티 GPU 지원)
        # 실시간 스트리밍에서는 배치 처리가 오히려 지연을 유발하므로 배치 파라미터 제거
        if 'cuda' in str(safety_system.device):
            if not safety_system.violation_uses_trt:
                violation_kwargs.update({
                    'half': True,  # Half precision 활성화 (GPU 성능 향상)
                    'agnostic_nms': False
                    # 배치 파라미터 제거: 실시간 처리에서는 즉시 처리 (배치=1)가 가장 빠름
                })
            if not safety_system.pose_uses_trt:
                pose_kwargs.update({
                    'half': True,  # Half precision 활성화 (GPU 성능 향상)
                    'agnostic_nms': False
                    # 배치 파라미터 제거: 실시간 처리에서는 즉시 처리 (배치=1)가 가장 빠름
                })
        else:
            if not safety_system.violation_uses_trt:
                violation_kwargs['half'] = False
            if not safety_system.pose_uses_trt:
                pose_kwargs['half'] = False
        
        # 람다 함수 오버헤드 제거: 직접 함수 호출로 최적화
        # 성능 최적화: resized_frame 사용 (이미 최적 크기로 리사이즈됨)
        # YOLO가 imgsz 파라미터로 추가 리사이즈를 처리하므로 resized_frame 사용이 더 빠름
        def run_violation_model():
            model_start = time.time()
            input_shape = resized_frame.shape[:2] if resized_frame is not None else (0, 0)
            logging.debug(f"[MODEL CAM-{cam_id}] YOLO Violation 모델 호출 시작: 입력 크기={input_shape}, kwargs={violation_kwargs}")
            try:
                result = safety_system.violation_model(resized_frame, **violation_kwargs)
                elapsed = (time.time() - model_start) * 1000
                # 결과 요약
                detections = 0
                if result and len(result) > 0:
                    boxes = result[0].boxes if hasattr(result[0], 'boxes') else None
                    if boxes is not None:
                        detections = len(boxes)
                logging.info(f"[MODEL CAM-{cam_id}] ✅ YOLO Violation 모델 완료: {elapsed:.1f}ms, 감지={detections}개")
                return result
            except Exception as e:
                elapsed = (time.time() - model_start) * 1000
                logging.error(f"[MODEL CAM-{cam_id}] ❌ YOLO Violation 모델 오류: {elapsed:.1f}ms, {e}", exc_info=True)
                raise
        
        def run_pose_model():
            model_start = time.time()
            input_shape = resized_frame.shape[:2] if resized_frame is not None else (0, 0)
            logging.debug(f"[MODEL CAM-{cam_id}] YOLO Pose 모델 호출 시작: 입력 크기={input_shape}, kwargs={pose_kwargs}")
            try:
                result = safety_system.pose_model(resized_frame, **pose_kwargs)
                elapsed = (time.time() - model_start) * 1000
                # 결과 요약
                detections = 0
                if result and len(result) > 0:
                    boxes = result[0].boxes if hasattr(result[0], 'boxes') else None
                    if boxes is not None:
                        detections = len(boxes)
                logging.info(f"[MODEL CAM-{cam_id}] ✅ YOLO Pose 모델 완료: {elapsed:.1f}ms, 감지={detections}개")
                return result
            except Exception as e:
                elapsed = (time.time() - model_start) * 1000
                logging.error(f"[MODEL CAM-{cam_id}] ❌ YOLO Pose 모델 오류: {elapsed:.1f}ms, {e}", exc_info=True)
                raise
        
        # ⭐ PPE(Violation)와 위험 감지(Pose) 모델을 병렬로 실행 (ThreadPoolExecutor)
        
        # Executor가 shutdown되었는지 확인하고, shutdown되었으면 재생성 (개선된 버전)
        max_retries = 2
        retry_count = 0
        violation_future = None
        pose_future = None
        
        while retry_count < max_retries:
            try:
                # Executor 상태 확인 (락으로 안전하게)
                executor_valid = False
                try:
                    if hasattr(yolo_executor, '_shutdown') and not yolo_executor._shutdown:
                        executor_valid = True
                    elif not hasattr(yolo_executor, '_shutdown'):
                        # _shutdown 속성이 없으면 유효한 것으로 간주
                        executor_valid = True
                except (AttributeError, RuntimeError):
                    executor_valid = False
                
                if not executor_valid:
                    logging.warning(f"⚠️ [CAM-{cam_id}] YOLO Executor가 shutdown되었습니다. 재생성합니다.")
                    from state import update_worker_executors
                    update_worker_executors()
                    # 재시도
                    from state import yolo_executor as new_yolo_executor
                    yolo_executor = new_yolo_executor
                
                violation_future = yolo_executor.submit(run_violation_model)  # PPE 위반 감지 모델 (병렬)
                pose_future = yolo_executor.submit(run_pose_model)  # 위험 행동 감지 모델 (병렬)
                break  # 성공하면 루프 종료
                
            except RuntimeError as e:
                if "shutdown" in str(e).lower() and retry_count < max_retries - 1:
                    # Executor가 shutdown된 경우 재생성
                    logging.warning(f"⚠️ [CAM-{cam_id}] Executor shutdown 오류 감지, 재생성합니다 (재시도 {retry_count + 1}/{max_retries}): {e}")
                    from state import update_worker_executors
                    update_worker_executors()
                    # 재시도
                    from state import yolo_executor as new_yolo_executor
                    yolo_executor = new_yolo_executor
                    retry_count += 1
                else:
                    logging.error(f"❌ [CAM-{cam_id}] Executor 재생성 실패 (최대 재시도 횟수 초과): {e}", exc_info=True)
                    raise
            except Exception as e:
                logging.error(f"❌ [CAM-{cam_id}] Executor 제출 중 예상치 못한 오류: {e}", exc_info=True)
                raise
        
        if violation_future is None or pose_future is None:
            raise RuntimeError(f"[CAM-{cam_id}] Executor 제출 실패: violation_future={violation_future}, pose_future={pose_future}")
        
        # 얼굴 감지도 병렬로 실행 (YOLO 얼굴 모델 사용, 전체 프레임에서)
        face_detection_future = None
        if face_model is not None and should_detect_faces_global:
            face_kwargs = {
                'conf': config.Thresholds.FACE_DETECTION_CONFIDENCE,
                'imgsz': config.Thresholds.FACE_DETECTION_SIZE
            }
            # TensorRT 모델도 iou, max_det 설정 적용 (NMS 최적화)
            # 주의: TensorRT 엔진은 빌드 시 설정을 사용하므로, 런타임 파라미터는 제한적입니다
            if safety_system.face_uses_trt:
                face_kwargs.update({
                    'verbose': False,
                    'iou': 0.7,  # TensorRT 최적화: 0.6 -> 0.7 (더 빠른 NMS, 더 적은 박스)
                    'max_det': 5,  # TensorRT 최적화: 10 -> 5 (NMS 속도 대폭 향상, 얼굴은 보통 1-3개)
                })
            else:
                face_kwargs['verbose'] = False
                face_kwargs['device'] = safety_system.device_face  # GPU 1 사용 (멀티 GPU인 경우)
            
            def run_face_model():
                model_start = time.time()
                input_shape = frame.shape[:2] if frame is not None else (0, 0)
                logging.debug(f"[MODEL CAM-{cam_id}] YOLO Face 모델 호출 시작: 입력 크기={input_shape}, kwargs={face_kwargs}")
                try:
                    result = face_model(frame, **face_kwargs)
                    elapsed = (time.time() - model_start) * 1000
                    # 결과 요약
                    detections = 0
                    if result and len(result) > 0:
                        boxes = result[0].boxes if hasattr(result[0], 'boxes') else None
                        if boxes is not None:
                            detections = len(boxes)
                    logging.info(f"[MODEL CAM-{cam_id}] ✅ YOLO Face 모델 완료: {elapsed:.1f}ms, 감지={detections}개")
                    return result
                except Exception as e:
                    elapsed = (time.time() - model_start) * 1000
                    logging.error(f"[MODEL CAM-{cam_id}] ❌ YOLO Face 모델 오류: {elapsed:.1f}ms, {e}", exc_info=True)
                    raise
            
            # Executor가 shutdown되었는지 확인
            try:
                if face_recognition_executor._shutdown:
                    logging.warning(f"⚠️ Face Executor가 shutdown되었습니다. 재생성합니다.")
                    from state import update_worker_executors
                    update_worker_executors()
                    from state import face_recognition_executor as new_face_executor
                    face_recognition_executor = new_face_executor
            except AttributeError:
                pass  # _shutdown 속성이 없는 경우 무시
            
            try:
                face_detection_future = face_recognition_executor.submit(run_face_model)
            except RuntimeError as e:
                if "shutdown" in str(e).lower():
                    # Executor가 shutdown된 경우 재생성
                    logging.warning(f"⚠️ Face Executor shutdown 오류 감지, 재생성합니다: {e}")
                    from state import update_worker_executors
                    update_worker_executors()
                    from state import face_recognition_executor as new_face_executor
                    face_recognition_executor = new_face_executor
                    face_detection_future = face_recognition_executor.submit(run_face_model)
                else:
                    raise
        
        # 모든 모델 결과 대기 (병렬 실행, 타임아웃 최적화: 실시간 처리 속도 향상)
        # GPU 환경에서는 첫 실행 시 warmup이 필요하므로 타임아웃 증가
        # 첫 실행 후에는 빠르게 처리되므로 타임아웃을 동적으로 조정
        # cam_id별 상태 관리 사용 (함수 속성 대신)
        if not frame_state.get('model_warmed_up', False):
            # 첫 실행: warmup을 위해 타임아웃 증가
            model_timeout = 5.0 if 'cuda' in str(safety_system.device) else 6.0
            frame_state['model_warmed_up'] = True
        else:
            # 이후 실행: 정상 처리 속도
            model_timeout = 3.0 if 'cuda' in str(safety_system.device) else 4.0
        try:
            violation_results = violation_future.result(timeout=model_timeout)
        except FuturesTimeoutError:
            logging.warning(f"[CAM-{cam_id}] YOLO violation 모델 타임아웃 ({model_timeout}s)")
            violation_results = []
        except Exception as e:
            logging.error(f"[CAM-{cam_id}] YOLO violation 모델 실행 오류: {e}", exc_info=True)
            violation_results = []
        
        try:
            pose_results = pose_future.result(timeout=model_timeout)
            if not (pose_results and len(pose_results) > 0):
                logging.warning(f"[CAM-{cam_id}] ⚠️ YOLO Pose 모델 실행 완료: 결과 없음 (pose_results={pose_results}, len={len(pose_results) if pose_results else 0})")
        except FuturesTimeoutError:
            logging.error(f"[CAM-{cam_id}] ❌ YOLO pose 모델 타임아웃 ({model_timeout}s)")
            pose_results = []
        except Exception as e:
            logging.error(f"[CAM-{cam_id}] ❌ YOLO pose 모델 실행 오류: {e}", exc_info=True)
            pose_results = []
        
        # YOLO 얼굴 감지 결과 처리
        yolo_face_results = None
        if face_detection_future is not None:
            try:
                yolo_face_results = face_detection_future.result(timeout=model_timeout)
            except (FuturesTimeoutError, Exception):
                yolo_face_results = None
        
        # 병렬 실행 시간 측정
        model_total_time = time.time() - model_start
        perf_timings['yolo_violation'] = model_total_time * 1000  # ms (병렬 실행 시간)
        perf_timings['yolo_pose'] = model_total_time * 1000  # ms (병렬 실행 시간)
        if face_detection_future is not None:
            perf_timings['face_recognition'] = model_total_time * 1000  # ms (병렬 실행 시간)

        # 5. 모든 모델 결과 합치기 (병렬 실행 결과를 통합)
        parse_start = time.time()
        
        # 5-1. YOLO violation 결과 파싱
        all_detections = {}
        if violation_results and len(violation_results) > 0:
            violation_box_count = len(violation_results[0].boxes) if violation_results[0].boxes is not None else 0
            logging.debug(f"[PARSE CAM-{cam_id}] YOLO Violation 결과 파싱 시작: 총 박스={violation_box_count}개")
            
            filtered_count = 0
            low_conf_count = 0
            class_counts = {}  # 클래스별 감지 수
            
            for det in violation_results[0].boxes:
                class_id = int(det.cls[0])
                class_name = safety_system.violation_model.names[class_id]
                conf = float(det.conf[0])
                
                # 클래스별 카운트
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                # Safety Con 등 오탐지 클래스 필터링
                if class_name in config.Thresholds.IGNORED_CLASSES:
                    filtered_count += 1
                    logging.debug(f"[PARSE CAM-{cam_id}] 필터링된 클래스: {class_name} (conf={conf:.3f})")
                    continue
                
                if conf >= config.Thresholds.YOLO_CONFIDENCE:
                    # 리사이즈된 프레임 기준 좌표를 원본 프레임 크기로 스케일링
                    bbox_resized = det.xyxy[0].cpu().numpy()
                    bbox_original = bbox_resized * np.array([w_scale, h_scale, w_scale, h_scale])
                    bbox_clipped = utils.clip_bbox_xyxy(bbox_original, orig_w, orig_h)
                    if bbox_clipped is not None:
                        if class_name not in all_detections:
                            all_detections[class_name] = []
                        # clip_bbox_xyxy는 tuple을 반환하므로 list()로 변환
                        all_detections[class_name].append({'bbox': list(bbox_clipped), 'conf': conf})
                        logging.debug(f"[PARSE CAM-{cam_id}] 감지: {class_name} conf={conf:.3f}, bbox={bbox_clipped}")
                else:
                    low_conf_count += 1
            
            # 파싱 결과 요약 로그
            total_valid = sum(len(dets) for dets in all_detections.values())
            logging.info(f"[PARSE CAM-{cam_id}] ✅ YOLO Violation 파싱 완료: "
                        f"유효={total_valid}개, 필터={filtered_count}개, 저신뢰={low_conf_count}개, "
                        f"클래스별={class_counts}")
            
        # 5-2. 얼굴 감지 결과 처리 (YOLO 결과를 InsightFace 형식으로 변환)
        recognized_faces = []
        violations_found = []
        face_detected_boxes = []  # 얼굴 기반 박스 (뒤에 있는 사람용)
        
        # YOLO 얼굴 감지 결과를 InsightFace 형식으로 변환
        faces_in_frame = []
        if yolo_face_results and len(yolo_face_results) > 0:
            result = yolo_face_results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                total_face_boxes = len(result.boxes)
                valid_face_count = 0
                logging.debug(f"[PARSE CAM-{cam_id}] YOLO Face 결과 파싱 시작: 총 박스={total_face_boxes}개")
                # Keypoints 전체 추출 (있으면)
                all_keypoints = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        all_keypoints = result.keypoints.xy.cpu().numpy()
                    except Exception as e:
                        pass  # Keypoints 변환 실패는 무시

                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0])
                    if conf >= config.Thresholds.FACE_DETECTION_CONFIDENCE:
                        bbox = box.xyxy[0].cpu().numpy()
                        fx1, fy1, fx2, fy2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        
                        # 스케일링 적용 (리사이즈된 프레임 기준이므로 원본 크기로 변환)
                        fx1 = int(fx1 * w_scale)
                        fy1 = int(fy1 * h_scale)
                        fx2 = int(fx2 * w_scale)
                        fy2 = int(fy2 * h_scale)
                        
                        # 프레임 경계 내로 클리핑
                        fx1 = max(0, min(fx1, orig_w))
                        fy1 = max(0, min(fy1, orig_h))
                        fx2 = max(0, min(fx2, orig_w))
                        fy2 = max(0, min(fy2, orig_h))
                        
                        # YOLO keypoints 추출 (5개 랜드마크) 및 스케일링
                        kps = None
                        if all_keypoints is not None and len(all_keypoints) > i:
                            try:
                                kps = all_keypoints[i].copy() # (5, 2)
                                # 스케일링 적용
                                kps[:, 0] *= w_scale
                                kps[:, 1] *= h_scale
                            except Exception as e:
                                pass  # 개별 Keypoints 처리 실패는 무시
                                kps = None

                        # 간단한 얼굴 객체 생성 (bbox, det_score, kps 속성)
                        class SimpleFace:
                            def __init__(self, bbox, det_score, kps=None):
                                self.bbox = bbox
                                self.det_score = det_score
                                self.kps = kps
                        
                        face_obj = SimpleFace([fx1, fy1, fx2, fy2], conf, kps)
                        faces_in_frame.append(face_obj)
                        
                        # 얼굴 박스를 사람 박스로 확장 (얼굴 크기의 3-4배)
                        face_w = fx2 - fx1
                        face_h = fy2 - fy1
                        
                        # 최소 얼굴 크기 필터링 완화 (더 작은 얼굴도 감지)
                        min_face_size = config.Thresholds.MIN_FACE_SIZE  # 16픽셀 이상
                        if face_w < min_face_size or face_h < min_face_size:
                            # 너무 작은 얼굴은 건너뛰지만, 로깅은 하지 않음 (노이즈 방지)
                            continue
                        
                        # 얼굴 중심점
                        face_cx = (fx1 + fx2) / 2
                        face_cy = (fy1 + fy2) / 2
                        # 얼굴 크기의 3.0배로 확장 (사람 전체 포함하도록 증가)
                        expanded_w = face_w * 3.0  # 2.5 -> 3.0
                        expanded_h = face_h * 3.5  # 2.5 -> 3.5 (세로 더 확장)
                        # 확장된 박스 (얼굴이 상단 중앙에 위치, 사람 전체 포함)
                        expanded_x1 = max(0, int(face_cx - expanded_w / 2))
                        expanded_y1 = max(0, int(face_cy - face_h * 0.5))  # 0.2 -> 0.5 (머리 위로 더 확장)
                        expanded_x2 = min(orig_w, int(face_cx + expanded_w / 2))
                        expanded_y2 = min(orig_h, int(face_cy + expanded_h * 1.0))  # 0.6 -> 1.0 (하체까지 충분히 포함)
                        
                        # 유효한 박스인지 확인
                        if expanded_x2 > expanded_x1 and expanded_y2 > expanded_y1:
                            face_detected_boxes.append({
                                'box': (expanded_x1, expanded_y1, expanded_x2, expanded_y2),
                                'face_bbox': (fx1, fy1, fx2, fy2),
                                'face': face_obj,
                                'confidence': conf
                            })
                            valid_face_count += 1
                            logging.debug(f"[PARSE CAM-{cam_id}] 얼굴 감지: conf={conf:.3f}, "
                                        f"face_bbox=({fx1},{fy1},{fx2},{fy2}), "
                                        f"expanded_box=({expanded_x1},{expanded_y1},{expanded_x2},{expanded_y2}), "
                                        f"keypoints={'있음' if kps is not None else '없음'}")
                
                logging.info(f"[PARSE CAM-{cam_id}] ✅ YOLO Face 파싱 완료: "
                           f"총={total_face_boxes}개, 유효={valid_face_count}개, "
                           f"최소 신뢰도={config.Thresholds.FACE_DETECTION_CONFIDENCE}")
        
        perf_timings['parse_results'] = (time.time() - parse_start) * 1000  # ms

        # 7. 사람 감지 및 상태 확인
        if pose_results and pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
            boxes = pose_results[0].boxes.xyxy.cpu().numpy()

            # 중복 사람 박스 제거 (NMS 유사) - 겹침이 큰 박스는 큰 박스 하나만 유지
            # 최적화: 거리 기반 필터링 먼저 수행하여 불필요한 IoU 계산 방지
            try:
                if boxes is not None and len(boxes) > 1:
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    order = np.argsort(-areas)  # 큰 박스 우선
                    keep_indices = []
                    suppressed = np.zeros(len(boxes), dtype=bool)
                    
                    for idx in order:
                        if suppressed[idx]:
                            continue
                        keep_indices.append(idx)
                        x1, y1, x2, y2 = boxes[idx]
                        box_center_x = (x1 + x2) / 2
                        box_center_y = (y1 + y2) / 2
                        box_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        max_distance = box_diagonal * 1.2  # 박스 대각선의 1.2배 이내만 고려
                        
                        for j in order:
                            if j == idx or suppressed[j]:
                                continue
                            
                            # 거리 기반 필터링 먼저 수행 (IoU 계산보다 빠름)
                            jx1, jy1, jx2, jy2 = boxes[j]
                            j_center_x = (jx1 + jx2) / 2
                            j_center_y = (jy1 + jy2) / 2
                            center_distance = ((box_center_x - j_center_x) ** 2 + (box_center_y - j_center_y) ** 2) ** 0.5
                            
                            # 거리가 너무 멀면 IoU 계산 생략 (성능 향상)
                            if center_distance > max_distance:
                                continue
                            
                            # IoU 계산 (거리 필터링 통과한 경우만)
                            iou = utils.calculate_iou((x1, y1, x2, y2), tuple(boxes[j]))
                            if iou > 0.8:  # 높은 겹침은 중복으로 간주 (5명 감지 확장: 0.7 -> 0.8)
                                suppressed[j] = True
                    
                    boxes = boxes[keep_indices]
                    if pose_results[0].keypoints is not None:
                        keypoints_list = [pose_results[0].keypoints[i] for i in keep_indices]
                    else:
                        keypoints_list = None
                else:
                    # boxes가 1개 이하인 경우 원본 유지
                    if pose_results[0].keypoints is not None:
                        keypoints_list = list(pose_results[0].keypoints) if hasattr(pose_results[0].keypoints, '__iter__') else None
                    else:
                        keypoints_list = None
            except Exception as e:
                # 예외 발생 시 로깅하고 원본 사용 (boxes와 keypoints_list 동기화 유지)
                logging.warning(f"[CAM-{cam_id}] 중복 박스 제거 중 오류 발생: {e}", exc_info=True)
                # boxes와 keypoints_list를 동기화하여 유지
                if pose_results[0].keypoints is not None:
                    keypoints_list = list(pose_results[0].keypoints) if hasattr(pose_results[0].keypoints, '__iter__') else None
                else:
                    keypoints_list = None
            
            # boxes와 keypoints_list 동기화 확인 (안전장치)
            if not isinstance(keypoints_list, list) and keypoints_list is not None:
                # keypoints_list가 리스트가 아니면 변환
                if hasattr(keypoints_list, '__iter__'):
                    keypoints_list = list(keypoints_list)
                else:
                    keypoints_list = None
            
            # boxes와 keypoints_list 길이 확인 (동기화 검증)
            if isinstance(keypoints_list, list) and len(boxes) != len(keypoints_list):
                logging.warning(f"[CAM-{cam_id}] ⚠️ boxes({len(boxes)})와 keypoints_list({len(keypoints_list)}) 길이 불일치! keypoints_list를 None으로 설정")
                keypoints_list = None
            confidences = pose_results[0].boxes.conf.cpu().numpy() if pose_results[0].boxes.conf is not None else None

            # 사람 박스 좌표를 원본 프레임 크기로 스케일링 및 필터링
            scaled_person_boxes = []
            valid_person_indices = []  # 유효한 사람 박스 인덱스
            filtered_boxes = []
            filtered_keypoints = []
            filtered_confidences = []

            # 필터링 전 박스 개수 로그
            initial_box_count = len(boxes)
            logging.info(f"[PARSE CAM-{cam_id}] Pose 박스 필터링 시작: 초기 박스={initial_box_count}개")
            
            # 필터링 단계별 제거 통계
            filter_stats = {
                'clipped_none': 0,
                'zero_size': 0,
                'zero_aspect': 0,
                'min_size': 0,
                'aspect_ratio': 0,
                'min_keypoints': 0,
                'keypoint_structure': 0,
                'upper_body_ratio': 0,
                'keypoint_confidence': 0,
                'keypoint_spread': 0,
                'keypoint_position': 0,
                'final': 0
            }

            for i, box in enumerate(boxes):
                scaled_box_np = box * np.array([w_scale, h_scale, w_scale, h_scale])
                clipped_box = utils.clip_bbox_xyxy(scaled_box_np, orig_w, orig_h)
                if clipped_box is None:
                    continue

                # ⭐ YOLO person detection 박스에 패딩 추가 (사람 전체 포함)
                x1, y1, x2, y2 = clipped_box
                box_w = x2 - x1
                box_h = y2 - y1
                # 박스 크기의 10% 패딩 추가 (위/아래 더 많이)
                padding_w = box_w * 0.1
                padding_h_top = box_h * 0.15  # 위로 15% 더 확장 (머리 포함)
                padding_h_bottom = box_h * 0.1  # 아래로 10% 더 확장
                
                # 패딩이 적용된 박스 계산
                padded_x1 = max(0, int(x1 - padding_w))
                padded_y1 = max(0, int(y1 - padding_h_top))
                padded_x2 = min(orig_w, int(x2 + padding_w))
                padded_y2 = min(orig_h, int(y2 + padding_h_bottom))
                
                original_box = (padded_x1, padded_y1, padded_x2, padded_y2)  # 패딩 적용된 박스
                x1, y1, x2, y2 = original_box
                box_w = x2 - x1
                box_h = y2 - y1
                
                # 박스 크기가 0이면 건너뛰기 (버그 방지)
                if box_w <= 0 or box_h <= 0:
                    filter_stats['zero_size'] += 1
                    continue
                
                box_area = box_w * box_h
                aspect_ratio = box_w / box_h if box_h > 0 else 0
                
                # aspect_ratio가 0이면 종횡비 필터링에서 문제가 될 수 있으므로 조기 종료
                if aspect_ratio == 0:
                    filter_stats['zero_aspect'] += 1
                    continue

                # 1. 키포인트 확인 및 박스 조정 (키포인트 기반으로 박스를 더 정확하게 조정)
                # 멀리 있는 사람도 감지하기 위해 완화 조건 완화
                num_valid_kpts = 0
                has_head_or_shoulders = False
                refined_box = None
                if isinstance(keypoints_list, list) and i < len(keypoints_list):
                    keypoints = keypoints_list[i]
                    if keypoints is not None and keypoints.conf is not None:
                        conf_arr = keypoints.conf[0].cpu().numpy()
                        valid_kpts_mask = conf_arr > config.Thresholds.POSE_CONFIDENCE
                        num_valid_kpts = int(np.sum(valid_kpts_mask))
                        # nose(0), left_shoulder(5), right_shoulder(6)
                        idxs = [0, 5, 6]
                        for idx in idxs:
                            if idx < len(valid_kpts_mask) and valid_kpts_mask[idx]:
                                has_head_or_shoulders = True
                                break
                        
                        # 키포인트 기반으로 박스 조정 (여러 사람 분리 개선)
                        # ⚠️ 박스 크기 안정화: 패딩 비율 감소하여 박스 크기 변동성 감소
                        if num_valid_kpts >= 4:  # 충분한 키포인트가 있을 때만 조정
                            refined_box = utils.refine_box_from_keypoints(
                                keypoints, original_box, orig_w, orig_h, padding_ratio=0.15  # 0.3 -> 0.15 (안정화)
                            )
                            if refined_box is not None:
                                # 조정된 박스 사용
                                x1, y1, x2, y2 = refined_box
                                box_w = x2 - x1
                                box_h = y2 - y1
                                
                                # 조정된 박스 크기가 0이면 원본 박스 사용
                                if box_w <= 0 or box_h <= 0:
                                    x1, y1, x2, y2 = original_box
                                    box_w = x2 - x1
                                    box_h = y2 - y1
                                
                                box_area = box_w * box_h
                                aspect_ratio = box_w / box_h if box_h > 0 else 0
                                
                                # aspect_ratio가 0이면 원본 박스로 복원
                                if aspect_ratio == 0:
                                    x1, y1, x2, y2 = original_box
                                    box_w = x2 - x1
                                    box_h = y2 - y1
                                    box_area = box_w * box_h
                                    aspect_ratio = box_w / box_h if box_h > 0 else 0

                # 오탐지 방지: 키포인트 검증 강화
                # 완화 조건을 더 엄격하게 적용하여 의자/책상 등 오탐지 방지
                # 최소 6개 키포인트와 머리/어깨가 있어야 완화 조건 적용
                use_relaxed = (num_valid_kpts >= 6 and has_head_or_shoulders) and (box_area < 5000)
                
                # 넘어진 사람 감지: 바운딩 박스 종횡비 기반 단순 판단
                # 가로가 세로보다 1.5배 이상이면 넘어짐으로 판단
                is_fallen_person = False
                if aspect_ratio >= config.Thresholds.FALL_ASPECT_RATIO:  # 1.5
                    # 추가 검증: 박스가 너무 작지 않아야 함 (오탐 방지)
                    if box_area >= 2000:  # 최소 면적 필터
                        is_fallen_person = True
                        logging.debug(f"🔍 [CAM-{cam_id}] 넘어진 사람 감지: 가로={box_w:.1f}, 세로={box_h:.1f}, "
                                    f"비율={aspect_ratio:.2f}, 면적={box_area:.0f}")
                
                min_w = config.Thresholds.RELAXED_MIN_PERSON_BOX_WIDTH if use_relaxed else config.Thresholds.MIN_PERSON_BOX_WIDTH
                min_h = config.Thresholds.RELAXED_MIN_PERSON_BOX_HEIGHT if use_relaxed else config.Thresholds.MIN_PERSON_BOX_HEIGHT
                min_area = config.Thresholds.RELAXED_MIN_PERSON_BOX_AREA if use_relaxed else config.Thresholds.MIN_PERSON_BOX_AREA
                max_ar = config.Thresholds.RELAXED_MAX_PERSON_ASPECT_RATIO if use_relaxed else config.Thresholds.MAX_PERSON_ASPECT_RATIO
                min_ar = config.Thresholds.RELAXED_MIN_PERSON_ASPECT_RATIO if use_relaxed else config.Thresholds.MIN_PERSON_ASPECT_RATIO
                
                # 넘어진 사람의 경우 필터링 완화 (하지만 더 엄격한 검증 후에만)
                if is_fallen_person:
                    # 종횡비 제한 완화 (넘어진 사람은 가로가 세로보다 훨씬 길 수 있음)
                    max_ar = max(max_ar, 5.0)  # 최대 5.0까지 허용
                    # 최소 면적 완화 (넘어진 사람은 면적이 작을 수 있음)
                    min_area = min(min_area, 1500)

                # 2. 최소 크기 필터링 (너무 작은 박스는 제외)
                if box_w < min_w or box_h < min_h or box_area < min_area:
                    filter_stats['min_size'] += 1
                    logging.debug(f"🔍 [CAM-{cam_id}] P{i} 최소 크기 필터링: w={box_w:.1f}<{min_w} or h={box_h:.1f}<{min_h} or area={box_area:.0f}<{min_area}")
                    continue

                # 3. 종횡비 필터링 (손처럼 세로로 긴 것 또는 너무 가로로 긴 것 제외)
                if aspect_ratio > max_ar or aspect_ratio < min_ar:
                    filter_stats['aspect_ratio'] += 1
                    logging.debug(f"🔍 [CAM-{cam_id}] P{i} 종횡비 필터링: {aspect_ratio:.2f} (범위: {min_ar:.2f}~{max_ar:.2f})")
                    continue

                # 4. 키포인트 검증 (사람 감지율 향상을 위해 완화)
                # 키포인트가 없는 경우: 박스 크기와 종횡비만 만족하면 통과 (YOLO Pose 모델이 감지했으므로)
                if num_valid_kpts == 0:
                    # 키포인트가 없어도 박스가 충분히 크고 적절한 종횡비를 가지면 통과
                    # (YOLO Pose 모델이 사람으로 감지했으므로)
                    if box_area >= 5000 and 0.3 <= aspect_ratio <= 3.0:
                        logging.debug(f"🔍 [CAM-{cam_id}] P{i} 키포인트 없음, 하지만 박스 크기/종횡비로 통과: area={box_area:.0f}, ratio={aspect_ratio:.2f}")
                        # 키포인트 검증 건너뛰고 다음 단계로
                        pass
                    else:
                        filter_stats['min_keypoints'] += 1
                        logging.debug(f"🔍 [CAM-{cam_id}] P{i} 키포인트 없음 + 박스 크기/종횡비 불만족: area={box_area:.0f}, ratio={aspect_ratio:.2f}")
                        continue
                else:
                    # 키포인트가 있는 경우: 최소 키포인트 개수 검증
                    if not has_head_or_shoulders:
                        min_kpts_required = 3  # 4 -> 3 (더 많은 사람 감지)
                    else:
                        min_kpts_required = 3 if use_relaxed else 3  # 4개 -> 3개 (더 많은 사람 감지)
                    
                    # 넘어진 사람의 경우 키포인트 요구사항 완화 (하지만 최소 3개는 필요)
                    if is_fallen_person:
                        min_kpts_required = max(3, min_kpts_required - 1)  # 최소 3개 이상 요구
                    
                    if num_valid_kpts < min_kpts_required:
                        filter_stats['min_keypoints'] += 1
                        logging.debug(f"🔍 [CAM-{cam_id}] P{i} 키포인트 개수 부족: {num_valid_kpts} < {min_kpts_required} (머리/어깨={has_head_or_shoulders})")
                        continue
                
                # 5. 추가 검증: 키포인트 분포 확인 (오탐지 방지)
                # 키포인트가 있는 경우에만 실행
                upper_body_ratio = 0.0
                keypoint_spread_ok = True
                if num_valid_kpts > 0 and isinstance(keypoints_list, list) and i < len(keypoints_list):
                    keypoints = keypoints_list[i]
                    if keypoints is not None and keypoints.conf is not None:
                        conf_arr = keypoints.conf[0].cpu().numpy()
                        points = keypoints.xy[0].cpu().numpy()
                        
                        # 상체 키포인트 인덱스: nose(0), eyes(1,2), ears(3,4), shoulders(5,6), elbows(7,8)
                        upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                        upper_body_valid = sum(1 for idx in upper_body_indices if idx < len(conf_arr) and conf_arr[idx] > config.Thresholds.POSE_CONFIDENCE)
                        upper_body_ratio = upper_body_valid / len(upper_body_indices) if len(upper_body_indices) > 0 else 0
                        
                        # 하체 키포인트 인덱스: hips(11,12), knees(13,14), ankles(15,16)
                        lower_body_indices = [11, 12, 13, 14, 15, 16]
                        lower_body_valid = sum(1 for idx in lower_body_indices if idx < len(conf_arr) and conf_arr[idx] > config.Thresholds.POSE_CONFIDENCE)
                        
                        # 키포인트 구조 검증 (사람 감지율 향상을 위해 완화)
                        # 재킷/가방 같은 객체는 상체 키포인트만 있고 하체 키포인트가 없을 가능성이 높음
                        # 사람은 상체 또는 하체 키포인트 중 하나만 있어도 통과 (더 많은 사람 감지)
                        has_upper_body = upper_body_valid >= 1  # 최소 1개 이상의 상체 키포인트 (2 -> 1, 더 많은 사람 감지)
                        has_lower_body = lower_body_valid >= 1  # 최소 1개 이상의 하체 키포인트 (2 -> 1, 더 많은 사람 감지)
                        
                        if not is_fallen_person:
                            # 서있는 사람은 상체 또는 하체 키포인트 중 하나만 있어도 통과 (더 많은 사람 감지)
                            if not (has_upper_body or has_lower_body):  # and -> or (더 많은 사람 감지)
                                filter_stats['keypoint_structure'] += 1
                                logging.debug(f"🔍 [CAM-{cam_id}] P{i} 키포인트 구조 검증 실패: 상체={upper_body_valid}개, 하체={lower_body_valid}개 (최소 각 1개 필요)")
                                continue
                        
                        # 넘어진 사람의 경우 상체 키포인트 비율 검증 완화
                        # 서있는 사람도 상체 키포인트가 부족할 수 있으므로 완화
                        # 똑바로 서있는 사람은 상체 키포인트가 더 적을 수 있으므로 완화
                        # 측면으로 서있는 사람은 일부 키포인트가 가려질 수 있으므로 더 완화
                        # 사람 감지율 향상: 상체 비율 임계값 완화 (0.10 -> 0.05)
                        upper_body_threshold = 0.05 if is_fallen_person else 0.05  # 서있는 사람: 0.05 (더 많은 사람 감지)
                        if upper_body_ratio < upper_body_threshold:
                            # 디버깅: 상체 키포인트 비율이 부족한 경우 로그
                            filter_stats['upper_body_ratio'] += 1
                            logging.debug(f"🔍 [CAM-{cam_id}] P{i} 상체 키포인트 비율 부족: {upper_body_ratio:.2f} < {upper_body_threshold:.2f} (필요: {upper_body_valid}/{len(upper_body_indices)})")
                            continue
                        
                        # 키포인트 분산 확인: 한 점에 몰려있으면 오탐지
                        valid_points = points[conf_arr > config.Thresholds.POSE_CONFIDENCE]
                        valid_conf = conf_arr[conf_arr > config.Thresholds.POSE_CONFIDENCE]
                        
                        # 오탐지 방지: 최소 키포인트 개수 요구사항 강화 (3개 -> 5개)
                        if len(valid_points) >= 5 and len(valid_points) > 0:
                            # valid_points가 비어있지 않은지 확인 (안정성 향상)
                            try:
                                kpt_x_std = np.std(valid_points[:, 0])
                                kpt_y_std = np.std(valid_points[:, 1])
                                
                                # 키포인트 평균 신뢰도 검증 (사람 감지율 향상을 위해 완화)
                                # 재킷/가방 같은 객체는 키포인트 신뢰도가 낮을 가능성이 높음
                                avg_kpt_conf = np.mean(valid_conf) if len(valid_conf) > 0 else 0
                                min_avg_conf = 0.15  # 평균 신뢰도 최소 0.15 이상 요구 (0.20 -> 0.15, 더 많은 사람 감지)
                                if avg_kpt_conf < min_avg_conf:
                                    filter_stats['keypoint_confidence'] += 1
                                    logging.debug(f"🔍 [CAM-{cam_id}] P{i} 키포인트 평균 신뢰도 부족: {avg_kpt_conf:.3f} < {min_avg_conf:.3f}")
                                    continue
                            except (IndexError, ValueError):
                                # valid_points 처리 중 오류 발생 시 필터링
                                continue
                            
                            # 키포인트가 박스 내에 적절히 분산되어 있는지 확인
                            # 서있는 사람은 팔을 내리면 X축 분산이 작을 수 있으므로 완화
                            # 박스 크기에 비례한 최소 분산 요구 (오탐지 방지)
                            # 서있는 사람: Y축 분산이 중요 (세로로 분산), X축은 완화
                            # 넘어진 사람: X축 분산이 중요 (가로로 분산)
                            
                            if is_fallen_person:
                                # 넘어진 사람은 이미 692-714줄에서 검증되었으므로
                                # 여기서는 추가 검증만 수행 (이미 넘어진 사람으로 판단된 경우)
                                # 하지만 안전을 위해 다시 한 번 검증
                                min_x_std = max(10, box_w * 0.03)  # 박스 너비의 3% 이상
                                if kpt_x_std < min_x_std or kpt_y_std < 3:  # Y축 최소 3px 분산 필요
                                    # 이미 넘어진 사람으로 판단되었지만 분산이 부족하면 필터링
                                    filter_stats['keypoint_spread'] += 1
                                    keypoint_spread_ok = False
                                    logging.debug(f"🔍 [CAM-{cam_id}] P{i} (넘어진) 키포인트 분산 부족: X={kpt_x_std:.1f}<{min_x_std}, Y={kpt_y_std:.1f}<3")
                                    continue
                            else:
                                # 서있는 사람은 Y축 분산이 중요 (세로로 분산)
                                # X축은 팔을 내리면 작을 수 있으므로 완화
                                # 똑바로 서있는 사람(1자로 서있는 사람)도 인식되도록 더 완화
                                # 사람 감지율 향상: 키포인트 분산 임계값 완화
                                min_y_std = max(5, box_h * 0.02)  # 박스 높이의 2% 이상 또는 최소 5px (8px -> 5px, 3% -> 2%)
                                min_x_std = max(3, box_w * 0.015)  # 박스 너비의 1.5% 이상 또는 최소 3px (5px -> 3px, 2% -> 1.5%)
                                
                                # Y축 분산이 충분하면 통과 (서있는 사람은 세로로 분산)
                                if kpt_y_std >= min_y_std:
                                    pass  # 통과
                                # X축 분산이 충분하면 통과
                                elif kpt_x_std >= min_x_std:
                                    pass  # 통과
                                # 둘 다 부족해도 키포인트가 충분히 많으면 통과 (똑바로 서있는 사람/1자로 서있는 사람 대응)
                                # 사람 감지율 향상: 키포인트 개수 요구사항 완화 (7개 -> 5개)
                                elif len(valid_points) >= 5:  # 키포인트가 5개 이상이면 통과 (7개 -> 5개)
                                    pass  # 통과
                                else:
                                    # 모든 조건 불만족 시 필터링
                                    filter_stats['keypoint_spread'] += 1
                                    keypoint_spread_ok = False
                                    logging.debug(f"🔍 [CAM-{cam_id}] P{i} 키포인트 분산 부족: Y={kpt_y_std:.1f} < {min_y_std:.1f}, X={kpt_x_std:.1f} < {min_x_std:.1f}, 키포인트={len(valid_points)}개")
                                    continue
                            
                            # 추가 검증: 키포인트가 박스의 적절한 위치에 있는지 확인
                            # 키포인트의 중심이 박스 중앙 근처에 있어야 함 (오탐지 방지)
                            # 하지만 서있는 사람은 키포인트가 상체에 몰려있을 수 있으므로 완화
                            # 오탐지 방지: 최소 키포인트 개수 요구사항 강화 (3개 -> 5개)
                            if len(valid_points) >= 5:
                                kpt_center_x = np.mean(valid_points[:, 0])
                                kpt_center_y = np.mean(valid_points[:, 1])
                                box_center_x = (x1 + x2) / 2
                                box_center_y = (y1 + y2) / 2
                                
                                # 키포인트 중심이 박스 중심에서 너무 멀면 오탐지 가능성
                                # 서있는 사람은 키포인트가 상체에 몰려있을 수 있으므로 완화
                                # 똑바로 서있는 사람도 인식되도록 더 완화
                                # 측면으로 서있는 사람은 키포인트가 한쪽으로 치우칠 수 있으므로 더 완화
                                # 오탐지 방지: 키포인트 위치 오프셋 허용 범위 축소
                                max_offset_x = box_w * 0.6  # 80% -> 60% (오탐지 방지)
                                max_offset_y = box_h * 0.6  # 80% -> 60% (오탐지 방지)
                                
                                # 키포인트가 박스 밖에 있으면 오탐지 가능성 높음
                                # 하지만 박스 경계 근처는 허용 (키포인트가 박스 경계에 있을 수 있음)
                                if abs(kpt_center_x - box_center_x) > max_offset_x or \
                                   abs(kpt_center_y - box_center_y) > max_offset_y:
                                    # 키포인트 중심이 박스 밖에 있으면 오탐지 가능성 높음
                                    # 하지만 키포인트 자체가 박스 내에 있는지도 확인
                                    # 똑바로 서있는 사람은 키포인트가 상체에 몰려있을 수 있으므로
                                    # 측면으로 서있는 사람은 키포인트가 한쪽으로 치우칠 수 있으므로
                                    # 대부분의 키포인트가 박스 내에 있으면 통과
                                    kpt_in_box_count = sum(
                                        1 for pt in valid_points
                                        if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2
                                    )
                                    kpt_in_box_ratio = kpt_in_box_count / len(valid_points) if len(valid_points) > 0 else 0
                                    # 사람 감지율 향상: 키포인트 박스 내 비율 요구사항 완화 (70% -> 50%)
                                    if kpt_in_box_ratio < 0.5:  # 50% 이상의 키포인트가 박스 내에 있어야 함 (더 많은 사람 감지)
                                        # 키포인트가 박스 밖에 있으면 오탐지 가능성
                                        filter_stats['keypoint_position'] += 1
                                        logging.debug(f"🔍 [CAM-{cam_id}] P{i} 키포인트 박스 내 비율 부족: {kpt_in_box_ratio:.2f} < 0.5, 박스 내={kpt_in_box_count}/{len(valid_points)}")
                                        continue

                # 4. violation_model에서 탐지된 작은 객체와 겹치는지 확인
                should_filter = False
                for class_name, detections in all_detections.items():
                    # 'person' 클래스는 제외 (pose_model과 중복)
                    if class_name.lower() == 'person':
                        continue
                    # 안전 장비는 제외
                    is_safety_gear = any(class_name in item.values() for item in config.Constants.SAFETY_RULES_MAP.values())
                    if is_safety_gear:
                        continue

                    # 작은 객체(machinery, hand 등)와 겹치면 필터링
                    for det in detections:
                        if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                            dx1, dy1, dx2, dy2 = det['bbox']
                            det_area = (dx2 - dx1) * (dy2 - dy1)

                            # 작은 객체가 사람 박스 내부나 가까이 있으면 필터링
                            det_center_x = (dx1 + dx2) / 2
                            det_center_y = (dy1 + dy2) / 2

                            if (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2) or \
                               (dx1 < x2 and dx2 > x1 and dy1 < y2 and dy2 > y1):
                                iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                                # 작은 객체가 사람 박스 면적의 25% 이상 차지하고 IOU가 0.25 이상이면 제외
                                # (30% -> 25%, 0.2 -> 0.25로 조정하여 더 엄격하게 오탐지 방지)
                                if det_area > box_area * 0.25 and iou > 0.25:
                                    should_filter = True
                                    break

                    if should_filter:
                        break

                if should_filter:
                    continue

                # 모든 필터링을 통과한 유효한 사람 박스
                scaled_person_boxes.append(scaled_box_np)
                valid_person_indices.append(i)
                filtered_boxes.append(box)
                if keypoints_list is not None and i < len(keypoints_list):
                    filtered_keypoints.append(keypoints_list[i])
                if confidences is not None:
                    filtered_confidences.append(confidences[i])

            # 필터링 통계 출력
            final_box_count = len(filtered_boxes)
            total_filtered = sum(filter_stats.values())
            logging.info(f"[PARSE CAM-{cam_id}] Pose 박스 필터링 완료: 초기={initial_box_count}개 → 최종={final_box_count}개 "
                        f"(제거={initial_box_count - final_box_count}개)")
            if initial_box_count > 0 and final_box_count == 0:
                logging.warning(f"[PARSE CAM-{cam_id}] ⚠️ 모든 박스가 필터링됨! 제거 이유: "
                              f"clipped_none={filter_stats['clipped_none']}, "
                              f"zero_size={filter_stats['zero_size']}, "
                              f"zero_aspect={filter_stats['zero_aspect']}, "
                              f"min_size={filter_stats['min_size']}, "
                              f"aspect_ratio={filter_stats['aspect_ratio']}, "
                              f"min_keypoints={filter_stats['min_keypoints']}, "
                              f"keypoint_structure={filter_stats['keypoint_structure']}, "
                              f"upper_body_ratio={filter_stats['upper_body_ratio']}, "
                              f"keypoint_confidence={filter_stats['keypoint_confidence']}, "
                              f"keypoint_spread={filter_stats['keypoint_spread']}, "
                              f"keypoint_position={filter_stats['keypoint_position']}")

            # 필터링된 결과로 업데이트
            boxes = np.array(filtered_boxes) if filtered_boxes else np.array([])
            keypoints_list = filtered_keypoints if filtered_keypoints else None
            if confidences is not None:
                confidences = np.array(filtered_confidences) if filtered_confidences else np.array([])
            
            # 얼굴 기반 박스를 기존 pose 박스와 병합 (뒤에 있는 사람 추가)
            # 최적화: 거리 기반 필터링 먼저 수행하여 불필요한 IoU 계산 방지
            if face_detected_boxes:
                for face_box_data in face_detected_boxes:
                    fx1, fy1, fx2, fy2 = face_box_data['box']
                    face_center_x = (fx1 + fx2) / 2
                    face_center_y = (fy1 + fy2) / 2
                    face_diagonal = ((fx2 - fx1) ** 2 + (fy2 - fy1) ** 2) ** 0.5
                    max_distance = face_diagonal * 1.5  # 박스 대각선의 1.5배 이내만 고려
                    
                    # 기존 pose 박스와 겹치는지 확인 (거리 기반 필터링 먼저)
                    is_duplicate = False
                    for existing_box in boxes:
                        ex_x1, ex_y1, ex_x2, ex_y2 = existing_box * np.array([w_scale, h_scale, w_scale, h_scale])
                        ex_clipped = utils.clip_bbox_xyxy((ex_x1, ex_y1, ex_x2, ex_y2), orig_w, orig_h)
                        if ex_clipped:
                            ex_x1, ex_y1, ex_x2, ex_y2 = ex_clipped
                            
                            # 거리 기반 필터링 먼저 수행 (IoU 계산보다 빠름)
                            ex_center_x = (ex_x1 + ex_x2) / 2
                            ex_center_y = (ex_y1 + ex_y2) / 2
                            center_distance = ((face_center_x - ex_center_x) ** 2 + (face_center_y - ex_center_y) ** 2) ** 0.5
                            
                            # 거리가 너무 멀면 IoU 계산 생략 (성능 향상)
                            if center_distance > max_distance:
                                continue
                            
                            # IoU 계산 (거리 필터링 통과한 경우만)
                            iou = utils.calculate_iou((fx1, fy1, fx2, fy2), (ex_x1, ex_y1, ex_x2, ex_y2))
                            if iou > 0.3:  # 기존 박스와 겹치면 중복으로 간주
                                is_duplicate = True
                                break
                    
                    # 중복이 아니면 얼굴 기반 박스를 추가
                    if not is_duplicate:
                        # 리사이즈된 프레임 기준으로 변환
                        scaled_face_box = np.array([fx1 / w_scale, fy1 / h_scale, fx2 / w_scale, fy2 / h_scale])
                        boxes = np.vstack([boxes, scaled_face_box.reshape(1, -1)]) if len(boxes) > 0 else scaled_face_box.reshape(1, -1)
                        # 키포인트는 None (얼굴 기반 박스는 키포인트 없음)
                        # boxes와 keypoints_list 동기화 유지
                        if keypoints_list is None:
                            keypoints_list = []
                        elif not isinstance(keypoints_list, list):
                            # keypoints_list가 리스트가 아니면 변환
                            keypoints_list = list(keypoints_list) if hasattr(keypoints_list, '__iter__') else []
                        keypoints_list.append(None)
                        
                        # 동기화 검증: boxes와 keypoints_list 길이가 일치해야 함
                        if len(boxes) != len(keypoints_list):
                            logging.warning(f"[CAM-{cam_id}] ⚠️ 얼굴 기반 박스 추가 후 길이 불일치: boxes={len(boxes)}, keypoints_list={len(keypoints_list)}")
                            # 길이 맞추기 (keypoints_list를 boxes 길이에 맞춤)
                            while len(keypoints_list) < len(boxes):
                                keypoints_list.append(None)
                            if len(keypoints_list) > len(boxes):
                                keypoints_list = keypoints_list[:len(boxes)]

            # 얼굴 인식 시간 측정 시작
            face_recognition_start = time.time()

            # 병렬 처리를 위한 작업 목록 준비
            face_recognition_tasks = []
            futures_with_index = []  # (person_data_list_index, future)
            person_data_list = []  # 순서대로 결과를 맞추기 위한 리스트
            
            # PPE 박스 중복 매칭 방지를 위한 추적 세트 초기화 (매 프레임마다 초기화)
            # cam_id별 상태 관리 사용 (함수 속성 대신)
            frame_state = state.get_frame_processing_state(cam_id)
            if 'used_ppe_boxes' not in frame_state:
                frame_state['used_ppe_boxes'] = set()
            frame_state['used_ppe_boxes'].clear()  # 매 프레임마다 초기화

            faces_scheduled = 0
            total_person_boxes = len(boxes)
            logging.debug(f"[PARSE CAM-{cam_id}] Pose 결과 처리 시작: 총 사람 박스={total_person_boxes}개")
            
            for i, box in enumerate(boxes):
                scaled_box_np = box * np.array([w_scale, h_scale, w_scale, h_scale])
                clipped_box = utils.clip_bbox_xyxy(scaled_box_np, orig_w, orig_h)
                if clipped_box is None:
                    continue
                x1, y1, x2, y2 = clipped_box
                person_area = max(1, (x2 - x1) * (y2 - y1))
                person_id_text = f"P{i}"
                
                # 사람 박스 상세 로깅 (처리 시작)
                logging.debug(f"[PARSE CAM-{cam_id}] P{i} 처리 시작: 박스=({x1},{y1},{x2},{y2}), "
                            f"면적={person_area}, 비율={person_area/(orig_w*orig_h)*100:.1f}%")
                
                # 바운딩 박스 크기 검증: 프레임의 90%를 넘으면 잘못된 감지로 간주
                frame_area = orig_w * orig_h
                box_ratio = person_area / frame_area if frame_area > 0 else 0
                if box_ratio > 0.9:
                    logging.warning(f"[CAM-{cam_id}] 잘못된 박스 감지 (너무 큼): 박스={clipped_box}, 비율={box_ratio:.2f}, 프레임 크기={orig_w}x{orig_h}")
                    continue

                # 사람 박스 영역 추출
                person_img = frame[y1:y2, x1:x2]

                if person_img.size == 0:
                    continue

                # 채널 변환 (3채널 BGR로 통일)
                if len(person_img.shape) == 2:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)
                elif len(person_img.shape) == 3 and person_img.shape[2] == 1:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)
                elif len(person_img.shape) == 3 and person_img.shape[2] == 4:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_RGBA2BGR)
                elif len(person_img.shape) == 3 and person_img.shape[2] != 3:
                    person_img = cv2.cvtColor(person_img[:,:,0], cv2.COLOR_GRAY2BGR) if person_img.shape[2] > 0 else person_img

                # 최종 확인: 반드시 3채널 BGR
                if len(person_img.shape) != 3 or person_img.shape[2] != 3:
                    person_img = frame[y1:y2, x1:x2]
                    if len(person_img.shape) == 2 or (len(person_img.shape) == 3 and person_img.shape[2] == 1):
                        person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)

                # 실시간 처리 최적화: copy() 대신 직접 사용 (메모리 복사 시간 절약)
                # 얼굴 인식 함수 내에서 필요한 경우에만 복사
                person_img_for_detection = person_img  # copy() 제거로 처리 속도 향상

                # 얼굴 인식 작업을 병렬 처리 대기열에 추가
                # 실시간 처리 최적화: 간격 체크 및 최대 얼굴 수 제한 적용
                # 얼굴 기반 박스인지 확인 (뒤에 있는 사람)
                is_face_based_box = False
                face_box_matched = False  # 얼굴 기반 박스 매칭 여부
                if face_detected_boxes:
                    for face_box_data in face_detected_boxes:
                        fx1, fy1, fx2, fy2 = face_box_data['box']
                        # IoU 기반 매칭으로 개선 (고정 픽셀 오차 대신)
                        face_box_tuple = (fx1, fy1, fx2, fy2)
                        person_box_tuple = (x1, y1, x2, y2)
                        iou = utils.calculate_iou(face_box_tuple, person_box_tuple)
                        
                        # IoU가 0.5 이상이면 같은 박스로 간주 (더 정확한 매칭)
                        if iou >= 0.5:
                            is_face_based_box = True
                            face_box_matched = True
                            # 얼굴 기반 박스는 얼굴 정보가 이미 있으므로 바로 사용
                            # 얼굴 기반 박스는 InsightFace 기준을 함께 적용
                            if person_area >= config.Thresholds.MIN_FACE_RECOGNITION_AREA:
                                face_obj = face_box_data['face']
                                if hasattr(face_obj, "normed_embedding") and face_obj.normed_embedding is not None:
                                    embedding = face_obj.normed_embedding
                                    logging.debug(f"[CAM-{cam_id}] 얼굴 기반 박스 FAISS 매칭 시도: person_id={person_id_text}, 박스={person_box_tuple}")
                                    person_name, similarity_score = find_best_match_faiss(
                                        embedding, face_database, config.Thresholds.SIMILARITY
                                    )
                                    if person_name == "Unknown":
                                        logging.info(f"[CAM-{cam_id}] 얼굴 기반 박스 FAISS 매칭 실패: person_id={person_id_text}, 유사도={similarity_score:.3f}, 임계값={config.Thresholds.SIMILARITY}")
                                    else:
                                        logging.info(f"[CAM-{cam_id}] 얼굴 기반 박스 FAISS 매칭 성공: person_id={person_id_text}, 이름={person_name}, 유사도={similarity_score:.3f}")
                                    # 얼굴 기반 박스는 person_data_list에 추가하고 중복 방지
                                    person_data_list.append({
                                        'index': i,
                                        'person_id': person_id_text,
                                        'box': (x1, y1, x2, y2),
                                        'img': person_img_for_detection,
                                        'keypoints': keypoints_list[i] if isinstance(keypoints_list, list) and i < len(keypoints_list) else None,
                                        'name': person_name,
                                        'similarity': similarity_score,
                                        'embedding': embedding
                                    })
                                    break  # 얼굴 기반 박스는 한 번만 매칭
                
                # 얼굴 기반 박스가 매칭되지 않은 경우 로깅
                if not face_box_matched and face_detected_boxes:
                    logging.debug(f"[CAM-{cam_id}] 얼굴 박스와 person_box 매칭 실패: person_id={person_id_text}, person_box={person_box_tuple}, face_boxes={[(fb['box'], utils.calculate_iou(person_box_tuple, fb['box'])) for fb in face_detected_boxes]}")
                
                # 얼굴 기반 박스가 아닌 경우에만 person_data_list에 추가 (중복 방지)
                if not face_box_matched:
                    # 모든 사람 박스에 대해 person_data_list에 추가 (PPE 감지는 항상 실행)
                    person_data_list.append({
                        'index': i,
                        'person_id': person_id_text,
                        'box': (x1, y1, x2, y2),
                        'img': person_img_for_detection,
                        'keypoints': keypoints_list[i] if isinstance(keypoints_list, list) and i < len(keypoints_list) else None
                    })
                
                # PPE 감지: 얼굴 인식과 독립적으로 항상 실행 (멀리 있어도 감지)
                # 모든 사람이 같은 all_detections를 공유하므로, 이미 매칭된 PPE 박스를 추적하여 중복 매칭 방지
                # cam_id별 상태 관리 사용 (함수 속성 대신)
                frame_state = state.get_frame_processing_state(cam_id)
                if 'used_ppe_boxes' not in frame_state:
                    frame_state['used_ppe_boxes'] = set()
                ppe_violations, ppe_boxes = _process_ppe_detection(
                    (x1, y1, x2, y2), 
                    all_detections, 
                    frame_state['used_ppe_boxes'],
                    person_id=f"CAM-{cam_id}_{person_id_text}"
                )
                
                # person_data_list가 비어있지 않은 경우에만 PPE 정보 추가 (IndexError 방지)
                if person_data_list:
                    person_data_list[-1]['ppe_violations'] = ppe_violations
                    person_data_list[-1]['ppe_boxes'] = ppe_boxes  # PPE 박스 정보 저장
                else:
                    # person_data_list가 비어있으면 PPE 정보와 함께 추가
                    person_data_list.append({
                        'index': i,
                        'person_id': person_id_text,
                        'box': (x1, y1, x2, y2),
                        'img': person_img_for_detection,
                        'keypoints': keypoints_list[i] if isinstance(keypoints_list, list) and i < len(keypoints_list) else None,
                        'ppe_violations': ppe_violations,
                        'ppe_boxes': ppe_boxes
                    })
                
                # 위험 행동 감지: 얼굴 인식 전에 수행 (위험 상황 확인용)
                is_dangerous_detected = False
                violation_type = ""
                # person_data_list가 비어있지 않은 경우에만 위험 행동 감지 수행
                person_keypoints = person_data_list[-1].get('keypoints') if person_data_list else None
                if person_keypoints is not None:
                    try:
                        # 키포인트 스케일링
                        scaled_kpts_data = person_keypoints.data.clone()
                        scaled_kpts_data[..., 0] *= w_scale
                        scaled_kpts_data[..., 1] *= h_scale
                        scaled_keypoints = Keypoints(scaled_kpts_data, (orig_h, orig_w))
                        
                        # person_box_key 생성 (위험 행동 감지용)
                        person_box_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                        
                        # 위험 행동 감지 (동기 실행, 위험할 때만 True 반환)
                        is_dangerous_detected, violation_type = _process_dangerous_behavior(
                            scaled_keypoints, (x1, y1, x2, y2), cam_id, person_box_key
                        )
                        # person_data_list가 비어있지 않은 경우에만 위험 행동 정보 저장
                        if person_data_list:
                            person_data_list[-1]['is_dangerous'] = is_dangerous_detected
                            person_data_list[-1]['violation_type'] = violation_type
                    except Exception as e:
                        logging.warning(f"[CAM-{cam_id}] 위험 행동 감지 중 오류 발생: {e}", exc_info=True)
                        # person_data_list가 비어있지 않은 경우에만 위험 행동 정보 저장
                        if person_data_list:
                            person_data_list[-1]['is_dangerous'] = False
                            person_data_list[-1]['violation_type'] = ""
                
                # ============================================================
                # 얼굴 인식 시스템 흐름 (기회형 인식: 상시 추적 + 기회형 인식)
                # ============================================================
                # [병렬 처리] 기본 모델 (항상 실행):
                #   - Yolo11n_PPE1.pt: PPE 위반 감지 (ThreadPoolExecutor로 병렬 실행)
                #   - yolo11n-pose.pt: 위험 행동 감지 (ThreadPoolExecutor로 병렬 실행)
                #
                # [기회형 얼굴 인식] 위반 여부와 상관없이 얼굴이 선명하고 정면일 때 인식:
                #   - 위반 발생 시: 즉시 인식 시도 (기존 로직)
                #   - 위반 없을 때: 얼굴이 선명하고 정면이면 미리 인식하여 캐시에 저장
                #   - 위반 발생 시: 캐시에서 이름을 꺼내서 사용 (인식률 비약적 상승)
                #   1. 사람 박스 크롭 (person_img_for_detection) ✅ 이미 완료
                #   2. yolov11n-face.pt 모델로 크롭된 이미지에서 얼굴 찾기
                #   3. 얼굴 영역 추출 (얼굴 자르기)
                #   4. buffalo_L 모델(InsightFace) 또는 AdaFace로 얼굴 임베딩 추출
                #   5. face_index.faiss, face_index.faiss.labels.npy를 사용하여 매칭
                #   6. 누구인지 특정 (person_name 반환) → 캐시에 저장
                # ============================================================
                
                has_violation_or_danger = len(ppe_violations) > 0 or is_dangerous_detected
                
                # 얼굴 인식 실행 조건 체크 (기회형 인식: 위반 여부와 상관없이 실행)
                check_face_based_box = not is_face_based_box
                check_face_model = face_model is not None
                check_face_analyzer = face_analyzer is not None
                check_face_database = face_database is not None
                check_area = person_area >= config.Thresholds.MIN_FACE_RECOGNITION_AREA
                # 거리 기반 게이팅: 사람 키가 프레임 높이의 일정 비율 이상일 때만 얼굴 인식
                person_height = max(1, y2 - y1)
                height_ratio = person_height / max(1, orig_h)
                min_height_ratio = float(config.Thresholds.MIN_PERSON_HEIGHT_RATIO_FOR_FACE)
                check_height_ratio = height_ratio >= min_height_ratio
                
                # 기회형 인식: 위반 여부와 상관없이 얼굴이 선명하고 정면일 때 인식
                # 위반이 있으면 즉시 인식, 없으면 기회형 인식 (얼굴이 선명할 때)
                # 얼굴 품질 체크: 키포인트 기반으로 얼굴 방향 추정 (정면일 가능성)
                face_quality_ok = False
                if person_keypoints is not None:
                    try:
                        # 키포인트에서 얼굴 방향 추정 (nose, eyes 등)
                        scaled_kpts_data = person_keypoints.data.clone()
                        scaled_kpts_data[..., 0] *= w_scale
                        scaled_kpts_data[..., 1] *= h_scale
                        kpts_conf = person_keypoints.conf[0].cpu().numpy() if person_keypoints.conf is not None else None
                        if kpts_conf is not None:
                            # nose(0), left_eye(1), right_eye(2) 키포인트가 모두 보이면 정면 가능성 높음
                            nose_visible = kpts_conf[0] > config.Thresholds.POSE_CONFIDENCE if len(kpts_conf) > 0 else False
                            left_eye_visible = kpts_conf[1] > config.Thresholds.POSE_CONFIDENCE if len(kpts_conf) > 1 else False
                            right_eye_visible = kpts_conf[2] > config.Thresholds.POSE_CONFIDENCE if len(kpts_conf) > 2 else False
                            # 얼굴 키포인트가 2개 이상 보이면 품질 OK (정면 가능성)
                            face_quality_ok = (nose_visible and (left_eye_visible or right_eye_visible)) or (left_eye_visible and right_eye_visible)
                    except Exception as e:
                        logging.debug(f"[CAM-{cam_id}] 얼굴 품질 확인 중 오류 발생: {e}")
                        pass
                
                # 기회형 인식 조건 (최적화: 조건 강화로 불필요한 호출 감소):
                # 1. 위반이 있으면 즉시 인식 (기존 로직)
                # 2. 위반이 없어도 얼굴 품질이 좋고, 충분히 가까운 거리일 때만 인식 (기회형 인식)
                # 거리 조건 완화: height_ratio가 더 낮아도 기회형 인식 (인식률 향상)
                min_height_for_opportunistic = 0.12  # 프레임 높이의 12% 이상 (0.15 -> 0.12, 더 멀리서도 인식)
                opportunistic_recognition = (
                    not has_violation_or_danger and 
                    face_quality_ok and 
                    height_ratio >= min_height_for_opportunistic  # 더 멀리서도 인식 가능
                )
                immediate_recognition = has_violation_or_danger
                
                # 사람 박스와 얼굴 박스 매칭 (IoU 기반)
                matched_face = None
                if faces_in_frame and len(faces_in_frame) > 0:
                    person_box = (x1, y1, x2, y2)
                    best_iou = 0.0
                    best_face = None
                    
                    for face_obj in faces_in_frame:
                        if face_obj.bbox is None:
                            continue
                        fx1, fy1, fx2, fy2 = face_obj.bbox
                        face_box = (fx1, fy1, fx2, fy2)
                        
                        # IoU 계산
                        iou = utils.calculate_iou(person_box, face_box)
                        # Fast Path 활성화: IoU 임계값 완화 (0.1 -> 0.05)로 더 많은 얼굴 매칭
                        if iou > best_iou and iou > 0.05:  # 최소 IoU 임계값 0.05 (더 관대한 매칭)
                            best_iou = iou
                            best_face = face_obj
                    
                    if best_face is not None:
                        matched_face = best_face
                
                # 캐시에서 이미 인식된 사람인지 확인 (재인식 스킵으로 성능 향상)
                cache_skip_recognition = False
                cached_name = "Unknown"
                if recent_identity_cache is not None:
                    try:
                        hold_sec = config.Thresholds.RECOGNITION_HOLD_SECONDS
                        cache_entries = recent_identity_cache.get_recent(cam_id, max_age=hold_sec)
                        
                        # 현재 박스와 IoU 높은 캐시 항목 찾기
                        person_box = (x1, y1, x2, y2)
                        best_iou = 0.0
                        best_entry = None
                        
                        current_box_center_x = (x1 + x2) / 2
                        current_box_center_y = (y1 + y2) / 2
                        current_box_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        max_distance = current_box_diagonal * 1.5
                        
                        for entry in cache_entries:
                            entry_box = entry.get('box', (0,0,0,0))
                            if len(entry_box) != 4:
                                continue
                            
                            # 거리 기반 필터링
                            entry_center_x = (entry_box[0] + entry_box[2]) / 2
                            entry_center_y = (entry_box[1] + entry_box[3]) / 2
                            center_distance = ((current_box_center_x - entry_center_x) ** 2 + 
                                             (current_box_center_y - entry_center_y) ** 2) ** 0.5
                            
                            if center_distance > max_distance:
                                continue
                            
                            iou = utils.calculate_iou(person_box, tuple(entry_box))
                            if iou >= 0.5 and iou > best_iou:
                                best_iou = iou
                                best_entry = entry
                        
                        # ⭐ 트래킹 강화: IoU 매칭 성공 시 이전 이름 무조건 사용 (재인식 스킵)
                        if best_entry is not None and best_iou >= 0.4:
                            cached_name = best_entry.get('name', 'Unknown')
                            cached_score = best_entry.get('score', 0.0)
                            # IoU 매칭이 성공했고 이전에 인식된 이름이 있으면 무조건 사용 (트래킹 유지)
                            if cached_name != "Unknown":
                                cache_skip_recognition = True
                                # 캐시된 이름을 미리 설정 (재인식 스킵)
                                if person_data_list:
                                    person_data_list[-1]['name'] = cached_name
                                    person_data_list[-1]['similarity'] = cached_score
                                # 캐시 항목 갱신 (위치/시간)
                                best_entry['box'] = (x1, y1, x2, y2)
                                best_entry['ts'] = time.time()
                                recent_identity_cache.add(cam_id, best_entry)
                                logging.debug(f"[CAM-{cam_id}] 🔄 트래킹으로 재인식 스킵: {cached_name} (IoU={best_iou:.3f})")
                    except Exception as e:
                        logging.debug(f"[CAM-{cam_id}] 캐시 확인 중 오류 발생: {e}")
                        pass  # 캐시 확인 실패는 무시
                
                allow_face_job = (
                    check_face_based_box and
                    check_face_model and
                    check_face_analyzer and
                    check_face_database and
                    (immediate_recognition or opportunistic_recognition) and  # ⭐ 기회형 인식: 위반 또는 얼굴 품질 좋을 때
                    check_area and
                    check_height_ratio and
                    not cache_skip_recognition  # 캐시에서 찾았으면 재인식 스킵
                )

                if allow_face_job:
                    # 얼굴 인식 실행: 기회형 인식 (위반 시 즉시 또는 얼굴 품질 좋을 때)
                    # 백그라운드에서 처리 (yolov11n-face.pt → 얼굴 자르기 → buffalo_L/AdaFace 임베딩 → FAISS 매칭)
                    # 복사 최적화: executor가 별도 스레드에서 실행되므로 복사 필요, 하지만 뷰(view) 사용으로 메모리 효율화
                    # person_img_for_detection은 이미 크롭된 이미지이므로 복사 비용이 낮음
                    # FastIndustrialRecognizer 가져오기 (랜드마크 기반 고속 처리용)
                    fast_recognizer = getattr(safety_system, 'fast_recognizer', None)
                    use_adaface = getattr(safety_system, 'use_adaface', False)
                    adaface_model_path = getattr(safety_system, 'adaface_model_path', None)
                    
                    # TensorRT 사용 여부 확인
                    face_uses_trt = getattr(safety_system, 'face_uses_trt', False)
                    
                    
                    # Executor가 shutdown되었는지 확인
                    try:
                        if face_recognition_executor._shutdown:
                            logging.warning(f"⚠️ Face Executor가 shutdown되었습니다. 재생성합니다.")
                            from state import update_worker_executors
                            update_worker_executors()
                            from state import face_recognition_executor as new_face_executor
                            face_recognition_executor = new_face_executor
                    except AttributeError:
                        pass  # _shutdown 속성이 없는 경우 무시
                    
                    try:
                        logging.debug(f"[CAM-{cam_id}] 얼굴 인식 작업 제출: person_id={person_id_text}, 박스={person_box}, matched_face={matched_face is not None}")
                        future = face_recognition_executor.submit(
                            _process_face_recognition,
                            person_img_for_detection.copy(),  # 스레드 안전성을 위해 복사 필요
                            person_id_text,
                            face_model,
                            face_analyzer,
                            face_database,
                            use_adaface,
                            adaface_model_path,
                            fast_recognizer,
                            matched_face,  # 미리 감지된 얼굴 (Fast Path)
                            resized_frame,  # 원본(리사이즈된) 프레임
                            face_uses_trt  # TensorRT 사용 여부 (640x640 고정 입력 크기)
                        )
                    except RuntimeError as e:
                        if "shutdown" in str(e).lower():
                            # Executor가 shutdown된 경우 재생성
                            logging.warning(f"⚠️ Face Executor shutdown 오류 감지, 재생성합니다: {e}")
                            from state import update_worker_executors
                            update_worker_executors()
                            from state import face_recognition_executor as new_face_executor
                            face_recognition_executor = new_face_executor
                            future = face_recognition_executor.submit(
                                _process_face_recognition,
                                person_img_for_detection.copy(),
                                person_id_text,
                                face_model,
                                face_analyzer,
                                face_database,
                                use_adaface,
                                adaface_model_path,
                                fast_recognizer,
                                matched_face,
                                resized_frame,
                                face_uses_trt
                            )
                        else:
                            raise
                    
                    
                    face_recognition_tasks.append(future)
                    futures_with_index.append((len(person_data_list) - 1, future))
                    faces_scheduled += 1
                    
                    # 얼굴 탐지 프레임 카운터 업데이트 (얼굴 인식 실행 시)
                    with face_detection_lock:
                        current_frame = frame_stats.get(cam_id, {}).get('frame_count', 0)
                        last_face_detection_frame[cam_id] = current_frame
                else:
                    # 얼굴 인식 스킵: 캐시된 라벨이 있으면 유지
                    try:
                        now_ts = time.time()
                        hold_sec = config.Thresholds.RECOGNITION_HOLD_SECONDS
                        # IdentityCache에서 최근 항목 가져오기 (자동 만료 처리)
                        cache_entries = recent_identity_cache.get_recent(cam_id, max_age=hold_sec)
                        if cache_entries:
                            best_iou = 0.0
                            best_entry = None
                            for entry in cache_entries:
                                iou = utils.calculate_iou((x1, y1, x2, y2), tuple(entry.get('box', (0,0,0,0))))
                                if iou > best_iou:
                                    best_iou = iou
                                    best_entry = entry
                            if best_entry is not None and best_iou >= 0.5:
                                # 캐시 라벨 사용
                                if person_data_list:
                                    person_data_list[-1]['name'] = best_entry.get('name', 'Unknown')
                                    person_data_list[-1]['similarity'] = float(best_entry.get('score', 0.0))
                                # 위치/시간 갱신 (캐시에 다시 추가하여 갱신)
                                best_entry['box'] = (x1, y1, x2, y2)
                                best_entry['ts'] = now_ts
                                recent_identity_cache.add(cam_id, best_entry)
                    except Exception as e:
                        logging.debug(f"[CAM-{cam_id}] 캐시 라벨 사용 중 오류 발생: {e}")
                        pass

            if faces_scheduled > 0:
                face_recognition_cooldown_ts[cam_id] = time.time()

            # 병렬로 얼굴 인식 결과 수집 (인원수 제한 해제)
            # GPU 사용 시 모든 사람 처리 가능
            # 제한 없이 모든 얼굴 인식 작업 처리
            
            face_recognition_results = {}
            try:
                # 모든 작업 완료 대기 (TensorRT 최적화: 타임아웃 대폭 증가)
                # TensorRT 사용 시에도 얼굴 인식은 시간이 걸릴 수 있으므로 타임아웃 증가
                num_workers = face_recognition_executor._max_workers
                # 타임아웃: 사람당 400ms, 최소 2.0s, 최대 5.0s (TensorRT NMS 지연 대응, 최적화: 500ms -> 400ms)
                # 사람이 많을 때도 타임아웃이 너무 길어지지 않도록 제한
                timeout_seconds = min(5.0, max(2.0, min(len(futures_with_index) * 0.4, 4.0)))
                
                for future in as_completed([f for _, f in futures_with_index], timeout=timeout_seconds):
                    try:
                        
                        # 개별 작업 타임아웃: 최대 3.0s (TensorRT NMS 지연 대응)
                        individual_timeout = min(3.0, max(1.0, 0.4 * (len(futures_with_index) / max(1, num_workers))))
                        # 얼굴 인식 결과만 수집 (PPE는 이미 처리됨)
                        person_name, similarity_score, embedding, face_bbox = future.result(timeout=individual_timeout)
                        # 매핑된 인덱스에 결과 기록 (person_idx를 올바르게 사용)
                        mapped_idx = next((person_idx for person_idx, f in futures_with_index if f is future), None)
                        if mapped_idx is not None and mapped_idx < len(person_data_list):
                            # 얼굴 인식 결과 로깅
                            if person_name == "Unknown":
                                logging.info(f"[CAM-{cam_id}] 얼굴 인식 실패: person_idx={mapped_idx}, 유사도={similarity_score:.3f}, 임계값={config.Thresholds.SIMILARITY}")
                            else:
                                logging.info(f"[CAM-{cam_id}] 얼굴 인식 성공: person_idx={mapped_idx}, 이름={person_name}, 유사도={similarity_score:.3f}")
                            person_data_list[mapped_idx]['name'] = person_name
                            person_data_list[mapped_idx]['similarity'] = similarity_score
                            person_data_list[mapped_idx]['embedding'] = embedding  # 센트로이드 계산용
                            person_data_list[mapped_idx]['face_bbox'] = face_bbox  # 얼굴 바운딩박스 저장
                            # PPE 위반 목록은 이미 person_data_list에 저장되어 있음
                            
                    except FaceRecognitionError as e:
                        # FaceRecognitionError는 로깅만 하고 계속 진행 (Unknown 유지)
                        pass  # 얼굴 인식 오류는 무시
                        # 타임아웃/오류 시 이미 기본 Unknown 유지
                    except Exception as e:
                        logging.warning(f"[CAM-{cam_id}] 얼굴 인식 작업 실패: {e}", exc_info=True)
                        pass  # 얼굴 인식 작업 실패는 무시
                        # 타임아웃/오류 시 이미 기본 Unknown 유지
            except FuturesTimeoutError:
                # 일부 작업이 시간 내 완료되지 않아도 진행
                logging.warning(f"⚠️ [CAM-{cam_id}] 얼굴 인식 일부 작업 타임아웃 (총 {len(futures_with_index)}개 중 일부 미완료, 타임아웃={timeout_seconds:.1f}s)")
                # 타임아웃된 Future 상태 확인 및 취소
                for idx, (person_idx, future) in enumerate(futures_with_index):
                    if not future.done():
                        logging.warning(f"⚠️ [CAM-{cam_id}] 얼굴 인식 Future {idx} 타임아웃: person_idx={person_idx}, done={future.done()}, cancelled={future.cancelled()}, running={future.running()}")
                        # 예외 정보 확인
                        try:
                            exception = future.exception(timeout=0.1)
                            if exception:
                                logging.error(f"❌ [CAM-{cam_id}] 얼굴 인식 Future {idx} 예외: {exception}", exc_info=exception)
                        except Exception as e:
                            logging.debug(f"[CAM-{cam_id}] 얼굴 인식 Future {idx} 예외 확인 중 오류: {e}")
                        # 타임아웃된 Future는 취소 시도 (리소스 정리)
                        try:
                            if not future.done() and not future.cancelled():
                                future.cancel()
                        except Exception as e:
                            logging.debug(f"[CAM-{cam_id}] 얼굴 인식 Future {idx} 취소 중 오류: {e}")
            
            # 얼굴 인식 시간 측정 종료
            perf_timings['face_recognition'] = (time.time() - face_recognition_start) * 1000  # ms

            # 결과를 순서대로 처리
            # 프레임 내 동일 이름 중복 방지: 이름별로 박스와 similarity 저장
            name_to_boxes: Dict[str, List[Tuple[Tuple[int,int,int,int], float, int]]] = {}  # (box, score, person_index)
            person_final_names: Dict[int, str] = {}  # person_index -> 최종 이름
            
            # 1단계: 모든 person_data를 순회하여 name_to_boxes 수집 (원본 이름 사용)
            for person_data in person_data_list:
                i = person_data['index']
                person_id_text = person_data['person_id']
                x1, y1, x2, y2 = person_data['box']
                person_name = person_data.get('name', 'Unknown')  # 원본 이름
                similarity_score = person_data.get('similarity', 0.0)

                # --- 얼굴 인식 안정화: 히스테리시스 + 홀드 ---
                matched_entry = None  # 센트로이드 및 넘어짐 감지에서 재사용
                try:
                    hold_sec = config.Thresholds.RECOGNITION_HOLD_SECONDS
                    up_th = config.Thresholds.SIMILARITY
                    down_th = max(0.0, up_th - config.Thresholds.RECOGNITION_HYSTERESIS_DELTA)
                    now_ts = time.time()

                    # IdentityCache에서 최근 항목 가져오기 (자동 만료 처리)
                    cache_entries = recent_identity_cache.get_recent(cam_id, max_age=hold_sec)

                    # 캐시에서 IoU 높은 최근 항목 찾기 (겹치는 사람 구분을 위해 IoU 임계값 상향)
                    # 최적화: 거리 기반 필터링 먼저 수행하여 불필요한 IoU 계산 방지
                    best_iou = 0.0
                    current_box_center_x = (x1 + x2) / 2
                    current_box_center_y = (y1 + y2) / 2
                    current_box_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    max_distance = current_box_diagonal * 1.5  # 박스 대각선의 1.5배 이내만 고려
                    
                    for entry in cache_entries:  # IdentityCache가 이미 만료된 항목 제거
                        entry_box = entry.get('box', (0,0,0,0))
                        if len(entry_box) != 4:
                            continue
                        
                        # 거리 기반 필터링 먼저 수행 (IoU 계산보다 빠름)
                        entry_center_x = (entry_box[0] + entry_box[2]) / 2
                        entry_center_y = (entry_box[1] + entry_box[3]) / 2
                        center_distance = ((current_box_center_x - entry_center_x) ** 2 + 
                                         (current_box_center_y - entry_center_y) ** 2) ** 0.5
                        
                        # 거리가 너무 멀면 IoU 계산 생략 (성능 향상)
                        if center_distance > max_distance:
                            continue
                        
                        # IoU 계산 (거리 필터링 통과한 항목만)
                        iou = utils.calculate_iou((x1, y1, x2, y2), tuple(entry_box))
                        # IoU 임계값 완화: 0.5 -> 0.4 (바운딩 박스 안정화, 깜빡임 방지)
                        if iou >= 0.4 and iou > best_iou:
                            best_iou = iou
                            matched_entry = entry

                    # ⭐ 트래킹 기반 이름 유지 강화: IoU 매칭 성공 시 이전 이름 무조건 유지
                    if matched_entry is not None and best_iou >= 0.4:
                        # IoU 매칭이 성공한 경우 (같은 사람으로 추적됨)
                        cached_name = matched_entry.get('name', 'Unknown')
                        cached_score = float(matched_entry.get('score', 0.0))
                        
                        # 이전에 인식된 이름이 있고 Unknown이 아니면 무조건 유지 (트래킹 유지)
                        if cached_name != 'Unknown':
                            # 트래킹으로 이름 유지 (얼굴 인식 실패해도 트래킹이 성공하면 이름 유지)
                            person_name = cached_name
                            # 유사도 점수는 이전 값 유지하거나 새 값 사용 (더 높은 값 선택)
                            if person_name != "Unknown" and similarity_score >= up_th:
                                # 새 인식 성공 시 EMA로 스무딩
                                smoothed = 0.7 * similarity_score + 0.3 * cached_score
                                similarity_score = smoothed
                            else:
                                # 얼굴 인식 실패 시 이전 점수 유지
                                similarity_score = cached_score
                            
                            # 캐시 항목 갱신 (트래킹 유지)
                            matched_entry['box'] = (x1, y1, x2, y2)
                            matched_entry['name'] = person_name
                            matched_entry['score'] = similarity_score
                            matched_entry['ts'] = now_ts
                            recent_identity_cache.add(cam_id, matched_entry)
                            logging.debug(f"[CAM-{cam_id}] 🔄 트래킹으로 이름 유지: {person_name} (IoU={best_iou:.3f}, 유사도={similarity_score:.3f})")
                        elif person_name != "Unknown" and similarity_score >= up_th:
                            # 새 인식 성공: 캐시에 저장
                            matched_entry['box'] = (x1, y1, x2, y2)
                            matched_entry['name'] = person_name
                            matched_entry['score'] = similarity_score
                            matched_entry['ts'] = now_ts
                            recent_identity_cache.add(cam_id, matched_entry)
                    elif person_name != "Unknown" and similarity_score >= up_th:
                        # 새 인식 성공 (IoU 매칭 실패): 새 항목 추가
                        recent_identity_cache.add(cam_id, {
                            'box': (x1, y1, x2, y2),
                            'name': person_name,
                            'score': similarity_score
                            # 'ts'는 IdentityCache.add()에서 자동 추가
                        })
                    else:
                        # Unknown 또는 낮은 점수: IoU 매칭 실패 시 홀드 조건으로 이름 유지 시도
                        if matched_entry is not None:
                            age = now_ts - matched_entry.get('ts', 0)
                            last_score = float(matched_entry.get('score', 0.0))
                            cached_name = matched_entry.get('name', 'Unknown')
                            # 홀드 조건: 시간 내이고 점수가 임계값 이상이면 이름 유지
                            if age <= hold_sec and last_score >= down_th and cached_name != 'Unknown':
                                person_name = cached_name
                                similarity_score = last_score
                                # 박스/시간 갱신
                                matched_entry.update({'box': (x1, y1, x2, y2), 'ts': now_ts})
                                recent_identity_cache.add(cam_id, matched_entry)
                except Exception as _stb_e:
                    # 안정화 로직 오류는 무시하고 원 결과 사용
                    pass
                
                # now_ts가 try 블록 안에서만 정의되었을 수 있으므로 재정의
                if 'now_ts' not in locals():
                    now_ts = time.time()

                # 센트로이드 임베딩: 여러 프레임의 임베딩을 평균내어 안정성 향상 (final 개선 기법)
                # matched_entry를 재사용하여 중복 계산 제거 (성능 최적화)
                person_box_key = None
                embedding = person_data.get('embedding')
                
                if embedding is not None:
                    # matched_entry를 재사용 (이미 위에서 계산됨)
                    if matched_entry is not None:
                        cached_name = matched_entry.get('name', 'Unknown')
                        if cached_name != "Unknown":
                            # 이름 기반 키 사용 (더 정확한 추적)
                            person_box_key = f"{cam_id}_{cached_name}"
                        else:
                            # 이름이 없으면 박스 기반 키 사용
                            person_box_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                    else:
                        # 새로운 사람: 박스 기반 키 생성
                        person_box_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                    
                    # 임베딩을 버퍼에 추가 (안전한 접근)
                    if cam_id not in embedding_buffers:
                        embedding_buffers[cam_id] = {}
                    if person_box_key not in embedding_buffers[cam_id]:
                        embedding_buffers[cam_id][person_box_key] = {'embeddings': [], 'last_update': 0.0}
                    buffer_data = embedding_buffers[cam_id][person_box_key]
                    # 임베딩 복사 최적화: numpy 배열은 view를 사용하거나 필요할 때만 복사
                    # 센트로이드 계산 시 복사가 필요하므로 여기서는 복사 유지 (안정성 우선)
                    buffer_data['embeddings'].append(embedding.copy())
                    buffer_data['last_update'] = now_ts
                    
                    # 버퍼 크기 제한 (최대 EMBEDDING_BUFFER_SIZE)
                    if len(buffer_data['embeddings']) > EMBEDDING_BUFFER_SIZE:
                        buffer_data['embeddings'] = buffer_data['embeddings'][-EMBEDDING_BUFFER_SIZE:]
                    
                    # 센트로이드 계산: 최소 3개 이상이면 계산 (정확도 향상)
                    if len(buffer_data['embeddings']) >= EMBEDDING_BUFFER_MIN_SIZE:
                        # 캐시 확인 (최근 2초 내 결과 재사용)
                        cached_centroid = centroid_cache[cam_id].get(person_box_key)
                        if cached_centroid:
                            person_name_centroid = cached_centroid.get('name', 'Unknown')
                            similarity_score_centroid = cached_centroid.get('score', 0.0)
                        else:
                            # 센트로이드 계산 최적화: numpy 배열을 한 번에 처리
                            embeddings_array = np.array(buffer_data['embeddings'])
                            if len(embeddings_array) > 0:
                                avg_embedding = np.mean(embeddings_array, axis=0)
                                norm = np.linalg.norm(avg_embedding)
                                if norm > 1e-6:
                                    normalized_avg_embedding = (avg_embedding / norm).astype('float32')
                                    # 센트로이드 임베딩으로 재검색 (약간 엄격한 임계값 적용)
                                    # 센트로이드는 여러 프레임 평균이므로 기본보다 +0.03만 상향
                                    centroid_threshold = config.Thresholds.SIMILARITY + 0.03  # 기본 임계값 + 0.03
                                    person_name_centroid, similarity_score_centroid = find_best_match_faiss(
                                        normalized_avg_embedding, face_database, centroid_threshold
                                    )
                                    # 캐시에 저장 (TTLCache가 자동으로 만료 처리)
                                    centroid_cache[cam_id].put(person_box_key, {
                                        'name': person_name_centroid,
                                        'score': similarity_score_centroid
                                    })
                                else:
                                    person_name_centroid = "Unknown"
                                    similarity_score_centroid = 0.0
                            else:
                                person_name_centroid = "Unknown"
                                similarity_score_centroid = 0.0
                        
                        # 센트로이드 결과가 더 좋으면 사용 (Unknown이 아니고 similarity가 더 높으면)
                        if person_name_centroid != "Unknown" and (person_name == "Unknown" or similarity_score_centroid > similarity_score):
                            person_name = person_name_centroid
                            similarity_score = similarity_score_centroid
                            # 버퍼 초기화 (인식 성공 시)
                            buffer_data['embeddings'] = []
                
                # 동일 이름 중복 방지: 같은 이름이 여러 박스에 할당되면 가장 높은 similarity만 유지
                if person_name != "Unknown":
                    if person_name not in name_to_boxes:
                        name_to_boxes[person_name] = []
                    name_to_boxes[person_name].append(((x1, y1, x2, y2), float(similarity_score), i))
            
            # 오래된 임베딩 버퍼 및 캐시 정리 (메모리 관리 개선)
            # 주기적 정리로 최적화: 매 프레임마다 실행하지 않고 10초마다 실행
            # cam_id별 상태 관리 사용 (함수 속성 대신)
            frame_state = state.get_frame_processing_state(cam_id)
            current_time_cleanup = time.time()
            CLEANUP_INTERVAL = 10.0  # 10초마다 정리
            
            if current_time_cleanup - frame_state.get('last_cleanup_time', 0.0) > CLEANUP_INTERVAL:
                cleanup_threshold = 5.0  # 5초 이상 사용되지 않은 버퍼 제거
                
                # embedding_buffers 정리 (최적화: list() 변환 최소화)
                for cam_id_cleanup in list(embedding_buffers.keys()):
                    if cam_id_cleanup not in embedding_buffers:  # 안전성 체크
                        continue
                    
                    # 카메라별 버퍼 수 제한 (메모리 최적화)
                    if len(embedding_buffers[cam_id_cleanup]) > MAX_EMBEDDING_BUFFERS_PER_CAM:
                        # 가장 오래된 버퍼부터 제거
                        sorted_keys = sorted(
                            embedding_buffers[cam_id_cleanup].keys(),
                            key=lambda k: embedding_buffers[cam_id_cleanup][k].get('last_update', 0)
                        )
                        # 초과분 제거
                        for key_to_remove in sorted_keys[:-MAX_EMBEDDING_BUFFERS_PER_CAM]:
                            if key_to_remove in embedding_buffers[cam_id_cleanup]:
                                del embedding_buffers[cam_id_cleanup][key_to_remove]
                            # 관련 캐시도 제거
                            if cam_id_cleanup in centroid_cache:
                                centroid_cache[cam_id_cleanup].remove(key_to_remove)
                    
                    # 키 목록을 한 번만 생성 (성능 최적화)
                    keys_to_check = list(embedding_buffers[cam_id_cleanup].keys())
                    for key in keys_to_check:
                        if key not in embedding_buffers[cam_id_cleanup]:  # 삭제되었을 수 있음
                            continue
                        buffer_data = embedding_buffers[cam_id_cleanup].get(key)
                        if buffer_data is None:
                            continue
                        # 버퍼가 비어있거나 오래되었으면 제거
                        if len(buffer_data.get('embeddings', [])) == 0 or (current_time_cleanup - buffer_data.get('last_update', 0)) > cleanup_threshold:
                            if key in embedding_buffers[cam_id_cleanup]:
                                del embedding_buffers[cam_id_cleanup][key]
                            # 관련 캐시도 제거 (안전한 접근)
                            if cam_id_cleanup in centroid_cache:
                                centroid_cache[cam_id_cleanup].remove(key)
                
                # 오래된 센트로이드 캐시 정리 (TTLCache가 자동으로 만료 처리하므로 간소화)
                # 주기적으로 만료된 항목만 제거 (성능 최적화)
                for cam_id_cleanup in list(centroid_cache.keys()):
                    if cam_id_cleanup in centroid_cache:
                        # TTLCache의 clear_expired() 호출하여 만료된 항목 제거
                        centroid_cache[cam_id_cleanup].clear_expired()
                
                # 정리 시간 업데이트 (cam_id별 상태 관리 사용)
                frame_state['last_cleanup_time'] = current_time_cleanup
            
            # 오래된 넘어짐 감지 시간 추적 정리 (최적화: list() 변환 최소화)
            for cam_id_cleanup in list(fall_start_times.keys()):
                if cam_id_cleanup not in fall_start_times:  # 안전성 체크
                    continue
                # 키 목록을 한 번만 생성 (성능 최적화)
                keys_to_check = list(fall_start_times[cam_id_cleanup].keys())
                for key in keys_to_check:
                    if key not in fall_start_times[cam_id_cleanup]:  # 삭제되었을 수 있음
                        continue
                    fall_time = fall_start_times[cam_id_cleanup].get(key)
                    if fall_time is None:
                        continue
                    if (current_time_cleanup - fall_time) > FALL_DURATION_THRESHOLD * 3:
                        if key in fall_start_times[cam_id_cleanup]:
                            del fall_start_times[cam_id_cleanup][key]
            
            # 오래된 얼굴 바운딩박스 캐시 정리 (TTLCache가 자동으로 만료 처리하므로 간소화)
            # 주기적으로 만료된 항목만 제거 (성능 최적화)
            for cam_id_cleanup in list(face_bbox_cache.keys()):
                if cam_id_cleanup in face_bbox_cache:
                    # TTLCache의 clear_expired() 호출하여 만료된 항목 제거
                    face_bbox_cache[cam_id_cleanup].clear_expired()
            
            # 2단계: name_to_boxes를 처리하여 각 person_index의 최종 이름 결정
            for name, boxes_scores_indices in name_to_boxes.items():
                if len(boxes_scores_indices) == 1:
                    # 이름이 1개만 있으면 그대로 사용
                    (x1, y1, x2, y2), score, person_idx = boxes_scores_indices[0]
                    person_final_names[person_idx] = name
                else:
                    # 같은 이름이 여러 박스에 할당됨: IoU 확인
                    # 가장 높은 similarity의 박스를 기준으로, IoU가 낮은 박스는 Unknown으로 처리
                    boxes_scores_indices_sorted = sorted(boxes_scores_indices, key=lambda x: x[1], reverse=True)
                    best_box, best_score, best_idx = boxes_scores_indices_sorted[0]
                    
                    # 최고 similarity 박스는 그대로 사용
                    person_final_names[best_idx] = name
                    
                    # 나머지 박스는 IoU와 similarity 차이를 고려하여 다른 사람 판단
                    for other_box, other_score, other_idx in boxes_scores_indices_sorted[1:]:
                        iou = utils.calculate_iou(best_box, other_box)
                        score_diff = abs(best_score - other_score)
                        
                        # 더 엄격한 중복 제거: IoU가 낮거나 similarity 차이가 크면 다른 사람
                        # 임계값을 더 엄격하게 조정: IoU 0.5 이상이고 similarity 차이 0.1 이하일 때만 같은 사람
                        if iou < 0.5 or score_diff > 0.1:  # IoU 0.4 -> 0.5 (더 엄격), similarity 차이 0.15 -> 0.1 (더 엄격)
                            person_final_names[other_idx] = "Unknown"
                        else:
                            # IoU가 높고 similarity 차이가 작으면 같은 사람이므로 같은 이름 유지
                            person_final_names[other_idx] = name
            
            # 3단계: 최종 이름으로 렌더링 및 처리
            # person_status를 저장하기 위한 딕셔너리
            person_status_map: Dict[int, str] = {}
            
            for person_data in person_data_list:
                i = person_data['index']
                person_id_text = person_data['person_id']
                x1, y1, x2, y2 = person_data['box']
                # 최종 이름 사용 (중복 제거된 결과)
                person_name = person_final_names.get(i, person_data.get('name', 'Unknown'))
                similarity_score = person_data.get('similarity', 0.0)

                # PPE 위반 목록 가져오기 (통합 함수에서 이미 처리됨)
                ppe_violations = person_data.get('ppe_violations', [])
                
                # 상태 초기화
                person_status = "SAFE"
                status_details = []
                current_violations = list(ppe_violations)  # PPE 위반 복사
                
                # PPE 위반이 있으면 VIOLATION 상태
                if ppe_violations:
                    person_status = "VIOLATION"
                    for rule in ppe_violations:
                        status_details.append(f"{rule}: VIOLATION")

                # 위험 행동 감지 결과 사용 (이미 얼굴 인식 전에 수행됨)
                is_dangerous_detected = person_data.get('is_dangerous', False)
                violation_type = person_data.get('violation_type', '')
                
                # person_box_key 생성 (로깅용)
                person_box_key = None
                if matched_entry is not None:
                    cached_name_fall = matched_entry.get('name', 'Unknown')
                    if cached_name_fall != "Unknown":
                        person_box_key = f"{cam_id}_{cached_name_fall}"
                    else:
                        person_box_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                else:
                    person_box_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                
                # 위험할 때만 상태 변경 및 위반 목록에 추가
                if is_dangerous_detected and violation_type:
                    person_status = "FALL"
                    status_details.append("넘어짐 감지")
                    current_violations.append("넘어짐")
                    logging.warning(f"⚠️ 위험 행동 감지: {person_box_key} - {violation_type}")
                
                # person_status 저장 (recognized_faces 필터링용)
                person_status_map[i] = person_status

                # 렌더링 정책: person_box는 그리지 않고, PPE 감지 박스만 그림
                # 얼굴 인식은 백그라운드에서 계속 진행 (텍스트로만 표시)
                ppe_boxes_list = person_data.get('ppe_boxes', [])
                face_bbox = person_data.get('face_bbox')
                
                # 얼굴 바운딩박스 캐시 처리 (깜빡임 방지)
                current_time = time.time()
                cached_face_bbox = None
                
                # 현재 프레임에서 얼굴이 감지되었으면 캐시 업데이트
                if face_bbox is not None:
                    # TTLCache에 저장 (자동 만료 처리)
                    face_bbox_cache[cam_id].put(person_box_key, {
                        'face_bbox': face_bbox,
                        'person_box': (x1, y1, x2, y2)
                    })
                    cached_face_bbox = face_bbox
                else:
                    # 캐시에서 이전 얼굴 바운딩박스 찾기 (IoU 기반 매칭)
                    # 먼저 person_box_key로 직접 찾기
                    cached_entry = face_bbox_cache[cam_id].get(person_box_key)
                    if cached_entry:
                        cached_person_box = cached_entry.get('person_box', (0, 0, 0, 0))
                        iou = utils.calculate_iou((x1, y1, x2, y2), cached_person_box)
                        
                        # IoU 임계값 완화: 0.3 -> 0.25 (바운딩 박스 안정화, 깜빡임 방지)
                        if iou >= 0.25:
                            cached_face_bbox = cached_entry.get('face_bbox')
                            # 캐시 업데이트 (TTLCache에 다시 저장하여 TTL 갱신)
                            face_bbox_cache[cam_id].put(person_box_key, {
                                'face_bbox': cached_face_bbox,
                                'person_box': (x1, y1, x2, y2)
                            })
                    else:
                        # person_box_key로 찾지 못하면 IoU 기반으로 모든 캐시 항목 검색
                        # TTLCache는 keys()를 지원하지 않으므로 다른 방법 사용
                        # 대신 person_box_key 기반 매칭만 사용 (성능 최적화)
                        # IoU 기반 전체 검색은 제거 (TTLCache 특성상 어려움)
                        cached_face_bbox = None
                
                # 캐시된 얼굴 바운딩박스 사용 (없으면 None)
                face_bbox_to_draw = face_bbox if face_bbox is not None else cached_face_bbox
                
                # 바운딩 박스 렌더링은 render_frame_results에서 통합 처리하므로 여기서는 비활성화
                # 기존 개별 렌더링 로직은 render_frame_results로 통합됨
                if False:  # render_frame_results에서 통합 렌더링하므로 비활성화
                    # 헬멧 박스 찾기 (얼굴 인식 결과 표시용)
                    helmet_box = None
                    for ppe_box_info in ppe_boxes_list:
                        ppe_class = ppe_box_info['class']
                        # Hardhat 또는 NO-Hardhat 클래스 찾기
                        if "Hardhat" in ppe_class:
                            helmet_box = ppe_box_info
                            break
                    
                    # 얼굴 바운딩박스 그리기 (얼굴이 감지되거나 캐시에 있으면 표시)
                    if face_bbox_to_draw is not None:
                        # person_img_for_detection의 좌표를 원본 프레임 좌표로 변환
                        # person_img_for_detection은 person_box 영역을 추출한 이미지
                        # face_bbox는 person_img_for_detection 내의 좌표
                        fx1, fy1, fx2, fy2 = face_bbox_to_draw
                        
                        # person_img_for_detection이 리사이즈되었을 수 있으므로 원본 person_box 크기로 스케일링
                        # person_img_for_detection의 원본 크기 확인
                        person_img = person_data.get('img')
                        if person_img is not None:
                            person_img_h, person_img_w = person_img.shape[:2]
                            # person_box의 실제 크기
                            person_box_w = x2 - x1
                            person_box_h = y2 - y1
                            
                            # 스케일 계산
                            scale_x = person_box_w / person_img_w if person_img_w > 0 else 1.0
                            scale_y = person_box_h / person_img_h if person_img_h > 0 else 1.0
                            
                            # 원본 프레임 좌표로 변환
                            face_x1 = int(x1 + fx1 * scale_x)
                            face_y1 = int(y1 + fy1 * scale_y)
                            face_x2 = int(x1 + fx2 * scale_x)
                            face_y2 = int(y1 + fy2 * scale_y)
                            
                            # 얼굴 박스 좌표 저장 (통합 박스용)
                            # (그리지는 않음)

                    # person_box 기준으로 바운딩 박스 그리기 (각 사람 독립적으로)
                    # 색상 및 투명도 결정
                    if person_status == "FALL":
                        unified_color = (0, 50, 255)  # 밝은 빨간색 (위험)
                        alpha = 0.25
                    elif current_violations:
                        unified_color = (0, 140, 255)  # 밝은 주황색 (위반)
                        alpha = 0.2
                    else:
                        unified_color = (50, 255, 50)  # 밝은 초록색 (준수)
                        alpha = 0.15

                    # person_box로 바운딩 박스 그리기 (현대적 스타일)
                    draw_modern_bbox(processed_frame, x1, y1, x2, y2, unified_color, thickness=3, corner_length=35, alpha=alpha)
                    
                    # 상태 텍스트 표시 (person_box 위에 표시)
                    if current_violations or person_status != "SAFE" or person_name != "Unknown":
                        # person_box 위치에 텍스트 표시
                        text_x, text_y = x1, y1

                        # 색상 결정 (박스와 동일)
                        text_color = unified_color

                        # 텍스트 내용 구성
                        status_text = ""
                        if person_name != "Unknown":
                            status_text = f"{person_name}"
                        else:
                            status_text = f"{person_id_text}"

                        if current_violations:
                            status_text += f": {', '.join(current_violations)}"
                        elif person_status == "SAFE":
                            status_text += " (안전)"

                        # person_box 위에 텍스트 표시
                        renderer.add_text(status_text, (text_x, text_y - 10), text_color)

                # 위반 사항 기록 (중복 제거: 같은 사람 박스에 대해 한 번만 기록)
                if current_violations:
                    # 박스 기반 중복 제거 키 생성
                    box_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                    
                    # 이미 같은 박스에 대한 위반이 기록되었는지 확인
                    is_duplicate = False
                    for existing_violation in violations_found:
                        ex_box = existing_violation.get("person_box", [])
                        if len(ex_box) == 4:
                            ex_x1, ex_y1, ex_x2, ex_y2 = ex_box
                            ex_box_key = f"{int(ex_x1)}_{int(ex_y1)}_{int(ex_x2)}_{int(ex_y2)}"
                            # IoU 계산하여 중복 확인
                            iou = utils.calculate_iou((x1, y1, x2, y2), (ex_x1, ex_y1, ex_x2, ex_y2))
                            if iou > 0.6:  # IoU 0.6 이상이면 같은 사람으로 간주
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        # cam_id를 area로 매핑 (0→A-1, 1→A-2, 2→B-1, 3→B-2)
                        area_map = {0: "A-1", 1: "A-2", 2: "B-1", 3: "B-2"}
                        area = area_map.get(cam_id, f"A-{cam_id+1}")
                        
                        # 위반 내용을 hazard 문자열로 변환
                        # 예: "PPE 위반내역: 안전모, 안전조끼"
                        # 중복 제거: 위반 유형을 set으로 변환하여 중복 제거
                        unique_violations = list(set(current_violations))  # 중복 제거
                        ppe_violations = []
                        other_violations = []
                        
                        for violation_type in unique_violations:
                            if violation_type == "넘어짐":
                                other_violations.append("넘어짐 감지")
                            elif violation_type == "안전모":
                                ppe_violations.append("안전모")
                            elif violation_type == "안전조끼":
                                ppe_violations.append("안전조끼")
                            else:
                                other_violations.append(f"위반: {violation_type}")
                        
                        # PPE 위반이 있으면 쉼표로 구분하여 표시
                        if ppe_violations:
                            hazard = f"PPE 위반내역: {', '.join(ppe_violations)}"
                            if other_violations:
                                hazard += f", {', '.join(other_violations)}"
                        elif other_violations:
                            hazard = ", ".join(other_violations)
                        else:
                            hazard = "위반 감지"
                        
                        # worker 이름: recognized_name이 있으면 사용, 없으면 "알 수 없음"
                        worker = person_name if person_name != "Unknown" else "알 수 없음"
                        
                        violations_found.append({
                            "person_box": [x1, y1, x2, y2],
                            "violations": unique_violations,  # 중복 제거된 위반 목록
                            "recognized_name": person_name,
                            "worker": worker,
                            "area": area,
                            "level": "WARNING",
                            "hazard": hazard
                        })

            # 프레임 내 동일 이름 중복 제거: person_final_names에서 최종 이름으로 recognized_faces 구성
            # KPI 계산을 위해 모든 감지된 사람을 추가 (VIOLATION 여부와 관계없이)
            # render_frame_results를 위해 ppe_violations 정보도 추가
            added_names = set()
            added_boxes = set()  # 박스 기반 중복 제거 (Unknown도 카운트하기 위해)
            for person_data in person_data_list:
                i = person_data['index']
                x1, y1, x2, y2 = person_data['box']
                final_name = person_final_names.get(i, person_data.get('name', 'Unknown'))
                similarity_score = person_data.get('similarity', 0.0)
                person_status = person_status_map.get(i, "SAFE")
                ppe_violations = person_data.get('ppe_violations', [])
                
                # 박스 기반 중복 제거 (같은 위치의 사람은 1명으로만 카운트)
                box_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                
                # 마스크 제외한 위반만 체크
                filtered_ppe = [v for v in ppe_violations if v != "마스크"]
                is_violation = person_status != "SAFE" or len(filtered_ppe) > 0
                
                # KPI 계산을 위해 모든 감지된 사람 추가 (Unknown 포함, 각 사람당 1개만)
                # 이름이 있으면 이름 기반 중복 제거, 없으면 박스 기반 중복 제거
                if final_name != "Unknown":
                    # 이름이 있는 경우: 이름 기반 중복 제거
                    if final_name not in added_names:
                        recognized_faces.append({
                            "box": [x1, y1, x2, y2],
                            "name": final_name,
                            "similarity": float(similarity_score),
                            "status": person_status,  # 상태 정보 추가 (KPI 계산용)
                            "ppe_violations": filtered_ppe,  # render_frame_results용
                            "isViolation": is_violation  # render_frame_results용
                        })
                        added_names.add(final_name)
                        added_boxes.add(box_key)
                else:
                    # Unknown인 경우: 박스 기반 중복 제거 (같은 위치의 Unknown은 1명으로만 카운트)
                    if box_key not in added_boxes:
                        recognized_faces.append({
                            "box": [x1, y1, x2, y2],
                            "name": "Unknown",
                            "similarity": 0.0,
                            "status": person_status,
                            "ppe_violations": filtered_ppe,  # render_frame_results용
                            "isViolation": is_violation  # render_frame_results용
                        })
                        added_boxes.add(box_key)

        # 8. 기타 객체 그리기 (안전 장비는 위에서 이미 처리했으므로 제외)
        for class_name, detections in all_detections.items():
            # 'person' 클래스는 pose_results에서 이미 처리하므로 제외
            if class_name.lower() == 'person':
                continue
            # Safety Con 등 오탐지 클래스 필터링
            if class_name in config.Thresholds.IGNORED_CLASSES:
                continue
            # 안전 장비 클래스는 사람 박스와 함께 위에서 처리하므로 제외
            is_safety_gear = any(class_name in item.values() for item in config.Constants.SAFETY_RULES_MAP.values())
            if not is_safety_gear and detections:
                color = (255, 0, 0)  # 파란색 (BGR)
                for det in detections:
                    if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                        x1_obj, y1_obj, x2_obj, y2_obj = map(int, det['bbox'])

                        # 손/작은 객체 필터링: 사람 박스와 겹치는 작은 객체는 무시
                        obj_area = (x2_obj - x1_obj) * (y2_obj - y1_obj)
                        obj_center_x = (x1_obj + x2_obj) / 2
                        obj_center_y = (y1_obj + y2_obj) / 2

                        # 사람 박스와의 IOU 확인 및 필터링 (최적화: 거리 기반 필터링 먼저)
                        should_filter = False
                        for person_box in scaled_person_boxes:
                            px1, py1, px2, py2 = person_box
                            person_area = (px2 - px1) * (py2 - py1)
                            
                            # 거리 기반 필터링 먼저 수행 (IoU 계산보다 빠름)
                            person_center_x = (px1 + px2) / 2
                            person_center_y = (py1 + py2) / 2
                            center_distance = ((obj_center_x - person_center_x) ** 2 + (obj_center_y - person_center_y) ** 2) ** 0.5
                            person_diagonal = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
                            
                            # 거리가 너무 멀면 IoU 계산 생략 (성능 향상)
                            if center_distance > person_diagonal * 1.5:
                                continue

                            # 작은 객체가 사람 박스 내부나 가까이 있으면 필터링
                            if (px1 <= obj_center_x <= px2 and py1 <= obj_center_y <= py2) or \
                               (x1_obj < px2 and x2_obj > px1 and y1_obj < py2 and y2_obj > py1):
                                # IOU 계산 (거리 필터링 통과한 경우만)
                                iou = utils.calculate_iou((px1, py1, px2, py2), (x1_obj, y1_obj, x2_obj, y2_obj))

                                # 작은 객체(machinery, hand 등)이고 사람 박스와 겹치면 필터링
                                # 또는 객체가 사람 박스 면적의 10% 미만이고 IOU가 0.1 이상이면 필터링
                                if obj_area < person_area * 0.1 and iou > 0.05:
                                    should_filter = True
                                    break

                        # machinery 클래스는 특히 엄격하게 필터링 (사람 박스와 겹치면 무시)
                        if class_name.lower() in ['machinery', 'hand', 'hands'] and should_filter:
                            continue

                        # 원본 프레임에 직접 그리기 (현대적 스타일, 이미 스케일링된 좌표)
                        draw_modern_bbox(processed_frame, x1_obj, y1_obj, x2_obj, y2_obj, color, thickness=1, corner_length=15, alpha=0.15)
                        display_name = class_name[:10]
                        renderer.add_text(f"{display_name}", (x1_obj, y1_obj - 5), color)

        # 스킵/누락 상황에서 박스/라벨 유지: 캐시로 보강 (강화 버전)
        # 렌더링 전에 recognized_faces가 비어있으면 캐시에서 강제로 가져와서 박스 그리기
        try:
            hold_sec = config.Thresholds.RECOGNITION_HOLD_SECONDS
            now_ts = time.time()
            
            # IdentityCache에서 최근 항목 가져오기 (자동 만료 처리)
            cache_entries = recent_identity_cache.get_recent(cam_id, max_age=hold_sec)
            
            if cache_entries:
                # recognized_faces가 비어있거나 부족하면 캐시에서 보강 (단, 최근 항목만)
                preserved = []
                
                for entry in cache_entries:
                    age = now_ts - entry.get('ts', 0)
                    if age <= hold_sec:  # 홀드 시간까지만 유지 (잔상 방지)
                        x1, y1, x2, y2 = entry.get('box', (0,0,0,0))
                        name = entry.get('name', 'Unknown')
                        score = float(entry.get('score', 0.0))
                        
                        # Unknown이 아니고 유효한 박스면 추가
                        if name != "Unknown" and (x2 > x1 and y2 > y1):
                            # 기존 recognized_faces에 같은 박스가 있는지 확인 (중복 방지)
                            # IoU 임계값 완화: 0.5 -> 0.4 (바운딩 박스 안정화)
                            is_duplicate = False
                            for existing in recognized_faces:
                                ex_box = existing.get("box", [])
                                if len(ex_box) == 4:
                                    ex_iou = utils.calculate_iou((x1, y1, x2, y2), tuple(ex_box))
                                    if ex_iou > 0.4:  # IoU 0.4 이상이면 중복 (바운딩 박스 안정화)
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate:
                                # 바운딩 박스 안정화: 캐시에서 가져온 항목도 recognized_faces에 추가
                                # 렌더링은 1634줄의 조건에서 처리됨 (모든 박스 렌더링)
                                preserved.append({
                                    "box": [int(x1), int(y1), int(x2), int(y2)],
                                    "name": name,
                                    "similarity": score
                                })
                
                # 보강된 항목을 recognized_faces에 추가
                if preserved:
                    recognized_faces.extend(preserved)

            # 바운딩 박스 안정화: 캐시 보강으로 박스가 사라지지 않도록 함
            # 렌더링은 1634줄의 조건에서 처리됨 (모든 박스 렌더링)
        except Exception as e:
            logging.debug(f"[CAM-{cam_id}] 캐시 보강 중 오류 발생: {e}")
            pass  # 캐시 보강 실패는 무시

        # 동일 이름 중복 제거: 가장 높은 similarity만 유지 (대시보드/KPI용)
        # Unknown도 포함하여 모든 사람을 카운트 (중복 제거는 이름 기반, Unknown은 박스 기반)
        try:
            if recognized_faces:
                best_by_name = {}
                unknown_faces = []  # Unknown은 별도로 처리
                added_unknown_boxes = set()  # Unknown 중복 제거용 (박스 기반)
                
                for face in recognized_faces:
                    name = face.get("name", "Unknown")
                    if name == "Unknown":
                        # Unknown은 박스 기반 중복 제거
                        box = tuple(face.get("box", []))
                        if len(box) == 4:
                            box_key = f"{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}"
                            if box_key not in added_unknown_boxes:
                                unknown_faces.append(face)
                                added_unknown_boxes.add(box_key)
                    else:
                        # 이름이 있는 경우: 이름 기반 중복 제거 (가장 높은 similarity만 유지)
                        score = float(face.get("similarity", 0.0))
                        if name not in best_by_name or score > float(best_by_name[name].get("similarity", 0.0)):
                            best_by_name[name] = face
                
                # 이름이 있는 사람들 + Unknown 사람들 합치기
                recognized_faces = list(best_by_name.values()) + unknown_faces
        except Exception as e:
            logging.warning(f"[CAM-{cam_id}] 중복 제거 중 오류 발생: {e}", exc_info=True)
            # 기본값 사용: person_data_list의 모든 사람을 recognized_faces로 변환
            recognized_faces = []
            for person_data in person_data_list:
                name = person_data.get('name', 'Unknown')
                similarity = person_data.get('similarity', 0.0)
                box = person_data.get('box', (0, 0, 0, 0))
                recognized_faces.append({
                    'name': name,
                    'similarity': similarity,
                    'box': box
                })

        # 8. 렌더링 (render_frame_results 사용하여 바운딩 박스 통합 렌더링)
        rendering_start = time.time()
        # render_frame_results를 사용하여 recognized_faces와 violations를 통합 렌더링
        processed_frame, unified_boxes = render_frame_results(
            processed_frame,
            recognized_faces,
            violations_found,
            cam_id,
            orig_w,
            orig_h
        )

        # 이번 프레임 렌더링 결과를 캐시에 저장 (다음 프레임 보강용)
        try:
            if recognized_faces:
                # TTLCache에 저장 (자동 만료 처리)
                last_render_cache[cam_id].put('render', {
                    'items': [{'box': tuple(face.get('box', (0,0,0,0))), 'name': face.get('name', 'Unknown'), 'similarity': face.get('similarity', 0.0)} for face in recognized_faces]
                })
        except Exception:
            pass
        perf_timings['rendering'] = (time.time() - rendering_start) * 1000  # ms

        # 9. 처리된 프레임을 JPEG 바이트로 인코딩 (품질 최적화: 100 → 85, 속도 향상)
        encoding_start = time.time()
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        perf_timings['encoding'] = (time.time() - encoding_start) * 1000  # ms
        if not ret:
            logging.error("JPEG 인코딩 실패")
            empty_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            return buffer.tobytes(), {}

        processed_frame_bytes = buffer.tobytes()

        # 전체 시간 계산
        perf_timings['total'] = (time.time() - total_start) * 1000  # ms
        
        # 최종 결과 요약 로깅
        recognized_count = len(recognized_faces)
        violations_count = len(violations_found)
        known_faces = [f for f in recognized_faces if f.get('name', 'Unknown') != 'Unknown']
        unknown_faces = recognized_count - len(known_faces)
        
        logging.info(f"[RESULT CAM-{cam_id}] ✅ 프레임 처리 완료: "
                    f"총 처리={perf_timings['total']:.1f}ms, "
                    f"인식된 얼굴={recognized_count}개(알려진={len(known_faces)}개, 알수없음={unknown_faces}개), "
                    f"위반={violations_count}개, "
                    f"YOLO Violation={perf_timings.get('yolo_violation', 0):.1f}ms, "
                    f"YOLO Pose={perf_timings.get('yolo_pose', 0):.1f}ms, "
                    f"Face Recognition={perf_timings.get('face_recognition', 0):.1f}ms, "
                    f"Rendering={perf_timings.get('rendering', 0):.1f}ms")
        
        # 알려진 얼굴 상세 정보
        if known_faces:
            face_details = [f"{f.get('name', 'Unknown')}({f.get('similarity', 0):.2f})" for f in known_faces[:5]]
            logging.debug(f"[RESULT CAM-{cam_id}] 알려진 얼굴 상세: {face_details}")
        
        # 위반 상세 정보
        if violations_count > 0:
            violation_types = {}
            for v in violations_found:
                for violation in v.get('violations', []):
                    violation_types[violation] = violation_types.get(violation, 0) + 1
            logging.info(f"[RESULT CAM-{cam_id}] 위반 상세: {violation_types}")
        
        # 적응형 워커 관리자에 성능 메트릭 업데이트
        try:
            from state import adaptive_worker_manager, frame_queues, queue_lock
            if adaptive_worker_manager:
                # 큐 크기 가져오기
                queue_size = 0
                with queue_lock:
                    if cam_id in frame_queues:
                        queue_size = frame_queues[cam_id].qsize()
                
                # 성능 메트릭 업데이트
                adaptive_worker_manager.update_metrics(
                    processing_time_ms=perf_timings.get('total', 0),
                    queue_size=queue_size,
                    latency_ms=perf_timings.get('total', 0),
                    fps=1000.0 / perf_timings.get('total', 1) if perf_timings.get('total', 0) > 0 else 0
                )
        except Exception as e:
            pass  # 적응형 워커 관리자 메트릭 업데이트 실패는 무시
        
        # 성능 데이터 로깅 (조건부 상세 로깅)
        # cam_id별 상태 관리 사용 (함수 속성 대신)
        frame_state = state.get_frame_processing_state(cam_id)
        frame_state['perf_log_count'] = frame_state.get('perf_log_count', 0) + 1
        
        total_time = perf_timings.get('total', 0)
        should_log = False
        log_level = logging.DEBUG
        
        # 조건 1: 총 처리 시간이 500ms 이상이면 항상 로깅
        if total_time > 500:
            should_log = True
            log_level = logging.WARNING
        
        # 조건 2: 총 처리 시간이 1000ms 이상이면 항상 상세 로깅
        if total_time > 1000:
            should_log = True
            log_level = logging.ERROR
        
        # 조건 3: 30프레임마다 한 번 로깅 (정기 모니터링)
        if frame_state['perf_log_count'] % 30 == 0:
            should_log = True
            log_level = logging.INFO
        
        # 조건 4: 병목이 전체의 30% 이상이면 로깅
        bottleneck = max(perf_timings.items(), key=lambda x: x[1] if x[0] != 'total' else 0)
        if bottleneck[1] > total_time * 0.3 and total_time > 100:
            should_log = True
            log_level = logging.WARNING
        
        if should_log:
            # 상세 성능 로깅
            perf_details = []
            for key in ['decode', 'resize', 'yolo_violation', 'yolo_pose', 'face_recognition', 
                       'parse_results', 'rendering', 'encoding', 'total']:
                if key in perf_timings:
                    value = perf_timings[key]
                    percentage = (value / total_time * 100) if total_time > 0 else 0
                    perf_details.append(f"{key}={value:.1f}ms({percentage:.0f}%)")
            
            perf_str = ", ".join(perf_details)
            
            if log_level >= logging.WARNING:
                logging.warning(f"[PERF CAM-{cam_id}] 성능 상세: {perf_str}")
                if bottleneck[1] > total_time * 0.3:
                    logging.warning(f"[PERF CAM-{cam_id}] ⚠️ 병목 지점: {bottleneck[0]} ({bottleneck[1]:.1f}ms, {bottleneck[1]/total_time*100:.0f}%)")
            elif log_level == logging.INFO:
                logging.info(f"[PERF CAM-{cam_id}] 성능: 총 {total_time:.1f}ms, 병목: {bottleneck[0]} ({bottleneck[1]:.1f}ms)")
            else:
                logging.debug(f"[PERF CAM-{cam_id}] 성능: {perf_str}")

        # 10. 결과 데이터 구성
        result_data = {
            "recognized_faces": recognized_faces,
            "violations": violations_found,
            "violation_count": len(violations_found),
            "performance": perf_timings,  # 성능 측정 데이터 포함
            "frame_width": orig_w,  # 원본 프레임 너비 (바운딩 박스 좌표 기준)
            "frame_height": orig_h,  # 원본 프레임 높이 (바운딩 박스 좌표 기준)
            "cam_id": cam_id,  # 카메라 ID 추가 (프론트엔드 디버깅용)
            "unified_boxes": unified_boxes  # 통합된 박스 정보 (프론트엔드 렌더링용, render_frame_results 결과)
        }

        return processed_frame_bytes, result_data

    except Exception as e:
        total_failed = (time.time() - total_start) * 1000
        error_msg = str(e)
        logging.error(f"AI 처리 실행 중 오류 (CAM-{cam_id}, 누적 {total_failed:.2f}ms): {e}", exc_info=True)
        
        # 오류 프레임 생성 (더 자세한 정보 포함)
        # orig_h, orig_w가 초기화되지 않았을 수 있으므로 안전하게 처리
        try:
            if frame is not None:
                error_frame = frame.copy()
                error_h, error_w = frame.shape[:2]
            else:
                # 기본값 사용
                error_h, error_w = orig_h if 'orig_h' in locals() and orig_h > 0 else 480, orig_w if 'orig_w' in locals() and orig_w > 0 else 640
                error_frame = np.zeros((error_h, error_w, 3), dtype=np.uint8)
        except Exception:
            # 모든 시도가 실패하면 최소한의 에러 프레임 생성
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            error_h, error_w = 480, 640
        
        # 오류 메시지를 여러 줄로 표시
        error_lines = [
            "Processing Error",
            error_msg[:50] + ("..." if len(error_msg) > 50 else ""),
            "Check backend logs"
        ]
        
        y_offset = 30
        for i, line in enumerate(error_lines):
            try:
                cv2.putText(error_frame, line, (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception:
                pass  # 텍스트 그리기 실패는 무시
        
        try:
            ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes(), {"error": error_msg, "recognized_faces": [], "violations": []}
        except Exception:
            # 인코딩 실패 시 빈 바이트 반환
            return b'', {"error": error_msg, "recognized_faces": [], "violations": []}
