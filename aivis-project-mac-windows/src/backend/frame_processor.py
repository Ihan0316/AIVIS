# frame_processor.py - 프레임 처리 로직
"""
단일 프레임 처리 모듈
AI 모델을 사용하여 프레임을 처리하고 결과를 반환합니다.
"""
import logging
import time
import torch  # 추가: torch 임포트
from typing import Dict, Tuple, Any, List, Optional
from concurrent.futures import as_completed, TimeoutError as FuturesTimeoutError

import cv2
import numpy as np
from ultralytics.engine.results import Keypoints

import utils
import config
from utils import find_best_match_faiss, find_best_matches_faiss_batch, draw_modern_bbox, draw_fast_bbox, calculate_iou_batch
from exceptions import (
    ProcessingError,
    FaceRecognitionError,
    ValidationError
)
import state
from state import (
    safety_system_lock,
    frame_stats,
    frame_stats_lock,
    yolo_executor,
    face_recognition_executor,
    dangerous_behavior_executor,
    frame_processing_executor,
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
    face_recognition_cooldown_ts,
    frame_buffer,
    MAX_BUFFER_SECONDS,
    frame_buffer_lock,
    model_results_cache,
    results_cache_lock,
    CACHE_TTL
)
from ai_processors import (
    _process_ppe_detection,
    _process_face_recognition,
    _process_dangerous_behavior
)
from pipeline_manager import (
    person_tracker,
    face_recognition_queue,
    get_person_crop,
    should_run_face_detection_for_frame,
    PersonTracker,
    FaceRecognitionQueue
)
from state import (
    track_states,
    track_states_lock,
    new_track_ids,
    NEW_TRACK_THRESHOLD,
    last_face_recognition_by_track,
    FACE_RECOGNITION_INTERVAL_PER_TRACK
)

# 마지막 렌더링된 프레임 캐시 (스킵 프레임에서 바운딩 박스 유지용)
_last_rendered_frames = {}  # {cam_id: (frame_bytes, result_dict)}


# ========================================
# 2단계 파이프라인 헬퍼 함수
# ========================================

def _update_track_state(cam_id: int, track_id: int, person_box: Tuple[int, int, int, int],
                        has_violation: bool = False, violation_types: List[str] = None) -> Dict:
    """
    Track ID 상태 업데이트 (2단계 파이프라인용)
    
    Args:
        cam_id: 카메라 ID
        track_id: 추적 ID
        person_box: 사람 바운딩 박스
        has_violation: 위반 발생 여부
        violation_types: 위반 유형 리스트
        
    Returns:
        업데이트된 상태 딕셔너리
    """
    now = time.time()
    
    with track_states_lock:
        if track_id not in track_states[cam_id]:
            # 새로운 Track ID
            track_states[cam_id][track_id] = {
                'name': None,
                'confidence': 0.0,
                'last_recognition': 0.0,
                'violations': [],
                'person_box': person_box,
                'first_seen': now,
                'last_update': now,
                'face_bbox': None,
                'embedding': None
            }
            # 새로운 Track ID 기록
            new_track_ids[cam_id][track_id] = now
        else:
            # 기존 Track ID 업데이트
            track_states[cam_id][track_id]['person_box'] = person_box
            track_states[cam_id][track_id]['last_update'] = now
        
        # 위반 상태 업데이트
        if has_violation and violation_types:
            track_states[cam_id][track_id]['violations'] = violation_types
        
        return track_states[cam_id][track_id]


def _should_recognize_face_for_track(cam_id: int, track_id: int, 
                                      has_violation: bool = False) -> Tuple[bool, str]:
    """
    해당 Track ID에 대해 얼굴 인식을 수행해야 하는지 판단
    
    조건:
    - 조건 A: 위반 사항 발생 시 (쓰러짐 OR 안전장비 미착용)
    - 조건 B: 새로운 사람 등장 시 (Track ID가 처음 생성되었을 때)
    - 조건 C: 주기적 확인 (해당 ID에 대해 1초에 1번만)
    
    Returns:
        (should_recognize, reason)
    """
    now = time.time()
    
    with track_states_lock:
        if track_id not in track_states[cam_id]:
            return False, "unknown_track"
        
        state = track_states[cam_id][track_id]
        
        # 조건 A: 위반 사항 발생 시 (최우선)
        if has_violation:
            last_face_recognition_by_track[cam_id][track_id] = now
            return True, "violation_detected"
        
        # 조건 B: 새로운 사람 등장 시
        if track_id in new_track_ids[cam_id]:
            first_seen = new_track_ids[cam_id][track_id]
            if now - first_seen < NEW_TRACK_THRESHOLD:
                # 새로운 사람이고 아직 인식 안 됨
                if state['name'] is None:
                    last_face_recognition_by_track[cam_id][track_id] = now
                    return True, "new_person"
        
        # 조건 C: 주기적 확인 (1초에 1번)
        last_recognition = last_face_recognition_by_track[cam_id].get(track_id, 0)
        if now - last_recognition >= FACE_RECOGNITION_INTERVAL_PER_TRACK:
            # 아직 인식되지 않은 경우에만
            if state['name'] is None or state['confidence'] < 0.5:
                last_face_recognition_by_track[cam_id][track_id] = now
                return True, "periodic_check"
        
        return False, "no_need"


def _set_track_recognized(cam_id: int, track_id: int, name: str, 
                          confidence: float, face_bbox: Tuple = None,
                          embedding: np.ndarray = None) -> None:
    """
    Track ID에 얼굴 인식 결과 저장
    """
    with track_states_lock:
        if track_id in track_states[cam_id]:
            track_states[cam_id][track_id]['name'] = name
            track_states[cam_id][track_id]['confidence'] = confidence
            track_states[cam_id][track_id]['last_recognition'] = time.time()
            if face_bbox:
                track_states[cam_id][track_id]['face_bbox'] = face_bbox
            if embedding is not None:
                track_states[cam_id][track_id]['embedding'] = embedding
            
            # 인식 성공 시 새로운 사람 목록에서 제거
            if name and name != "Unknown" and track_id in new_track_ids[cam_id]:
                del new_track_ids[cam_id][track_id]


def _cleanup_expired_tracks(cam_id: int, expiry_time: float = 30.0) -> None:
    """
    만료된 Track ID 정리
    """
    now = time.time()
    
    with track_states_lock:
        expired_tracks = []
        for track_id, state in track_states[cam_id].items():
            if now - state['last_update'] > expiry_time:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del track_states[cam_id][track_id]
            if track_id in new_track_ids[cam_id]:
                del new_track_ids[cam_id][track_id]
            if track_id in last_face_recognition_by_track[cam_id]:
                del last_face_recognition_by_track[cam_id][track_id]
        
        if expired_tracks:
            logging.debug(f"[CAM-{cam_id}] 만료된 Track ID 정리: {len(expired_tracks)}개")

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
    (얼굴 박스와 person_box 통합 및 렌더링)
    """
    processed_frame = frame.copy()
    
    # 1. recognized_faces와 violations를 통합
    all_boxes = []
    box_to_info = {}  # box_tuple -> (name, ppe_violations, is_violation)
    
    # recognized_faces 처리: 안전한 사람(위반 없음)도 박스 표시
    face_boxes_info = {}  # 얼굴 박스 정보 임시 저장 (person_box와 매칭용)
    for face in recognized_faces:
        box = face.get("box") or face.get("bbox")
        if box and len(box) == 4:
            box_tuple = tuple(map(int, box))
            name = face.get("name", "Unknown")
            ppe_violations = face.get("ppe_violations", [])
            # 마스크 제외
            filtered_ppe = [v for v in ppe_violations if v != "마스크"]
            is_violation = face.get("isViolation", False) or len(filtered_ppe) > 0
            
            # 얼굴 박스 정보 저장
            face_boxes_info[box_tuple] = {
                'name': name,
                'ppe_violations': filtered_ppe,
                'is_violation': is_violation
            }
            
            # ⭐ 안전한 사람(위반 없음)도 all_boxes에 추가 - 초록 박스 표시!
            if not is_violation:
                if box_tuple not in all_boxes:
                    all_boxes.append(box_tuple)
                    box_to_info[box_tuple] = (name, filtered_ppe, False)  # is_violation=False
    
    # violations 처리: person_box를 기준으로 얼굴 박스와 매칭하여 이름 통합
    # ⭐ 이미 매칭된 얼굴 박스 추적 (중복 매칭 방지)
    matched_face_boxes = set()
    
    for violation in violations:
        box = violation.get("person_box") or violation.get("bbox") or violation.get("box")
        if box and len(box) == 4:
            box_tuple = tuple(map(int, box))
            # 위반 정보에서 얼굴 인식된 이름 가져오기 (우선순위: recognized_name > worker)
            recognized_name = violation.get("recognized_name", "Unknown")
            worker = violation.get("worker", "알 수 없음")
            # recognized_name이 "Unknown"이 아니면 사용, 아니면 worker 사용
            violation_name = recognized_name if recognized_name != "Unknown" else (worker if worker != "알 수 없음" else "Unknown")
            violations_list = violation.get("violations", [])
            
            # 얼굴 박스와 매칭 (얼굴 박스가 person_box 내부에 있으면 매칭)
            # ⭐ 가장 IoU가 높은 얼굴만 매칭 (이미 매칭된 얼굴 제외)
            matched_face_box = None
            matched_face_info = None
            best_iou = 0.0
            
            for face_box_tuple, face_info in face_boxes_info.items():
                # ⭐ 이미 매칭된 얼굴은 건너뛰기
                if face_box_tuple in matched_face_boxes:
                    continue
                
                # IoU 계산
                iou = utils.calculate_iou(box_tuple, face_box_tuple)
                if iou > 0.3 and iou > best_iou:  # IoU 0.3 이상이고 더 높은 IoU면 업데이트
                    best_iou = iou
                    matched_face_box = face_box_tuple
                    matched_face_info = face_info
                    continue  # 더 좋은 매칭을 찾기 위해 계속 검색
                
                # 얼굴 박스 중심점이 person_box 내부에 있는지 확인
                fx1, fy1, fx2, fy2 = face_box_tuple
                v_x1, v_y1, v_x2, v_y2 = box_tuple
                face_center_x = (fx1 + fx2) / 2
                face_center_y = (fy1 + fy2) / 2
                if (v_x1 <= face_center_x <= v_x2 and v_y1 <= face_center_y <= v_y2):
                    # 얼굴 박스가 person_box 내부에 있으면 매칭 (IoU보다 우선)
                    if matched_face_box is None or iou < 0.3:  # IoU 매칭이 없을 때만
                        matched_face_box = face_box_tuple
                        matched_face_info = face_info
            
            # ⭐ 매칭된 얼굴 박스 기록
            if matched_face_box:
                matched_face_boxes.add(matched_face_box)
            
            # 얼굴 박스와 매칭된 경우: 이름과 위반 정보 통합
            if matched_face_info:
                face_name = matched_face_info['name']
                face_ppe = matched_face_info.get('ppe_violations', [])
                # 얼굴 인식 결과의 이름이 "Unknown"이 아니면 우선 사용
                final_name = face_name if face_name != "Unknown" else violation_name
                # 위반 정보 병합 (중복 제거) - 마스크 제외
                merged_ppe = [v for v in list(set(face_ppe + violations_list)) if v != "마스크"]
                # ⭐ 위반이 있을 때만 is_violation=True
                box_to_info[box_tuple] = (final_name, merged_ppe, len(merged_ppe) > 0)
            else:
                # 얼굴 박스와 매칭되지 않은 경우: 위반 정보만 사용
                # 마스크 제외
                filtered_violations = [v for v in violations_list if v != "마스크"]
                # ⭐ 위반이 있을 때만 is_violation=True
                box_to_info[box_tuple] = (violation_name, filtered_violations, len(filtered_violations) > 0)
            
            # person_box는 항상 all_boxes에 추가 (얼굴 박스는 추가하지 않음)
            if box_tuple not in all_boxes:
                all_boxes.append(box_tuple)
    
    # 얼굴 박스가 person_box와 매칭되지 않은 경우 처리 (이름만 표시, 위반 없음)
    for face_box_tuple, face_info in face_boxes_info.items():
        # 이미 person_box와 매칭되었는지 확인
        is_matched = False
        for person_box_tuple in all_boxes:
            iou = utils.calculate_iou(face_box_tuple, person_box_tuple)
            if iou > 0.3:
                is_matched = True
                break
            # 얼굴 박스 중심점이 person_box 내부에 있는지 확인
            fx1, fy1, fx2, fy2 = face_box_tuple
            px1, py1, px2, py2 = person_box_tuple
            face_center_x = (fx1 + fx2) / 2
            face_center_y = (fy1 + fy2) / 2
            if (px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2):
                is_matched = True
                break
        
        # 매칭되지 않은 얼굴 박스는 제외 (이름만 있는 경우는 표시하지 않음)
        # 위반이 있거나 이름이 "Unknown"이 아니면 person_box와 매칭되어야 함
        if not is_matched:
            # 얼굴 박스만 있고 person_box가 없는 경우는 무시 (이름 라벨 통합을 위해)
            pass
    
    # IoU 기반 중복 제거 (같은 사람에 대한 중복 박스 제거)
    final_boxes = []
    for box_tuple in all_boxes:
        is_duplicate = False
        for final_box in final_boxes:
            iou = utils.calculate_iou(box_tuple, final_box)
            if iou > 0.7:  # IoU 0.7 이상이면 같은 사람으로 간주 (0.98 -> 0.7로 완화)
                is_duplicate = True
                # 더 큰 박스로 통합
                box_area = (box_tuple[2] - box_tuple[0]) * (box_tuple[3] - box_tuple[1])
                final_area = (final_box[2] - final_box[0]) * (final_box[3] - final_box[1])
                if box_area > final_area:
                    # 더 큰 박스로 교체
                    final_boxes.remove(final_box)
                    final_boxes.append(box_tuple)
                    # 정보도 업데이트
                    box_to_info[box_tuple] = box_to_info.get(final_box, box_to_info[box_tuple])
                    if final_box in box_to_info:
                        del box_to_info[final_box]
                break
        
        if not is_duplicate:
            final_boxes.append(box_tuple)
    
    # 좌표 스무딩: 이전 프레임과 매칭된 박스만 좌표 스무딩 (잔상 방지)
    if cam_id in _last_rendered_frames:
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
                if iou > 0.5:  # IoU 0.5 이상이면 같은 사람
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
            
            # 얼굴 박스 필터링: 박스 크기가 너무 작으면 얼굴 박스로 간주하여 제외
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            frame_area = orig_w * orig_h
            box_ratio = box_area / frame_area if frame_area > 0 else 0
            
            # 박스가 프레임의 2% 미만이면 얼굴 박스로 간주하여 제외 (5% -> 2%로 완화)
            # 단, 위반이 있으면 작은 박스도 표시
            if box_ratio < 0.02 and not is_violation and len(ppe_violations) == 0:
                # 얼굴 박스는 person_box와 통합되어야 하므로 별도로 렌더링하지 않음
                logging.debug(f"[CAM-{cam_id}] 얼굴 박스 제외: 박스={box_tuple}, 비율={box_ratio:.3f}")
                continue
            
            # 모든 사람에게 박스 표시 (위반 여부 관계없이)
            # 마스크 제외한 위반만 체크
            filtered_violations = [v for v in ppe_violations if v != "마스크"]
            
            if "넘어짐" in filtered_violations:
                unified_color = (0, 50, 255)  # 빨간색
                alpha = 0.25
            elif len(filtered_violations) > 0:
                unified_color = (0, 140, 255)  # 주황색
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
    
    return renderer.render_on(processed_frame)

def _submit_models_background_simple(
    frame: np.ndarray,
    resized_frame: np.ndarray,
    cam_id: int,
    timestamp: float,
    safety_system: Any,
    violation_future: Any,
    pose_future: Any,
    fall_future: Optional[Any],
    face_detection_future: Optional[Any],
    violation_kwargs: Dict,
    pose_kwargs: Dict,
    face_model: Optional[Any],
    face_analyzer: Optional[Any],  # 🦬 buffalo_l 추가!
    fast_recognizer: Optional[Any],
    face_database: Optional[Any],
    orig_w: int,
    orig_h: int,
    w_scale: float,
    h_scale: float
):
    """
    프레임 보장 방식: 모든 모델을 백그라운드로 처리하고 결과를 캐시에 저장
    """
    # 공유 변수: Violation과 Pose 결과를 함께 사용하기 위해
    violation_data = {'all_detections': {}, 'ready': False}
    pose_data = {'person_boxes': [], 'ready': False}
    fall_data = {'fall_detections': [], 'ready': False}
    face_data = {'yolo_face_results': None, 'ready': False}
    
    # 동기화를 위한 Lock 생성
    import threading
    data_lock = threading.Lock()
    
    # 결과 콜백: 완료되면 캐시에 저장
    def save_violation_result(future):
        try:
            violation_results = future.result()  # 타임아웃 없이 완료 대기
            # 결과 파싱 (PPE 모델의 Person 박스를 기준으로 통합)
            all_detections = {}
            ppe_person_boxes = []  # PPE 모델에서 감지된 Person 박스 목록
            
            if violation_results and len(violation_results) > 0:
                for det in violation_results[0].boxes:
                    class_id = int(det.cls[0])
                    class_name = safety_system.violation_model.names[class_id]
                    conf = float(det.conf[0])
                    
                    if class_name in config.Thresholds.IGNORED_CLASSES:
                        continue
                    
                    # Person 클래스는 별도로 저장 (기준 박스로 사용)
                    if class_name == 'Person':
                        person_conf_threshold = config.Thresholds.PERSON_CONFIDENCE
                        if conf >= person_conf_threshold:
                            bbox_resized = det.xyxy[0].cpu().numpy()
                            bbox_original = bbox_resized * np.array([w_scale, h_scale, w_scale, h_scale])
                            bbox_clipped = utils.clip_bbox_xyxy(bbox_original, orig_w, orig_h)
                            if bbox_clipped is not None:
                                # 박스 크기 검증 (손/작은 물체 제외)
                                box_w = bbox_clipped[2] - bbox_clipped[0]
                                box_h = bbox_clipped[3] - bbox_clipped[1]
                                box_area = box_w * box_h
                                box_ratio = box_h / box_w if box_w > 0 else 0  # 세로/가로 비율
                                
                                # ⭐ 사람 조건 강화: 최소 크기 + 세로로 긴 박스 (손/부분 제외)
                                # - 최소 너비 80, 높이 120, 면적 15000
                                # - 세로/가로 비율 >= 1.0 (사람은 보통 세로가 더 김)
                                is_valid_person = (
                                    box_w >= 80 and 
                                    box_h >= 120 and 
                                    box_area >= 15000 and
                                    box_ratio >= 1.0  # 세로가 가로보다 긴 박스만
                                )
                                
                                if is_valid_person:
                                    ppe_person_boxes.append({
                                        'bbox': list(bbox_clipped),
                                        'conf': conf,
                                        'class': 'Person'
                                    })
                                else:
                                    logging.debug(f"[CAM-{cam_id}] PPE Person 필터링: w={box_w:.0f}, h={box_h:.0f}, area={box_area:.0f}, ratio={box_ratio:.2f}")
                        continue
                    
                    class_threshold = config.Thresholds.CLASS_CONFIDENCE_THRESHOLDS.get(
                        class_name, config.Thresholds.YOLO_CONFIDENCE
                    )
                    
                    if conf >= class_threshold:
                        bbox_resized = det.xyxy[0].cpu().numpy()
                        bbox_original = bbox_resized * np.array([w_scale, h_scale, w_scale, h_scale])
                        bbox_clipped = utils.clip_bbox_xyxy(bbox_original, orig_w, orig_h)
                        if bbox_clipped is not None:
                            if class_name not in all_detections:
                                all_detections[class_name] = []
                            all_detections[class_name].append({'bbox': list(bbox_clipped), 'conf': conf})
            
            # PPE Person 박스 로깅
            if ppe_person_boxes:
                logging.debug(f"[CAM-{cam_id}] 🔍 PPE Person 감지: {len(ppe_person_boxes)}명")
            
            with data_lock:
                violation_data['all_detections'] = all_detections
                violation_data['ppe_person_boxes'] = ppe_person_boxes  # PPE Person 박스 저장
                violation_data['ready'] = True
                
                # Pose 결과도 준비되었으면 violations 생성
                if pose_data['ready']:
                    _create_violations_from_results(cam_id, timestamp, violation_data, pose_data)
        except Exception as e:
            logging.debug(f"Violation 모델 결과 저장 실패: {e}")
    
    def save_pose_result(future):
        try:
            pose_results = future.result()
            # Pose 결과에서 사람 박스 추출 (confidence 및 크기 검증)
            person_boxes = []
            if pose_results and len(pose_results) > 0 and pose_results[0].boxes is not None:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                confidences = pose_results[0].boxes.conf.cpu().numpy() if pose_results[0].boxes.conf is not None else None
                keypoints = pose_results[0].keypoints if hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None else None
                
                # 디버깅: 전체 감지된 사람 수 로깅
                logging.debug(f"[CAM-{cam_id}] 🔍 Pose 모델 감지: {len(boxes)}명, conf 범위: {confidences.min():.2f}~{confidences.max():.2f}" if confidences is not None and len(confidences) > 0 else f"[CAM-{cam_id}] 🔍 Pose 모델 감지: {len(boxes)}명")
                
                for idx, box in enumerate(boxes):
                    # Confidence 필터링 (더 엄격하게)
                    if confidences is not None and len(confidences) > idx:
                        conf = float(confidences[idx])
                        # Pose confidence를 낮춰서 누운 사람도 감지 (0.15 -> 0.10)
                        min_pose_conf = 0.10
                        if conf < min_pose_conf:
                            logging.debug(f"[CAM-{cam_id}] Pose confidence 낮음: {conf:.3f} < {min_pose_conf:.3f}, 제외")
                            continue
                        
                        # 디버깅: 각 박스의 비율 로깅
                        box_w_debug = (box[2] - box[0]) * w_scale
                        box_h_debug = (box[3] - box[1]) * h_scale
                        ratio_debug = box_w_debug / box_h_debug if box_h_debug > 0 else 0
                        logging.debug(f"[CAM-{cam_id}] 사람 {idx}: conf={conf:.2f}, 박스비율={ratio_debug:.2f}, 크기={box_w_debug:.0f}x{box_h_debug:.0f}")
                    
                    # 박스 크기 검증
                    box_w = (box[2] - box[0]) * w_scale
                    box_h = (box[3] - box[1]) * h_scale
                    box_area = box_w * box_h
                    
                    # 최소 크기 검증
                    if box_w < config.Thresholds.MIN_PERSON_BOX_WIDTH or box_h < config.Thresholds.MIN_PERSON_BOX_HEIGHT:
                        continue
                    
                    # 최대 크기 검증 (너무 큰 박스는 사람이 아닐 가능성 높음)
                    # 프레임의 30% 이상을 차지하면 제외 (50% -> 30%로 강화)
                    max_box_area = orig_w * orig_h * 0.3
                    if box_area > max_box_area:
                        logging.debug(f"[CAM-{cam_id}] Pose 박스 너무 큼: {box_area:.0f} > {max_box_area:.0f}, 제외")
                        continue
                    
                    # 박스 비율 검증 (넘어짐 감지를 위해 가로로 긴 박스도 허용)
                    box_ratio = box_w / box_h if box_h > 0 else 0
                    # 사람 박스: 세로로 긴 경우 0.25~1.0, 가로로 긴 경우(넘어짐) 1.0~3.5
                    # 너무 극단적인 비율만 제외 (0.25 미만 또는 3.5 초과)
                    is_horizontal_pose = box_ratio > 1.2  # 가로로 긴 박스 (넘어짐 가능성)
                    if box_ratio < 0.25 or box_ratio > 3.5:
                        logging.debug(f"[CAM-{cam_id}] Pose 박스 비율 이상: {box_ratio:.2f}, 제외")
                        continue
                    
                    # ⭐ 발/신발 필터링: 가로로 긴 박스(넘어짐)도 최소 높이 130px 이상
                    if is_horizontal_pose and box_h < 130:
                        logging.debug(f"[CAM-{cam_id}] Pose 박스 필터링 (발/신발 의심): 가로로 긴 박스인데 h={box_h:.0f} < 130")
                        continue
                    
                    # 가로로 긴 박스(넘어짐 후보)는 별도 플래그 설정
                    if is_horizontal_pose:
                        logging.info(f"[CAM-{cam_id}] 🔻 넘어짐 후보 감지: 박스 비율={box_ratio:.2f} (가로로 긴 박스)")
                    
                    # 최소 박스 면적 검증 (너무 작은 박스는 노이즈일 가능성)
                    min_box_area = config.Thresholds.MIN_PERSON_BOX_WIDTH * config.Thresholds.MIN_PERSON_BOX_HEIGHT * 1.5
                    if box_area < min_box_area:
                        logging.debug(f"[CAM-{cam_id}] Pose 박스 너무 작음: {box_area:.0f} < {min_box_area:.0f}, 제외")
                        continue
                    
                    # 키포인트 검증 (키포인트가 있으면 최소 개수 확인, 없어도 confidence가 높으면 허용)
                    keypoint_valid = False
                    person_keypoints = None  # 쓰러짐 감지용 키포인트 저장
                    
                    # 디버깅: 키포인트 객체 상태 확인
                    if keypoints is None:
                        logging.debug(f"[CAM-{cam_id}] 🔍 Pose 키포인트 객체 없음 (keypoints is None)")
                    else:
                        logging.debug(f"[CAM-{cam_id}] 🔍 Pose 키포인트 객체 존재: type={type(keypoints)}, "
                                    f"has_xy={hasattr(keypoints, 'xy')}, has_conf={hasattr(keypoints, 'conf')}")
                    
                    if keypoints is not None:
                        try:
                            kpts = keypoints.xy[idx] if hasattr(keypoints, 'xy') and keypoints.xy is not None else None
                            kpts_conf = None
                            if hasattr(keypoints, 'conf') and keypoints.conf is not None:
                                if isinstance(keypoints.conf, (list, tuple)) and len(keypoints.conf) > idx:
                                    kpts_conf = keypoints.conf[idx]
                                elif hasattr(keypoints.conf, '__getitem__'):
                                    try:
                                        kpts_conf = keypoints.conf[idx]
                                    except (IndexError, TypeError):
                                        pass
                            
                            if kpts_conf is not None:
                                # numpy 배열로 변환
                                if not isinstance(kpts_conf, np.ndarray):
                                    try:
                                        kpts_conf = kpts_conf.cpu().numpy() if hasattr(kpts_conf, 'cpu') else np.array(kpts_conf)
                                    except:
                                        kpts_conf = None
                                
                                if kpts_conf is not None and kpts_conf.size > 0:
                                    # 최소 키포인트 개수 확인 (confidence > 0.3인 키포인트, 0.5 -> 0.3으로 완화)
                                    visible_kpts = int(np.sum(kpts_conf > 0.3))
                                    avg_conf = float(np.mean(kpts_conf))
                                    logging.debug(f"[CAM-{cam_id}] 🔍 사람 {idx} 키포인트: visible={visible_kpts}/17, avg_conf={avg_conf:.3f}, min_required={config.Thresholds.MIN_VISIBLE_KEYPOINTS}")
                                    
                                    if visible_kpts >= config.Thresholds.MIN_VISIBLE_KEYPOINTS:
                                        keypoint_valid = True
                                        # 쓰러짐 감지용 키포인트 저장
                                        if kpts is not None:
                                            kpts_np = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else np.array(kpts)
                                            person_keypoints = {
                                                'xy': kpts_np,
                                                'conf': kpts_conf,
                                                'visible_count': visible_kpts
                                            }
                                            logging.debug(f"[CAM-{cam_id}] ✅ 사람 {idx} 키포인트 저장 완료: {visible_kpts}개")
                        except Exception as e:
                            logging.debug(f"[CAM-{cam_id}] 키포인트 검증 실패: {e}")
                    
                    # ⭐ 키포인트가 유효하지 않으면 더 엄격한 기준 적용 (손/부분 감지 방지)
                    if not keypoint_valid:
                        # 키포인트 없으면 무조건 제외! (손이나 부분만 잡히는 것 방지)
                        logging.debug(f"[CAM-{cam_id}] Pose 키포인트 부족 ({config.Thresholds.MIN_VISIBLE_KEYPOINTS}개 미만), 제외")
                        continue
                    
                    scaled_box_np = box * np.array([w_scale, h_scale, w_scale, h_scale])
                    clipped_box = utils.clip_bbox_xyxy(scaled_box_np, orig_w, orig_h)
                    if clipped_box is not None:
                        # 키포인트 정보도 함께 저장 (쓰러짐 감지용)
                        if person_keypoints is not None:
                            person_boxes.append({
                                'box': clipped_box,
                                'keypoints': person_keypoints,
                                'confidence': float(confidences[idx]) if confidences is not None and len(confidences) > idx else 0.0
                            })
                        else:
                            person_boxes.append({
                                'box': clipped_box,
                                'keypoints': None,
                                'confidence': float(confidences[idx]) if confidences is not None and len(confidences) > idx else 0.0
                            })
            
            with data_lock:
                pose_data['person_boxes'] = person_boxes
                pose_data['ready'] = True
                
                # Violation 결과도 준비되었으면 violations 생성
                if violation_data['ready']:
                    _create_violations_from_results(cam_id, timestamp, violation_data, pose_data)
        except Exception as e:
            logging.debug(f"Pose 모델 결과 저장 실패: {e}")
    
    def save_face_result(future):
        try:
            # ⭐ buffalo_l로 Person 크롭 방식 얼굴 인식
            # 전체 프레임 대신 Person 박스로 크롭해서 더 정확하게 인식
            
            # 🔍 디버그: face_analyzer와 face_database 상태 확인
            logging.debug(f"[CAM-{cam_id}] 🔍 save_face_result 진입: face_analyzer={face_analyzer is not None}, face_database={face_database is not None}")
            
            with data_lock:
                face_data['ready'] = True
            
            recognized_faces = []
            
            # 🦬 buffalo_l Person 크롭 방식 (더 정확한 인식!)
            if face_analyzer is None:
                logging.warning(f"[CAM-{cam_id}] ⚠️ face_analyzer가 None입니다!")
            if face_database is None:
                logging.warning(f"[CAM-{cam_id}] ⚠️ face_database가 None입니다!")
            
            if face_analyzer is not None and face_database is not None:
                try:
                    # Pose 데이터가 준비될 때까지 잠시 대기 (최대 100ms)
                    wait_start = time.time()
                    while not pose_data.get('ready', False) and (time.time() - wait_start) < 0.1:
                        time.sleep(0.01)
                    
                    # Person 박스 가져오기
                    person_boxes = pose_data.get('person_boxes', [])
                    
                    if not person_boxes:
                        # Person이 없으면 전체 프레임으로 폴백
                        logging.debug(f"[CAM-{cam_id}] 🔍 frame 상태: shape={frame.shape if frame is not None else 'None'}, dtype={frame.dtype if frame is not None else 'None'}")
                        faces = face_analyzer.get(frame)
                        logging.debug(f"[CAM-{cam_id}] 🦬 buffalo_l (전체 프레임): {len(faces)}개 얼굴")
                        
                        for face in faces:
                            try:
                                # ⭐ det_score 최소 임계값 체크 (오탐지 방지: 손바닥 등)
                                # 0.5 → 0.4로 낮춤 (누운 상태 얼굴 인식률 향상)
                                MIN_FACE_DET_SCORE = 0.4
                                if face.det_score < MIN_FACE_DET_SCORE:
                                    logging.debug(f"[CAM-{cam_id}] 얼굴 신뢰도 부족: {face.det_score:.3f} < {MIN_FACE_DET_SCORE} - 스킵")
                                    continue
                                
                                bbox = face.bbox.astype(int)
                                fx1, fy1, fx2, fy2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                fx1 = max(0, min(fx1, orig_w))
                                fy1 = max(0, min(fy1, orig_h))
                                fx2 = max(0, min(fx2, orig_w))
                                fy2 = max(0, min(fy2, orig_h))
                                
                                # 최소 크기 체크 (40x40 이상)
                                if (fx2 - fx1) < 40 or (fy2 - fy1) < 40:
                                    logging.debug(f"[CAM-{cam_id}] 얼굴 크기 부족: {fx2-fx1}x{fy2-fy1} < 40x40 - 스킵")
                                    continue
                                
                                embedding = face.embedding
                                if embedding is None:
                                    continue
                                embedding = embedding / np.linalg.norm(embedding)
                                
                                person_name, similarity_score = utils.find_best_match_faiss(
                                    embedding, face_database, config.Thresholds.SIMILARITY
                                )
                                
                                if person_name != "Unknown":
                                    logging.info(f"[CAM-{cam_id}] ✅ 🦬 buffalo_l 인식: {person_name}, 유사도={similarity_score:.3f}")
                                
                                recognized_faces.append({
                                    "box": [fx1, fy1, fx2, fy2],
                                    "bbox": [fx1, fy1, fx2, fy2],
                                    "name": person_name,
                                    "similarity": similarity_score,
                                    "isViolation": False,
                                    "ppe_violations": []
                                })
                            except Exception as e:
                                continue
                    else:
                        # ⭐ Person 크롭 방식 (더 정확!)
                        logging.debug(f"[CAM-{cam_id}] 🦬 buffalo_l Person 크롭: {len(person_boxes)}명 처리")
                        
                        # 🔒 이미 사용된 얼굴 좌표 추적 (중복 할당 방지)
                        used_face_centers = set()
                        
                        for person_info in person_boxes:
                            try:
                                # Person 박스 추출
                                if isinstance(person_info, dict):
                                    person_box = person_info.get('box', person_info.get('bbox', []))
                                else:
                                    person_box = list(person_info)
                                
                                if len(person_box) < 4:
                                    continue
                                
                                px1, py1, px2, py2 = int(person_box[0]), int(person_box[1]), int(person_box[2]), int(person_box[3])
                                
                                # 경계 체크
                                px1 = max(0, min(px1, orig_w))
                                py1 = max(0, min(py1, orig_h))
                                px2 = max(0, min(px2, orig_w))
                                py2 = max(0, min(py2, orig_h))
                                
                                # Person 크기 체크 (최소 80x120)
                                if (px2 - px1) < 80 or (py2 - py1) < 120:
                                    continue
                                
                                # Person 크롭
                                person_crop = frame[py1:py2, px1:px2]
                                if person_crop.size == 0:
                                    continue
                                
                                # 크롭된 이미지에서 얼굴 감지
                                faces = face_analyzer.get(person_crop)
                                
                                if len(faces) == 0:
                                    continue
                                
                                # ⭐ det_score 최소 임계값으로 필터링 (오탐지 방지)
                                # 0.5 → 0.4로 낮춤 (누운 상태 얼굴 인식률 향상)
                                MIN_FACE_DET_SCORE = 0.4
                                valid_faces = [f for f in faces if f.det_score >= MIN_FACE_DET_SCORE]
                                
                                if len(valid_faces) == 0:
                                    logging.debug(f"[CAM-{cam_id}] Person 크롭 내 유효 얼굴 없음 (det_score < {MIN_FACE_DET_SCORE})")
                                    continue
                                
                                # 가장 큰 얼굴 선택 (det_score 기준)
                                best_face = max(valid_faces, key=lambda f: f.det_score)
                                
                                # 최소 얼굴 크기 체크 (크롭 내에서 20x20 이상, 완화됨)
                                face_bbox = best_face.bbox.astype(int)
                                face_w = face_bbox[2] - face_bbox[0]
                                face_h = face_bbox[3] - face_bbox[1]
                                if face_w < 20 or face_h < 20:
                                    logging.debug(f"[CAM-{cam_id}] Person 크롭 내 얼굴 크기 부족: {face_w}x{face_h} < 20x20")
                                    continue
                                
                                # 크롭 내 좌표를 원본 프레임 좌표로 변환
                                bbox = best_face.bbox.astype(int)
                                fx1 = px1 + bbox[0]
                                fy1 = py1 + bbox[1]
                                fx2 = px1 + bbox[2]
                                fy2 = py1 + bbox[3]
                                
                                # 🔒 얼굴 중심이 Person 박스 중앙 영역 안에 있는지 확인 (이름 뺏어감 방지)
                                face_center_x = (fx1 + fx2) // 2
                                face_center_y = (fy1 + fy2) // 2
                                
                                # Person 박스의 중앙 80% 영역 계산
                                person_w = px2 - px1
                                person_h = py2 - py1
                                margin_x = int(person_w * 0.1)  # 좌우 10% 마진
                                margin_y = int(person_h * 0.1)  # 상하 10% 마진
                                
                                inner_px1 = px1 + margin_x
                                inner_py1 = py1 + margin_y
                                inner_px2 = px2 - margin_x
                                inner_py2 = py2 - margin_y
                                
                                # 얼굴 중심이 중앙 영역 밖이면 건너뛰기
                                if not (inner_px1 <= face_center_x <= inner_px2 and inner_py1 <= face_center_y <= inner_py2):
                                    logging.debug(f"[CAM-{cam_id}] 얼굴 중심이 Person 박스 가장자리에 있음 - 스킵 (이름 뺏어감 방지)")
                                    continue
                                
                                # 🔒 이미 사용된 얼굴인지 체크 (중복 할당 방지)
                                # 얼굴 중심을 50픽셀 단위로 그룹화 (같은 얼굴 판정)
                                face_center_key = (face_center_x // 50, face_center_y // 50)
                                if face_center_key in used_face_centers:
                                    logging.debug(f"[CAM-{cam_id}] 이미 사용된 얼굴 - 스킵 (중복 할당 방지): center={face_center_key}")
                                    continue
                                used_face_centers.add(face_center_key)
                                
                                # 경계 체크
                                fx1 = max(0, min(fx1, orig_w))
                                fy1 = max(0, min(fy1, orig_h))
                                fx2 = max(0, min(fx2, orig_w))
                                fy2 = max(0, min(fy2, orig_h))
                                
                                # 임베딩
                                embedding = best_face.embedding
                                if embedding is None:
                                    continue
                                embedding = embedding / np.linalg.norm(embedding)
                                
                                # FAISS 검색
                                person_name, similarity_score = utils.find_best_match_faiss(
                                    embedding, face_database, config.Thresholds.SIMILARITY
                                )
                                
                                if person_name != "Unknown":
                                    logging.info(f"[CAM-{cam_id}] ✅ 🦬 buffalo_l 크롭 인식: {person_name}, 유사도={similarity_score:.3f}")
                                
                                # Person 박스도 함께 저장 (매칭용)
                                recognized_faces.append({
                                    "box": [fx1, fy1, fx2, fy2],
                                    "bbox": [fx1, fy1, fx2, fy2],
                                    "person_box": [px1, py1, px2, py2],
                                    "name": person_name,
                                    "similarity": similarity_score,
                                    "isViolation": False,
                                    "ppe_violations": []
                                })
                            except Exception as e:
                                logging.debug(f"[CAM-{cam_id}] Person 크롭 처리 오류: {e}")
                                continue
                                
                except Exception as e:
                    logging.debug(f"[CAM-{cam_id}] buffalo_l 처리 오류: {e}")
            
            with results_cache_lock:
                merged = False
                for ts, rd in model_results_cache[cam_id]:
                    # 타임스탬프 매칭 정밀도: 0.1초 (지연 허용 범위 확대)
                    if abs(ts - timestamp) < 0.1:
                        # 기존 recognized_faces와 병합
                        if 'recognized_faces' not in rd:
                            rd['recognized_faces'] = []
                        existing_faces = rd.get('recognized_faces', [])
                        
                        # 중복 제거를 위해 IoU 기반으로 확인 (박스 좌표 정확 일치가 아닌 IoU 사용)
                        new_faces = []
                        for new_face in recognized_faces:
                            new_box = new_face.get('box', [])
                            if len(new_box) != 4:
                                continue
                            new_box_tuple = tuple(map(int, new_box))
                            is_duplicate = False
                            for existing_face in existing_faces:
                                existing_box = existing_face.get('box', [])
                                if len(existing_box) != 4:
                                    continue
                                existing_box_tuple = tuple(map(int, existing_box))
                                # IoU 기반 중복 확인 (0.5 이상이면 같은 사람)
                                iou = utils.calculate_iou(new_box_tuple, existing_box_tuple)
                                if iou > 0.5:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                new_faces.append(new_face)
                        
                        rd['recognized_faces'].extend(new_faces)
                        merged = True
                        break
                if not merged:
                    model_results_cache[cam_id].append((timestamp, {'recognized_faces': recognized_faces}))
            logging.debug(f"[CAM-{cam_id}] 백그라운드 Face 결과 저장 완료: {len(recognized_faces)}개")
        except Exception as e:
            logging.debug(f"Face 모델 결과 저장 실패: {e}")
    
    def _create_violations_from_results(cam_id, timestamp, violation_data, pose_data):
        """Violation과 Pose 결과를 결합하여 violations 생성 (Pose 박스 기본 사용)"""
        try:
            all_detections = violation_data['all_detections']
            
            # Pose + PPE Person 박스 통합 (넘어진 사람 감지 강화)
            pose_person_boxes = pose_data['person_boxes']
            ppe_person_boxes = violation_data.get('ppe_person_boxes', [])
            
            # 1. Pose 박스를 기본으로 사용
            person_boxes = list(pose_person_boxes) if pose_person_boxes else []
            
            # 2. PPE Person 박스 중 Pose에 없는 것 추가 (넘어진 사람 감지)
            for ppe_person in ppe_person_boxes:
                ppe_bbox = ppe_person['bbox']
                is_new = True
                
                # Pose 박스와 IoU 비교
                for pose_box in pose_person_boxes:
                    if isinstance(pose_box, dict):
                        pose_bbox = pose_box.get('box', pose_box)
                    else:
                        pose_bbox = pose_box
                    
                    iou = utils.calculate_iou(tuple(ppe_bbox), tuple(pose_bbox))
                    if iou > 0.3:  # 30% 이상 겹치면 같은 사람
                        is_new = False
                        break
                
                if is_new:
                    # 넘어진 사람 후보 (Pose가 못 잡은 사람)
                    box_w = ppe_bbox[2] - ppe_bbox[0]
                    box_h = ppe_bbox[3] - ppe_bbox[1]
                    box_area = box_w * box_h
                    box_ratio = box_w / box_h if box_h > 0 else 0
                    
                    # ===== 손바닥/부분 감지 필터링 강화 =====
                    # 1. 최소 면적 조건: 20000 픽셀 이상 (15000 -> 20000, 발/신발 제외 강화)
                    if box_area < 20000:
                        logging.debug(f"[CAM-{cam_id}] PPE Person 필터링 (면적 부족): area={box_area:.0f} < 20000")
                        continue
                    
                    # 2. 최소 세로 크기: 사람은 세로가 길어야 함 (손바닥 제외)
                    # 넘어진 사람(가로로 긴 경우)은 예외 처리
                    MIN_PPE_PERSON_HEIGHT = 120  # 최소 세로 120px
                    MIN_PPE_PERSON_WIDTH = 50    # 최소 가로 50px
                    
                    if box_ratio < 1.2:  # 서있는 사람 (세로가 더 긴 경우)
                        if box_h < MIN_PPE_PERSON_HEIGHT:
                            logging.debug(f"[CAM-{cam_id}] PPE Person 필터링 (세로 부족 - 손바닥 의심): h={box_h:.0f} < {MIN_PPE_PERSON_HEIGHT}")
                            continue
                        if box_w < MIN_PPE_PERSON_WIDTH:
                            logging.debug(f"[CAM-{cam_id}] PPE Person 필터링 (가로 부족): w={box_w:.0f} < {MIN_PPE_PERSON_WIDTH}")
                            continue
                    else:  # 넘어진 사람 (가로가 더 긴 경우)
                        # 넘어짐은 가로가 길어야 함
                        if box_w < MIN_PPE_PERSON_HEIGHT:  # 가로가 최소 120px
                            logging.debug(f"[CAM-{cam_id}] PPE Person 필터링 (넘어짐 후보 가로 부족): w={box_w:.0f} < {MIN_PPE_PERSON_HEIGHT}")
                            continue
                        # ⭐ 발/신발 필터링: 넘어진 사람도 최소 높이 130px 이상 (발/신발 제외 강화)
                        MIN_FALL_HEIGHT = 130
                        if box_h < MIN_FALL_HEIGHT:
                            logging.debug(f"[CAM-{cam_id}] PPE Person 필터링 (넘어짐 높이 부족 - 발/신발 의심): h={box_h:.0f} < {MIN_FALL_HEIGHT}")
                            continue
                    
                    # 3. 비율 검증: 너무 정사각형에 가까우면 손바닥 의심 (0.7~1.3 범위)
                    if 0.7 <= box_ratio <= 1.3 and box_area < 30000:
                        logging.debug(f"[CAM-{cam_id}] PPE Person 필터링 (정사각형 - 손바닥 의심): ratio={box_ratio:.2f}, area={box_area:.0f}")
                        continue
                    
                    person_boxes.append({
                        'box': ppe_bbox, 
                        'source': 'ppe',
                        'is_fall_candidate': box_ratio >= 1.2  # 가로로 긴 박스는 넘어짐 후보
                    })
                    
                    if box_ratio >= 1.2:
                        logging.warning(f"[CAM-{cam_id}] 🔻 넘어짐 후보 (PPE Person): 비율={box_ratio:.2f}, 박스={ppe_bbox}, 면적={box_area:.0f}")
                    else:
                        logging.debug(f"[CAM-{cam_id}] PPE Person 추가: 비율={box_ratio:.2f}, 면적={box_area:.0f}, w={box_w:.0f}, h={box_h:.0f}")
            
            logging.debug(f"[CAM-{cam_id}] 통합 Person 박스: Pose={len(pose_person_boxes)}개 + PPE={len(ppe_person_boxes)}개 → 총 {len(person_boxes)}개")
            
            violations = []
            # recognized_faces는 기존 결과를 유지하기 위해 빈 리스트로 초기화하지 않음
            # 대신 위반 정보만 추가할 새로운 리스트 생성
            new_recognized_faces = []  # 위반 정보만 추가할 리스트
            used_ppe_boxes = set()
            used_face_indices = set()  # ⭐ 이미 매칭된 얼굴 인덱스 추적 (중복 매칭 방지)
            
            # 기존 얼굴 인식 결과 가져오기 (최근 0.5초 이내 + 영역 포함 기반 매칭)
            # ⭐ 얼굴 박스 좌표도 함께 저장하여 Person 박스 내부 포함 여부 확인
            existing_faces_list = []  # [(center_x, center_y, name, index, face_box), ...]
            current_time = time.time()
            face_index = 0  # 얼굴 인덱스
            with results_cache_lock:
                if cam_id in model_results_cache:
                    for ts, rd in reversed(model_results_cache[cam_id]):
                        if current_time - ts <= 1.5:  # 1.5초 이내 결과 사용 (얼굴 인식 타이밍 개선: 0.5 -> 1.5초)
                            for face in rd.get('recognized_faces', []):
                                box = face.get('box', [])
                                name = face.get('name', 'Unknown')
                                if len(box) == 4 and name != 'Unknown':
                                    cx = (box[0] + box[2]) / 2
                                    cy = (box[1] + box[3]) / 2
                                    # ⭐ 얼굴 박스 좌표도 함께 저장
                                    existing_faces_list.append((cx, cy, name, face_index, box))
                                    face_index += 1
                        else:
                            break
            
            if existing_faces_list:
                logging.debug(f"[CAM-{cam_id}] 기존 얼굴 인식 결과: {len(existing_faces_list)}명")
            
            # ⭐⭐ 현재 프레임에서 buffalo_l이 감지한 얼굴 박스 목록 (뒷모습 이름 할당 방지)
            # save_face_result에서 저장한 최신 얼굴 결과만 사용
            current_frame_faces = []  # [(face_box, name), ...]
            with results_cache_lock:
                if cam_id in model_results_cache and model_results_cache[cam_id]:
                    latest_ts, latest_rd = model_results_cache[cam_id][-1]
                    # 1.5초 이내의 최신 결과 사용 (얼굴 인식 타이밍 개선: 0.3 -> 1.5초)
                    if current_time - latest_ts <= 1.5:
                        for face in latest_rd.get('recognized_faces', []):
                            box = face.get('box', [])
                            name = face.get('name', 'Unknown')
                            if len(box) == 4 and name != 'Unknown':
                                current_frame_faces.append((box, name))
            
            logging.debug(f"[CAM-{cam_id}] 현재 프레임 얼굴: {len(current_frame_faces)}명")
            
            # 각 사람 박스에 대해 PPE 위반 확인
            for person_box in person_boxes:
                # person_box가 dict인 경우 (키포인트 정보 포함) 처리
                is_fall_candidate = False
                if isinstance(person_box, dict):
                    box = person_box.get('box', person_box)
                    x1, y1, x2, y2 = map(int, box)
                    is_fall_candidate = person_box.get('is_fall_candidate', False)
                else:
                    x1, y1, x2, y2 = map(int, person_box)
                
                # ⭐ 박스 비율로 넘어짐 직접 감지 (가로 > 세로 = 쓰러짐)
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                box_ratio = box_width / box_height if box_height > 0 else 0
                # 넘어짐 조건: 비율 >= 1.8 AND 면적 >= 15000 (손/작은 물체 제외)
                if box_ratio >= 1.8 and box_area >= 15000:
                    is_fall_candidate = True
                    logging.warning(f"[CAM-{cam_id}] 🔻 박스 비율 기반 넘어짐 감지 (보조): ratio={box_ratio:.2f}, area={box_area}")
                
                ppe_violations, ppe_boxes = _process_ppe_detection(
                    (x1, y1, x2, y2), 
                    all_detections, 
                    used_ppe_boxes
                )
                
                # 넘어짐 후보면 위반에 추가
                if is_fall_candidate:
                    if '넘어짐' not in ppe_violations:
                        ppe_violations.append('넘어짐')
                        logging.warning(f"[CAM-{cam_id}] ⚠️ 넘어짐 위반 추가 (박스 비율={box_ratio:.2f})")
                
                # 마스크 제외한 위반만 체크
                filtered_violations = [v for v in ppe_violations if v != "마스크"]
                
                # cam_id를 area로 매핑
                area_map = {0: "A-1", 1: "A-2", 2: "B-1", 3: "B-2"}
                area = area_map.get(cam_id, f"A-{cam_id+1}")
                
                # 기존 얼굴 인식 결과에서 이름 찾기 (영역 포함 + 거리 기반 매칭)
                person_box_tuple = tuple(map(int, [x1, y1, x2, y2]))
                recognized_name = "Unknown"
                worker = "알 수 없음"
                
                # ⭐⭐⭐ 1단계: 현재 프레임에서 감지된 얼굴만 매칭 (뒷모습 이름 할당 완전 방지)
                # 현재 프레임에서 buffalo_l이 감지하지 못한 사람에게는 이름 할당 안 함
                person_cx = (x1 + x2) / 2
                person_cy = (y1 + y2) / 2
                person_w = x2 - x1
                person_h = y2 - y1
                
                # 현재 프레임 얼굴에서 먼저 매칭 시도
                for face_box_current, face_name_current in current_frame_faces:
                    fx1, fy1, fx2, fy2 = face_box_current
                    face_cx = (fx1 + fx2) / 2
                    face_cy = (fy1 + fy2) / 2
                    
                    # 얼굴 중심이 Person 박스 안에 있는지 확인
                    if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:
                        # 얼굴이 Person 상단 60%에 있는지 확인
                        person_top_60 = y1 + person_h * 0.6
                        if face_cy <= person_top_60:
                            recognized_name = face_name_current
                            worker = face_name_current
                            logging.debug(f"[CAM-{cam_id}] 현재 프레임 얼굴 매칭: {face_name_current}")
                            break
                
                # 현재 프레임에서 매칭 실패 시, 기존 캐시에서 시도 (단, 더 엄격한 조건)
                best_match_score = -1  # 매칭 점수 (높을수록 좋음)
                best_face_index = None
                
                # 현재 프레임 매칭 성공 시 캐시 매칭 스킵
                if recognized_name != "Unknown":
                    best_face_index = -1  # 캐시 매칭 불필요 플래그
                
                for face_cx, face_cy, face_name, face_idx, face_box in existing_faces_list:
                    # ⭐ 현재 프레임에서 이미 매칭됨 → 캐시 매칭 스킵
                    if best_face_index == -1:
                        break
                    
                    # ⭐ 이미 다른 Person에 매칭된 얼굴은 건너뛰기
                    if face_idx in used_face_indices:
                        continue
                    
                    fx1, fy1, fx2, fy2 = face_box
                    face_w = fx2 - fx1
                    face_h = fy2 - fy1
                    
                    # ===== 1단계: 얼굴 박스가 Person 박스 안에 있는지 확인 =====
                    # 얼굴 중심이 Person 박스 내부에 있어야 함
                    if not (x1 <= face_cx <= x2 and y1 <= face_cy <= y2):
                        logging.debug(f"[CAM-{cam_id}] 얼굴 중심이 Person 박스 밖: face=({face_cx:.0f},{face_cy:.0f}), person=({x1},{y1},{x2},{y2}) - 스킵")
                        continue
                    
                    # ===== 2단계: 얼굴이 Person 박스 상단 60%에 있어야 함 (머리 위치) =====
                    # 뒷모습 Person의 경우 앞사람의 얼굴이 하단에 위치할 수 있음
                    person_top_60_percent = y1 + person_h * 0.6
                    if face_cy > person_top_60_percent:
                        logging.debug(f"[CAM-{cam_id}] 얼굴이 Person 박스 하단에 있음 (뒷모습 의심): face_cy={face_cy:.0f}, top60%={person_top_60_percent:.0f} - 스킵")
                        continue
                    
                    # ===== 3단계: 얼굴 박스의 대부분이 Person 박스 안에 포함 =====
                    # 얼굴 박스와 Person 박스의 교집합 계산
                    inter_x1 = max(x1, fx1)
                    inter_y1 = max(y1, fy1)
                    inter_x2 = min(x2, fx2)
                    inter_y2 = min(y2, fy2)
                    
                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        face_area = max(face_w * face_h, 1)
                        containment_ratio = inter_area / face_area  # 얼굴이 Person에 포함된 비율
                        
                        # 얼굴의 70% 이상이 Person 박스 안에 있어야 함
                        if containment_ratio < 0.7:
                            logging.debug(f"[CAM-{cam_id}] 얼굴 포함 비율 부족: {containment_ratio:.2f} < 0.7 - 스킵")
                            continue
                    else:
                        # 교집합 없음
                        continue
                    
                    # ===== 4단계: 거리 기반 점수 계산 =====
                    distance = ((person_cx - face_cx) ** 2 + (person_cy - face_cy) ** 2) ** 0.5
                    if distance > 200:  # 200픽셀 초과면 제외
                        continue
                    
                    # 매칭 점수: 거리가 가까울수록, 포함 비율이 높을수록 좋음
                    match_score = containment_ratio * (1.0 - distance / 200.0)
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        recognized_name = face_name
                        worker = face_name
                        best_face_index = face_idx
                        logging.debug(f"[CAM-{cam_id}] 얼굴 매칭 후보: {face_name}, 점수={match_score:.3f}, 거리={distance:.0f}, 포함비율={containment_ratio:.2f}")
                
                # ⭐ 매칭된 얼굴 인덱스 기록 (중복 매칭 방지)
                if best_face_index is not None:
                    used_face_indices.add(best_face_index)
                    logging.debug(f"[CAM-{cam_id}] 최종 얼굴 매칭: {recognized_name}, 점수={best_match_score:.3f}")
                
                # ⭐ 모든 사람 추가 (위반 여부 관계없이) - 초록 박스 표시를 위해!
                if filtered_violations:
                    # 위반이 있는 경우
                    unique_violations = list(set(filtered_violations))
                    ppe_violations_display = []
                    for v in unique_violations:
                        if v == "안전모":
                            ppe_violations_display.append("안전모")
                        elif v == "안전조끼":
                            ppe_violations_display.append("안전조끼")
                        elif v == "넘어짐":
                            ppe_violations_display.append("넘어짐")
                    
                    # 넘어짐이 있으면 hazard를 넘어짐으로 설정
                    if "넘어짐" in ppe_violations_display:
                        hazard = "⚠️ 넘어짐 감지"
                    else:
                        hazard = f"PPE 위반내역: {', '.join(ppe_violations_display)}" if ppe_violations_display else "위반 감지"
                    
                    violations.append({
                        "person_box": [x1, y1, x2, y2],
                        "violations": unique_violations,
                        "recognized_name": recognized_name,
                        "worker": worker,
                        "area": area,
                        "level": "WARNING",
                        "hazard": hazard
                    })
                    
                    # new_recognized_faces에 위반 정보 추가
                    new_recognized_faces.append({
                        "box": [x1, y1, x2, y2],
                        "bbox": [x1, y1, x2, y2],
                        "name": recognized_name,
                        "type": "Violation",
                        "isViolation": True,
                        "ppe_violations": unique_violations,
                    })
                else:
                    # ⭐ 위반이 없는 경우 - 안전! (초록 박스)
                    violations.append({
                        "person_box": [x1, y1, x2, y2],
                        "violations": [],  # 빈 리스트 = 안전
                        "recognized_name": recognized_name,
                        "worker": worker,
                        "area": area,
                        "level": "SAFE",
                        "hazard": "안전"
                    })
                    
                    new_recognized_faces.append({
                        "box": [x1, y1, x2, y2],
                        "bbox": [x1, y1, x2, y2],
                        "name": recognized_name,
                        "type": "Safe",
                        "isViolation": False,
                        "ppe_violations": [],  # 빈 리스트 = 안전
                    })
            
            # 결과 저장 (기존 결과에 병합)
            result_dict = {
                'violations': violations, 
                'violation_count': len(violations), 
                'all_detections': all_detections,
                'recognized_faces': new_recognized_faces  # 위반 정보만 포함
            }
            with results_cache_lock:
                # 기존 결과 찾기 및 병합
                merged = False
                for idx, (ts, rd) in enumerate(model_results_cache[cam_id]):
                    # 타임스탬프 매칭 정밀도: 0.1초 (지연 허용 범위 확대)
                    if abs(ts - timestamp) < 0.1:
                        # 기존 결과에 병합 (recognized_faces와 violations는 리스트로 합치기)
                        # 키가 없으면 초기화
                        if 'recognized_faces' not in rd:
                            rd['recognized_faces'] = []
                        if 'violations' not in rd:
                            rd['violations'] = []
                        if 'all_detections' not in rd:
                            rd['all_detections'] = {}
                        
                        existing_faces = rd.get('recognized_faces', [])
                        existing_violations = rd.get('violations', [])
                        
                        # 중복 제거를 위해 IoU 기반으로 확인 (박스 좌표 정확 일치가 아닌 IoU 사용)
                        new_faces = []
                        for new_face in new_recognized_faces:
                            new_box = new_face.get('box', [])
                            if len(new_box) != 4:
                                continue
                            new_box_tuple = tuple(map(int, new_box))
                            is_duplicate = False
                            for existing_face in existing_faces:
                                existing_box = existing_face.get('box', [])
                                if len(existing_box) != 4:
                                    continue
                                existing_box_tuple = tuple(map(int, existing_box))
                                # IoU 기반 중복 확인 (0.5 이상이면 같은 사람)
                                iou = utils.calculate_iou(new_box_tuple, existing_box_tuple)
                                if iou > 0.5:
                                    # 같은 사람이면 기존 얼굴 인식 정보는 유지하고, 위반 정보만 업데이트
                                    if 'name' in existing_face and existing_face['name'] != 'Unknown':
                                        # 기존 얼굴 인식 정보가 있으면 위반 정보만 업데이트
                                        existing_face['isViolation'] = True
                                        existing_face['ppe_violations'] = new_face.get('ppe_violations', [])
                                        existing_face['type'] = new_face.get('type', '')
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                new_faces.append(new_face)
                        
                        new_violations = []
                        for new_viol in violations:
                            new_box = new_viol.get('person_box', [])
                            if len(new_box) != 4:
                                continue
                            new_box_tuple = tuple(map(int, new_box))
                            is_duplicate = False
                            for existing_viol in existing_violations:
                                existing_box = existing_viol.get('person_box', [])
                                if len(existing_box) != 4:
                                    continue
                                existing_box_tuple = tuple(map(int, existing_box))
                                # IoU 기반 중복 확인 (0.5 이상이면 같은 사람)
                                iou = utils.calculate_iou(new_box_tuple, existing_box_tuple)
                                if iou > 0.5:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                new_violations.append(new_viol)
                        
                        # 병합
                        rd['recognized_faces'].extend(new_faces)
                        rd['violations'].extend(new_violations)
                        rd['violation_count'] = len(rd['violations'])
                        rd['all_detections'].update(all_detections)
                        merged = True
                        logging.info(f"[CAM-{cam_id}] 백그라운드 결과 병합: 기존 위반={len(existing_violations)}개, 새 위반={len(new_violations)}개, 최종={len(rd['violations'])}개")
                        break
                
                if not merged:
                    # 저장할 때 현재 시간 사용 (정리 로직에서 즉시 삭제 방지)
                    save_time = time.time()
                    model_results_cache[cam_id].append((save_time, result_dict))
                    logging.info(f"[CAM-{cam_id}] 백그라운드 결과 신규 저장: {len(violations)}개 위반, {len(new_recognized_faces)}개 얼굴")
                
                # 오래된 결과 제거 (CACHE_TTL 사용 - 3.0초)
                current_time = time.time()
                model_results_cache[cam_id] = [
                    (ts, rd) for ts, rd in model_results_cache[cam_id]
                    if current_time - ts <= CACHE_TTL
                ]
            
            logging.info(f"[CAM-{cam_id}] 백그라운드 Violation+Pose 결과 저장 완료: {len(violations)}개 위반, {len(new_recognized_faces)}개 얼굴")
        except Exception as e:
            logging.error(f"Violations 생성 실패: {e}", exc_info=True)
    
    def save_fall_result(future):
        """Fall Detection 모델 결과 처리 (누운 사람 직접 감지)"""
        try:
            fall_results = future.result()
            if fall_results is None:
                with data_lock:
                    fall_data['ready'] = True
                return
            
            fall_detections = []
            if fall_results and len(fall_results) > 0 and fall_results[0].boxes is not None:
                boxes = fall_results[0].boxes.xyxy.cpu().numpy()
                confidences = fall_results[0].boxes.conf.cpu().numpy() if fall_results[0].boxes.conf is not None else None
                classes = fall_results[0].boxes.cls.cpu().numpy() if fall_results[0].boxes.cls is not None else None
                
                for idx, box in enumerate(boxes):
                    conf = float(confidences[idx]) if confidences is not None and len(confidences) > idx else 0.5
                    cls = int(classes[idx]) if classes is not None and len(classes) > idx else 0
                    
                    # Fall 클래스만 감지 (클래스 0=Person, 1=Fall)
                    # ⭐ 클래스 1("Fall")이고 confidence가 0.75 이상인 경우 처리 (오탐지 방지)
                    if cls == 1 and conf >= 0.75:
                        # ⭐ resized_frame 사용하므로 다른 모델들과 동일하게 스케일링
                        bbox_resized = box
                        bbox_original = bbox_resized * np.array([w_scale, h_scale, w_scale, h_scale])
                        bbox_clipped = utils.clip_bbox_xyxy(bbox_original, orig_w, orig_h)
                        
                        if bbox_clipped is not None:
                            # 박스 크기 필터링 (너무 작은 박스 제외 - 최소 60x60 픽셀)
                            box_width = bbox_clipped[2] - bbox_clipped[0]
                            box_height = bbox_clipped[3] - bbox_clipped[1]
                            box_area = box_width * box_height
                            frame_area = orig_w * orig_h
                            
                            if box_width < 60 or box_height < 60:
                                logging.debug(f"[CAM-{cam_id}] Fall 박스 필터링: 크기 너무 작음 ({box_width:.0f}x{box_height:.0f})")
                                continue
                            
                            # ⭐ 너무 큰 박스 제외 (화면의 40% 이상 차지하면 오탐지)
                            if box_area > frame_area * 0.4:
                                logging.debug(f"[CAM-{cam_id}] Fall 박스 필터링: 크기 너무 큼 (화면의 {box_area/frame_area*100:.1f}%)")
                                continue
                            
                            # ⭐ 박스 비율 확인 (넘어진 자세 = 가로 > 세로)
                            box_ratio = box_width / box_height if box_height > 0 else 0
                            if box_ratio < 1.2:
                                logging.debug(f"[CAM-{cam_id}] Fall 박스 필터링: 비율 부적합 (ratio={box_ratio:.2f} < 1.2)")
                                continue
                            
                            # ⭐ 발/신발 필터링: 넘어진 사람도 최소 높이 130px 이상
                            if box_height < 130:
                                logging.debug(f"[CAM-{cam_id}] Fall 박스 필터링: 높이 부족 (발/신발 의심): h={box_height:.0f} < 130")
                                continue
                            
                            fall_detections.append({
                                'bbox': list(bbox_clipped),
                                'conf': conf,
                                'class': cls
                            })
                            logging.warning(f"[CAM-{cam_id}] 🔻 Fall 모델 감지: 클래스=Fall, conf={conf:.2f}, 박스={bbox_clipped}")
            
            with data_lock:
                fall_data['fall_detections'] = fall_detections
                fall_data['ready'] = True
                
                # Fall Detection 결과를 기존 Pose 결과와 병합
                if fall_detections:
                    current_time = time.time()
                    with results_cache_lock:
                        if cam_id not in model_results_cache:
                            model_results_cache[cam_id] = []
                        
                        for fall_det in fall_detections:
                            x1, y1, x2, y2 = [int(c) for c in fall_det['bbox']]
                            fall_center_x = (x1 + x2) / 2
                            fall_center_y = (y1 + y2) / 2
                            
                            # 중복 체크 (같은 위치에서 0.5초 내 중복 방지 - 빠른 업데이트 허용)
                            is_duplicate = False
                            for ts, rd in model_results_cache[cam_id]:
                                if current_time - ts < 0.5:
                                    for v in rd.get('violations', []):
                                        if v.get('violation_type') == '넘어짐':
                                            v_center_x = (v.get('x1', 0) + v.get('x2', 0)) / 2
                                            v_center_y = (v.get('y1', 0) + v.get('y2', 0)) / 2
                                            if abs(v_center_x - fall_center_x) < 100 and abs(v_center_y - fall_center_y) < 100:
                                                is_duplicate = True
                                                break
                            
                            if is_duplicate:
                                continue
                            
                            # 가장 가까운 기존 결과 찾기 (이름 정보 가져오기)
                            best_match_name = "Unknown"
                            best_match_distance = float('inf')
                            
                            for ts, rd in model_results_cache[cam_id]:
                                if current_time - ts < 2.0:  # 2초 이내 결과
                                    for face in rd.get('recognized_faces', []):
                                        face_box = face.get('box', [0, 0, 0, 0])
                                        if len(face_box) >= 4:
                                            face_center_x = (face_box[0] + face_box[2]) / 2
                                            face_center_y = (face_box[1] + face_box[3]) / 2
                                            distance = ((fall_center_x - face_center_x) ** 2 + (fall_center_y - face_center_y) ** 2) ** 0.5
                                            if distance < best_match_distance and distance < 200:  # 200픽셀 이내
                                                best_match_distance = distance
                                                best_match_name = face.get('name', 'Unknown')
                            
                            # 넘어짐 위반 + 얼굴 정보 함께 저장
                            new_violation = {
                                'violation_type': '넘어짐',
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'bbox': [x1, y1, x2, y2],
                                'person_box': [x1, y1, x2, y2],
                                'box': [x1, y1, x2, y2],
                                'confidence': fall_det['conf'],
                                'timestamp': timestamp,
                                'cam_id': cam_id
                            }
                            
                            # 얼굴 정보도 함께 저장 (이름 라벨링용)
                            new_face = {
                                'name': best_match_name,
                                'box': [x1, y1, x2, y2],
                                'bbox': [x1, y1, x2, y2],
                                'ppe_violations': ['넘어짐'],
                                'isViolation': True
                            }
                            
                            # 저장할 때 현재 시간 사용 (정리 로직에서 즉시 삭제 방지)
                            save_time = time.time()
                            model_results_cache[cam_id].append((save_time, {
                                'violations': [new_violation],
                                'recognized_faces': [new_face],
                                'frame_timestamp': save_time
                            }))
                            
                            if best_match_name != "Unknown":
                                logging.warning(f"[CAM-{cam_id}] ⚠️ 넘어짐 감지! 이름={best_match_name}, conf={fall_det['conf']:.2f}, 위치=({x1}, {y1})")
                            else:
                                logging.warning(f"[CAM-{cam_id}] ⚠️ 넘어짐 감지! conf={fall_det['conf']:.2f}, 위치=({x1}, {y1})")
        except Exception as e:
            logging.debug(f"Fall 모델 결과 저장 실패: {e}")
            with data_lock:
                fall_data['ready'] = True
    
    # 백그라운드에서 결과 저장
    yolo_executor.submit(save_violation_result, violation_future)
    yolo_executor.submit(save_pose_result, pose_future)
    if fall_future:
        yolo_executor.submit(save_fall_result, fall_future)
    
    # 🦬 buffalo_l 얼굴 인식: face_analyzer가 있으면 항상 실행 (face_detection_future 없어도 됨)
    if face_analyzer is not None:
        face_recognition_executor.submit(save_face_result, face_detection_future)


def _generate_person_box_key(cam_id: int, matched_entry: Optional[Dict], x1: int, y1: int, x2: int, y2: int) -> str:
    """
    person_box_key 생성 헬퍼 함수
    
    Args:
        cam_id: 카메라 ID
        matched_entry: 캐시에서 찾은 항목 (None 가능)
        x1, y1, x2, y2: 사람 박스 좌표
    
    Returns:
        person_box_key 문자열
    """
    if matched_entry is not None:
        cached_name = matched_entry.get('name', 'Unknown')
        if cached_name != "Unknown":
            return f"{cam_id}_{cached_name}"
    return f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

def process_single_frame(
    frame_bytes: bytes,
    cam_id: int
) -> Tuple[bytes, Dict[str, Any]]:
    """
    단일 프레임을 처리하고 결과를 반환합니다.
    
    Args:
        frame_bytes: 프레임 이미지 바이트 데이터
        cam_id: 카메라 ID
        
    Returns:
        Tuple[bytes, Dict[str, Any]]: 처리된 프레임 바이트와 결과 딕셔너리
    """
    # cam_id 타입 통일 (int)
    try:
        cam_id = int(cam_id)
    except (ValueError, TypeError):
        pass
    
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
    
    # 1단계 최적화: 프레임 스킵 30% (30 FPS 목표)
    # 10프레임 중 3개 스킵 = 30% 스킵
    if not hasattr(process_single_frame, '_frame_counters'):
        process_single_frame._frame_counters = {}
    if cam_id not in process_single_frame._frame_counters:
        process_single_frame._frame_counters[cam_id] = 0
    process_single_frame._frame_counters[cam_id] += 1
    frame_counter = process_single_frame._frame_counters[cam_id]
    
    # 프레임 스킵 비활성화 (실시간 처리 - 모든 프레임)
    # skip_pattern = [3, 6, 9]  # 비활성화
    should_skip = False  # 모든 프레임 처리
    
    if should_skip:
        # 스킵된 프레임: 프로덕션 최적화 - 최소한의 처리만 수행 (PIL 제거, OpenCV 직접 사용)
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                # 디코딩 실패 시 이전 프레임 재사용
                if cam_id in _last_rendered_frames:
                    last_frame_bytes, last_result = _last_rendered_frames[cam_id]
                    logging.debug(f"[CAM-{cam_id}] 프레임 스킵 (30% 최적화): {frame_counter}번째 프레임, 디코딩 실패로 이전 프레임 재사용")
                    return last_frame_bytes, last_result
                # 이전 프레임도 없으면 빈 프레임
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return buffer.tobytes(), {"timestamp": time.time(), "recognized_faces": [], "violations": [], "violation_count": 0, "performance": {"skipped": True}}
            
            orig_h, orig_w = frame.shape[:2]
            
            # 이전 결과에서 바운딩 박스 정보 가져오기
            last_result = None
            if cam_id in _last_rendered_frames:
                _, last_result = _last_rendered_frames[cam_id]
            
            # 렌더링 필요 여부 확인 (프레임 복사 최소화)
            needs_rendering = last_result and (len(last_result.get("recognized_faces", [])) > 0 or len(last_result.get("violations", [])) > 0)
            
            if needs_rendering:
                # 렌더링이 필요할 때만 프레임 복사 (메모리 최적화)
                processed_frame = frame.copy()
                recognized_faces = last_result.get("recognized_faces", [])
                violations = last_result.get("violations", [])
                
                # 공통 렌더링 함수 호출
                processed_frame = render_frame_results(
                    processed_frame,
                    recognized_faces,
                    violations,
                    cam_id,
                    orig_w,
                    orig_h
                )
            else:
                # 렌더링이 필요 없으면 프레임 복사 없이 원본 사용
                processed_frame = frame

            
            # 리사이즈 최적화: 필요할 때만 리사이즈
            stream_width = 1280
            if processed_frame.shape[1] > stream_width:
                aspect_ratio = processed_frame.shape[0] / processed_frame.shape[1]
                stream_height = int(stream_width * aspect_ratio)
                processed_frame = cv2.resize(processed_frame, (stream_width, stream_height), 
                                           interpolation=cv2.INTER_LINEAR)
            
            # 인코딩 품질 조정 (프로덕션: 95 - 고화질)
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 이전 결과 재사용 (바운딩 박스 정보 유지)
            result = last_result.copy() if last_result else {
                "recognized_faces": [],
                "violations": [],
                "violation_count": 0,
                "frame_width": orig_w,
                "frame_height": orig_h,
                "cam_id": cam_id
            }
            # 타임스탬프 갱신 (새 프레임이므로)
            result["timestamp"] = time.time()
            result["performance"] = {"skipped": True}
            
            logging.debug(f"[CAM-{cam_id}] 프레임 스킵 (30% 최적화): {frame_counter}번째 프레임, AI 처리 스킵 (프로덕션 최적화)")
            return buffer.tobytes(), result
            
        except Exception as e:
            logging.warning(f"[CAM-{cam_id}] 스킵 프레임 처리 오류: {e}")
            # 오류 시 이전 프레임 재사용
            if cam_id in _last_rendered_frames:
                last_frame_bytes, last_result = _last_rendered_frames[cam_id]
                return last_frame_bytes, last_result
            # 이전 프레임도 없으면 빈 결과 반환
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes(), {"recognized_faces": [], "violations": [], "violation_count": 0, "performance": {"skipped": True}}
    
    # SafetySystem 초기화 확인 및 에러 처리 개선
    # 전역 변수 안전하게 읽기 (멀티스레드 환경 대비 - 락 사용)
    # state 모듈을 직접 import하여 최신 값을 읽도록 수정
    with safety_system_lock:
        safety_system = state.safety_system_instance
        is_none = safety_system is None
        logging.debug(f"[CAM-{cam_id}] SafetySystem 확인: 존재={not is_none}")
    
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
    
    # 함수 시작 (로깅 최소화로 성능 향상)
    logging.debug(f"[CAM-{cam_id}] process_single_frame 시작")

    # 함수 내에서 orig_h, orig_w 기본값 설정 (오류 방지)
    orig_h, orig_w = 480, 640
    frame = None # 오류 발생 시 사용하기 위해 초기화

    # 프레임 보장 방식: 타임스탬프 기록 (프레임 버퍼 및 모델 결과 캐시에 사용)
    timestamp = time.time()
    
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
            return buffer.tobytes(), {"timestamp": time.time(), "recognized_faces": [], "violations": [], "violation_count": 0}
        orig_h, orig_w = frame.shape[:2]
        
        # 프레임 보장 방식: 프레임 버퍼에 저장 (최근 1초만 유지)
        with frame_buffer_lock:
            frame_buffer[cam_id].append((timestamp, frame_bytes, frame.copy()))
            # 오래된 프레임 제거 (1초 이상 지난 프레임)
            frame_buffer[cam_id] = [
                (ts, fb, f) for ts, fb, f in frame_buffer[cam_id]
                if timestamp - ts <= MAX_BUFFER_SECONDS
            ]
        
        # 프레임 보장 방식: 캐시에서 가장 최근 결과 찾기
        best_result = None
        best_time_diff = float('inf')
        MAX_CACHE_AGE = 3.0  # 캐시 유효 시간 (2.0 -> 3.0초로 확대, 깜빡임 방지)
        current_time = time.time()
        
        with results_cache_lock:
            if cam_id in model_results_cache and len(model_results_cache[cam_id]) > 0:
                # 가장 최근 결과 가져오기 (타임스탬프가 아닌 현재 시간 기준)
                for result_ts, result_dict in reversed(model_results_cache[cam_id]):
                    time_diff = current_time - result_ts  # 현재 시간과의 차이로 계산
                    if time_diff <= MAX_CACHE_AGE:
                        # 가장 최근 결과를 무조건 사용 (faces/violations 유무 관계없이)
                        best_result = result_dict
                        best_time_diff = time_diff
                        # 디버깅 로그 추가
                        violations_count = len(result_dict.get('violations', []))
                        faces_count = len(result_dict.get('recognized_faces', []))
                        recognized_names = [f.get('name', 'Unknown') for f in result_dict.get('recognized_faces', [])]
                        logging.debug(f"[CAM-{cam_id}] 캐시에서 결과 찾음: age={time_diff:.3f}s, violations={violations_count}개, faces={faces_count}개, 이름={recognized_names}")
                        break  # 가장 최근의 유효한 결과 사용
                
                # 오래된 결과 제거 (5초 이상, 깜빡임 방지)
                model_results_cache[cam_id] = [
                    (ts, rd) for ts, rd in model_results_cache[cam_id]
                    if current_time - ts <= 5.0
                ]
        
        # 프레임 유효성 검사: 크기가 너무 작거나 비어있는지 확인
        if orig_h < 100 or orig_w < 100:
            logging.warning(f"프레임 크기가 너무 작음: {orig_w}x{orig_h} (CAM-{cam_id})")
            empty_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            return buffer.tobytes(), {"timestamp": time.time(), "recognized_faces": [], "violations": [], "violation_count": 0}
        
        # 검은 프레임 체크 (캠이 꺼진 상태): 평균 밝기가 매우 낮으면 AI 처리만 건너뜀
        # 기준: BGR 평균이 2.0 미만이면 거의 완전히 검은 프레임으로 간주
        frame_mean = np.mean(frame)
        if frame_mean < 2.0:  # 평균 밝기가 2 미만이면 검은 프레임으로 간주
            logging.debug(f"[CAM-{cam_id}] 검은 프레임 감지 - AI 처리 건너뜀")
            # 검은 프레임도 스트림에는 표시하되, 위반 감지는 하지 않음 (품질 최적화: 100 → 85)
            # 복사 최적화: 인코딩만 필요하므로 원본 프레임 직접 사용
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes(), {"timestamp": time.time(), "recognized_faces": [], "violations": [], "violation_count": 0, "performance": {}}

        # 2. 모델 입력 크기에 맞게 리사이즈 (원본 비율 유지, 최적화)
        # 각 카메라의 원본 비율을 유지하면서 최대 크기를 제한
        # 예: 1920x1080 (16:9) -> 1024x576 (16:9 유지, 최적)
        #     1280x720 (16:9) -> 1024x576 (16:9 유지, 최적)
        resize_start = time.time()
        max_input_size = max(config.SystemConfig.MODEL_INPUT_WIDTH, config.SystemConfig.MODEL_INPUT_HEIGHT)
        
        # 원본 비율 계산
        orig_ratio = orig_w / orig_h
        
        # 최대 크기를 제한하면서 비율 유지 (다운스케일링만 수행, 업스케일링은 하지 않음)
        if orig_w > max_input_size or orig_h > max_input_size:
            # 다운스케일링: 비율 유지하면서 최대 크기 제한
            if orig_w > orig_h:
                # 가로가 더 긴 경우 (landscape)
                new_w = max_input_size
                new_h = int(max_input_size / orig_ratio)
            else:
                # 세로가 더 긴 경우 (portrait)
                new_h = max_input_size
                new_w = int(max_input_size * orig_ratio)
            # INTER_LINEAR가 INTER_AREA보다 약간 빠름 (속도 최적화)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            logging.debug(f"[CAM-{cam_id}] 프레임 리사이즈 (다운스케일, 비율 유지): {orig_w}x{orig_h} -> {new_w}x{new_h} (비율: {orig_ratio:.3f})")
        else:
            # 원본이 작은 경우: 그대로 사용 (업스케일링 하지 않음 - 품질 저하 방지)
            # 메모리 최적화: 복사 대신 뷰 사용 (리사이즈가 없으므로 원본 프레임 재사용)
            resized_frame = frame  # copy() 제거: 리사이즈가 없으므로 원본 프레임 재사용
            logging.debug(f"[CAM-{cam_id}] 프레임 리사이즈 건너뜀 (원본 크기 유지): {orig_w}x{orig_h} (비율: {orig_ratio:.3f})")
        
        # 바운딩 박스 좌표 변환을 위한 스케일 계산 (리사이즈된 크기 기준)
        # 정확한 스케일 계산: 원본 크기 / 리사이즈된 크기
        resized_w = resized_frame.shape[1]
        resized_h = resized_frame.shape[0]
        w_scale = orig_w / resized_w
        h_scale = orig_h / resized_h
        
        # 스케일 값 로깅 (디버깅용)
        logging.debug(f"[CAM-{cam_id}] 스케일 계산: 원본={orig_w}x{orig_h}, 리사이즈={resized_w}x{resized_h}, w_scale={w_scale:.4f}, h_scale={h_scale:.4f}")
        perf_timings['resize'] = (time.time() - resize_start) * 1000  # ms

        # 3. 처리된 프레임 생성 (렌더링이 필요할 때만 복사)
        # 메모리 최적화: 렌더링이 필요할 때만 복사 (나중에 복사)
        processed_frame = None  # 나중에 필요할 때만 복사
        renderer = utils.TextRenderer(frame.shape)

        # 4. 모든 모델을 병렬로 실행 (개별 실행 후 결과만 합치기)
        model_start = time.time()
        
        logging.debug(f"[CAM-{cam_id}] 프레임 처리 시작 (수신 크기: {len(frame_bytes)} bytes)")

        # SafetySystem은 이미 위에서 락으로 읽었으므로 재사용 (1013번 줄 제거)
        # safety_system 변수는 이미 929번 줄에서 설정됨
        
        # GPU 최적화 설정 (half precision, 배치 처리 등)
        base_half_precision = config.SystemConfig.ENABLE_HALF_PRECISION and 'cuda' in str(safety_system.device)
        
        # YOLO 모델 입력 크기 설정 (모델별로 다름)
        # TensorRT 엔진이 640x640으로 빌드되었으므로 640x640 사용 (고정 크기)
        # 동적 입력 크기를 지원하지 않으므로 엔진 빌드 크기와 일치해야 함
        violation_imgsz = 640  # Violation 모델: 640x640 (TensorRT 엔진 크기와 일치)
        pose_imgsz = 640       # Pose 모델: 640x640 (TensorRT 엔진 크기와 일치)
        logging.debug(f"TensorRT 엔진 사용: Violation={violation_imgsz}x{violation_imgsz}, Pose={pose_imgsz}x{pose_imgsz}")
        
        # FPS 기반 max_det 동적 조정 (인원이 많을 때 FPS 저하 방지)
        # NMS 처리 시간이 max_det에 비례하여 증가하므로, FPS가 낮을 때 max_det를 낮춰 처리 속도 향상
        violation_max_det = 50  # 기본값
        pose_max_det = 30  # 기본값
        try:
            with frame_stats_lock:
                cam_stats = frame_stats.get(cam_id, {})
                recent_frames = cam_stats.get('recent_frame_times', [])
                if len(recent_frames) >= 2:
                    time_span = recent_frames[-1] - recent_frames[0]
                    if time_span > 0:
                        current_fps = (len(recent_frames) - 1) / time_span
                        
                        # FPS 기반 max_det 조정 (NMS 처리 시간 최적화, 더 공격적인 감소)
                        if current_fps >= 30:
                            violation_max_det = 30  # 높은 FPS: 50 -> 30 (40% 감소)
                            pose_max_det = 20       # 높은 FPS: 30 -> 20 (33% 감소)
                        elif current_fps >= 25:
                            violation_max_det = 25  # 중간 FPS: 35 -> 25 (29% 감소)
                            pose_max_det = 18       # 중간 FPS: 25 -> 18 (28% 감소)
                        elif current_fps >= 20:
                            violation_max_det = 20  # 중간 FPS: 35 -> 20 (43% 감소)
                            pose_max_det = 15       # 중간 FPS: 25 -> 15 (40% 감소)
                        elif current_fps >= 15:
                            violation_max_det = 15  # 낮은 FPS: 25 -> 15 (40% 감소)
                            pose_max_det = 12       # 낮은 FPS: 18 -> 12 (33% 감소)
                        else:
                            violation_max_det = 12  # 매우 낮은 FPS: 20 -> 12 (40% 감소)
                            pose_max_det = 10       # 매우 낮은 FPS: 15 -> 10 (33% 감소)
                        
                        logging.debug(f"[CAM-{cam_id}] FPS 기반 max_det 조정: FPS={current_fps:.1f}, Violation={violation_max_det}, Pose={pose_max_det}")
        except Exception:
            pass  # 예외 시 기본값 사용
        
        # 오인식 방지를 위해 NMS IoU와 max_det도 조정
        # 오인식 방지를 위해 NMS IoU와 max_det 조정
        # ONNX 모델은 ONNX Runtime을 사용하므로 PyTorch device 파라미터를 전달하지 않음
        # ONNX Runtime은 모델 로드 시 이미 GPU/CPU를 설정했으므로 device 파라미터 불필요
        # PyTorch CUDA가 감지되지 않아도 ONNX Runtime CUDA Provider는 사용 가능할 수 있음
        # YOLO가 device 파라미터를 요구하므로, ONNX 모델일 때는 None 또는 전달하지 않음
        violation_kwargs = {
            'conf': config.Thresholds.YOLO_CONFIDENCE,
            'verbose': False,
            'iou': 0.55,  # NMS IoU (0.5 -> 0.55, 중복 박스 제거 강화, 오인식 방지)
            'max_det': violation_max_det,  # FPS 기반 동적 조정 (기본값: 50)
            # device 파라미터 제거: ONNX 모델은 ONNX Runtime이 자동으로 처리
            # YOLO가 내부적으로 device를 요구하면 None으로 설정하거나 전달하지 않음
        }
        pose_kwargs = {
            'conf': config.Thresholds.POSE_CONFIDENCE,
            'verbose': False,
            'iou': 0.55,  # NMS IoU (0.5 -> 0.55, 중복 박스 제거 강화, 오인식 방지)
            'max_det': pose_max_det,  # FPS 기반 동적 조정 (기본값: 30)
            # device 파라미터 제거: ONNX 모델은 ONNX Runtime이 자동으로 처리
            # YOLO가 내부적으로 device를 요구하면 None으로 설정하거나 전달하지 않음
        }
        
        # ONNX 모델만 사용하므로 항상 imgsz 설정
        violation_kwargs.update({
            'half': base_half_precision,
            'imgsz': violation_imgsz,  # Violation 모델: 832x832
        })
        pose_kwargs.update({
            'half': base_half_precision,
            'imgsz': pose_imgsz,  # Pose 모델: 832x832
        })
        
        # 얼굴 인식 모델 및 DB 가져오기 (병렬 실행 준비)
        face_model = safety_system.face_model
        face_analyzer = safety_system.face_analyzer  # buffalo_l (실제 사용!)
        fast_recognizer = safety_system.fast_recognizer  # 폴백용
        face_database = safety_system.face_database
        
        # 🔍 디버그: face_analyzer 상태 확인 (처음 1번만)
        if not hasattr(process_single_frame, '_face_analyzer_logged'):
            logging.warning(f"🔍 [초기화] face_analyzer={face_analyzer is not None}, face_database={face_database is not None}")
            process_single_frame._face_analyzer_logged = True
        
        # 얼굴 탐지 간격 체크 (병렬 실행 전에 확인)
        should_detect_faces_global = True
        with face_detection_lock:
            current_frame = frame_stats.get(cam_id, {}).get('frame_count', 0)
            last_frame = last_face_detection_frame.get(cam_id, -config.Thresholds.FACE_DETECTION_INTERVAL)
            if current_frame - last_frame < config.Thresholds.FACE_DETECTION_INTERVAL:
                should_detect_faces_global = False
        
        # 모든 모델을 병렬로 실행 (개별 실행 후 결과만 합치기)
        # GPU 메모리 정리는 필요시에만 (매 프레임은 오버헤드)
        # 100 프레임마다 한 번씩만 정리 (멀티 GPU 지원)
        # ONNX Runtime은 자체적으로 메모리 관리를 하므로 PyTorch 메모리 정리는 선택적
        if 'cuda' in str(safety_system.device) and frame_stats.get(cam_id, {}).get('frame_count', 0) % 100 == 0:
            # PyTorch CUDA 메모리 정리 (호환성을 위해, ONNX Runtime은 자체 관리)
            try:
                if torch.cuda.is_available():
                    for gpu_id in range(torch.cuda.device_count()):
                        torch.cuda.empty_cache()
            except:
                pass
        
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
            # ONNX 모델은 ONNX Runtime을 사용하므로 device 파라미터 없이 실행
            # ONNX Runtime은 모델 로드 시 이미 GPU/CPU Provider를 설정했으므로 자동 처리
            return safety_system.violation_model(resized_frame, **violation_kwargs)
        
        def run_pose_model():
            # ONNX 모델은 ONNX Runtime을 사용하므로 device 파라미터 없이 실행
            # ONNX Runtime은 모델 로드 시 이미 GPU/CPU Provider를 설정했으므로 자동 처리
            return safety_system.pose_model(resized_frame, **pose_kwargs)
        
        def run_fall_model():
            # Fall Detection 모델 활성화 - 완전히 쓰러진 사람은 Pose가 못잡으므로 필요
            if safety_system.fall_model is None:
                return None
            fall_kwargs = {
                'conf': 0.45,  # Fall 감지 임계값
                'iou': 0.5,
                'verbose': False,
                'classes': [1],  # Fall 클래스만
            }
            # ⭐ 다른 모델들과 동일하게 resized_frame 사용 (좌표 스케일링 통일)
            return safety_system.fall_model(resized_frame, **fall_kwargs)
        
        # ⭐ PPE(Violation)와 위험 감지(Pose), Fall Detection 모델을 병렬로 실행
        import sys
        logging.debug(f"[CAM-{cam_id}] 🔄 YOLO 모델 실행 준비: Violation={safety_system.violation_model is not None}, Pose={safety_system.pose_model is not None}, 입력 크기={resized_frame.shape}")
        violation_future = yolo_executor.submit(run_violation_model)  # PPE 위반 감지 모델 (병렬)
        pose_future = yolo_executor.submit(run_pose_model)  # 위험 행동 감지 모델 (병렬)
        fall_future = yolo_executor.submit(run_fall_model)  # 넘어짐 감지 모델 (병렬, 전체 프레임)
        # print 제거 - 로그 파일에만 기록 (콘솔 노이즈 감소)
        logging.debug(f"[CAM-{cam_id}] YOLO 모델 병렬 실행 시작")
        
        # ========================================
        # 2단계 파이프라인: 얼굴 감지는 조건부 실행
        # ========================================
        # Step 1: YOLO Pose + PPE는 항상 실행 (위에서 이미 실행됨)
        # Step 2: 얼굴 감지는 아래 조건 중 하나라도 만족할 때만 실행
        #   - 조건 A: 위반 사항 발생 시 (쓰러짐 OR 안전장비 미착용)
        #   - 조건 B: 새로운 사람 등장 시 (Track ID가 처음 생성되었을 때)
        #   - 조건 C: 주기적 확인 (해당 ID에 대해 1초에 1번만)
        
        face_detection_future = None
        should_run_face_detection = False
        face_detection_reason = "none"
        
        # ⭐ buffalo_l로 얼굴 감지 (face_analyzer 사용)
        # face_model은 None (YOLO Face 대신 buffalo_l 사용)
        if face_analyzer is None:
            logging.debug(f"🔍 [CAM-{cam_id}] 얼굴 감지 스킵: face_analyzer=None")
        elif not should_detect_faces_global:
            with face_detection_lock:
                current_frame = frame_stats.get(cam_id, {}).get('frame_count', 0)
                last_frame = last_face_detection_frame.get(cam_id, -config.Thresholds.FACE_DETECTION_INTERVAL)
                frame_interval = current_frame - last_frame
            logging.debug(f"🔍 [CAM-{cam_id}] 얼굴 감지 스킵: 간격 부족 (현재={current_frame}, 마지막={last_frame}, 간격={frame_interval}, 최소={config.Thresholds.FACE_DETECTION_INTERVAL})")
        
        # ⭐ buffalo_l 사용 시 YOLO Face 실행하지 않음
        # face_analyzer가 있으면 save_face_result에서 buffalo_l로 통합 처리
        if face_analyzer is not None and should_detect_faces_global:
            logging.debug(f"🦬 [CAM-{cam_id}] buffalo_l 얼굴 감지 예정 (save_face_result에서 처리)")
            # YOLO Face 대신 buffalo_l 사용 (save_face_result에서 face_analyzer.get() 호출)
            # face_detection_future는 None으로 유지 (buffalo_l은 별도 처리)
        
        # 모든 모델 결과 대기 (병렬 실행, 타임아웃 최적화: 실시간 처리 속도 향상)
        # GPU 환경에서는 첫 실행 시 warmup이 필요하므로 타임아웃 증가
        # CPU 모드에서는 처리 시간이 길어서 타임아웃을 더 길게 설정
        if not hasattr(process_single_frame, '_model_warmed_up'):
            # 첫 실행: warmup을 위해 타임아웃 증가
            model_timeout = 5.0 if 'cuda' in str(safety_system.device) else 10.0  # CPU: 6.0 -> 10.0 (병목 해결)
            process_single_frame._model_warmed_up = True
        else:
            # 이후 실행: 정상 처리 속도 (CPU 모드에서는 더 긴 타임아웃 필요)
            model_timeout = 3.0 if 'cuda' in str(safety_system.device) else 10.0  # CPU: 4.0 -> 10.0 (병목 해결)
        
        # ========================================
        # 동기식 처리: 모델 결과를 기다렸다가 바로 렌더링 (실시간, 지연 없음)
        # ========================================
        
        # 1. Violation + Pose 모델 결과 동시 기다리기 (타임아웃 없음)
        violation_start = time.time()
        try:
            violation_results = violation_future.result() or []
        except Exception:
            violation_results = []
        perf_timings['yolo_violation'] = (time.time() - violation_start) * 1000
        
        pose_start = time.time()
        try:
            pose_results = pose_future.result() or []
        except Exception:
            pose_results = []
        perf_timings['yolo_pose'] = (time.time() - pose_start) * 1000
        
        # 3. 결과 즉시 파싱 (캐시 사용 안 함)
        best_result = None
        recognized_faces = []
        violations_found = []
        all_detections = {}
        
        # Violation 결과 파싱 (Person 포함)
        if violation_results and len(violation_results) > 0:
            for det in violation_results[0].boxes:
                class_id = int(det.cls[0])
                class_name = safety_system.violation_model.names[class_id]
                conf = float(det.conf[0])
                
                if class_name in config.Thresholds.IGNORED_CLASSES:
                    continue
                
                # Person 클래스는 별도 임계값 사용
                if class_name == 'Person':
                    class_threshold = config.Thresholds.PERSON_CONFIDENCE
                else:
                    class_threshold = config.Thresholds.CLASS_CONFIDENCE_THRESHOLDS.get(
                        class_name, config.Thresholds.YOLO_CONFIDENCE
                    )
                
                if conf >= class_threshold:
                    bbox_resized = det.xyxy[0].cpu().numpy()
                    bbox_original = bbox_resized * np.array([w_scale, h_scale, w_scale, h_scale])
                    bbox_clipped = utils.clip_bbox_xyxy(bbox_original, orig_w, orig_h)
                    if bbox_clipped is not None:
                        if class_name not in all_detections:
                            all_detections[class_name] = []
                        all_detections[class_name].append({'bbox': list(bbox_clipped), 'conf': conf})
        
        # Pose 결과 파싱 (person boxes)
        person_boxes = []
        frame_area = orig_w * orig_h
        if pose_results and len(pose_results) > 0 and pose_results[0].boxes is not None:
            for idx, box in enumerate(pose_results[0].boxes.xyxy.cpu().numpy()):
                conf = float(pose_results[0].boxes.conf.cpu().numpy()[idx]) if pose_results[0].boxes.conf is not None else 0.5
                if conf >= 0.25:
                    scaled_box = box * np.array([w_scale, h_scale, w_scale, h_scale])
                    clipped = utils.clip_bbox_xyxy(scaled_box, orig_w, orig_h)
                    if clipped is not None:
                        # ⭐ Pose 박스도 비율 확인해서 넘어짐 감지
                        box_w = clipped[2] - clipped[0]
                        box_h = clipped[3] - clipped[1]
                        box_area = box_w * box_h
                        box_ratio = box_w / box_h if box_h > 0 else 0
                        
                        # 넘어짐: 가로가 세로보다 1.6배 이상 (Fall 모델 보조, 1.3 → 1.6)
                        is_fall = (box_ratio >= 1.6 and box_area >= 10000 and box_area <= frame_area * 0.4)
                        
                        if is_fall:
                            logging.warning(f"[CAM-{cam_id}] 🔻 Pose 넘어짐 감지: 비율={box_ratio:.2f}, 면적={box_area:.0f}")
                        
                        person_boxes.append({'box': list(clipped), 'source': 'pose', 'is_fall': is_fall})
        
        # PPE Person 박스 추가 (Pose가 못 잡은 넘어진 사람 감지)
        if 'Person' in all_detections:
            for ppe_person in all_detections['Person']:
                ppe_bbox = ppe_person['bbox']
                is_new = True
                
                # Pose 박스와 비교 (중복 제거)
                for pose_box in person_boxes:
                    if utils.calculate_iou(ppe_bbox, pose_box['box']) > 0.3:
                        is_new = False
                        break
                
                if is_new:
                    # 넘어진 사람 후보 판단 (엄격한 기준)
                    box_w = ppe_bbox[2] - ppe_bbox[0]
                    box_h = ppe_bbox[3] - ppe_bbox[1]
                    box_area = box_w * box_h
                    box_ratio = box_w / box_h if box_h > 0 else 0
                    
                    # 넘어짐 조건 (Fall 모델 보조):
                    # 1. 가로가 세로보다 1.8배 이상 (Fall 모델이 주력이므로 보조는 엄격하게)
                    # 2. 박스 면적이 최소 12000 픽셀 이상 (노이즈 제외)
                    # 3. 박스 면적이 프레임의 30% 이하
                    frame_area = orig_w * orig_h
                    is_fall = (box_ratio >= 1.8 and 
                              box_area >= 12000 and 
                              box_area <= frame_area * 0.30)
                    
                    person_boxes.append({
                        'box': ppe_bbox,
                        'source': 'ppe',
                        'is_fall': is_fall
                    })
                    
                    if is_fall:
                        logging.warning(f"[CAM-{cam_id}] 🔻 넘어짐 감지: 비율={box_ratio:.2f}, 면적={box_area:.0f}")
        
        # PPE 위반 확인 (person_box와 PPE 매칭)
        for person_data in person_boxes:
            box = person_data['box']
            x1, y1, x2, y2 = map(int, box)
            is_fall = person_data.get('is_fall', False)
            ppe_violations = []
            
            # 넘어짐이면 바로 위반 추가
            if is_fall:
                ppe_violations.append('넘어짐')
            
            # 안전모 체크 (거리 기반 매칭 - 더 정확)
            if 'NO-Hardhat' in all_detections:
                person_cx, person_cy = (x1 + x2) / 2, (y1 + y2) / 2
                person_h = y2 - y1
                for ppe in all_detections['NO-Hardhat']:
                    ppe_bbox = ppe['bbox']
                    ppe_cx = (ppe_bbox[0] + ppe_bbox[2]) / 2
                    ppe_cy = (ppe_bbox[1] + ppe_bbox[3]) / 2
                    # 안전모는 머리 위치 (상단 30%)에 있어야 함
                    if abs(ppe_cx - person_cx) < (x2 - x1) * 0.5 and ppe_cy < y1 + person_h * 0.4:
                        ppe_violations.append('안전모')
                        break
            
            # 마스크 체크 비활성화 (사용자 요청)
            # if 'NO-Mask' in all_detections:
            #     pass
            
            # 안전조끼 체크 (거리 기반)
            if 'NO-Safety Vest' in all_detections:
                person_cx = (x1 + x2) / 2
                person_h = y2 - y1
                for ppe in all_detections['NO-Safety Vest']:
                    ppe_bbox = ppe['bbox']
                    ppe_cx = (ppe_bbox[0] + ppe_bbox[2]) / 2
                    ppe_cy = (ppe_bbox[1] + ppe_bbox[3]) / 2
                    # 안전조끼는 몸통 위치 (30%~80%)에 있어야 함
                    if abs(ppe_cx - person_cx) < (x2 - x1) * 0.6 and y1 + person_h * 0.2 < ppe_cy < y1 + person_h * 0.8:
                        ppe_violations.append('안전조끼')
                        break
            
            # 모든 사람 박스 표시 (위반 여부와 관계없이)
            has_violation = bool(ppe_violations) or is_fall
            recognized_faces.append({
                'box': [x1, y1, x2, y2],
                'bbox': [x1, y1, x2, y2],
                'name': 'Unknown',
                'ppe_violations': ppe_violations,
                'isViolation': has_violation
            })
            if has_violation:
                violations_found.append({
                    'person_box': [x1, y1, x2, y2],
                    'violations': ppe_violations
                })
        
        # 백그라운드에서 얼굴 인식 계속 실행 (결과는 캐시에 저장)
        _submit_models_background_simple(
            frame, resized_frame, cam_id, timestamp, safety_system,
            violation_future, pose_future, fall_future, face_detection_future,
            violation_kwargs, pose_kwargs, face_model, face_analyzer, fast_recognizer,
            face_database, orig_w, orig_h, w_scale, h_scale
        )
        
        perf_timings['face_recognition'] = 0.0  # 백그라운드 처리
        
        # 캐시에서 얼굴 인식 이름 가져오기 (있으면)
        from state import get_latest_cache
        cached_result = get_latest_cache(cam_id, max_age=CACHE_TTL)
            
        # 캐시에서 얼굴 이름 가져와서 동기식 결과와 병합
        if cached_result:
            cached_faces = cached_result.get('recognized_faces', [])
            for rf in recognized_faces:
                rf_cx = (rf['box'][0] + rf['box'][2]) / 2
                rf_cy = (rf['box'][1] + rf['box'][3]) / 2
                
                for cf in cached_faces:
                    cf_box = cf.get('box', [])
                    cf_name = cf.get('name', 'Unknown')
                    if len(cf_box) == 4 and cf_name != 'Unknown':
                        cf_cx = (cf_box[0] + cf_box[2]) / 2
                        cf_cy = (cf_box[1] + cf_box[3]) / 2
                        distance = ((rf_cx - cf_cx)**2 + (rf_cy - cf_cy)**2)**0.5
                        if distance < 150:  # 150픽셀 이내면 같은 사람
                            rf['name'] = cf_name
                            break
        
        # 동기식 결과로 렌더링 (캐시 대기 없음)
        if recognized_faces or violations_found:
            if processed_frame is None:
                processed_frame = frame.copy()
            
            # ⭐⭐ 중복 박스 제거 (같은 위치에 여러 박스 방지)
            # violations_found 중복 제거
            deduplicated_violations = []
            used_violation_boxes = set()
            for vf in violations_found:
                box = vf.get('person_box', [])
                if len(box) != 4:
                    continue
                box_key = (int(box[0]//20), int(box[1]//20), int(box[2]//20), int(box[3]//20))
                if box_key not in used_violation_boxes:
                    used_violation_boxes.add(box_key)
                    deduplicated_violations.append(vf)
            violations_found = deduplicated_violations
            
            # recognized_faces 중복 제거
            # 박스 좌표가 유사하면 (20픽셀 단위 그룹화) 하나만 유지
            deduplicated_faces = []
            used_boxes = set()
            for rf in recognized_faces:
                box = rf.get('box', [])
                if len(box) != 4:
                    continue
                box_key = (int(box[0]//20), int(box[1]//20), int(box[2]//20), int(box[3]//20))  # 20픽셀 단위로 그룹화
                if box_key not in used_boxes:
                    used_boxes.add(box_key)
                    deduplicated_faces.append(rf)
                else:
                    # 이미 있는 박스면 이름이 있는 쪽 우선
                    for i, existing_rf in enumerate(deduplicated_faces):
                        existing_box = existing_rf.get('box', [])
                        if len(existing_box) == 4:
                            existing_key = (int(existing_box[0]//20), int(existing_box[1]//20), int(existing_box[2]//20), int(existing_box[3]//20))
                            if existing_key == box_key:
                                # 이름이 있는 쪽 우선
                                if rf.get('name', 'Unknown') != 'Unknown' and existing_rf.get('name', 'Unknown') == 'Unknown':
                                    deduplicated_faces[i] = rf
                                break
            
            recognized_faces = deduplicated_faces
            
            # 공통 렌더링 함수 호출
            processed_frame = render_frame_results(
                processed_frame,
                recognized_faces,
                violations_found,
                cam_id,
                orig_w,
                orig_h
            )
        
            # 렌더링 완료 후 바로 리턴 (프레임 보장 방식)
            perf_timings['total'] = (time.time() - total_start) * 1000
            logging.debug(f"[CAM-{cam_id}] 캐시 결과 렌더링 완료: 얼굴={len(recognized_faces)}개, 위반={len(violations_found)}개, 처리시간={perf_timings['total']:.1f}ms")
            
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes(), {
                "timestamp": time.time(),  # 타임스탬프 추가
                "recognized_faces": recognized_faces,
                "violations": violations_found,
                "violation_count": len(violations_found),
                "performance": perf_timings,
                "frame_width": orig_w,
                "frame_height": orig_h,
                "cam_id": cam_id
            }
        
        # 프레임 보장 방식: 캐시에 결과가 없으면 원본 프레임 반환
        # 백그라운드에서 모델 처리 중이므로, 다음 프레임에서 결과가 표시됨
        if not best_result:
            logging.debug(f"[CAM-{cam_id}] 캐시에 결과 없음, 원본 프레임 반환")
            perf_timings['total'] = (time.time() - total_start) * 1000
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes(), {
                "timestamp": time.time(),  # 타임스탬프 추가
                "recognized_faces": [],
                "violations": [],
                "violation_count": 0,
                "performance": perf_timings,
                "frame_width": orig_w,
                "frame_height": orig_h,
                "cam_id": cam_id
            }
        
        # 아래 코드는 실행되지 않음 (위에서 모두 리턴)
        # 기존 파싱 로직은 백그라운드에서 처리됨
        parse_start = time.time()
        perf_timings['parse_results'] = 0.0
        
        if False:  # 기존 파싱 로직 비활성화 (백그라운드에서 처리)
            # 5-1. YOLO violation 결과 파싱
            all_detections = {}
            filtered_count = 0  # if 블록 밖에서 초기화
            low_conf_count = 0  # if 블록 밖에서 초기화
            if violation_results and len(violation_results) > 0:
                violation_box_count = len(violation_results[0].boxes) if violation_results[0].boxes is not None else 0
                logging.debug(f"[CAM-{cam_id}] 📦 YOLO Violation 결과 파싱 시작: {violation_box_count}개 박스, confidence 임계값={config.Thresholds.YOLO_CONFIDENCE}")
                for det in violation_results[0].boxes:
                    class_id = int(det.cls[0])
                    class_name = safety_system.violation_model.names[class_id]
                    conf = float(det.conf[0])
                    
                    # Safety Con 등 오탐지 클래스 필터링
                    if class_name in config.Thresholds.IGNORED_CLASSES:
                        filtered_count += 1
                        logging.debug(f"[CAM-{cam_id}] 클래스 필터링: {class_name} (IGNORED_CLASSES)")
                        continue
                    
                    # 클래스별 confidence 임계값 적용 (안전조끼 인식 개선)
                    class_threshold = config.Thresholds.CLASS_CONFIDENCE_THRESHOLDS.get(
                        class_name, 
                        config.Thresholds.YOLO_CONFIDENCE
                    )
                    
                    if conf >= class_threshold:
                        # 리사이즈된 프레임 기준 좌표를 원본 프레임 크기로 스케일링
                        bbox_resized = det.xyxy[0].cpu().numpy()
                        bbox_original = bbox_resized * np.array([w_scale, h_scale, w_scale, h_scale])
                        bbox_clipped = utils.clip_bbox_xyxy(bbox_original, orig_w, orig_h)
                        if bbox_clipped is not None:
                            if class_name not in all_detections:
                                all_detections[class_name] = []
                            # clip_bbox_xyxy는 tuple을 반환하므로 list()로 변환
                            all_detections[class_name].append({'bbox': list(bbox_clipped), 'conf': conf})
                            # PPE 클래스인 경우에만 상세 로깅 (디버깅)
                            is_ppe_class = class_name in ['Hardhat', 'NO-Hardhat', 'Mask', 'NO-Mask', 'Safety Vest', 'NO-Safety Vest']
                            if is_ppe_class:
                                logging.info(f"[CAM-{cam_id}] ✅ PPE 감지: {class_name} (conf={conf:.3f}, threshold={class_threshold:.3f}, bbox={bbox_clipped})")
                            else:
                                logging.debug(f"[CAM-{cam_id}] ✅ Violation 감지: {class_name} (conf={conf:.3f}, threshold={class_threshold:.3f}, bbox={bbox_clipped})")
                    else:
                        low_conf_count += 1
                        logging.debug(f"[CAM-{cam_id}] 낮은 confidence: {class_name} (conf={conf:.3f} < {class_threshold:.3f})")
            
            if filtered_count > 0 or low_conf_count > 0:
                logging.info(f"[CAM-{cam_id}] 필터링 통계: 필터링됨={filtered_count}개, 낮은 confidence={low_conf_count}개, 최종 감지={sum(len(v) for v in all_detections.values())}개")
            else:
                logging.warning(f"[CAM-{cam_id}] ⚠️ YOLO Violation 모델 결과 없음 또는 빈 결과 (violation_results={violation_results})")
            
            if all_detections:
                logging.debug(f"[CAM-{cam_id}] 총 {sum(len(v) for v in all_detections.values())}개 violation 감지: {list(all_detections.keys())}")
            
            # 5-2. 얼굴 감지 결과 처리 (YOLO 결과를 InsightFace 형식으로 변환)
            recognized_faces = []
            violations_found = []
            face_detected_boxes = []  # 얼굴 기반 박스 (뒤에 있는 사람용)
            
            # YOLO 얼굴 감지 결과를 InsightFace 형식으로 변환
            faces_in_frame = []
            result = None  # 초기화 추가 (NameError 방지)
            if yolo_face_results and len(yolo_face_results) > 0:
                result = yolo_face_results[0]
            
            # 디버깅: YOLO Face 전체 감지 결과 로깅
            total_boxes = 0
            has_keypoints = False
            if result is not None:
                total_boxes = len(result.boxes) if result.boxes is not None else 0
                has_keypoints = result.keypoints is not None
                logging.debug(f"🔍 [CAM-{cam_id}] YOLO Face 전체 감지: 박스={total_boxes}개, 키포인트={has_keypoints}")
            else:
                logging.debug(f"🔍 [CAM-{cam_id}] YOLO Face 결과 없음 (yolo_face_results=None 또는 빈 리스트)")
            
            # 얼굴 감지 수 기록 (동적 confidence 조정용) - 결과가 없어도 0 기록
            if not hasattr(process_single_frame, '_face_detection_history'):
                process_single_frame._face_detection_history = {}
            if cam_id not in process_single_frame._face_detection_history:
                process_single_frame._face_detection_history[cam_id] = []
            process_single_frame._face_detection_history[cam_id].append(total_boxes)
            # 최근 30프레임만 유지 (메모리 최적화)
            if len(process_single_frame._face_detection_history[cam_id]) > 30:
                process_single_frame._face_detection_history[cam_id] = process_single_frame._face_detection_history[cam_id][-30:]
            
            if result is not None and result.boxes is not None and len(result.boxes) > 0:
                # Keypoints 전체 추출 (있으면)
                all_keypoints = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        all_keypoints = result.keypoints.xy.cpu().numpy()
                    except Exception as e:
                        logging.debug(f"Keypoints 전체 변환 실패: {e}")

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
                        
                        # YOLO keypoints 추출 (측면 얼굴 지원: 일부 키포인트만 있어도 처리)
                        kps = None
                        if all_keypoints is not None and len(all_keypoints) > i:
                            try:
                                kps = all_keypoints[i].copy() # (5, 2) 또는 일부만 있을 수 있음
                                # 스케일링 적용
                                kps[:, 0] *= w_scale
                                kps[:, 1] *= h_scale
                                
                                # 측면 얼굴 지원: 키포인트가 2개 이상이면 사용 (5개 미만도 허용)
                                if len(kps) < 2:
                                    logging.debug(f"키포인트가 2개 미만입니다: {len(kps)}개 (얼굴 박스만 사용)")
                                    kps = None  # 키포인트가 너무 적으면 None으로 설정 (얼굴 박스 기반 정렬 사용)
                            except Exception as e:
                                logging.debug(f"개별 Keypoints 처리 실패: {e}")
                                kps = None
                        
                        # 키포인트가 없어도 얼굴 박스만으로 처리 가능 (측면 얼굴 지원)
                        # kps가 None이어도 얼굴 인식 시도 (얼굴 박스 기반 정렬 사용)

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
                        # 얼굴 크기의 3.5배로 확장 (상체 포함)
                        expanded_w = face_w * 3.5
                        expanded_h = face_h * 3.5
                        # 확장된 박스 (얼굴이 상단 중앙에 위치)
                        expanded_x1 = max(0, int(face_cx - expanded_w / 2))
                        expanded_y1 = max(0, int(face_cy - face_h * 0.3))  # 얼굴이 상단에
                        expanded_x2 = min(orig_w, int(face_cx + expanded_w / 2))
                        expanded_y2 = min(orig_h, int(face_cy + expanded_h * 0.7))  # 하체 포함
                        
                        # 유효한 박스인지 확인
                        if expanded_x2 > expanded_x1 and expanded_y2 > expanded_y1:
                            face_detected_boxes.append({
                                'box': (expanded_x1, expanded_y1, expanded_x2, expanded_y2),
                                'face_bbox': (fx1, fy1, fx2, fy2),
                                'face': face_obj,
                                'confidence': conf
                            })
                            logging.debug(f"얼굴 기반 박스 감지: ({expanded_x1}, {expanded_y1}, {expanded_x2}, {expanded_y2}), 얼굴 크기: {face_w:.1f}x{face_h:.1f}")
            
            perf_timings['parse_results'] = (time.time() - parse_start) * 1000  # ms

            # 7. 사람 감지 및 상태 확인
            if pose_results and pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                logging.info(f"[CAM-{cam_id}] YOLO Pose 모델 결과: {len(boxes)}명 감지")

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
                            # config에서 IoU 임계값 가져오기 (기본값 0.5로 더 적극적인 중복 제거)
                            iou_threshold = config.Thresholds.IOU_PERSON_DEDUP
                            if iou > iou_threshold:  # 높은 겹침은 중복으로 간주
                                suppressed[j] = True
                    
                    boxes = boxes[keep_indices]
                    if pose_results[0].keypoints is not None:
                        keypoints_list = [pose_results[0].keypoints[i] for i in keep_indices]
            except Exception:
                pass
            
            # pose_results가 비어있는지 먼저 확인
            if pose_results and len(pose_results) > 0:
                keypoints_list = pose_results[0].keypoints if pose_results[0].keypoints else None
                confidences = pose_results[0].boxes.conf.cpu().numpy() if pose_results[0].boxes.conf is not None else None
                tracker_ids = pose_results[0].boxes.id.cpu().numpy() if pose_results[0].boxes.id is not None else None
            else:
                keypoints_list = None
                confidences = None
                tracker_ids = None
                boxes = np.array([])  # 빈 박스 배열

            # 필터링된 인덱스를 저장할 리스트
            valid_indices = []
            
            for i, box in enumerate(boxes):
                scaled_box_np = box * np.array([w_scale, h_scale, w_scale, h_scale])
                clipped_box = utils.clip_bbox_xyxy(scaled_box_np, orig_w, orig_h)
                if clipped_box is None:
                    continue

                # 키포인트 기반 필터링
                if keypoints_list is not None and len(keypoints_list) > i:
                    kpts = keypoints_list[i]
                    if kpts is not None and kpts.conf is not None:
                        visible_kpts_count = torch.sum(kpts.conf > config.Thresholds.POSE_CONFIDENCE).item()
                        if visible_kpts_count < config.Thresholds.MIN_VISIBLE_KEYPOINTS:
                            continue
                else:
                    continue
                
                # 모든 필터링을 통과한 경우, 인덱스를 저장
                valid_indices.append(i)
            
            logging.info(f"[CAM-{cam_id}] 필터링 후 유효한 사람 수: {len(valid_indices)}")

            # 사람 박스 좌표를 원본 프레임 크기로 스케일링 및 필터링
            scaled_person_boxes = []
            valid_person_indices = []  # 유효한 사람 박스 인덱스
            filtered_boxes = []
            filtered_keypoints = []
            filtered_confidences = []
            filtered_tracker_ids = []

            for i, box in enumerate(boxes):
                scaled_box_np = box * np.array([w_scale, h_scale, w_scale, h_scale])
                clipped_box = utils.clip_bbox_xyxy(scaled_box_np, orig_w, orig_h)
                if clipped_box is None:
                    continue
                x1, y1, x2, y2 = map(int, clipped_box) # 정수형으로 변환

                original_box = clipped_box  # 원본 박스 저장
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                aspect_ratio = box_w / box_h if box_h > 0 else 0

                # 1. 키포인트 확인 및 박스 조정 (키포인트 기반으로 박스를 더 정확하게 조정)
                # 멀리 있는 사람도 감지하기 위해 완화 조건 완화
                num_valid_kpts = 0
                has_head_or_shoulders = False
                refined_box = None
                if keypoints_list is not None and i < len(keypoints_list):
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
                        if num_valid_kpts >= 4:  # 충분한 키포인트가 있을 때만 조정
                            refined_box = utils.refine_box_from_keypoints(
                                keypoints, original_box, orig_w, orig_h, padding_ratio=0.15
                            )
                            if refined_box is not None:
                                # 조정된 박스 사용
                                x1, y1, x2, y2 = refined_box
                                box_w = x2 - x1
                                box_h = y2 - y1
                                box_area = box_w * box_h
                                aspect_ratio = box_w / box_h if box_h > 0 else 0
                                logging.debug(f"키포인트 기반 박스 조정: {original_box} -> {refined_box}")

                # 오탐지 방지: 키포인트 검증 강화
                # 완화 조건을 더 엄격하게 적용하여 의자/책상 등 오탐지 방지
                # 최소 6개 키포인트와 머리/어깨가 있어야 완화 조건 적용
                use_relaxed = (num_valid_kpts >= 6 and has_head_or_shoulders) and (box_area < 5000)
                min_w = config.Thresholds.RELAXED_MIN_PERSON_BOX_WIDTH if use_relaxed else config.Thresholds.MIN_PERSON_BOX_WIDTH
                min_h = config.Thresholds.RELAXED_MIN_PERSON_BOX_HEIGHT if use_relaxed else config.Thresholds.MIN_PERSON_BOX_HEIGHT
                min_area = config.Thresholds.RELAXED_MIN_PERSON_BOX_AREA if use_relaxed else config.Thresholds.MIN_PERSON_BOX_AREA
                max_ar = config.Thresholds.RELAXED_MAX_PERSON_ASPECT_RATIO if use_relaxed else config.Thresholds.MAX_PERSON_ASPECT_RATIO
                min_ar = config.Thresholds.RELAXED_MIN_PERSON_ASPECT_RATIO if use_relaxed else config.Thresholds.MIN_PERSON_ASPECT_RATIO

                # 2. 최소 크기 필터링 (너무 작은 박스는 제외)
                if box_w < min_w or box_h < min_h or box_area < min_area:
                    logging.debug(f"사람 박스 필터링 (크기 작음): {box_w}x{box_h}, 면적={box_area} (relaxed={use_relaxed})")
                    continue

                # 3. 종횡비 필터링 (손처럼 세로로 긴 것 또는 너무 가로로 긴 것 제외)
                if aspect_ratio > max_ar or aspect_ratio < min_ar:
                    logging.debug(f"사람 박스 필터링 (종횡비 이상): {aspect_ratio:.2f} (relaxed={use_relaxed})")
                    continue

                # 4. 키포인트 검증 (사람이 아닌 객체 필터링 강화)
                # 사람이 아닌 객체(의자, 안전모 등)를 제외하기 위해 머리/어깨가 반드시 있어야 함
                if not has_head_or_shoulders:
                    # 머리나 어깨가 없으면 최소 6개 키포인트 필요 (사람이 아닌 객체 필터링)
                    min_kpts_required = 6
                    logging.debug(f"사람 박스 필터링 (머리/어깨 없음): {num_valid_kpts} < {min_kpts_required} (사람이 아닌 객체 의심)")
                    continue
                else:
                    # 머리/어깨가 있으면 완화된 조건 사용
                    min_kpts_required = 4 if use_relaxed else config.Thresholds.MIN_VISIBLE_KEYPOINTS
                
                if num_valid_kpts < min_kpts_required:
                    logging.debug(f"사람 박스 필터링 (키포인트 부족): {num_valid_kpts} < {min_kpts_required} (has_head={has_head_or_shoulders}, relaxed={use_relaxed})")
                    continue
                
                # 5. 추가 검증: 키포인트 분포 확인 (오탐지 방지)
                # 상체 키포인트(머리, 어깨, 팔꿈치)가 일정 비율 이상 있어야 함
                upper_body_ratio = 0.0
                keypoint_spread_ok = True
                if keypoints_list is not None and i < len(keypoints_list):
                    keypoints = keypoints_list[i]
                    if keypoints is not None and keypoints.conf is not None:
                        conf_arr = keypoints.conf[0].cpu().numpy()
                        points = keypoints.xy[0].cpu().numpy()
                        
                        # 상체 키포인트 인덱스: nose(0), eyes(1,2), ears(3,4), shoulders(5,6), elbows(7,8)
                        upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                        upper_body_valid = sum(1 for idx in upper_body_indices if idx < len(conf_arr) and conf_arr[idx] > config.Thresholds.POSE_CONFIDENCE)
                        upper_body_ratio = upper_body_valid / len(upper_body_indices) if len(upper_body_indices) > 0 else 0
                        
                        # 상체 키포인트가 30% 미만이면 오탐지 가능성 높음 (25% -> 30% 강화, 사람이 아닌 객체 필터링)
                        # 안전모, 의자 등은 상체 키포인트가 거의 없으므로 이를 강화
                        if upper_body_ratio < 0.30:
                            logging.debug(f"사람 박스 필터링 (상체 키포인트 부족): {upper_body_valid}/{len(upper_body_indices)} ({upper_body_ratio:.2%})")
                            continue
                        
                        # 키포인트 분산 확인: 한 점에 몰려있으면 오탐지 (사람이 아닌 객체 필터링 강화)
                        valid_points = points[conf_arr > config.Thresholds.POSE_CONFIDENCE]
                        if len(valid_points) >= 3:
                            kpt_x_std = np.std(valid_points[:, 0])
                            kpt_y_std = np.std(valid_points[:, 1])
                            # 분산이 너무 작으면 (표준편차 < 12px) 한 점에 몰려있음 (8px -> 12px 강화, 사람이 아닌 객체 필터링)
                            # 사람이 아닌 객체(안전모, 의자 등)는 키포인트가 한 곳에 집중되어 있음
                            min_spread = 12.0
                            if box_area > 10000:  # 큰 박스는 더 큰 분산 요구
                                min_spread = 20.0
                            elif box_area > 50000:  # 매우 큰 박스는 더 엄격
                                min_spread = 30.0
                            
                            if kpt_x_std < min_spread or kpt_y_std < min_spread:
                                logging.debug(f"사람 박스 필터링 (키포인트 분산 부족, 사람이 아닌 객체 의심): std_x={kpt_x_std:.1f}, std_y={kpt_y_std:.1f}, box_area={box_area}")
                                keypoint_spread_ok = False
                                continue
                            
                            # 추가 검증: 키포인트가 박스 전체에 분산되어 있는지 확인
                            # 박스 크기 대비 키포인트 분산 비율 확인
                            box_width = x2 - x1
                            box_height = y2 - y1
                            if box_width > 0 and box_height > 0:
                                # 키포인트 분산이 박스 크기의 일정 비율 이상이어야 함
                                spread_ratio_x = kpt_x_std / box_width
                                spread_ratio_y = kpt_y_std / box_height
                                # 박스 크기 대비 분산이 너무 작으면 (10% 미만) 사람이 아닌 객체 의심
                                if spread_ratio_x < 0.10 and spread_ratio_y < 0.10:
                                    logging.debug(f"사람 박스 필터링 (박스 대비 키포인트 분산 부족): spread_x={spread_ratio_x:.2%}, spread_y={spread_ratio_y:.2%}")
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
                                # 작은 객체가 사람 박스 면적의 20% 이상 차지하고 IOU가 0.15 이상이면 제외 (더 엄격하게)
                                # 사람이 아닌 객체(의자, 안전모 등)가 사람 박스와 겹치면 제외
                                if det_area > box_area * 0.2 and iou > 0.15:
                                    logging.debug(f"사람 박스 필터링 (작은 객체와 겹침, 사람이 아닌 객체 의심): {class_name}, IOU={iou:.2f}, det_area={det_area}, box_area={box_area}")
                                    should_filter = True
                                    break
                                # IOU가 매우 높으면(0.4 이상) 사람 박스 내부에 객체가 있는 것으로 간주하여 제외
                                if iou > 0.4:
                                    logging.debug(f"사람 박스 필터링 (높은 IOU로 인한 겹침, 사람이 아닌 객체 의심): {class_name}, IOU={iou:.2f}")
                                    should_filter = True
                                    break

                    if should_filter:
                        break

                if should_filter:
                    continue

                # 화면 가장자리에 붙은 좁은 박스 필터링 (오탐지 방지 - 왼쪽 구석 오탐지 해결용)
                # 예: x1이 0에 가깝고 너비가 50px 미만인 경우
                if (x1 < 10 or x2 > orig_w - 10) and box_w < 60:
                     logging.debug(f"가장자리 좁은 박스 제거됨 (오탐지 의심): {box_w}x{box_h} at x={x1}")
                     continue

                # 중복 박스 방지 (scaled_person_boxes에 추가하기 전 확인)
                # YOLO Pose가 동일한 사람에 대해 중복 박스를 내뱉는 경우 방지
                is_box_duplicate = False
                for existing_box in scaled_person_boxes:
                    # 기존 박스와 현재 박스의 IoU 계산
                    # existing_box는 float array일 수 있으므로 int로 변환
                    ex_x1, ex_y1, ex_x2, ex_y2 = map(int, existing_box)
                    iou = utils.calculate_iou((x1, y1, x2, y2), (ex_x1, ex_y1, ex_x2, ex_y2))
                    
                    # 70% 이상 겹치면 중복으로 간주하고 건너뜀
                    if iou > 0.7: 
                        is_box_duplicate = True
                        logging.debug(f"중복 박스 필터링됨: IoU={iou:.2f}, Box={x1},{y1},{x2},{y2}")
                        break
                
                if is_box_duplicate:
                    continue

                # 모든 필터링을 통과한 유효한 사람 박스
                scaled_person_boxes.append(scaled_box_np)
                valid_person_indices.append(i)
                filtered_boxes.append(box)
                if keypoints_list is not None and i < len(keypoints_list):
                    filtered_keypoints.append(keypoints_list[i])
                if confidences is not None:
                    filtered_confidences.append(confidences[i])
                if tracker_ids is not None:
                    filtered_tracker_ids.append(tracker_ids[i])

            # 필터링된 결과로 업데이트
            boxes = np.array(filtered_boxes) if filtered_boxes else np.array([])
            keypoints_list = filtered_keypoints if filtered_keypoints else None
            if confidences is not None:
                confidences = np.array(filtered_confidences) if filtered_confidences else np.array([])
            if tracker_ids is not None:
                tracker_ids = np.array(filtered_tracker_ids) if filtered_tracker_ids else np.array([])
            
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
                        if keypoints_list is None:
                            keypoints_list = []
                        keypoints_list.append(None)
                        logging.debug(f"얼굴 기반 박스 추가: ({fx1}, {fy1}, {fx2}, {fy2})")

            num_people = len(boxes)
            if num_people == 0:
                logging.debug("필터링 후 유효한 사람 탐지 없음")
            else:
                logging.debug(f"필터링 후 유효한 사람 수: {num_people}")

            # 얼굴 인식 시간 측정 시작
            face_recognition_start = time.time()

            # 병렬 처리를 위한 작업 목록 준비
            face_recognition_tasks = []
            futures_with_index = []  # (person_data_list_index, future)
            person_data_list = []  # 순서대로 결과를 맞추기 위한 리스트
            
            # PPE 박스 중복 매칭 방지를 위한 추적 세트 초기화 (매 프레임마다 초기화)
            frame_state = state.get_frame_processing_state(cam_id)
            if 'used_ppe_boxes' not in frame_state:
                frame_state['used_ppe_boxes'] = set()
            frame_state['used_ppe_boxes'].clear()  # 매 프레임마다 초기화

            # 얼굴 인식 우선순위 계산을 위한 임시 리스트
            face_recognition_candidates = []  # (priority_score, person_index, box, ...)

            faces_scheduled = 0
            # 수정: valid_indices 대신 scaled_person_boxes를 순회하도록 변경 (좌표 오류 수정)
            # scaled_person_boxes에는 Pose 기반 박스와 Face 기반 박스가 모두 포함되어 있음
            num_pose_boxes = len(filtered_boxes) # Pose 기반 박스 개수 (Face 기반은 그 뒤에 추가됨)
            
            for i, scaled_box in enumerate(scaled_person_boxes):
                # 변수 초기화
                # Pose 기반인지 Face 기반인지 구분
                if i < num_pose_boxes:
                    # Pose 기반 박스
                    original_idx = valid_person_indices[i] # 원본 인덱스
                    person_id_text = f"P{original_idx}"
                    # tracker_ids는 필터링된 Pose 박스들에 대한 ID 리스트임 (928라인에서 업데이트됨)
                    tracker_id = int(tracker_ids[i]) if tracker_ids is not None and len(tracker_ids) > i else None
                else:
                    # Face 기반 박스
                    person_id_text = f"F{i}"
                    tracker_id = None
                
                # scaled_box는 이미 원본 프레임 크기에 맞춰져 있음 (float -> int 변환)
                x1, y1, x2, y2 = map(int, scaled_box)
                
                person_area = max(1, (x2 - x1) * (y2 - y1))
                person_height = max(1, y2 - y1)
                height_ratio = person_height / max(1, orig_h)
                matched_face = None
                face_quality_ok = False
                has_violation_or_danger = False
                immediate_recognition = False
                opportunistic_recognition = False
                cache_skip_recognition = False # 변수 초기화 추가

                # 사람 박스 영역 추출 (스레드 안전성을 위해 복사)
                person_img_for_detection = frame[y1:y2, x1:x2].copy()
                if person_img_for_detection.size == 0:
                    continue

                person_keypoints = keypoints_list[i] if keypoints_list and len(keypoints_list) > i else None

                person_data_list.append({
                    'index': i,
                    'person_id': person_id_text,
                    'box': (x1, y1, x2, y2),
                    'img': person_img_for_detection,
                    'keypoints': person_keypoints,
                    'tracker_id': tracker_id
                })
                
                # PPE 및 위험 행동 분석
                ppe_violations, ppe_boxes = _process_ppe_detection((x1, y1, x2, y2), all_detections, frame_state['used_ppe_boxes'])
                person_data_list[-1]['ppe_violations'] = ppe_violations
                person_data_list[-1]['ppe_boxes'] = ppe_boxes
                
                # 위험 행동(넘어짐) 감지 - 키포인트가 없어도 박스 비율로 감지 시도
                is_dangerous_detected, violation_type = False, ""
                try:
                    person_box_key = _generate_person_box_key(cam_id, None, x1, y1, x2, y2)
                    # FallSafe 모델용 person_crop (이미 생성된 person_img_for_detection 사용)
                    person_crop = person_img_for_detection if person_img_for_detection.size > 0 else None
                    # FallSafe 모델 가져오기
                    fall_model = getattr(safety_system, 'fall_model', None)
                    
                    # 키포인트가 없어도 박스 비율이 높으면 (가로로 긴 박스) 넘어짐 감지 시도
                    box_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                    should_check_fall = person_keypoints is not None or box_ratio >= 1.5
                    
                    if should_check_fall:
                        is_dangerous_detected, violation_type = _process_dangerous_behavior(
                            person_keypoints, (x1, y1, x2, y2), cam_id, person_box_key,
                            person_crop=person_crop,
                            fall_model=fall_model
                        )
                        if is_dangerous_detected:
                            logging.warning(f"⚠️ [CAM-{cam_id}] 넘어짐 감지됨: box_ratio={box_ratio:.2f}, "
                                          f"키포인트={'있음' if person_keypoints else '없음'}")
                    
                    person_data_list[-1]['is_dangerous'] = is_dangerous_detected
                    person_data_list[-1]['violation_type'] = violation_type
                except Exception as e:
                    logging.debug(f"위험 행동 감지 처리 오류: {e}")
                    person_data_list[-1]['is_dangerous'] = False
                    person_data_list[-1]['violation_type'] = ""
                
                has_violation_or_danger = len(ppe_violations) > 0 or is_dangerous_detected
                
                # 얼굴 품질 및 인식 조건 계산 (완화: 일부만 보여도 인식 시도)
                if person_keypoints is not None:
                    try:
                        kpts_conf = person_keypoints.conf[0].cpu().numpy() if person_keypoints.conf is not None else None
                        if kpts_conf is not None:
                            nose_visible = kpts_conf[0] > config.Thresholds.POSE_CONFIDENCE
                            left_eye_visible = kpts_conf[1] > config.Thresholds.POSE_CONFIDENCE
                            right_eye_visible = kpts_conf[2] > config.Thresholds.POSE_CONFIDENCE
                            # 완화: 일부만 보여도 얼굴 품질 OK로 간주 (측면 얼굴 지원)
                            face_quality_ok = nose_visible or left_eye_visible or right_eye_visible
                    except Exception:
                        pass
                
                # 기회형 인식 조건 완화 (더 많은 사람 인식 시도)
                opportunistic_recognition = not has_violation_or_danger and (
                    face_quality_ok or 
                    height_ratio >= 0.06 or  # 0.10 -> 0.06 (더 멀리서도 인식)
                    person_area >= 800  # 작은 사람도 인식 시도
                )
                # 위반이 있으면 항상 즉시 인식 시도 (얼굴 인식 실패해도 위반은 감지되어야 함)
                immediate_recognition = has_violation_or_danger
                
                # --- 얼굴 인식 실행 여부 결정 ---
                priority_score = 0.0  # 기본값 초기화
                allow_face_job = False

                # 1. 기본 조건 확인 (위반이 있으면 조건 완화, 없으면 더 엄격하게)
                # 위반이 있으면 얼굴 인식 조건을 완화하여 최대한 인식 시도
                # 위반이 없으면 더 엄격한 조건으로 불필요한 인식 시도 감소
                if has_violation_or_danger:
                    # 위반이 있으면 조건 완화 (더 작은 얼굴도 인식 시도)
                    min_area_for_recognition = max(400, config.Thresholds.MIN_FACE_RECOGNITION_AREA // 2)  # 최소 400, 또는 기존값의 절반
                    min_height_ratio_for_recognition = max(0.04, float(config.Thresholds.MIN_PERSON_HEIGHT_RATIO_FOR_FACE) * 0.75)  # 최소 0.04, 또는 기존값의 75%
                else:
                    # 위반이 없으면 더 엄격한 조건 (불필요한 인식 시도 감소)
                    min_area_for_recognition = 2500  # 1200 -> 2500 (약 2배 증가)
                    min_height_ratio_for_recognition = 0.15  # 0.06 -> 0.15 (약 2.5배 증가)
                
                base_conditions_met = (
                    face_analyzer is not None and  # buffalo_l 사용
                    face_database is not None and
                    person_area >= min_area_for_recognition and
                    height_ratio >= min_height_ratio_for_recognition
                )
                
                # 2. 캐시 확인 (tracker_id가 있을 때만)
                # 위반이 있으면 캐시 스킵하지 않음 (재인식 시도)
                # 인식률 향상: 캐시 사용 시에도 유사도 점수 검증
                if tracker_id is not None and recent_identity_cache is not None:
                    cache = recent_identity_cache[cam_id]
                    if tracker_id in cache:
                        cached_entry = cache[tracker_id]
                        cached_name = cached_entry.get('name', 'Unknown')
                        cached_score = cached_entry.get('score', 0.0)
                        cache_age = time.time() - cached_entry.get('ts', 0)
                        
                        # 캐시 사용 조건: 이름이 있고, 유사도가 임계값 이상이며, 쿨다운 시간 내
                        if cached_name != "Unknown" and \
                           cached_score >= config.Thresholds.SIMILARITY and \
                           cache_age < config.Thresholds.FACE_RECOGNITION_COOLDOWN_SECONDS:
                            # 위반이 없을 때만 캐시 스킵 (위반이 있으면 재인식 시도)
                            if not has_violation_or_danger:
                                cache_skip_recognition = True
                                person_data_list[-1]['name'] = cached_name
                                person_data_list[-1]['similarity'] = cached_score
                                logging.debug(f"✅ 캐시 사용: tracker_id={tracker_id}, name={cached_name}, score={cached_score:.3f}")
                            else:
                                logging.debug(f"🔄 위반 감지로 캐시 스킵: 재인식 시도 (tracker_id={tracker_id})")

                # 3. 최종 실행 여부 결정
                # 위반이 있으면 항상 인식 시도 (조건 완화됨)
                if base_conditions_met and not cache_skip_recognition and (immediate_recognition or opportunistic_recognition):
                    allow_face_job = True
                    # 얼굴 인식 우선순위 계산
                    face_size_score = 0.0
                    if matched_face and matched_face.bbox:
                        fx1, fy1, fx2, fy2 = matched_face.bbox
                        face_area = (fx2 - fx1) * (fy2 - fy1)
                        face_size_score = min(1.0, face_area / (orig_w * orig_h * 0.1))
                    
                    front_face_score = 0.0
                    if matched_face and hasattr(matched_face, 'kps') and matched_face.kps is not None:
                        front_face_score = 1.0
                    elif face_quality_ok:
                        front_face_score = 0.7
                    
                    urgency_score = 1.0 if immediate_recognition else 0.3
                    
                    priority_score = (face_size_score * 0.4 + front_face_score * 0.3 + urgency_score * 0.3)
                
                if allow_face_job:
                    face_recognition_candidates.append((
                        priority_score,
                        len(person_data_list) - 1,
                        person_img_for_detection,
                        person_id_text,
                        tracker_id,
                        matched_face,
                        immediate_recognition,
                        face_quality_ok
                    ))
                    faces_scheduled += 1
                else:
                    # 얼굴 인식 스킵 이유 로깅
                    if has_violation_or_danger or face_quality_ok or height_ratio >= 0.06 or person_area >= 800:
                        skip_reasons = []
                        if tracker_id is None: skip_reasons.append("추적 ID 없음")
                        if face_analyzer is None: skip_reasons.append("buffalo_l 모델 없음")
                        if face_database is None: skip_reasons.append("FAISS DB 없음")
                        if not (immediate_recognition or opportunistic_recognition): skip_reasons.append("인식 조건 불만족")
                        if not (person_area >= config.Thresholds.MIN_FACE_RECOGNITION_AREA): skip_reasons.append(f"영역 부족 (area={person_area}, 최소={config.Thresholds.MIN_FACE_RECOGNITION_AREA})")
                        if not (height_ratio >= float(config.Thresholds.MIN_PERSON_HEIGHT_RATIO_FOR_FACE)): skip_reasons.append(f"거리 제한 (키 비율 {height_ratio:.2f}, 최소={config.Thresholds.MIN_PERSON_HEIGHT_RATIO_FOR_FACE})")
                        if cache_skip_recognition: skip_reasons.append("캐시에서 이름 발견 (재인식 스킵)")
                        
                        logging.debug(f"⚠️ 얼굴 인식 스킵: person_id={person_id_text}, tracker_id={tracker_id}, 이유={', '.join(skip_reasons)}")
                
                # 얼굴 탐지 프레임 카운터 업데이트 (얼굴 인식 실행 시)
                with face_detection_lock:
                    current_frame = frame_stats.get(cam_id, {}).get('frame_count', 0)
                    last_face_detection_frame[cam_id] = current_frame

            # 얼굴 인식 우선순위 기반 작업 제출 (프레임 드랍 방지: 동적 제한)
            if len(face_recognition_candidates) > 0:
                # 우선순위 점수 기준으로 정렬 (높은 점수 우선)
                face_recognition_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # 산업 현장 대응: 병렬 처리 최적화
                # 1. 기본 제한: config에서 설정된 최대 작업 수 (산업 현장 대응: 기본값 증가)
                max_jobs_base = config.Thresholds.MAX_FACE_RECOGNITION_JOBS_PER_FRAME
                
                # 2. 워커 수 기반 동적 조정: 워커가 많아도 작업 수 제한 (우선순위 높은 작업만)
                num_workers_available = face_recognition_executor._max_workers
                # 워커 수의 0.5배까지 처리 가능 (작업 수 제한 강화)
                max_jobs_by_workers = max(max_jobs_base, int(num_workers_available * 0.5))
                
                # 3. FPS 기반 동적 조정: 30 FPS 목표로 작업 수 제한 (프레임 드랍 방지)
                max_jobs_dynamic = max_jobs_by_workers
                try:
                    with frame_stats_lock:
                        cam_stats = frame_stats.get(cam_id, {})
                        recent_frames = cam_stats.get('recent_frame_times', [])
                        if len(recent_frames) >= 2:
                            time_span = recent_frames[-1] - recent_frames[0]
                            if time_span > 0:
                                current_fps = (len(recent_frames) - 1) / time_span
                                # 30 FPS 목표: FPS가 낮으면 작업 수 감소
                                if current_fps < 20:
                                    max_jobs_dynamic = max(1, int(max_jobs_by_workers * 0.5))  # 50% 감소
                                elif current_fps < 25:
                                    max_jobs_dynamic = max(1, int(max_jobs_by_workers * 0.7))  # 30% 감소
                                elif current_fps >= 30:
                                    max_jobs_dynamic = max_jobs_by_workers  # 최대 작업 수 유지
                            else:
                                current_fps = (len(recent_frames) - 1) / time_span
                                # FPS가 낮을 때만 제한 (산업 현장 대응: 더 관대한 기준)
                                if current_fps < 10:
                                    max_jobs_dynamic = max(3, max_jobs_by_workers // 2)  # 최소 3개
                                elif current_fps < 15:
                                    max_jobs_dynamic = max(5, int(max_jobs_by_workers * 0.7))
                                elif current_fps < 20:
                                    max_jobs_dynamic = max(7, int(max_jobs_by_workers * 0.85))
                                # FPS >= 20이면 워커 기반 제한 사용 (제한 없음)
                except Exception:
                    pass  # FPS 계산 실패 시 워커 기반 제한 사용
                
                # 4. 후보 수가 매우 많을 때만 제한 (산업 현장 대응: 더 많은 사람 처리)
                if len(face_recognition_candidates) > max_jobs_dynamic * 3:
                    # 후보가 워커 기반 제한의 3배 이상이면 약간 감소
                    max_jobs_dynamic = max(max_jobs_base, max_jobs_dynamic - 2)
                
                # 5. 최대 인원 제한: FPS 기반 동적 조정 (프레임 드랍 방지)
                # FPS가 높으면 더 많은 사람 처리 가능, 낮으면 제한
                try:
                    with frame_stats_lock:
                        cam_stats = frame_stats.get(cam_id, {})
                        recent_frames = cam_stats.get('recent_frame_times', [])
                        if len(recent_frames) >= 2:
                            time_span = recent_frames[-1] - recent_frames[0]
                            if time_span > 0:
                                current_fps = (len(recent_frames) - 1) / time_span
                                # FPS 기반 최대 인원 제한
                                if current_fps >= 30:
                                    MAX_PEOPLE_LIMIT = 8  # 높은 FPS: 최대 8명
                                elif current_fps >= 20:
                                    MAX_PEOPLE_LIMIT = 6  # 중간 FPS: 최대 6명
                                elif current_fps >= 15:
                                    MAX_PEOPLE_LIMIT = 5  # 낮은 FPS: 최대 5명
                                else:
                                    MAX_PEOPLE_LIMIT = 3  # 매우 낮은 FPS: 최대 3명
                            else:
                                MAX_PEOPLE_LIMIT = 5  # 기본값
                        else:
                            MAX_PEOPLE_LIMIT = 5  # 기본값
                except Exception:
                    MAX_PEOPLE_LIMIT = 5  # 예외 시 기본값
                
                max_jobs_dynamic = min(max_jobs_dynamic, MAX_PEOPLE_LIMIT)
                
                # 최종 제한 적용
                limited_candidates = face_recognition_candidates[:max_jobs_dynamic]
                
                if len(face_recognition_candidates) > max_jobs_dynamic:
                    skipped_count = len(face_recognition_candidates) - max_jobs_dynamic
                    logging.info(f"[CAM-{cam_id}] 얼굴 인식 작업 제한: {len(face_recognition_candidates)}개 후보 중 {max_jobs_dynamic}개만 처리 (최대 5명 제한, 프레임 드랍 방지), {skipped_count}개 스킵 (우선순위 낮은 작업)")
                
                # FastIndustrialRecognizer 가져오기 (랜드마크 기반 고속 처리용)
                fast_recognizer = getattr(safety_system, 'fast_recognizer', None)
                use_adaface = getattr(safety_system, 'use_adaface', False)
                adaface_model_path = getattr(safety_system, 'adaface_model_path', None)
                face_uses_trt = getattr(safety_system, 'face_uses_trt', False)
                
                logging.debug(f"얼굴 인식 우선순위 정렬: {len(limited_candidates)}개 작업 제출, 최고 우선순위={limited_candidates[0][0]:.3f}")
                
                # 배치 처리 최적화: Fast Path 얼굴들을 모아서 배치 처리
                fast_path_candidates = []  # (person_idx, tracker_id, matched_face, original_frame)
                fallback_candidates = []   # (priority_score, person_idx, person_img, person_id, tracker_id, matched_face, immediate, face_quality)
                
                for priority_score, person_idx, person_img, person_id, tracker_id, matched_face, immediate, face_quality in limited_candidates:
                    # Fast Path: 미리 감지된 얼굴이 있고 키포인트가 있으면 배치 처리 대상
                    has_fast_path = matched_face is not None and hasattr(matched_face, 'kps') and matched_face.kps is not None
                    if has_fast_path and fast_recognizer is not None:
                        fast_path_candidates.append((person_idx, tracker_id, matched_face, resized_frame))
                    else:
                        fallback_candidates.append((priority_score, person_idx, person_img, person_id, tracker_id, matched_face, immediate, face_quality))
                
                # Fast Path 배치 처리 (GPU 활용률 향상)
                if len(fast_path_candidates) > 0 and fast_recognizer is not None:
                    try:
                        # 배치 처리: 여러 얼굴을 한 번에 처리
                        batch_frames = []
                        batch_kps = []
                        batch_indices = []  # (person_idx, tracker_id, matched_face)
                        
                        for person_idx, tracker_id, matched_face, original_frame in fast_path_candidates:
                            if matched_face.kps is not None:
                                batch_frames.append(original_frame)
                                batch_kps.append(matched_face.kps)
                                batch_indices.append((person_idx, tracker_id, matched_face))
                        
                        if len(batch_frames) > 0:
                            # 배치 임베딩 추출 (데이터베이스 구축과 동일한 전처리 적용)
                            batch_results = fast_recognizer.get_embeddings_batch(
                                batch_frames, 
                                batch_kps,
                                use_enhanced_preprocessing=False,  # aivis-project1 방식: 기본 전처리만 사용 (CLAHE)
                                use_tta=False  # 데이터베이스 구축 시와 동일 (USE_TTA_FOR_DATABASE=False)
                            )
                            
                            # 결과 처리
                            for (person_idx, tracker_id, matched_face), (embedding, aligned_face) in zip(batch_indices, batch_results):
                                if embedding is not None:
                                    face_bbox = tuple(map(int, matched_face.bbox)) if hasattr(matched_face, 'bbox') else None
                                    # 배치 FAISS 검색을 위해 임베딩 저장
                                    embeddings_for_batch.append((person_idx, tracker_id, embedding, face_bbox))
                                    logging.debug(f"[CAM-{cam_id}] ✅ Fast Path 배치 처리 완료: person_idx={person_idx}, tracker_id={tracker_id}")
                                else:
                                    # 배치 처리 실패 시 Fallback으로 전환
                                    fallback_candidates.append((0.5, person_idx, None, f"P{person_idx}", tracker_id, matched_face, True, True))
                    except Exception as e:
                        logging.warning(f"⚠️ Fast Path 배치 처리 실패, 개별 처리로 폴백: {e}")
                        # 배치 처리 실패 시 모든 Fast Path 후보를 Fallback으로 전환
                        for person_idx, tracker_id, matched_face, _ in fast_path_candidates:
                            fallback_candidates.append((0.5, person_idx, None, f"P{person_idx}", tracker_id, matched_face, True, True))
                
                # Fallback: 개별 작업 제출 (YOLO로 다시 감지하는 경우)
                for priority_score, person_idx, person_img, person_id, tracker_id, matched_face, immediate, face_quality in fallback_candidates:
                    recognition_type = "즉시" if immediate else "기회형"
                    logging.debug(f"[CAM-{cam_id}] 얼굴 인식 작업 제출 (우선순위={priority_score:.3f}, {recognition_type}): person_idx={person_idx}, person_id={person_id}, 얼굴품질={face_quality}, FastPath=False")
                    
                    future = face_recognition_executor.submit(
                        _process_face_recognition,
                        person_img.copy() if person_img is not None else None,  # 스레드 안전성을 위해 복사 필요
                        person_id,
                        face_model,
                        face_database,
                        fast_recognizer,  # AdaFace용 (실제 사용)
                        matched_face,  # 미리 감지된 얼굴 (Fast Path)
                        resized_frame  # 원본(리사이즈된) 프레임
                    )
                    
                    face_recognition_tasks.append(future)
                    # tracker_id를 함께 저장하여 나중에 결과를 매핑
                    futures_with_index.append((person_idx, tracker_id, future))
                
                # 쿨다운 타임스탬프는 작업 완료 후 업데이트 (아래에서 처리)

            # 병렬로 얼굴 인식 결과 수집 (인원수 제한 해제)
            # GPU 사용 시 모든 사람 처리 가능
            # 제한 없이 모든 얼굴 인식 작업 처리
            
            face_recognition_results = {}
            # 배치 FAISS 검색을 위한 임베딩 수집
            embeddings_for_batch = []  # (person_idx, tracker_id, embedding, face_bbox)
            
            # 비동기 처리: 프레임 드랍 방지를 위한 타임아웃 최적화
            # GPU 사용 시 더 긴 타임아웃 허용 (배치 처리로 더 많은 작업 완료 가능)
            num_workers = face_recognition_executor._max_workers
            
            # GPU 사용 여부 확인 (얼굴 인식 모델이 GPU를 사용하는지)
            is_gpu_available = False
            try:
                if torch.cuda.is_available():
                    # 얼굴 인식 모델이 GPU를 사용하는지 확인
                    if hasattr(safety_system, 'device_face') and safety_system.device_face.type == 'cuda':
                        is_gpu_available = True
                    elif hasattr(safety_system, 'fast_recognizer') and hasattr(safety_system.fast_recognizer, 'session'):
                        # ONNX Runtime 세션의 Provider 확인
                        session = safety_system.fast_recognizer.session
                        if session and 'CUDAExecutionProvider' in session.get_providers():
                            is_gpu_available = True
            except:
                pass
            
            # 30 FPS 목표: 실제 처리 시간 고려하여 타임아웃 증가 (타임아웃으로 인한 작업 실패 방지)
            # 실제 처리 시간: YOLO Face 감지 + 임베딩 추출 = 약 50-100ms (로그 확인: 1046ms까지 소요)
            # 타임아웃을 150-200ms로 증가하여 작업 완료율 향상
            if is_gpu_available:
                base_timeout = 0.15  # GPU 사용 시 기본 타임아웃 150ms (50ms -> 150ms, 작업 완료율 향상)
                max_timeout = 0.20   # GPU 사용 시 최대 타임아웃 200ms (80ms -> 200ms)
            else:
                base_timeout = 0.15  # CPU 사용 시 기본 타임아웃 150ms (50ms -> 150ms, 작업 완료율 향상)
                max_timeout = 0.20   # CPU 사용 시 최대 타임아웃 200ms (80ms -> 200ms)
            
            # 작업 수 기반 타임아웃 조정 (30 FPS 목표, 실제 처리 시간 고려)
            if len(futures_with_index) > 0:
                if is_gpu_available:
                    # GPU 사용 시: 작업 수에 따라 타임아웃 조정
                    if len(futures_with_index) <= 2:
                        timeout_seconds = 0.15  # 150ms (50ms -> 150ms, 작업 완료율 향상)
                    elif len(futures_with_index) <= 4:
                        timeout_seconds = 0.18  # 180ms (50ms -> 180ms, 작업 완료율 향상)
                    else:
                        timeout_seconds = 0.20  # 200ms (50ms -> 200ms, 작업 완료율 향상)
                else:
                    # CPU 사용 시: 작업 수에 따라 타임아웃 조정
                    if len(futures_with_index) <= 2:
                        timeout_seconds = 0.15  # 150ms (50ms -> 150ms, 작업 완료율 향상)
                    elif len(futures_with_index) <= 4:
                        timeout_seconds = 0.18  # 180ms (50ms -> 180ms, 작업 완료율 향상)
                    else:
                        timeout_seconds = 0.20  # 200ms (50ms -> 200ms, 작업 완료율 향상)
            else:
                timeout_seconds = base_timeout
            
            # 우선순위 기반 타임아웃 조정 (높은 우선순위 작업이 많으면 약간 증가)
            if len(face_recognition_candidates) > 0:
                # 제출된 작업만 고려 (limited_candidates는 스코프 밖이므로 futures_with_index 기준)
                submitted_candidates = [c for c in face_recognition_candidates if any(c[1] == idx for idx, _, _ in futures_with_index)]
                if len(submitted_candidates) > 0:
                    avg_priority = sum(c[0] for c in submitted_candidates) / len(submitted_candidates)
                    # 평균 우선순위가 매우 높으면(>0.8) 타임아웃 약간 증가 (최대 200ms)
                    if avg_priority > 0.8 and len(futures_with_index) <= 3:
                        timeout_seconds = min(max_timeout, timeout_seconds + 0.05)
                else:
                    avg_priority = 0.0
            else:
                avg_priority = 0.0
            
            logging.debug(f"얼굴 인식 비동기 처리: 타임아웃={timeout_seconds:.3f}s (프레임 드랍 방지 최적화), 작업 수={len(futures_with_index)}, 워커 수={num_workers}, 평균 우선순위={avg_priority:.3f}")
            
            try:
                
                # 타임아웃 내 완료된 작업만 수집 (프레임 드랍 방지)
                # 배치 FAISS 검색을 위해 모든 임베딩을 모아서 처리
                completed_count = 0
                completed_futures = set()
                fallback_embeddings = []  # Fallback 개별 처리 결과 임베딩 수집
                try:
                    for future in as_completed([f for _, _, f in futures_with_index], timeout=timeout_seconds):
                        completed_futures.add(future)
                        try:
                            # 완료된 작업 결과 수집 (타임아웃 없음: 이미 완료됨)
                            _, _, embedding, face_bbox = future.result(timeout=0.1)
                            
                            # 매핑된 인덱스 찾기
                            mapped_idx, tracker_id = next(((idx, tid) for idx, tid, f in futures_with_index if f is future), (None, None))
                            if mapped_idx is not None and embedding is not None:
                                # Fast Path 배치 처리 결과와 Fallback 개별 처리 결과 모두 수집
                                embeddings_for_batch.append((mapped_idx, tracker_id, embedding, face_bbox))
                                fallback_embeddings.append((mapped_idx, tracker_id, embedding, face_bbox))
                            else:
                                # 임베딩이 없으면 Unknown으로 설정
                                person_data_list[mapped_idx]['name'] = "Unknown"
                                person_data_list[mapped_idx]['similarity'] = 0.0
                                person_data_list[mapped_idx]['embedding'] = None
                                person_data_list[mapped_idx]['face_bbox'] = face_bbox
                            completed_count += 1
                        except FaceRecognitionError as e:
                            # FaceRecognitionError는 로깅만 하고 계속 진행 (Unknown 유지)
                            # 얼굴 인식 실패해도 person_box는 있으므로 face_bbox는 None으로 설정
                            logging.debug(f"얼굴 인식 오류 (무시): {e.message} (error_code={e.error_code})")
                            # 매핑된 인덱스 찾기
                            mapped_idx, tracker_id = next(((idx, tid) for idx, tid, f in futures_with_index if f is future), (None, None))
                            if mapped_idx is not None and mapped_idx < len(person_data_list):
                                # 얼굴 인식 실패해도 person_box는 있으므로 Unknown으로 설정
                                person_data_list[mapped_idx]['name'] = "Unknown"
                                person_data_list[mapped_idx]['similarity'] = 0.0
                                person_data_list[mapped_idx]['embedding'] = None
                                person_data_list[mapped_idx]['face_bbox'] = None  # 얼굴 인식 실패 시 None
                            completed_count += 1
                        except Exception as e:
                            logging.debug(f"얼굴 인식 작업 실패: {e}")
                            # 매핑된 인덱스 찾기
                            mapped_idx, tracker_id = next(((idx, tid) for idx, tid, f in futures_with_index if f is future), (None, None))
                            if mapped_idx is not None and mapped_idx < len(person_data_list):
                                # 얼굴 인식 실패해도 person_box는 있으므로 Unknown으로 설정
                                person_data_list[mapped_idx]['name'] = "Unknown"
                                person_data_list[mapped_idx]['similarity'] = 0.0
                                person_data_list[mapped_idx]['embedding'] = None
                                person_data_list[mapped_idx]['face_bbox'] = None  # 얼굴 인식 실패 시 None
                            completed_count += 1
                except FuturesTimeoutError:
                    # 타임아웃 발생: 완료된 작업만 사용하고 나머지는 캐시 활용
                    logging.debug(f"얼굴 인식 타임아웃: {len(completed_futures)}/{len(futures_with_index)}개 완료, 나머지는 캐시 사용")
                
                # 미완료 작업 처리: 캐시에서 결과 찾기 또는 Unknown 유지
                for person_idx, tracker_id, future in futures_with_index:
                    if future not in completed_futures:
                        # 미완료 작업: 캐시에서 결과 찾기 시도
                        if person_idx < len(person_data_list):
                            person_data = person_data_list[person_idx]
                            # 이미 캐시에서 이름을 찾았는지 확인
                            cached_name = person_data.get('name', 'Unknown')
                            if cached_name == "Unknown":
                                # 캐시에서 추가로 찾기 시도 (tracker_id 기반)
                                try:
                                    # tracker_id가 있으면 직접 접근
                                    if tracker_id is not None:
                                        cache = recent_identity_cache.get(cam_id)
                                        if cache is not None:
                                            cached_entry = cache.get(tracker_id)
                                            if cached_entry is not None:
                                                cached_name = cached_entry.get('name', 'Unknown')
                                                cached_score = cached_entry.get('score', 0.0)
                                                if cached_name != "Unknown" and cached_score >= config.Thresholds.SIMILARITY:
                                                    person_data_list[person_idx]['name'] = cached_name
                                                    person_data_list[person_idx]['similarity'] = cached_score
                                                    logging.debug(f"미완료 작업 캐시 사용 (tracker_id): person_idx={person_idx}, tracker_id={tracker_id}, 이름={cached_name}")
                                except Exception as cache_error:
                                    logging.debug(f"캐시 검색 실패: {cache_error}")
                            
                            # 캐시에서도 찾지 못하면 Unknown 유지 (백그라운드 작업은 계속 실행)
                            if person_data_list[person_idx].get('name', 'Unknown') == "Unknown":
                                logging.debug(f"얼굴 인식 미완료: person_idx={person_idx}, 백그라운드에서 계속 실행 중 (현재 프레임은 Unknown)")
                
                # 배치 FAISS 검색 수행 (성능 최적화)
                if len(embeddings_for_batch) > 0 and face_database is not None:
                    try:
                        embeddings_array = np.array([emb for _, _, emb, _ in embeddings_for_batch], dtype=np.float32)
                        logging.info(f"[CAM-{cam_id}] 🔍 배치 FAISS 검색 시작: {len(embeddings_for_batch)}개 임베딩, 임계값={config.Thresholds.SIMILARITY}")
                        
                        batch_results = find_best_matches_faiss_batch(
                            embeddings_array,
                            face_database,
                            config.Thresholds.SIMILARITY
                        )
                        
                        for (mapped_idx, tracker_id, embedding, face_bbox), (person_name, similarity_score) in zip(embeddings_for_batch, batch_results):
                            if mapped_idx < len(person_data_list):
                                person_data_list[mapped_idx]['name'] = person_name
                                person_data_list[mapped_idx]['similarity'] = similarity_score
                                
                                # FAISS 검색 결과 로깅
                                if person_name != "Unknown":
                                    logging.info(f"[CAM-{cam_id}] ✅ FAISS 매칭 성공: person_idx={mapped_idx}, 이름={person_name}, 유사도={similarity_score:.3f}, 임계값={config.Thresholds.SIMILARITY}")
                                else:
                                    logging.warning(f"[CAM-{cam_id}] ⚠️ FAISS 매칭 실패: person_idx={mapped_idx}, 유사도={similarity_score:.3f} < 임계값={config.Thresholds.SIMILARITY} (차이: {config.Thresholds.SIMILARITY - similarity_score:.3f})")
                                
                                # 캐시 업데이트 (tracker_id 기준)
                                if tracker_id is not None:
                                    recent_identity_cache[cam_id][tracker_id] = {
                                        'name': person_name,
                                        'score': similarity_score,
                                        'ts': time.time()
                                    }
                                    logging.debug(f"[CAM-{cam_id}] 캐시 업데이트: tracker_id={tracker_id}, name={person_name}")
                    except Exception as e:
                        logging.error(f"❌ 배치 FAISS 검색 실패: {e}", exc_info=True)
                        
                        # 검색 실패해도 쿨다운 업데이트 (재시도 방지)
                        if len(embeddings_for_batch) > 0:
                            face_recognition_cooldown_ts[cam_id] = time.time()
                        # 배치 검색 실패 시 개별 검색으로 폴백 (기존 로직 유지)
                        for mapped_idx, tracker_id, embedding, face_bbox in embeddings_for_batch:
                            if mapped_idx < len(person_data_list):
                                try:
                                    person_name, similarity_score = utils.find_best_match_faiss(
                                        embedding, face_database, config.Thresholds.SIMILARITY
                                    )
                                    person_data_list[mapped_idx]['name'] = person_name
                                    person_data_list[mapped_idx]['similarity'] = similarity_score
                                    person_data_list[mapped_idx]['embedding'] = embedding
                                    person_data_list[mapped_idx]['face_bbox'] = face_bbox
                                except Exception as fallback_error:
                                    logging.error(f"❌ 개별 FAISS 검색 폴백 실패: {fallback_error}")
                                    person_data_list[mapped_idx]['name'] = "Unknown"
                                    person_data_list[mapped_idx]['similarity'] = 0.0
                                    person_data_list[mapped_idx]['embedding'] = embedding
                                    person_data_list[mapped_idx]['face_bbox'] = face_bbox
                        
                        # 개별 검색 폴백 완료 후에도 쿨다운 업데이트
                        if len(embeddings_for_batch) > 0:
                            face_recognition_cooldown_ts[cam_id] = time.time()
                            logging.debug(f"[CAM-{cam_id}] 얼굴 인식 쿨다운 업데이트: 개별 검색 폴백 완료 ({len(embeddings_for_batch)}개)")
            except Exception as e:
                # 예외 발생 시 처리: 완료된 작업만 사용하고 나머지는 캐시 활용
                logging.debug(f"얼굴 인식 처리 중 예외: {e}")
                # 예외가 발생해도 완료된 작업은 이미 처리되었으므로 계속 진행
            
            # 얼굴 인식 시간 측정 종료 (타임아웃 시간만 측정: 프레임 드랍 방지)
            face_recognition_elapsed = (time.time() - face_recognition_start) * 1000  # ms
            # 프레임 드랍 방지: 타임아웃 시간만 측정 (실제 처리 시간은 백그라운드에서 계속 실행)
            perf_timings['face_recognition'] = min(face_recognition_elapsed, timeout_seconds * 1000)
            
            # 얼굴 인식 상세 로깅 (병목 분석용)
            if face_recognition_elapsed > 1000:  # 1초 이상 걸리면 경고
                logging.warning(f"[PERF CAM-{cam_id}] ⚠️ 얼굴 인식 시간 초과: {face_recognition_elapsed:.1f}ms (제출 작업 수: {len(futures_with_index)}, 완료: {completed_count}, 타임아웃: {timeout_seconds:.3f}s)")

            # 결과를 순서대로 처리
            # 프레임 내 동일 이름 중복 방지: 이름별로 박스와 similarity 저장
            name_to_boxes: Dict[str, List[Tuple[Tuple[int,int,int,int], float, int]]] = {}  # (box, score, person_index)
            person_final_names: Dict[int, str] = {}  # person_index -> 최종 이름
            
            # 1단계: 모든 person_data를 순회하여 name_to_boxes 수집 (원본 이름 사용)
            # 최적화: 캐시 검색을 배치로 처리하여 중첩 루프 제거
            hold_sec = config.Thresholds.RECOGNITION_HOLD_SECONDS
            up_th = config.Thresholds.SIMILARITY
            down_th = max(0.0, up_th - config.Thresholds.RECOGNITION_HYSTERESIS_DELTA)
            now_ts = time.time()
            
            # TTLCache에서 모든 항목 가져오기 (tracker_id 기반)
            cache = recent_identity_cache.get(cam_id)
            cache_entries = []
            if cache is not None:
                # TTLCache는 딕셔너리처럼 사용 가능, 모든 항목 순회
                for tracker_id_key, entry in cache.items():
                    # 만료된 항목은 자동으로 제거되므로 여기서는 유효한 항목만 처리
                    entry_ts = entry.get('ts', 0)
                    age = now_ts - entry_ts
                    if age <= hold_sec:
                        # tracker_id와 함께 저장
                        entry_with_tracker = entry.copy()
                        entry_with_tracker['tracker_id'] = tracker_id_key
                        cache_entries.append(entry_with_tracker)
            
            # 배치 IoU 계산을 위한 박스 배열 준비 (tracker_id 기반으로 직접 매칭)
            if len(cache_entries) > 0 and len(person_data_list) > 0:
                # 캐시 박스 배열 준비
                cache_boxes = []
                valid_cache_indices = []
                for idx, entry in enumerate(cache_entries):
                    entry_box = entry.get('box', (0,0,0,0))
                    if len(entry_box) == 4:
                        cache_boxes.append(entry_box)
                        valid_cache_indices.append(idx)
                
                if len(cache_boxes) > 0:
                    cache_boxes_array = np.array(cache_boxes, dtype=np.float32)
                    
                    # person_data 박스 배열 준비
                    person_boxes = []
                    person_indices = []
                    for person_data in person_data_list:
                        person_boxes.append(person_data['box'])
                        person_indices.append(person_data['index'])
                    
                    person_boxes_array = np.array(person_boxes, dtype=np.float32)
                    
                    # 배치 IoU 계산
                    iou_matrix = calculate_iou_batch(person_boxes_array, cache_boxes_array)
                    
                    # 각 person_data에 대해 최적의 캐시 항목 찾기
                    matched_entries_map = {}  # person_index -> matched_entry
                    for p_idx, person_idx in enumerate(person_indices):
                        person_box = person_boxes[p_idx]
                        x1, y1, x2, y2 = person_box
                        current_box_center_x = (x1 + x2) / 2
                        current_box_center_y = (y1 + y2) / 2
                        current_box_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        max_distance = current_box_diagonal * 1.5
                        
                        best_iou = 0.0
                        best_cache_idx = None
                        
                        # IoU 행렬에서 최적 항목 찾기 (거리 필터링도 적용)
                        for c_idx, cache_idx in enumerate(valid_cache_indices):
                            iou = float(iou_matrix[p_idx, c_idx])
                            if iou < 0.5:  # IoU 임계값
                                continue
                            
                            entry_box = cache_boxes[c_idx]
                            entry_center_x = (entry_box[0] + entry_box[2]) / 2
                            entry_center_y = (entry_box[1] + entry_box[3]) / 2
                            center_distance = ((current_box_center_x - entry_center_x) ** 2 + 
                                             (current_box_center_y - entry_center_y) ** 2) ** 0.5
                            
                            if center_distance <= max_distance and iou > best_iou:
                                best_iou = iou
                                best_cache_idx = cache_idx
                        
                        if best_cache_idx is not None:
                            matched_entries_map[person_idx] = cache_entries[best_cache_idx]
            else:
                matched_entries_map = {}
            
            for person_data in person_data_list:
                i = person_data['index']
                person_id_text = person_data['person_id']
                x1, y1, x2, y2 = person_data['box']
                person_name = person_data.get('name', 'Unknown')  # 원본 이름
                similarity_score = person_data.get('similarity', 0.0)

                # --- 얼굴 인식 안정화: 히스테리시스 + 홀드 ---
                matched_entry = matched_entries_map.get(i)  # 배치 계산 결과 사용
                try:

                    # 상승/하강 임계 적용
                    if person_name != "Unknown" and similarity_score >= up_th:
                        # 확정 또는 갱신(EMA)
                        if matched_entry is not None:
                            prev_score = float(matched_entry.get('score', similarity_score))
                            smoothed = 0.7 * similarity_score + 0.3 * prev_score
                            # 캐시 항목 갱신 (IdentityCache에 다시 추가)
                            matched_entry['box'] = (x1, y1, x2, y2)
                            matched_entry['name'] = person_name
                            matched_entry['score'] = smoothed
                            matched_entry['ts'] = now_ts
                            recent_identity_cache.add(cam_id, matched_entry)
                        else:
                            # 새 항목 추가 (IdentityCache가 자동으로 크기 제한)
                            recent_identity_cache.add(cam_id, {
                                'box': (x1, y1, x2, y2),
                                'name': person_name,
                                'score': similarity_score
                                # 'ts'는 IdentityCache.add()에서 자동 추가
                            })
                    else:
                        # Unknown 또는 낮은 점수: 홀드 조건 충족 시 직전 라벨 유지
                        if matched_entry is not None:
                            age = now_ts - matched_entry.get('ts', 0)
                            last_score = float(matched_entry.get('score', 0.0))
                            if age <= hold_sec and last_score >= down_th:
                                person_name = matched_entry.get('name', person_name)
                                similarity_score = last_score
                                # 박스/시간 갱신
                                matched_entry.update({'box': (x1, y1, x2, y2), 'ts': now_ts})
                except Exception as _stb_e:
                    # 안정화 로직 오류는 무시하고 원 결과 사용
                    pass
                
                # now_ts가 try 블록 안에서만 정의되었을 수 있으므로 재정의
                if 'now_ts' not in locals():
                    now_ts = time.time()

                # 센트로이드 임베딩: 여러 프레임의 임베딩을 평균내어 안정성 향상 (final 개선 기법)
                # matched_entry를 재사용하여 중복 계산 제거 (성능 최적화)
                embedding = person_data.get('embedding')
                
                if embedding is not None:
                    # person_box_key 생성 (헬퍼 함수 사용)
                    person_box_key = _generate_person_box_key(cam_id, matched_entry, x1, y1, x2, y2)
                    
                    # 임베딩을 버퍼에 추가 (안전한 접근)
                    if cam_id not in embedding_buffers:
                        embedding_buffers[cam_id] = {}
                    if person_box_key not in embedding_buffers[cam_id]:
                        embedding_buffers[cam_id][person_box_key] = {'embeddings': [], 'last_update': 0.0}
                    buffer_data = embedding_buffers[cam_id][person_box_key]
                    # 임베딩 복사 최적화: 버퍼에 추가할 때는 복사 (안정성 우선)
                    # 센트로이드 계산 시에는 이미 복사된 배열을 사용하므로 추가 복사 불필요
                    buffer_data['embeddings'].append(embedding.copy())
                    buffer_data['last_update'] = now_ts
                    
                    # 버퍼 크기 제한 (최대 EMBEDDING_BUFFER_SIZE)
                    if len(buffer_data['embeddings']) > EMBEDDING_BUFFER_SIZE:
                        buffer_data['embeddings'] = buffer_data['embeddings'][-EMBEDDING_BUFFER_SIZE:]
                    
                    # 센트로이드 계산 최적화: 버퍼 크기가 충분히 클 때만 계산 (5개 이상)
                    # 최적화: 계산 주기를 늘려 CPU 사용량 감소 (5개 이상일 때만 계산)
                    if len(buffer_data['embeddings']) >= 5:
                        logging.debug(f"🔍 {person_id_text} 센트로이드 계산 시작: 버퍼 크기={len(buffer_data['embeddings'])}, person_box_key={person_box_key}")
                        # 캐시 확인 (최근 2초 내 결과 재사용)
                        cached_centroid = centroid_cache[cam_id].get(person_box_key)
                        if cached_centroid:
                            person_name_centroid = cached_centroid.get('name', 'Unknown')
                            similarity_score_centroid = cached_centroid.get('score', 0.0)
                            logging.debug(f"🔍 {person_id_text} 센트로이드 캐시 사용: {person_name_centroid} (유사도={similarity_score_centroid:.3f})")
                        else:
                            # 센트로이드 계산 최적화: numpy 배열을 한 번에 처리
                            # 버퍼에 이미 복사된 배열이므로 추가 복사 불필요 (메모리 최적화)
                            embeddings_array = np.array(buffer_data['embeddings'], dtype=np.float32)
                            if len(embeddings_array) > 0:
                                avg_embedding = np.mean(embeddings_array, axis=0)
                                norm = np.linalg.norm(avg_embedding)
                                if norm > 1e-6:
                                    normalized_avg_embedding = (avg_embedding / norm).astype('float32')
                                    # 센트로이드 임베딩으로 재검색 (약간 엄격한 임계값 적용)
                                    # 센트로이드는 여러 프레임 평균이므로 기본보다 +0.03만 상향
                                    centroid_threshold = config.Thresholds.SIMILARITY + 0.03  # 기본 임계값 + 0.03
                                    logging.debug(f"🔍 {person_id_text} 센트로이드 FAISS 검색: 임계값={centroid_threshold:.3f}")
                                    person_name_centroid, similarity_score_centroid = find_best_match_faiss(
                                        normalized_avg_embedding, face_database, centroid_threshold
                                    )
                                    logging.debug(f"🔍 {person_id_text} 센트로이드 결과: {person_name_centroid} (유사도={similarity_score_centroid:.3f})")
                                    # 캐시에 저장 (TTLCache가 자동으로 만료 처리)
                                    centroid_cache[cam_id].put(person_box_key, {
                                        'name': person_name_centroid,
                                        'score': similarity_score_centroid
                                    })
                                else:
                                    person_name_centroid = "Unknown"
                                    similarity_score_centroid = 0.0
                                    logging.debug(f"⚠️ {person_id_text} 센트로이드 정규화 실패: norm={norm}")
                            else:
                                person_name_centroid = "Unknown"
                                similarity_score_centroid = 0.0
                        
                        # 센트로이드 결과가 더 좋으면 사용 (Unknown이 아니고 similarity가 더 높으면)
                        if person_name_centroid != "Unknown" and (person_name == "Unknown" or similarity_score_centroid > similarity_score):
                            logging.debug(f"✅ {person_id_text} 센트로이드 결과 적용: {person_name_centroid} (유사도={similarity_score_centroid:.3f}, 기존={person_name}, 유사도={similarity_score:.3f})")
                            person_name = person_name_centroid
                            similarity_score = similarity_score_centroid
                            # 버퍼 초기화 (인식 성공 시)
                            buffer_data['embeddings'] = []
                        else:
                            logging.debug(f"🔍 {person_id_text} 센트로이드 결과 미사용: 센트로이드={person_name_centroid} (유사도={similarity_score_centroid:.3f}), 기존={person_name} (유사도={similarity_score:.3f})")
                
                # ⭐ 위치 기반 투표로 이름 안정화 (깜빡거림 방지)
                if person_name != "Unknown":
                    from state import vote_for_name
                    voted_name, voted_score = vote_for_name(
                        cam_id, int(x1), int(y1), int(x2), int(y2), 
                        person_name, float(similarity_score)
                    )
                    if voted_name != person_name:
                        logging.debug(f"[CAM-{cam_id}] 투표로 이름 안정화: {person_name} → {voted_name}")
                    person_name = voted_name
                    similarity_score = voted_score
                
                # 동일 이름 중복 방지: 같은 이름이 여러 박스에 할당되면 가장 높은 similarity만 유지
                if person_name != "Unknown":
                    if person_name not in name_to_boxes:
                        name_to_boxes[person_name] = []
                    name_to_boxes[person_name].append(((x1, y1, x2, y2), float(similarity_score), i))
            
            # 오래된 임베딩 버퍼 및 캐시 정리 (메모리 관리 개선)
            # 주기적 정리로 최적화: 매 프레임마다 실행하지 않고 10초마다 실행
            if not hasattr(process_single_frame, '_last_cleanup_time'):
                process_single_frame._last_cleanup_time = 0.0
            
            current_time_cleanup = time.time()
            CLEANUP_INTERVAL = 10.0  # 10초마다 정리
            
            if current_time_cleanup - process_single_frame._last_cleanup_time > CLEANUP_INTERVAL:
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
                        logging.debug(f"CAM-{cam_id_cleanup} 버퍼 수 제한: {MAX_EMBEDDING_BUFFERS_PER_CAM}개로 축소")
                    
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
                
                # 정리 시간 업데이트
                process_single_frame._last_cleanup_time = current_time_cleanup
            
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
                    # 같은 이름이 여러 박스에 할당됨: 모두 같은 이름 허용
                    # (실제로 여러 사람이 같은 이름으로 인식될 수 있음 - 투표 시스템이 처리)
                    for box, score, idx in boxes_scores_indices:
                        person_final_names[idx] = name
            
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
                
                # person_box_key 생성 (헬퍼 함수 사용)
                person_box_key = _generate_person_box_key(cam_id, matched_entry, x1, y1, x2, y2)
                
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
                        
                        # IoU가 0.3 이상이면 같은 사람으로 간주하고 캐시된 바운딩박스 사용
                        if iou >= 0.3:
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
                
                    # PPE 박스가 있거나 위반이 있거나 얼굴이 감지되면 렌더링
                if ppe_boxes_list or current_violations or person_status != "SAFE" or person_name != "Unknown" or face_bbox_to_draw is not None:
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

                    # 렌더링이 필요한 시점에 프레임 복사 (메모리 최적화: 필요할 때만 복사)
                    if processed_frame is None:
                        processed_frame = frame.copy()
                    
                    # person_box로 바운딩 박스 그리기 (현대적 스타일)
                    draw_modern_bbox(processed_frame, x1, y1, x2, y2, unified_color, thickness=3, corner_length=35, alpha=alpha)
                    
                    # 상태 텍스트 표시 (person_box 위에 표시) - 프론트엔드와 동일한 형식
                    # 라벨 텍스트 구성: "이름: 위반내역" 형식
                    display_name = person_name if person_name != "Unknown" else "알 수 없음"
                    
                    # 위반 정보 수집
                    violation_parts = []
                    if "넘어짐" in current_violations:
                        violation_parts.append("넘어짐 감지")
                    # PPE 위반 정보 수집
                    ppe_violations_display = []
                    for v in current_violations:
                        if v == "안전모":
                            ppe_violations_display.append("안전모")
                        elif v == "안전조끼":
                            ppe_violations_display.append("안전조끼")
                        elif v == "마스크":
                            ppe_violations_display.append("마스크")
                    
                    # 위반 문자열 구성
                    violation_str = ""
                    if violation_parts or ppe_violations_display:
                        violation_list = violation_parts + ppe_violations_display
                        if "넘어짐 감지" in violation_list and len(violation_list) > 1:
                            # 넘어짐과 PPE 위반이 함께 있을 때
                            ppe_only = [v for v in violation_list if v != "넘어짐 감지"]
                            violation_str = f"넘어짐 감지, {', '.join(ppe_only)} 미착용"
                        elif "넘어짐 감지" in violation_list:
                            violation_str = "넘어짐 감지"
                        else:
                            violation_str = f"{', '.join(ppe_violations_display)} 미착용"
                    
                    # 최종 라벨 텍스트 구성
                    if violation_str:
                        status_text = f"{display_name}: {violation_str}"
                    else:
                        status_text = display_name
                    
                    # 라벨 표시 조건: 얼굴 인식이 성공하면 트래킹으로 계속 이름 표시
                    # 위반이 있으면 얼굴이 없어도 위반 정보는 표시
                    should_show_label = False
                    if current_violations:
                        # 위반이 있으면 항상 표시
                        should_show_label = True
                    elif person_name != "Unknown":
                        # 얼굴 인식이 성공한 경우 (한 번 인식되면 트래킹으로 계속 표시)
                        # person_name이 "Unknown"이 아니면 이미 인식된 이름이므로 계속 표시
                        # 현재 프레임에서 얼굴이 감지되었거나, 이전에 인식된 이름이 있으면 표시
                        if face_bbox is not None:
                            # 현재 프레임에서 얼굴 감지: 항상 표시
                            should_show_label = True
                        else:
                            # 얼굴이 감지되지 않아도, person_name이 "Unknown"이 아니면
                            # 이전에 인식된 이름이므로 트래킹으로 계속 표시
                            should_show_label = True
                    
                    if should_show_label:
                        # person_box 위치에 텍스트 표시
                        text_x, text_y = x1, y1

                        # 색상 결정 (박스와 동일)
                        text_color = unified_color

                        # person_box 위에 텍스트 표시
                        renderer.add_text(status_text, (text_x, text_y - 10), text_color)

                # 위반 사항 기록 (중복 제거: 같은 사람 박스에 대해 한 번만 기록)
                if current_violations:
                    # 배치 IoU 계산으로 중복 확인 최적화
                    is_duplicate = False
                    if len(violations_found) > 0:
                        # 기존 위반 박스 배열 준비
                        existing_boxes = []
                        for existing_violation in violations_found:
                            ex_box = existing_violation.get("person_box", [])
                            if len(ex_box) == 4:
                                existing_boxes.append(ex_box)
                        
                        if len(existing_boxes) > 0:
                            # 배치 IoU 계산
                            current_box_array = np.array([(x1, y1, x2, y2)], dtype=np.float32)
                            existing_boxes_array = np.array(existing_boxes, dtype=np.float32)
                            iou_matrix = calculate_iou_batch(current_box_array, existing_boxes_array)
                            
                            # 최대 IoU가 0.6 이상이면 중복
                            max_iou = float(np.max(iou_matrix))
                            if max_iou > 0.6:
                                is_duplicate = True
                    
                    if not is_duplicate:
                        # cam_id를 area로 매핑 (0→A-1, 1→A-2, 2→B-1, 3→B-2)
                        area_map = {0: "A-1", 1: "A-2", 2: "B-1", 3: "B-2"}
                        area = area_map.get(cam_id, f"A-{cam_id+1}")
                        
                        # 위반 내용을 hazard 문자열로 변환
                        # 예: "PPE 위반내역: 안전모, 마스크, 안전조끼"
                        # 중복 제거: 위반 유형을 set으로 변환하여 중복 제거
                        # 최적화: current_violations가 이미 리스트이므로 set 변환 후 리스트 변환
                        if isinstance(current_violations, (set, tuple)):
                            unique_violations = list(current_violations)
                        else:
                            unique_violations = list(set(current_violations))  # 중복 제거
                        ppe_violations = []
                        other_violations = []
                        
                        for violation_type in unique_violations:
                            if violation_type == "넘어짐":
                                other_violations.append("넘어짐 감지")
                            elif violation_type == "안전모":
                                ppe_violations.append("안전모")
                            elif violation_type == "마스크":
                                ppe_violations.append("마스크")
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
            # 위반이 있는 사람만 recognized_faces에 추가 (얼굴 인식은 계속 실행하되 위반 시에만 전송)
            # 중요: 각 사람마다 별도의 항목을 보내야 함 (같은 이름이어도 다른 person_box를 가지면 다른 사람)
            
            # 디버깅: person_data_list 개수 확인
            logging.debug(f"[CAM-{cam_id}] person_data_list: {len(person_data_list)}명 감지")
            
            added_boxes = set()  # 박스 튜플 기반 중복 제거 (IoU 계산용)
            skipped_count = 0  # 중복으로 스킵된 사람 수
            added_count = 0  # 추가된 사람 수
            
            for person_data in person_data_list:
                i = person_data['index']
                x1, y1, x2, y2 = person_data['box']
                final_name = person_final_names.get(i, person_data.get('name', 'Unknown'))
                similarity_score = person_data.get('similarity', 0.0)
                person_status = person_status_map.get(i, "SAFE")
                
                # 박스 기반 중복 제거 (IoU 기반으로 매우 엄격하게)
                # IoU 0.98 이상이면 같은 사람으로 간주 (거의 완전히 겹치는 경우만)
                # 멀리 있는 사람은 IoU가 낮으므로 중복으로 간주되지 않음
                is_duplicate = False
                for seen_box in added_boxes:
                    if len(seen_box) == 4:
                        # 거리 기반 필터링 먼저 수행 (성능 향상 및 정확도 향상)
                        current_center_x = (x1 + x2) / 2
                        current_center_y = (y1 + y2) / 2
                        seen_center_x = (seen_box[0] + seen_box[2]) / 2
                        seen_center_y = (seen_box[1] + seen_box[3]) / 2
                        
                        # 현재 박스의 대각선 길이 계산
                        current_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        center_distance = ((current_center_x - seen_center_x) ** 2 + (current_center_y - seen_center_y) ** 2) ** 0.5
                        
                        # 거리가 대각선 길이의 0.5배 이상이면 중복이 아님 (IoU 계산 생략)
                        if center_distance > current_diagonal * 0.5:
                            continue  # 멀리 있으면 중복 아님
                        
                        # 거리가 가까우면 IoU 계산
                        iou = utils.calculate_iou((x1, y1, x2, y2), seen_box)
                        # IoU 0.98 이상이면 같은 사람으로 간주 (거의 완전히 겹치는 경우만)
                        # 멀리 있는 사람은 IoU가 낮으므로 중복으로 간주되지 않음
                        if iou > 0.98:  # 0.95 -> 0.98로 더 엄격하게 (거의 완전히 겹치는 경우만)
                            is_duplicate = True
                            logging.debug(f"[CAM-{cam_id}] 중복 제거: person_idx={i}, box=({x1}, {y1}, {x2}, {y2}), IoU={iou:.3f}, seen_box={seen_box}, 거리={center_distance:.1f}")
                            break
                        elif iou > 0.5:  # IoU가 0.5 이상이면 디버깅 로그 출력
                            logging.debug(f"[CAM-{cam_id}] IoU 체크: person_idx={i}, box=({x1}, {y1}, {x2}, {y2}), IoU={iou:.3f}, seen_box={seen_box}, 거리={center_distance:.1f} (중복 아님)")
                
                if is_duplicate:
                    # 거의 완전히 겹치는 사람은 이미 추가됨 (중복 방지)
                    skipped_count += 1
                    continue
                
                # PPE 위반 정보 가져오기 (프론트엔드 표시용)
                ppe_violations = person_data.get('ppe_violations', [])
                
                # 디버깅: PPE 위반 정보 확인
                if ppe_violations:
                    logging.info(f"[CAM-{cam_id}] PPE 위반 확인: person_idx={i}, 위반={ppe_violations}, 상태={person_status}")
                
                # 위반이 있는 사람만 recognized_faces에 추가 (얼굴 인식은 계속 실행하되 위반 시에만 전송)
                # 위반이 있으면 얼굴 인식이 실패해도 "Unknown"으로 보내기
                if person_status != "SAFE" or ppe_violations:
                    # 얼굴 인식 결과 사용 (실패해도 "Unknown"으로 설정됨)
                    face_name = final_name if final_name != "Unknown" else "Unknown"
                    
                    # ⭐⭐ 중복 박스 방지: 하나의 Person에 하나의 박스만 추가
                    # 위반 정보와 얼굴 정보를 하나의 객체에 통합
                    recognized_faces.append({
                        "box": [x1, y1, x2, y2],  # person_box 사용
                        "bbox": [x1, y1, x2, y2],  # 프론트엔드 호환성을 위해 bbox도 추가
                        "name": face_name,  # 얼굴 인식 결과 (실패 시 "Unknown")
                        "worker": face_name,  # 프론트엔드 호환성 (worker 필드)
                        "similarity": float(similarity_score),
                        "status": person_status,  # 상태 정보 추가 (KPI 계산용)
                        "ppe_violations": ppe_violations,  # PPE 위반 정보 추가 (프론트엔드 표시용)
                        "isFace": True,  # 얼굴 인식 시도했음을 표시
                        "isViolation": len(ppe_violations) > 0,  # 위반 플래그 (프론트엔드 필수)
                    })
                    added_boxes.add((x1, y1, x2, y2))  # 박스 중복 방지
                    added_count += 1
            
            # 디버깅: recognized_faces 추가 결과 확인
            logging.debug(f"[CAM-{cam_id}] recognized_faces 추가: person_data_list={len(person_data_list)}명, 추가={added_count}명, 중복제거={skipped_count}명, 최종={len(recognized_faces)}명")

        # 8. 기타 객체 그리기 (안전 장비는 위에서 이미 처리했으므로 제외)
        # violations와 recognized_faces에서 사람 박스 추출
        person_boxes_for_filter = []
        for v in violations:
            if 'person_box' in v and v['person_box']:
                person_boxes_for_filter.append(v['person_box'])
        for rf in recognized_faces:
            if 'box' in rf and rf['box']:
                person_boxes_for_filter.append(rf['box'])
        
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
                        for person_box in person_boxes_for_filter:
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
                            logging.debug(f"작은 객체 필터링: {class_name} (사람 박스와 겹침)")
                            continue

                        # 렌더링이 필요한 시점에 프레임 복사 (메모리 최적화)
                        if processed_frame is None:
                            processed_frame = frame.copy()
                        
                        # 원본 프레임에 직접 그리기 (현대적 스타일, 이미 스케일링된 좌표)
                        draw_modern_bbox(processed_frame, x1_obj, y1_obj, x2_obj, y2_obj, color, thickness=1, corner_length=15, alpha=0.15)
                        display_name = class_name[:10]
                        renderer.add_text(f"{display_name}", (x1_obj, y1_obj - 5), color)

        # 스킵/누락 상황에서 박스/라벨 유지: 캐시로 보강 (강화 버전)
        # 렌더링 전에 recognized_faces가 비어있으면 캐시에서 강제로 가져와서 박스 그리기
        try:
            hold_sec = config.Thresholds.RECOGNITION_HOLD_SECONDS
            now_ts = time.time()
            
            # TTLCache에서 최근 항목 가져오기 (자동 만료 처리)
            cache = recent_identity_cache.get(cam_id)
            cache_entries = []
            if cache is not None:
                # TTLCache는 딕셔너리처럼 사용 가능, 모든 항목 순회
                for tracker_id_key, entry in cache.items():
                    # 만료된 항목은 자동으로 제거되므로 여기서는 유효한 항목만 처리
                    entry_ts = entry.get('ts', 0)
                    age = now_ts - entry_ts
                    if age <= hold_sec:
                        # tracker_id와 함께 저장
                        entry_with_tracker = entry.copy()
                        entry_with_tracker['tracker_id'] = tracker_id_key
                        cache_entries.append(entry_with_tracker)
            
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
                            # 겹치는 사람 구분을 위해 IoU 임계값 상향
                            is_duplicate = False
                            for existing in recognized_faces:
                                ex_box = existing.get("box", [])
                                if len(ex_box) == 4:
                                    ex_iou = utils.calculate_iou((x1, y1, x2, y2), tuple(ex_box))
                                    if ex_iou > 0.5:  # IoU 0.5 이상이면 중복 (겹치는 사람 구분)
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate:
                                # 캐시에서 가져온 항목은 VIOLATION 상태를 알 수 없으므로 박스를 그리지 않음
                                # SAFE 바운딩 박스는 표시하지 않음 (사용자 요청)
                                # preserved에만 추가하고 렌더링은 하지 않음
                                preserved.append({
                                    "box": [int(x1), int(y1), int(x2), int(y2)],
                                    "name": name,
                                    "similarity": score
                                })
                                # 초록색 박스 제거: VIOLATION 상태를 알 수 없는 캐시 항목은 렌더링하지 않음
                
                # 보강된 항목을 recognized_faces에 추가
                if preserved:
                    recognized_faces.extend(preserved)

            # 렌더링 캐시 보강 제거: SAFE 바운딩 박스는 표시하지 않음 (사용자 요청)
            # VIOLATION 상태를 알 수 없는 캐시 항목은 렌더링하지 않음
        except Exception as e:
            logging.debug(f"캐시 보강 중 오류 (무시): {e}")

        # 중복 제거: 박스 기반으로만 중복 제거 (같은 person_box를 가진 항목만 제거)
        # 중요: 각 사람마다 별도의 항목을 유지해야 함 (같은 이름이어도 다른 person_box를 가지면 다른 사람)
        # 박스 기반 중복 제거는 이미 위에서 수행했으므로 여기서는 추가 중복 제거하지 않음
        try:
            if recognized_faces:
                # 박스 기반 중복 제거 (IoU 기반)
                unique_faces = []
                seen_boxes = set()
                
                for face in recognized_faces:
                    box = face.get("box", [])
                    if len(box) == 4:
                        box_tuple = tuple(box)
                        # 같은 박스를 가진 항목이 이미 있으면 IoU로 확인
                        is_duplicate = False
                        for seen_box in seen_boxes:
                            if len(seen_box) == 4:
                                iou = utils.calculate_iou(box_tuple, seen_box)
                                if iou > 0.7:  # IoU 0.7 이상이면 같은 사람으로 간주 (0.98 -> 0.7로 완화하여 중복 제거 강화)
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            unique_faces.append(face)
                            seen_boxes.add(box_tuple)
                    else:
                        # 박스가 없으면 그대로 추가
                        unique_faces.append(face)
                
                recognized_faces = unique_faces
                
                # 디버깅: recognized_faces 개수 확인
                logging.debug(f"[CAM-{cam_id}] 최종 recognized_faces: {len(recognized_faces)}개")
        except Exception as e:
            logging.debug(f"중복 제거 중 오류 (무시): {e}")

        # 바운딩 박스 스무딩 제거: 잔상 방지를 위해 이전 프레임 결과를 현재 프레임에 추가하지 않음
        # 좌표 스무딩만 유지 (이미 위에서 처리됨)
        # 이전 프레임의 박스를 현재 프레임에 추가하는 로직은 잔상을 유발하므로 제거

        # 8. 렌더링 (텍스트 오버레이)
        rendering_start = time.time()
        # 렌더링이 필요한 경우에만 프레임 복사 (메모리 최적화)
        if processed_frame is None:
            processed_frame = frame.copy()
        processed_frame = renderer.render_on(processed_frame)

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

        # 9. 처리된 프레임을 JPEG 바이트로 인코딩 (프로덕션 최적화)
        # 스트리밍 지연 해결: 이미지 크기 리사이즈 (FHD -> HD급으로 축소하여 전송량 50% 절감)
        # 원본 분석은 고해상도로 했으므로 정확도는 유지됨
        stream_width = 1280  # 1280px (HD) 정도면 충분히 선명함
        processed_frame_resized = processed_frame
        # 리사이즈 최적화: 필요할 때만 리사이즈 (프레임 크기가 같으면 스킵)
        if processed_frame.shape[1] > stream_width:
            try:
                aspect_ratio = processed_frame.shape[0] / processed_frame.shape[1]
                stream_height = int(stream_width * aspect_ratio)
                # 빠른 리사이즈 (INTER_LINEAR가 속도와 품질의 균형)
                processed_frame_resized = cv2.resize(processed_frame, (stream_width, stream_height), interpolation=cv2.INTER_LINEAR)
            except Exception:
                processed_frame_resized = processed_frame

        encoding_start = time.time()
        # 프로덕션 품질 조정: 95 (고화질)
        ret, buffer = cv2.imencode('.jpg', processed_frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        perf_timings['encoding'] = (time.time() - encoding_start) * 1000  # ms
        if not ret:
            logging.error("JPEG 인코딩 실패")
            empty_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            return buffer.tobytes(), {"timestamp": time.time(), "recognized_faces": [], "violations": [], "violation_count": 0}

        processed_frame_bytes = buffer.tobytes()

        # 프레임 보장 방식: best_result는 이미 위에서 찾았고, 렌더링도 완료되었으므로
        # 여기서는 인코딩된 프레임만 반환

        # 전체 시간 계산
        perf_timings['total'] = (time.time() - total_start) * 1000  # ms
        
        # 성능 데이터 로깅 (주기적으로만 출력 - 성능 최적화)
        if not hasattr(process_single_frame, '_perf_log_count'):
            process_single_frame._perf_log_count = {}
        if cam_id not in process_single_frame._perf_log_count:
            process_single_frame._perf_log_count[cam_id] = 0
        process_single_frame._perf_log_count[cam_id] += 1
        
        # FPS 향상을 위해 성능 로깅 빈도 감소 (매 프레임 -> 10프레임마다)
        if process_single_frame._perf_log_count[cam_id] % 10 == 0:
            # 개별 모델 처리 시간 포함 (GPU 사용 여부 확인용)
            perf_msg = f"[PERF CAM-{cam_id}] 총 처리: {perf_timings['total']:.1f}ms | "
            perf_msg += f"Decode={perf_timings['decode']:.1f}ms | "
            perf_msg += f"Resize={perf_timings['resize']:.1f}ms | "
            if 'yolo_violation_actual' in perf_timings:
                perf_msg += f"Violation={perf_timings['yolo_violation_actual']:.1f}ms | "
            else:
                perf_msg += f"Violation={perf_timings['yolo_violation']:.1f}ms | "
            if 'yolo_pose_actual' in perf_timings:
                perf_msg += f"Pose={perf_timings['yolo_pose_actual']:.1f}ms | "
            else:
                perf_msg += f"Pose={perf_timings['yolo_pose']:.1f}ms | "
            if 'yolo_face_actual' in perf_timings:
                perf_msg += f"Face={perf_timings['yolo_face_actual']:.1f}ms | "
            perf_msg += f"Parse={perf_timings.get('parse_results', 0):.1f}ms | "
            perf_msg += f"얼굴인식={perf_timings.get('face_recognition', 0):.1f}ms | "
            perf_msg += f"Render={perf_timings.get('rendering', 0):.1f}ms | "
            perf_msg += f"Encode={perf_timings.get('encoding', 0):.1f}ms"
            logging.info(perf_msg)  # debug -> info (병목 분석용)
            
            # GPU 사용 여부 추정 (처리 시간 기준)
            if 'yolo_violation_actual' in perf_timings and 'yolo_pose_actual' in perf_timings:
                v_time = perf_timings['yolo_violation_actual']
                p_time = perf_timings['yolo_pose_actual']
                if v_time < 50 and p_time < 50:
                    logging.info(f"[PERF CAM-{cam_id}] ✅ GPU 사용 추정: Violation={v_time:.1f}ms, Pose={p_time:.1f}ms (GPU 속도 범위)")
                elif v_time > 150 or p_time > 150:
                    logging.warning(f"[PERF CAM-{cam_id}] ⚠️ CPU 사용 가능성: Violation={v_time:.1f}ms, Pose={p_time:.1f}ms (CPU 속도 범위)")
            
            # 병목 지점 식별 (매 프레임마다, 상위 3개 표시)
            if perf_timings['total'] > 0:
                bottlenecks = []
                for key, value in perf_timings.items():
                    if key != 'total' and value > 0:
                        percentage = (value / perf_timings['total']) * 100
                        if percentage > 10:  # 10% 이상 차지하는 단계만 표시
                            bottlenecks.append((key, value, percentage))
                
                # 시간 순으로 정렬
                bottlenecks.sort(key=lambda x: x[1], reverse=True)
                if bottlenecks:
                    bottleneck_msg = f"[BOTTLENECK CAM-{cam_id}] "
                    for i, (key, value, pct) in enumerate(bottlenecks[:3]):  # 상위 3개만
                        bottleneck_msg += f"{key}={value:.1f}ms({pct:.0f}%) "
                    logging.warning(bottleneck_msg)

        # 10. 결과 데이터 구성
        # 로깅 최소화 (성능 최적화)
        faces_count = len(recognized_faces)
        violations_count = len(violations_found)
        
        # PPE 위반이 있는 사람 수 확인 (디버깅용)
        ppe_violation_count = sum(1 for face in recognized_faces if face.get('ppe_violations', []))
        
        if faces_count > 0 or violations_count > 0:
            logging.debug(f"[CAM-{cam_id}] AI 결과: 얼굴={faces_count}개, 위반={violations_count}개, PPE위반 있는 사람={ppe_violation_count}개")
        
        result_data = {
            "timestamp": time.time(),  # 타임스탬프 추가 (WebSocket 전송용)
            "recognized_faces": recognized_faces,
            "violations": violations_found,
            "violation_count": len(violations_found),
            "performance": perf_timings,  # 성능 측정 데이터 포함
            "frame_width": orig_w,  # 원본 프레임 너비 (바운딩 박스 좌표 기준)
            "frame_height": orig_h,  # 원본 프레임 높이 (바운딩 박스 좌표 기준)
            "cam_id": cam_id  # 카메라 ID 추가 (프론트엔드 디버깅용)
        }

        total_elapsed = (time.time() - total_start) * 1000
        # 실제 FPS는 camera_worker에서 계산하므로, 여기서는 처리 시간만 로깅
        logging.debug(f"[CAM-{cam_id}] 처리 완료: 얼굴={faces_count}개, 위반={violations_count}개, 처리시간={total_elapsed:.1f}ms")
        
        # 주기적 GPU/워커 상태 로깅 (10초마다)
        current_time = time.time()
        if not hasattr(process_single_frame, '_last_status_log_time'):
            process_single_frame._last_status_log_time = {}
        if cam_id not in process_single_frame._last_status_log_time:
            process_single_frame._last_status_log_time[cam_id] = current_time
        
        if current_time - process_single_frame._last_status_log_time[cam_id] >= 10.0:  # 10초마다
            process_single_frame._last_status_log_time[cam_id] = current_time
            
            # GPU 사용률 확인
            gpu_stats = {}
            if torch.cuda.is_available():
                try:
                    for gpu_id in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(gpu_id)
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                        memory_total = props.total_memory / 1024**3  # GB
                        memory_util = (memory_reserved / memory_total) * 100 if memory_total > 0 else 0
                        
                        gpu_stats[gpu_id] = {
                            "name": props.name,
                            "memory_allocated_gb": memory_allocated,
                            "memory_reserved_gb": memory_reserved,
                            "memory_total_gb": memory_total,
                            "memory_util_percent": memory_util
                        }
                except Exception as e:
                    logging.debug(f"GPU 통계 수집 오류: {e}")
            
            # 워커 상태 확인
            try:
                # 워커 수 (최대 워커 수)
                face_max_workers = face_recognition_executor._max_workers
                yolo_max_workers = yolo_executor._max_workers
                danger_max_workers = dangerous_behavior_executor._max_workers
                frame_max_workers = frame_processing_executor._max_workers
                
                # Executor의 내부 큐 크기 추정 (안전한 방법)
                face_queue_size = 0
                yolo_queue_size = 0
                try:
                    # ThreadPoolExecutor의 내부 큐는 직접 접근 불가, 대신 통계로 추정
                    # _work_queue는 내부 구현이므로 안전하게 접근
                    if hasattr(face_recognition_executor, '_work_queue'):
                        try:
                            face_queue_size = face_recognition_executor._work_queue.qsize()
                        except:
                            pass
                    if hasattr(yolo_executor, '_work_queue'):
                        try:
                            yolo_queue_size = yolo_executor._work_queue.qsize()
                        except:
                            pass
                except:
                    pass
                
                # 프레임 큐 크기
                frame_queue_size = 0
                try:
                    if cam_id in state.frame_queues:
                        frame_queue_size = state.frame_queues[cam_id].qsize()
                except:
                    pass
                
                # FPS 계산
                with frame_stats_lock:
                    cam_stat = frame_stats.get(cam_id, {})
                    recent_frames = cam_stat.get('recent_frame_times', [])
                    if len(recent_frames) >= 2:
                        time_span = recent_frames[-1] - recent_frames[0]
                        current_fps = (len(recent_frames) - 1) / time_span if time_span > 0 else 0
                    else:
                        current_fps = 0
                
                # 평균 처리 시간
                avg_processing_time = total_elapsed
                if 'processing_times' in cam_stat:
                    times = cam_stat['processing_times']
                    if len(times) > 0:
                        avg_processing_time = sum(times) / len(times)
                
                # 상태 로깅
                logging.info(f"📊 [CAM-{cam_id}] 시스템 상태 (10초 주기):")
                logging.info(f"   FPS: {current_fps:.1f} | 평균 처리시간: {avg_processing_time:.1f}ms")
                logging.info(f"   워커: Face={face_max_workers}, YOLO={yolo_max_workers}, Danger={danger_max_workers}, Frame={frame_max_workers}")
                logging.info(f"   큐: Face={face_queue_size}, YOLO={yolo_queue_size}, Frame={frame_queue_size}")
                
                if gpu_stats:
                    for gpu_id, stat in gpu_stats.items():
                        logging.info(f"   GPU {gpu_id} ({stat['name']}): 메모리={stat['memory_reserved_gb']:.2f}GB/{stat['memory_total_gb']:.2f}GB ({stat['memory_util_percent']:.1f}%)")
                        logging.warning(f"   ⚠️ 참고: TensorRT는 PyTorch CUDA 메모리와 별도로 작동합니다. 실제 GPU 사용률은 nvidia-smi로 확인하세요.")
            except Exception as e:
                logging.debug(f"워커 상태 로깅 오류: {e}")
        
        # 렌더링된 프레임 캐시 저장 (다음 스킵 프레임에서 재사용하여 바운딩 박스 유지)
        _last_rendered_frames[cam_id] = (processed_frame_bytes, result_data)
        
        return processed_frame_bytes, result_data

    except Exception as e:
        total_failed = (time.time() - total_start) * 1000
        error_msg = str(e)
        logging.error(f"AI 처리 실행 중 오류 (CAM-{cam_id}, 누적 {total_failed:.2f}ms): {e}", exc_info=True)
        
        # 오류 프레임 생성 (더 자세한 정보 포함)
        error_frame = frame if frame is not None else np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        
        # 오류 메시지를 여러 줄로 표시
        error_lines = [
            "Processing Error",
            error_msg[:50] + ("..." if len(error_msg) > 50 else ""),
            "Check backend logs"
        ]
        
        y_offset = 30
        for i, line in enumerate(error_lines):
            cv2.putText(error_frame, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes(), {"error": error_msg, "recognized_faces": [], "violations": []}