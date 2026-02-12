# ai_processors.py - AI 처리 로직
"""
AI 모델 처리 함수 모듈
PPE 감지, 얼굴 인식, 위험 행동 감지 등
"""
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np
from ultralytics.engine.results import Keypoints

import utils
import config
from utils import find_best_match_faiss
from exceptions import (
    ProcessingError,
    FaceRecognitionError
)
from state import (
    fall_start_times,
    FALL_DURATION_THRESHOLD
)
from fast_face_recognizer import FastIndustrialRecognizer


def _process_ppe_detection(
    person_box: Tuple[int, int, int, int], 
    all_detections: Dict[str, List[Dict[str, Any]]],
    used_ppe_boxes: Optional[Set[Tuple[int, int, int, int]]] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    PPE 감지 전용 함수 (얼굴 인식과 독립적으로 항상 실행)
    멀리 있는 사람도 잘 잡기 위해 최고 성능 설정
    
    Returns:
        ppe_violations: PPE 위반 목록 (예: ["안전모", "마스크"])
        ppe_boxes: PPE 감지된 박스 정보 리스트 [{"bbox": (x1,y1,x2,y2), "class": "Safety Vest", "conf": 0.9}, ...]
    """
    ppe_violations = []
    ppe_boxes: List[Dict[str, Any]] = []  # PPE 감지 박스 정보
    
    if used_ppe_boxes is None:
        used_ppe_boxes = set()
    
    try:
        x1, y1, x2, y2 = person_box
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # 멀리 있는 사람(작은 박스)을 위한 동적 IoU 임계값 조정 (다른 사람 PPE 오매칭 방지)
        # 작은 박스일수록 더 낮은 IoU 임계값 사용하되, 너무 낮으면 다른 사람 PPE도 매칭됨
        if box_area < 5000:  # 매우 작은 박스 (멀리 있는 사람)
            ppe_iou_threshold = 0.05  # 0.0001 -> 0.05 (다른 사람 PPE 오매칭 방지)
        elif box_area < 10000:  # 작은 박스
            ppe_iou_threshold = 0.08  # 0.001 -> 0.08
        elif box_area < 20000:  # 중간 박스
            ppe_iou_threshold = 0.10  # 0.005 -> 0.10
        else:
            ppe_iou_threshold = 0.15  # 0.01 -> 0.15 (다른 사람 PPE 오매칭 방지 강화)
        
        # 모든 PPE 클래스 수집 (준수 및 위반 모두)
        ppe_class_names = set()
        for rule, classes in config.Constants.SAFETY_RULES_MAP.items():
            ppe_class_names.add(classes["compliance"])
            ppe_class_names.add(classes["violation"])
        
        # person_box와 겹치는 모든 PPE 박스 수집
        for ppe_class in ppe_class_names:
            if ppe_class in all_detections and all_detections[ppe_class]:
                # 거리 임계값 미리 계산 (다른 사람 PPE 오매칭 방지: 더 엄격하게)
                box_diagonal = ((box_w ** 2 + box_h ** 2) ** 0.5)
                distance_threshold = box_diagonal * (0.6 if box_area < 10000 else 0.5)  # 1.3->0.6, 1.0->0.5 (다른 사람 PPE 오매칭 방지)
                
                for det in all_detections[ppe_class]:
                    if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                        dx1, dy1, dx2, dy2 = det['bbox']
                        det_bbox_tuple = (int(dx1), int(dy1), int(dx2), int(dy2))
                        
                        # 이미 다른 사람에게 매칭된 PPE 박스는 제외 (중복 매칭 방지)
                        if det_bbox_tuple in used_ppe_boxes:
                            continue
                        
                        conf = det.get('conf', 0.9)
                        
                        # 중심점 거리 기반 판정 먼저 (IoU보다 빠름)
                        det_center_x = (dx1 + dx2) / 2
                        det_center_y = (dy1 + dy2) / 2
                        
                        # person_box 내부 포함 조건 추가 (다른 사람 PPE 오매칭 방지)
                        # PPE 박스의 중심점이 person_box 내부에 있어야 함
                        ppe_center_in_person = (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2)
                        if not ppe_center_in_person:
                            continue  # person_box 밖의 PPE는 무시
                        
                        center_distance = ((box_center_x - det_center_x) ** 2 + (box_center_y - det_center_y) ** 2) ** 0.5
                        
                        is_match = False
                        if center_distance < distance_threshold:
                            is_match = True
                        else:
                            # 거리 임계값을 넘으면 IoU 계산 (더 정확하지만 느림)
                            iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                            if iou > ppe_iou_threshold:
                                is_match = True
                        
                        if is_match:
                            # PPE 박스 정보 저장
                            ppe_boxes.append({
                                "bbox": det_bbox_tuple,
                                "class": ppe_class,
                                "conf": conf
                            })
        
        # PPE 박스 중복 제거 (IoU 기반, 같은 클래스 내에서만)
        if len(ppe_boxes) > 1:
            # 클래스별로 그룹화
            ppe_by_class: Dict[str, List[Dict[str, Any]]] = {}
            for ppe_box in ppe_boxes:
                ppe_class = ppe_box['class']
                if ppe_class not in ppe_by_class:
                    ppe_by_class[ppe_class] = []
                ppe_by_class[ppe_class].append(ppe_box)
            
            # 각 클래스별로 중복 제거 (IoU > 0.6이면 중복으로 간주, confidence가 높은 것만 유지)
            # IoU 임계값을 0.5 -> 0.6으로 상향하여 더 엄격한 중복 제거
            filtered_ppe_boxes: List[Dict[str, Any]] = []
            for ppe_class, boxes in ppe_by_class.items():
                if len(boxes) == 1:
                    filtered_ppe_boxes.append(boxes[0])
                else:
                    # confidence 기준으로 정렬 (높은 것부터)
                    boxes_sorted = sorted(boxes, key=lambda x: x['conf'], reverse=True)
                    kept_indices: List[int] = []
                    
                    for i, box1 in enumerate(boxes_sorted):
                        bx1, by1, bx2, by2 = box1['bbox']
                        is_duplicate = False
                        
                        # 이미 유지된 박스와 IoU 계산
                        for j in kept_indices:
                            box2 = boxes_sorted[j]
                            bx3, by3, bx4, by4 = box2['bbox']
                            iou = utils.calculate_iou((bx1, by1, bx2, by2), (bx3, by3, bx4, by4))
                            
                            # IoU가 0.6 이상이면 중복으로 간주 (0.5 -> 0.6, 더 엄격한 중복 제거)
                            if iou > 0.6:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            kept_indices.append(i)
                    
                    # 유지된 박스만 추가
                    for idx in kept_indices:
                        filtered_ppe_boxes.append(boxes_sorted[idx])
            
            ppe_boxes = filtered_ppe_boxes
        
        # 위반 판정 로직 (origin 버전 사용)
        for rule, classes in config.Constants.SAFETY_RULES_MAP.items():
            comp_cls, viol_cls = classes["compliance"], classes["violation"]
            is_compliance = False
            is_violation = False
            
            # 준수(착용) 판정: 중심점 거리 먼저 계산, IoU는 나중에 (성능 최적화)
            if comp_cls in all_detections and all_detections[comp_cls]:
                # 거리 임계값 미리 계산 (다른 사람 PPE 오매칭 방지: 더 엄격하게)
                box_diagonal = ((box_w ** 2 + box_h ** 2) ** 0.5)
                distance_threshold = box_diagonal * (0.6 if box_area < 10000 else 0.5)  # 1.3->0.6, 1.0->0.5 (다른 사람 PPE 오매칭 방지)
                
                for det in all_detections[comp_cls]:
                    if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                        dx1, dy1, dx2, dy2 = det['bbox']
                        det_bbox_tuple = (int(dx1), int(dy1), int(dx2), int(dy2))
                        
                        # 이미 다른 사람에게 매칭된 PPE 박스는 제외 (중복 매칭 방지)
                        if det_bbox_tuple in used_ppe_boxes:
                            continue
                        
                        # 중심점 거리 기반 판정 먼저 (IoU보다 빠름)
                        det_center_x = (dx1 + dx2) / 2
                        det_center_y = (dy1 + dy2) / 2
                        
                        # person_box 내부 포함 조건 추가 (다른 사람 PPE 오매칭 방지)
                        # PPE 박스의 중심점이 person_box 내부에 있어야 함
                        ppe_center_in_person = (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2)
                        if not ppe_center_in_person:
                            continue  # person_box 밖의 PPE는 무시
                        
                        center_distance = ((box_center_x - det_center_x) ** 2 + (box_center_y - det_center_y) ** 2) ** 0.5
                        
                        if center_distance < distance_threshold:
                            is_compliance = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            logging.debug(f"✅ PPE 준수 매칭 성공: {rule} - 거리={center_distance:.1f}, ppe_box={det_bbox_tuple}, person_box=({x1},{y1},{x2},{y2})")
                            break
                        
                        # 거리 임계값을 넘으면 IoU 계산 (더 정확하지만 느림)
                        iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                        if iou > ppe_iou_threshold:
                            is_compliance = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            logging.debug(f"✅ PPE 준수 매칭 성공: {rule} - IoU={iou:.4f}, ppe_box={det_bbox_tuple}, person_box=({x1},{y1},{x2},{y2})")
                            break
            
            # 위반(미착용) 판정: 중심점 거리 먼저 계산, IoU는 나중에 (성능 최적화)
            if viol_cls in all_detections and all_detections[viol_cls]:
                # 거리 임계값 미리 계산 (다른 사람 PPE 오매칭 방지: 더 엄격하게)
                box_diagonal = ((box_w ** 2 + box_h ** 2) ** 0.5)
                distance_threshold = box_diagonal * (0.6 if box_area < 10000 else 0.5)  # 1.3->0.6, 1.0->0.5 (다른 사람 PPE 오매칭 방지)
                
                for det in all_detections[viol_cls]:
                    if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                        dx1, dy1, dx2, dy2 = det['bbox']
                        det_bbox_tuple = (int(dx1), int(dy1), int(dx2), int(dy2))
                        
                        # 이미 다른 사람에게 매칭된 PPE 박스는 제외 (중복 매칭 방지)
                        if det_bbox_tuple in used_ppe_boxes:
                            continue
                        
                        # 중심점 거리 기반 판정 먼저 (IoU보다 빠름)
                        det_center_x = (dx1 + dx2) / 2
                        det_center_y = (dy1 + dy2) / 2
                        
                        # person_box 내부 포함 조건 추가 (다른 사람 PPE 오매칭 방지)
                        # PPE 박스의 중심점이 person_box 내부에 있어야 함
                        ppe_center_in_person = (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2)
                        if not ppe_center_in_person:
                            continue  # person_box 밖의 PPE는 무시
                        
                        center_distance = ((box_center_x - det_center_x) ** 2 + (box_center_y - det_center_y) ** 2) ** 0.5
                        
                        if center_distance < distance_threshold:
                            is_violation = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            break
                        
                        # 거리 임계값을 넘으면 IoU 계산 (더 정확하지만 느림)
                        iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                        if iou > ppe_iou_threshold:
                            is_violation = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            break
            
            # [수정] 준수 우선 정책 (Compliance Priority)
            # 착용(Compliance)이 감지되었다면, 위반(Violation) 감지가 있더라도 무시 (오탐지 방지)
            # 예: 조끼를 입었는데 조끼 주름 때문에 NO-Vest로 오인식되는 경우 방지
            if is_compliance:
                is_violation = False
                logging.debug(f"PPE 준수 감지: {rule} (Compliance Priority 적용, 위반 무시)")

            if is_violation:
                ppe_violations.append(rule)
                # 위반 감지는 중요하므로 info 레벨 유지 (단, 매 프레임마다 출력되지 않도록 조절 필요)
                logging.debug(f"PPE 위반 감지: {rule}")
        
        return ppe_violations, ppe_boxes
    except Exception as e:
        logging.warning(f"PPE 감지 처리 실패: {e}")
        return [], []


def _process_face_recognition(
    person_img_for_detection: np.ndarray, 
    person_id_text: str,
    face_model: Any, 
    face_database: Any,
    fast_recognizer: Optional[Any] = None,
    pre_detected_face: Optional[Any] = None,
    original_frame: Optional[np.ndarray] = None
) -> Tuple[str, float, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    얼굴 인식 처리 헬퍼 함수 (병렬 처리 가능, 최적화 버전)
    이제 이 함수는 임베딩 추출까지만 담당합니다.
    """
    try:
        import time
        import numpy as np
        
        # 0. Fast Path: 미리 감지된 얼굴 정보 사용 (중복 감지 제거)
        if pre_detected_face and original_frame is not None and fast_recognizer is not None:
            has_kps = hasattr(pre_detected_face, 'kps') and pre_detected_face.kps is not None
            if has_kps:
                try:
                    result = fast_recognizer.get_embedding_fast(
                        original_frame, 
                        pre_detected_face.kps
                    )
                    if result is not None:
                        embedding, _ = result
                        if embedding is not None:
                            face_bbox = tuple(map(int, pre_detected_face.bbox))
                            return "Unknown", 0.0, embedding, face_bbox
                except Exception as e:
                    logging.error(f"⚠️ {person_id_text} Fast Path 실패 (Fallback 진행): {e}", exc_info=True)

        # Fallback: YOLO로 다시 감지하여 랜드마크 추출 (측면 얼굴 지원)
        # 이미지 전처리 개선: 밝기/대비 조정 및 업스케일링
        import cv2
        img_h, img_w = person_img_for_detection.shape[:2]
        min_size = 64  # 최소 64x64 픽셀 (32 -> 64로 증가)
        
        # 이미지가 작으면 업스케일링 (더 큰 해상도로 감지 성공률 향상)
        processed_img = person_img_for_detection.copy()
        if img_h < min_size or img_w < min_size:
            scale = max(min_size / img_h, min_size / img_w)
            new_h, new_w = int(img_h * scale), int(img_w * scale)
            processed_img = cv2.resize(processed_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logging.debug(f"🔍 {person_id_text} 이미지 확대: {img_h}x{img_w} -> {new_h}x{new_w}")
        
        # 이미지 품질 개선: CLAHE (대비 향상) 및 가우시안 블러 제거 (샤프닝)
        # 작은 얼굴 감지를 위해 이미지 품질 향상
        if processed_img.shape[0] < 128 or processed_img.shape[1] < 128:
            # 작은 이미지는 CLAHE로 대비 향상
            lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed_img = cv2.merge([l, a, b])
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_LAB2BGR)
            logging.debug(f"🔍 {person_id_text} CLAHE 적용: 대비 향상")
        
        # confidence를 매우 낮게 설정하여 최대한 얼굴 감지 시도 (0.01 -> 0.005)
        # YOLO Face 모델은 640x640으로 ONNX 변환되었으므로 640만 사용
        yolo_results = None
        conf_levels = [0.005, 0.01, 0.02]  # 낮은 confidence부터 시도
        imgsz_options = [640]  # YOLO Face 모델은 640x640으로 고정 (ONNX 변환 시 해상도)
        
        for conf in conf_levels:
            for imgsz in imgsz_options:
                try:
                    yolo_results = face_model(processed_img, verbose=False, imgsz=imgsz, conf=conf)
                    if yolo_results and yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
                        logging.debug(f"🔍 {person_id_text} YOLO Face 감지 성공: conf={conf}, imgsz={imgsz}")
                        break
                except Exception as e:
                    logging.debug(f"🔍 {person_id_text} YOLO Face 시도 실패: conf={conf}, imgsz={imgsz}, error={e}")
                    continue
                if yolo_results and yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
                    break
            if yolo_results and yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
                break
        
        # 디버깅: YOLO Face 감지 결과 확인
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            box_count = len(result.boxes) if result.boxes is not None else 0
            has_keypoints = result.keypoints is not None
            kp_count = 0
            if has_keypoints and box_count > 0:
                try:
                    kp_count = len(result.keypoints.xy[0]) if len(result.keypoints.xy) > 0 else 0
                except:
                    pass
            logging.debug(f"🔍 {person_id_text} YOLO Face 결과: 박스={box_count}개, 키포인트={kp_count}개")
        
        if yolo_results and yolo_results[0].boxes:
            best_idx = yolo_results[0].boxes.xywh.prod(1).argmax()
            face_bbox_raw = yolo_results[0].boxes.xyxy[best_idx].cpu().numpy()
            face_bbox = tuple(map(int, face_bbox_raw))
            
            # 키포인트 추출 (측면 얼굴 지원: 2개 이상이면 사용)
            kps = None
            if yolo_results[0].keypoints is not None:
                try:
                    kps = yolo_results[0].keypoints.xy[best_idx].cpu().numpy()
                    # 측면 얼굴 지원: 최소 2개 키포인트만 있어도 처리
                    if kps.shape[0] < 2:
                        kps = None
                except Exception as e:
                    logging.debug(f"Fallback 키포인트 추출 실패: {e}")
                    kps = None
            
            # 키포인트가 있으면 사용, 없으면 얼굴 박스 기반 정렬 시도
            if kps is not None:
                result = fast_recognizer.get_embedding_fast(
                    processed_img, 
                    kps,
                    use_enhanced_preprocessing=False,  # aivis-project1 방식: 기본 전처리만 사용 (CLAHE)
                    use_tta=False  # 데이터베이스 구축 시와 동일 (USE_TTA_FOR_DATABASE=False)
                )
                if result is not None:
                    embedding, _ = result
                    if embedding is not None:
                        return "Unknown", 0.0, embedding, face_bbox
            else:
                # 키포인트가 없어도 얼굴 박스 기반으로 시도 (측면 얼굴)
                # 얼굴 박스 중심을 키포인트로 사용
                fx1, fy1, fx2, fy2 = face_bbox
                face_center = np.array([(fx1 + fx2) / 2, (fy1 + fy2) / 2], dtype=np.float32)
                face_size = max(fx2 - fx1, fy2 - fy1) * 0.3
                # 가상 키포인트 생성 (얼굴 박스 기반)
                fake_kps = np.array([
                    [face_center[0] - face_size, face_center[1] - face_size * 0.3],  # 왼쪽 눈 위치 추정
                    [face_center[0] + face_size, face_center[1] - face_size * 0.3],  # 오른쪽 눈 위치 추정
                    [face_center[0], face_center[1]],  # 코
                    [face_center[0] - face_size * 0.5, face_center[1] + face_size * 0.5],  # 왼쪽 입꼬리
                    [face_center[0] + face_size * 0.5, face_center[1] + face_size * 0.5],  # 오른쪽 입꼬리
                ], dtype=np.float32)
                
                result = fast_recognizer.get_embedding_fast(
                    processed_img, 
                    fake_kps
                )
                if result is not None:
                    embedding, _ = result
                    if embedding is not None:
                        logging.debug(f"측면 얼굴: 얼굴 박스 기반 임베딩 추출 성공")
                        return "Unknown", 0.0, embedding, face_bbox
        
        # 최종 시도: person_img_for_detection 전체 영역을 얼굴로 간주하고 시도
        # (YOLO Face가 실패해도 person_box 영역에서 얼굴을 찾을 수 있을 수 있음)
        if processed_img.shape[0] >= 32 and processed_img.shape[1] >= 32:
            try:
                # person_box의 상단 1/3 영역을 얼굴로 간주 (일반적인 얼굴 위치)
                face_region = processed_img[:processed_img.shape[0] // 3, :]
                if face_region.shape[0] >= 32 and face_region.shape[1] >= 32:
                    # 얼굴 영역 중심을 키포인트로 사용
                    face_center = np.array([face_region.shape[1] / 2, face_region.shape[0] / 2], dtype=np.float32)
                    face_size = min(face_region.shape[0], face_region.shape[1]) * 0.3
                    fake_kps = np.array([
                        [face_center[0] - face_size, face_center[1] - face_size * 0.3],
                        [face_center[0] + face_size, face_center[1] - face_size * 0.3],
                        [face_center[0], face_center[1]],
                        [face_center[0] - face_size * 0.5, face_center[1] + face_size * 0.5],
                        [face_center[0] + face_size * 0.5, face_center[1] + face_size * 0.5],
                    ], dtype=np.float32)
                    
                    result = fast_recognizer.get_embedding_fast(
                        face_region, 
                        fake_kps
                    )
                    if result is not None:
                        embedding, _ = result
                        if embedding is not None:
                            # 얼굴 영역의 bbox 계산 (원본 person_img 기준)
                            face_bbox = (0, 0, processed_img.shape[1], processed_img.shape[0] // 3)
                            logging.debug(f"최종 시도 성공: person_box 상단 영역 기반 임베딩 추출")
                            return "Unknown", 0.0, embedding, face_bbox
            except Exception as e:
                logging.debug(f"최종 시도 실패: {e}")
        
        raise FaceRecognitionError("얼굴 임베딩 추출 실패. 랜드마크를 찾을 수 없습니다.")

    except Exception as e:
        logging.error(f"❌ {person_id_text} 얼굴 인식 중 예상치 못한 오류: {e}", exc_info=True)
        raise FaceRecognitionError(f"얼굴 처리 중 예상치 못한 오류: {e}") from e


def _process_dangerous_behavior(
    keypoints: Keypoints, 
    person_box: Tuple[int, int, int, int], 
    cam_id: int, 
    person_box_key: str,
    person_crop: Optional[np.ndarray] = None,
    fall_model: Optional[Any] = None
) -> Tuple[bool, str]:
    """
    위험 행동 감지 처리 함수 (넘어짐 등) - 백그라운드에서 실행
    위험할 때만 True 반환하여 알림 생성
    
    Args:
        keypoints: 키포인트 객체
        person_box: 사람 바운딩 박스
        cam_id: 카메라 ID
        person_box_key: 사람 박스 키
        person_crop: 사람 영역 크롭 이미지 (FallSafe 모델용, 선택적)
        fall_model: FallSafe 모델 객체 (선택적)
    
    Returns:
        (is_dangerous, violation_type)
        - is_dangerous: 위험 행동 감지 여부 (위험할 때만 True)
        - violation_type: 위반 유형 (예: "넘어짐")
    """
    try:
        x1, y1, x2, y2 = person_box
        
        # 키포인트가 없어도 박스 비율로 넘어짐 감지 시도
        # 박스 비율이 매우 높으면 (가로가 세로보다 2배 이상) 넘어짐 후보
        box_w = x2 - x1
        box_h = y2 - y1
        box_ratio = box_w / box_h if box_h > 0 else 0
        
        if keypoints is None and box_ratio < 1.5:
            # 키포인트도 없고 박스 비율도 낮으면 스킵
            return False, ""
        
        is_fallen_horizontal = utils.is_person_horizontal(
            keypoints, (x1, y1, x2, y2),
            person_crop=person_crop,
            fall_model=fall_model
        )
        
        now_ts = time.time()
        
        # 디버깅: 키포인트 상태 확인 (dict 또는 Keypoints 객체 모두 지원)
        if keypoints is not None:
            try:
                if isinstance(keypoints, dict):
                    # dict 형태 (frame_processor에서 전달)
                    points = keypoints.get('xy')
                    confidences = keypoints.get('conf')
                    if points is not None and confidences is not None:
                        valid_count = (confidences > config.Thresholds.POSE_CONFIDENCE).sum()
                        logging.debug(f"🔍 위험 감지 분석 (dict): person_box_key={person_box_key}, "
                                    f"유효 키포인트={valid_count}/{len(confidences)}, "
                                    f"넘어짐 자세={is_fallen_horizontal}")
                elif hasattr(keypoints, 'data') and keypoints.data is not None:
                    # Keypoints 객체
                    points = keypoints.xy[0].cpu().numpy()
                    confidences = keypoints.conf[0].cpu().numpy()
                    valid_count = (confidences > config.Thresholds.POSE_CONFIDENCE).sum()
                    logging.debug(f"🔍 위험 감지 분석: person_box_key={person_box_key}, "
                            f"유효 키포인트={valid_count}/{len(confidences)}, "
                            f"넘어짐 자세={is_fallen_horizontal}")
            except Exception as e:
                logging.debug(f"키포인트 디버깅 오류: {e}")
        
        if is_fallen_horizontal:
            # 넘어짐 감지 시간 추적
            if cam_id not in fall_start_times:
                fall_start_times[cam_id] = {}
            if person_box_key not in fall_start_times[cam_id]:
                fall_start_times[cam_id][person_box_key] = now_ts
            
            # 0.5초 이상 지속되면 넘어짐으로 판정 (위험할 때만 True 반환)
            fall_duration = now_ts - fall_start_times[cam_id][person_box_key]
            if fall_duration >= FALL_DURATION_THRESHOLD:
                # 위험 감지: 알림 생성
                logging.warning(f"⚠️ 위험 행동 감지: {person_box_key} - 넘어짐 (지속 시간: {fall_duration:.2f}초)")
                return True, "넘어짐"
            else:
                # 아직 시간이 부족하면 False (위험하지 않음)
                return False, ""
        else:
            # 넘어짐이 아니면 시간 추적 초기화
            if cam_id in fall_start_times and person_box_key in fall_start_times[cam_id]:
                del fall_start_times[cam_id][person_box_key]
        
        return False, ""
    except Exception as e:
        logging.debug(f"위험 행동 감지 처리 오류: {e}")
        return False, ""
