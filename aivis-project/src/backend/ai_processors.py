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
    used_ppe_boxes: Optional[Set[Tuple[int, int, int, int]]] = None,
    person_id: Optional[str] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    PPE 감지 전용 함수 (얼굴 인식과 독립적으로 항상 실행)
    멀리 있는 사람도 잘 잡기 위해 최고 성능 설정
    
    Args:
        person_box: 사람 바운딩 박스 (x1, y1, x2, y2)
        all_detections: 모든 감지 결과 딕셔너리
        used_ppe_boxes: 이미 사용된 PPE 박스 집합
        person_id: 사람 식별자 (로깅용)
    
    Returns:
        ppe_violations: PPE 위반 목록 (예: ["안전모"])
        ppe_boxes: PPE 감지된 박스 정보 리스트 [{"bbox": (x1,y1,x2,y2), "class": "Safety Vest", "conf": 0.9}, ...]
    """
    ppe_violations = []
    ppe_boxes: List[Dict[str, Any]] = []  # PPE 감지 박스 정보
    person_id_text = person_id or "UNKNOWN"
    
    try:
        x1, y1, x2, y2 = person_box
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        logging.debug(f"[PPE {person_id_text}] PPE 감지 시작: person_box=({x1},{y1},{x2},{y2}), "
                     f"크기={box_w}x{box_h}, 면적={box_area:.0f}px²")
        
        # 멀리 있는 사람(작은 박스)을 위한 동적 IoU 임계값 조정
        # 착용한 PPE 감지와 오탐지 방지의 균형 유지
        if box_area < 5000:  # 매우 작은 박스 (멀리 있는 사람)
            ppe_iou_threshold = 0.05  # 착용한 PPE 감지 개선: 0.10 -> 0.05
        elif box_area < 10000:  # 작은 박스
            ppe_iou_threshold = 0.08  # 착용한 PPE 감지 개선: 0.15 -> 0.08
        elif box_area < 20000:  # 중간 박스
            ppe_iou_threshold = 0.12  # 착용한 PPE 감지 개선: 0.20 -> 0.12
        else:
            ppe_iou_threshold = 0.15  # 일반 박스 (착용한 PPE 감지 개선: 0.25 -> 0.15)
        
        relaxed_iou_threshold = ppe_iou_threshold * 0.5  # 완화된 임계값 미리 계산
        
        logging.debug(f"[PPE {person_id_text}] IoU 임계값 설정: 기본={ppe_iou_threshold:.3f}, "
                     f"완화={relaxed_iou_threshold:.3f}, 박스면적={box_area:.0f}px²")
        
        # 이미 사용된 PPE 박스 추적 (중복 매칭 방지)
        if used_ppe_boxes is None:
            used_ppe_boxes = set()
        
        # 모든 PPE 클래스 수집 (준수 및 위반 모두) - 마스크 제외
        ppe_class_names = set()
        for rule, classes in config.Constants.SAFETY_RULES_MAP.items():
            # 마스크는 제외
            if rule == "마스크":
                continue
            ppe_class_names.add(classes["compliance"])
            ppe_class_names.add(classes["violation"])
        
        # person_box와 겹치는 모든 PPE 박스 수집 (준수 클래스만, 위반 클래스는 위반 판정 단계에서 처리)
        compliance_classes = set()
        violation_classes = set()
        for rule, classes in config.Constants.SAFETY_RULES_MAP.items():
            if rule == "마스크":
                continue
            compliance_classes.add(classes["compliance"])
            violation_classes.add(classes["violation"])
        
        for ppe_class in ppe_class_names:
            # 위반 클래스는 위반 판정 단계에서만 처리 (중복 매칭 방지)
            if ppe_class in violation_classes:
                continue
                
            if ppe_class in all_detections and all_detections[ppe_class]:
                detection_count = len(all_detections[ppe_class])
                logging.debug(f"[PPE {person_id_text}] {ppe_class} 클래스 감지: {detection_count}개")
                
                # 거리 임계값 미리 계산 (착용한 PPE 감지와 오탐지 방지의 균형)
                box_diagonal = ((box_w ** 2 + box_h ** 2) ** 0.5)
                # 거리 임계값 조정 (착용한 PPE 감지 개선: 0.4/0.3 -> 0.5/0.4)
                distance_threshold = box_diagonal * (0.5 if box_area < 10000 else 0.4)
                
                for det_idx, det in enumerate(all_detections[ppe_class]):
                    if det and 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                        dx1, dy1, dx2, dy2 = det['bbox']
                        det_bbox_tuple = (int(dx1), int(dy1), int(dx2), int(dy2))
                        
                        # 이미 다른 사람에게 매칭된 PPE 박스는 제외 (중복 매칭 방지)
                        if det_bbox_tuple in used_ppe_boxes:
                            logging.debug(f"[PPE {person_id_text}] {ppe_class}[{det_idx}] 스킵: 이미 사용된 박스")
                            continue
                        
                        conf = det.get('conf', 0.9)
                        
                        # PPE 박스 크기 및 위치 검증 추가 (오탐지 방지)
                        det_w = dx2 - dx1
                        det_h = dy2 - dy1
                        det_area = det_w * det_h
                        det_center_x = (dx1 + dx2) / 2
                        det_center_y = (dy1 + dy2) / 2
                        
                        # PPE 박스가 너무 작거나 너무 크면 제외 (오탐지 방지, 하지만 완화)
                        # 사람 박스의 0.5% 미만이거나 60% 초과면 제외 (착용한 PPE 감지 개선)
                        min_ppe_area = box_area * 0.005  # 최소 0.5% (1% -> 0.5%, 작은 PPE도 감지)
                        max_ppe_area = box_area * 0.60  # 최대 60% (50% -> 60%, 큰 PPE도 감지)
                        area_ratio = (det_area / box_area) * 100 if box_area > 0 else 0
                        
                        if det_area < min_ppe_area or det_area > max_ppe_area:
                            logging.debug(f"[PPE {person_id_text}] {ppe_class}[{det_idx}] 스킵: 크기 범위 초과 "
                                        f"(면적={det_area:.0f}px², 비율={area_ratio:.2f}%, "
                                        f"범위={min_ppe_area:.0f}~{max_ppe_area:.0f}px²)")
                            continue
                        
                        # person_box 내부 포함 조건 추가 (다른 사람 PPE 오매칭 방지, 의자 등 오탐지 방지)
                        # PPE 박스의 중심점이 person_box 내부에 있어야 함
                        ppe_center_in_person = (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2)
                        if not ppe_center_in_person:
                            logging.debug(f"[PPE {person_id_text}] {ppe_class}[{det_idx}] 스킵: person_box 밖의 PPE (중심점=({det_center_x:.1f},{det_center_y:.1f}), person_box=({x1},{y1},{x2},{y2}))")
                            continue  # person_box 밖의 PPE는 무시 (의자 등 오탐지 방지)
                        
                        # 중심점 거리 계산
                        center_distance = ((box_center_x - det_center_x) ** 2 + (box_center_y - det_center_y) ** 2) ** 0.5
                        
                        # IoU 계산
                        iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                        
                        is_match = False
                        match_reason = ""
                        
                        # 중심점 거리 기반 판정 먼저 (IoU보다 빠름)
                        if center_distance < distance_threshold:
                            is_match = True
                            match_reason = f"거리근접({center_distance:.1f}<{distance_threshold:.1f})"
                        else:
                            # 거리 임계값을 넘으면 IoU 계산 (더 정확하지만 느림)
                            if iou > ppe_iou_threshold:
                                is_match = True
                                match_reason = f"IoU({iou:.3f}>{ppe_iou_threshold:.3f})"
                            else:
                                match_reason = f"IoU부족({iou:.3f}<={ppe_iou_threshold:.3f})"
                        
                        logging.debug(f"[PPE {person_id_text}] {ppe_class}[{det_idx}] 평가: "
                                    f"bbox=({dx1},{dy1},{dx2},{dy2}), conf={conf:.3f}, "
                                    f"IoU={iou:.3f}, 중심거리={center_distance:.1f}px, "
                                    f"면적비율={area_ratio:.2f}%, 매칭={'✅' if is_match else '❌'}({match_reason})")
                        
                        if is_match:
                            # PPE 박스 정보 저장 및 사용된 박스로 표시
                            ppe_boxes.append({
                                "bbox": det_bbox_tuple,
                                "class": ppe_class,
                                "conf": conf
                            })
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            logging.info(f"[PPE {person_id_text}] ✅ {ppe_class} 매칭 성공: "
                                       f"bbox=({dx1},{dy1},{dx2},{dy2}), conf={conf:.3f}, IoU={iou:.3f}")
        
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
        
        # 위반 판정 로직 - 마스크 제외
        logging.debug(f"[PPE {person_id_text}] 위반 판정 시작: 규칙 수={len([r for r in config.Constants.SAFETY_RULES_MAP.keys() if r != '마스크'])}")
        
        for rule, classes in config.Constants.SAFETY_RULES_MAP.items():
            # 마스크는 제외
            if rule == "마스크":
                continue
            comp_cls, viol_cls = classes["compliance"], classes["violation"]
            is_compliance = False
            is_violation = False
            
            logging.debug(f"[PPE {person_id_text}] 규칙 '{rule}' 평가 시작: "
                        f"준수클래스={comp_cls}, 위반클래스={viol_cls}")
            
            # 준수(착용) 판정: 중심점 거리 먼저 계산, IoU는 나중에 (성능 최적화)
            if comp_cls in all_detections and all_detections[comp_cls]:
                # 거리 임계값 미리 계산 (다른 사람 PPE 오매칭 방지: 더 엄격하게)
                box_diagonal = ((box_w ** 2 + box_h ** 2) ** 0.5)
                distance_threshold = box_diagonal * (0.6 if box_area < 10000 else 0.5)  # 0.5->0.6, 0.4->0.5 (다른 사람 PPE 오매칭 방지)
                
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
                        
                        # person_box 내부 포함 조건 추가 (다른 사람 PPE 오매칭 방지, 의자 등 오탐지 방지)
                        # PPE 박스의 중심점이 person_box 내부에 있어야 함
                        ppe_center_in_person = (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2)
                        if not ppe_center_in_person:
                            continue  # person_box 밖의 PPE는 무시
                        
                        center_distance = ((box_center_x - det_center_x) ** 2 + (box_center_y - det_center_y) ** 2) ** 0.5
                        
                        if center_distance < distance_threshold:
                            is_compliance = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            logging.debug(f"[PPE {person_id_text}] ✅ PPE 준수 매칭 성공: {rule} - 거리={center_distance:.1f}, ppe_box={det_bbox_tuple}, person_box=({x1},{y1},{x2},{y2})")
                            break
                        
                        # 거리 임계값을 넘으면 IoU 계산 (더 정확하지만 느림)
                        iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                        if iou > ppe_iou_threshold:
                            is_compliance = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            logging.debug(f"[PPE {person_id_text}] ✅ PPE 준수 매칭 성공: {rule} - IoU={iou:.4f}, ppe_box={det_bbox_tuple}, person_box=({x1},{y1},{x2},{y2})")
                            break
            
            # 위반(미착용) 판정: 중심점 거리 먼저 계산, IoU는 나중에 (성능 최적화)
            if viol_cls in all_detections and all_detections[viol_cls]:
                # 거리 임계값 미리 계산 (다른 사람 PPE 오매칭 방지: 더 엄격하게)
                box_diagonal = ((box_w ** 2 + box_h ** 2) ** 0.5)
                distance_threshold = box_diagonal * (0.6 if box_area < 10000 else 0.5)  # 0.5->0.6, 0.4->0.5 (다른 사람 PPE 오매칭 방지)
                
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
                        
                        # person_box 내부 포함 조건 추가 (다른 사람 PPE 오매칭 방지, 의자 등 오탐지 방지)
                        # PPE 박스의 중심점이 person_box 내부에 있어야 함
                        ppe_center_in_person = (x1 <= det_center_x <= x2 and y1 <= det_center_y <= y2)
                        if not ppe_center_in_person:
                            continue  # person_box 밖의 PPE는 무시
                        
                        center_distance = ((box_center_x - det_center_x) ** 2 + (box_center_y - det_center_y) ** 2) ** 0.5
                        
                        if center_distance < distance_threshold:
                            is_violation = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            # 위반 클래스도 ppe_boxes에 추가 (로깅 및 시각화용)
                            ppe_boxes.append({
                                "bbox": det_bbox_tuple,
                                "class": viol_cls,
                                "conf": det.get('conf', 0.9)
                            })
                            logging.info(f"[PPE {person_id_text}] ✅ {viol_cls} 위반 매칭 성공: "
                                       f"bbox=({dx1},{dy1},{dx2},{dy2}), conf={det.get('conf', 0.9):.3f}, 거리={center_distance:.1f}")
                            break
                        
                        # 거리 임계값을 넘으면 IoU 계산 (더 정확하지만 느림)
                        iou = utils.calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                        if iou > ppe_iou_threshold:
                            is_violation = True
                            used_ppe_boxes.add(det_bbox_tuple)  # 중복 매칭 방지
                            # 위반 클래스도 ppe_boxes에 추가 (로깅 및 시각화용)
                            ppe_boxes.append({
                                "bbox": det_bbox_tuple,
                                "class": viol_cls,
                                "conf": det.get('conf', 0.9)
                            })
                            logging.info(f"[PPE {person_id_text}] ✅ {viol_cls} 위반 매칭 성공: "
                                       f"bbox=({dx1},{dy1},{dx2},{dy2}), conf={det.get('conf', 0.9):.3f}, IoU={iou:.3f}")
                            break
            
            # [수정] 준수 우선 정책 (Compliance Priority)
            # 착용(Compliance)이 감지되었다면, 위반(Violation) 감지가 있더라도 무시 (오탐지 방지)
            # 예: 조끼를 입었는데 조끼 주름 때문에 NO-Vest로 오인식되는 경우 방지
            if is_compliance:
                is_violation = False
                logging.debug(f"[PPE {person_id_text}] PPE 준수 감지: {rule} (Compliance Priority 적용, 위반 무시)")

            if is_violation:
                ppe_violations.append(rule)
                # 위반 감지는 중요하므로 info 레벨 유지 (단, 매 프레임마다 출력되지 않도록 조절 필요)
                logging.debug(f"[PPE {person_id_text}] PPE 위반 감지: {rule}")
        
        logging.info(f"[PPE {person_id_text}] ✅ PPE 감지 완료: 위반={len(ppe_violations)}개{ppe_violations if ppe_violations else ''}, "
                    f"매칭된 PPE 박스={len(ppe_boxes)}개")
        
        return ppe_violations, ppe_boxes
    except Exception as e:
        logging.warning(f"PPE 감지 처리 실패: {e}")
        return [], []


def _process_face_recognition(
    person_img_for_detection: np.ndarray, 
    person_id_text: str,
    face_model: Any, 
    face_analyzer: Any, 
    face_database: Any,
    use_adaface: bool = False,
    adaface_model_path: Optional[str] = None,
    fast_recognizer: Optional[Any] = None,
    pre_detected_face: Optional[Any] = None,
    original_frame: Optional[np.ndarray] = None,
    face_uses_trt: bool = False
) -> Tuple[str, float, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    얼굴 인식 처리 헬퍼 함수 (병렬 처리 가능, 최적화 버전)
    """
    
    try:
        import time
        import cv2
        import numpy as np
        import config
        from utils import find_best_match_faiss
        
        # 0. Fast Path: 미리 감지된 얼굴 정보 사용 (중복 감지 제거)
        # frame_processor에서 이미 얼굴을 찾았으므로, 랜드마크를 이용해 바로 임베딩 추출
        if pre_detected_face and original_frame is not None and fast_recognizer is not None:
            has_kps = hasattr(pre_detected_face, 'kps') and pre_detected_face.kps is not None
            
            if has_kps:
                try:
                    # 원본 프레임과 절대 좌표 랜드마크 사용 -> 화질 저하 없음, 속도 최상
                    result = fast_recognizer.get_embedding_fast(original_frame, pre_detected_face.kps)
                    
                    if result is not None:
                        embedding, aligned_face = result
                        if embedding is not None:
                            # 정규화
                            embedding = embedding / np.linalg.norm(embedding)
                            
                            # FAISS 매칭
                            adaptive_threshold = config.Thresholds.SIMILARITY
                            faiss_start = time.time()
                            person_name, similarity_score = find_best_match_faiss(
                                embedding, face_database, adaptive_threshold
                            )
                            faiss_time = (time.time() - faiss_start) * 1000  # ms
                            
                            logging.info(f"[FACE {person_id_text}] [Fast Path] FAISS 검색 완료: {faiss_time:.1f}ms, "
                                       f"결과={person_name}, 유사도={similarity_score:.3f}, "
                                       f"임계값={adaptive_threshold:.3f}, 통과={'✅' if similarity_score >= adaptive_threshold else '❌'}")
                            
                            # bbox는 정수형 튜플로 변환
                            face_bbox = tuple(map(int, pre_detected_face.bbox))
                            return person_name, similarity_score, embedding, face_bbox
                except Exception:
                    pass
        
        # ---------------------------------------------------------
        # Fallback: YOLO로 다시 감지하여 랜드마크 추출 (측면 얼굴 지원, MPS 최적화)
        # ---------------------------------------------------------
        
        # 이미지 전처리 개선: 밝기/대비 조정 및 업스케일링 (MPS 최적화)
        import cv2
        img_h, img_w = person_img_for_detection.shape[:2]
        min_size = 64  # 최소 64x64 픽셀 (MPS 최적화: 32 -> 64로 증가)
        
        # 이미지가 작으면 업스케일링 (더 큰 해상도로 감지 성공률 향상)
        processed_img = person_img_for_detection.copy()
        if img_h < min_size or img_w < min_size:
            scale = max(min_size / img_h, min_size / img_w)
            new_h, new_w = int(img_h * scale), int(img_w * scale)
            processed_img = cv2.resize(processed_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logging.debug(f"🔍 {person_id_text} 이미지 확대: {img_h}x{img_w} -> {new_h}x{new_w}")
        
        # 이미지 품질 개선: CLAHE (대비 향상) - 작은 얼굴 감지 개선
        if processed_img.shape[0] < 128 or processed_img.shape[1] < 128:
            # 작은 이미지는 CLAHE로 대비 향상
            lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed_img = cv2.merge([l, a, b])
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_LAB2BGR)
            logging.debug(f"🔍 {person_id_text} CLAHE 적용: 대비 향상")
        
        h, w = processed_img.shape[:2]
        min_size_check = config.Thresholds.MIN_FACE_SIZE  # 최소 16픽셀 이상
        if h < min_size_check or w < min_size_check:
            return "Unknown", 0.0, None, None
        
        # confidence를 매우 낮게 설정하여 최대한 얼굴 감지 시도 (MPS 최적화)
        # YOLO Face 모델은 640x640으로 ONNX 변환되었으므로 640만 사용
        face_start_time = time.time()
        if face_model is None:
            logging.warning(f"{person_id_text} 얼굴 모델이 None입니다")
            return "Unknown", 0.0, None, None
        
        yolo_results = None
        conf_levels = [0.005, 0.01, 0.02]  # 낮은 confidence부터 시도 (MPS 최적화)
        imgsz_options = [640]  # YOLO Face 모델은 640x640으로 고정 (MPS 최적화)
        
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
        
        face_detection_time = time.time() - face_start_time
        
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
        
        # YOLO 얼굴 감지 결과 처리 (MPS 최적화)
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
            
            # 키포인트가 있으면 사용, 없으면 얼굴 박스 기반 정렬 시도 (측면 얼굴 지원)
            if fast_recognizer is not None:
                if kps is not None:
                    result = fast_recognizer.get_embedding_fast(
                        processed_img, 
                        kps
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
        if fast_recognizer is not None and processed_img.shape[0] >= 32 and processed_img.shape[1] >= 32:
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
        
        # 모든 시도 실패 시 기존 InsightFace 방식으로 폴백
        # YOLO 결과가 있고 fast_recognizer로 처리하지 못한 경우에만 실행
        if yolo_results and yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
            # YOLO 결과가 있으면 기존 방식으로 처리 (InsightFace fallback)
            boxes = yolo_results[0].boxes
            best_idx = boxes.xywh.prod(1).argmax()
            face_bbox_raw = boxes.xyxy[best_idx].cpu().numpy()
            fx1, fy1, fx2, fy2 = int(face_bbox_raw[0]), int(face_bbox_raw[1]), int(face_bbox_raw[2]), int(face_bbox_raw[3])
            face_bbox = (fx1, fy1, fx2, fy2)
            
            # 2단계: 얼굴 영역 추출 (얼굴 자르기) - 실시간 처리 최적화
            # 패딩 최소화: 배경 포함을 줄여 임베딩 품질 향상 (5% 패딩으로 감소)
            face_w = fx2 - fx1
            face_h = fy2 - fy1
            
            # 동적 패딩: 작은 얼굴일수록 패딩을 늘려 이마/턱 포함률 향상 (인식률 최대화)
            # 짧은 변이 60px 미만이면 15%, 아니면 8% (더 많은 얼굴 영역 포함)
            padding_ratio = 0.15 if min(face_w, face_h) < 60 else 0.08
            padding_w = int(face_w * padding_ratio)
            padding_h = int(face_h * padding_ratio)
            
            # 경계 체크 및 패딩 적용 (반드시 int로 변환)
            h, w = processed_img.shape[:2]
            fx1_padded = int(max(0, fx1 - padding_w))
            fy1_padded = int(max(0, fy1 - padding_h))
            fx2_padded = int(min(w, fx2 + padding_w))
            fy2_padded = int(min(h, fy2 + padding_h))
            
            # 얼굴 영역 추출 (패딩 포함)
            face_img = processed_img[fy1_padded:fy2_padded, fx1_padded:fx2_padded]
            if face_img.size == 0:
                return "Unknown", 0.0, None, None
            
            # 화질 개선: 크롭 후 이미지 품질 향상 (인식률 개선)
            # 1. 노이즈 제거 (Bilateral Filter) - 너무 느려서 제거 (GPU로 처리하거나 생략)
            # face_img = cv2.bilateralFilter(face_img, d=5, sigmaColor=50, sigmaSpace=50)
            
            # 2. 대비 향상 (CLAHE: Contrast Limited Adaptive Histogram Equalization) - 빠르고 효과 좋음 (유지)
            # LAB 색공간으로 변환하여 밝기 채널만 처리 (색상 왜곡 방지)
            try:
                lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                face_img = cv2.merge([l_channel, a, b])
                face_img = cv2.cvtColor(face_img, cv2.COLOR_LAB2BGR)
            except Exception:
                pass # 오류 발생 시 원본 사용
            
            # 3. 샤프닝 (Unsharp Masking) - 느려서 제거
            # gaussian = cv2.GaussianBlur(face_img, (0, 0), 2.0)
            # face_img = cv2.addWeighted(face_img, 1.5, gaussian, -0.5, 0)
            
            # logging.info(f"🔍 {person_id_text} 화질 개선 완료: 노이즈 제거 + 대비 향상 + 샤프닝")
            
            # 최소 얼굴 크기 확인 및 리사이즈 (너무 작으면 리사이즈하여 처리)
            min_face_size = 32  # 최소 얼굴 크기 (40 -> 32로 완화: 더 많은 얼굴 인식)
            if face_img.shape[0] < min_face_size or face_img.shape[1] < min_face_size:
                # 너무 작은 얼굴은 최소 크기로 리사이즈 (비율 유지)
                scale = min_face_size / min(face_img.shape[0], face_img.shape[1])
                new_h = max(min_face_size, int(face_img.shape[0] * scale))
                new_w = max(min_face_size, int(face_img.shape[1] * scale))
                # 작은 얼굴 업스케일은 속도를 위해 Linear 사용 (LANCZOS4 -> LINEAR)
                face_img = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # logging.debug(f"🔍 {person_id_text} 얼굴 리사이즈 (작은 얼굴): {face_img.shape[1]}x{face_img.shape[0]}")
            
            # InsightFace 최적 크기로 리사이즈 (112x112 권장)
            # 모든 얼굴 이미지를 112x112로 리사이즈하여 일관성 유지 (InsightFace 최적 크기)
            target_size = 112  # InsightFace buffalo_L 모델 최적 크기
            if face_img.shape[0] != target_size or face_img.shape[1] != target_size:
                # 모든 얼굴을 112x112로 리사이즈 (비율 유지하지 않고 정확히 112x112)
                # 속도를 위해 Linear 사용 (LANCZOS4 -> LINEAR)
                face_img = cv2.resize(face_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                # logging.debug(f"🔍 {person_id_text} 얼굴 리사이즈: {face_img.shape[1]}x{face_img.shape[0]}")
            
            # 3단계: buffalo_L 모델(InsightFace)로 얼굴 임베딩 추출
            embedding_start_time = time.time()
            if face_analyzer is None:
                logging.warning(f"{person_id_text} InsightFace 분석기가 None입니다")
                return "Unknown", 0.0, None, face_bbox
            if face_database is None:
                logging.error(f"{person_id_text} FAISS 데이터베이스가 None입니다 - 얼굴 인식 불가능!")
                return "Unknown", 0.0, None, face_bbox
            
            # InsightFace recognition 모듈로 임베딩 추출 (buffalo_L 모델 사용)
            # 이미 크롭된 얼굴 이미지이므로 rec_model을 직접 사용해야 함
            embedding: Optional[np.ndarray] = None
            try:
                # rec_model 접근 방법: face_analyzer.models['recognition'] 또는 face_analyzer.rec_model
                rec_model = None
                if hasattr(face_analyzer, 'models') and 'recognition' in face_analyzer.models:
                    rec_model = face_analyzer.models['recognition']
                elif hasattr(face_analyzer, 'rec_model'):
                    rec_model = face_analyzer.rec_model
                
                if rec_model is not None:
                    # 이미 크롭된 얼굴 이미지에 대해 직접 임베딩 추출
                    # face_img는 이미 112x112로 리사이즈된 얼굴 이미지
                    embedding = rec_model.get_feat(face_img)
                    if embedding is not None:
                        # 정규화 (L2 norm)
                        embedding = embedding / np.linalg.norm(embedding)
                    else:
                        logging.warning(f"⚠️ {person_id_text} rec_model.get_feat() 반환값이 None입니다")
                        return "Unknown", 0.0, None, face_bbox
                else:
                    # rec_model을 찾을 수 없는 경우 fallback: get() 메서드 사용
                    # 하지만 이미 크롭된 이미지이므로 실패할 가능성이 높음
                    logging.warning(f"⚠️ {person_id_text} rec_model을 찾을 수 없음. face_analyzer.get() 사용 (fallback, 실패 가능성 높음)")
                    faces = face_analyzer.get(face_img)
                    if faces and len(faces) > 0:
                        biggest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        embedding = biggest_face.normed_embedding
                    else:
                        logging.warning(f"⚠️ {person_id_text} face_analyzer.get() 실패: 얼굴 감지 결과 없음 (이미 크롭된 이미지이므로 정상)")
                        return "Unknown", 0.0, None, face_bbox
            except Exception as e:
                logging.error(f"❌ {person_id_text} InsightFace 임베딩 추출 실패: {e}", exc_info=True)
                raise FaceRecognitionError(
                    f"임베딩 추출 실패: {e}",
                    error_code="EMBEDDING_EXTRACTION_FAILED",
                    details={"person_id": person_id_text}
                ) from e
            
            embedding_time = time.time() - embedding_start_time
            
            # 임베딩 추출 결과 상세 로깅
            if embedding is not None and embedding.size > 0:
                embedding_norm = np.linalg.norm(embedding)
                embedding_dim = embedding.shape[0] if embedding.ndim == 1 else embedding.shape[1]
                logging.info(f"[FACE {person_id_text}] ✅ 임베딩 추출 성공: {embedding_time*1000:.1f}ms, "
                           f"shape={embedding.shape}, dim={embedding_dim}, norm={embedding_norm:.3f}, "
                           f"얼굴 크기={face_img.shape[:2]}")
            else:
                logging.warning(f"[FACE {person_id_text}] ❌ 임베딩 추출 실패: embedding=None 또는 빈 배열")
            
            # 임베딩 추출 확인 로깅
            if embedding is None or embedding.size == 0:
                raise FaceRecognitionError(
                    f"임베딩이 None이거나 비어있음",
                    error_code="EMPTY_EMBEDDING",
                    details={"person_id": person_id_text}
                )
            
            # 4단계: FAISS 인덱스(face_index.faiss)와 레이블(face_index.faiss.labels.npy)을 사용하여 매칭
            # face_database는 튜플 (index, labels) 형태
            try:
                if face_database is None:
                    raise FaceRecognitionError(
                        "FAISS 데이터베이스가 None입니다",
                        error_code="FAISS_DATABASE_NONE",
                        details={"person_id": person_id_text}
                    )
                
                # face_database가 튜플인지 확인
                if isinstance(face_database, tuple):
                    faiss_index, faiss_labels = face_database
                    if faiss_index is None:
                        raise FaceRecognitionError(
                            "FAISS 인덱스가 None입니다",
                            error_code="FAISS_INDEX_NONE",
                            details={"person_id": person_id_text}
                        )
                    if not hasattr(faiss_index, 'ntotal'):
                        raise FaceRecognitionError(
                            "FAISS 인덱스에 ntotal 속성이 없습니다",
                            error_code="FAISS_INDEX_INVALID",
                            details={"person_id": person_id_text}
                        )
                else:
                    raise FaceRecognitionError(
                        f"FAISS 데이터베이스가 튜플이 아닙니다: {type(face_database)}",
                        error_code="FAISS_DATABASE_INVALID",
                        details={"person_id": person_id_text}
                    )
                
                # 작은 얼굴은 약간 낮춘 임계값 적용 (오인식 방지와 인식률 균형)
                base_threshold = config.Thresholds.SIMILARITY
                adaptive_threshold = base_threshold
                fh, fw = face_img.shape[:2]
                face_min_size = min(fh, fw)
                # 작은 얼굴에만 제한적으로 낮춘 임계값 적용 (오인식 방지 강화)
                if face_min_size < 80:
                    # 작은 얼굴일수록 약간 낮은 임계값 적용 (오인식 방지 강화)
                    if face_min_size < 50:
                        adaptive_threshold = max(0.32, adaptive_threshold - 0.04)  # 매우 작은 얼굴: -0.04 (최소 0.32 유지, 0.28 -> 0.32)
                    elif face_min_size < 65:
                        adaptive_threshold = max(0.32, adaptive_threshold - 0.02)  # 작은 얼굴: -0.02 (최소 0.32 유지, 0.30 -> 0.32)
                    else:
                        adaptive_threshold = max(0.30, adaptive_threshold)  # 중간 크기: 기본값 유지 (0.30)
                
                logging.info(f"[FACE {person_id_text}] FAISS 검색 시작: 임계값={adaptive_threshold:.3f} "
                           f"(기본={base_threshold:.3f}, 얼굴크기={face_min_size}px)")
                
                faiss_start = time.time()
                person_name, similarity_score = find_best_match_faiss(
                    embedding, face_database, adaptive_threshold
                )
                faiss_time = (time.time() - faiss_start) * 1000  # ms
                
                logging.info(f"[FACE {person_id_text}] FAISS 검색 완료: {faiss_time:.1f}ms, "
                           f"결과={person_name}, 유사도={similarity_score:.3f}, "
                           f"임계값={adaptive_threshold:.3f}, 통과={'✅' if similarity_score >= adaptive_threshold else '❌'}")
            except FaceRecognitionError:
                # 이미 FaceRecognitionError이면 그대로 전파
                raise
            except Exception as e:
                logging.error(f"❌ {person_id_text} FAISS 매칭 중 예외 발생: {e}", exc_info=True)
                raise FaceRecognitionError(
                    f"FAISS 매칭 실패: {e}",
                    error_code="FAISS_MATCHING_FAILED",
                    details={"person_id": person_id_text}
                ) from e
            
            # 얼굴 인식 결과 로깅 (성능 최적화: DEBUG 레벨로 변경)
            if person_name == "Unknown":
                pass
                # logging.debug(f"⚠️ {person_id_text} 얼굴 인식 실패: Unknown (유사도={similarity_score:.3f})")
            else:
                pass
                # logging.debug(f"✅ {person_id_text} 얼굴 인식 성공: {person_name} (유사도={similarity_score:.3f})")
            
            total_time = face_detection_time + embedding_time
            if total_time > 1.0:  # 1.0초 이상 걸린 경우만 로깅 (0.5 -> 1.0)
                logging.warning(f"{person_id_text} 얼굴 인식 시간: {total_time:.3f}s (YOLO: {face_detection_time:.3f}s, Embedding: {embedding_time:.3f}s) -> {person_name}")
            
            return person_name, similarity_score, embedding, face_bbox
        else:
            logging.warning(f"⚠️ {person_id_text} 얼굴 감지 실패: YOLO 결과 없음 또는 박스 없음 (yolo_results={yolo_results is not None}, len={len(yolo_results) if yolo_results else 0}, boxes={len(yolo_results[0].boxes) if yolo_results and len(yolo_results) > 0 else 0})")
            return "Unknown", 0.0, None, None
    except FaceRecognitionError:
        # FaceRecognitionError는 그대로 전파 (호출자가 처리)
        raise
    except Exception as e:
        logging.warning(f"{person_id_text} 얼굴 인식 처리 실패: {e}", exc_info=True)
        # 예상치 못한 오류는 ProcessingError로 변환
        raise ProcessingError(
            f"얼굴 인식 처리 중 예상치 못한 오류: {e}",
            error_code="FACE_RECOGNITION_UNEXPECTED_ERROR",
            details={"person_id": person_id_text}
        ) from e


def _process_dangerous_behavior(
    keypoints: Keypoints, 
    person_box: Tuple[int, int, int, int], 
    cam_id: int, 
    person_box_key: str
) -> Tuple[bool, str]:
    """
    위험 행동 감지 처리 함수 (넘어짐 등) - 백그라운드에서 실행
    위험할 때만 True 반환하여 알림 생성
    
    Returns:
        (is_dangerous, violation_type)
        - is_dangerous: 위험 행동 감지 여부 (위험할 때만 True)
        - violation_type: 위반 유형 (예: "넘어짐")
    """
    try:
        if keypoints is None:
            return False, ""
        
        x1, y1, x2, y2 = person_box
        is_fallen_horizontal = utils.is_person_horizontal(keypoints, (x1, y1, x2, y2))
        
        now_ts = time.time()
        
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
    except Exception:
        return False, ""

