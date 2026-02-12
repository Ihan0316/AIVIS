"""
간단한 유틸리티 함수들
얼굴 임베딩 작업에 필요한 함수들만 포함
"""
from typing import Tuple, List


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """두 바운딩 박스의 IoU (Intersection over Union)를 계산합니다."""
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def remove_duplicate_faces(faces: List, iou_threshold: float = 0.4, min_face_size: int = 40) -> List:
    """
    중복된 얼굴을 제거합니다.
    
    Args:
        faces: 얼굴 객체 리스트
        iou_threshold: IoU 임계값 (기본값: 0.4)
        min_face_size: 최소 얼굴 크기 (기본값: 40)
    
    Returns:
        중복이 제거된 얼굴 리스트
    """
    if not faces or len(faces) <= 1:
        return faces
    
    # 얼굴 크기로 필터링
    valid_faces = []
    for face in faces:
        try:
            bbox = face.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width >= min_face_size and height >= min_face_size:
                valid_faces.append(face)
        except (AttributeError, IndexError):
            continue
    
    if len(valid_faces) <= 1:
        return valid_faces
    
    # IoU 기반 중복 제거
    filtered_faces = []
    for i, face1 in enumerate(valid_faces):
        is_duplicate = False
        try:
            bbox1 = face1.bbox
            for j, face2 in enumerate(valid_faces):
                if i >= j:  # 이미 처리된 얼굴과 비교하지 않음
                    continue
                try:
                    bbox2 = face2.bbox
                    iou = calculate_iou(bbox1, bbox2)
                    if iou > iou_threshold:
                        # 더 큰 얼굴을 유지
                        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                        if area1 < area2:
                            is_duplicate = True
                            break
                except (AttributeError, IndexError):
                    continue
            
            if not is_duplicate:
                filtered_faces.append(face1)
        except (AttributeError, IndexError):
            continue
    
    return filtered_faces

