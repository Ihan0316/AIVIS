# -*- coding: utf-8 -*-
"""
입력 검증 모듈
API 요청 및 데이터 검증을 위한 유틸리티 함수
"""
import logging
from typing import Any, Optional, Tuple, List
import numpy as np
from exceptions import ValidationError


def validate_camera_id(cam_id: Any) -> int:
    """
    카메라 ID 검증
    
    Args:
        cam_id: 검증할 카메라 ID
    
    Returns:
        검증된 카메라 ID (int)
    
    Raises:
        ValidationError: 유효하지 않은 카메라 ID
    """
    try:
        cam_id_int = int(cam_id)
        if cam_id_int < 0 or cam_id_int > 10:
            raise ValidationError(
                f"카메라 ID는 0-10 사이여야 합니다: {cam_id_int}",
                error_code="INVALID_CAMERA_ID"
            )
        return cam_id_int
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"유효하지 않은 카메라 ID 형식: {cam_id}",
            error_code="INVALID_CAMERA_ID_FORMAT"
        ) from e


def validate_frame_bytes(frame_bytes: Any, max_size: int = 10 * 1024 * 1024) -> bytes:
    """
    프레임 바이트 데이터 검증
    
    Args:
        frame_bytes: 검증할 프레임 바이트
        max_size: 최대 크기 (바이트), 기본값 10MB
    
    Returns:
        검증된 프레임 바이트
    
    Raises:
        ValidationError: 유효하지 않은 프레임 데이터
    """
    if not isinstance(frame_bytes, bytes):
        raise ValidationError(
            f"프레임 데이터는 bytes 타입이어야 합니다: {type(frame_bytes)}",
            error_code="INVALID_FRAME_TYPE"
        )
    
    if len(frame_bytes) == 0:
        raise ValidationError(
            "프레임 데이터가 비어있습니다",
            error_code="EMPTY_FRAME"
        )
    
    if len(frame_bytes) > max_size:
        raise ValidationError(
            f"프레임 크기가 너무 큽니다: {len(frame_bytes)} bytes (최대: {max_size} bytes)",
            error_code="FRAME_TOO_LARGE"
        )
    
    return frame_bytes


def validate_bbox(bbox: Any, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """
    바운딩 박스 검증 및 정규화
    
    Args:
        bbox: 바운딩 박스 (x1, y1, x2, y2) 또는 리스트
        frame_width: 프레임 너비
        frame_height: 프레임 높이
    
    Returns:
        검증된 바운딩 박스 (x1, y1, x2, y2)
    
    Raises:
        ValidationError: 유효하지 않은 바운딩 박스
    """
    try:
        if isinstance(bbox, (list, tuple)):
            if len(bbox) != 4:
                raise ValidationError(
                    f"바운딩 박스는 4개의 좌표가 필요합니다: {len(bbox)}개",
                    error_code="INVALID_BBOX_LENGTH"
                )
            x1, y1, x2, y2 = map(float, bbox)
        else:
            raise ValidationError(
                f"바운딩 박스는 리스트 또는 튜플이어야 합니다: {type(bbox)}",
                error_code="INVALID_BBOX_TYPE"
            )
        
        # 좌표 검증
        if x1 >= x2 or y1 >= y2:
            raise ValidationError(
                f"바운딩 박스 좌표가 잘못되었습니다: ({x1}, {y1}, {x2}, {y2})",
                error_code="INVALID_BBOX_COORDS"
            )
        
        # 프레임 경계 내로 제한
        x1 = max(0, min(int(x1), frame_width - 1))
        y1 = max(0, min(int(y1), frame_height - 1))
        x2 = max(x1 + 1, min(int(x2), frame_width))
        y2 = max(y1 + 1, min(int(y2), frame_height))
        
        return (x1, y1, x2, y2)
    
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"바운딩 박스 파싱 오류: {e}",
            error_code="BBOX_PARSING_ERROR"
        ) from e


def validate_embedding(embedding: Any, expected_dim: int = 512) -> np.ndarray:
    """
    임베딩 벡터 검증
    
    Args:
        embedding: 검증할 임베딩 벡터
        expected_dim: 예상 차원, 기본값 512
    
    Returns:
        검증된 임베딩 벡터 (numpy array)
    
    Raises:
        ValidationError: 유효하지 않은 임베딩
    """
    try:
        if embedding is None:
            raise ValidationError(
                "임베딩이 None입니다",
                error_code="EMPTY_EMBEDDING"
            )
        
        embedding_array = np.array(embedding, dtype=np.float32)
        
        if embedding_array.size == 0:
            raise ValidationError(
                "임베딩이 비어있습니다",
                error_code="EMPTY_EMBEDDING"
            )
        
        # 차원 검증
        if embedding_array.ndim == 1:
            if embedding_array.shape[0] != expected_dim:
                raise ValidationError(
                    f"임베딩 차원이 일치하지 않습니다: {embedding_array.shape[0]} (예상: {expected_dim})",
                    error_code="INVALID_EMBEDDING_DIM"
                )
        elif embedding_array.ndim == 2:
            if embedding_array.shape[1] != expected_dim:
                raise ValidationError(
                    f"임베딩 차원이 일치하지 않습니다: {embedding_array.shape[1]} (예상: {expected_dim})",
                    error_code="INVALID_EMBEDDING_DIM"
                )
            # 2D 배열을 1D로 변환
            embedding_array = embedding_array.flatten()
        else:
            raise ValidationError(
                f"임베딩 차원이 잘못되었습니다: {embedding_array.ndim}D (예상: 1D 또는 2D)",
                error_code="INVALID_EMBEDDING_SHAPE"
            )
        
        return embedding_array
    
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"임베딩 파싱 오류: {e}",
            error_code="EMBEDDING_PARSING_ERROR"
        ) from e


def validate_similarity_threshold(threshold: Any) -> float:
    """
    유사도 임계값 검증
    
    Args:
        threshold: 검증할 임계값
    
    Returns:
        검증된 임계값 (float)
    
    Raises:
        ValidationError: 유효하지 않은 임계값
    """
    try:
        threshold_float = float(threshold)
        if not 0.0 <= threshold_float <= 1.0:
            raise ValidationError(
                f"유사도 임계값은 0.0-1.0 사이여야 합니다: {threshold_float}",
                error_code="INVALID_THRESHOLD_RANGE"
            )
        return threshold_float
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"유효하지 않은 임계값 형식: {threshold}",
            error_code="INVALID_THRESHOLD_FORMAT"
        ) from e

