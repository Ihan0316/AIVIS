# -*- coding: utf-8 -*-
"""
커스텀 예외 클래스
에러 처리를 표준화하고 더 명확한 에러 메시지를 제공합니다.
"""
from typing import Dict, Any, Optional


class AIVISError(Exception):
    """AIVIS 시스템 기본 예외 클래스"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Args:
            message: 에러 메시지
            error_code: 에러 코드 (선택)
            details: 추가 상세 정보 (선택)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """에러 정보를 딕셔너리로 변환"""
        result: Dict[str, Any] = {
            'error': self.message,
            'error_code': self.error_code
        }
        if self.details:
            result['details'] = self.details
        return result


class ProcessingError(AIVISError):
    """프로세싱 관련 예외"""
    pass


class FaceRecognitionError(ProcessingError):
    """얼굴 인식 관련 예외"""
    pass


class ModelLoadError(ProcessingError):
    """모델 로딩 관련 예외"""
    pass


class CameraError(AIVISError):
    """카메라 관련 예외"""
    pass


class DatabaseError(AIVISError):
    """데이터베이스 관련 예외"""
    pass


class ValidationError(AIVISError):
    """입력 검증 관련 예외"""
    pass


class ConfigurationError(AIVISError):
    """설정 관련 예외"""
    pass

