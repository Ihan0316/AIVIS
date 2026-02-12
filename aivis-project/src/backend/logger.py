"""
구조화된 로깅 시스템
JSON 형식 로그, 성능 메트릭, 에러 추적
"""
import logging
import json
import time
import traceback
import os
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
from pathlib import Path


class StructuredLogger:
    """JSON 구조화 로깅 클래스"""

    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 콘솔 핸들러 (사람이 읽기 쉬운 형식)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 파일 핸들러 (JSON 형식)
        if log_file:
            # 상대 경로인 경우 config에서 절대 경로로 변환
            if not os.path.isabs(log_file):
                try:
                    import config
                    log_dir = config.Paths.LOG_FOLDER
                    # 로그 디렉토리 생성
                    os.makedirs(log_dir, exist_ok=True)
                    # 파일명만 추출하여 절대 경로 생성
                    filename = os.path.basename(log_file)
                    log_file = os.path.join(log_dir, filename)
                except (ImportError, AttributeError):
                    # config를 가져올 수 없으면 현재 디렉토리 기준으로 처리
                    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs'
                    os.makedirs(log_dir, exist_ok=True)
                    if not os.path.isabs(log_file):
                        log_file = os.path.abspath(log_file)
            else:
                # 절대 경로인 경우에도 디렉토리 생성
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

    def log_event(self, event_type: str, data: Dict[str, Any], level: str = 'INFO'):
        """이벤트 로그 (구조화)"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data,
        }
        self.logger.log(getattr(logging, level), json.dumps(log_entry, ensure_ascii=False))

    def log_performance(self, operation: str, duration_ms: float, metadata: Optional[Dict] = None):
        """성능 메트릭 로그"""
        data = {
            'operation': operation,
            'duration_ms': round(duration_ms, 2),
            **(metadata or {})
        }
        self.log_event('performance', data, 'INFO')

    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """에러 로그 (스택 트레이스 포함)"""
        data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        self.log_event('error', data, 'ERROR')

    def log_camera_frame(self, camera_id: int, processing_time_ms: float, detections: int):
        """카메라 프레임 처리 로그"""
        data = {
            'camera_id': camera_id,
            'processing_time_ms': round(processing_time_ms, 2),
            'detections': detections
        }
        self.log_event('camera_frame', data, 'DEBUG')

    def log_violation(self, violation_data: Dict):
        """위반 사항 로그"""
        self.log_event('violation', violation_data, 'WARNING')

    def log_face_recognition(self, camera_id: int, person_name: str, similarity: float):
        """얼굴 인식 로그"""
        data = {
            'camera_id': camera_id,
            'person_name': person_name,
            'similarity': round(similarity, 3)
        }
        self.log_event('face_recognition', data, 'INFO')


class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포맷터"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def log_execution_time(logger: StructuredLogger, operation: str):
    """함수 실행 시간 측정 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, {'success': True})
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, {'success': False, 'error': str(e)})
                logger.log_error(e, {'function': func.__name__, 'args': str(args)[:100]})
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, {'success': True})
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, {'success': False, 'error': str(e)})
                logger.log_error(e, {'function': func.__name__, 'args': str(args)[:100]})
                raise

        # 비동기 함수인지 확인
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# 전역 로거 인스턴스
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, log_file: Optional[str] = None) -> StructuredLogger:
    """로거 인스턴스 가져오기 (싱글톤)"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, log_file)
    return _loggers[name]


# 기본 로거들 (config에서 로그 폴더 경로 가져오기)
try:
    import config
    log_folder = config.Paths.LOG_FOLDER
    os.makedirs(log_folder, exist_ok=True)
    camera_logger = get_logger('aivis.camera', os.path.join(log_folder, 'camera.log'))
    violation_logger = get_logger('aivis.violation', os.path.join(log_folder, 'violation.log'))
    face_logger = get_logger('aivis.face', os.path.join(log_folder, 'face.log'))
    system_logger = get_logger('aivis.system', os.path.join(log_folder, 'system.log'))
except (ImportError, AttributeError):
    # config를 가져올 수 없으면 상대 경로 사용 (하위 호환성)
    camera_logger = get_logger('aivis.camera', 'logs/camera.log')
    violation_logger = get_logger('aivis.violation', 'logs/violation.log')
    face_logger = get_logger('aivis.face', 'logs/face.log')
    system_logger = get_logger('aivis.system', 'logs/system.log')
