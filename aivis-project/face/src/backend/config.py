# config.py
# 간소화된 설정 파일 - 임베딩 작업에 필요한 설정만 포함
import os
from pathlib import Path

# 이 파일의 실제 위치를 기준으로 절대 경로를 생성합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # final/


class Paths:
    """모델 및 데이터 경로 설정"""
    # 환경변수에서 모델 경로를 가져오거나 기본값 사용
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', os.path.join(PARENT_DIR, "models"))
    
    YOLO_VIOLATION_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "best.pt"))
    YOLO_POSE_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "yolov8n-pose.pt"))
    EDSR_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "EDSR_x4.pb"))

    LOG_FOLDER: str = os.path.join(PARENT_DIR, "logs")
    LOG_CSV: str = os.path.join(LOG_FOLDER, "system_log.csv")

    # FAISS 파일은 data/embeddings에 있음
    FAISS_INDEX: str = os.path.normpath(os.path.join(PARENT_DIR, "data", "embeddings", "face_index.faiss"))
    FAISS_LABELS: str = os.path.normpath(os.path.join(PARENT_DIR, "data", "embeddings", "face_index.faiss.labels.npy"))


class Thresholds:
    """탐지 및 인식 관련 임계값 설정 (인식률 95% 목표)"""
    # 얼굴 인식 임계값
    SIMILARITY: float = 0.34  # 인식률 95% 목표 (0.36 -> 0.34, 더 관대한 매칭)
    MIN_FACE_SIZE: int = 20  # CCTV 환경: 작은 얼굴도 감지 (15 -> 20, 너무 작으면 오인식)
    UPSCALE_THRESHOLD: int = 150  # 업스케일 임계값
    
    # 실시간 처리 최적화 설정 (1~2초 지연 해결)
    FACE_DETECTION_INTERVAL: int = 5  # 프레임 간 얼굴 탐지 간격 (2 -> 5로 증가, 속도 향상)
    MAX_FACES_PER_FRAME: int = 5  # 프레임당 최대 얼굴 수 (10 -> 5로 감소, 속도 향상)
    FACE_DETECTION_CONFIDENCE: float = 0.6  # 얼굴 탐지 최소 신뢰도 (0.5 -> 0.6, 더 빠른 필터링)
    
    # 얼굴 탐지 해상도 (실시간 서버용 최적화)
    DET_SIZE: tuple = (640, 640)  # 832 -> 640으로 감소, 약 1.7배 속도 향상


# 로그 폴더 생성
os.makedirs(Paths.LOG_FOLDER, exist_ok=True)
