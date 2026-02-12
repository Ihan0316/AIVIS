# config.py - 시스템 설정 및 구성
import os
import logging
import platform
from typing import List, Tuple, Dict, Any

# 환경 변수 로드 (선택적)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 없어도 작동

# 이 파일의 실제 위치(src/backend)를 기준으로 절대 경로를 생성합니다.
# final 구조에 맞게 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # -> /app/src/backend

class Paths:
    """모델 및 데이터 경로 설정"""
    # 환경변수에서 모델 경로를 가져오거나 기본값 사용
    # final 구조: src/backend에서 ../../models, ../../data/embeddings, ../../logs

    # 모델 경로 설정 (src/backend에서 두 단계 위인 프로젝트 루트의 model 폴더)
    # 주의: 현재 aivis-project는 model/ (단수)를 사용
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', os.path.normpath(os.path.join(BASE_DIR, "../../model")))

    YOLO_VIOLATION_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "Yolo11n_PPE1.pt"))
    YOLO_POSE_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "yolo11n-pose.pt"))
    YOLO_FACE_MODEL: str = os.path.normpath(os.path.join(MODEL_BASE_DIR, "yolov8n-face.pt"))
    YOLO_VIOLATION_ENGINE: str = os.path.normpath(os.getenv('YOLO_VIOLATION_ENGINE', os.path.join(MODEL_BASE_DIR, "Yolo11n_PPE1.engine")))
    YOLO_POSE_ENGINE: str = os.path.normpath(os.getenv('YOLO_POSE_ENGINE', os.path.join(MODEL_BASE_DIR, "yolo11n-pose.engine")))
    YOLO_FACE_ENGINE: str = os.path.normpath(os.getenv('YOLO_FACE_ENGINE', os.path.join(MODEL_BASE_DIR, "yolov8n-face.engine")))
    
    # AdaFace 모델 경로 (선택적, 없으면 buffalo_l 사용)
    ADAFACE_MODEL: str = os.path.normpath(os.getenv('ADAFACE_MODEL', os.path.join(MODEL_BASE_DIR, "adaface_ir50_ms1mv2.onnx")))

    # 로그 폴더 (프로젝트 루트의 logs 폴더)
    LOG_FOLDER: str = os.path.normpath(os.path.join(BASE_DIR, "../../logs"))
    LOG_CSV: str = os.path.join(LOG_FOLDER, "system_log.csv")

    # FAISS 파일 (face/data 폴더에 위치)
    # face/data/face_index.faiss, face/data/face_index.faiss.labels.npy
    FAISS_INDEX: str = os.path.normpath(os.path.join(BASE_DIR, "../..", "face", "data", "face_index.faiss"))
    FAISS_LABELS: str = os.path.normpath(os.path.join(BASE_DIR, "../..", "face", "data", "face_index.faiss.labels.npy"))

    _os_name = platform.system()
    if _os_name == "Windows":
        _font_path = "C:/Windows/Fonts/malgun.ttf"
    elif _os_name == "Darwin":
        _font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    else:
        _font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    FONT_PATH: str = _font_path

class Thresholds:
    """탐지 및 인식 관련 임계값 설정 - CCTV 환경 최적화"""
    # 얼굴 인식 매칭 개선: 기본값 0.30 (오인식 방지 강화)
    # 환경변수로 조정 가능 (0.25-0.45 권장)
    # 0.25: 매우 낮은 임계값 (많은 인식, 오인식 가능), 0.28: 인식률 우선, 0.32: 균형, 0.36: 오인식 방지 강화, 0.37: 오인식 방지 강화(권장), 0.45: 높은 임계값 (엄격한 매칭)
    # 환경변수 우선, 없으면 0.30 사용
    SIMILARITY: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.30'))  # 유사도 임계값 (0.22 -> 0.30, 오인식 방지 강화)
    
    # 1위와 2위 유사도 차이 임계값 (오인식 방지 강화)
    # 1위와 2위의 차이가 이 값 미만이면 불확실한 매칭으로 간주하여 Unknown 반환
    # 값이 클수록 더 엄격한 검증 (0.08 -> 0.12, 오인식 방지 강화)
    SIMILARITY_DIFF_THRESHOLD: float = float(os.getenv('SIMILARITY_DIFF_THRESHOLD', '0.12'))  # 0.08 -> 0.12 (오인식 방지 강화)

    # 객체 탐지 임계값 (감지율 향상을 위해 하향 조정)
    YOLO_CONFIDENCE: float = float(os.getenv('YOLO_CONFIDENCE', '0.08'))  # 0.05 -> 0.08 (바운딩 박스 안정화, 깜빡임 방지)
    POSE_CONFIDENCE: float = float(os.getenv('POSE_CONFIDENCE', '0.08'))  # 0.10 -> 0.08 (더 많은 사람 감지)

    # 추적 관련 임계값 (정확도 향상)
    IOU_MATCHING: float = float(os.getenv('IOU_MATCHING', '0.6'))  # 매칭 IoU 상향으로 더 정확한 매칭
    IOU_VIOLATION: float = float(os.getenv('IOU_VIOLATION', '0.15'))  # 위반 탐지 IoU 상향으로 정확도 향상

    # 넘어짐 감지 (단순화된 로직)
    FALL_ASPECT_RATIO: float = 1.5  # 가로가 세로보다 1.5배 이상이면 넘어짐으로 판단
    FALL_TIME: float = float(os.getenv('FALL_TIME', '1.5'))  # 넘어짐 감지 시간 단축
    FALL_MOVEMENT: int = 15

    # 넘어짐 감지 세부 임계값 (개선된 로직용)
    FALL_VERTICAL_RATIO_THRESHOLD: float = 0.3  # 수직 비율 임계값 (낮을수록 넘어짐 가능성 높음)
    FALL_HORIZONTAL_SPREAD_RATIO: float = 1.5  # 수평 분산 비율 임계값
    FALL_SCORE_THRESHOLD: float = 0.6  # 넘어짐 점수 임계값 (0.6 이상이면 넘어짐으로 판단)

    # 얼굴 인식 중복 제거 설정 (정확도 향상)
    FACE_IOU_THRESHOLD: float = 0.4  # 얼굴 중복 제거 IoU 임계값 상향 (더 엄격한 중복 제거)
    MIN_FACE_SIZE_DEDUP: int = 40  # 중복 제거 시 최소 얼굴 크기 증가 (더 큰 얼굴만 처리)

    # 얼굴 인식 (CCTV 환경 최적화 - 실시간 처리 최적화)
    MIN_FACE_SIZE: int = int(os.getenv('MIN_FACE_SIZE', '16'))  # 최소 얼굴 크기 (기본값: 16, 작은 얼굴 감지 개선)
    UPSCALE_THRESHOLD: int = 100  # 업스케일 임계값 감소로 인식률 향상
    # 얼굴 탐지 신뢰도 (작은 얼굴 감지 개선: 0.12 -> 0.10)
    FACE_DETECTION_CONFIDENCE: float = float(os.getenv('FACE_DETECTION_CONFIDENCE', '0.10'))  # 0.12 -> 0.10 (작은 얼굴 감지 개선)
    # 얼굴 인식 시 신뢰도 (인식률 최대화: 0.10 -> 0.08)
    FACE_CONFIDENCE_STRICT: float = float(os.getenv('FACE_CONFIDENCE_STRICT', '0.06'))  # 0.08 -> 0.06 (인식률 최대화, 작은 얼굴 감지 개선)
    # 얼굴 인식 허용 최소 크기(짧은 변 기준, px) - 가까운 거리/큰 얼굴도 인식하도록 완화
    MIN_FACE_SHORT_SIDE_FOR_RECOG: int = int(os.getenv('MIN_FACE_SHORT_SIDE_FOR_RECOG', '15'))  # 20 -> 15 (더 작은 얼굴도 인식)
    # 얼굴이 사람 박스의 상단 영역에 위치해야 함 (뒤통수/몸 일부 오검출 차단) - 가까운 거리 대응
    FACE_TOP_MAX_RATIO: float = float(os.getenv('FACE_TOP_MAX_RATIO', '0.80'))   # 0.70 -> 0.80 (더 낮은 위치도 허용)
    FACE_BOTTOM_MIN_RATIO: float = float(os.getenv('FACE_BOTTOM_MIN_RATIO', '0.00'))  # 0.01 -> 0.00 (위치 제한 완전 제거)
    # 얼굴 탐지 크기 (속도 최적화: 832 -> 640, 약 30% 속도 향상)
    _det_size_str = os.getenv('FACE_DETECTION_SIZE', '640')  # 832 -> 640 (속도 최적화)
    if ',' in _det_size_str:
        _det_size_list = list(map(int, _det_size_str.split(',')))
        _det_size_tuple: Tuple[int, int] = (_det_size_list[0], _det_size_list[1] if len(_det_size_list) > 1 else _det_size_list[0])
    else:
        _det_size_int = int(_det_size_str)
        _det_size_tuple = (_det_size_int, _det_size_int)
    FACE_DETECTION_SIZE: Tuple[int, int] = _det_size_tuple
    # 성능 최대화: 얼굴 탐지 간격 최소화 (모든 프레임에서 탐지)
    # GPU 최대 활용을 위해 간격 최소화
    FACE_DETECTION_INTERVAL: int = int(os.getenv('FACE_DETECTION_INTERVAL', '1'))  # 25 -> 1 (모든 프레임 탐지, 성능 최대화)
    # 프레임당 최대 얼굴 수 (M4 Pro 최적화: 3 -> 4, 20코어 GPU 성능 활용)
    # M4 Pro GPU 사용 시: 4개로 증가 (GPU 성능 활용)
    MAX_FACES_PER_FRAME: int = int(os.getenv('MAX_FACES_PER_FRAME', '4'))  # 3 -> 4 (M4 Pro GPU 성능 활용)

    # 얼굴 인식 안정화 (히스테리시스/홀드) - 깜빡임 완전 제거
    RECOGNITION_HOLD_SECONDS: float = float(os.getenv('RECOGNITION_HOLD_SECONDS', '2.0'))  # 1.2 -> 2.0초 (바운딩 박스 안정화, 깜빡임 방지)
    RECOGNITION_HYSTERESIS_DELTA: float = float(os.getenv('RECOGNITION_HYSTERESIS_DELTA', '0.08'))  # 0.06 -> 0.08 (더 안정적)
    FACE_RECOGNITION_COOLDOWN_SECONDS: float = float(os.getenv('FACE_RECOGNITION_COOLDOWN_SECONDS', '0.5'))
    MIN_FACE_RECOGNITION_AREA: int = int(os.getenv('MIN_FACE_RECOGNITION_AREA', '1200'))  # 1600 -> 1200 (약 35x35, 더 작은 얼굴도 인식 시도)
    # 사람 키(픽셀)가 프레임 높이 대비 이 비율 이상일 때만 얼굴 인식 시도 (거리 제한 완화)
    MIN_PERSON_HEIGHT_RATIO_FOR_FACE: float = float(os.getenv('MIN_PERSON_HEIGHT_RATIO_FOR_FACE', '0.06'))  # 0.08 -> 0.06 (더 멀리서도 인식)

    # FAISS GPU 사용 옵션 (기본 활성화: 성능 향상)
    USE_FAISS_GPU: bool = os.getenv('USE_FAISS_GPU', '1') in ['1', 'true', 'True']

    # 포즈 감지 품질 (정확도 향상)
    # 멀리 있는 사람도 감지하기 위해 키포인트 요구사항 대폭 완화
    # 가까운 거리 대응: 키포인트 요구사항 완화
    MIN_VISIBLE_KEYPOINTS: int = 4  # 5 -> 4 (가까운 거리에서 일부 키포인트만 보여도 감지)
    MIN_VERTICAL_RATIO: float = 0.6  # 0.7 -> 0.6 (가까운 거리에서 다양한 자세 허용)

    # 사람 탐지 필터링 (손/작은 객체 오인식 방지)
    # 멀리 있는 사람도 감지하기 위해 임계값 대폭 완화
    # 가까운 거리 대응: 최소 크기 기준 완화 (큰 박스도 허용)
    MIN_PERSON_BOX_WIDTH: int = 20   # 30 -> 20 (가까운 거리/큰 박스 감지 개선)
    MIN_PERSON_BOX_HEIGHT: int = 40  # 60 -> 40 (가까운 거리/큰 박스 감지 개선)
    MIN_PERSON_BOX_AREA: int = 800   # 1800 -> 800 (가까운 거리/큰 박스 감지 개선)
    MAX_PERSON_ASPECT_RATIO: float = 3.0  # 2.5 -> 3.0 (가까운 거리에서 다양한 자세 허용)
    MIN_PERSON_ASPECT_RATIO: float = 0.2  # 0.3 -> 0.2 (가까운 거리에서 다양한 자세 허용)

    # 상반신만 보이거나 멀리 있는 경우를 위한 완화 임계값
    # 매우 작은 박스도 허용하여 멀리 있는 사람 감지
    # 가까운 거리 대응: 더 완화된 기준
    RELAXED_MIN_PERSON_BOX_WIDTH: int = 15  # 20 -> 15 (가까운 거리/큰 박스 감지 개선)
    RELAXED_MIN_PERSON_BOX_HEIGHT: int = 30  # 40 -> 30 (가까운 거리/큰 박스 감지 개선)
    RELAXED_MIN_PERSON_BOX_AREA: int = 450   # 800 -> 450 (가까운 거리/큰 박스 감지 개선)
    RELAXED_MAX_PERSON_ASPECT_RATIO: float = 3.0  # 2.5 -> 3.0 (가까운 거리에서 다양한 자세 허용)
    RELAXED_MIN_PERSON_ASPECT_RATIO: float = 0.2  # 0.3 -> 0.2 (가까운 거리에서 다양한 자세 허용)

    # 무시할 클래스 목록 (오탐지 방지)
    IGNORED_CLASSES: List[str] = [
        'Safety Con', 'safety con', 'Safety Cone', 'safety_cone', 
        'safetycone', 'SafetyCone', 'safety_cone', 'Safety_Cone',
        'machinery', 'Machinery', 'MACHINERY',
        'vehicle', 'Vehicle', 'VEHICLE'
    ]


class SystemConfig:
    """실시간 시스템 동작 관련 설정 - MPS 최적화"""
    HEADLESS: bool = False

    # TensorRT 활성화 (속도 2-3배 향상)
    # ⚠️ 주의: TensorRT는 키포인트를 제공하지 않아 포즈/얼굴 모델에 사용 불가
    # 위반 모델에만 사용 가능 (키포인트 불필요)
    # 권장: ONNX 사용 (키포인트 지원, 해상도 조절 가능, 속도 충분)
    ENABLE_TENSORRT: bool = os.getenv('ENABLE_TENSORRT', '0') in ['1', 'true', 'True']  # 기본값: False (ONNX 권장)
    
    # 반정밀도(FP16) 사용 (GPU 메모리 절약 및 속도 향상)
    ENABLE_HALF_PRECISION: bool = os.getenv('ENABLE_HALF_PRECISION', 'true').lower() == 'true'

    @classmethod
    def get_device_config(cls) -> Dict[str, Any]:
        """디바이스 설정 (CUDA 우선, MPS 지원, 멀티 GPU 지원)"""
        import torch
        import platform

        device_config = {'device': 'cpu', 'device_face': 'cpu', 'gpu_count': 0, 'gpu_memory_gb': 0}

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_config['gpu_count'] = gpu_count

            if gpu_count >= 2:
                # 2개 이상의 GPU가 있으면 분산 사용
                device_config['device'] = 'cuda:0'  # GPU 0: YOLO Violation, Pose
                device_config['device_face'] = 'cuda:1'  # GPU 1: YOLO Face, InsightFace
                device_config['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logging.info(f"멀티 GPU 활성화: GPU 0 (YOLO Violation/Pose), GPU 1 (얼굴 인식)")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logging.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                # GPU 1개만 있으면 모두 GPU 0 사용
                device_config['device'] = 'cuda:0'
                device_config['device_face'] = 'cuda:0'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                device_config['gpu_memory_gb'] = gpu_memory
                logging.info(f"CUDA GPU 활성화: {gpu_name} ({gpu_memory:.1f}GB)")
        elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
            device_config['device'] = 'mps'
            device_config['device_face'] = 'mps'
            device_config['gpu_count'] = 1  # MPS는 단일 GPU로 처리
            
            # Mac 시스템 메모리 정보 가져오기 (MPS는 통합 메모리 사용)
            try:
                import psutil
                total_memory_gb = psutil.virtual_memory().total / (1024**3)
                
                # Mac 모델 정보 가져오기 (M4 Pro 감지)
                try:
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                           capture_output=True, text=True, timeout=1)
                    cpu_info = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"
                    
                    # M4 Pro 감지 (20코어 GPU)
                    is_m4_pro = 'M4' in cpu_info and 'Pro' in cpu_info
                    
                    if is_m4_pro:
                        # M4 Pro: 20코어 GPU, 14코어 CPU
                        # 통합 메모리에서 GPU가 더 많은 메모리 사용 가능
                        if total_memory_gb >= 36:
                            device_config['gpu_memory_gb'] = 12  # 36GB+ 시스템: 12GB GPU 메모리로 추정
                        elif total_memory_gb >= 18:
                            device_config['gpu_memory_gb'] = 10  # 18GB+ 시스템: 10GB GPU 메모리로 추정
                        else:
                            device_config['gpu_memory_gb'] = 8  # 18GB 미만: 8GB GPU 메모리로 추정
                        logging.info(f"MPS (Metal Performance Shaders) 활성화됨 - {cpu_info}")
                        logging.info(f"  M4 Pro 감지: 20코어 GPU, 14코어 CPU")
                        logging.info(f"  추정 GPU 메모리: {device_config['gpu_memory_gb']:.1f}GB (통합 메모리 사용)")
                    else:
                        # 다른 Apple Silicon
                        if total_memory_gb >= 16:
                            device_config['gpu_memory_gb'] = 8
                        elif total_memory_gb >= 8:
                            device_config['gpu_memory_gb'] = 6
                        else:
                            device_config['gpu_memory_gb'] = 4
                        logging.info(f"MPS (Metal Performance Shaders) 활성화됨 - {cpu_info}")
                        logging.info(f"  추정 GPU 메모리: {device_config['gpu_memory_gb']:.1f}GB (통합 메모리 사용)")
                except Exception:
                    # 모델 정보를 가져올 수 없으면 기본값 사용
                    if total_memory_gb >= 16:
                        device_config['gpu_memory_gb'] = 8
                    elif total_memory_gb >= 8:
                        device_config['gpu_memory_gb'] = 6
                    else:
                        device_config['gpu_memory_gb'] = 4
                    logging.info("MPS (Metal Performance Shaders) 활성화됨")
            except ImportError:
                # psutil이 없으면 기본값 사용 (M4 Pro 가정)
                device_config['gpu_memory_gb'] = 10
                logging.info("MPS (Metal Performance Shaders) 활성화됨 (M4 Pro 가정: 10GB GPU 메모리)")
        else:
            logging.warning("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")

        return device_config

    @classmethod
    def get_optimal_batch_size(cls) -> int:
        """GPU 메모리에 따라 동적 배치 크기 결정 (성능 최대화)"""
        device_config = cls.get_device_config()
        gpu_memory_gb = device_config.get('gpu_memory_gb', 0)
        device = device_config.get('device', 'cpu')

        # 성능 최대화: MPS에서 배치 크기 증가로 GPU 활용률 극대화
        if device == 'mps' and gpu_memory_gb >= 10:
            # M4 Pro: 20코어 GPU 최대 활용, 배치 크기 8로 GPU 병렬 처리 극대화
            return 8  # 배치 처리로 GPU 활용률 극대화
        elif device == 'mps' and gpu_memory_gb >= 6:
            # M1/M2/M3: 배치 크기 4로 GPU 활용률 향상
            return 4
        elif 'cuda' in str(device):
            # CUDA: 실시간 스트리밍에서는 배치 크기 1이 가장 빠름 (즉시 처리)
            return 1
        else:
            # CPU: 배치 크기 1 (실시간 처리 최적화)
            return 1

    @classmethod
    def get_optimal_input_size(cls) -> Tuple[int, int]:
        """GPU 메모리에 따라 동적 입력 해상도 결정 (정확도와 속도 균형)"""
        device_config = cls.get_device_config()
        gpu_memory_gb = device_config.get('gpu_memory_gb', 0)
        device = device_config.get('device', 'cpu')

        # 환경변수로 설정된 값이 있으면 우선 사용
        env_width = int(os.getenv('MODEL_INPUT_WIDTH', '0'))
        env_height = int(os.getenv('MODEL_INPUT_HEIGHT', '0'))
        if env_width > 0 and env_height > 0:
            return (env_width, env_height)

        # M4 Pro (20코어 GPU) 최적화
        if device == 'mps' and gpu_memory_gb >= 10:
            # M4 Pro: 20코어 GPU 성능 활용, 640x480 최적화
            return (640, 480)
        elif gpu_memory_gb >= 12:
            # 고성능 GPU: 640x480 (성능 최적화)
            return (640, 480)
        elif gpu_memory_gb >= 6:
            # 중급 GPU: 640x480 (성능 최적화)
            return (640, 480)
        elif gpu_memory_gb > 0:
            # 저급 GPU: 512x384 (메모리 절약)
            return (512, 384)
        else:
            # CPU: 640x480 (실시간 처리 최적)
            return (640, 480)

    @classmethod
    def get_camera_indices(cls) -> List[int]:
        """환경 변수에서 카메라 인덱스를 가져옵니다."""
        indices_str = os.getenv('CAMERA_INDICES', '0')  # 기본값을 0으로 변경

        if indices_str.lower() == 'auto':
            # 서버 환경에서는 자동 감지가 의미 없을 수 있음
            logging.warning("CAMERA_INDICES=auto. 서버 환경에서는 지원되지 않을 수 있습니다. [0] 사용")
            return [0]

        try:
            cameras = [int(x.strip()) for x in indices_str.split(',')]
            logging.info(f"환경 변수에서 설정된 카메라: {cameras}")
            return cameras
        except ValueError:
            logging.warning("CAMERA_INDICES 환경 변수 형식 오류. 기본값 [0] 사용")
            return [0]

    # (server.py가 카메라를 직접 제어하지 않으므로 이 설정들은 대부분 client.py로 이동)
    CAMERA_INDICES: List[int] = [0]

    # 디스플레이 해상도 (서버 처리 기준)
    DISPLAY_WIDTH: int = int(os.getenv('DISPLAY_WIDTH', '960'))
    DISPLAY_HEIGHT: int = int(os.getenv('DISPLAY_HEIGHT', '540'))

    # 모델 입력 해상도 (동적 최적화)
    # 환경변수가 설정되어 있으면 그 값 사용, 없으면 기본값 640x480 (성능 최적화)
    # 640x480은 실시간 처리에 최적화된 크기 (640x640보다 약 25% 빠름, 메모리 사용량 감소)
    MODEL_INPUT_WIDTH: int = int(os.getenv('MODEL_INPUT_WIDTH', '640'))
    MODEL_INPUT_HEIGHT: int = int(os.getenv('MODEL_INPUT_HEIGHT', '480'))
    # ONNX 모델 해상도 (convert_to_onnx.py에서 변환한 해상도와 일치해야 함)
    # GPU 성능에 맞춰 자동 추천: 11GB GPU 2개 = 832x832 권장
    ONNX_MODEL_SIZE: int = int(os.getenv('ONNX_MODEL_SIZE', '832'))  # ONNX 모델 변환 시 사용한 해상도 (기본값: 832)

    # 성능 최대화: 모든 프레임에서 감지 (간격 없음)
    DETECTION_INTERVAL: int = int(os.getenv('DETECTION_INTERVAL', '1'))  # 1 (모든 프레임 처리)
    MAX_INACTIVE_FRAMES: int = int(os.getenv('MAX_INACTIVE_FRAMES', '60'))
    # Pose 모델 실행 간격 (CPU 병목 해소를 위해 N프레임마다 실행, 기본값: 3)
    # 1 = 모든 프레임, 2 = 2프레임마다, 3 = 3프레임마다 (기본값: 3으로 CPU 부하 감소)
    POSE_INTERVAL: int = int(os.getenv('POSE_INTERVAL', '3'))  # 3프레임마다 Pose 실행 (CPU 부하 감소)
    POSE_SMOOTHING_FACTOR: float = 0.4
    MAX_PEOPLE_TO_TRACK: int = int(os.getenv('MAX_PEOPLE_TO_TRACK', '20')) # 10 -> 20 (추적 인원 증가)
    EMBEDDING_BUFFER_SIZE: int = int(os.getenv('EMBEDDING_BUFFER_SIZE', '10')) # 5 -> 10 (버퍼 증가)

    # 성능 최적화 (동적 최적화)
    ENABLE_HALF_PRECISION: bool = os.getenv('ENABLE_HALF_PRECISION', '1') in ['1', 'true', 'True']
    ENABLE_MODEL_FUSION: bool = os.getenv('ENABLE_MODEL_FUSION', '1') in ['1', 'true', 'True']
    ENABLE_TENSORRT: bool = os.getenv('ENABLE_TENSORRT', '1') in ['1', 'true', 'True']  # TensorRT 엔진 파일 생성됨, 기본값 True
    ENABLE_ONNX_OPTIMIZATION: bool = os.getenv('ENABLE_ONNX_OPTIMIZATION', '1') in ['1', 'true', 'True']
    ENABLE_MODEL_QUANTIZATION: bool = os.getenv('ENABLE_MODEL_QUANTIZATION', '0') in ['1', 'true', 'True']

    # 배치 크기: 환경변수 우선, 없으면 기본값 8 (런타임에 get_optimal_batch_size() 사용 권장)
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '8'))
    # M4 Pro (14코어 CPU) 최적화: CPU 코어 수에 맞게 워커 수 조정
    # 14코어 = 성능 코어(10) + 효율 코어(4), 실효 코어 약 12개 활용
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '12'))  # 8 -> 12 (M4 Pro 14코어 활용)
    
    # 성능 최대화: 모델 추론 타임아웃 증가 (더 많은 프레임 처리)
    MODEL_INFERENCE_TIMEOUT: float = float(os.getenv('MODEL_INFERENCE_TIMEOUT', '10.0'))  # 8.0 -> 10.0 (성능 최대화)

    ENABLE_MODEL_CACHING: bool = True
    # 성능 최대화: 프레임 스킵 비활성화 (모든 프레임 처리)
    ENABLE_FRAME_SKIPPING: bool = os.getenv('ENABLE_FRAME_SKIPPING', 'false').lower() == 'true'
    # 성능 최대화: 프레임 스킵 비율 0 (모든 프레임 처리, GPU 최대 활용)
    FRAME_SKIP_RATIO: float = float(os.getenv('FRAME_SKIP_RATIO', '0.0'))  # 0.0 (프레임 스킵 없음, 최대 성능)
    ENABLE_ADAPTIVE_QUALITY: bool = True


class Policy:
    """이벤트 로그 및 재인식 정책 설정"""
    EVENT_LOG_COOLDOWN: int = int(os.getenv('EVENT_LOG_COOLDOWN', '15'))
    UNKNOWN_RECOGNITION_COOLDOWN: int = int(os.getenv('UNKNOWN_RECOGNITION_COOLDOWN', '10'))
    KNOWN_RECOGNITION_COOLDOWN: int = int(os.getenv('KNOWN_RECOGNITION_COOLDOWN', '120'))


class Constants:
    """고정 상수 값"""
    SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    SAFETY_RULES_MAP: Dict[str, Dict[str, str]] = {
        "안전모": {"compliance": "Hardhat", "violation": "NO-Hardhat"},
        "마스크": {"compliance": "Mask", "violation": "NO-Mask"},
        "안전조끼": {"compliance": "Safety Vest", "violation": "NO-Safety Vest"}
    }

# 로그 폴더 생성
os.makedirs(Paths.LOG_FOLDER, exist_ok=True)
