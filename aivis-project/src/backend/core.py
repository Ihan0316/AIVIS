# core.py - SafetySystem 클래스 (모델 로딩 및 관리)
import os
import sys
import cv2
import torch
import logging
import numpy as np
import platform
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints

# Conda 환경의 faiss-gpu를 사용할 수 있도록 경로 추가 (venv에서도 사용 가능하도록)
_conda_paths_added = False
if not _conda_paths_added:
    import site
    # 일반적인 conda 설치 경로 확인
    _possible_conda_paths = [
        os.path.join(os.environ.get('USERPROFILE', ''), 'anaconda3', 'Lib', 'site-packages'),
        os.path.join(os.environ.get('USERPROFILE', ''), 'miniconda3', 'Lib', 'site-packages'),
        os.path.join('C:', 'ProgramData', 'anaconda3', 'Lib', 'site-packages'),
        os.path.join('C:', 'ProgramData', 'miniconda3', 'Lib', 'site-packages'),
    ]
    # 현재 Python 실행 파일 경로에서 conda 경로 추출
    _python_dir = os.path.dirname(sys.executable)
    if 'conda' in _python_dir.lower() or 'anaconda' in _python_dir.lower():
        _conda_base = _python_dir
        while _conda_base and os.path.basename(_conda_base).lower() not in ['anaconda3', 'miniconda3', 'conda']:
            _conda_base = os.path.dirname(_conda_base)
        if _conda_base:
            _conda_site_packages = os.path.join(_conda_base, 'Lib', 'site-packages')
            if os.path.exists(_conda_site_packages) and _conda_site_packages not in sys.path:
                sys.path.insert(0, _conda_site_packages)
                _conda_paths_added = True
    
    # 가능한 경로 확인
    if not _conda_paths_added:
        for _conda_path in _possible_conda_paths:
            if os.path.exists(_conda_path) and _conda_path not in sys.path:
                sys.path.insert(0, _conda_path)
                _conda_paths_added = True
                break

# faiss 임포트 (conda 경로 추가 후)
try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("⚠️ FAISS를 찾을 수 없습니다. 얼굴 인식 기능이 제한될 수 있습니다.")

# InsightFace는 선택적 (설치 실패 시 얼굴 인식 기능 비활성화)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None
    logging.warning("insightface 모듈을 찾을 수 없습니다. 얼굴 인식 기능이 비활성화됩니다.")
    logging.warning("설치 방법: .\\install_insightface.bat")

import config
from utils import calculate_iou, clip_bbox_xyxy, is_person_horizontal, log_violation



class SafetySystem:
    def __init__(self):
        # 0. 성능 최적화 설정
        if torch.cuda.is_available():
            # 고정된 입력 크기에서 최적의 알고리즘을 찾아 속도 향상
            torch.backends.cudnn.benchmark = True
            # TensorFloat-32(TF32) 활성화 (Ampere 이상 GPU에서 성능 향상, 2080Ti는 무시됨)
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            logging.info("✅ GPU 성능 최적화 설정 완료 (CuDNN Benchmark=True)")
        elif torch.backends.mps.is_available():
            # MPS 최적화 설정 (처리 속도 향상)
            # MPS는 자동으로 최적화되지만, 명시적으로 설정 가능한 옵션들
            try:
                # MPS 캐시 활성화 (반복 연산 속도 향상)
                if hasattr(torch.backends.mps, 'is_built'):
                    logging.info("✅ MPS (Metal Performance Shaders) 최적화 활성화")
                    logging.info("  - 통합 메모리 아키텍처 활용")
                    logging.info("  - Metal GPU 가속 활성화")
                    logging.info("  - MPS 캐시 활성화 (반복 연산 최적화)")
                
                # MPS 메모리 관리 최적화
                # MPS는 통합 메모리를 사용하므로 명시적인 메모리 관리 불필요
                # 하지만 가비지 컬렉션 최적화를 위해 힌트 제공
                import gc
                gc.collect()  # 초기화 시 메모리 정리
                
            except Exception as e:
                logging.warning(f"MPS 최적화 설정 중 경고: {e}")
            
            logging.info("✅ MPS (Metal Performance Shaders) 최적화 활성화")
            logging.info("  - 통합 메모리 아키텍처 활용")
            logging.info("  - Metal GPU 가속 활성화")

        # 1. 장치 설정 (멀티 GPU 지원)
        self.device_config = config.SystemConfig.get_device_config()
        self.device = self.device_config['device']  # GPU 0: YOLO Violation, Pose
        self.device_face = self.device_config.get('device_face', self.device)  # GPU 1: YOLO Face, InsightFace
        self.gpu_count = self.device_config.get('gpu_count', 0)
        logging.info(f"SafetySystem: YOLO Violation/Pose 장치: {self.device.upper()}")
        logging.info(f"SafetySystem: 얼굴 인식 장치: {self.device_face.upper()}")

        # 2. 모델 로딩
        (self.violation_model,
         self.pose_model,
         self.violation_uses_trt,
         self.pose_uses_trt) = self._initialize_tracking_models()
        (self.face_model,
         self.face_analyzer,
         self.face_database,
         self.face_uses_trt,
         self.use_adaface,
         self.adaface_model_path) = self._initialize_face_recognition_models()
        
        # FastIndustrialRecognizer 초기화 (랜드마크 기반 고속 처리용)
        self.fast_recognizer = None
        # MPS/CUDA 디바이스에 맞는 ctx_id 설정
        if 'mps' in str(self.device_face):
            ctx_id_face = 0  # MPS는 단일 GPU
        elif 'cuda' in str(self.device_face):
            ctx_id_face = int(self.device_face.split(':')[-1]) if ':' in str(self.device_face) else 0
        else:
            ctx_id_face = -1  # CPU
        
        if self.use_adaface and self.adaface_model_path:
            try:
                from fast_face_recognizer import FastIndustrialRecognizer
                self.fast_recognizer = FastIndustrialRecognizer(
                    model_path=self.adaface_model_path,
                    ctx_id=ctx_id_face,
                    use_adaface=True
                )
                logging.info(f"✅ FastIndustrialRecognizer 초기화 완료 (AdaFace 모델: {self.adaface_model_path})")
            except Exception as e:
                logging.warning(f"⚠️ FastIndustrialRecognizer 초기화 실패, 기존 방식 사용: {e}")
                self.fast_recognizer = None
        elif self.face_analyzer is not None:
            # AdaFace가 아니어도 FastIndustrialRecognizer 사용 가능 (랜드마크 기반 처리)
            try:
                from fast_face_recognizer import FastIndustrialRecognizer
                self.fast_recognizer = FastIndustrialRecognizer(
                    model_path=None,  # InsightFace 기본 모델 사용
                    ctx_id=ctx_id_face,
                    use_adaface=False
                )
                logging.info(f"✅ FastIndustrialRecognizer 초기화 완료 (buffalo_l 모델, 랜드마크 기반 처리)")
            except Exception as e:
                logging.warning(f"⚠️ FastIndustrialRecognizer 초기화 실패, 기존 방식 사용: {e}")
                self.fast_recognizer = None

        if self.violation_model is None or self.pose_model is None:
            logging.error("필수 모델(Violation or Pose) 로딩에 실패했습니다.")
        else:
             logging.info("YOLO 모델 로딩 완료.")

        if self.face_model is None or self.face_analyzer is None or self.face_database is None:
            logging.warning("=" * 80)
            logging.warning("⚠️  얼굴 인식 모델 또는 DB 로딩에 실패했습니다.")
            logging.warning("⚠️  얼굴 인식 기능이 비활성화됩니다.")
            if not INSIGHTFACE_AVAILABLE:
                logging.warning("⚠️  InsightFace 모듈이 설치되지 않았습니다.")
                logging.warning("⚠️  설치 방법: .\\install_insightface.bat")
            logging.warning("=" * 80)
        else:
            logging.info("✅ 얼굴 인식 모델 및 DB 로딩 완료 (YOLO 얼굴 감지 + InsightFace 임베딩).")

    def _load_yolo_variant(self, weight_path: str, engine_path: str, task_description: str, task_type: str) -> Tuple[Optional[YOLO], bool]:
        """
        YOLO 모델 로드 (PyTorch .pt 파일 직접 사용, MPS/CUDA 최적화)
        
        :param weight_path: PyTorch 모델 경로 (.pt)
        :param engine_path: TensorRT 엔진 경로 (.engine) - 사용 안 함
        :param task_description: 작업 설명 (로깅용)
        :param task_type: 작업 타입 ('detect', 'pose', 'segment')
        :return: (모델, TensorRT 사용 여부) 튜플 - 항상 False 반환
        """
        # PyTorch 모델 직접 로드 (ONNX/Engine 우회)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"{task_description} 모델 파일 없음: {weight_path}")
        
        try:
            logging.info(f"{task_description} PyTorch 모델 로드: {weight_path}")
            model = YOLO(weight_path, task=task_type)
            # 디바이스 정보는 나중에 실제 이동 후 로깅됨
            return model, False  # TensorRT 사용 안 함
        except Exception as e:
            logging.error(f"{task_description} 모델 로드 실패: {e}", exc_info=True)
            return None, False
    
    @staticmethod
    def _is_onnx_model(model_path: str) -> bool:
        """모델 경로가 ONNX 모델인지 확인"""
        if model_path.endswith('.onnx'):
            return True
        onnx_path = os.path.splitext(model_path)[0] + ".onnx"
        return os.path.exists(onnx_path)

    def _initialize_tracking_models(self) -> Tuple[Optional[YOLO], Optional[YOLO], bool, bool]:
        try:
            violation_model, violation_trt = self._load_yolo_variant(
                config.Paths.YOLO_VIOLATION_MODEL,
                config.Paths.YOLO_VIOLATION_ENGINE,
                "Violation",
                "detect"
            )
            pose_model, pose_trt = self._load_yolo_variant(
                config.Paths.YOLO_POSE_MODEL,
                config.Paths.YOLO_POSE_ENGINE,
                "Pose",
                "pose"
            )

            if violation_model is None or pose_model is None:
                raise RuntimeError("필수 YOLO 모델을 로드하지 못했습니다.")

            # MPS의 경우 Pose 모델은 CPU 사용 (알려진 버그)
            import torch
            pose_device = self.device
            if 'mps' in str(self.device) and torch.backends.mps.is_available():
                # MPS Pose 모델 버그로 인해 CPU 사용
                pose_device = 'cpu'
                logging.warning("⚠️ MPS Pose 모델 알려진 버그로 인해 CPU 모드 사용 (https://github.com/ultralytics/ultralytics/issues/4031)")
            
            if not violation_trt:
                # PyTorch 모델인 경우에만 float() 및 .to() 호출
                underlying_violation = getattr(violation_model, "model", None)
                if underlying_violation is not None:
                    # 문자열이 아니고 float() 메서드가 있는 경우에만 호출 (PyTorch 모델)
                    if not isinstance(underlying_violation, str) and hasattr(underlying_violation, "float"):
                        try:
                            underlying_violation.float()
                        except (AttributeError, TypeError):
                            pass  # ONNX 모델이거나 다른 타입인 경우 무시
                
                # .to() 메서드 호출 (MPS 디바이스로 명시적 이동)
                try:
                    violation_model.to(self.device)
                    # MPS 최적화: 추론 모드로 설정 (드롭아웃, 배치 정규화 최적화)
                    if hasattr(violation_model, 'eval'):
                        violation_model.eval()
                    logging.info(f"✅ Violation 모델을 {self.device} 디바이스로 이동 완료 (추론 모드 활성화)")
                except (AttributeError, TypeError) as e:
                    logging.warning(f"⚠️ Violation 모델 디바이스 이동 실패: {e}")
                except Exception as e:
                    logging.error(f"❌ Violation 모델 디바이스 이동 오류: {e}")

            if not pose_trt:
                # PyTorch 모델인 경우에만 float() 및 .to() 호출
                underlying_pose = getattr(pose_model, "model", None)
                if underlying_pose is not None:
                    # 문자열이 아니고 float() 메서드가 있는 경우에만 호출 (PyTorch 모델)
                    if not isinstance(underlying_pose, str) and hasattr(underlying_pose, "float"):
                        try:
                            underlying_pose.float()
                        except (AttributeError, TypeError):
                            pass  # ONNX 모델이거나 다른 타입인 경우 무시
                
                # .to() 메서드 호출 (MPS인 경우 CPU 사용)
                try:
                    pose_model.to(pose_device)
                    logging.info(f"✅ Pose 모델을 {pose_device} 디바이스로 이동 완료")
                except (AttributeError, TypeError) as e:
                    logging.warning(f"⚠️ Pose 모델 디바이스 이동 실패: {e}")
                except Exception as e:
                    logging.error(f"❌ Pose 모델 디바이스 이동 오류: {e}")

            if 'cuda' in str(self.device) and (not violation_trt or not pose_trt):
                import torch
                if torch.cuda.is_available():
                    # GPU 0 정보 가져오기
                    gpu_id = int(self.device.split(':')[-1]) if ':' in str(self.device) else 0
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
                    logging.info(f"GPU {gpu_id} ({gpu_name}) 최적화: YOLO Violation/Pose 모델 실행 (메모리: {gpu_memory:.1f}GB)")
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    logging.info("✅ cuDNN 최적화 활성화")
            elif 'mps' in str(self.device) and (not violation_trt or not pose_trt):
                import torch
                if torch.backends.mps.is_available():
                    device_config = config.SystemConfig.get_device_config()
                    gpu_memory = device_config.get('gpu_memory_gb', 8)
                    logging.info(f"MPS 최적화: YOLO Violation/Pose 모델 실행 (추정 메모리: {gpu_memory:.1f}GB)")
                    logging.info("✅ MPS Metal GPU 가속 활성화")
                    # MPS 최적화: 메모리 관리 힌트
                    try:
                        # MPS는 통합 메모리를 사용하므로 명시적인 메모리 관리 불필요
                        # 하지만 PyTorch에 힌트 제공
                        if hasattr(torch, 'mps'):
                            logging.info("  - MPS 메모리 관리 최적화 활성화")
                    except:
                        pass

            # 모델 타입 로깅 (모두 PyTorch)
            violation_device_str = str(self.device).upper()
            if 'mps' in violation_device_str:
                violation_device_info = "MPS (Metal GPU)"
            elif 'cuda' in violation_device_str:
                violation_device_info = "CUDA GPU"
            else:
                violation_device_info = "CPU"
            
            pose_device_str = str(pose_device).upper()
            if 'mps' in pose_device_str:
                pose_device_info = "MPS (Metal GPU)"
            elif 'cuda' in pose_device_str:
                pose_device_info = "CUDA GPU"
            else:
                pose_device_info = "CPU"
            
            logging.info(f"✅ Violation 모델: PyTorch ({violation_device_info})")
            logging.info(f"✅ Pose 모델: PyTorch ({pose_device_info})")

            return violation_model, pose_model, violation_trt, pose_trt
        except Exception as e:
            logging.error(f"YOLO 모델 초기화 실패: {e}", exc_info=True)
            return None, None, False, False

    def _initialize_face_recognition_models(self):
        face_model = None
        face_analyzer = None
        face_database = None
        face_uses_trt = False
        use_adaface = False
        adaface_model_path = None
        face_model_name = 'buffalo_l'  # 기본값 설정 (InsightFace detection 모듈용)

        # InsightFace가 없으면 얼굴 인식 기능 비활성화
        if not INSIGHTFACE_AVAILABLE:
            logging.warning("InsightFace가 설치되지 않아 얼굴 인식 기능이 비활성화됩니다.")
            logging.warning("설치 방법: .\\install_insightface.bat")
            return None, None, None, False, False, None

        try:
            # 1. YOLO 얼굴 감지 모델 로드 (PyTorch .pt 파일 직접 사용, MPS/CUDA 최적화)
            face_model, face_uses_trt = self._load_yolo_variant(
                config.Paths.YOLO_FACE_MODEL,
                config.Paths.YOLO_FACE_ENGINE,
                "Face",
                "detect"
            )
            
            if face_model is None:
                raise RuntimeError("YOLO 얼굴 감지 모델을 로드하지 못했습니다.")
            
            # PyTorch 모델 최적화: float() 및 .to() 호출
            underlying_face = getattr(face_model, "model", None)
            if underlying_face is not None:
                # float() 메서드로 모델을 float32로 변환 (MPS/CUDA 최적화)
                if not isinstance(underlying_face, str) and hasattr(underlying_face, "float"):
                    try:
                        underlying_face.float()
                    except (AttributeError, TypeError):
                        pass
            
            # 디바이스로 이동 (MPS/CUDA/CPU)
            try:
                face_model.to(self.device_face)
                # MPS 최적화: 추론 모드로 설정 (드롭아웃, 배치 정규화 최적화)
                if hasattr(face_model, 'eval'):
                    face_model.eval()
                logging.info(f"✅ Face 모델을 {self.device_face} 디바이스로 이동 완료 (추론 모드 활성화)")
            except (AttributeError, TypeError) as e:
                logging.warning(f"⚠️ Face 모델 디바이스 이동 실패: {e}")
            except Exception as e:
                logging.error(f"❌ Face 모델 디바이스 이동 오류: {e}")
            
            # 디바이스 정보 로깅
            device_str = str(self.device_face).upper()
            if 'mps' in device_str:
                device_info = "MPS (Metal GPU)"
            elif 'cuda' in device_str:
                gpu_id_face = int(self.device_face.split(':')[-1]) if ':' in str(self.device_face) else 0
                device_info = f"CUDA GPU {gpu_id_face}"
            else:
                device_info = "CPU"
            
            logging.info(f"✅ YOLO 얼굴 감지 모델 로드 완료: PyTorch ({device_info})")
            
            # 2. InsightFace 모델 경로 설정
            # InsightFace는 기본적으로 ~/.insightface/models/ 경로를 찾음
            # 로컬 모델 경로가 있으면 환경 변수로 설정
            insightface_models_dir = os.path.normpath(os.path.join(config.BASE_DIR, "../../models/insightface"))
            if os.path.exists(insightface_models_dir):
                # 환경 변수로 InsightFace 모델 경로 설정
                os.environ['INSIGHTFACE_ROOT'] = os.path.normpath(os.path.join(config.BASE_DIR, "../../models"))
                logging.info(f"InsightFace 모델 경로 설정: {insightface_models_dir}")
            else:
                # 기본 경로 사용 (사용자 홈 디렉토리)
                default_insightface_dir = os.path.expanduser("~/.insightface/models")
                logging.info(f"로컬 InsightFace 모델 경로 없음. 기본 경로 사용: {default_insightface_dir}")
            
            # 2. AdaFace 모델 지원 확인 (환경 변수로 활성화)
            # 기본값을 true로 변경하여 adaface_ir50_ms1mv2 모델 우선 사용
            use_adaface = os.getenv('USE_ADA_FACE', 'true').lower() == 'true'
            adaface_model_path = config.Paths.ADAFACE_MODEL if hasattr(config.Paths, 'ADAFACE_MODEL') else None
            
            # AdaFace 모델 파일 존재 여부 확인
            if use_adaface and adaface_model_path:
                if os.path.exists(adaface_model_path):
                    logging.info(f"✅ AdaFace 모델 파일 발견: {adaface_model_path}")
                else:
                    logging.warning(f"⚠️ AdaFace 모델 파일을 찾을 수 없습니다: {adaface_model_path}")
                    logging.info("💡 buffalo_l 모델로 폴백합니다.")
                    use_adaface = False
                    adaface_model_path = None
            else:
                use_adaface = False
                adaface_model_path = None
            
            # InsightFace는 항상 'buffalo_l' 모델을 사용 (detection 모듈용)
            # 실제 임베딩 추출은 FastIndustrialRecognizer를 통해 AdaFace를 사용할 수 있음
            face_model_name = 'buffalo_l'
            
            # 3. buffalo_L 모델 로드 (InsightFace - 얼굴 임베딩 추출용)
            # 시스템 흐름: yolov11n-face.pt로 얼굴 감지 → 얼굴 자르기 → buffalo_L로 임베딩 추출 → FAISS 매칭
            # AdaFace를 사용하는 경우: yolov11n-face.pt로 얼굴 감지 + 랜드마크 추출 → FastIndustrialRecognizer로 AdaFace 임베딩 추출
            import platform
            if 'cuda' in str(self.device_face):
                # CUDA 우선, 실패 시 CPU 폴백
                # GPU 1 사용 (멀티 GPU인 경우)
                gpu_id_face = int(self.device_face.split(':')[-1]) if ':' in str(self.device_face) else 0
                
                # InsightFace는 CUDAExecutionProvider를 사용하고 device_id 옵션으로 GPU 지정
                # ctx_id는 prepare()에서 사용하지만, CUDAExecutionProvider에도 device_id를 명시적으로 지정
                providers = [
                    ('CUDAExecutionProvider', {'device_id': gpu_id_face}),  # GPU ID 명시적 지정
                    'CPUExecutionProvider'
                ]
                ctx_id = gpu_id_face  # ctx_id로 특정 GPU 지정
                model_info = f"AdaFace 모델" if use_adaface else "buffalo_L 모델"
                logging.info(f"InsightFace GPU {gpu_id_face} 모드 활성화 ({model_info} - 임베딩 추출용, ctx_id={gpu_id_face}, device_id={gpu_id_face})")
            elif 'mps' in str(self.device_face) and platform.system() == 'Darwin':
                # Mac MPS 지원: CoreML Execution Provider 사용
                try:
                    import onnxruntime
                    available_providers = onnxruntime.get_available_providers()
                    if 'CoreMLExecutionProvider' in available_providers:
                        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                        ctx_id = 0  # MPS는 단일 GPU로 처리
                        model_info = f"AdaFace 모델" if use_adaface else "buffalo_L 모델"
                        logging.info(f"InsightFace MPS 모드 활성화 ({model_info} - 임베딩 추출용, CoreML 사용)")
                    else:
                        providers = ['CPUExecutionProvider']
                        ctx_id = -1
                        model_info = f"AdaFace 모델" if use_adaface else "buffalo_L 모델"
                        logging.warning(f"CoreML Provider를 사용할 수 없습니다. CPU 모드로 실행됩니다 ({model_info})")
                except ImportError:
                    providers = ['CPUExecutionProvider']
                    ctx_id = -1
                    model_info = f"AdaFace 모델" if use_adaface else "buffalo_L 모델"
                    logging.warning(f"onnxruntime를 찾을 수 없습니다. CPU 모드로 실행됩니다 ({model_info})")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1  # CPU 사용
                model_info = f"AdaFace 모델" if use_adaface else "buffalo_L 모델"
                logging.info(f"InsightFace CPU 모드 활성화 ({model_info} - 임베딩 추출용)")
            
            # InsightFace 초기화 (buffalo_l 모델 사용 - detection 모듈용)
            # 실제 얼굴 감지는 YOLO(yolov11n-face.pt)를 사용하지만, 
            # InsightFace는 detection 모듈이 있어야 정상 작동하므로 포함
            # AdaFace를 사용하는 경우에도 InsightFace는 'buffalo_l'을 사용하고, 
            # 실제 임베딩 추출은 FastIndustrialRecognizer를 통해 AdaFace를 사용
            face_analyzer = FaceAnalysis(
                name=face_model_name,  # 항상 'buffalo_l' (detection 모듈용)
                providers=providers,
                allowed_modules=['detection', 'recognition']  # detection 모듈 포함 (필수)
            )
            
            # det_size 설정 (실시간 처리 최적화)
            # ctx_id로 특정 GPU 지정 (멀티 GPU 지원)
            det_size = config.Thresholds.FACE_DETECTION_SIZE
            face_analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
            
            # 실제 사용 중인 Provider 확인
            try:
                actual_providers = face_analyzer.models['recognition'].session.get_providers()
                logging.info(f"{face_model_name} 모델(InsightFace) 로드 완료 (Provider: {actual_providers}, ctx_id={ctx_id})")
            except Exception as e:
                logging.info(f"{face_model_name} 모델(InsightFace) 로드 완료 (Provider: {providers}, ctx_id={ctx_id})")
            
            # AdaFace 모델 사용 여부 로깅
            if use_adaface and adaface_model_path:
                logging.info(f"✅ AdaFace 모델 활성화: {adaface_model_path}")
                logging.info("💡 실제 임베딩 추출은 FastIndustrialRecognizer를 통해 AdaFace를 사용합니다.")
            else:
                logging.info("✅ buffalo_l 모델 사용 (InsightFace 기본 모델)")

            # 3. FAISS 인덱스 로드 (face_index.faiss, face_index.faiss.labels.npy)
            # face_embeddings.npy는 FAISS 인덱스에 이미 포함되어 있음
            logging.info(f"🔍 FAISS 인덱스 경로 확인: {config.Paths.FAISS_INDEX} (존재: {os.path.exists(config.Paths.FAISS_INDEX)})")
            logging.info(f"🔍 FAISS 레이블 경로 확인: {config.Paths.FAISS_LABELS} (존재: {os.path.exists(config.Paths.FAISS_LABELS)})")
            face_index, face_labels = self._load_face_database(config.Paths.FAISS_INDEX)
            face_database = (face_index, face_labels)  # 튜플로 저장
            logging.info(f"✅ FAISS 데이터베이스 로드 완료: 인덱스={face_index.ntotal if face_index else 0}개, 레이블={len(face_labels) if face_labels is not None else 0}개")
            if face_index is None or (hasattr(face_index, 'ntotal') and face_index.ntotal == 0):
                logging.error(f"❌ FAISS 인덱스가 None이거나 비어있습니다! 얼굴 인식이 작동하지 않습니다.")
            if face_labels is None or len(face_labels) == 0:
                logging.error(f"❌ FAISS 레이블이 None이거나 비어있습니다! 얼굴 인식이 작동하지 않습니다.")

        except Exception as e:
            logging.error(f"얼굴 인식 시스템 초기화 실패: {e}", exc_info=True)
            face_model = None
            face_analyzer = None # 실패 시 None으로 설정
            face_database = None
            face_uses_trt = False
            use_adaface = False
            adaface_model_path = None

        return face_model, face_analyzer, face_database, face_uses_trt, use_adaface, adaface_model_path

    @staticmethod
    def _load_face_database(index_path: str) -> Tuple[Optional[object], Optional[np.ndarray]]:
        """
        FAISS 인덱스와 레이블 파일을 함께 로드
        
        로드하는 파일:
        - face_index.faiss: FAISS 인덱스 (face_embeddings.npy의 데이터가 포함됨)
        - face_index.faiss.labels.npy: 인물 이름 레이블
        
        Returns:
            (faiss_index, labels): FAISS 인덱스와 레이블 배열 튜플
        """
        try:
            # config에서 설정된 경로 사용 (face/data/face_index.faiss)
            if not os.path.exists(index_path):
                # 폴백 1: 프로젝트 루트에서 찾기 (하위 호환성)
                project_root_index = os.path.normpath(os.path.join(config.BASE_DIR, "../..", "face_index.faiss"))
                if os.path.exists(project_root_index):
                    index_path = project_root_index
                    logging.info(f"✅ FAISS 인덱스 발견 (프로젝트 루트): {index_path}")
                else:
                    # 폴백 2: face/data 폴더에서 찾기 (새 경로)
                    face_data_index = os.path.normpath(os.path.join(config.BASE_DIR, "../..", "face", "data", "face_index.faiss"))
                    if os.path.exists(face_data_index):
                        index_path = face_data_index
                        logging.info(f"✅ FAISS 인덱스 발견 (face/data): {index_path}")
                    else:
                        logging.warning(f"Faiss 인덱스 파일을 찾을 수 없습니다: {config.Paths.FAISS_INDEX}")
                        # 빈 인덱스 반환
                        empty_index = faiss.IndexFlatIP(512)
                        empty_labels = np.array([])
                        return empty_index, empty_labels

            # 레이블 파일 경로 찾기
            labels_path = config.Paths.FAISS_LABELS
            logging.info(f"FAISS 레이블 파일 경로 확인: {labels_path} (존재: {os.path.exists(labels_path)})")
            if not os.path.exists(labels_path):
                # 폴백 1: 프로젝트 루트에서 찾기 (하위 호환성)
                project_root_labels = os.path.normpath(os.path.join(config.BASE_DIR, "../..", "face_index.faiss.labels.npy"))
                if os.path.exists(project_root_labels):
                    labels_path = project_root_labels
                    logging.info(f"✅ FAISS 레이블 발견 (프로젝트 루트): {labels_path}")
                else:
                    # 폴백 2: face/data 폴더에서 찾기 (새 경로)
                    face_data_labels = os.path.normpath(os.path.join(config.BASE_DIR, "../..", "face", "data", "face_index.faiss.labels.npy"))
                    if os.path.exists(face_data_labels):
                        labels_path = face_data_labels
                        logging.info(f"✅ FAISS 레이블 발견 (face/data): {labels_path}")
                    else:
                        abs_path = os.path.abspath(labels_path)
                        logging.warning(f"Faiss 레이블 파일을 찾을 수 없습니다: {labels_path} (절대 경로: {abs_path})")
                        labels_path = None

            # 인덱스 로드
            dimension = 512  # InsightFace 임베딩 차원
            index = faiss.read_index(index_path)

            # FAISS 인덱스를 GPU로 이전 (기본 활성화: 성능 향상)
            # 얼굴 인식이 GPU 1을 사용하므로 FAISS도 GPU 1을 사용하여 부하 분산
            try:
                import torch as _torch
                if config.Thresholds.USE_FAISS_GPU and _torch.cuda.is_available():
                    # 얼굴 인식이 사용하는 GPU ID 확인 (멀티 GPU인 경우 GPU 1)
                    gpu_count = _torch.cuda.device_count()
                    faiss_gpu_id = 1 if gpu_count >= 2 else 0  # GPU 1 사용 (멀티 GPU인 경우)
                    
                    # faiss-gpu 패키지 확인
                    try:
                        gpu_res = faiss.StandardGpuResources()
                    except AttributeError:
                        # faiss-cpu만 설치된 경우
                        logging.warning("⚠️ FAISS GPU 기능을 사용할 수 없습니다 (faiss-cpu만 설치됨)")
                        logging.info("💡 FAISS GPU를 사용하려면 다음 중 하나를 선택하세요:")
                        logging.info("   1. Conda 사용: conda install -c pytorch faiss-gpu")
                        logging.info("   2. CPU 버전 계속 사용 (현재 설정)")
                        logging.info("   3. .env 파일에 USE_FAISS_GPU=0 설정하여 CPU 모드 명시")
                        raise AttributeError("faiss.StandardGpuResources not available (faiss-cpu only)")
                    
                    # 임시 메모리 사용 최소화로 안정성 향상
                    try:
                        gpu_res.setTempMemory(0)
                    except Exception:
                        pass
                    index = faiss.index_cpu_to_gpu(gpu_res, faiss_gpu_id, index)
                    logging.info(f"✅ FAISS 인덱스를 GPU {faiss_gpu_id}로 이전 완료 (성능 향상, USE_FAISS_GPU=1)")
                else:
                    logging.info("FAISS CPU 인덱스 사용 (USE_FAISS_GPU=0 또는 CUDA 비활성)")
            except AttributeError as attr_e:
                # faiss-gpu가 설치되지 않은 경우
                logging.warning(f"⚠️ FAISS GPU 이전 실패: {attr_e}")
                logging.info("💡 CPU 모드로 계속 실행됩니다 (성능 저하 가능)")
                logging.info("   FAISS GPU를 사용하려면 conda를 통해 faiss-gpu를 설치하세요")
            except Exception as gpu_e:
                logging.warning(f"⚠️ FAISS GPU 이전 실패, CPU 인덱스 사용: {gpu_e}")
                logging.info("💡 CPU 모드로 계속 실행됩니다 (성능 저하 가능)")

            # 레이블 로드
            labels = np.array([])
            if labels_path and os.path.exists(labels_path):
                labels = np.load(labels_path, allow_pickle=True)
                logging.info(f"✅ Faiss 인덱스 및 레이블 로드 완료. 인덱스={index.ntotal}개 임베딩, 레이블={len(labels)}개 포함.")
                if len(labels) == 0:
                    logging.error(f"❌ 레이블 파일이 비어있습니다! 얼굴 데이터베이스를 다시 구축해야 합니다.")
                elif index.ntotal != len(labels):
                    logging.warning(f"⚠️ 인덱스 크기({index.ntotal})와 레이블 크기({len(labels)})가 일치하지 않습니다!")
            else:
                logging.error(f"❌ Faiss 레이블 파일 없음. 인덱스만 로드: {index.ntotal}개 임베딩. 얼굴 인식이 작동하지 않습니다!")

            return index, labels
        except Exception as e:
            logging.error(f"Faiss 데이터베이스 로드 실패: {e}", exc_info=True)
            # 빈 인덱스 반환
            empty_index = faiss.IndexFlatIP(512)
            empty_labels = np.array([])
            return empty_index, empty_labels

    def cleanup(self):
        logging.info("SafetySystem 정리됨.")

    # --- 헬퍼 함수 (Static Methods) ---
    # 이 함수들은 server.py의 process_single_frame에서 호출됩니다.

    @staticmethod
    def _scale_boxes(
        boxes: Any, 
        w_scale: float, 
        h_scale: float, 
        names: Dict[int, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        scaled: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if boxes is None or len(boxes) == 0:
            return scaled

        for box in boxes:
            try:
                class_name = names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                scaled[class_name].append({
                    'bbox': (x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale),
                    'confidence': confidence
                })
            except Exception as e:
                logging.warning(f"박스 스케일링 중 오류: {e}")
        return scaled

    @staticmethod
    def _scale_poses(
        pose_result: Any, 
        w_scale: float, 
        h_scale: float, 
        orig_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        scaled: List[Dict[str, Any]] = []
        if pose_result.keypoints and pose_result.boxes is not None and pose_result.boxes.id is not None:
            tracker_ids = pose_result.boxes.id.int().cpu().numpy()
            for idx, (kpts, tracker_id) in enumerate(zip(pose_result.keypoints, tracker_ids)):
                try:
                    if torch.sum(kpts.conf > config.Thresholds.POSE_CONFIDENCE) >= config.Thresholds.MIN_VISIBLE_KEYPOINTS:
                        kpts_data = kpts.data.clone()
                        kpts_data[..., 0] *= w_scale
                        kpts_data[..., 1] *= h_scale
                        box = pose_result.boxes[idx].xyxy[0].cpu().numpy()
                        scaled_box = (box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale)

                        scaled.append({'keypoints': Keypoints(kpts_data, orig_shape), 'bbox_xyxy': scaled_box,
                                       'tracker_id': tracker_id})
                except Exception as e:
                    logging.warning(f"포즈 스케일링 중 오류: {e}")
        return scaled

    # ( ... _get_frame_batch, cleanup, _update_people_states, _draw_results, _create_grid_display 등등 ... 모두 삭제 ...)
    # ( ... _match_and_update_people, _check_fall_status, _check_safety_gear_status 등등 ... 모두 삭제 ...)
    # SafetySystem 클래스는 이제 모델 로딩과 헬퍼 함수(_scale_boxes, _scale_poses)만 제공합니다.
