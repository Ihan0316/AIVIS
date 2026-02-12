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

# NOTE: conda 환경에서 실행되므로 수동 경로 추가 불필요 (충돌 방지)
# conda activate aivis-gpu 후 실행하면 자동으로 올바른 site-packages가 설정됨

# onnxruntime을 먼저 import (InsightFace보다 먼저 로드하여 DLL 충돌 방지)
ONNXRUNTIME_AVAILABLE = False
try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
    logging.info(f"✅ onnxruntime {onnxruntime.__version__} import 성공")
    logging.info(f"   Providers: {onnxruntime.get_available_providers()}")
except ImportError as e:
    logging.warning(f"⚠️ onnxruntime import 실패: {e}")
except Exception as e:
    logging.warning(f"⚠️ onnxruntime import 중 예외: {type(e).__name__}: {e}")

# faiss 임포트 (conda 경로 추가 후)
try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("⚠️ FAISS를 찾을 수 없습니다. 얼굴 인식 기능이 제한될 수 있습니다.")

# InsightFace는 선택적 (설치 실패 시 얼굴 인식 기능 비활성화)
# 주의: onnxruntime이 먼저 로드되어야 InsightFace가 정상 작동할 수 있음
try:
    print("[DEBUG] InsightFace import 시도...")
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    print("[DEBUG] ✅ InsightFace FaceAnalysis import 성공")
    logging.info("✅ InsightFace FaceAnalysis import 성공")
except ImportError as e:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None
    print(f"[DEBUG] ❌ InsightFace ImportError: {e}")
    logging.warning(f"insightface 모듈을 찾을 수 없습니다 (ImportError: {e}). 얼굴 인식 기능이 비활성화됩니다.")
    logging.warning("설치 방법: .\\install_insightface.bat")
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None
    print(f"[DEBUG] ❌ InsightFace Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    logging.error(f"InsightFace import 중 예외 발생: {type(e).__name__}: {e}")
    logging.warning("설치 방법: .\\install_insightface.bat")

# TensorRT는 선택적 (엔진 파일 로드 시 필요)
# conda 경로 추가 후에도 TensorRT를 찾지 못할 수 있으므로, 추가 경로 확인
TENSORRT_AVAILABLE = False
try:
    import tensorrt
    TENSORRT_AVAILABLE = True
    logging.info("✅ TensorRT Python 패키지 사용 가능")
except ImportError:
    # conda 환경의 TensorRT 확인 (sys.path에 추가되기 전일 수 있음)
    try:
        # conda 경로에서 직접 확인
        conda_site_packages = os.path.join(os.environ.get('USERPROFILE', ''), 'anaconda3', 'Lib', 'site-packages')
        if os.path.exists(conda_site_packages):
            tensorrt_path = os.path.join(conda_site_packages, 'tensorrt')
            if os.path.exists(tensorrt_path):
                # conda 경로를 sys.path에 추가하고 재시도
                if conda_site_packages not in sys.path:
                    sys.path.insert(0, conda_site_packages)
                import tensorrt
                TENSORRT_AVAILABLE = True
                logging.info("✅ TensorRT Python 패키지 사용 가능 (conda 환경에서 찾음)")
    except Exception as e:
        TENSORRT_AVAILABLE = False
    
    if not TENSORRT_AVAILABLE:
        logging.info("ℹ️ TensorRT Python 패키지가 없습니다. TensorRT 엔진 파일 대신 ONNX 모델을 사용합니다.")
        logging.info("  TensorRT를 사용하려면: pip install nvidia-tensorrt (CUDA와 호환되는 버전 필요)")

import config
from utils import calculate_iou, clip_bbox_xyxy, is_person_horizontal, log_violation



class SafetySystem:
    def __init__(self):
        # 0. 성능 최적화 설정 (ONNX Runtime 기반)
        # ONNX Runtime과 PyTorch 모두 확인
        try:
            import onnxruntime as ort
            onnx_providers = ort.get_available_providers()
            has_cuda_provider = 'CUDAExecutionProvider' in onnx_providers
        except:
            has_cuda_provider = False
            onnx_providers = []
        
        # PyTorch CUDA 확인 (호환성을 위해)
        pytorch_cuda_available = False
        gpu_count = 0
        try:
            if torch.cuda.is_available():
                pytorch_cuda_available = True
                gpu_count = torch.cuda.device_count()
        except:
            pass
        
        # ONNX Runtime GPU 사용 가능 여부 확인
        if has_cuda_provider:
            logging.info(f"✅ ONNX Runtime CUDA Provider 사용 가능")
            logging.info(f"  - 사용 가능한 Providers: {onnx_providers}")
            if pytorch_cuda_available:
                logging.info(f"  - PyTorch CUDA도 사용 가능 (GPU 개수: {gpu_count})")
                # PyTorch 최적화 설정 (호환성을 위해)
                torch.backends.cudnn.benchmark = True
                if gpu_count >= 2:
                    for i in range(gpu_count):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                    logging.info(f"✅ 멀티 GPU ({gpu_count}개) 최적화 설정 완료")
        else:
            logging.warning("⚠️ ONNX Runtime CUDA Provider를 사용할 수 없습니다.")
            logging.info(f"  - 사용 가능한 Providers: {onnx_providers}")
            if pytorch_cuda_available:
                logging.info(f"  - PyTorch CUDA는 사용 가능하지만, ONNX Runtime은 CPU로 실행됩니다.")
            else:
                logging.warning("  - CPU 모드로 실행됩니다.")

        # 1. 장치 설정 (멀티 GPU 지원)
        self.device_config = config.SystemConfig.get_device_config()
        self.device = self.device_config['device']  # GPU 0: YOLO Violation, Pose
        self.device_face = self.device_config.get('device_face', self.device)  # GPU 1: YOLO Face, InsightFace
        self.gpu_count = self.device_config.get('gpu_count', 0)
        
        # GPU 강제 사용 시도 (ONNX Runtime 기반)
        # ONNX Runtime CUDA Provider 확인
        try:
            import onnxruntime as ort
            onnx_providers = ort.get_available_providers()
            has_onnx_cuda = 'CUDAExecutionProvider' in onnx_providers
        except:
            has_onnx_cuda = False
        
        if self.device == 'cpu':
            # GPU가 감지되지 않았지만 GPU 사용 시도
            try:
                # ONNX Runtime CUDA Provider 확인
                if has_onnx_cuda:
                    self.device = 'cuda:0'
                    self.device_face = 'cuda:0' if self.gpu_count < 2 else 'cuda:1'
                    logging.warning("⚠️ ONNX Runtime CUDA Provider 사용 가능. GPU로 강제 전환합니다.")
                elif torch.cuda.is_available():
                    # PyTorch CUDA도 확인 (호환성)
                    self.device = 'cuda:0'
                    self.device_face = 'cuda:0' if self.gpu_count < 2 else 'cuda:1'
                    logging.warning("⚠️ PyTorch CUDA 사용 가능. GPU로 강제 전환합니다.")
                else:
                    # CUDA가 없으면 CPU 유지
                    logging.warning("⚠️ CUDA를 사용할 수 없습니다. CPU 모드로 계속 진행합니다.")
                    self.device = 'cpu'
                    self.device_face = 'cpu'
            except Exception as e:
                logging.warning(f"⚠️ GPU 강제 사용 시도 중 오류: {e}. CPU 모드로 계속 진행합니다.")
                self.device = 'cpu'
                self.device_face = 'cpu'
        
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
        
        # 넘어짐 감지 모델 로드 (선택적) - TensorRT 엔진 우선
        self.fall_model = None
        self.fall_uses_trt = False
        try:
            fall_engine_path = config.Paths.YOLO_FALL_DETECTION_ENGINE
            fall_model_path = config.Paths.YOLO_FALL_DETECTION_MODEL
            
            # TensorRT 엔진 우선 로드 (2배 빠름: 24ms → 12ms)
            if os.path.exists(fall_engine_path):
                logging.info(f"🔍 Fall TensorRT Engine 로드 시도: {fall_engine_path}")
                self.fall_model = YOLO(fall_engine_path, task='detect')
                self.fall_uses_trt = True
                logging.info("✅ 넘어짐 감지 모델 로드 완료 (TensorRT)")
            elif os.path.exists(fall_model_path):
                logging.info(f"🔍 Fall Detection 모델 로드 시도: {fall_model_path}")
                self.fall_model = YOLO(fall_model_path, task='detect')
                if torch.cuda.is_available():
                    self.fall_model.to('cuda:0')
                    logging.info("✅ 넘어짐 감지 모델 로드 완료 (GPU)")
                else:
                    logging.info("✅ 넘어짐 감지 모델 로드 완료 (CPU)")
            else:
                logging.info(f"ℹ️ 넘어짐 감지 모델 파일 없음")
                logging.info("   (키포인트 기반 분석만 사용)")
        except Exception as e:
            logging.warning(f"⚠️ 넘어짐 감지 모델 로드 중 오류 (계속 진행): {e}")
            self.fall_model = None
        
        # FastIndustrialRecognizer 초기화 (랜드마크 기반 고속 처리용)
        self.fast_recognizer = None
        # CUDA 디바이스에 맞는 ctx_id 설정
        if 'cuda' in str(self.device_face):
            ctx_id_face = int(self.device_face.split(':')[-1]) if ':' in str(self.device_face) else 0
        else:
            ctx_id_face = -1  # CPU
        
        # AdaFace 모델이 있으면 InsightFace 없이도 FastIndustrialRecognizer 사용 가능
        if self.use_adaface and self.adaface_model_path and os.path.exists(self.adaface_model_path):
            try:
                from fast_face_recognizer import FastIndustrialRecognizer
                self.fast_recognizer = FastIndustrialRecognizer(
                    model_path=self.adaface_model_path,
                    ctx_id=ctx_id_face,
                    use_adaface=True
                )
                logging.info(f"✅ FastIndustrialRecognizer 초기화 완료 (AdaFace 모델: {self.adaface_model_path})")
            except Exception as e:
                logging.warning(f"⚠️ FastIndustrialRecognizer 초기화 실패: {e}")
                self.fast_recognizer = None
        elif self.face_analyzer is not None:
            # AdaFace가 없고 InsightFace가 있으면 buffalo_l 사용
            try:
                from fast_face_recognizer import FastIndustrialRecognizer
                self.fast_recognizer = FastIndustrialRecognizer(
                    model_path=None,  # InsightFace 기본 모델 사용
                    ctx_id=ctx_id_face,
                    use_adaface=False
                )
                logging.info(f"✅ FastIndustrialRecognizer 초기화 완료 (buffalo_l 모델, 랜드마크 기반 처리)")
            except Exception as e:
                logging.warning(f"⚠️ FastIndustrialRecognizer 초기화 실패: {e}")
                self.fast_recognizer = None
        else:
            logging.warning("⚠️ FastIndustrialRecognizer 사용 불가: AdaFace 모델 또는 InsightFace 필요")

        if self.violation_model is None or self.pose_model is None:
            logging.error("필수 모델(Violation or Pose) 로딩에 실패했습니다.")
        else:
             logging.info("YOLO 모델 로딩 완료.")

        # 🦬 buffalo_l만 사용: face_model은 None이어도 됨!
        # face_analyzer와 face_database만 있으면 얼굴 인식 가능
        if self.face_analyzer is None or self.face_database is None:
            logging.warning("=" * 80)
            logging.warning("⚠️  얼굴 인식 모델 또는 DB 로딩에 실패했습니다.")
            logging.warning("⚠️  얼굴 인식 기능이 비활성화됩니다.")
            if not INSIGHTFACE_AVAILABLE:
                logging.warning("⚠️  InsightFace 모듈이 설치되지 않았습니다.")
                logging.warning("⚠️  설치 방법: .\\install_insightface.bat")
            logging.warning("=" * 80)
        else:
            logging.info("✅ 얼굴 인식 모델 및 DB 로딩 완료 (buffalo_l + InsightFace 임베딩).")

    def _load_yolo_variant(self, weight_path: str, engine_path: str, task_description: str, task_type: str) -> Tuple[Optional[YOLO], bool]:
        """
        YOLO 모델 로드 (TensorRT Engine 우선, 없으면 ONNX 모델 사용)
        
        :param weight_path: ONNX 모델 경로 (.onnx)
        :param engine_path: TensorRT 엔진 경로 (.engine)
        :param task_description: 작업 설명 (로깅용)
        :param task_type: 작업 타입 ('detect', 'pose', 'segment')
        :return: (모델, TensorRT 사용 여부) 튜플
        """
        # 1. TensorRT Engine 파일이 있으면 우선 사용
        # TensorRT Python 패키지가 없어도 Ultralytics가 내부적으로 로드 시도할 수 있음
        if engine_path and os.path.exists(engine_path):
            logging.info(f"🔍 {task_description} 엔진 파일 발견: {engine_path}")
            try:
                logging.info(f"{task_description} TensorRT Engine 로드 시도: {engine_path}")
                model = YOLO(engine_path, task=task_type)
                
                # 로드 성공 확인 (속성 접근)
                _ = model.names
                
                # TensorRT 엔진은 GPU에서만 실행되므로 device 설정 불필요
                logging.info(f"✅ {task_description} TensorRT Engine 로드 완료 (GPU 최적화)")
                
                # TensorRT GPU 사용 확인 (실제 GPU 메모리 확인)
                try:
                    if torch.cuda.is_available():
                        gpu_id = 0 if task_description != "Face" else (1 if torch.cuda.device_count() >= 2 else 0)
                        gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                        gpu_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                        logging.info(f"🔍 {task_description} TensorRT 로드 후 GPU {gpu_id} 메모리: 할당={gpu_mem:.3f}GB, 예약={gpu_reserved:.3f}GB")
                        logging.warning(f"⚠️ 참고: TensorRT는 PyTorch CUDA 메모리와 별도로 작동하므로, 실제 GPU 사용률은 nvidia-smi로 확인하세요.")
                except Exception as mem_e:
                    logging.debug(f"{task_description} GPU 메모리 확인 중 오류: {mem_e}")
                
                return model, True  # TensorRT 사용
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"⚠️ {task_description} TensorRT Engine 로드 실패: {error_msg}")
                logging.info(f"   {task_description} ONNX 모델로 대체합니다.")
        else:
            logging.info(f"ℹ️ {task_description} 엔진 파일이 없습니다: {engine_path}")
        
        # 2. ONNX 모델 로드 (Engine이 없거나 로드 실패 시)
        # weight_path가 .onnx로 끝나지 않으면 .onnx 확장자 추가
        if not weight_path.endswith('.onnx'):
            onnx_path = os.path.splitext(weight_path)[0] + ".onnx"
        else:
            onnx_path = weight_path
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"{task_description} 모델 파일 없음 (Engine: {engine_path}, ONNX: {onnx_path})")
        
        try:
            logging.info(f"{task_description} ONNX 모델 로드: {onnx_path}")
            
            # GPU 사용을 위한 ONNX Runtime 세션 옵션 설정
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                
                # GPU 사용 가능 여부 확인
                if 'CUDAExecutionProvider' in available_providers:
                    # GPU ID 결정
                    if task_description == "Face":
                        gpu_id = 1 if torch.cuda.device_count() >= 2 else 0
                    else:
                        gpu_id = 0
                    
                    # ONNX Runtime 환경 변수 설정 (YOLO가 내부적으로 사용)
                    os.environ['ORT_DEVICE'] = 'cuda'
                    os.environ['ORT_EXECUTION_PROVIDERS'] = 'CUDAExecutionProvider;CPUExecutionProvider'
                    os.environ['ORT_CUDA_DEVICE_ID'] = str(gpu_id)
                    
                    logging.info(f"{task_description} ONNX Runtime GPU 옵션 설정 완료 (GPU {gpu_id})")
                else:
                    logging.warning(f"{task_description} ONNX Runtime: CUDA Provider를 사용할 수 없어 CPU로 실행됩니다.")
            except Exception as opt_e:
                logging.warning(f"{task_description} ONNX Runtime 옵션 설정 중 오류 (계속 진행): {opt_e}")
            
            # YOLO ONNX 모델 로드 (YOLO가 내부적으로 ONNX Runtime 사용)
            model = YOLO(onnx_path, task=task_type)
            
            # YOLO가 내부적으로 device를 설정하지 않도록 명시적으로 device 속성 설정
            # ONNX 모델은 ONNX Runtime이 device를 처리하므로, YOLO의 device 속성을 None 또는 'cpu'로 설정
            try:
                # YOLO 모델의 device 속성을 명시적으로 설정하여 내부 device 설정 방지
                if hasattr(model, 'device'):
                    # ONNX 모델은 device를 사용하지 않으므로 None으로 설정
                    model.device = None
                elif hasattr(model, 'overrides'):
                    # YOLO v8+ 스타일: overrides에 device를 명시적으로 설정하지 않음
                    if 'device' in model.overrides:
                        del model.overrides['device']
                # YOLO 모델의 predictor device 설정도 확인
                if hasattr(model, 'predictor') and hasattr(model.predictor, 'device'):
                    model.predictor.device = None
            except Exception as device_set_e:
                logging.debug(f"{task_description} YOLO device 속성 설정 중 오류 (계속 진행): {device_set_e}")
            
            logging.info(f"{task_description} ONNX 모델 로드 완료")
            
            # ✅ GPU 사용 여부 및 입력/출력 확인
            try:
                import onnxruntime as ort
                
                # 먼저 ONNX Runtime의 사용 가능한 Providers 확인
                available_providers = ort.get_available_providers()
                logging.info(f"🔍 {task_description} ONNX Runtime 사용 가능한 Providers: {available_providers}")
                
                if 'CUDAExecutionProvider' not in available_providers:
                    logging.warning(f"⚠️ {task_description} ONNX Runtime: CUDAExecutionProvider가 설치되지 않았습니다.")
                    logging.warning(f"   GPU를 사용하려면: pip install onnxruntime-gpu")
                    logging.warning(f"   현재 사용 가능한 Providers: {available_providers}")
                
                # YOLO 모델의 내부 ONNX Runtime 세션 찾기
                session_obj = None
                session_path = None
                
                # 여러 가능한 경로로 세션 찾기
                if hasattr(model, 'model'):
                    # 경로 1: model.model.session
                    if hasattr(model.model, 'session'):
                        try:
                            session_obj = model.model.session
                            session_path = 'model.model.session'
                        except:
                            pass
                    
                    # 경로 2: model.model.predictor.session
                    if session_obj is None and hasattr(model.model, 'predictor'):
                        try:
                            if hasattr(model.model.predictor, 'session'):
                                session_obj = model.model.predictor.session
                                session_path = 'model.model.predictor.session'
                        except:
                            pass
                    
                    # 경로 3: model.model.overrides['session']
                    if session_obj is None and hasattr(model.model, 'overrides'):
                        try:
                            if isinstance(model.model.overrides, dict) and 'session' in model.model.overrides:
                                session_obj = model.model.overrides.get('session')
                                if session_obj:
                                    session_path = 'model.model.overrides["session"]'
                        except:
                            pass
                    
                    # 경로 4: 재귀적으로 session 속성 찾기
                    if session_obj is None:
                        def find_session(obj, path="", depth=0):
                            if depth > 3:  # 최대 3단계 깊이
                                return None, None
                            if hasattr(obj, 'get_providers'):
                                return obj, path
                            for attr_name in dir(obj):
                                if attr_name.startswith('_') or attr_name in ['session']:
                                    continue
                                try:
                                    attr = getattr(obj, attr_name)
                                    if attr is None or isinstance(attr, (str, int, float, bool)):
                                        continue
                                    if 'session' in attr_name.lower():
                                        found_session, found_path = find_session(attr, f"{path}.{attr_name}" if path else attr_name, depth+1)
                                        if found_session:
                                            return found_session, found_path
                                except:
                                    pass
                            return None, None
                        
                        found_session, found_path = find_session(model.model, "model.model")
                        if found_session:
                            session_obj = found_session
                            session_path = found_path
                
                if session_obj:
                    # 1. GPU 사용 여부 확인
                    actual_providers = session_obj.get_providers()
                    logging.info(f"🔍 {task_description} ONNX 모델 세션 정보:")
                    logging.info(f"   세션 경로: {session_path}")
                    logging.info(f"   활성화된 Providers: {actual_providers}")
                    
                    if 'CUDAExecutionProvider' in actual_providers:
                        # CUDA Provider가 첫 번째인지 확인 (우선순위)
                        if actual_providers[0] == 'CUDAExecutionProvider':
                            logging.info(f"✅ {task_description} ONNX 모델: GPU 사용 중 (CUDAExecutionProvider 우선순위 1)")
                        else:
                            logging.warning(f"⚠️ {task_description} ONNX 모델: CUDA Provider는 있지만 우선순위가 낮음 (현재: {actual_providers[0]})")
                    else:
                        logging.warning(f"⚠️ {task_description} ONNX 모델: CPU로 실행 중 (CUDA Provider 없음)")
                    
                    # 2. 입력/출력 shape 확인
                    try:
                        inputs = session_obj.get_inputs()
                        outputs = session_obj.get_outputs()
                        
                        logging.info(f"🔍 {task_description} ONNX 모델 입력/출력 정보:")
                        for i, inp in enumerate(inputs):
                            logging.info(f"   입력[{i}]: name={inp.name}, shape={inp.shape}, type={inp.type}")
                        
                        for i, out in enumerate(outputs):
                            logging.info(f"   출력[{i}]: name={out.name}, shape={out.shape}, type={out.type}")
                        
                        # 입력 shape 검증
                        if len(inputs) > 0:
                            input_shape = inputs[0].shape
                            if len(input_shape) == 4:  # [batch, channels, height, width]
                                expected_h = 832 if task_description != "Face" else 640
                                expected_w = 832 if task_description != "Face" else 640
                                # 동적 shape인 경우 (None 또는 -1)는 체크하지 않음
                                if input_shape[2] not in [None, -1] and input_shape[3] not in [None, -1]:
                                    if input_shape[2] != expected_h or input_shape[3] != expected_w:
                                        logging.warning(f"⚠️ {task_description} 모델 입력 크기 불일치: 예상={expected_h}x{expected_w}, 실제={input_shape[2]}x{input_shape[3]}")
                                    else:
                                        logging.info(f"✅ {task_description} 모델 입력 크기 확인: {input_shape[2]}x{input_shape[3]}")
                    except Exception as io_e:
                        logging.warning(f"⚠️ {task_description} 모델 입력/출력 확인 중 오류: {io_e}")
                    
                    # 3. GPU 메모리 변화 확인 (실제 GPU 사용 여부 검증)
                    if 'CUDAExecutionProvider' in actual_providers and torch.cuda.is_available():
                        try:
                            gpu_id = 0 if task_description != "Face" else (1 if torch.cuda.device_count() >= 2 else 0)
                            gpu_mem_before = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
                            
                            # 더미 추론 실행
                            dummy_input = np.random.randn(1, 3, 832 if task_description != "Face" else 640, 832 if task_description != "Face" else 640).astype(np.float32)
                            input_name = inputs[0].name if inputs else 'images'
                            _ = session_obj.run(None, {input_name: dummy_input})
                            
                            gpu_mem_after = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
                            mem_increase = gpu_mem_after - gpu_mem_before
                            
                            if mem_increase > 0.1:  # 0.1MB 이상 증가하면 GPU 사용 중
                                logging.info(f"✅ {task_description} ONNX 모델: GPU 실제 사용 확인됨 (메모리 증가: {mem_increase:.2f}MB)")
                            else:
                                logging.warning(f"⚠️ {task_description} ONNX 모델: GPU 메모리 변화 없음 ({mem_increase:.2f}MB) - CPU에서 실행 중일 수 있음")
                        except Exception as mem_check_error:
                            logging.debug(f"{task_description} GPU 메모리 확인 중 오류 (무시): {mem_check_error}")
                else:
                    logging.warning(f"⚠️ {task_description} ONNX 모델: 세션 객체를 찾을 수 없음 (YOLO 내부 구조 확인 필요)")
            except Exception as check_e:
                logging.warning(f"⚠️ {task_description} ONNX 모델 검증 중 오류 (계속 진행): {check_e}")
            
            return model, False  # TensorRT 사용 안 함
        except Exception as e:
            logging.error(f"{task_description} ONNX 모델 로드 실패: {e}", exc_info=True)
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

            # CUDA 디바이스 사용 (GPU 강제 사용)
            pose_device = self.device
            
            # GPU로 강제 이동 시도
            target_device = self.device
            if target_device == 'cpu':
                # CPU로 설정되어 있어도 GPU 사용 시도
                if torch.cuda.is_available():
                    target_device = 'cuda:0'
                    pose_device = 'cuda:0'
                    logging.info("🔄 GPU 감지됨. YOLO Violation/Pose 모델을 GPU로 이동합니다.")
                else:
                    # CUDA가 실제로 사용 불가능하면 CPU로 유지
                    target_device = 'cpu'
                    pose_device = 'cpu'
                    logging.warning("⚠️ CUDA가 사용 불가능합니다. CPU 모드로 계속 진행합니다.")
            
            # ONNX 모델은 YOLO가 내부적으로 ONNX Runtime을 사용하므로
            # .to() 메서드 호출이 필요 없고, 호출하면 YOLO가 내부적으로 device를 설정하여 오류 발생 가능
            # ONNX Runtime은 모델 로드 시 이미 GPU/CPU Provider를 설정했으므로 추가 설정 불필요
            if not violation_trt:
                try:
                    # ONNX 모델은 .to() 메서드를 호출하지 않음 (YOLO가 내부적으로 device 설정 시도하여 오류 발생)
                    # eval()만 호출 (있을 경우)
                    if hasattr(violation_model, 'eval'):
                        violation_model.eval()
                    logging.info(f"✅ Violation ONNX 모델 로드 완료 (ONNX Runtime이 자동으로 GPU/CPU 처리)")
                except Exception as e:
                    logging.debug(f"Violation 모델 설정: {e} (ONNX 모델은 내부적으로 처리됨)")

            if not pose_trt:
                try:
                    # ONNX 모델은 .to() 메서드를 호출하지 않음
                    if hasattr(pose_model, 'eval'):
                        pose_model.eval()
                    logging.info(f"✅ Pose ONNX 모델 로드 완료 (ONNX Runtime이 자동으로 GPU/CPU 처리)")
                except Exception as e:
                    logging.debug(f"Pose 모델 설정: {e} (ONNX 모델은 내부적으로 처리됨)")

            if 'cuda' in str(self.device) and (not violation_trt or not pose_trt):
                if torch.cuda.is_available():
                    # GPU 0 정보 가져오기
                    gpu_id = int(self.device.split(':')[-1]) if ':' in str(self.device) else 0
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
                    logging.info(f"GPU {gpu_id} ({gpu_name}) 최적화: YOLO Violation/Pose 모델 실행 (메모리: {gpu_memory:.1f}GB)")
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    logging.info("✅ cuDNN 최적화 활성화")

            # 모델 타입 로깅
            violation_device_str = str(self.device).upper()
            if 'cuda' in violation_device_str:
                violation_device_info = "CUDA GPU"
            else:
                violation_device_info = "CPU"
            
            pose_device_str = str(pose_device).upper()
            if 'cuda' in pose_device_str:
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
        face_model_name = 'buffalo_l'  # buffalo_l 사용 (InsightFace)
        face_model = None  # YOLO Face 대신 buffalo_l 사용
        
        # ⭐ buffalo_l 사용 (얼굴 감지 + 임베딩 통합!)
        # AdaFace, YOLO Face 대신 InsightFace buffalo_l 사용
        logging.info("🦬 buffalo_l 모델 사용 (얼굴 감지 + 임베딩 통합)")
        use_adaface = False
        adaface_model_path = None

        # FAISS 인덱스 로드 (InsightFace 없이도 사용 가능)
        face_database = None
        try:
            index_path = config.Paths.FAISS_INDEX
            face_database = self._load_face_database(index_path)
            if face_database and face_database[0] is not None:
                logging.info(f"✅ FAISS 인덱스 로드 완료: {index_path}")
            else:
                logging.warning("⚠️ FAISS 인덱스 로드 실패 또는 없음")
                face_database = None
        except Exception as e:
            logging.warning(f"⚠️ FAISS 인덱스 로드 실패: {e}")
            face_database = None

        # InsightFace가 없으면 AdaFace + FAISS로 얼굴 인식 시도
        if not INSIGHTFACE_AVAILABLE:
            logging.warning("InsightFace가 설치되지 않아 buffalo_l 모델을 사용할 수 없습니다.")
            if use_adaface and face_database:
                logging.info("✅ AdaFace + FAISS로 얼굴 인식 기능을 대체합니다.")
            else:
                logging.warning("얼굴 감지는 YOLO 모델로 작동합니다. 이름 인식은 비활성화됩니다.")
            return face_model, None, face_database, face_uses_trt, use_adaface, adaface_model_path

        try:
            # InsightFace buffalo_l 관련 초기화 (얼굴 감지 + 임베딩 통합)
            # NOTE: buffalo_l만 사용하므로 YOLO Face 모델 체크 제거
            # face_model은 None이어도 됨 (buffalo_l이 얼굴 감지 + 임베딩 모두 처리)
            
            # PyTorch 모델 최적화 (YOLO Face 사용 시에만)
            underlying_face = getattr(face_model, "model", None) if face_model else None
            if underlying_face is not None:
                # float() 메서드로 모델을 float32로 변환 (CUDA 최적화)
                if not isinstance(underlying_face, str) and hasattr(underlying_face, "float"):
                    try:
                        underlying_face.float()
                    except (AttributeError, TypeError):
                        pass
            
            # ONNX 모델은 YOLO가 내부적으로 ONNX Runtime을 사용하므로
            # device 설정은 YOLO API 호환성을 위해 유지
            target_device_face = self.device_face
            
            # ONNX Runtime CUDA Provider 확인
            try:
                import onnxruntime as ort
                onnx_providers = ort.get_available_providers()
                has_onnx_cuda = 'CUDAExecutionProvider' in onnx_providers
            except:
                has_onnx_cuda = False
            
            if target_device_face == 'cpu' and has_onnx_cuda:
                target_device_face = 'cuda:0'
                logging.info("🔄 ONNX Runtime CUDA Provider 감지됨. YOLO Face 모델이 GPU를 사용합니다.")
            
            # YOLO Face 모델 설정 (buffalo_l만 사용 시 스킵)
            if face_model is not None:
                try:
                    # ONNX 모델도 YOLO 래퍼를 통해 .to() 메서드 지원 가능
                    face_model.to(target_device_face)
                    if hasattr(face_model, 'eval'):
                        face_model.eval()
                    logging.info(f"✅ Face ONNX 모델 로드 완료 (디바이스: {target_device_face})")
                except (AttributeError, TypeError) as e:
                    # ONNX 모델은 .to() 메서드가 없을 수 있음 (정상)
                    logging.debug(f"Face 모델 디바이스 설정: {e} (ONNX 모델은 내부적으로 처리됨)")
                except Exception as e:
                    logging.warning(f"⚠️ Face 모델 설정 중 오류 (계속 진행): {e}")
            else:
                logging.info("🦬 YOLO Face 모델 없음 - buffalo_l로 얼굴 감지 처리")
            
            # ✅ Face 모델도 GPU 사용 여부 및 입력/출력 확인 (이미 _load_yolo_variant에서 확인했지만 재확인)
            try:
                import onnxruntime as ort
                session_obj = None
                if hasattr(face_model, 'model'):
                    if hasattr(face_model.model, 'session'):
                        session_obj = face_model.model.session
                    elif hasattr(face_model.model, 'predictor') and hasattr(face_model.model.predictor, 'session'):
                        session_obj = face_model.model.predictor.session
                
                if session_obj:
                    face_providers = session_obj.get_providers()
                    if 'CUDAExecutionProvider' in face_providers and face_providers[0] == 'CUDAExecutionProvider':
                        logging.info(f"✅ Face ONNX 모델: GPU 사용 확인됨")
                    else:
                        logging.warning(f"⚠️ Face ONNX 모델: GPU 미사용 또는 우선순위 낮음 (Providers: {face_providers})")
            except Exception as face_check_e:
                logging.debug(f"Face 모델 검증 중 오류 (무시): {face_check_e}")
            
            # 디바이스 정보 로깅
            device_str = str(self.device_face).upper()
            if 'cuda' in device_str:
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
            
            # 2. buffalo_l 모델만 사용 (AdaFace 비활성화)
            # 🦬 buffalo_l로 통합: 얼굴 감지 + 임베딩 모두 buffalo_l 사용
            use_adaface = False
            adaface_model_path = None
            logging.info("🦬 buffalo_l 모델로 얼굴 감지 + 임베딩 통합")
            
            # InsightFace는 항상 'buffalo_l' 모델을 사용 (detection 모듈용)
            # 실제 임베딩 추출은 FastIndustrialRecognizer를 통해 AdaFace를 사용할 수 있음
            face_model_name = 'buffalo_l'
            
            # 3. buffalo_L 모델 로드 (InsightFace - 얼굴 임베딩 추출용)
            # 시스템 흐름: yolov11n-face.pt로 얼굴 감지 → 얼굴 자르기 → buffalo_L로 임베딩 추출 → FAISS 매칭
            # AdaFace를 사용하는 경우: yolov11n-face.pt로 얼굴 감지 + 랜드마크 추출 → FastIndustrialRecognizer로 AdaFace 임베딩 추출
            if 'cuda' in str(self.device_face):
                # CUDA 우선, 실패 시 CPU 폴백
                # 🦬 buffalo_l: GPU 0 고정 (GPU 1에서 CAM-0 감지 실패 문제)
                gpu_id_face = 0  # GPU 0 사용 - 안정적으로 작동
                
                # ⭐ TensorRT 우선 → CUDA → CPU 순서로 Provider 설정
                # TensorRT 엔진(.engine)이 있으면 TensorRT 사용, 없으면 CUDA 폴백
                trt_options = {
                    'device_id': gpu_id_face,
                    'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                    'trt_fp16_enable': True,  # FP16 가속
                }
                cuda_options = {
                    'device_id': gpu_id_face,
                    'arena_extend_strategy': 'kNextPowerOfTwo',  # 메모리 할당 최적화
                    'gpu_mem_limit': 10 * 1024 * 1024 * 1024,  # 10GB 제한 (11GB GPU)
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # 최적 알고리즘 검색
                    'do_copy_in_default_stream': True,  # 스트림 최적화
                }
                
                # TensorRT Provider가 사용 가능한지 확인
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                
                # 🦬 buffalo_l: TensorRT 비활성화 (안정성 + 빠른 warmup)
                # TensorRT는 매번 엔진을 빌드해서 30초+ warmup이 필요함
                # CUDA만 사용하면 2-3초로 단축됨
                providers = []
                # TensorRT 비활성화 (주석 처리)
                # if 'TensorrtExecutionProvider' in available_providers:
                #     providers.append(('TensorrtExecutionProvider', trt_options))
                #     logging.info(f"🚀 TensorRT Provider 활성화 (GPU {gpu_id_face})")
                
                providers.append(('CUDAExecutionProvider', cuda_options))
                providers.append('CPUExecutionProvider')
                logging.info(f"🦬 buffalo_l: CUDA Provider 사용 (TensorRT 비활성화 - 안정성 향상)")
                
                ctx_id = gpu_id_face  # ctx_id로 특정 GPU 지정
                logging.info(f"🦬 InsightFace buffalo_l GPU {gpu_id_face} 모드 활성화 (ctx_id={gpu_id_face})")
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
            
            # 실제 사용 중인 Provider 확인 및 입력/출력 확인
            try:
                rec_session = face_analyzer.models['recognition'].session
                actual_providers = rec_session.get_providers()
                logging.info(f"{face_model_name} 모델(InsightFace) 로드 완료 (Provider: {actual_providers}, ctx_id={ctx_id})")
                
                # InsightFace 모델 입력/출력 확인
                try:
                    rec_inputs = rec_session.get_inputs()
                    rec_outputs = rec_session.get_outputs()
                    logging.info(f"🔍 InsightFace 모델 입력/출력 정보:")
                    for i, inp in enumerate(rec_inputs):
                        logging.info(f"   입력[{i}]: name={inp.name}, shape={inp.shape}, type={inp.type}")
                    for i, out in enumerate(rec_outputs):
                        logging.info(f"   출력[{i}]: name={out.name}, shape={out.shape}, type={out.type}")
                except Exception as io_e:
                    logging.debug(f"InsightFace 모델 입력/출력 확인 중 오류: {io_e}")
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
