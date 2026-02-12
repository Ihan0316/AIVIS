# fast_face_recognizer.py - 고속 얼굴 인식 모듈
"""
랜드마크 기반 고속 얼굴 인식
YOLO Face의 랜드마크를 직접 사용하여 InsightFace의 Detection 단계 생략
"""
import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Any
import os

try:
    from insightface.utils import face_align
    INSIGHTFACE_UTILS_AVAILABLE = True
except ImportError:
    INSIGHTFACE_UTILS_AVAILABLE = False
    logging.warning("insightface.utils를 찾을 수 없습니다. face_align을 사용할 수 없습니다.")

try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logging.warning("onnxruntime를 찾을 수 없습니다. 직접 ONNX 추론을 사용할 수 없습니다.")

# TensorRT 지원 (ONNX보다 12배 빠름)
try:
    import tensorrt as trt
    import torch
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class FastIndustrialRecognizer:
    """
    고속 얼굴 인식 클래스
    YOLO Face의 랜드마크를 사용하여 InsightFace의 Detection 단계를 생략
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        ctx_id: int = 0,
        use_adaface: bool = False
    ):
        """
        :param model_path: ONNX 모델 경로 (None이면 InsightFace 기본 모델 사용)
        :param ctx_id: GPU ID (0, 1, ... 또는 -1 for CPU)
        :param use_adaface: AdaFace 모델 사용 여부 (True면 AdaFace, False면 기존 buffalo_l)
        """
        self.ctx_id = ctx_id
        self.use_adaface = use_adaface
        self.session = None
        self.input_name = None
        self.output_name = None
        self.use_direct_onnx = False
        
        # TensorRT 관련
        self.use_tensorrt = False
        self.trt_context = None
        self.trt_input_tensor = None
        self.trt_output_tensor = None
        self.trt_output_div = None
        
        # TensorRT 엔진 우선 로드 (ONNX 대비 2배 빠름: 21ms → 10ms)
        if model_path and TENSORRT_AVAILABLE:
            # .engine 파일이 직접 전달된 경우
            if model_path.endswith('.engine'):
                engine_path = model_path
            else:
                engine_path = model_path.replace('.onnx', '.engine')
            
            if os.path.exists(engine_path):
                try:
                    self._init_tensorrt(engine_path, ctx_id)
                except Exception as e:
                    logging.warning(f"⚠️ TensorRT 초기화 실패, ONNX로 폴백: {e}")
        
        # TensorRT 실패 시 ONNX 추론 사용
        if not self.use_tensorrt and model_path and os.path.exists(model_path) and ONNXRUNTIME_AVAILABLE:
            try:
                # ONNX Runtime 세션 생성
                available_providers = onnxruntime.get_available_providers()
                providers = []
                
                # CUDA 사용 가능한 경우 (GPU 2대 최대 활용)
                if ctx_id >= 0 and 'CUDAExecutionProvider' in available_providers:
                    # CUDA Execution Provider 최적화 옵션
                    cuda_options = {
                        'device_id': ctx_id,
                        'arena_extend_strategy': 'kNextPowerOfTwo',  # 메모리 할당 최적화
                        'gpu_mem_limit': 10 * 1024 * 1024 * 1024,  # 10GB 제한 (11GB GPU)
                        'cudnn_conv_algo_search': 'DEFAULT',  # EXHAUSTIVE -> DEFAULT (안정성 및 초기화 속도 향상)
                        'do_copy_in_default_stream': True,  # 스트림 최적화
                    }
                    providers.append(('CUDAExecutionProvider', cuda_options))
                    logging.info(f"✅ CUDA GPU {ctx_id} 감지: CUDA Execution Provider 사용 (최적화 활성화)")
                
                # CPU는 항상 폴백으로 추가
                providers.append('CPUExecutionProvider')
                
                # 세션 옵션 설정 (GPU 2대 최대 활용 - GPU 사용률 극대화)
                sess_options = onnxruntime.SessionOptions()
                # GPU 사용 시 병렬 처리 활성화 (GPU 사용률 극대화)
                if ctx_id >= 0 and 'CUDAExecutionProvider' in available_providers:
                    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL  # SEQUENTIAL -> PARALLEL (GPU 사용률 극대화)
                    sess_options.intra_op_num_threads = 4  # 2 -> 4 (GPU 병렬 처리 증가)
                    sess_options.inter_op_num_threads = 4  # 2 -> 4 (GPU 병렬 처리 증가)
                else:
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
                # 메모리 최적화 활성화
                sess_options.enable_mem_pattern = True
                sess_options.enable_cpu_mem_arena = True
                # 그래프 최적화 레벨 (모든 최적화 활성화)
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.session = onnxruntime.InferenceSession(
                    model_path, 
                    providers=providers,
                    sess_options=sess_options
                )
                
                # 실제 사용 중인 Provider 확인
                active_providers = self.session.get_providers()
                logging.info(f"✅ FastIndustrialRecognizer: 직접 ONNX 모델 로드 완료 ({model_path})")
                logging.info(f"🔍 활성화된 ONNX Providers: {active_providers}")
                
                if 'CUDAExecutionProvider' in active_providers:
                    logging.info("✅ CUDA 가속 활성화됨")
                else:
                    logging.warning("⚠️ GPU Provider가 활성화되지 않았습니다! CPU로 실행됩니다.")
                
                # 입력/출력 이름 확인
                inputs = self.session.get_inputs()
                outputs = self.session.get_outputs()
                self.input_name = inputs[0].name
                # AdaFace ONNX 모델은 출력이 2개 (output, onnx::Div_704) - 첫 번째 출력만 사용
                self.output_name = outputs[0].name
                
                # 입력/출력 형식 로깅
                logging.info(f"🔍 AdaFace ONNX 모델 정보:")
                logging.info(f"   입력: {inputs[0].name}, shape={inputs[0].shape}, type={inputs[0].type}")
                logging.info(f"   출력: {outputs[0].name}, shape={outputs[0].shape}, type={outputs[0].type}")
                if len(outputs) > 1:
                    logging.debug(f"   추가 출력 (무시): {outputs[1].name}, shape={outputs[1].shape}")
                
                self.use_direct_onnx = True
                logging.info(f"✅ FastIndustrialRecognizer: AdaFace ONNX 모델 로드 완료 ({model_path})")
            except Exception as e:
                logging.warning(f"⚠️ 직접 ONNX 모델 로드 실패, InsightFace 사용: {e}")
                self.use_direct_onnx = False
        else:
            logging.info("FastIndustrialRecognizer: InsightFace 기본 모델 사용")
    
    def _init_tensorrt(self, engine_path: str, ctx_id: int):
        """TensorRT 엔진 초기화 (PyTorch CUDA 메모리 사용)"""
        # GPU 디바이스 설정 (TensorRT 초기화 전에 설정!)
        device_id = ctx_id if ctx_id >= 0 else 0
        device = f'cuda:{device_id}'
        
        # CUDA 디바이스 명시적 설정 및 동기화
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize(device_id)
        
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            raise RuntimeError("TensorRT 엔진 로드 실패")
        
        self.trt_context = engine.create_execution_context()
        self.trt_engine = engine  # 엔진 참조 유지 (GC 방지)
        self.trt_device_id = device_id
        
        # 버퍼 미리 할당 (재사용으로 속도 향상)
        self.trt_input_tensor = torch.empty((1, 3, 112, 112), dtype=torch.float32, device=device).contiguous()
        self.trt_output_tensor = torch.empty((1, 512), dtype=torch.float32, device=device).contiguous()
        self.trt_output_div = torch.empty((1, 1), dtype=torch.float32, device=device).contiguous()
        self.trt_stream = torch.cuda.Stream(device=device)
        
        # 초기화 후 동기화
        torch.cuda.synchronize(device_id)
        
        self.use_tensorrt = True
        logging.info(f"✅ AdaFace TensorRT 엔진 로드 완료: {engine_path} (GPU {device_id})")
        logging.info(f"   (ONNX 대비 12배 빠름: 27ms → 2ms)")
    
    def get_embedding_fast(
        self, 
        frame: np.ndarray, 
        kps: np.ndarray,
        face_analyzer: Optional[Any] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        YOLO Face의 랜드마크(kps)를 이용해 즉시 정렬 및 임베딩 추출 (Detection 생략)
        
        :param frame: 원본 프레임 (Crop된 이미지가 아님! 원본에서 좌표로 자르는게 더 정확함)
        :param kps: YOLO Face가 리턴한 5개 랜드마크 좌표 [[x1,y1], ... [x5,y5]]
                    형식: (5, 2) numpy array 또는 list of [x, y]
        :param face_analyzer: InsightFace 분석기 (직접 ONNX를 사용하지 않는 경우)
        
        :return: (embedding, aligned_face) 튜플
                 - embedding: 512차원 임베딩 벡터 (정규화됨) 또는 None
                 - aligned_face: 정렬된 얼굴 이미지 (112x112) 또는 None
        """
        try:
            # 랜드마크 형식 변환 및 검증
            if kps is None or len(kps) < 5:
                return None, None
            
            # kps를 numpy array로 변환
            if isinstance(kps, list):
                kps = np.array(kps, dtype=np.float32)
            elif not isinstance(kps, np.ndarray):
                kps = np.array(kps, dtype=np.float32)
            
            # 형식 변환: (5, 2) 또는 (10,) -> (5, 2)
            if kps.shape == (10,):
                kps = kps.reshape(5, 2)
            elif len(kps.shape) != 2 or kps.shape[1] != 2:
                logging.warning(f"랜드마크 형식이 올바르지 않습니다: {kps.shape}")
                return None, None
            
            # 5개 랜드마크 확인 (왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입꼬리, 오른쪽 입꼬리)
            if kps.shape[0] < 5:
                return None, None
            
            # Face Alignment (Affine Transformation)
            # 위에서 30도 각도로 촬영된 얼굴을 위한 개선된 정렬
            # 1. 먼저 2D 정렬 수행 (평면 회전 보정)
            # 2. 위에서 본 각도(pitch) 보정을 위한 추가 전처리
            if not INSIGHTFACE_UTILS_AVAILABLE:
                # 경고는 한 번만 출력 (매 프레임마다 출력 방지)
                if not getattr(self, '_insightface_warning_shown', False):
                    logging.info("ℹ️ insightface.utils 없음 → 기본 정렬(_simple_align) 사용")
                    self._insightface_warning_shown = True
                # 기본 정렬 (간단한 크롭)
                aligned_face = self._simple_align(frame, kps)
            else:
                try:
                    # standard output size: 112x112
                    aligned_face = face_align.norm_crop(frame, kps)
                    
                    # 위에서 본 각도 보정: 얼굴이 위에서 30도 각도로 촬영된 경우
                    # 코와 눈의 수직 위치 차이를 분석하여 pitch 각도 추정
                    if aligned_face is not None and kps.shape[0] >= 3:
                        # 코 위치 (kps[2])
                        nose_y = kps[2][1]
                        # 눈 위치 평균 (kps[0], kps[1])
                        eye_y = (kps[0][1] + kps[1][1]) / 2.0
                        # 코와 눈의 수직 거리
                        vertical_diff = nose_y - eye_y
                        
                        # 위에서 본 각도가 크면 (코가 눈보다 아래에 있으면) pitch 보정 필요
                        # 일반적으로 정면 얼굴에서는 코가 눈보다 약간 아래에 있지만,
                        # 위에서 본 각도가 크면 이 차이가 더 커짐
                        if vertical_diff > 0:  # 코가 눈보다 아래에 있음
                            # 위에서 본 각도 보정: 얼굴을 약간 위로 회전시킨 것처럼 보정
                            # perspective transformation을 사용하여 얼굴을 정면으로 보정
                            aligned_face = self._correct_pitch_angle(aligned_face, vertical_diff)
                            
                except Exception as e:
                    logging.warning(f"face_align.norm_crop 실패, 기본 정렬 사용: {e}")
                    aligned_face = self._simple_align(frame, kps)
            
            if aligned_face is None or aligned_face.size == 0:
                return None, None
            
            # 화질 개선: 대비 향상 (CLAHE) - build_database와 동일한 YCrCb 색공간 사용
            try:
                face_size = max(aligned_face.shape[0], aligned_face.shape[1])
                if face_size >= 100:
                    # YCrCb 색공간 사용 (build_database와 동일)
                    ycrcb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    y = clahe.apply(y)
                    ycrcb = cv2.merge([y, cr, cb])
                    aligned_face = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            except Exception:
                pass  # 오류 발생 시 원본 사용
            
            # 임베딩 추출 (TensorRT > ONNX > InsightFace 순서, 자동 폴백)
            embedding = None
            
            if self.use_tensorrt and self.trt_context is not None:
                embedding = self._get_embedding_from_tensorrt(aligned_face)
            
            # TensorRT 실패 시 ONNX로 폴백
            if embedding is None and self.use_direct_onnx and self.session is not None:
                embedding = self._get_embedding_from_onnx(aligned_face)
            
            # ONNX도 실패 시 InsightFace로 폴백
            if embedding is None and face_analyzer is not None:
                embedding = self._get_embedding_from_insightface(aligned_face, face_analyzer)
            
            if embedding is None:
                return None, None
            
            if embedding is None:
                return None, None
            
            # Normalize Embedding (L2 Norm) - Cosine Similarity를 위해 필수
            norm_val = np.linalg.norm(embedding)
            if norm_val > 0:
                embedding = embedding / norm_val
            else:
                return None, None
            
            return embedding, aligned_face
            
        except Exception as e:
            logging.debug(f"get_embedding_fast 오류: {e}")
            return None, None
    
    def _correct_pitch_angle(self, aligned_face: np.ndarray, vertical_diff: float) -> np.ndarray:
        """
        위에서 본 각도(pitch) 보정: 얼굴을 정면으로 보정
        :param aligned_face: 정렬된 얼굴 이미지 (112x112)
        :param vertical_diff: 코와 눈의 수직 거리 차이
        :return: 보정된 얼굴 이미지
        """
        try:
            h, w = aligned_face.shape[:2]
            
            # 위에서 본 각도가 크면 (vertical_diff가 크면) 얼굴을 약간 위로 회전시킨 것처럼 보정
            # perspective transformation을 사용하여 얼굴 상단을 약간 확대하고 하단을 약간 축소
            # 이렇게 하면 위에서 본 각도가 줄어든 것처럼 보임
            
            # 보정 강도: vertical_diff가 클수록 더 강한 보정
            # 일반적으로 정면 얼굴에서 코-눈 거리는 얼굴 높이의 약 10-15% 정도
            # 위에서 30도 각도로 촬영되면 이 거리가 20-30% 정도로 증가
            correction_strength = min(0.15, vertical_diff / h)  # 최대 15% 보정
            
            if correction_strength < 0.05:  # 보정이 너무 작으면 스킵
                return aligned_face
            
            # Perspective transformation을 위한 source points
            src_points = np.float32([
                [0, 0],           # 왼쪽 상단
                [w, 0],           # 오른쪽 상단
                [w, h],           # 오른쪽 하단
                [0, h]            # 왼쪽 하단
            ])
            
            # Destination points: 상단을 약간 확대하고 하단을 약간 축소
            offset = int(w * correction_strength * 0.3)  # 보정 오프셋
            dst_points = np.float32([
                [offset, 0],           # 왼쪽 상단 (약간 오른쪽으로)
                [w - offset, 0],       # 오른쪽 상단 (약간 왼쪽으로)
                [w - offset, h],       # 오른쪽 하단
                [offset, h]            # 왼쪽 하단
            ])
            
            # Perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Transform 적용
            corrected_face = cv2.warpPerspective(aligned_face, M, (w, h), 
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REPLICATE)
            
            return corrected_face
            
        except Exception as e:
            logging.debug(f"pitch 각도 보정 실패 (원본 사용): {e}")
            return aligned_face
    
    def _simple_align(self, frame: np.ndarray, kps: np.ndarray) -> Optional[np.ndarray]:
        """
        ArcFace 표준 정렬 (face_align.norm_crop과 동일한 방식)
        InsightFace utils를 사용할 수 없을 때 대체 구현
        """
        try:
            # ArcFace 표준 랜드마크 위치 (112x112 기준)
            # 왼쪽눈, 오른쪽눈, 코, 왼쪽입꼬리, 오른쪽입꼬리
            arcface_dst = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
            
            # 입력 랜드마크 (5개)
            src_pts = kps[:5].astype(np.float32)
            
            # Similarity Transform 계산 (회전, 스케일, 이동)
            # cv2.estimateAffinePartial2D는 similarity transform을 계산
            tform, _ = cv2.estimateAffinePartial2D(src_pts, arcface_dst, method=cv2.LMEDS)
            
            if tform is None:
                # fallback: 단순 affine transform
                tform = cv2.getAffineTransform(src_pts[:3], arcface_dst[:3])
            
            if tform is None:
                logging.warning("_simple_align: Transform 계산 실패")
                return None
            
            # Affine Transform 적용하여 112x112로 정렬
            aligned_face = cv2.warpAffine(
                frame, tform, (112, 112), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            if aligned_face is None or aligned_face.size == 0:
                return None
            
            return aligned_face
            
        except Exception as e:
            logging.error(f"_simple_align 오류: {e}", exc_info=True)
            return None
    
    def _get_embedding_from_onnx(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        ONNX 모델로 직접 임베딩 추출
        AdaFace 전처리 방식: BGR 이미지, [0, 255] -> [0, 1] -> [-1, 1] 정규화
        """
        # ONNX 세션이 없으면 (TensorRT만 사용 중) None 반환
        if self.session is None:
            return None
        
        try:
            # AdaFace 전처리 방식 적용
            # aligned_face는 BGR 형식 (OpenCV 기본)
            # 1. [0, 255] -> [0, 1] 정규화
            np_img = aligned_face.astype(np.float32) / 255.0
            
            # 2. [0, 1] -> [-1, 1] 정규화: ((img / 255.) - 0.5) / 0.5
            np_img = (np_img - 0.5) / 0.5
            
            # 3. BGR 순서 유지 (AdaFace는 BGR 입력 사용)
            # 4. (H, W, C) -> (1, C, H, W) 변환
            tensor = np_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
            
            # Inference (Only Recognition)
            import time
            t0 = time.time()
            
            # 입력 텐서 형식 확인 (디버깅)
            logging.debug(f"AdaFace ONNX 추론 입력: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min():.3f}, max={tensor.max():.3f}")
            
            # ONNX 추론 실행
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            t1 = time.time()
            
            # 추론 시간 측정
            inference_time_ms = (t1 - t0) * 1000
            logging.debug(f"AdaFace ONNX 추론 완료: {inference_time_ms:.2f}ms")
            
            # GPU 2대 최대 활용: 경고 임계값 조정 (실제 병목은 150ms 이상)
            if inference_time_ms > 150:
                logging.warning(f"⚠️ AdaFace 추론 느림: {inference_time_ms:.1f}ms (목표: <100ms)")
            
            embedding = outputs[0]
            
            # 출력 형식 확인 (디버깅)
            logging.debug(f"AdaFace ONNX 추론 출력: shape={embedding.shape}, dtype={embedding.dtype}")
            
            # Flatten (이미 (1, 512) 형태일 수 있음)
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # 최종 형식 확인
            if embedding.shape[0] != 512:
                logging.error(f"❌ AdaFace 임베딩 차원 오류: 예상=512, 실제={embedding.shape[0]}")
                return None
            
            return embedding
            
        except Exception as e:
            logging.error(f"_get_embedding_from_onnx 오류: {e}", exc_info=True)
            return None
    
    def _get_embedding_from_tensorrt(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        TensorRT 엔진으로 직접 임베딩 추출 (ONNX 대비 12배 빠름)
        실패 시 자동으로 ONNX로 폴백
        """
        try:
            # 올바른 CUDA 디바이스로 전환
            if hasattr(self, 'trt_device_id'):
                torch.cuda.set_device(self.trt_device_id)
            
            # AdaFace 전처리
            np_img = aligned_face.astype(np.float32) / 255.0
            np_img = (np_img - 0.5) / 0.5
            np_img = np_img.transpose(2, 0, 1)  # HWC -> CHW
            
            # 입력 텐서에 복사 (동기화 포함)
            input_tensor = torch.from_numpy(np_img).unsqueeze(0).to(self.trt_input_tensor.device)
            self.trt_input_tensor.copy_(input_tensor)
            torch.cuda.synchronize(self.trt_device_id)
            
            # TensorRT 실행
            self.trt_context.set_tensor_address('input', self.trt_input_tensor.data_ptr())
            self.trt_context.set_tensor_address('output', self.trt_output_tensor.data_ptr())
            self.trt_context.set_tensor_address('onnx::Div_704', self.trt_output_div.data_ptr())
            
            with torch.cuda.stream(self.trt_stream):
                success = self.trt_context.execute_async_v3(self.trt_stream.cuda_stream)
            
            # 완전한 동기화
            self.trt_stream.synchronize()
            torch.cuda.synchronize(self.trt_device_id)
            
            if not success:
                logging.warning("TensorRT 추론 실패, ONNX로 폴백")
                self.use_tensorrt = False  # 다음부터 ONNX 사용
                return None
            
            embedding = self.trt_output_tensor.cpu().numpy()[0]
            return embedding
            
        except RuntimeError as e:
            # CUDA 에러 특별 처리
            error_msg = str(e)
            if "CUDA" in error_msg or "illegal memory" in error_msg:
                logging.error(f"❌ CUDA 메모리 에러 발생: {e}")
                logging.warning("TensorRT 비활성화, ONNX로 폴백")
                self.use_tensorrt = False
                # CUDA 상태 복구 시도
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except:
                    pass
            return None
        except Exception as e:
            logging.warning(f"TensorRT 오류, ONNX로 폴백: {e}")
            self.use_tensorrt = False  # 다음부터 ONNX 사용
            return None
    
    def _get_embedding_from_insightface(
        self, 
        aligned_face: np.ndarray, 
        face_analyzer: Any
    ) -> Optional[np.ndarray]:
        """
        InsightFace 분석기로 임베딩 추출
        """
        try:
            # rec_model 접근
            rec_model = None
            if hasattr(face_analyzer, 'models') and 'recognition' in face_analyzer.models:
                rec_model = face_analyzer.models['recognition']
            elif hasattr(face_analyzer, 'rec_model'):
                rec_model = face_analyzer.rec_model
            
            if rec_model is None:
                logging.warning("rec_model을 찾을 수 없습니다")
                return None
            
            # 이미 정렬된 얼굴 이미지에 대해 직접 임베딩 추출
            embedding = rec_model.get_feat(aligned_face)
            
            if embedding is not None:
                # 정규화 (L2 norm)
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logging.error(f"_get_embedding_from_insightface 오류: {e}", exc_info=True)
            return None


