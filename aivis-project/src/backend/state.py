# state.py - 전역 상태 관리
"""
전역 변수 및 상태 관리 모듈
모든 전역 상태를 중앙에서 관리합니다.
"""
import threading
import time
import queue
from typing import Dict, Set, Optional, Any
from collections import defaultdict

import os
import logging
import asyncio
from aiohttp import web
from cache_manager import IdentityCache, TTLCache
from concurrent.futures import ThreadPoolExecutor
import config
import torch
import threading

# 프레임 저장
latest_frames: Dict[int, bytes] = {}  # 처리된 프레임 저장
latest_result_data: Dict[int, dict] = {}  # 최신 결과 데이터 저장 (대시보드용)
frame_lock = threading.Lock()

# 프레임 처리 동시성 제어
processing_lock = threading.Lock()  # 프레임 처리 동시성 제어용
processing_flags: Dict[int, bool] = {}  # cam_id별 처리 중 플래그

# 프레임 큐 시스템 (최신 프레임 우선 처리, 딜레이 최소화)
frame_queues: Dict[int, queue.Queue] = {}  # cam_id별 프레임 큐
# 프레임 유지율 최대화: 큐 크기 증가 (MPS 환경 최적화: 10 -> 20, 프레임 드롭 방지)
MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', '20'))  # 10 -> 20 (MPS 환경 프레임 유지율 향상)
queue_lock = threading.Lock()

# 프레임 간격 제어 (튐 방지)
last_frame_processed_time: Dict[int, float] = {}  # cam_id별 마지막 프레임 처리 시간
# 프레임 유지율 최대화: 프레임 간격 최소화 (모든 프레임 처리)
MIN_FRAME_INTERVAL = 1.0 / 30.0  # 최소 프레임 간격 (30 FPS 기준, 약 33.33ms - 프레임 드롭 방지)
frame_interval_lock = threading.Lock()

# WebSocket 연결 관리
connected_websockets: Set[web.WebSocketResponse] = set()
dashboard_websockets: Set[web.WebSocketResponse] = set()  # 대시보드 전용 연결

# SafetySystem 및 StorageManager (나중에 초기화됨)
safety_system_instance: Optional[Any] = None  # core.SafetySystem 타입
safety_system_lock = threading.Lock()  # SafetySystem 접근용 락 (멀티스레드 안전성)
storage_manager: Optional[Any] = None  # LocalStorageManager 타입

# 프레임 처리 태스크 추적 (WebSocket 연결별)
processing_tasks: Dict[int, Dict[int, asyncio.Task]] = {}  # {cam_id: {client_id: task}}
processing_tasks_lock = asyncio.Lock()  # 태스크 추적용 락

# 대시보드 브로드캐스트 제어 (깜빡거림 완화용)
DASHBOARD_BROADCAST_INTERVAL = float(os.getenv("DASHBOARD_BROADCAST_INTERVAL", "0.5"))
last_dashboard_broadcast_ts = 0.0
last_dashboard_payload = ""
dashboard_broadcast_lock: Optional[asyncio.Lock] = None

# 프레임 추적 통계 (실시간 FPS 측정용)
frame_stats: Dict[int, dict] = {}  # cam_id별 프레임 통계
frame_stats_lock = threading.Lock()

# 얼굴 탐지 최적화를 위한 프레임 추적 (CCTV 효율 인식 설정)
last_face_detection_frame: Dict[int, int] = {}  # cam_id별 마지막 얼굴 탐지 프레임 번호
face_detection_lock = threading.Lock()
face_recognition_cooldown_ts: Dict[int, float] = defaultdict(lambda: 0.0)

# 최근 식별 결과 캐시 (라벨 안정화) - 메모리 누수 방지
# IdentityCache 사용: 자동 크기 제한 및 TTL 관리
MAX_IDENTITY_CACHE_PER_CAM = 30  # 50 -> 30 (멀티캠 메모리 최적화)
recent_identity_cache = IdentityCache(
    max_items_per_cam=MAX_IDENTITY_CACHE_PER_CAM,
    ttl=config.Thresholds.RECOGNITION_HOLD_SECONDS
)

# 마지막 렌더링된 박스/라벨 캐시 (렌더링 보강)
# TTLCache 사용: 자동 만료 처리
# cam_id -> TTLCache['render' -> {'items': List[{box: (x1,y1,x2,y2), name: str}]}]
last_render_cache: Dict[int, TTLCache] = defaultdict(lambda: TTLCache(default_ttl=2.0))

# 센트로이드 임베딩 버퍼 (final의 개선 기법 도입)
# cam_id -> person_box_key -> {'embeddings': [embedding1, ...], 'last_update': timestamp}
# person_box_key는 IoU 기반으로 같은 사람을 식별하는 키
embedding_buffers: Dict[int, Dict[str, dict]] = defaultdict(dict)
EMBEDDING_BUFFER_SIZE = 5  # 3 -> 5개 프레임 평균 (정확도 향상)
EMBEDDING_BUFFER_MIN_SIZE = 2  # 최소 2개 있어야 센트로이드 계산 (빠른 인식을 위해 2로 조정)
MAX_EMBEDDING_BUFFERS_PER_CAM = 20  # 50 -> 20 (멀티캠 메모리 최적화)

# 넘어짐 감지 시간 추적 (final의 개선 기법 도입)
# cam_id -> person_box_key -> fall_start_time
fall_start_times: Dict[int, Dict[str, float]] = defaultdict(dict)
FALL_DURATION_THRESHOLD = 0.5  # final과 동일: 0.5초 지속 시 넘어짐 판정

# 센트로이드 결과 캐시 (재사용으로 성능 향상)
# TTLCache 사용: 자동 만료 처리
# cam_id -> TTLCache[person_box_key -> {'name': str, 'score': float}]
CENTROID_CACHE_TTL = 2.0  # 2초간 캐시 유지
centroid_cache: Dict[int, TTLCache] = defaultdict(lambda: TTLCache(default_ttl=CENTROID_CACHE_TTL))

# 얼굴 바운딩박스 캐시 (깜빡임 방지)
# TTLCache 사용: 자동 만료 처리
# cam_id -> TTLCache[person_box_key -> {'face_bbox': (x1,y1,x2,y2), 'person_box': (x1,y1,x2,y2)}]
FACE_BBOX_CACHE_TTL = 2.0  # 1.0 -> 2.0초 (바운딩 박스 안정화, 깜빡임 방지, 홀드 시간과 통일)
face_bbox_cache: Dict[int, TTLCache] = defaultdict(lambda: TTLCache(default_ttl=FACE_BBOX_CACHE_TTL))

# 모델 결과 데이터 (final과 동일한 구조)
model_results = {
    "alerts": [],
    "violations": {},
    "heatmap_counts": {"A-1": 0, "A-2": 0, "B-1": 0, "B-2": 0},
    "profile": {"name": "시스템", "status": "정상", "area": "전체"},
    "logs": [],
    "kpi_data": {"totalWorkers": 0, "attendees": 0, "ppeRate": 0, "riskLevel": 0},
    "detected_workers": {}  # 구역별 감지된 작업자 정보
}
results_lock = threading.Lock()

# 중복 알림 방지를 위한 최근 알림 추적
# 심각한 위반(넘어짐, 사고)은 즉시 알림, PPE 위반은 쿨다운 적용
recent_alerts_cache: Dict[str, float] = {}  # key: "{worker}|{area}|{violation_types}", value: timestamp
ALERT_COOLDOWN_SECONDS = 30.0  # PPE 위반 쿨다운 (30초)
CRITICAL_VIOLATIONS = ["넘어짐", "사고", "FALL", "ACCIDENT"]  # 즉시 알림 위반 (쿨다운 없음)

# 시스템 상태 모니터링
system_stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "error_count": 0,
    "last_health_check": time.time(),
    "memory_usage": 0,
    "cpu_usage": 0,
    "response_times": [],  # 응답 시간 추적 (최근 100개)
    "gpu_stats": {}  # GPU 사용량 통계
}
stats_lock = threading.Lock()
MAX_RESPONSE_TIMES = 100  # 최근 응답 시간 최대 저장 개수

# 프레임 처리 상태 관리 (cam_id별) - 함수 속성 대신 중앙 관리
# 메모리 누수 방지 및 멀티스레드 안전성 향상
frame_processing_state: Dict[int, Dict[str, Any]] = defaultdict(dict)
frame_processing_state_lock = threading.Lock()

def get_frame_processing_state(cam_id: int) -> Dict[str, Any]:
    """cam_id별 프레임 처리 상태 가져오기 (없으면 초기화)"""
    with frame_processing_state_lock:
        if cam_id not in frame_processing_state:
            frame_processing_state[cam_id] = {
                'frame_count': 0,
                'used_ppe_boxes': set(),
                'last_cleanup_time': 0.0,
                'perf_log_count': 0,
                'model_warmed_up': False
            }
        return frame_processing_state[cam_id]

def clear_frame_processing_state(cam_id: int) -> None:
    """cam_id별 프레임 처리 상태 초기화"""
    with frame_processing_state_lock:
        if cam_id in frame_processing_state:
            frame_processing_state[cam_id].clear()
            frame_processing_state[cam_id] = {
                'frame_count': 0,
                'used_ppe_boxes': set(),
                'last_cleanup_time': 0.0,
                'perf_log_count': 0,
                'model_warmed_up': False
            }

# ThreadPoolExecutor 인스턴스 (AI 모델 병렬 실행용)

# GPU 메모리에 따라 동적으로 워커 수 계산 (main.py와 동일한 로직)
def _calculate_optimal_workers():
    """GPU 메모리에 따라 최적의 워커 수를 계산합니다."""
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            
            if gpu_count >= 2:
                # 멀티 GPU: 각 GPU의 메모리를 합산하여 계산
                total_memory_gb = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count))
                avg_memory_gb = total_memory_gb / gpu_count
                
                # 멀티 GPU: GPU 여유 활용하여 성능 개선
                # GPU 사용률이 낮으므로 워커 수 대폭 증가 (GPU 활용률 향상)
                # 처리 시간(97ms) > 프레임 간격(33ms)이므로 더 많은 워커 필요
                if avg_memory_gb >= 10:
                    # 워커 수가 너무 많으면 CPU 오버헤드(Context Switching) 발생 -> 8개로 최적화
                    face_workers = 8   # 16 → 8 (최적화)
                    yolo_workers = 8   # 16 → 8 (최적화)
                    danger_workers = 6  # 10 → 6
                    frame_workers = 16  # 24 → 16
                else:
                    face_workers = 6   # 14 → 6
                    yolo_workers = 6   # 14 → 6
                    danger_workers = 6  # 10 → 6
                    frame_workers = 12  # 26 → 12
                
                logging.info(f"멀티 GPU 감지 ({gpu_count}개) - 워커 수: Face={face_workers}, YOLO={yolo_workers}, Danger={danger_workers}, Frame={frame_workers}")
                return face_workers, yolo_workers, danger_workers, frame_workers
            else:
                # 단일 GPU
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if gpu_memory_gb >= 16:
                    face_workers = 8
                    yolo_workers = 8
                    danger_workers = 6
                    frame_workers = 12
                elif gpu_memory_gb >= 10:
                    face_workers = 6
                    yolo_workers = 6
                    danger_workers = 4
                    frame_workers = 10
                else:
                    face_workers = 4
                    yolo_workers = 4
                    danger_workers = 3
                    frame_workers = 8
                
                logging.info(f"단일 GPU 감지 ({gpu_memory_gb:.1f}GB) - 워커 수: Face={face_workers}, YOLO={yolo_workers}, Danger={danger_workers}, Frame={frame_workers}")
                return face_workers, yolo_workers, danger_workers, frame_workers
        except Exception as e:
            logging.warning(f"GPU 정보 가져오기 실패, 기본값 사용: {e}")
            return 4, 4, 3, 8
    elif hasattr(torch, "backends") and torch.backends.mps.is_available():
        # MPS (Apple Silicon) - M4 Pro 최적화
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                   capture_output=True, text=True, timeout=1)
            cpu_info = result.stdout.strip() if result.returncode == 0 else ""
            is_m4_pro = 'M4' in cpu_info and 'Pro' in cpu_info
            
            if is_m4_pro:
                # M4 Pro: 20코어 GPU, 14코어 CPU
                # 성능 최대화: 워커 수 극대화로 GPU 병렬 처리 극대화 (인원 증가 대응)
                face_workers = 18  # 얼굴 인식 워커 (16 -> 18, 인원 증가 대응)
                yolo_workers = 22  # YOLO 모델 워커 (20 -> 22, GPU 코어 수 활용)
                danger_workers = 14  # 위험 행동 감지 워커 (12 -> 14)
                frame_workers = 24  # 프레임 처리 워커 (20 -> 24, CPU 코어 + GPU 병렬 처리)
                logging.info(f"M4 Pro 감지 (20코어 GPU, 14코어 CPU) - 성능 최대화: Face={face_workers}, YOLO={yolo_workers}, Danger={danger_workers}, Frame={frame_workers}")
            else:
                # 다른 Apple Silicon (M1/M2/M3 등) - 인원 증가 대응
                face_workers = 8  # 6 -> 8 (인원 증가 대응)
                yolo_workers = 8  # 6 -> 8
                danger_workers = 6  # 4 -> 6
                frame_workers = 12  # 8 -> 12
                logging.info(f"Apple Silicon 감지 - 워커 수: Face={face_workers}, YOLO={yolo_workers}, Danger={danger_workers}, Frame={frame_workers}")
            return face_workers, yolo_workers, danger_workers, frame_workers
        except Exception as e:
            logging.warning(f"Mac 모델 정보 가져오기 실패, 기본값 사용: {e}")
            # 기본값 (M4 Pro 가정)
            return 8, 10, 6, 12
        return 8, 3, 2, 4
    else:
        # CPU
        return 2, 2, 2, 3

# 환경 변수 또는 동적 계산으로 워커 수 결정
_DEFAULT_FACE_WORKERS, _DEFAULT_YOLO_WORKERS, _DEFAULT_DANGER_WORKERS, _DEFAULT_FRAME_WORKERS = _calculate_optimal_workers()

# 적응형 워커 관리자 초기화
try:
    from adaptive_worker_manager import initialize_adaptive_worker_manager
    adaptive_worker_manager = initialize_adaptive_worker_manager(
        initial_face_workers=_DEFAULT_FACE_WORKERS,
        initial_yolo_workers=_DEFAULT_YOLO_WORKERS,
        initial_danger_workers=_DEFAULT_DANGER_WORKERS,
        initial_frame_workers=_DEFAULT_FRAME_WORKERS
    )
    logging.info("✅ 적응형 워커 관리자 초기화 완료 (GPU 사용률 및 지연 시간 기반 자동 조정)")
except Exception as e:
    logging.warning(f"⚠️ 적응형 워커 관리자 초기화 실패, 기본값 사용: {e}")
    adaptive_worker_manager = None

# MongoDB 배치 저장 시스템 (DB 부하 감소)
violation_batch_queue = queue.Queue()  # 위반 사항 배치 큐
violation_batch_lock = threading.Lock()  # 배치 큐 접근용 락
VIOLATION_BATCH_SIZE = int(os.getenv('VIOLATION_BATCH_SIZE', '20'))  # 배치 크기 (기본 20개, 증가)
VIOLATION_BATCH_INTERVAL = float(os.getenv('VIOLATION_BATCH_INTERVAL', '10.0'))  # 배치 간격 (초, 기본 10초, 5초에서 증가)
VIOLATION_MIN_INTERVAL = float(os.getenv('VIOLATION_MIN_INTERVAL', '30.0'))  # 같은 위반 최소 저장 간격 (초, 기본 30초, 10초에서 증가)
violation_last_saved: Dict[str, float] = {}  # 마지막 저장 시간 추적 (key: f"{worker_id}_{violation_type}_{cam_id}")
image_last_saved: Dict[str, float] = {}  # 이미지 저장 시간 추적 (key: f"{worker_id}_{violation_type}_{cam_id}")
IMAGE_SAVE_MIN_INTERVAL = 1.0  # 이미지 저장 최소 간격 (초, 1초)

# ThreadPoolExecutor 생성 (동적 조정 가능하도록 함수로 래핑)
def _create_executors():
    """워커 수에 따라 ThreadPoolExecutor 생성/업데이트"""
    if adaptive_worker_manager:
        face_w, yolo_w, danger_w, frame_w = adaptive_worker_manager.get_current_workers()
    else:
        face_w = int(os.getenv("FACE_RECOGNITION_WORKERS", str(_DEFAULT_FACE_WORKERS)))
        yolo_w = int(os.getenv("YOLO_WORKERS", str(_DEFAULT_YOLO_WORKERS)))
        danger_w = int(os.getenv("DANGEROUS_BEHAVIOR_WORKERS", str(_DEFAULT_DANGER_WORKERS)))
        frame_w = int(os.getenv("FRAME_PROCESSING_WORKERS", str(_DEFAULT_FRAME_WORKERS)))
    
    return (
        ThreadPoolExecutor(max_workers=face_w, thread_name_prefix="face_recognition"),
        ThreadPoolExecutor(max_workers=yolo_w, thread_name_prefix="yolo_inference"),
        ThreadPoolExecutor(max_workers=danger_w, thread_name_prefix="dangerous_behavior"),
        ThreadPoolExecutor(max_workers=frame_w, thread_name_prefix="frame_processing")
    )

# 초기 Executor 생성
face_recognition_executor, yolo_executor, dangerous_behavior_executor, frame_processing_executor = _create_executors()

# Executor 업데이트용 락
_executor_update_lock = threading.Lock()

def update_worker_executors():
    """적응형 워커 관리자에 따라 Executor 업데이트 (안전한 교체)"""
    global face_recognition_executor, yolo_executor, dangerous_behavior_executor, frame_processing_executor
    
    # 락을 사용하여 동시 업데이트 방지
    with _executor_update_lock:
        if adaptive_worker_manager:
            face_w, yolo_w, danger_w, frame_w = adaptive_worker_manager.adjust_workers()
            
            # 워커 수가 변경되었으면 새로운 Executor 생성
            if (face_recognition_executor._max_workers != face_w or
                yolo_executor._max_workers != yolo_w or
                dangerous_behavior_executor._max_workers != danger_w or
                frame_processing_executor._max_workers != frame_w):
                
                # 새로운 Executor 먼저 생성 (기존 Executor 종료 전에 생성)
                new_face_executor, new_yolo_executor, new_danger_executor, new_frame_executor = _create_executors()
                
                # 기존 Executor를 임시 변수에 저장 (참조 유지)
                old_face_executor = face_recognition_executor
                old_yolo_executor = yolo_executor
                old_danger_executor = dangerous_behavior_executor
                old_frame_executor = frame_processing_executor
                
                # 새로운 Executor로 즉시 교체 (새 작업은 새 Executor로)
                face_recognition_executor = new_face_executor
                yolo_executor = new_yolo_executor
                dangerous_behavior_executor = new_danger_executor
                frame_processing_executor = new_frame_executor
                
                logging.info(f"🔄 Executor 업데이트 완료: Face={face_w}, YOLO={yolo_w}, Danger={danger_w}, Frame={frame_w}")
                
                # 기존 Executor는 백그라운드에서 안전하게 종료 (기존 작업 완료 대기)
                def shutdown_old_executor(old_exec, name):
                    try:
                        old_exec.shutdown(wait=True, timeout=10.0)
                    except Exception as e:
                        logging.warning(f"⚠️ {name} Executor 종료 중 오류 (무시): {e}")
                
                # 백그라운드 스레드에서 기존 Executor 종료 (블로킹 방지)
                import threading
                threading.Thread(target=shutdown_old_executor, args=(old_face_executor, "Face"), daemon=True).start()
                threading.Thread(target=shutdown_old_executor, args=(old_yolo_executor, "YOLO"), daemon=True).start()
                threading.Thread(target=shutdown_old_executor, args=(old_danger_executor, "Danger"), daemon=True).start()
                threading.Thread(target=shutdown_old_executor, args=(old_frame_executor, "Frame"), daemon=True).start()

