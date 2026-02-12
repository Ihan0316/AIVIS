# state.py - 전역 상태 관리 (최적화됨)
"""
전역 변수 및 상태 관리 모듈
관련 변수를 그룹으로 정리하여 관리합니다.
"""
import threading
import time
import queue
from typing import Dict, Set, Optional, Any, List, Tuple
from collections import defaultdict, deque
import os
import logging
import asyncio
from aiohttp import web
from cache_manager import IdentityCache, TTLCache
from concurrent.futures import ThreadPoolExecutor
import config
import torch

# ========================================
# 1. 프레임 상태 관리
# ========================================
latest_frames: Dict[int, bytes] = {}  # 처리된 프레임 저장
latest_result_data: Dict[int, dict] = {}  # 최신 결과 데이터 저장 (대시보드용)
frame_lock = threading.Lock()

# 프레임 처리 동시성 제어
processing_lock = threading.Lock()
processing_flags: Dict[int, bool] = {}

# 프레임 큐 시스템
frame_queues: Dict[int, queue.Queue] = {}
MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', '20'))
queue_lock = threading.Lock()

# 프레임 간격 제어
last_frame_processed_time: Dict[int, float] = {}
MIN_FRAME_INTERVAL = 1.0 / 30.0  # 30 FPS
frame_interval_lock = threading.Lock()

# 프레임 버퍼 (최근 1초)
frame_buffer: Dict[int, List[Tuple[float, bytes, Any]]] = defaultdict(list)
MAX_BUFFER_SECONDS = 1.0
frame_buffer_lock = threading.Lock()

# 프레임 추적 통계
frame_stats: Dict[int, dict] = {}
frame_stats_lock = threading.Lock()

# ========================================
# 2. 통합 결과 캐시 (UnifiedResultCache)
# ========================================
# 모든 모델 결과를 단일 캐시에서 관리
model_results_cache: Dict[int, List[Tuple[float, Dict]]] = defaultdict(list)
results_cache_lock = threading.Lock()
MAX_CACHE_ITEMS = 10  # 캐시당 최대 항목 수 (20 -> 10, 빠른 갱신)
CACHE_TTL = 0.5  # 캐시 TTL (1.5 -> 0.5초, 실시간 응답)

def add_to_cache(cam_id: int, result: Dict) -> None:
    """결과를 캐시에 추가 (자동 정리)"""
    current_time = time.time()
    with results_cache_lock:
        if cam_id not in model_results_cache:
            model_results_cache[cam_id] = []
        
        model_results_cache[cam_id].append((current_time, result))
        
        # 오래된 항목 및 초과 항목 제거
        model_results_cache[cam_id] = [
            (ts, r) for ts, r in model_results_cache[cam_id]
            if current_time - ts <= CACHE_TTL
        ][-MAX_CACHE_ITEMS:]

# 마지막 유의미한 결과 캐시 (깜빡거림 방지)
_last_valid_results: Dict[int, Tuple[float, Dict]] = {}
_last_valid_results_lock = threading.Lock()
LAST_VALID_RESULT_TTL = 3.0  # 마지막 유의미한 결과 유지 시간 (10초 -> 3초, 빠른 갱신)

# ========================================
# 얼굴 위치 기반 이름 투표 캐시 (깜빡거림 방지)
# ========================================
# 키: (cam_id, position_key), 값: {'name': str, 'score': float, 'votes': deque, 'ts': float}
face_position_voting: Dict[Tuple[int, str], Dict] = {}
face_position_voting_lock = threading.Lock()
VOTING_WINDOW_SIZE = 7  # 최근 7프레임 투표
VOTING_MIN_VOTES = 4    # 최소 4표 이상이어야 이름 변경
VOTING_TTL = 2.0        # 2초 후 만료

def get_position_key(x1: int, y1: int, x2: int, y2: int, grid_size: int = 80) -> str:
    """바운딩 박스 위치를 그리드 키로 변환 (근사 위치)"""
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    gx, gy = cx // grid_size, cy // grid_size
    return f"{gx}_{gy}"

def vote_for_name(cam_id: int, x1: int, y1: int, x2: int, y2: int, 
                  new_name: str, new_score: float) -> Tuple[str, float]:
    """위치 기반 투표로 안정적인 이름 반환"""
    position_key = get_position_key(x1, y1, x2, y2)
    cache_key = (cam_id, position_key)
    current_time = time.time()
    
    with face_position_voting_lock:
        # 만료된 항목 정리
        expired_keys = [k for k, v in face_position_voting.items() 
                       if current_time - v.get('ts', 0) > VOTING_TTL]
        for k in expired_keys:
            del face_position_voting[k]
        
        # 기존 캐시 확인
        if cache_key in face_position_voting:
            entry = face_position_voting[cache_key]
            votes = entry['votes']
            
            # 새 투표 추가
            votes.append(new_name)
            if len(votes) > VOTING_WINDOW_SIZE:
                votes.popleft()
            
            # 투표 집계
            from collections import Counter
            vote_counts = Counter(votes)
            most_common_name, most_common_count = vote_counts.most_common(1)[0]
            
            # 현재 이름과 다른 경우
            if most_common_name != entry['name']:
                # 충분한 투표가 있으면 이름 변경
                if most_common_count >= VOTING_MIN_VOTES:
                    entry['name'] = most_common_name
                    entry['score'] = new_score
                    entry['ts'] = current_time
                    return most_common_name, new_score
                else:
                    # 투표 부족 - 기존 이름 유지
                    entry['ts'] = current_time
                    return entry['name'], entry['score']
            else:
                # 같은 이름 - 점수 업데이트
                entry['score'] = max(entry['score'], new_score)
                entry['ts'] = current_time
                return entry['name'], entry['score']
        else:
            # 새 위치 - 바로 등록
            face_position_voting[cache_key] = {
                'name': new_name,
                'score': new_score,
                'votes': deque([new_name], maxlen=VOTING_WINDOW_SIZE),
                'ts': current_time
            }
            return new_name, new_score

def get_latest_cache(cam_id: int, max_age: float = CACHE_TTL) -> Optional[Dict]:
    """캐시에서 최신 결과 가져오기 (유의미한 결과만 반환, 깜빡거림 방지)"""
    # cam_id 타입 통일 (int)
    try:
        cam_id = int(cam_id)
    except (ValueError, TypeError):
        pass

    current_time = time.time()
    result = None
    
    with results_cache_lock:
        if cam_id in model_results_cache:
            # 유의미한 결과(위반/얼굴 있음)만 반환 (빈 결과는 무시)
            for ts, r in reversed(model_results_cache[cam_id]):
                if current_time - ts <= max_age:
                    if r.get("violations") or r.get("recognized_faces"):
                        result = r
                        break
    
    # 유의미한 결과를 찾으면 마지막 유의미한 결과로 저장
    if result:
        with _last_valid_results_lock:
            _last_valid_results[cam_id] = (current_time, result)
        return result
    
    # 유의미한 결과가 없으면 마지막 유의미한 결과 사용 (깜빡거림 방지)
    with _last_valid_results_lock:
        if cam_id in _last_valid_results:
            last_ts, last_result = _last_valid_results[cam_id]
            if current_time - last_ts <= LAST_VALID_RESULT_TTL:
                return last_result
    
    return None

# ========================================
# 3. WebSocket 연결 관리
# ========================================
connected_websockets: Set[web.WebSocketResponse] = set()
dashboard_websockets: Set[web.WebSocketResponse] = set()

# 대시보드 브로드캐스트 제어
DASHBOARD_BROADCAST_INTERVAL = float(os.getenv("DASHBOARD_BROADCAST_INTERVAL", "0.5"))
last_dashboard_broadcast_ts = 0.0
last_dashboard_payload = ""
dashboard_broadcast_lock: Optional[asyncio.Lock] = None

# 프레임 처리 태스크 추적
processing_tasks: Dict[int, Dict[int, asyncio.Task]] = {}
processing_tasks_lock = asyncio.Lock()

# ========================================
# 4. SafetySystem 및 StorageManager
# ========================================
safety_system_instance: Optional[Any] = None
safety_system_lock = threading.Lock()
storage_manager: Optional[Any] = None

# ========================================
# 5. 얼굴 인식 관련 캐시
# ========================================
last_face_detection_frame: Dict[int, int] = {}
face_detection_lock = threading.Lock()
face_recognition_cooldown_ts: Dict[int, float] = defaultdict(lambda: 0.0)

# 식별 결과 캐시
MAX_IDENTITY_CACHE_PER_CAM = 30
recent_identity_cache = IdentityCache(
    max_items_per_cam=MAX_IDENTITY_CACHE_PER_CAM,
    ttl=config.Thresholds.RECOGNITION_HOLD_SECONDS
)

# 렌더링 캐시
last_render_cache: Dict[int, TTLCache] = defaultdict(lambda: TTLCache(default_ttl=1.5))

# 센트로이드 임베딩 버퍼
embedding_buffers: Dict[int, Dict[str, dict]] = defaultdict(dict)
EMBEDDING_BUFFER_SIZE = 5
EMBEDDING_BUFFER_MIN_SIZE = 2
MAX_EMBEDDING_BUFFERS_PER_CAM = 20

# 센트로이드 결과 캐시
CENTROID_CACHE_TTL = 1.5  # 2.0 -> 1.5초
centroid_cache: Dict[int, TTLCache] = defaultdict(lambda: TTLCache(default_ttl=CENTROID_CACHE_TTL))

# 얼굴 바운딩박스 캐시
FACE_BBOX_CACHE_TTL = 1.5  # 2.0 -> 1.5초
face_bbox_cache: Dict[int, TTLCache] = defaultdict(lambda: TTLCache(default_ttl=FACE_BBOX_CACHE_TTL))

# ========================================
# 6. 넘어짐 감지 관련
# ========================================
fall_start_times: Dict[int, Dict[str, float]] = defaultdict(dict)
FALL_DURATION_THRESHOLD = 1.0  # 1초

# ========================================
# 7. Track ID 상태 관리
# ========================================
track_states: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
track_states_lock = threading.Lock()

new_track_ids: Dict[int, Dict[int, float]] = defaultdict(dict)
NEW_TRACK_THRESHOLD = 2.0

last_face_recognition_by_track: Dict[int, Dict[int, float]] = defaultdict(dict)
FACE_RECOGNITION_INTERVAL_PER_TRACK = 1.0

# ========================================
# 8. 알림 및 위반 관리
# ========================================
recent_alerts_cache: Dict[str, float] = {}
ALERT_COOLDOWN_SECONDS = 30.0
CRITICAL_VIOLATIONS = ["넘어짐", "사고", "FALL", "ACCIDENT"]

# MongoDB 배치 저장
violation_batch_queue = queue.Queue()
violation_batch_lock = threading.Lock()
VIOLATION_BATCH_SIZE = int(os.getenv('VIOLATION_BATCH_SIZE', '20'))
VIOLATION_BATCH_INTERVAL = float(os.getenv('VIOLATION_BATCH_INTERVAL', '1.0'))
VIOLATION_MIN_INTERVAL = float(os.getenv('VIOLATION_MIN_INTERVAL', '1.0'))  # 1초로 변경
violation_last_saved: Dict[str, float] = {}
image_last_saved: Dict[str, float] = {}
IMAGE_SAVE_MIN_INTERVAL = 1.0  # 이미지 저장 최소 간격 1초로 변경

# ⭐ 전역 저장 쿨다운 (1초에 1건만 저장)
GLOBAL_SAVE_INTERVAL = float(os.getenv('GLOBAL_SAVE_INTERVAL', '1.0'))  # 전체 저장 간격 (초)
global_last_saved_time: float = 0.0

# ========================================
# 9. 시스템 모니터링
# ========================================
system_stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "error_count": 0,
    "last_health_check": time.time(),
    "memory_usage": 0,
    "cpu_usage": 0,
    "response_times": [],
    "gpu_stats": {}
}
stats_lock = threading.Lock()
MAX_RESPONSE_TIMES = 100

# ========================================
# 10. 프레임 처리 상태
# ========================================
frame_processing_state: Dict[int, Dict[str, Any]] = defaultdict(dict)
frame_processing_state_lock = threading.Lock()

def get_frame_processing_state(cam_id: int) -> Dict[str, Any]:
    """cam_id별 프레임 처리 상태 가져오기"""
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
            frame_processing_state[cam_id] = {
                'frame_count': 0,
                'used_ppe_boxes': set(),
                'last_cleanup_time': 0.0,
                'perf_log_count': 0,
                'model_warmed_up': False
            }

# ========================================
# 11. ThreadPoolExecutor 관리
# ========================================
def _calculate_optimal_workers():
    """GPU 메모리에 따라 최적의 워커 수를 계산 (안정성 우선)"""
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            
            if gpu_count >= 2:
                total_memory_gb = sum(
                    torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                    for i in range(gpu_count)
                )
                avg_memory_gb = total_memory_gb / gpu_count
                
                if avg_memory_gb >= 10:
                    # RTX 2080 Ti 2대 - 안정성 우선 (워커 수 감소)
                    # Face: 12, YOLO: 10, Danger: 6, Frame: 16
                    return 12, 10, 6, 16
                else:
                    return 6, 6, 4, 10
            else:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if gpu_memory_gb >= 16:
                    return 6, 6, 4, 10
                elif gpu_memory_gb >= 10:
                    return 4, 4, 3, 8
                else:
                    return 3, 3, 2, 6
        except Exception:
            return 3, 3, 2, 6
    else:
        return 2, 2, 2, 3

_DEFAULT_FACE_WORKERS, _DEFAULT_YOLO_WORKERS, _DEFAULT_DANGER_WORKERS, _DEFAULT_FRAME_WORKERS = _calculate_optimal_workers()

# Executor 생성
face_recognition_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("FACE_RECOGNITION_WORKERS", str(_DEFAULT_FACE_WORKERS))),
    thread_name_prefix="face_recognition"
)
yolo_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("YOLO_WORKERS", str(_DEFAULT_YOLO_WORKERS))),
    thread_name_prefix="yolo_inference"
)
dangerous_behavior_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("DANGEROUS_BEHAVIOR_WORKERS", str(_DEFAULT_DANGER_WORKERS))),
    thread_name_prefix="dangerous_behavior"
)
frame_processing_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("FRAME_PROCESSING_WORKERS", str(_DEFAULT_FRAME_WORKERS))),
    thread_name_prefix="frame_processing"
)

# ========================================
# 12. 대시보드 상태 (model_results)
# ========================================
# 대시보드 UI에 표시되는 실시간 데이터
model_results: Dict[str, Any] = {
    "kpi_data": {},
    "alerts": [],
    "heatmap_counts": {"A-1": 0, "A-2": 0, "B-1": 0, "B-2": 0},
    "violations": {},
    "profile": {},
    "detected_workers": {}
}
results_lock = results_cache_lock  # 별칭 (하위 호환성)