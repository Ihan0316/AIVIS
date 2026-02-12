# main.py - AIVIS AI 서버 (얼굴 인식, PPE, 위험행동 탐지)
import asyncio
import logging
import os
import threading
import time
import json
import signal
import queue
import tempfile
import ssl
import subprocess
import sys
from typing import Dict, Set, Tuple, List, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# MongoDB 저장 비활성화 옵션 (성능 문제 시 False로 설정, 기본값: False)
ENABLE_MONGODB_SAVE = os.getenv('ENABLE_MONGODB_SAVE', 'false').lower() == 'true'
if not ENABLE_MONGODB_SAVE:
    logging.info("[MongoDB] 위반 사항 저장이 비활성화되어 있습니다 (ENABLE_MONGODB_SAVE=False). 성능 최적화를 위해 MongoDB 저장을 건너뜁니다.")

import aiohttp_cors
import cv2
import numpy as np
import torch
from aiohttp import web, WSMsgType

import core
import utils
import config
from utils import setup_logging, create_standard_response, find_best_match_faiss
import datetime
from datetime import datetime, timedelta
from pathlib import Path
from storage_manager import LocalStorageManager
from cache_manager import IdentityCache, TTLCache

# 전역 캐시 인스턴스 (DB 부하 감소를 위한 TTL 캐시)
violations_cache = TTLCache(default_ttl=5.0)  # 5초 캐시 (위반 사항 조회)
stats_cache = TTLCache(default_ttl=10.0)  # 10초 캐시 (통계 조회)
workers_cache = TTLCache(default_ttl=30.0)  # 30초 캐시 (작업자 조회)

# Rate Limiting을 위한 간단한 구현
class RateLimiter:
    """간단한 Rate Limiter (IP 기반)"""
    def __init__(self, max_calls: int = 100, period: float = 60.0):
        """
        Args:
            max_calls: 기간 내 최대 요청 수
            period: 기간 (초)
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = defaultdict(list)  # IP별 요청 시간 기록
        self.lock = threading.Lock()
    
    def is_allowed(self, ip: str) -> bool:
        """요청이 허용되는지 확인"""
        current_time = time.time()
        
        with self.lock:
            # 오래된 요청 기록 제거
            self.calls[ip] = [t for t in self.calls[ip] if current_time - t < self.period]
            
            # 최대 요청 수 확인
            if len(self.calls[ip]) >= self.max_calls:
                return False
            
            # 요청 기록 추가
            self.calls[ip].append(current_time)
            return True
    
    def get_remaining(self, ip: str) -> int:
        """남은 요청 수 반환"""
        current_time = time.time()
        
        with self.lock:
            # 오래된 요청 기록 제거
            self.calls[ip] = [t for t in self.calls[ip] if current_time - t < self.period]
            return max(0, self.max_calls - len(self.calls[ip]))

# 전역 Rate Limiter 인스턴스
# 환경 변수로 설정 가능 (기본값: 분당 500회로 증가 - 실시간 대시보드 대응)
# 실시간 API는 Rate Limit에서 제외되므로, 나머지 API에 대해서는 여유있게 설정
RATE_LIMIT_MAX_CALLS = int(os.getenv('RATE_LIMIT_MAX_CALLS', '500'))  # 100 -> 500 (실시간 대시보드 대응)
RATE_LIMIT_PERIOD = float(os.getenv('RATE_LIMIT_PERIOD', '60.0'))
rate_limiter = RateLimiter(max_calls=RATE_LIMIT_MAX_CALLS, period=RATE_LIMIT_PERIOD)
from exceptions import (
    FaceRecognitionError,
    CameraError,
    ValidationError
)
from validators import (
    validate_camera_id,
    validate_frame_bytes
)
# 서버 유틸리티 import
from utils_server import (
    get_gpu_usage_stats,
    log_gpu_optimization_recommendations,
    create_compressed_response,
    filter_model_results
)
# 전역 상태 import
from state import (
    latest_frames, latest_result_data, frame_lock,
    processing_lock, processing_flags,
    frame_queues, MAX_QUEUE_SIZE, queue_lock,
    last_frame_processed_time, MIN_FRAME_INTERVAL, frame_interval_lock,
    connected_websockets, dashboard_websockets,
    safety_system_instance, safety_system_lock, storage_manager,
    processing_tasks, processing_tasks_lock,
    DASHBOARD_BROADCAST_INTERVAL, last_dashboard_broadcast_ts, last_dashboard_payload, dashboard_broadcast_lock,
    frame_stats, frame_stats_lock,
    last_face_detection_frame, face_detection_lock, face_recognition_cooldown_ts,
    recent_identity_cache, MAX_IDENTITY_CACHE_PER_CAM,
    last_render_cache,
    embedding_buffers, EMBEDDING_BUFFER_SIZE, EMBEDDING_BUFFER_MIN_SIZE, MAX_EMBEDDING_BUFFERS_PER_CAM,
    fall_start_times, FALL_DURATION_THRESHOLD,
    centroid_cache, CENTROID_CACHE_TTL,
    face_bbox_cache, FACE_BBOX_CACHE_TTL,
    model_results, results_lock,
    recent_alerts_cache, ALERT_COOLDOWN_SECONDS, CRITICAL_VIOLATIONS,
    system_stats, stats_lock, MAX_RESPONSE_TIMES,
    face_recognition_executor, yolo_executor, dangerous_behavior_executor, frame_processing_executor
)

# Ultralytics 환경변수 기본값 설정 (컨테이너 권한 이슈 대비)
# Windows 호환: tempfile.gettempdir()를 사용하여 플랫폼 독립적인 임시 디렉토리 사용
_UL_BASE = os.path.join(tempfile.gettempdir(), "Ultralytics")
_UL_ENV_MAP = {
    "YOLO_CONFIG_DIR": os.path.join(_UL_BASE, "config"),
    "YOLO_RUNS_DIR": os.path.join(_UL_BASE, "runs"),
    "YOLO_DATASETS_DIR": os.path.join(_UL_BASE, "datasets"),
}

for _env_key, _env_path in _UL_ENV_MAP.items():
    if not os.environ.get(_env_key):
        os.environ[_env_key] = _env_path

# 미리 디렉터리 생성 (Ultralytics가 내부에서 추가 경로를 붙여도 실패하지 않도록)
for _path in {_UL_BASE, *(_UL_ENV_MAP.values())}:
    try:
        os.makedirs(_path, exist_ok=True)
        os.makedirs(os.path.join(_path, "Ultralytics"), exist_ok=True)
    except OSError:
        pass

# torch matmul precision (MPS/CPU 최적화를 위해 조정 가능)
try:
    # MPS 최적화: medium precision으로 설정 (속도와 정확도 균형)
    # high: 더 정확하지만 느림, medium: 균형, highest: 가장 정확하지만 매우 느림
    precision = os.getenv("TORCH_MATMUL_PRECISION", "medium")
    torch.set_float32_matmul_precision(precision)
    if torch.backends.mps.is_available():
        logging.info(f"✅ MPS 최적화: matmul precision={precision}")
except AttributeError:
    pass

# GPU 모니터링 함수는 utils_server.py로 이동됨

# 환경에 따른 기본 워커 수 (실제 GPU에 맞게 동적 설정, 멀티 GPU 지원)
if torch.cuda.is_available():
    try:
        gpu_count = torch.cuda.device_count()
        
        if gpu_count >= 2:
            # 멀티 GPU: 각 GPU의 메모리를 합산하여 계산
            total_memory_gb = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count))
            avg_memory_gb = total_memory_gb / gpu_count
            
            # 멀티 GPU: GPU 여유 활용하여 성능 개선
            # GPU 사용률이 낮으므로 워커 수 대폭 증가 (GPU 활용률 향상)
            if avg_memory_gb >= 10:
                _DEFAULT_FACE_WORKERS = 16  # 12 → 16 (얼굴 감지 + 얼굴 인식 병렬 처리)
                _DEFAULT_YOLO_WORKERS = 12  # 8 → 12 (GPU 활용률 향상)
                _DEFAULT_DANGER_WORKERS = 8  # 6 → 8 (GPU 활용률 향상)
                _DEFAULT_FRAME_WORKERS = 20  # 16 → 20 (GPU 활용률 향상)
            else:
                _DEFAULT_FACE_WORKERS = 14  # 10 → 14 (얼굴 감지 + 얼굴 인식 병렬 처리)
                _DEFAULT_YOLO_WORKERS = 10
                _DEFAULT_DANGER_WORKERS = 8
                _DEFAULT_FRAME_WORKERS = 22
            
            gpu_info = []
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info.append(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            logging.info(f"멀티 GPU 감지 ({gpu_count}개): {', '.join(gpu_info)}")
            logging.info(f"⭐ 멀티캠 2대 최적화 모드 활성화")
            logging.info(f"워커 수: Face={_DEFAULT_FACE_WORKERS} (GPU 1), YOLO={_DEFAULT_YOLO_WORKERS} (GPU 0), Danger={_DEFAULT_DANGER_WORKERS}, Frame={_DEFAULT_FRAME_WORKERS}")
        else:
            # 단일 GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory_gb >= 16:
                _DEFAULT_FACE_WORKERS = 8  # 최고 성능: 증가
                _DEFAULT_YOLO_WORKERS = 8  # 최고 성능: 증가
                _DEFAULT_DANGER_WORKERS = 6  # 최고 성능: 증가
                _DEFAULT_FRAME_WORKERS = 12  # 최고 성능: 증가
            elif gpu_memory_gb >= 10:
                _DEFAULT_FACE_WORKERS = 6  # 최고 성능: 증가
                _DEFAULT_YOLO_WORKERS = 6  # 최고 성능: 증가
                _DEFAULT_DANGER_WORKERS = 4  # 최고 성능: 증가
                _DEFAULT_FRAME_WORKERS = 10  # 최고 성능: 증가
            else:
                _DEFAULT_FACE_WORKERS = 4
                _DEFAULT_YOLO_WORKERS = 4
                _DEFAULT_DANGER_WORKERS = 3
                _DEFAULT_FRAME_WORKERS = 8
            
            logging.info(f"GPU 감지: {gpu_name} ({gpu_memory_gb:.1f}GB) - 워커 수: Face={_DEFAULT_FACE_WORKERS}, YOLO={_DEFAULT_YOLO_WORKERS}, Danger={_DEFAULT_DANGER_WORKERS}, Frame={_DEFAULT_FRAME_WORKERS}")
    except Exception as e:
        logging.warning(f"GPU 정보 가져오기 실패, 기본값 사용: {e}")
        _DEFAULT_FACE_WORKERS = 4
        _DEFAULT_YOLO_WORKERS = 4
        _DEFAULT_DANGER_WORKERS = 3
        _DEFAULT_FRAME_WORKERS = 8

# 프레임 처리 함수 import
from frame_processor import process_single_frame


# --- 대시보드 브로드캐스트 함수 ---
async def broadcast_to_dashboards(result_data: Dict[str, Any], cam_id: int = 0) -> None:
    """대시보드 연결들에게 결과 데이터 브로드캐스트"""
    if not dashboard_websockets:
        return

    message = json.dumps(result_data)
    disconnected = set()

    for ws in dashboard_websockets:
        try:
            await ws.send_str(message)
        except (ConnectionResetError, ConnectionError, OSError):
            disconnected.add(ws)

    # 끊어진 연결 제거
    for ws in disconnected:
        dashboard_websockets.discard(ws)


async def _send_embedding_notification(notification: Dict[str, Any]) -> None:
    """임베딩 완료 알림을 WebSocket으로 전송"""
    if not dashboard_websockets:
        logging.warning(f"[EMBEDDING] WebSocket 연결이 없습니다. 알림을 전송할 수 없습니다. (연결된 대시보드: 0개)")
        return
    
    message = json.dumps(notification)
    disconnected = set()
    sent_count = 0
    
    for ws in dashboard_websockets:
        try:
            await ws.send_str(message)
            sent_count += 1
        except (ConnectionResetError, ConnectionError, OSError) as e:
            logging.debug(f"[EMBEDDING] WebSocket 전송 실패: {e}")
            disconnected.add(ws)
    
    # 끊어진 연결 제거
    for ws in disconnected:
        dashboard_websockets.discard(ws)
    
    if sent_count > 0:
        logging.info(f"[EMBEDDING] ✅ 알림 전송 완료: {sent_count}개 대시보드에 전송됨")
    else:
        logging.warning(f"[EMBEDDING] ⚠️ 알림 전송 실패: 모든 WebSocket 연결이 끊어짐")

# --- 모델 결과 업데이트 및 브로드캐스트 함수 ---
def update_kpi_from_all_cameras() -> None:
    """모든 카메라의 데이터를 합산하여 KPI를 계산합니다."""
    global model_results
    
    with results_lock, frame_lock:
        # 모든 카메라의 데이터 수집
        all_recognized_faces = []
        all_violations = []
        
        for cam_id, result_data in latest_result_data.items():
            recognized_faces = result_data.get("recognized_faces", [])
            violations = result_data.get("violations", [])
            
            
            all_recognized_faces.extend(recognized_faces)
            all_violations.extend(violations)
        
        # 현재 인원 계산 (중복 제거)
        unique_names = set()
        for face in all_recognized_faces:
            name = face.get("name", "Unknown")
            if name != "Unknown":
                unique_names.add(name)
        
        # 위반이 있지만 얼굴이 인식되지 않은 사람도 카운트
        violation_workers = set()
        for v in all_violations:
            worker = v.get("worker", "알 수 없음")
            if worker and worker != "알 수 없음":
                violation_workers.add(worker)
        
        # 실제 작업자 수 계산: 인식된 얼굴 수(Unknown 포함) 또는 위반이 있는 사람 수 중 큰 값
        # recognized_faces에는 이미 중복 제거가 되어 있으므로 길이를 직접 사용
        recognized_count = len(all_recognized_faces)  # Unknown 포함 모든 감지된 사람 수
        
        # 위반이 있지만 얼굴이 인식되지 않은 사람도 카운트
        total_workers = max(recognized_count, len(violation_workers))
        
        # 위반이 있으면 최소 1명 이상
        if len(all_violations) > 0 and total_workers == 0:
            total_workers = len(set(v.get("worker", "알 수 없음") for v in all_violations))
        
        # 최소값 보장: recognized_faces가 있으면 최소 그만큼은 있어야 함
        if recognized_count > 0:
            total_workers = max(total_workers, recognized_count)
        
        # 안전장비 위반 수 계산 (실제 위반 건수)
        # 위반이 있는 사람 수를 카운트 (동일 사람의 여러 위반도 1건으로 카운트)
        safety_violation_count = 0
        helmet_violations = 0
        vest_violations = 0
        violation_workers_set = set()  # 위반이 있는 사람 추적 (중복 제거)
        
        for v in all_violations:
            violations = v.get("violations", [])
            worker = v.get("worker", "알 수 없음")
            
            # PPE 위반이 있는지 확인
            has_ppe_violation = any("안전모" in str(vt) or "안전조끼" in str(vt) for vt in violations)
            
            # 위반 유형별 카운트
            for vt in violations:
                vt_str = str(vt)
                if "안전모" in vt_str or "헬멧" in vt_str:
                    helmet_violations += 1
                if "안전조끼" in vt_str or "조끼" in vt_str or "vest" in vt_str.lower():
                    vest_violations += 1
            
            if has_ppe_violation:
                # 동일 사람의 위반은 1건으로만 카운트
                if worker not in violation_workers_set:
                    safety_violation_count += 1
                    violation_workers_set.add(worker)
        
        # 위험 행동 감지 수 계산 (실제 위반 건수)
        dangerous_behavior_count = 0
        fall_detections = 0
        for v in all_violations:
            violations = v.get("violations", [])
            for vt in violations:
                vt_str = str(vt)
                if any(keyword in vt_str for keyword in ["넘어짐", "사고", "FALL", "ACCIDENT"]):
                    dangerous_behavior_count += 1
                    if "넘어짐" in vt_str or "FALL" in vt_str:
                        fall_detections += 1
                    break  # 한 사람당 1건으로 카운트
        
        # PPE 비율 계산
        ppe_count = sum(1 for face in all_recognized_faces if face.get("has_ppe", False))
        # 위반이 없는 사람도 PPE 착용으로 간주
        no_violation_count = len([v for v in all_violations if not v.get("violations") or len(v.get("violations", [])) == 0])
        ppe_count += no_violation_count
        ppe_rate = (ppe_count / total_workers * 100) if total_workers > 0 else 0
        ppe_rate = round(ppe_rate, 1)
        
        # 위험도 계산
        violation_count = len(all_violations)
        if total_workers == 0:
            risk_level = 0
        else:
            risk_score = violation_count * 2  # 위반당 2점
            max_possible_score = total_workers * 2
            risk_level = min(100, (risk_score / max_possible_score) * 100) if max_possible_score > 0 else 0
            risk_level = round(risk_level, 1)
        
        # 안전 점수 계산 (위반이 적을수록 높은 점수)
        # 위반이 없으면 100점, 위반이 있으면 감점 (위반 1건당 5점 감점, 최소 0점)
        safety_score = 100
        if total_workers > 0:
            violation_penalty = len(all_violations) * 5
            safety_score = max(0, 100 - violation_penalty)
        else:
            safety_score = 100  # 작업자가 없으면 100점
        
        # 정확도 지표 계산 (실제 감지 성공률 기반)
        # 얼굴 인식 정확도: 인식된 얼굴 수 / 전체 감지된 사람 수
        facial_recognition_rate = 0.0
        if recognized_count > 0:
            recognized_faces_count = len([f for f in all_recognized_faces if f.get("name") != "Unknown"])
            facial_recognition_rate = (recognized_faces_count / recognized_count) * 100
        
        # PPE 탐지 정확도: PPE 착용 감지 성공률 (위반이 없는 사람 비율)
        equipment_detection_rate = ppe_rate  # PPE 착용률을 탐지 정확도로 사용
        
        # 행동 탐지 정확도: 위험 행동 감지 성공률 (위험 행동이 감지된 경우)
        behavior_detection_rate = 91.5  # 기본값 (향후 실제 감지 성공률로 계산 가능)
        if dangerous_behavior_count > 0:
            # 위험 행동이 감지되면 정확도가 높다고 가정
            behavior_detection_rate = 95.0
        
        # KPI 데이터 업데이트 (프론트엔드 필드명과 일치)
        model_results["kpi_data"] = {
            "totalWorkers": total_workers,
            "attendees": total_workers,
            "ppeRate": ppe_rate,
            "riskLevel": risk_level,
            "safetyViolations": safety_violation_count,
            "dangerousBehaviors": dangerous_behavior_count,
            "helmet_violations": helmet_violations,
            "vest_violations": vest_violations,
            "fall_detections": fall_detections,
            "total_violations": len(all_violations),
            # 프론트엔드 호환 필드명 추가
            "helmet": helmet_violations,
            "vest": vest_violations,
            "fall": fall_detections,
            "total": len(all_violations),  # 전체 위반 건수
            # 대시보드 추가 지표
            "safetyScore": round(safety_score, 1),
            "facialRecognitionAccuracy": round(facial_recognition_rate, 1),
            "equipmentDetectionAccuracy": round(equipment_detection_rate, 1),
            "behaviorDetectionAccuracy": round(behavior_detection_rate, 1)
        }
        
        # 디버깅: KPI 계산 결과 로깅 (주기적으로 INFO 레벨로 출력)
        if not hasattr(update_kpi_from_all_cameras, '_log_count'):
            update_kpi_from_all_cameras._log_count = 0
        update_kpi_from_all_cameras._log_count += 1
        
        if update_kpi_from_all_cameras._log_count % 10 == 0:  # 10번에 1번만 INFO 로깅
            logging.info(f"[KPI] 업데이트 - 인원: {total_workers}, 안전모위반: {helmet_violations}, 조끼위반: {vest_violations}, 낙상: {fall_detections}, 전체위반: {len(all_violations)}, 얼굴인식: {len(all_recognized_faces)}개")

async def save_violation_to_mongodb(worker: str, violations: List[str], area: str, cam_id: int) -> None:
    """MongoDB에 위반 사항을 배치 큐에 추가 (배치 처리로 DB 부하 감소)"""
    # MongoDB 저장이 비활성화되어 있으면 즉시 반환
    if not ENABLE_MONGODB_SAVE:
        return
    
    try:
        from database import get_database
        from state import violation_batch_queue, violation_batch_lock, VIOLATION_MIN_INTERVAL, violation_last_saved
        import time
        
        db = get_database()
        if not db or not db.is_connected():
            return
        
        db_service = db.db_service if hasattr(db, 'db_service') else None
        if not db_service:
            return
        
        current_time = time.time()
        
        # 위반 사항을 배치 큐에 추가
        for violation_type in violations:
            if not violation_type:
                continue
            
            # 중복 저장 방지: 같은 위반에 대해 최소 간격 내 저장 방지
            cache_key = f"{worker}_{violation_type}_{cam_id}"
            with violation_batch_lock:
                last_saved_time = violation_last_saved.get(cache_key, 0)
                if current_time - last_saved_time < VIOLATION_MIN_INTERVAL:
                    continue
                violation_last_saved[cache_key] = current_time
            
            # 배치 큐에 추가
            batch_item = {
                'violations': [{
                    'worker': worker,
                    'violations': [violation_type],
                    'work_zone': area
                }],
                'cam_id': cam_id,
                'recognized_faces': [],
                'db_service': db_service,
                'timestamp': current_time
            }
            violation_batch_queue.put(batch_item)
    
    except Exception as e:
        logging.error(f"[MongoDB] 위반 사항 배치 큐 추가 실패: worker={worker}, violations={violations}, area={area}, cam_id={cam_id}, error={e}", exc_info=True)

def update_model_results_from_frame(result_data: Dict[str, Any], cam_id: int = 0) -> None:
    """프레임 처리 결과를 기반으로 model_results를 업데이트합니다."""
    global model_results, recent_alerts_cache
    
    with results_lock:
        # 위반 감지 결과 처리
        violations_list = result_data.get("violations", [])
        recognized_faces = result_data.get("recognized_faces", [])
        
        # 위반이 없고 인식된 얼굴도 없으면 처리하지 않음 (검은 프레임 등)
        # 하지만 인식된 얼굴이 있으면 KPI 업데이트를 위해 처리 (현재 인원 표시)
        if not violations_list and not recognized_faces:
            return
        
        # KPI 업데이트: 인식된 얼굴이 있으면 현재 인원 업데이트
        # 위반이 없어도 작업자가 있으면 KPI에 반영
        
        current_time = time.time()
        
        # 오래된 알림 캐시 정리 (1분 이상 지난 항목 제거)
        recent_alerts_cache = {k: v for k, v in recent_alerts_cache.items() 
                              if current_time - v < 60.0}
        
        # 캐시 크기 제한 (안전성 향상: 최대 1000개 항목만 유지)
        MAX_ALERTS_CACHE_SIZE = 1000
        if len(recent_alerts_cache) > MAX_ALERTS_CACHE_SIZE:
            # 최신 항목만 유지 (timestamp 기준 정렬)
            sorted_items = sorted(recent_alerts_cache.items(), key=lambda x: x[1], reverse=True)
            recent_alerts_cache = dict(sorted_items[:MAX_ALERTS_CACHE_SIZE])
        
        # 위반 처리 및 MongoDB 저장
        for violation in violations_list:
            # area 추론: violation에 area가 없으면 cam_id 기반으로 추론
            area = violation.get("area")
            if not area or area == "":
                # cam_id를 area로 매핑 (0→A-1, 1→A-2, 2→B-1, 3→B-2)
                area_map = {0: "A-1", 1: "A-2", 2: "B-1", 3: "B-2"}
                area = area_map.get(cam_id, f"A-{cam_id+1}")
            
            level = violation.get("level", "WARNING")
            worker = violation.get("worker", "알 수 없음")
            hazard = violation.get("hazard", "위반 감지")
            violations = violation.get("violations", [])  # 위반 유형 리스트
            
            # 심각한 위반(넘어짐, 사고) 체크: 즉시 알림 (쿨다운 없음)
            is_critical = any(crit in str(violations) or crit in hazard for crit in CRITICAL_VIOLATIONS)
            
            # 중복 알림 방지: 위반 유형을 정렬하여 정규화 (같은 위반 조합은 같은 키로 인식)
            # 예: ["안전모", "안전조끼"]와 ["안전조끼", "안전모"]는 같은 키로 인식
            violation_types_sorted = sorted([str(v) for v in violations])
            violation_key = "|".join(violation_types_sorted) if violation_types_sorted else "unknown"
            alert_key = f"{worker}|{area}|{violation_key}"
            last_alert_time = recent_alerts_cache.get(alert_key, 0)
            
            if not is_critical and (current_time - last_alert_time < ALERT_COOLDOWN_SECONDS):
                # PPE 위반이고 쿨다운 시간 내이므로 알림 생성하지 않음
                continue
            
            # 새로운 알림 생성
            alert = {
                "level": level,
                "area": area,
                "worker": worker,
                "hazard": hazard,
                "timestamp": current_time
            }
            model_results["alerts"].append(alert)
            
            # 알림 캐시 업데이트
            recent_alerts_cache[alert_key] = current_time
            
            # MongoDB에 위반 사항 저장 (비동기 태스크로 실행, 블로킹 완전 방지)
            if violations:
                asyncio.create_task(save_violation_to_mongodb(
                    worker=worker,
                    violations=violations,
                    area=area,
                    cam_id=cam_id
                ))
        
        # 알림 히스토리 크기 제한 (20 → 100으로 증가: 더 긴 히스토리 제공)
        MAX_ALERTS_HISTORY = 100
        if len(model_results["alerts"]) > MAX_ALERTS_HISTORY:
            model_results["alerts"] = model_results["alerts"][-MAX_ALERTS_HISTORY:]
        
        # 히트맵 카운트 및 위반 카운트 업데이트 (각 위반별로)
        last_violation = None
        for violation in violations_list:
            area = violation.get("area", "A-1")
            
            # 히트맵 카운트 증가
            if area in model_results["heatmap_counts"]:
                model_results["heatmap_counts"][area] += 1
            
            # 위반 카운트 업데이트
            if area not in model_results["violations"]:
                model_results["violations"][area] = 0
            model_results["violations"][area] += 1
            
            # 마지막 위반 저장 (프로필 업데이트용)
            last_violation = violation
        
        # 프로필 업데이트 (가장 최근 위반만)
        if last_violation:
            model_results["profile"] = {
                "name": last_violation.get("worker", "알 수 없음"),
                "status": last_violation.get("hazard", "위반 감지"),
                "area": last_violation.get("area", "A-1")
            }
        
        # 작업자 수 계산 (인식된 얼굴 + 위반이 있는 사람 모두 포함)
        # 위반이 있지만 얼굴이 인식되지 않은 경우도 카운트
        total_workers = len(recognized_faces)
        # 위반이 있지만 얼굴이 인식되지 않은 사람도 카운트
        violation_workers = set()
        for v in violations_list:
            worker = v.get("worker", "알 수 없음")
            if worker and worker != "알 수 없음":
                violation_workers.add(worker)
        # 실제 작업자 수 = 인식된 얼굴 수 + 위반이 있지만 얼굴 미인식 수
        total_workers = max(total_workers, len(violation_workers) + len([v for v in violations_list if not v.get("worker") or v.get("worker") == "알 수 없음"]))
        # 최소 1명 이상 (위반이 있으면 사람이 있다는 의미)
        if len(violations_list) > 0 and total_workers == 0:
            total_workers = len(violations_list)
        attendees = total_workers
        
        # PPE 비율 계산
        ppe_count = sum(1 for face in recognized_faces if face.get("has_ppe", False))
        # 위반이 없는 사람도 PPE 착용으로 간주
        no_violation_count = len([v for v in violations_list if not v.get("violations") or len(v.get("violations", [])) == 0])
        ppe_count += no_violation_count
        ppe_rate = (ppe_count / total_workers * 100) if total_workers > 0 else 0
        ppe_rate = round(ppe_rate, 1)
        
        # 안전장비 위반 수 계산
        safety_violation_count = 0
        for v in violations_list:
            violations = v.get("violations", [])
            if any("안전모" in str(vt) or "안전조끼" in str(vt) for vt in violations):
                safety_violation_count += 1
        
        # 위험 행동 감지 수 계산
        dangerous_behavior_count = 0
        for v in violations_list:
            violations = v.get("violations", [])
            if any("넘어짐" in str(vt) or "사고" in str(vt) or "FALL" in str(vt) or "ACCIDENT" in str(vt) for vt in violations):
                dangerous_behavior_count += 1
        
        # 위험도 계산
        violation_count = len(violations_list)
        if total_workers == 0:
            risk_level = 0
        else:
            risk_score = violation_count * 2  # 위반당 2점
            max_possible_score = total_workers * 2
            risk_level = min(100, (risk_score / max_possible_score) * 100) if max_possible_score > 0 else 0
            risk_level = round(risk_level, 1)
        
        # KPI 데이터는 모든 카메라를 합산하여 계산하므로 여기서는 업데이트하지 않음
        # update_kpi_from_all_cameras() 함수에서 통합 계산
        
        # 감지된 작업자 정보 업데이트
        model_results["detected_workers"] = {}
        for face in recognized_faces:
            name = face.get("name", "Unknown")
            area = face.get("area", "A-1")
            if area not in model_results["detected_workers"]:
                model_results["detected_workers"][area] = []
            model_results["detected_workers"][area].append({
                "name": name,
                "has_ppe": face.get("has_ppe", False)
            })

async def broadcast_model_results(force: bool = False) -> None:
    """연결된 모든 웹소켓 클라이언트에게 모델 결과를 전송합니다."""
    # 대시보드 연결에만 전송 (클라이언트 연결은 프레임 전송으로 충분)
    if not dashboard_websockets:
        return
    
    global dashboard_broadcast_lock, last_dashboard_broadcast_ts, last_dashboard_payload
    
    if dashboard_broadcast_lock is None:
        dashboard_broadcast_lock = asyncio.Lock()
    
    async with dashboard_broadcast_lock:
        try:
            now_ts = time.time()
            if not force and (now_ts - last_dashboard_broadcast_ts) < DASHBOARD_BROADCAST_INTERVAL:
                return
            
            with results_lock:
                # KPI 데이터 확인 및 로깅 (주기적으로)
                kpi_data = model_results.get("kpi_data", {})
                if not hasattr(broadcast_model_results, '_log_count'):
                    broadcast_model_results._log_count = 0
                broadcast_model_results._log_count += 1
                
                if broadcast_model_results._log_count % 20 == 0:  # 20번에 1번만 로깅
                    logging.info(f"[대시보드 브로드캐스트] KPI 데이터: 인원={kpi_data.get('totalWorkers', 0)}, 안전모={kpi_data.get('helmet', 0)}, 조끼={kpi_data.get('vest', 0)}, 낙상={kpi_data.get('fall', 0)}, 전체위반={kpi_data.get('total', 0)}, 연결된 대시보드={len(dashboard_websockets)}개")
                
                # 디버깅: 연결된 대시보드가 있으면 항상 로깅 (처음 몇 번만)
                if len(dashboard_websockets) > 0 and broadcast_model_results._log_count <= 5:
                    logging.info(f"[대시보드 브로드캐스트] 디버그 - 연결된 대시보드: {len(dashboard_websockets)}개, KPI 데이터 전송 중...")
                
                results_json = json.dumps({
                    "type": "model_results",
                    "data": model_results
                })
            
            # 동일한 페이로드 반복 전송을 피하여 UI 깜빡임을 완화
            if not force and results_json == last_dashboard_payload:
                last_dashboard_broadcast_ts = now_ts
                return
            
            last_dashboard_payload = results_json
            last_dashboard_broadcast_ts = now_ts
        except Exception as e:
            logging.error(f"broadcast_model_results 준비 중 오류: {e}")
            return
    
    try:
        # 동시 전송을 위해 gather 사용 (타임아웃 추가)
        disconnected = set()
        tasks = []
        for ws in dashboard_websockets.copy():  # copy로 반복 중 수정 방지
            tasks.append(_send_to_websocket(ws, results_json, disconnected))
        
        # 최대 2초 타임아웃으로 전송
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=2.0
        )
        
        # 끊어진 연결 제거
        for ws in disconnected:
            dashboard_websockets.discard(ws)
    except asyncio.TimeoutError:
        logging.warning("broadcast_model_results 타임아웃 (2초 초과)")
    except Exception as e:
        logging.error(f"broadcast_model_results 오류: {e}")

async def _send_to_websocket(
    ws: web.WebSocketResponse, 
    message: str, 
    disconnected_set: Set[web.WebSocketResponse]
) -> None:
    """개별 WebSocket에 메시지 전송 (헬퍼 함수)"""
    try:
        await ws.send_str(message)
    except (ConnectionResetError, ConnectionError, OSError):
        disconnected_set.add(ws)
    except Exception as e:
        disconnected_set.add(ws)

async def broadcast_logs(logs: List[Dict[str, Any]]) -> None:
    """연결된 모든 웹소켓 클라이언트에게 로그 메시지를 전송합니다."""
    if not connected_websockets:
        return
    
    full_log_message = "".join(logs)
    # 동시 전송을 위해 gather 사용
    disconnected = set()
    for ws in connected_websockets:
        try:
            await ws.send_str(full_log_message)
        except (ConnectionResetError, ConnectionError, OSError):
            disconnected.add(ws)
    
    # 끊어진 연결 제거
    for ws in disconnected:
        connected_websockets.discard(ws)


# --- 대시보드용 웹소켓 핸들러 ---
async def dashboard_websocket_handler(request: web.Request):
    """대시보드 전용 WebSocket 핸들러 (데이터만 받기)"""
    ws = web.WebSocketResponse()
    client_id = None
    
    try:
        await ws.prepare(request)
        client_id = id(ws)
        dashboard_websockets.add(ws)
        logging.info(f"대시보드 웹소켓 클라이언트 {client_id} 연결됨. (현재 {len(dashboard_websockets)}명 접속 중)")

        # 연결 확인 메시지 먼저 전송
        try:
            await ws.send_str(json.dumps({
                "type": "connected",
                "message": "대시보드 WebSocket 연결 성공",
                "timestamp": int(time.time() * 1000)
            }))
        except Exception as e:
            logging.warning(f"대시보드 클라이언트 {client_id} 연결 확인 메시지 전송 실패: {e}")

        # 연결 시 최신 데이터 전송 (대시보드용 model_results 포함)
        try:
            # 1. 최신 프레임 데이터 전송
            with frame_lock:
                latest_data = latest_result_data.get(0, {})

            if latest_data:
                try:
                    await ws.send_str(json.dumps(latest_data))
                except Exception:
                    pass

            # 2. 대시보드용 model_results 전송 (초기 상태 포함)
            with results_lock:
                initial_results = {
                    "type": "model_results",
                    "data": model_results
                }

            await ws.send_str(json.dumps(initial_results))
            logging.info(f"대시보드 클라이언트 {client_id}에게 초기 데이터 전송 완료")
        except (ConnectionResetError, ConnectionError, OSError) as e:
            logging.warning(f"대시보드 클라이언트 {client_id} 초기 데이터 전송 실패: {e}")
            return ws
        except Exception as e:
            logging.error(f"대시보드 클라이언트 {client_id} 초기 데이터 전송 오류: {e}", exc_info=True)
            # 오류가 발생해도 연결은 유지
    except Exception as e:
        logging.error(f"대시보드 WebSocket 연결 준비 오류: {e}", exc_info=True)
        if client_id:
            dashboard_websockets.discard(ws)
        return ws

    try:
        # 연결 유지를 위한 무한 루프
        while True:
            try:
                # 메시지 수신 대기 (타임아웃 없음, 연결 유지)
                msg = await asyncio.wait_for(ws.receive(), timeout=30.0)

                if msg.type == WSMsgType.TEXT:
                    # ping/pong 메시지 처리
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "ping":
                            await ws.send_str(json.dumps({"type": "pong"}))
                    except:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    logging.warning(f"대시보드 클라이언트 {client_id} 오류 발생: {ws.exception()}")
                    break
                elif msg.type == WSMsgType.CLOSE:
                    break

            except asyncio.TimeoutError:
                # 타임아웃 시 연결 확인 메시지 전송
                try:
                    await ws.send_str(json.dumps({"type": "heartbeat"}))
                except (ConnectionResetError, ConnectionError, OSError):
                    break
            except Exception as e:
                logging.warning(f"대시보드 클라이언트 {client_id} 메시지 수신 오류: {e}")
                break

    except (ConnectionResetError, ConnectionError, OSError) as e:
        logging.info(f"대시보드 클라이언트 {client_id} 연결이 끊어졌습니다. ({e})")
    finally:
        dashboard_websockets.discard(ws)
        logging.info(f"대시보드 웹소켓 클라이언트 {client_id} 연결 종료. (현재 {len(dashboard_websockets)}명 접속 중)")



# 프레임 큐 처리 함수 (비동기)
async def process_frame_queue(cam_id: int, ws: web.WebSocketResponse, client_id: int):
    """프레임 큐에서 프레임을 처리하는 비동기 함수 (최신 프레임 우선)"""
    try:
        while True:
            # 태스크가 취소되면 즉시 종료
            try:
                await asyncio.sleep(0)  # 취소 포인트
            except asyncio.CancelledError:
                raise
            
            # 연결 상태 확인 (연결이 끊겼으면 즉시 종료)
            if ws.closed:
                break
            
            # 프레임 간격 제어 (튐 방지)
            current_time_check = time.time()
            with frame_interval_lock:
                last_time = last_frame_processed_time.get(cam_id, 0)
                time_since_last = current_time_check - last_time
                
                # 최소 프레임 간격보다 짧으면 대기
                if time_since_last < MIN_FRAME_INTERVAL:
                    await asyncio.sleep(MIN_FRAME_INTERVAL - time_since_last)
            
            # 큐에서 프레임 가져오기
            frame_bytes = None
            frame_timestamp = None
            
            iteration_start = time.time()

            remaining_queue = 0

            with queue_lock:
                if cam_id not in frame_queues or frame_queues[cam_id].empty():
                    # 큐가 비어있으면 처리 종료
                    with processing_lock:
                        processing_flags[cam_id] = False
                    logging.info(f"[CAM-{cam_id}] 프레임 큐가 비어 처리 종료 (클라이언트 {client_id})")
                    break
                
                # 프레임 순차 처리 (튐 방지)
                # 큐에서 가장 오래된 프레임부터 처리 (FIFO)
                # 너무 오래된 프레임은 스킵 (처리 속도가 느릴 때를 고려하여 완화)
                temp_frames = []
                current_time = time.time()
                # 프레임 유지율 최대화: 오래된 프레임 스킵 기준 완화 (MPS 환경 최적화: 0.3 -> 0.5초)
                MAX_FRAME_AGE = 0.5  # 0.5초 이상 오래된 프레임만 스킵 (MPS 환경 프레임 유지율 향상)
                # 프레임 유지율 최대화: 더 많은 프레임 확인 (MPS 환경 최적화: 10 -> 15개)
                MAX_QUEUE_CHECK = 15  # 최대 15개 확인 (MPS 환경 프레임 유지율 향상)
                
                checked_count = 0
                skipped_count = 0
                while not frame_queues[cam_id].empty() and checked_count < MAX_QUEUE_CHECK:
                    try:
                        frame_data = frame_queues[cam_id].get_nowait()
                        frame_ts = frame_data[1]
                        frame_age = current_time - frame_ts
                        checked_count += 1
                        
                        # 너무 오래된 프레임은 스킵
                        if frame_age > MAX_FRAME_AGE:
                            skipped_count += 1
                            continue
                        
                        temp_frames.append(frame_data)
                    except queue.Empty:
                        break
                
                # 프레임 유지율 최대화: 큐 정리 기준 완화 (더 많은 프레임 유지)
                # 큐가 매우 많이 쌓인 경우에만 정리 (MAX_QUEUE_SIZE * 1.5 이상)
                if frame_queues[cam_id].qsize() > int(MAX_QUEUE_SIZE * 1.5):
                    skipped_count = 0
                    max_cleanup = MAX_QUEUE_SIZE  # 정리 개수 제한 완화
                    cleanup_count = 0
                    while not frame_queues[cam_id].empty() and cleanup_count < max_cleanup:
                        try:
                            old_frame = frame_queues[cam_id].get_nowait()
                            frame_ts = old_frame[1] if isinstance(old_frame, tuple) and len(old_frame) > 1 else current_time
                            frame_age = current_time - frame_ts
                            # 0.5초 이상 오래된 프레임만 제거 (프레임 유지율 향상)
                            if frame_age > 0.5:
                                skipped_count += 1
                            else:
                                # 최신 프레임은 다시 큐에 넣기 (프레임 유지)
                                frame_queues[cam_id].put(old_frame)
                            cleanup_count += 1
                        except queue.Empty:
                            break
                    if skipped_count > 0:
                        logging.warning(f"[CAM-{cam_id}] 큐에 쌓인 오래된 프레임 {skipped_count}개 제거 (프레임 유지율 최적화)")
                
                if temp_frames:
                    # 타임스탬프 기준으로 정렬 (가장 최신 프레임 선택)
                    temp_frames.sort(key=lambda x: x[1], reverse=True)
                    frame_bytes, frame_timestamp = temp_frames[0]
                    
                    # 나머지 프레임은 모두 버림 (실시간성 우선, 큐 쌓임 방지)
                else:
                    # 큐에 유효한 프레임이 없으면 이전 프레임 유지하여 끊김 방지
                    with frame_lock:
                        if cam_id in latest_frames:
                            # 이전 프레임을 사용하여 끊김 방지
                            frame_bytes = latest_frames[cam_id]
                            frame_timestamp = current_time
                if cam_id in frame_queues and frame_queues[cam_id] is not None:
                    remaining_queue = frame_queues[cam_id].qsize()
                
                # 큐 크기 로깅 (큐가 많이 쌓일 때 경고)
                if remaining_queue > 10:
                    logging.warning(f"[PERF CAM-{cam_id}] 프레임 큐 크기 큼: {remaining_queue}개 (처리 대기 중)")
                elif remaining_queue > 5:
                    logging.info(f"[PERF CAM-{cam_id}] 프레임 큐 크기: {remaining_queue}개")
            
            # 프레임 대기 시간 계산 (프레임 타임스탬프 기준)
            if frame_bytes is not None and frame_timestamp is not None:
                frame_wait_time = (time.time() - frame_timestamp) * 1000  # ms
                if frame_wait_time > 200:  # 200ms 이상 대기한 프레임은 경고
                    logging.warning(f"[PERF CAM-{cam_id}] 오래된 프레임 처리: {frame_wait_time:.1f}ms 대기 (프레임 지연)")
            
            if frame_bytes is None:
                with processing_lock:
                    processing_flags[cam_id] = False
                break
            
            # AI 처리 수행 (전용 스레드 풀 사용, 비동기로 완전 분리)
            # 프레임 처리를 백그라운드 태스크로 실행하여 서버 응답성 유지
            try:
                # 연결 상태 재확인 (취소 전에 체크)
                if ws.closed:
                    break
                
                # GPU 사용 시 빠른 처리가 가능하므로 타임아웃 단축
                executor_start = time.time()
                try:
                    processed_frame_bytes, result_data = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            frame_processing_executor, 
                            process_single_frame, 
                            frame_bytes, 
                            cam_id
                        ),
                        timeout=2.5  # 프레임 처리 타임아웃 2.5초 (실시간 성능 최적화)
                    )
                    executor_elapsed = (time.time() - executor_start) * 1000
                    
                    # 프레임 처리 시간 로깅 (느린 처리 감지)
                    if executor_elapsed > 500:  # 500ms 이상이면 경고
                        logging.warning(f"[PERF CAM-{cam_id}] 느린 프레임 처리: {executor_elapsed:.1f}ms (목표: <500ms)")
                    
                    # 성능 데이터에서 병목 확인
                    if result_data and 'performance' in result_data:
                        perf = result_data['performance']
                        total_ms = perf.get('total', 0)
                        if total_ms > 1000:  # 1초 이상이면 상세 로깅
                            logging.warning(f"[PERF CAM-{cam_id}] 프레임 처리 상세: 총 {total_ms:.1f}ms, "
                                          f"YOLO={perf.get('yolo_violation', 0):.1f}ms, "
                                          f"Pose={perf.get('yolo_pose', 0):.1f}ms, "
                                          f"Face={perf.get('face_recognition', 0):.1f}ms, "
                                          f"Rendering={perf.get('rendering', 0):.1f}ms")
                    
                except asyncio.TimeoutError:
                    executor_elapsed = (time.time() - executor_start) * 1000
                    logging.error(f"[PERF CAM-{cam_id}] ⚠️ 프레임 처리 타임아웃: {executor_elapsed:.1f}ms (타임아웃: 2500ms)")
                    # 타임아웃 시 이전 프레임 유지
                    with frame_lock:
                        if cam_id in latest_frames:
                            processed_frame_bytes = latest_frames[cam_id]
                            result_data = latest_result_data.get(cam_id, {})
                        else:
                            # 프레임이 없으면 건너뛰기
                            with processing_lock:
                                processing_flags[cam_id] = False
                            continue
                
                # 처리된 프레임 저장 (MJPEG 스트림용 및 final과 호환)
                current_time = time.time()
                with frame_lock:
                    latest_frames[cam_id] = processed_frame_bytes
                    
                    # latest_frames 크기 제한 (메모리 최적화: 최대 10개 카메라 프레임만 캐시)
                    MAX_FRAMES_CACHE = 10
                    if len(latest_frames) > MAX_FRAMES_CACHE:
                        # 가장 오래 업데이트되지 않은 카메라 제거
                        with frame_stats_lock:
                            oldest_cam = min(latest_frames.keys(),
                                            key=lambda k: frame_stats.get(k, {}).get('last_frame_time', 0))
                            if oldest_cam != cam_id:  # 현재 카메라가 아니면 제거
                                del latest_frames[oldest_cam]
                    
                # 프레임 통계 업데이트 (FPS 계산용)
                with frame_stats_lock:
                    if cam_id not in frame_stats:
                        frame_stats[cam_id] = {
                            'frame_count': 0,
                            'last_frame_time': current_time,
                            'first_frame_time': current_time,
                            'frame_times': []  # 최근 60프레임의 시간 기록 (약 2초간)
                        }
                    
                    stats = frame_stats[cam_id]
                    stats['frame_count'] += 1
                    stats['last_frame_time'] = current_time
                    
                    # 최근 프레임 시간 기록 (FPS 계산용)
                    stats['frame_times'].append(current_time)
                    # 최근 60프레임만 유지 (약 2초간, 30 FPS 기준)
                    if len(stats['frame_times']) > 60:
                        stats['frame_times'] = stats['frame_times'][-60:]

                # 프레임 처리 시간 기록 (간격 제어용)
                with frame_interval_lock:
                    last_frame_processed_time[cam_id] = time.time()
                
                # 프레임 처리 통계 로깅 (매 30프레임마다)
                with frame_stats_lock:
                    frame_count = frame_stats.get(cam_id, {}).get('frame_count', 0)
                    if frame_count % 30 == 0 and frame_count > 0:
                        # FPS 계산
                        frame_times = frame_stats.get(cam_id, {}).get('frame_times', [])
                        if len(frame_times) >= 2:
                            time_span = frame_times[-1] - frame_times[0]
                            if time_span > 0:
                                current_fps = (len(frame_times) - 1) / time_span
                                logging.info(f"[PERF CAM-{cam_id}] FPS: {current_fps:.1f}, 큐: {remaining_queue}개, "
                                           f"처리 시간: {executor_elapsed:.1f}ms")
                
                # 처리된 프레임은 latest_frames에 저장 (MJPEG 스트림용)
                # JPEG 프레임은 전송하지 않음 - JSON 결과만 전송하여 네트워크 대역폭 절약

                # 결과 데이터를 JSON으로 전송 및 저장
                # result_data가 비어있어도 프레임은 저장해야 함 (검은 프레임도 스트림에 표시)
                # 디버깅: result_data 상태 확인
                if cam_id in frame_stats:
                    with frame_stats_lock:
                        stats = frame_stats.get(cam_id, {})
                        frame_count = stats.get('frame_count', 0)
                
                # 연결 상태 재확인 (처리 중에 연결이 끊겼을 수 있음)
                if ws.closed:
                    logging.warning(f"[CAM-{cam_id}] WebSocket이 닫혀있어 결과 전송 불가 (클라이언트 {client_id}): result_data={result_data is not None}, 위반={result_data.get('violation_count', 0) if result_data else 0}, 얼굴={len(result_data.get('recognized_faces', [])) if result_data else 0}")
                    break
                
                # result_data 검증 및 로깅 (디버깅용)
                if result_data is None:
                    pass
                elif not result_data:  # 빈 딕셔너리 체크
                    pass
                else:
                    pass
                
                if result_data:
                    # 타임스탬프/프레임ID/지연 정보 부여 (프론트 동기화용)
                    try:
                        with frame_stats_lock:
                            stats = frame_stats.get(cam_id, {})
                            frame_id = int(stats.get('frame_count', 0))
                        result_data['cam_id'] = cam_id
                        result_data['frame_id'] = frame_id
                        result_data['ts_ms'] = int(current_time * 1000)
                        if frame_timestamp is not None:
                            result_data['latency_ms'] = int((current_time - frame_timestamp) * 1000)
                    except Exception:
                        pass
                    
                    # 최신 결과 데이터 저장 (HTTP 폴링용)
                    with frame_lock:
                        latest_result_data[cam_id] = result_data.copy() if result_data else {}
                        # 디버깅: 결과 데이터 저장 확인
                        faces_count = len(result_data.get('recognized_faces', [])) if result_data else 0
                        violations_count = len(result_data.get('violations', [])) if result_data else 0

                    # model_results 업데이트 (final과 동일한 구조)
                    # KPI 업데이트: 위반이 있거나 얼굴이 인식되면 항상 업데이트
                    # 검은 프레임이 아닌 경우 항상 KPI 업데이트 (작업자 수 표시를 위해)
                    if result_data.get("violation_count", 0) > 0 or len(result_data.get("recognized_faces", [])) > 0:
                        update_model_results_from_frame(result_data, cam_id)
                    # 모든 카메라 데이터를 합산하여 KPI 업데이트 (항상 실행)
                    update_kpi_from_all_cameras()
                    # 대시보드에 업데이트된 결과 전송 (비동기로 실행, 블로킹 방지)
                    asyncio.create_task(broadcast_model_results())

                    # 연결 상태 확인 후 전송
                    if ws.closed:
                        logging.warning(f"[CAM-{cam_id}] WebSocket이 닫혀있어 결과 전송 불가 (클라이언트 {client_id}): ws.closed={ws.closed}, ws.id={id(ws)}")
                        break
                    
                    # WebSocket 인스턴스 확인 (디버깅용)
                    ws_id = id(ws)
                    is_in_connected = ws in connected_websockets
                    
                    try:
                        result_json = json.dumps(result_data)
                        
                        # 전송 전 최종 확인
                        if ws.closed:
                            logging.warning(f"[CAM-{cam_id}] WebSocket이 전송 직전에 닫혔습니다 (클라이언트 {client_id}, ws.id={ws_id})")
                            break
                        
                        # 실제 전송 (모든 결과 전송 - 빈 결과도 포함)
                        await ws.send_str(result_json)
                        
                        # 전송 후 확인
                        if ws.closed:
                            logging.warning(f"[CAM-{cam_id}] WebSocket이 전송 직후에 닫혔습니다 (클라이언트 {client_id}, ws.id={ws_id})")
                            break
                        
                        # 전송 성공 로그 (모든 결과 로깅 - 빈 결과도 포함)
                    except (ConnectionResetError, ConnectionError, OSError) as conn_err:
                        logging.warning(f"[CAM-{cam_id}] 결과 전송 실패 - 연결 끊김 (클라이언트 {client_id}): {conn_err}")
                        break
                    except Exception as send_err:
                        # 기타 예외도 연결 끊김으로 간주
                        if "closing transport" in str(send_err).lower() or "closed" in str(send_err).lower():
                            break
                        else:
                            logging.warning(f"[CAM-{cam_id}] 결과 전송 오류 (클라이언트 {client_id}): {send_err}")
                else:
                    # result_data가 없어도 빈 결과로 저장 및 전송 (스트림 표시용)
                    empty_result = {
                        "recognized_faces": [], 
                        "violations": [], 
                        "violation_count": 0, 
                        "cam_id": cam_id,
                        "frame_id": 0,
                        "ts_ms": int(time.time() * 1000)
                    }
                    with frame_lock:
                        latest_result_data[cam_id] = empty_result
                    
                    # 빈 프레임이어도 KPI 업데이트 (인원 수가 0으로 변경될 수 있음)
                    update_kpi_from_all_cameras()
                    
                    # 대시보드 연결들에게 model_results 브로드캐스트 (빈 상태도 전송)
                    asyncio.create_task(broadcast_model_results())
                    
                    # 빈 결과도 클라이언트에 전송 (프론트엔드가 데이터 수신 확인 가능)
                    if ws.closed:
                        break
                    else:
                        try:
                            empty_result_json = json.dumps(empty_result)
                            ws_id = id(ws)
                            is_in_connected = ws in connected_websockets
                            
                            await ws.send_str(empty_result_json)
                            
                        except (ConnectionResetError, ConnectionError, OSError) as conn_err:
                            logging.warning(f"[CAM-{cam_id}] 빈 결과 전송 실패 - 연결 끊김 (클라이언트 {client_id}): {conn_err}")
                            break
                        except Exception as send_err:
                            if "closing transport" in str(send_err).lower() or "closed" in str(send_err).lower():
                                break
                            else:
                                logging.warning(f"[CAM-{cam_id}] 빈 결과 전송 오류 (클라이언트 {client_id}): {send_err}")
                        
            except asyncio.TimeoutError:
                # 프레임 처리 타임아웃 - 큐 정리 및 이전 프레임 유지하여 끊김 방지
                timeout_elapsed = (time.time() - iteration_start) * 1000
                logging.warning(f"프레임 처리 타임아웃 (5초 초과, 누적 {timeout_elapsed:.2f}ms). 큐 정리 및 이전 프레임 유지. (클라이언트 {client_id}, CAM-{cam_id})")
                
                # 큐에 쌓인 오래된 프레임 모두 제거 (백엔드 멈춤 방지)
                with queue_lock:
                    if cam_id in frame_queues:
                        queue_size_before = frame_queues[cam_id].qsize()
                        # 큐 비우기 (오래된 프레임 모두 제거)
                        while not frame_queues[cam_id].empty():
                            try:
                                frame_queues[cam_id].get_nowait()
                            except queue.Empty:
                                break
                        if queue_size_before > 0:
                            logging.warning(f"[CAM-{cam_id}] 타임아웃 발생으로 큐에 쌓인 프레임 {queue_size_before}개 모두 제거")
                
                # 이전 프레임이 있으면 유지하여 끊김 방지
                with frame_lock:
                    if cam_id in latest_frames:
                        # 이전 프레임을 다시 전송하여 끊김 방지
                        try:
                            await ws.send_bytes(latest_frames[cam_id])
                            # 빈 결과 데이터 전송
                            empty_result = {"recognized_faces": [], "violations": [], "violation_count": 0, "cam_id": cam_id}
                            await ws.send_str(json.dumps(empty_result))
                        except (ConnectionResetError, ConnectionError, OSError):
                            break
                
                # 처리 플래그 해제하여 다음 프레임 처리 가능하도록
                with processing_lock:
                    processing_flags[cam_id] = False
                
                # 짧은 대기 후 다음 프레임 처리 (백엔드 부하 완화)
                await asyncio.sleep(0.1)
                continue  # 다음 프레임으로 진행

            except Exception as e:
                logging.error(f"AI 처리 실행 중 오류: {e}", exc_info=True)
                
                # 오류 프레임 생성 및 전송
                try:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    error_msg_short = str(e)[:40] + ("..." if len(str(e)) > 40 else "")
                    cv2.putText(error_frame, "Processing Error", (50, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(error_frame, error_msg_short, (50, 260), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    await ws.send_bytes(buffer.tobytes())
                    
                    # 오류 메시지도 JSON으로 전송
                    error_msg = json.dumps({"type": "error", "message": f"Server processing error: {e}"})
                    await ws.send_str(error_msg)
                except (ConnectionResetError, ConnectionError, OSError):
                    pass
                except Exception as send_err:
                    logging.warning(f"오류 프레임 전송 실패: {send_err}")
                
                # 오류 발생 시에도 플래그 해제하여 다음 프레임 처리 가능
                with processing_lock:
                    processing_flags[cam_id] = False
                continue  # 오류 발생해도 다음 프레임 처리 계속

    except (ConnectionResetError, ConnectionError, OSError) as e:
        logging.info(f"프레임 큐 처리 중 연결 오류 (클라이언트 {client_id}, CAM-{cam_id}): {e}")
        with processing_lock:
            if cam_id in processing_flags:
                processing_flags[cam_id] = False
    except asyncio.CancelledError:
        logging.info(f"프레임 큐 처리 취소됨 (클라이언트 {client_id}, CAM-{cam_id})")
        # 취소 시 즉시 종료 (리소스 정리는 finally에서 처리)
        with processing_lock:
            if cam_id in processing_flags:
                processing_flags[cam_id] = False
        raise  # CancelledError를 다시 발생시켜 즉시 종료
    except RuntimeError as e:
        # WebSocket이 이미 닫혔거나 취소된 경우
        error_msg = str(e).lower()
        if "closed" in error_msg or "cancelled" in error_msg:
            logging.info(f"프레임 큐 처리 중 연결 종료 (클라이언트 {client_id}, CAM-{cam_id}): {e}")
        else:
            logging.warning(f"프레임 큐 처리 중 RuntimeError (클라이언트 {client_id}, CAM-{cam_id}): {e}")
        with processing_lock:
            if cam_id in processing_flags:
                processing_flags[cam_id] = False
    except Exception as e:
        # 예상치 못한 오류는 로깅하되 서버를 멈추지 않음
        error_msg = str(e).lower()
        # 일반적인 연결 종료 관련 오류는 INFO 레벨로 처리
        if any(keyword in error_msg for keyword in ['closed', 'cancelled', 'disconnect', 'reset', 'broken pipe', 'connection']):
            logging.info(f"프레임 큐 처리 중 연결 종료 관련 오류 (클라이언트 {client_id}, CAM-{cam_id}): {e}")
        else:
            logging.error(f"프레임 큐 처리 중 예상치 못한 오류 (클라이언트 {client_id}, CAM-{cam_id}): {e}", exc_info=True)
        with processing_lock:
            if cam_id in processing_flags:
                processing_flags[cam_id] = False
    finally:
        # 태스크 추적에서 제거
        try:
            async with processing_tasks_lock:
                if cam_id in processing_tasks and client_id in processing_tasks[cam_id]:
                    del processing_tasks[cam_id][client_id]
                    if not processing_tasks[cam_id]:
                        del processing_tasks[cam_id]
        except Exception as e:
            logging.warning(f"태스크 추적에서 제거 중 오류 (CAM-{cam_id}, 클라이언트 {client_id}): {e}")
        
        # 항상 플래그 해제 및 큐 정리 (재연결을 위해 필수, 메모리 누수 방지)
        try:
            with processing_lock:
                if cam_id in processing_flags:
                    processing_flags[cam_id] = False
        except Exception as e:
            logging.warning(f"프레임 처리 플래그 해제 중 오류 (CAM-{cam_id}, 클라이언트 {client_id}): {e}")
        
        # 큐 정리 (메모리 누수 방지)
        try:
            with queue_lock:
                if cam_id in frame_queues:
                    try:
                        while not frame_queues[cam_id].empty():
                            frame_queues[cam_id].get_nowait()
                    except queue.Empty:
                        pass
                    except Exception:
                        pass
        except Exception as e:
            logging.warning(f"프레임 큐 정리 중 오류 (CAM-{cam_id}, 클라이언트 {client_id}): {e}")
        


# --- 카메라 리소스 완전 정리 함수 ---
async def cleanup_camera_resources(cam_id: int, client_id: Optional[int] = None) -> None:
    """
    카메라별 모든 리소스를 완전히 정리
    카메라 전환 시 이전 카메라 데이터가 남지 않도록 보장
    """
    logging.info(f"[CLEANUP] 카메라 {cam_id} 리소스 정리 시작 (클라이언트 {client_id})")
    
    # 1. 프레임 및 결과 데이터 정리
    try:
        with frame_lock:
            if cam_id in latest_frames:
                del latest_frames[cam_id]
            if cam_id in latest_result_data:
                del latest_result_data[cam_id]
    except Exception as e:
        logging.warning(f"[CLEANUP] 프레임/결과 데이터 정리 중 오류 (CAM-{cam_id}): {e}")
    
    # 2. 프레임 통계 정리
    try:
        with frame_stats_lock:
            if cam_id in frame_stats:
                del frame_stats[cam_id]
    except Exception as e:
        logging.warning(f"[CLEANUP] 프레임 통계 정리 중 오류 (CAM-{cam_id}): {e}")
    
    # 3. 얼굴 인식 캐시 정리
    try:
        recent_identity_cache.clear(cam_id)
        if cam_id in embedding_buffers:
            del embedding_buffers[cam_id]
        if cam_id in face_bbox_cache:
            face_bbox_cache[cam_id].clear()
        if cam_id in centroid_cache:
            centroid_cache[cam_id].clear()
        if cam_id in fall_start_times:
            del fall_start_times[cam_id]
    except Exception as e:
        logging.warning(f"[CLEANUP] 얼굴 인식 캐시 정리 중 오류 (CAM-{cam_id}): {e}")
    
    # 4. 기타 카메라별 캐시 정리
    try:
        with face_detection_lock:
            if cam_id in last_face_detection_frame:
                del last_face_detection_frame[cam_id]
        
        if cam_id in face_recognition_cooldown_ts:
            del face_recognition_cooldown_ts[cam_id]
        
        if cam_id in last_render_cache:
            last_render_cache[cam_id].clear()
    except Exception as e:
        logging.warning(f"[CLEANUP] 기타 캐시 정리 중 오류 (CAM-{cam_id}): {e}")
    
    # 5. 프레임 간격 제어 데이터 정리
    try:
        with frame_interval_lock:
            if cam_id in last_frame_processed_time:
                del last_frame_processed_time[cam_id]
    except Exception as e:
        logging.warning(f"[CLEANUP] 프레임 간격 데이터 정리 중 오류 (CAM-{cam_id}): {e}")
    
    # 6. 처리 플래그 정리 (마지막에 수행)
    try:
        with processing_lock:
            if cam_id in processing_flags:
                processing_flags[cam_id] = False
    except Exception as e:
        logging.warning(f"[CLEANUP] 처리 플래그 정리 중 오류 (CAM-{cam_id}): {e}")
    
    # 7. 큐 정리 (마지막에 수행)
    try:
        with queue_lock:
            if cam_id in frame_queues:
                try:
                    while not frame_queues[cam_id].empty():
                        frame_queues[cam_id].get_nowait()
                except queue.Empty:
                    pass
                del frame_queues[cam_id]
    except Exception as e:
        logging.warning(f"[CLEANUP] 프레임 큐 정리 중 오류 (CAM-{cam_id}): {e}")
    
    logging.info(f"[CLEANUP] 카메라 {cam_id} 리소스 정리 완료 (클라이언트 {client_id})")


# --- 웹소켓 핸들러 ---
async def websocket_handler(request: web.Request):
    """클라이언트용 WebSocket 핸들러 - 프레임 수신 및 처리"""
    try:
        # 입력 검증
        cam_id = validate_camera_id(request.query.get('cam_id', '0'))
    except ValidationError as e:
        logging.error(f"[WebSocket] 카메라 ID 검증 실패: {e.message}")
        return web.json_response(e.to_dict(), status=400)
    
    ws = web.WebSocketResponse()
    
    # WebSocket 연결 준비
    try:
        await ws.prepare(request)
    except asyncio.TimeoutError as e:
        logging.error(f"[WebSocket] 연결 준비 타임아웃: {e} (cam_id={cam_id})", exc_info=True)
        return ws
    except Exception as e:
        logging.error(f"[WebSocket] 연결 준비 실패: {e} (cam_id={cam_id})", exc_info=True)
        raise CameraError(
            f"WebSocket 연결 준비 실패: {e}",
            error_code="WEBSOCKET_PREPARE_FAILED",
            details={"cam_id": cam_id}
        ) from e
    
    connected_websockets.add(ws)
    client_id = id(ws)  # 클라이언트 식별용

    # 연결 확인 메시지 전송 (선택적)
    try:
        await ws.send_str(json.dumps({"type": "connected", "message": "서버 연결 성공", "cam_id": cam_id, "ts_ms": int(time.time() * 1000)}))
    except Exception:
        pass

    # 백그라운드 워커 결과 브로드캐스트 태스크 시작
    broadcast_task = None
    try:
        # 백그라운드 워커 결과 브로드캐스트
        async def broadcast_worker_results():
            """백그라운드 워커의 결과를 WebSocket으로 전송"""
            from camera_worker import get_camera_buffer
            last_sent_timestamp = 0
            
            while True:
                try:
                    await asyncio.sleep(1/60)  # 60fps로 체크 (실제 전송은 새 결과가 있을 때만)
                    
                    # 연결 상태 확인
                    if ws.closed:
                        break
                    
                    # 카메라 버퍼에서 최신 결과 가져오기
                    buffer = get_camera_buffer(cam_id)
                    if not buffer:
                        continue
                    
                    result = buffer.get("latest_result")
                    if not result:
                        continue
                    
                    # 새로운 결과가 있을 때만 전송
                    result_timestamp = result.get("timestamp", 0) or result.get("ts_ms", 0)
                    if result_timestamp > last_sent_timestamp:
                        # 타임스탬프 추가 (없으면 추가)
                        if "ts_ms" not in result:
                            result["ts_ms"] = result_timestamp
                        
                        # WebSocket으로 전송 (프론트엔드 형식에 맞춤)
                        try:
                            # 프론트엔드가 기대하는 형식: {"type": "model_results", "result": {...}}
                            message = {
                                "type": "model_results",
                                "result": result
                            }
                            result_json = json.dumps(message)
                            await ws.send_str(result_json)
                            last_sent_timestamp = result_timestamp
                        except Exception as send_err:
                            if "closing" in str(send_err).lower() or "closed" in str(send_err).lower():
                                break
                            logging.warning(f"결과 전송 오류 (CAM-{cam_id}): {send_err}")
                            break
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"브로드캐스트 오류 (CAM-{cam_id}): {e}", exc_info=True)
                    break
        
        broadcast_task = asyncio.create_task(broadcast_worker_results())
        
        # WebSocket 메시지 수신 루프 시작 로그
        
        async for msg in ws:
            # 메시지 타입 확인 및 로깅 (처음 몇 개만)
            if not hasattr(ws, '_msg_count'):
                ws._msg_count = 0
            ws._msg_count += 1
            
            if msg.type == WSMsgType.BINARY:
                # 클라이언트로부터 프레임 수신 (선택적 - 백그라운드 워커가 있으면 사용하지 않음)
                # 백그라운드 워커가 실행 중이면 프레임을 받지 않고 결과만 수신
                frame_bytes = msg.data
                
                # 프레임 수신 로깅 (처음 10개는 상세 로깅)
                if not hasattr(ws, '_frame_receive_count'):
                    ws._frame_receive_count = 0
                ws._frame_receive_count += 1
                
                if ws._frame_receive_count <= 10 or ws._frame_receive_count % 30 == 0:
                    logging.info(f"[CAM-{cam_id}] 참고: 백그라운드 워커가 실행 중이면 프레임은 무시되고 결과만 전송됩니다")

                # 백그라운드 워커가 실행 중이면 프레임 처리 건너뛰기
                # (브로드캐스트 태스크가 결과를 전송함)
                from camera_worker import get_camera_buffer
                buffer = get_camera_buffer(cam_id)
                if buffer and buffer.get("latest_result") is not None:
                    # 백그라운드 워커가 실행 중이므로 프레임 무시
                    continue

                # 프레임 크기 검증: 너무 작으면 처리하지 않음
                if len(frame_bytes) < 100:  # 최소 100 bytes 이상이어야 함
                    continue

                # 프레임 큐 시스템: 최신 프레임 우선 처리 (딜레이 최소화)
                # 큐가 가득 찬 경우 오래된 프레임 제거하고 최신 프레임 추가
                with queue_lock:
                    if cam_id not in frame_queues:
                        frame_queues[cam_id] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
                    
                    frame_queue = frame_queues[cam_id]
                    
                    # 큐가 가득 찬 경우 오래된 프레임 제거 (최신 프레임 우선, 처리 속도 개선)
                    # 큐가 가득 차면 여러 개의 오래된 프레임을 버려서 처리 속도 향상
                    # FPS 저하 방지를 위해 큐 크기의 50%를 비움
                    if frame_queue.full():
                        dropped_count = 0
                        drain_count = max(1, int(frame_queue.maxsize * 0.5))  # 큐의 절반을 비움
                        for _ in range(drain_count):
                            try:
                                frame_queue.get_nowait()
                                dropped_count += 1
                            except queue.Empty:
                                break
                    
                    # 최신 프레임 추가
                    try:
                        frame_queue.put_nowait((frame_bytes, time.time()))
                    except queue.Full:
                        pass
                
                # 프레임 처리 중이 아니면 즉시 처리 시작 (비동기, 완전 분리)
                # 연결이 살아있는지 확인 후 처리 시작
                # 프레임 처리는 백그라운드에서 실행되므로 서버 응답성에 영향 없음
                if not ws.closed:
                    with processing_lock:
                        if not processing_flags.get(cam_id, False):
                            processing_flags[cam_id] = True
                            # 프레임 처리 태스크 시작 (비동기로 실행, 서버 블로킹 방지)
                            # create_task로 완전히 분리하여 서버가 다른 요청 처리 가능
                            task = asyncio.create_task(process_frame_queue(cam_id, ws, client_id))
                            # 태스크 추적 (연결 종료 시 취소하기 위해)
                            async with processing_tasks_lock:
                                if cam_id not in processing_tasks:
                                    processing_tasks[cam_id] = {}
                                processing_tasks[cam_id][client_id] = task
                        # 이미 처리 중이면 큐에 프레임이 추가되면 다음 처리 시 자동으로 처리됨
                else:
                    logging.warning(f"클라이언트 {client_id} 연결이 이미 닫혔습니다. 프레임 처리 건너뜀 (CAM-{cam_id})")

            elif msg.type == WSMsgType.TEXT:
                # 텍스트 메시지 처리 (ping/pong 등)
                try:
                    data = json.loads(msg.data)
                    if data.get("type") == "ping":
                        await ws.send_str(json.dumps({"type": "pong"}))
                except:
                    pass
            elif msg.type == WSMsgType.CLOSE:
                logging.info(f"웹소켓 클라이언트 {client_id} 종료 요청 수신 (CAM-{cam_id})")
                break
            elif msg.type == WSMsgType.ERROR:
                logging.warning(f"웹소켓 클라이언트 {client_id} 오류 발생: {ws.exception()}")
                break

    except (ConnectionResetError, ConnectionError, OSError) as e:
        logging.info(f"클라이언트 {client_id} 연결이 끊어졌습니다. ({e})")
    except asyncio.CancelledError:
        logging.info(f"클라이언트 {client_id} 연결이 취소되었습니다. (CAM-{cam_id})")
    except RuntimeError as e:
        # WebSocket이 이미 닫혔거나 취소된 경우
        if "closed" in str(e).lower() or "cancelled" in str(e).lower():
            logging.info(f"클라이언트 {client_id} 연결이 이미 종료되었습니다. ({e})")
        else:
            logging.warning(f"WebSocket 핸들러 RuntimeError: {e}")
    except Exception as e:
        # 예상치 못한 오류는 로깅하되 서버를 멈추지 않음
        error_msg = str(e)
        # 일반적인 연결 종료 관련 오류는 INFO 레벨로 처리
        if any(keyword in error_msg.lower() for keyword in ['closed', 'cancelled', 'disconnect', 'reset', 'broken pipe']):
            logging.info(f"클라이언트 {client_id} 연결 종료 관련 오류 (무시): {error_msg}")
        else:
            logging.error(f"WebSocket 핸들러 예상치 못한 오류: {e}", exc_info=True)
    finally:
        # 브로드캐스트 태스크 취소
        if broadcast_task and not broadcast_task.done():
            broadcast_task.cancel()
            try:
                await asyncio.wait_for(broadcast_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # 실행 중인 프레임 처리 태스크 취소 (우선순위 높음)
        # 타임아웃을 추가하여 블로킹 방지
        try:
            async with processing_tasks_lock:
                if cam_id in processing_tasks and client_id in processing_tasks[cam_id]:
                    task = processing_tasks[cam_id][client_id]
                    if not task.done():
                        task.cancel()
                        try:
                            # 태스크 취소 대기 시 타임아웃 추가 (1초) - 블로킹 방지
                            await asyncio.wait_for(task, timeout=1.0)
                        except asyncio.TimeoutError:
                            logging.warning(f"프레임 처리 태스크 취소 타임아웃 (CAM-{cam_id}, 클라이언트 {client_id}): 1초 내에 완료되지 않음 - 강제 종료")
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logging.warning(f"프레임 처리 태스크 취소 중 오류: {e}")
                    del processing_tasks[cam_id][client_id]
                    if not processing_tasks[cam_id]:
                        del processing_tasks[cam_id]
                        logging.info(f"[CLEANUP] 카메라 {cam_id}의 모든 처리 태스크 종료됨")
        except Exception as e:
            logging.warning(f"프레임 처리 태스크 취소 중 오류: {e}")
        
        # 안전한 cleanup (예외 발생해도 실행)
        try:
            connected_websockets.discard(ws)
            dashboard_websockets.discard(ws)
        except Exception as e:
            logging.warning(f"WebSocket 세트에서 제거 중 오류: {e}")
        
        # 카메라에 연결된 다른 클라이언트가 있는지 확인
        has_other_clients = False
        try:
            async with processing_tasks_lock:
                if cam_id in processing_tasks and len(processing_tasks[cam_id]) > 0:
                    has_other_clients = True
                    logging.info(f"[CLEANUP] 카메라 {cam_id}에 다른 클라이언트 {len(processing_tasks[cam_id])}개 연결 중 - 리소스 유지")
        except Exception as e:
            logging.warning(f"다른 클라이언트 확인 중 오류: {e}")
        
        # 다른 클라이언트가 없으면 카메라 리소스 완전 정리
        if not has_other_clients:
            logging.info(f"[CLEANUP] 카메라 {cam_id}에 연결된 클라이언트 없음 - 전체 리소스 정리 시작")
            try:
                # 비동기로 정리하여 블로킹 방지 (타임아웃을 2초로 단축 - 더 빠른 응답)
                # cleanup은 백그라운드에서 실행되도록 create_task로 분리
                cleanup_task = asyncio.create_task(
                    asyncio.wait_for(cleanup_camera_resources(cam_id, client_id), timeout=2.0)
                )
                # cleanup 작업을 기다리지 않고 즉시 반환 (백그라운드에서 실행)
                # 이렇게 하면 새로고침 시 백엔드가 멈추지 않음
                try:
                    await asyncio.wait_for(cleanup_task, timeout=0.5)  # 최대 0.5초만 대기
                except asyncio.TimeoutError:
                    # 백그라운드에서 계속 실행되도록 태스크는 유지
                    pass
            except asyncio.TimeoutError:
                logging.warning(f"[CLEANUP] 카메라 리소스 정리 타임아웃 (CAM-{cam_id}): 2초 내에 완료되지 않음")
            except Exception as e:
                logging.error(f"[CLEANUP] 카메라 리소스 정리 중 오류 (CAM-{cam_id}): {e}", exc_info=True)
        else:
            # 다른 클라이언트가 있으면 최소한의 정리만 수행
            try:
                with processing_lock:
                    if cam_id in processing_flags:
                        processing_flags[cam_id] = False
            except Exception as e:
                logging.warning(f"처리 플래그 해제 중 오류: {e}")
            
            # 큐는 정리하지 않음 (다른 클라이언트가 사용 중)
            logging.info(f"[CLEANUP] 카메라 {cam_id} 최소 정리 완료 (다른 클라이언트 존재)")
        
        # WebSocket 명시적 종료 (안전하게)
        try:
            if not ws.closed:
                await ws.close()
        except Exception as e:
            # 이미 닫혔거나 닫는 중 오류는 무시
            pass
        
        logging.info(f"웹소켓 클라이언트 {client_id} 연결 종료. (현재 {len(connected_websockets)}명 접속 중)")


# --- API 핸들러들 ---
async def api_status_handler(request: web.Request):
    """시스템 상태 API 엔드포인트"""
    try:
        # 시스템 상태 확인
        system_status = "running"
        camera_count = len(latest_frames)
        
        # 카메라가 없으면 테스트 모드로 표시
        if camera_count == 0:
            system_status = "test_mode"
        
        # 전체 프레임 통계 계산
        total_frames = 0
        total_fps = 0.0
        current_time = time.time()
        
        with frame_stats_lock:
            for cam_id, stats in frame_stats.items():
                total_frames += stats['frame_count']
                
                # 각 카메라의 FPS 계산
                frame_times = stats['frame_times']
                if len(frame_times) >= 2:
                    recent_frames = [t for t in frame_times if current_time - t <= 2.0]
                    if len(recent_frames) >= 2:
                        time_span = recent_frames[-1] - recent_frames[0]
                        if time_span > 0:
                            total_fps += (len(recent_frames) - 1) / time_span
        
        status_data = {
            "system_status": system_status,
            "cameras": camera_count,
            "connected_clients": len(connected_websockets),
            "uptime": time.time() - system_stats["start_time"],
            "test_mode": camera_count == 0,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_stats": {
                "total_frames_processed": total_frames,
                "average_fps": round(total_fps, 2),
                "active_cameras": camera_count
            }
        }
        
        response = create_standard_response(data=status_data, message="시스템 상태 조회 성공")
        return web.json_response(response)
    except Exception as e:
        logging.error(f"API 상태 조회 중 오류: {e}")
        response = create_standard_response(
            status="error", 
            message=f"시스템 상태 조회 실패: {str(e)}", 
            error_code="STATUS_ERROR"
        )
        return web.json_response(response, status=500)

async def api_cameras_handler(request: web.Request):
    """카메라 목록 API 엔드포인트"""
    try:
        cameras_data = []
        
        # 현재 활성 카메라 수 파악
        # 1. latest_frames에서 카메라 확인 (프레임이 있는 카메라)
        with frame_lock:
            active_cameras_from_frames = set(latest_frames.keys())
        
        # 2. frame_stats에서 카메라 확인 (통계가 있는 카메라)
        with frame_stats_lock:
            active_cameras_from_stats = set(frame_stats.keys())
        
        # 3. 웹소켓 연결이 있으면 기본 카메라(0)도 포함 (클라이언트가 연결되어 있지만 아직 프레임이 없을 수 있음)
        active_cameras = active_cameras_from_frames | active_cameras_from_stats
        if len(connected_websockets) > 0 and 0 not in active_cameras:
            # 웹소켓 연결이 있지만 카메라 정보가 없으면 기본 카메라(0) 추가
            active_cameras.add(0)
        
        # 카메라 ID 정렬
        active_cameras = sorted(list(active_cameras))
        
        # 각 카메라의 상세 정보 수집
        current_time = time.time()
        for cam_id in active_cameras:
            # 프레임 통계에서 실시간 FPS 계산
            actual_fps = 0.0
            frame_count = 0
            last_frame_time = 0.0
            time_since_first = 0.0
            
            with frame_stats_lock:
                if cam_id in frame_stats:
                    stats = frame_stats[cam_id]
                    frame_count = stats['frame_count']
                    last_frame_time = stats['last_frame_time']
                    time_since_first = current_time - stats['first_frame_time']
                    
                    # 최근 프레임 시간을 기반으로 FPS 계산
                    frame_times = stats['frame_times']
                    if len(frame_times) >= 2:
                        # 최근 2초간의 프레임 간격으로 FPS 계산
                        recent_frames = [t for t in frame_times if current_time - t <= 2.0]
                        if len(recent_frames) >= 2:
                            time_span = recent_frames[-1] - recent_frames[0]
                            if time_span > 0:
                                actual_fps = (len(recent_frames) - 1) / time_span
                        elif time_since_first > 0:
                            # 전체 평균 FPS
                            actual_fps = frame_count / time_since_first
            
            # FPS가 0이면 기본값 사용
            if actual_fps == 0:
                actual_fps = 0.0
            
            # 카메라 상태 결정
            # 1. FPS가 있으면 active
            # 2. 프레임 통계가 있지만 FPS가 0이면 connecting (처리 중)
            # 3. 웹소켓 연결이 있지만 통계가 없으면 connecting (연결 대기)
            # 4. 그 외는 no_frames
            if actual_fps > 0:
                camera_status = "active"
            elif frame_count > 0 or time_since_first > 0:
                camera_status = "connecting"  # 처리 중이지만 아직 프레임 없음
            elif len(connected_websockets) > 0:
                camera_status = "connecting"  # 웹소켓 연결은 있지만 아직 프레임 없음
            else:
                camera_status = "no_frames"
            
            cameras_data.append({
                "id": cam_id,
                "name": f"카메라 {cam_id}",
                "status": camera_status,
                "last_frame_time": last_frame_time,
                "stream_url": f"/video_feed/{cam_id}",
                "resolution": "1280x720",
                "fps": round(actual_fps, 2),  # 소수점 2자리까지
                "total_frames": frame_count,
                "uptime_seconds": round(time_since_first, 2),
                "is_streaming": actual_fps > 0
            })
        
        # 카메라 개수 정보 및 시스템 상태 포함
        camera_info = {
            "cameras": cameras_data,
            "total_cameras": len(cameras_data),
            "active_cameras": len(cameras_data),
            "system_status": "running" if cameras_data else "no_cameras",
            "last_updated": time.time()
        }
        
        response_data = create_standard_response(data=camera_info, message=f"카메라 목록 조회 성공 - {len(cameras_data)}개 카메라 감지")
        return create_compressed_response(response_data)
    except Exception as e:
        logging.error(f"카메라 목록 조회 중 오류: {e}")
        response = create_standard_response(status="error", message=f"카메라 목록 조회 실패: {str(e)}", error_code="CAMERAS_ERROR")
        return web.json_response(response, status=500)

async def api_camera_results_handler(request: web.Request):
    """카메라별 AI 결과 API 엔드포인트 (HTTP 폴링용)"""
    try:
        # 입력 검증
        cam_id = validate_camera_id(request.query.get('cam_id', '0'))
        
        # 처리 중인지 확인
        is_processing = False
        with processing_lock:
            is_processing = processing_flags.get(cam_id, False)
        
        # 모든 카메라의 latest_result_data 상태 확인 (디버깅)
        with frame_lock:
            all_cameras_data = {}
            for cid in latest_result_data.keys():
                cam_data = latest_result_data.get(cid, {})
                all_cameras_data[cid] = {
                    'has_data': bool(cam_data),
                    'faces': len(cam_data.get('recognized_faces', [])) if cam_data else 0,
                    'violations': len(cam_data.get('violations', [])) if cam_data else 0,
                    'frame_id': cam_data.get('frame_id', 'N/A') if cam_data else 'N/A'
                }
            
            result_data = latest_result_data.get(cam_id, {
                "recognized_faces": [],
                "violations": [],
                "violation_count": 0
            })
            
            # 처리 중이고 결과가 없으면 이전 결과 유지 (빈 결과 반환 방지)
            if is_processing and (not result_data or (not result_data.get('recognized_faces') and not result_data.get('violations'))):
                # 이전 결과가 있으면 그대로 반환 (처리 중임을 표시)
                if cam_id in latest_result_data and latest_result_data[cam_id]:
                    prev_data = latest_result_data[cam_id]
                    # 이전 데이터에 얼굴이나 위반이 있으면 그대로 반환
                    if prev_data.get('recognized_faces') or prev_data.get('violations'):
                        result_data = prev_data.copy()
                        result_data['processing'] = True  # 처리 중 플래그 추가
        
        # 모든 카메라 상태 로깅 (10번에 1번만)
        if not hasattr(api_camera_results_handler, '_log_count'):
            api_camera_results_handler._log_count = {}
        if cam_id not in api_camera_results_handler._log_count:
            api_camera_results_handler._log_count[cam_id] = 0
        api_camera_results_handler._log_count[cam_id] += 1
        
        # 타임스탬프 추가 (복사본에 추가하여 원본 데이터 보호)
        result_data = result_data.copy() if result_data else {}
        result_data['cam_id'] = cam_id
        result_data['ts_ms'] = int(time.time() * 1000)
        
        # 디버깅: 결과 데이터 상세 로깅
        faces_count = len(result_data.get('recognized_faces', []))
        violations_count = len(result_data.get('violations', []))
        frame_id = result_data.get('frame_id', 'N/A')
        
        if faces_count == 0 and violations_count == 0:
            pass  # 빈 결과도 정상 응답
        else:
            # 데이터 구조 확인
            if faces_count > 0:
                pass  # 얼굴 데이터가 있는 경우
        
        response_data = create_standard_response(
            data=result_data,
            message=f"카메라 {cam_id} AI 결과 조회 성공"
        )
        return web.json_response(response_data)
    except Exception as e:
        logging.error(f"카메라 결과 조회 중 오류: {e}", exc_info=True)
        response = create_standard_response(
            status="error",
            message=f"카메라 결과 조회 실패: {str(e)}",
            error_code="CAMERA_RESULTS_ERROR"
        )
        return web.json_response(response, status=500)

async def api_model_results_handler(request: web.Request):
    """모델 결과 API 엔드포인트"""
    try:
        # 캐시 미스 - 실제 데이터 조회
        with results_lock:
            # 데이터 필터링 적용
            filtered_data = filter_model_results(model_results)
            response_data = create_standard_response(data=filtered_data, message="모델 결과 조회 성공")
        
        return create_compressed_response(response_data)
    except Exception as e:
        logging.error(f"모델 결과 조회 중 오류: {e}")
        response = create_standard_response(status="error", message=f"모델 결과 조회 실패: {str(e)}", error_code="MODEL_RESULTS_ERROR")
        return web.json_response(response, status=500)

async def api_health_handler(request):
    """시스템 헬스체크 API"""
    # 서버가 시작되었는지 확인 (즉시 응답하여 연결 확인)
    # SafetySystem이 초기화되지 않았으면 503 반환 (서버 아직 준비 중)
    # 이 함수는 최대한 빠르게 응답해야 함 (타임아웃 방지)
    # 항상 응답을 반환해야 함 (연결 실패 방지)
    # 프레임 처리와 완전히 분리되어 항상 즉시 응답
    global safety_system_instance
    
    # 최소한의 응답을 즉시 반환 (연결 확인용)
    # 프레임 처리 상태와 무관하게 항상 빠르게 응답
    try:
        if safety_system_instance is None:
            # 초기화 중이지만 서버는 실행 중이므로 503 반환
            # 503은 정상적인 상태 (초기화 중)이므로 연결은 성공
            return web.json_response(
                {"status": "initializing", "message": "서버 초기화 중입니다."},
                status=503  # Service Unavailable (하지만 연결은 성공)
            )
    except Exception as e:
        # 예외 발생 시에도 응답 반환 (서버 응답성 보장)
        logging.warning(f"Health check 중 오류 (무시): {e}")
        return web.json_response(
            {"status": "error", "message": "Health check 오류"},
            status=503
        )
    
    # SafetySystem이 준비된 경우에만 상세 정보 수집
    try:
        
        # psutil 선택적 import (없어도 작동하도록)
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False
        
        with stats_lock:
            current_time = time.time()
            uptime = current_time - system_stats["start_time"]
            
            # 시스템 리소스 사용률 (psutil이 있으면 사용, 없으면 기본값)
            if psutil_available:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    memory_available = memory.available
                except Exception as e:
                    logging.warning(f"psutil 사용 중 오류: {e}")
                    cpu_percent = 0
                    memory_percent = 0
                    memory_available = 0
            else:
                cpu_percent = 0
                memory_percent = 0
                memory_available = 0
            
            # SafetySystem 상태 확인
            safety_status = "healthy" if safety_system_instance is not None else "unhealthy"
            
            # 카메라 상태 확인
            camera_count = len(latest_frames)
            
            health_data = {
                "status": "healthy",
                "timestamp": current_time,
                "uptime": uptime,
                "system": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_percent,
                    "memory_available": memory_available,
                    "total_requests": system_stats["total_requests"],
                    "error_count": system_stats["error_count"],
                    "error_rate": system_stats["error_count"] / max(system_stats["total_requests"], 1) * 100
                },
                "safety_system": {
                    "status": safety_status,
                    "camera_count": camera_count,
                    "connected_websockets": len(connected_websockets)
                },
                "performance": {
                    "avg_response_time": sum(system_stats["response_times"]) / max(len(system_stats["response_times"]), 1) if system_stats["response_times"] else 0,
                    "throughput": system_stats["total_requests"] / max(uptime, 1)
                }
            }
            
            # 상태 업데이트
            system_stats["last_health_check"] = current_time
            if psutil_available:
                try:
                    system_stats["memory_usage"] = memory_percent
                    system_stats["cpu_usage"] = cpu_percent
                except:
                    pass
            
            response = create_standard_response(data=health_data, message="시스템 헬스체크 성공")
            return web.json_response(response)
    except Exception as e:
        logging.error(f"헬스체크 중 오류: {e}", exc_info=True)
        # 예외 발생 시에도 항상 응답 반환 (연결 확인용)
        # 서버가 시작되었지만 일부 정보를 가져올 수 없음
        try:
            return web.json_response(
                {"status": "error", "message": f"Health check error: {str(e)}"},
                status=503  # 연결은 성공하지만 서비스는 준비되지 않음
            )
        except:
            # 최악의 경우에도 응답 반환
            return web.Response(text="Health check error", status=503)

async def api_violations_handler(request: web.Request):
    """MongoDB 위반 이벤트 조회 API"""
    try:
        from datetime import datetime, timedelta
        
        
        # 캐시 키 생성 (query string 기반)
        cache_key = f"violations_{request.query_string}"
        
        # 캐시 확인
        cached_result = violations_cache.get(cache_key)
        if cached_result is not None:
            return web.json_response(cached_result)
        
        # 입력 검증
        query_params = request.query
        camera_id = query_params.get('camera_id')
        if camera_id:
            try:
                camera_id = validate_camera_id(camera_id)
            except ValidationError as e:
                return web.json_response(e.to_dict(), status=400)
        
        days_param = query_params.get('days', '7')
        # days가 0이거나 매우 크면 모든 데이터 반환 (제한 없음)
        if days_param == '0':
            days = None  # None은 모든 데이터를 의미
        else:
            try:
                days_int = int(days_param)
                # 1년 이상이면 모든 데이터로 처리
                if days_int >= 365:
                    days = None
                elif days_int < 1:
                    return web.json_response({
                        'error': 'days는 1 이상이어야 합니다 (0은 모든 데이터)',
                        'error_code': 'INVALID_DAYS_RANGE'
                    }, status=400)
                else:
                    days = days_int
            except ValueError:
                return web.json_response({
                    'error': 'days는 숫자여야 합니다',
                    'error_code': 'INVALID_DAYS_FORMAT'
            }, status=400)
        
        limit_param = query_params.get('limit', '100')
        # limit이 0이면 모든 데이터를 반환 (제한 없음)
        if limit_param == '0':
            limit = None  # None은 제한 없음을 의미
        else:
            limit = int(limit_param)
            if limit < 1:
                return web.json_response({
                    'error': 'limit는 1 이상이어야 합니다 (0은 모든 데이터)',
                    'error_code': 'INVALID_LIMIT_RANGE'
                }, status=400)
            # limit이 매우 크면 제한 없음으로 처리 (10만 이상)
            if limit >= 100000:
                limit = None
        
        # MongoDB 연결 시도 (선택적)
        try:
            from database import get_database  # type: ignore
            db = None
            try:
                db = get_database()
            except Exception as db_init_error:
                logging.warning(f"[API] MongoDB 초기화 실패: {db_init_error}", exc_info=True)
                db = None
            
            if db and db.is_connected():
                # 쿼리 파라미터 파싱
                worker_name = query_params.get('worker_name')
                event_type = query_params.get('event_type')
                
                # days가 None이면 모든 데이터 (시간 제한 없음)
                if days is None:
                    start_time = None
                    end_time = None
                elif days == 1:
                    # days=1일 때는 오늘 0시부터 현재까지 (금일 데이터)
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    start_time = today
                    end_time = datetime.now()
                else:
                    # days > 1일 때는 지난 N일간의 데이터
                    start_time = datetime.now() - timedelta(days=days)
                    end_time = datetime.now()
                
                # MongoDB에서 조회 (가이드 스키마 호환)
                try:
                    violations = db.get_violations(
                        worker_name=worker_name,
                        camera_id=camera_id,
                        event_type=event_type,
                        start_time=start_time,
                        end_time=end_time,
                        limit=limit
                    )
                except Exception as query_error:
                    logging.error(f"[API] MongoDB 위반 사항 조회 쿼리 실패: {query_error}", exc_info=True)
                    violations = []
                
                # 실제 DB 데이터를 그대로 사용하되, 필수 필드만 보장
                try:
                    for v in violations:
                        # _id 변환
                        if '_id' in v:
                            v['_id'] = str(v['_id'])
                        
                        
                        # timestamp 처리 (실제 DB에 있는 형식 그대로 사용)
                        if 'timestamp' in v:
                            if isinstance(v['timestamp'], datetime):
                                v['timestamp'] = int(v['timestamp'].timestamp() * 1000)
                            elif isinstance(v['timestamp'], str):
                                try:
                                    dt = datetime.fromisoformat(v['timestamp'].replace('T', ' ').split('.')[0])
                                    v['timestamp'] = int(dt.timestamp() * 1000)
                                except:
                                    pass
                        
                        # violation_datetime 생성 (프론트엔드에서 사용하므로 API 응답에서만 생성)
                        # DB에는 저장하지 않고 API 응답 시에만 추가
                        if 'violation_datetime' not in v or not v.get('violation_datetime'):
                            if 'timestamp' in v:
                                try:
                                    timestamp = v['timestamp']
                                    if isinstance(timestamp, (int, float)):
                                        if timestamp > 1e12:  # 밀리초
                                            dt = datetime.fromtimestamp(timestamp / 1000)
                                        else:  # 초 단위
                                            dt = datetime.fromtimestamp(timestamp)
                                        v['violation_datetime'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                                except:
                                    v['violation_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                v['violation_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                if 'timestamp' not in v:
                                    v['timestamp'] = int(datetime.now().timestamp() * 1000)
                        
                        # 하위 호환 필드 제거 (프론트엔드가 표준 필드를 우선 사용하므로 불필요)
                        # 프론트엔드는 이미 표준 필드(worker_name, type, cam_id, timestamp)를 우선 사용
                        
                        # worker_name <-> worker_id (실제 DB에 있는 값 우선 사용)
                        if 'worker_name' not in v or not v.get('worker_name'):
                            v['worker_name'] = v.get('worker_id', 'Unknown')
                        if 'worker_id' not in v or not v.get('worker_id'):
                            v['worker_id'] = v.get('worker_name', 'Unknown')
                        
                        # work_zone (실제 DB에 있으면 그대로 사용, 없으면 cam_id 기반 생성)
                        if 'work_zone' not in v or not v.get('work_zone'):
                            cam_id = v.get('cam_id', 0)
                            area_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                            v['work_zone'] = area_map.get(cam_id, f"A-{cam_id+1}")
                        
                        # severity (실제 DB에 있으면 그대로 사용, 없으면 기본값)
                        if 'severity' not in v:
                            violation_type = v.get('type', '')
                            if "안전모" in violation_type or "helmet" in violation_type.lower():
                                v['severity'] = "high"
                            elif "안전조끼" in violation_type or "vest" in violation_type.lower():
                                v['severity'] = "medium"
                            elif "넘어짐" in violation_type or "fall" in violation_type.lower():
                                v['severity'] = "critical"
                            else:
                                v['severity'] = "medium"
                        
                        # status (실제 DB에 있으면 그대로 사용, 없으면 기본값)
                        if 'status' not in v:
                            v['status'] = 'new'
                        
                except Exception as format_error:
                    logging.warning(f"[API] 위반 사항 데이터 포맷팅 실패: {format_error}", exc_info=True)
                
                if violations:
                    pass  # 위반 사항이 있는 경우
                
                # 프론트엔드 호환 형식으로 변환
                response_data = {
                    "success": True,
                    "violations": violations,
                    "count": len(violations),
                    "message": "위반 이벤트 조회 성공",
                    "mongodb_connected": True  # MongoDB 연결 상태
                }
                
                # 결과 캐싱
                violations_cache.put(cache_key, response_data)
                return web.json_response(response_data)
        except ImportError as import_err:
            logging.warning(f"[API] MongoDB 모듈 import 실패: {import_err}")
            pass  # database 모듈이 없으면 아래 코드 실행
        except Exception as db_error:
            logging.error(f"[API] MongoDB 위반 사항 조회 실패: {db_error}", exc_info=True)
        
        # MongoDB가 없으면 빈 결과 반환 (프론트엔드 호환 형식)
        response_data = {
            "success": True,
            "violations": [],
            "count": 0,
            "message": "위반 이벤트 조회 성공 (MongoDB 미연결)",
            "mongodb_connected": False  # MongoDB 연결 상태
        }
        
        # 결과 캐싱 (MongoDB 미연결 상태도 캐싱)
        violations_cache.put(cache_key, response_data)
        return web.json_response(response_data)
    
    except Exception as e:
        logging.error(f"[API] 위반 이벤트 조회 중 예외 발생: {e}", exc_info=True)
        # MongoDB 없이도 정상 작동하도록 빈 결과 반환
        response_data = {
            "success": True,
            "violations": [],
            "count": 0,
            "message": "위반 이벤트 조회 성공 (MongoDB 미연결)"
        }
        
        # 에러 발생 시에도 캐싱 (짧은 TTL로 재시도 방지)
        cache_key = f"violations_{request.query_string}"
        violations_cache.put(cache_key, response_data, ttl=2.0)  # 2초만 캐싱
        return web.json_response(response_data)

async def api_update_violation_status_handler(request: web.Request):
    """위반 사항 상태 업데이트 API"""
    try:
        from database import get_database  # type: ignore
        
        # 요청 본문 파싱
        try:
            data = await request.json()
        except Exception as e:
            logging.warning(f"[API] 위반 사항 상태 업데이트 - JSON 파싱 실패: {e}")
            return web.json_response({
                "success": False,
                "error": "잘못된 요청 형식",
                "error_code": "INVALID_REQUEST"
            }, status=400)
        
        worker_id = data.get('worker_id')
        violation_datetime = data.get('violation_datetime')
        status = data.get('status', 'done')
        
        if not worker_id or not violation_datetime:
            return web.json_response({
                "success": False,
                "error": "worker_id와 violation_datetime은 필수입니다",
                "error_code": "MISSING_REQUIRED_FIELDS"
            }, status=400)
        
        # MongoDB 연결 확인
        try:
            db = get_database()
            if not db or not db.is_connected():
                return web.json_response({
                    "success": False,
                    "error": "MongoDB 연결 실패",
                    "error_code": "MONGODB_NOT_CONNECTED"
                }, status=500)
        except Exception as db_error:
            logging.error(f"[API] 위반 사항 상태 업데이트 - MongoDB 연결 실패: {db_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": "MongoDB 연결 실패",
                "error_code": "MONGODB_NOT_CONNECTED"
            }, status=500)
        
        # 위반 사항 상태 업데이트
        try:
            violations_collection = db.db_service.get_violations_collection()
            if violations_collection is None:
                return web.json_response({
                    "success": False,
                    "error": "violation 컬렉션을 가져올 수 없습니다",
                    "error_code": "COLLECTION_NOT_FOUND"
                }, status=500)
            
            # 처리 시간 계산 (status가 'done'일 때만)
            processing_time_seconds = None
            if status == 'done':
                try:
                    # 위반 사항의 violation_datetime 가져오기
                    violation = violations_collection.find_one({
                        'worker_id': worker_id,
                        'violation_datetime': violation_datetime
                    })
                    
                    if violation and violation.get('violation_datetime'):
                        # violation_datetime 파싱
                        from datetime import datetime
                        violation_dt = datetime.strptime(violation_datetime, '%Y-%m-%d %H:%M:%S')
                        current_dt = datetime.now()
                        
                        # 처리 시간 계산 (초 단위)
                        time_diff = current_dt - violation_dt
                        processing_time_seconds = int(time_diff.total_seconds())
                        
                except Exception as time_calc_error:
                    logging.warning(f"[API] 처리 시간 계산 실패: {time_calc_error}")
                    processing_time_seconds = None
            
            # 업데이트 쿼리
            update_fields = {
                'status': status
            }
            
            # status가 'done'이고 처리 시간이 계산되었으면 추가
            if status == 'done' and processing_time_seconds is not None:
                update_fields['processing_time'] = processing_time_seconds
            
            # 위반 사항 검색 쿼리 (violation_datetime이 None인 경우도 처리)
            # 1차: worker_id와 violation_datetime으로 검색
            query = {
                    'worker_id': worker_id,
                    'violation_datetime': violation_datetime
            }
            
            update_result = violations_collection.update_one(
                query,
                {
                    '$set': update_fields
                }
            )
            
            # 찾지 못한 경우, violation_datetime이 None인 경우를 고려하여 timestamp로 검색
            if update_result.matched_count == 0:
                try:
                    # violation_datetime을 timestamp로 변환
                    from datetime import datetime
                    violation_dt = datetime.strptime(violation_datetime, '%Y-%m-%d %H:%M:%S')
                    timestamp_ms = int(violation_dt.timestamp() * 1000)
                    
                    # timestamp로 검색 (약간의 오차 허용: ±5초)
                    timestamp_tolerance = 5000  # 5초
                    query = {
                        'worker_id': worker_id,
                        'timestamp': {
                            '$gte': timestamp_ms - timestamp_tolerance,
                            '$lte': timestamp_ms + timestamp_tolerance
                        }
                    }
                    
                    # violation_datetime이 None이거나 없는 문서도 찾기 위해 추가 조건
                    query_with_null = {
                        'worker_id': worker_id,
                        '$or': [
                            {'violation_datetime': None},
                            {'violation_datetime': {'$exists': False}},
                            {'timestamp': {
                                '$gte': timestamp_ms - timestamp_tolerance,
                                '$lte': timestamp_ms + timestamp_tolerance
                            }}
                        ]
                    }
                    
                    # 먼저 timestamp로 정확히 매칭되는 문서 찾기
                    update_result = violations_collection.update_one(
                        query,
                        {
                            '$set': {
                                **update_fields,
                                'violation_datetime': violation_datetime  # violation_datetime도 업데이트
                            }
                        }
                    )
                    
                    # 여전히 찾지 못한 경우, violation_datetime이 None인 문서 찾기
                    if update_result.matched_count == 0:
                        update_result = violations_collection.update_one(
                            query_with_null,
                            {
                                '$set': {
                                    **update_fields,
                                    'violation_datetime': violation_datetime  # violation_datetime도 업데이트
                                }
                            }
                        )
                    
                    if update_result.matched_count > 0:
                        # timestamp로 찾아서 업데이트 성공
                        pass
                except Exception as timestamp_error:
                    logging.warning(f"[API] timestamp 변환 실패: {timestamp_error}")
            
            if update_result.matched_count == 0:
                logging.warning(f"[API] 위반 사항을 찾을 수 없음: worker_id={worker_id}, violation_datetime={violation_datetime}")
                # 디버깅: 해당 worker_id의 최근 위반 사항 조회
                try:
                    recent_violations = list(violations_collection.find(
                        {'worker_id': worker_id}
                    ).sort('timestamp', -1).limit(5))
                except:
                    pass
                return web.json_response({
                    "success": False,
                    "error": "위반 사항을 찾을 수 없습니다",
                    "error_code": "VIOLATION_NOT_FOUND"
                }, status=404)
            
            if update_result.modified_count == 0:
                # 수정된 내용이 없어도 성공으로 처리 (이미 같은 상태일 수 있음)
                pass
            
            return web.json_response({
                "success": True,
                "message": "위반 사항 상태가 업데이트되었습니다",
                "matched_count": update_result.matched_count,
                "modified_count": update_result.modified_count
            })
            
        except Exception as update_error:
            logging.error(f"[API] 위반 사항 상태 업데이트 실패: {update_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": f"상태 업데이트 실패: {str(update_error)}",
                "error_code": "UPDATE_FAILED"
            }, status=500)
    
    except Exception as e:
        logging.error(f"[API] 위반 사항 상태 업데이트 중 예외 발생: {e}", exc_info=True)
        return web.json_response({
            "success": False,
            "error": str(e),
            "error_code": "INTERNAL_ERROR"
        }, status=500)

async def api_fps_handler(request: web.Request):
    """실시간 FPS 추적 API 엔드포인트"""
    try:
        current_time = time.time()
        fps_data = {}
        
        # GPU 사용량 정보 추가
        gpu_info = {}
        if safety_system_instance and 'cuda' in str(safety_system_instance.device):
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info = {
                        "device_name": torch.cuda.get_device_name(0),
                        "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2),
                        "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2),
                        "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 2),
                        "memory_usage_percent": round((torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100, 2),
                    }
            except Exception as e:
                gpu_info = {"error": str(e)}
        
        with frame_stats_lock:
            for cam_id, stats in frame_stats.items():
                # 최근 2초간의 FPS 계산
                frame_times = stats['frame_times']
                recent_fps = 0.0
                
                if len(frame_times) >= 2:
                    recent_frames = [t for t in frame_times if current_time - t <= 2.0]
                    if len(recent_frames) >= 2:
                        time_span = recent_frames[-1] - recent_frames[0]
                        if time_span > 0:
                            recent_fps = (len(recent_frames) - 1) / time_span
                
                # 전체 평균 FPS
                time_since_first = current_time - stats['first_frame_time']
                average_fps = stats['frame_count'] / time_since_first if time_since_first > 0 else 0.0
                
                fps_data[cam_id] = {
                    "camera_id": cam_id,
                    "recent_fps": round(recent_fps, 2),  # 최근 2초간 FPS
                    "average_fps": round(average_fps, 2),  # 전체 평균 FPS
                    "total_frames": stats['frame_count'],
                    "last_frame_time": stats['last_frame_time'],
                    "time_since_last_frame": round(current_time - stats['last_frame_time'], 2),
                    "uptime_seconds": round(time_since_first, 2),
                    "is_active": recent_fps > 0
                }
        
        # 전체 통계
        total_frames = sum(s['frame_count'] for s in frame_stats.values())
        total_recent_fps = sum(data['recent_fps'] for data in fps_data.values())
        
        response_data = create_standard_response(
            data={
                "cameras": fps_data,
                "gpu_info": gpu_info,
                "summary": {
                    "total_cameras": len(fps_data),
                    "active_cameras": len([d for d in fps_data.values() if d['is_active']]),
                    "total_frames_processed": total_frames,
                    "total_recent_fps": round(total_recent_fps, 2),
                    "timestamp": current_time
                }
            },
            message="실시간 FPS 추적 정보 조회 성공"
        )
        return web.json_response(response_data)
    except Exception as e:
        logging.error(f"FPS 추적 정보 조회 중 오류: {e}", exc_info=True)
        response = create_standard_response(
            status="error",
            message=f"FPS 추적 정보 조회 실패: {str(e)}",
            error_code="FPS_ERROR"
        )
        return web.json_response(response, status=500)

async def api_violation_image_handler(request: web.Request):
    """위반 이미지 제공 API"""
    try:
        import os
        from pathlib import Path
        
        # 쿼리 파라미터에서 이미지 경로 가져오기
        image_path = request.query.get('path', '')
        if not image_path:
            return web.json_response({
                'error': '이미지 경로가 필요합니다',
                'error_code': 'MISSING_IMAGE_PATH'
            }, status=400)
        
        # 경로 디코딩
        import urllib.parse
        image_path = urllib.parse.unquote(image_path)
        
        # 절대 경로로 변환 (상대 경로인 경우)
        if not os.path.isabs(image_path):
            # logs 폴더 기준으로 상대 경로 처리
            from config import Paths
            log_folder = Paths.LOG_FOLDER
            # 파일명만 추출하여 logs 폴더와 결합
            image_path = os.path.join(log_folder, os.path.basename(image_path))
        else:
            # 절대 경로인 경우 그대로 사용
            # 경로 정규화 (슬래시/백슬래시 통일)
            image_path = os.path.normpath(image_path)
        
        # 파일 존재 확인
        if not os.path.exists(image_path):
            logging.warning(f"[이미지 API] 파일이 존재하지 않음: {image_path}")
            logging.warning(f"[이미지 API] 원본 경로: {request.query.get('path', '')}")
            # logs 폴더의 파일 목록 확인 (디버깅)
            try:
                from config import Paths
                log_folder = Paths.LOG_FOLDER
                if os.path.exists(log_folder):
                    files = os.listdir(log_folder)
                    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
                    logging.info(f"[이미지 API] logs 폴더의 JPG 파일 수: {len(jpg_files)}")
                    if jpg_files:
                        logging.info(f"[이미지 API] 샘플 파일: {jpg_files[:3]}")
            except Exception as debug_err:
                logging.warning(f"[이미지 API] 디버깅 정보 조회 실패: {debug_err}")
            
            return web.json_response({
                'error': '이미지 파일을 찾을 수 없습니다',
                'error_code': 'IMAGE_NOT_FOUND',
                'requested_path': image_path
            }, status=404)
        
        # 파일 읽기
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # MIME 타입 결정
        mime_type = 'image/jpeg'
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.gif'):
            mime_type = 'image/gif'
        
        return web.Response(
            body=image_data,
            content_type=mime_type,
            headers={
                'Cache-Control': 'public, max-age=3600',
            }
        )
    except Exception as e:
        logging.error(f"[이미지 API] 오류: {e}", exc_info=True)
        return web.json_response({
            'error': str(e),
            'error_code': 'IMAGE_LOAD_ERROR'
        }, status=500)


async def api_frontend_logs_handler(request: web.Request):
    """프론트엔드 로그를 받아서 서버 로그 파일에 저장"""
    try:
        data = await request.json()
        logs = data.get('logs', [])
        user_agent = data.get('userAgent', 'Unknown')
        url = data.get('url', 'Unknown')
        
        # 각 로그를 서버 로그 파일에 저장
        for log_entry in logs:
            level = log_entry.get('level', 'LOG')
            message = log_entry.get('message', '')
            timestamp = log_entry.get('timestamp', '')
            
            # 로그 레벨에 따라 적절한 로깅 메서드 사용
            log_message = f"[Frontend] [{level}] {message} (URL: {url}, UserAgent: {user_agent[:100]})"
            
            if level == 'ERROR':
                logging.error(log_message)
            elif level == 'WARN':
                logging.warning(log_message)
            elif level == 'INFO':
                logging.info(log_message)
            elif level == 'DEBUG':
                logging.debug(log_message)
            else:
                logging.info(log_message)
        
        return web.json_response({
            'status': 'success',
            'message': f'{len(logs)}개의 로그가 저장되었습니다.',
            'count': len(logs)
        })
    except Exception as e:
        logging.error(f"프론트엔드 로그 저장 실패: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'message': f'로그 저장 실패: {str(e)}'
        }, status=500)

async def api_performance_handler(request: web.Request):
    """파이프라인별 성능 통계 API 엔드포인트"""
    try:
        current_time = time.time()
        performance_data = {}
        
        # latest_result_data에서 performance 정보 추출
        with frame_lock:
            for cam_id, result_data in latest_result_data.items():
                if 'performance' in result_data:
                    perf = result_data['performance']
                    
                    # 병목 지점 식별
                    bottleneck = None
                    bottleneck_time = 0.0
                    if perf:
                        # total을 제외한 최대 시간을 가진 단계 찾기
                        perf_items = [(k, v) for k, v in perf.items() if k != 'total' and v > 0]
                        if perf_items:
                            bottleneck_tuple = max(perf_items, key=lambda x: x[1])
                            bottleneck = bottleneck_tuple[0]
                            bottleneck_time = bottleneck_tuple[1]
                    
                    performance_data[cam_id] = {
                        "camera_id": cam_id,
                        "timings": {
                            "total_ms": round(perf.get('total', 0), 2),
                            "decode_ms": round(perf.get('decode', 0), 2),
                            "resize_ms": round(perf.get('resize', 0), 2),
                            "yolo_violation_ms": round(perf.get('yolo_violation', 0), 2),
                            "yolo_pose_ms": round(perf.get('yolo_pose', 0), 2),
                            "parse_results_ms": round(perf.get('parse_results', 0), 2),
                            "face_recognition_ms": round(perf.get('face_recognition', 0), 2),
                            "rendering_ms": round(perf.get('rendering', 0), 2),
                            "encoding_ms": round(perf.get('encoding', 0), 2)
                        },
                        "bottleneck": {
                            "stage": bottleneck,
                            "time_ms": round(bottleneck_time, 2),
                            "percentage": round((bottleneck_time / perf.get('total', 1)) * 100, 1) if perf.get('total', 0) > 0 else 0.0
                        } if bottleneck else None,
                        "last_update": result_data.get('timestamp', current_time)
                    }
        
        # 전체 통계 계산
        if performance_data:
            all_timings = {
                "total_ms": [],
                "decode_ms": [],
                "resize_ms": [],
                "yolo_violation_ms": [],
                "yolo_pose_ms": [],
                "parse_results_ms": [],
                "face_recognition_ms": [],
                "rendering_ms": [],
                "encoding_ms": []
            }
            
            for cam_data in performance_data.values():
                timings = cam_data['timings']
                for key in all_timings.keys():
                    if timings[key] > 0:
                        all_timings[key].append(timings[key])
            
            summary = {}
            for key, values in all_timings.items():
                if values:
                    summary[key] = {
                        "avg": round(sum(values) / len(values), 2),
                        "min": round(min(values), 2),
                        "max": round(max(values), 2)
                    }
        else:
            summary = {}
        
        response_data = create_standard_response(
            data={
                "cameras": performance_data,
                "summary": summary,
                "timestamp": current_time
            },
            message="파이프라인별 성능 통계 조회 성공"
        )
        return web.json_response(response_data)
    except Exception as e:
        logging.error(f"성능 통계 조회 중 오류: {e}", exc_info=True)
        response = create_standard_response(
            status="error",
            message=f"성능 통계 조회 실패: {str(e)}",
            error_code="PERFORMANCE_ERROR"
        )
        return web.json_response(response, status=500)

async def api_workers_handler(request: web.Request):
    """MongoDB 작업자 조회 API (GET)"""
    try:
        from datetime import datetime
        
        
        # 캐시 키 생성 (query string 기반)
        cache_key = f"workers_{request.query_string}"
        
        # 캐시 확인
        cached_result = workers_cache.get(cache_key)
        if cached_result is not None:
            return web.json_response(cached_result)
        
        # MongoDB 연결 시도 (선택적)
        try:
            from database import get_database  # type: ignore
            db = None
            try:
                db = get_database()
            except Exception as db_init_error:
                logging.warning(f"[API] MongoDB 초기화 실패: {db_init_error}", exc_info=True)
                db = None
            
            if db and db.is_connected():
                # 쿼리 파라미터
                active_only = request.query.get('active_only', 'true').lower() == 'true'
                
                # 디버깅: 실제 DB에서 직접 조회하여 비교
                try:
                    workers_collection = db.db_service.get_workers_collection()
                    if workers_collection is not None:
                        raw_count = workers_collection.count_documents({})
                        logging.info(f"[API] MongoDB worker 컬렉션 총 레코드 수: {raw_count}개")
                        
                except Exception as debug_error:
                    logging.warning(f"[API] 디버깅 정보 조회 실패: {debug_error}")
                
                try:
                    workers = db.get_all_workers(active_only=active_only)
                except Exception as query_error:
                    logging.error(f"[API] MongoDB 작업자 조회 쿼리 실패: {query_error}", exc_info=True)
                    workers = []
                    # 쿼리 실패 시에도 빈 배열 반환 (프론트엔드에서 에러로 인식하지 않도록)
                
                # WorkerService가 이미 포맷팅된 데이터를 반환하므로 추가 변환 불필요
                # 단, datetime 필드가 있으면 ISO 형식으로 변환
                # 필드명 정규화: worker_id -> workerId, name -> workerName (프론트엔드 호환)
                try:
                    for w in workers:
                        if '_id' in w:
                            w['_id'] = str(w['_id'])
                        
                        # 필드명 정규화: worker_id -> workerId (프론트엔드 호환)
                        # DB에 worker_id만 있는 경우를 대비하여 명확하게 매핑
                        if 'worker_id' in w and w['worker_id']:
                            worker_id_value = str(w['worker_id']).strip()
                            # workerId가 없거나 unknown_으로 시작하는 경우 worker_id로 덮어쓰기
                            if not w.get('workerId') or (w.get('workerId') and str(w.get('workerId', '')).startswith('unknown_')):
                                w['workerId'] = worker_id_value
                        elif 'worker_id' in w and 'workerId' not in w:
                            # worker_id가 있지만 workerId가 없는 경우
                            w['workerId'] = w['worker_id']
                        
                        # name 필드 정규화
                        if 'name' in w and 'workerName' not in w:
                            w['workerName'] = w['name']
                        
                        for date_field in ['registered_at', 'last_seen', 'datetime']:
                            if date_field in w and isinstance(w[date_field], datetime):
                                w[date_field] = w[date_field].isoformat()
                except Exception as format_error:
                    logging.warning(f"[API] 작업자 데이터 포맷팅 실패: {format_error}")
                
                
                # 프론트엔드 호환 형식으로 변환
                response_data = {
                    "success": True,
                    "workers": workers,
                    "count": len(workers),
                    "message": "작업자 조회 성공",
                    "mongodb_connected": True  # MongoDB 연결 상태
                }
                
                # 결과 캐싱
                workers_cache.put(cache_key, response_data)
                return web.json_response(response_data)
        except ImportError as import_err:
            logging.warning(f"[API] database 모듈 import 실패: {import_err}")
            pass  # database 모듈이 없으면 아래 코드 실행
        except Exception as db_error:
            logging.error(f"[API] MongoDB 조회 실패: {db_error}", exc_info=True)
        
        # MongoDB가 없으면 빈 결과 반환 (프론트엔드 호환 형식)
        response_data = {
            "success": True,
            "workers": [],
            "count": 0,
            "message": "작업자 조회 성공 (MongoDB 미연결)",
            "mongodb_connected": False  # MongoDB 연결 상태
        }
        
        # 결과 캐싱 (MongoDB 미연결 상태도 캐싱)
        workers_cache.put(cache_key, response_data)
        return web.json_response(response_data)
    
    except Exception as e:
        logging.error(f"[API] 작업자 조회 중 예외 발생: {e}", exc_info=True)
        # MongoDB 없이도 정상 작동하도록 빈 결과 반환 (프론트엔드에서 에러로 인식하지 않도록)
        # 에러가 발생해도 success=True로 반환하여 프론트엔드에서 정상 처리되도록 함
        response_data = {
            "success": True,
            "workers": [],
            "count": 0,
            "message": "작업자 조회 성공 (MongoDB 미연결 또는 오류)",
            "mongodb_connected": False,
            "error": str(e)  # 디버깅을 위한 에러 메시지 (선택적)
        }
        
        # 에러 발생 시에도 캐싱 (짧은 TTL로 재시도 방지)
        cache_key = f"workers_{request.query_string}"
        workers_cache.put(cache_key, response_data, ttl=5.0)  # 5초만 캐싱
        return web.json_response(response_data)

async def api_workers_post_handler(request: web.Request):
    """작업자 생성 API (POST)"""
    try:
        from database import get_database  # type: ignore
        
        # 요청 본문 파싱
        try:
            data = await request.json()
        except Exception as e:
            logging.warning(f"[API] 작업자 생성 - JSON 파싱 실패: {e}")
            return web.json_response({
                "success": False,
                "error": "잘못된 요청 형식",
                "error_code": "INVALID_REQUEST"
            }, status=400)
        
        # MongoDB 연결 확인
        try:
            db = get_database()
            if not db or not db.is_connected():
                return web.json_response({
                    "success": False,
                    "error": "MongoDB 연결 실패",
                    "error_code": "MONGODB_NOT_CONNECTED"
                }, status=500)
        except Exception as db_error:
            logging.error(f"[API] 작업자 생성 - MongoDB 연결 실패: {db_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": "MongoDB 연결 실패",
                "error_code": "MONGODB_NOT_CONNECTED"
            }, status=500)
        
        # 작업자 생성
        try:
            workers_collection = db.db_service.get_workers_collection()
            if workers_collection is None:
                return web.json_response({
                    "success": False,
                    "error": "worker 컬렉션을 가져올 수 없습니다",
                    "error_code": "COLLECTION_NOT_FOUND"
                }, status=500)
            
            # 중복 확인 (workerId 또는 worker_id로 검색)
            worker_id = data.get('workerId') or data.get('worker_id')
            if worker_id:
                existing = workers_collection.find_one({
                    '$or': [
                        {'workerId': worker_id},
                        {'worker_id': worker_id}
                    ]
                })
                if existing:
                    return web.json_response({
                        "success": False,
                        "error": f"이미 존재하는 작업자 ID입니다: {worker_id}",
                        "error_code": "DUPLICATE_WORKER_ID"
                    }, status=400)
            
            # 작업자 데이터 삽입
            result = workers_collection.insert_one(data)
            
            
            return web.json_response({
                "success": True,
                "message": "작업자가 생성되었습니다",
                "data": {
                    "id": str(result.inserted_id),
                    "workerId": worker_id
                }
            })
            
        except Exception as create_error:
            logging.error(f"[API] 작업자 생성 실패: {create_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": f"작업자 생성 실패: {str(create_error)}",
                "error_code": "CREATE_FAILED"
            }, status=500)
    
    except Exception as e:
        logging.error(f"[API] 작업자 생성 중 예외 발생: {e}", exc_info=True)
        return web.json_response({
            "success": False,
            "error": str(e),
            "error_code": "INTERNAL_ERROR"
        }, status=500)

async def api_workers_put_handler(request: web.Request):
    """작업자 업데이트 API (PUT)"""
    try:
        from database import get_database  # type: ignore
        
        # URL에서 workerId 추출
        worker_id = request.match_info.get('workerId') or request.match_info.get('worker_id')
        if not worker_id:
            return web.json_response({
                "success": False,
                "error": "workerId가 필요합니다",
                "error_code": "MISSING_WORKER_ID"
            }, status=400)
        
        # 요청 본문 파싱
        try:
            data = await request.json()
        except Exception as e:
            logging.warning(f"[API] 작업자 업데이트 - JSON 파싱 실패: {e}")
            return web.json_response({
                "success": False,
                "error": "잘못된 요청 형식",
                "error_code": "INVALID_REQUEST"
            }, status=400)
        
        # MongoDB 연결 확인
        try:
            db = get_database()
            if not db or not db.is_connected():
                return web.json_response({
                    "success": False,
                    "error": "MongoDB 연결 실패",
                    "error_code": "MONGODB_NOT_CONNECTED"
                }, status=500)
        except Exception as db_error:
            logging.error(f"[API] 작업자 업데이트 - MongoDB 연결 실패: {db_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": "MongoDB 연결 실패",
                "error_code": "MONGODB_NOT_CONNECTED"
            }, status=500)
        
        # 작업자 업데이트
        try:
            workers_collection = db.db_service.get_workers_collection()
            if workers_collection is None:
                return web.json_response({
                    "success": False,
                    "error": "worker 컬렉션을 가져올 수 없습니다",
                    "error_code": "COLLECTION_NOT_FOUND"
                }, status=500)
            
            # _id 필드 제거 (업데이트 시 사용하지 않음)
            data.pop('_id', None)
            
            # 필드명 정규화: worker_id -> workerId (DB에 worker_id만 있는 경우 대비)
            if 'worker_id' in data and data['worker_id']:
                data['workerId'] = data['worker_id']
            # name -> workerName 정규화
            if 'name' in data and 'workerName' not in data:
                data['workerName'] = data['name']
            
            # 업데이트 쿼리 (workerId 또는 worker_id로 검색)
            # DB에 worker_id만 있는 경우를 대비하여 둘 다 검색
            update_result = workers_collection.update_one(
                {'$or': [
                    {'workerId': worker_id},
                    {'worker_id': worker_id}
                ]},
                {'$set': data}
            )
            
            if update_result.matched_count == 0:
                # 디버깅: 실제 DB에 어떤 필드가 있는지 확인
                sample = workers_collection.find_one({})
                if sample:
                    logging.warning(f"[API] 작업자 검색 실패 - 샘플 레코드: keys={list(sample.keys())}, workerId={sample.get('workerId')}, worker_id={sample.get('worker_id')}")
                return web.json_response({
                    "success": False,
                    "error": f"작업자를 찾을 수 없습니다: {worker_id}",
                    "error_code": "WORKER_NOT_FOUND"
                }, status=404)
            
            
            return web.json_response({
                "success": True,
                "message": "작업자가 업데이트되었습니다",
                "matched_count": update_result.matched_count,
                "modified_count": update_result.modified_count
            })
            
        except Exception as update_error:
            logging.error(f"[API] 작업자 업데이트 실패: {update_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": f"작업자 업데이트 실패: {str(update_error)}",
                "error_code": "UPDATE_FAILED"
            }, status=500)
    
    except Exception as e:
        logging.error(f"[API] 작업자 업데이트 중 예외 발생: {e}", exc_info=True)
        return web.json_response({
            "success": False,
            "error": str(e),
            "error_code": "INTERNAL_ERROR"
        }, status=500)

async def api_workers_delete_handler(request: web.Request):
    """작업자 삭제 API (DELETE)"""
    try:
        from database import get_database  # type: ignore
        
        # URL에서 workerId 추출
        worker_id = request.match_info.get('workerId') or request.match_info.get('worker_id')
        if not worker_id:
            return web.json_response({
                "success": False,
                "error": "workerId가 필요합니다",
                "error_code": "MISSING_WORKER_ID"
            }, status=400)
        
        # MongoDB 연결 확인
        try:
            db = get_database()
            if not db or not db.is_connected():
                return web.json_response({
                    "success": False,
                    "error": "MongoDB 연결 실패",
                    "error_code": "MONGODB_NOT_CONNECTED"
                }, status=500)
        except Exception as db_error:
            logging.error(f"[API] 작업자 삭제 - MongoDB 연결 실패: {db_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": "MongoDB 연결 실패",
                "error_code": "MONGODB_NOT_CONNECTED"
            }, status=500)
        
        # 작업자 삭제
        try:
            workers_collection = db.db_service.get_workers_collection()
            if workers_collection is None:
                return web.json_response({
                    "success": False,
                    "error": "worker 컬렉션을 가져올 수 없습니다",
                    "error_code": "COLLECTION_NOT_FOUND"
                }, status=500)
            
            # 삭제 쿼리 (workerId 또는 worker_id로 검색)
            delete_result = workers_collection.delete_one({
                '$or': [
                    {'workerId': worker_id},
                    {'worker_id': worker_id}
                ]
            })
            
            if delete_result.deleted_count == 0:
                return web.json_response({
                    "success": False,
                    "error": f"작업자를 찾을 수 없습니다: {worker_id}",
                    "error_code": "WORKER_NOT_FOUND"
                }, status=404)
            
            
            return web.json_response({
                "success": True,
                "message": "작업자가 삭제되었습니다",
                "deleted_count": delete_result.deleted_count
            })
            
        except Exception as delete_error:
            logging.error(f"[API] 작업자 삭제 실패: {delete_error}", exc_info=True)
            return web.json_response({
                "success": False,
                "error": f"작업자 삭제 실패: {str(delete_error)}",
                "error_code": "DELETE_FAILED"
            }, status=500)
    
    except Exception as e:
        logging.error(f"[API] 작업자 삭제 중 예외 발생: {e}", exc_info=True)
        return web.json_response({
            "success": False,
            "error": str(e),
            "error_code": "INTERNAL_ERROR"
        }, status=500)

async def api_gpu_handler(request: web.Request):
    """GPU 사용량 모니터링 API"""
    try:
        gpu_stats = get_gpu_usage_stats()
        
        if not gpu_stats:
            # GPU가 없거나 사용 불가능한 경우
            return web.json_response({
                "success": True,
                "gpu_available": False,
                "message": "GPU를 사용할 수 없습니다 (CUDA 미지원 또는 GPU 없음)",
                "gpus": []
            })
        
        # GPU 정보를 리스트로 변환
        gpus = []
        for gpu_id, stats in gpu_stats.items():
            gpus.append({
                "id": gpu_id,
                "name": stats["name"],
                "memory_allocated_gb": round(stats["memory_allocated_gb"], 2),
                "memory_reserved_gb": round(stats["memory_reserved_gb"], 2),
                "memory_total_gb": round(stats["memory_total_gb"], 2),
                "memory_free_gb": round(stats["memory_free_gb"], 2),
                "memory_used_gb": round(stats["memory_reserved_gb"], 2),  # 사용 중인 메모리 (reserved)
                "memory_util_percent": round(stats["memory_util_percent"], 1)
            })
        
        return web.json_response({
            "success": True,
            "gpu_available": True,
            "gpu_count": len(gpus),
            "gpus": gpus,
            "message": f"{len(gpus)}개의 GPU가 감지되었습니다"
        })
    
    except Exception as e:
        logging.error(f"GPU 사용량 조회 중 오류: {e}", exc_info=True)
        return web.json_response({
            "success": False,
            "error": str(e),
            "gpu_available": False,
            "gpus": []
        }, status=500)

async def api_statistics_handler(request: web.Request):
    """MongoDB 통계 조회 API"""
    global model_results
    try:
        # 캐시 키 생성 (query string 기반)
        cache_key = f"stats_{request.query_string}"
        
        # 캐시 확인 (MongoDB 미연결 상태는 캐시 무시)
        cached_result = stats_cache.get(cache_key)
        if cached_result is not None:
            # MongoDB가 연결되어 있는 경우에만 캐시 사용
            # (MongoDB 미연결 상태가 캐시에 저장되어 있을 수 있으므로)
            if cached_result.get('mongodb_connected', False):
                return web.json_response(cached_result)
            else:
                # MongoDB 미연결 상태 캐시는 무시하고 실제 조회 수행
                pass
        
        # MongoDB 연결 시도 (선택적)
        mongodb_connected = False
        db = None
        try:
            from database import get_database  # type: ignore
            try:
                db = get_database()
                if db:
                    mongodb_connected = db.is_connected()
                else:
                    logging.warning(f"[API] 통계 조회 - get_database()가 None 반환")
            except Exception as db_init_error:
                logging.warning(f"[API] 통계 조회 - MongoDB 초기화 실패: {db_init_error}", exc_info=True)
                db = None
                mongodb_connected = False
            
            if db and mongodb_connected:
                # 쿼리 파라미터
                days = int(request.query.get('days', 7))
                
                # days=1일 때는 오늘 0시부터의 데이터만 집계
                start_timestamp = None
                end_timestamp = None
                if days == 1:
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    start_timestamp = int(today.timestamp() * 1000)
                    end_timestamp = int(datetime.now().timestamp() * 1000)
                
                # 통계 조회 (SimpleViolationService 사용)
                try:
                    stats = db.get_violation_statistics(days=days, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
                except Exception as stats_error:
                    logging.error(f"[API] 통계 조회 실패: {stats_error}", exc_info=True)
                    # 기본 chart_data 생성
                    default_chart_data = []
                    for i in range(7):
                        default_chart_data.append({
                            "week": i,
                            "violations": 0,
                            "helmet_violations": 0,
                            "fall_detections": 0,
                            "completed": 0
                        })
                    stats = {
                        "total_violations": 0,
                        "event_type_stats": {},
                        "chart_data": default_chart_data
                    }
                
                # 가이드 스키마 호환 형식으로 변환
                event_type_stats = stats.get('event_type_stats', {})
                
                # 오늘의 위반 사항 수 (간단한 방법: 전체 통계에서 추정)
                today_count = 0
                try:
                    # SimpleViolationService를 통해 오늘의 위반 사항 조회
                    violations_collection = db.db_service.get_violations_collection()
                    if violations_collection is not None:
                        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                        today_str = today.strftime('%Y-%m-%d')
                        today_count = violations_collection.count_documents({
                            "$or": [
                                {"violation_datetime": {"$regex": f"^{today_str}"}},
                                {"timestamp": {"$gte": int(today.timestamp() * 1000)}}
                            ]
                        })
                except Exception as e:
                    logging.debug(f"오늘의 위반 사항 수 조회 실패 (무시): {e}")
                    today_count = 0
                
                # 타입별 집계 (event_type_stats 사용)
                by_type = {
                    "NO_HELMET": event_type_stats.get('helmet', 0),
                    "NO_VEST": event_type_stats.get('vest', 0),
                    "FALL": event_type_stats.get('fall', 0)
                }
                
                # 실시간 KPI 데이터 가져오기 (model_results에서)
                realtime_total_workers = 0
                realtime_safety_score = None
                try:
                    if model_results and "kpi_data" in model_results:
                        kpi_data = model_results["kpi_data"]
                        realtime_total_workers = kpi_data.get("totalWorkers", 0)
                        realtime_safety_score = kpi_data.get("safetyScore", None)
                except Exception as e:
                    logging.debug(f"실시간 KPI 데이터 조회 실패 (무시): {e}")
                
                # 가이드 스키마 응답 형식
                response_data = {
                    "success": True,
                    "total": stats.get('total_violations', 0),
                    "today": today_count,
                    "by_type": by_type,
                    "kpi": {
                        "helmet": by_type.get("NO_HELMET", 0),
                        "vest": by_type.get("NO_VEST", 0),
                        "fall": by_type.get("FALL", 0),
                        "total": stats.get('total_violations', 0),
                        "total_workers": realtime_total_workers,  # 실시간 작업자 수 추가
                        "safety_score": realtime_safety_score  # 실시간 안전 점수 추가
                    },
                    "chart_data": stats.get('chart_data', []),
                    "mongodb_connected": True  # MongoDB 연결 상태
                }
                
                
                # 결과 캐싱
                stats_cache.put(cache_key, response_data)
                return web.json_response(response_data)
        except ImportError as import_err:
            logging.warning(f"[API] 통계 조회 - MongoDB 모듈 import 실패: {import_err}")
            mongodb_connected = False
            pass  # database 모듈이 없으면 아래 코드 실행
        except Exception as db_error:
            logging.error(f"[API] 통계 조회 - MongoDB 통계 조회 실패: {db_error}", exc_info=True)
            mongodb_connected = False
        
        # MongoDB가 없으면 빈 결과 반환 (프론트엔드 호환 형식)
        chart_data = []
        for i in range(7):
            chart_data.append({
                "week": i,
                "violations": 0,
                "helmet_violations": 0,
                "fall_detections": 0,
                "completed": 0
            })
        
        # 실시간 KPI 데이터 가져오기 (model_results에서)
        realtime_total_workers = 0
        realtime_safety_score = None
        try:
            if model_results and "kpi_data" in model_results:
                kpi_data = model_results["kpi_data"]
                realtime_total_workers = kpi_data.get("totalWorkers", 0)
                realtime_safety_score = kpi_data.get("safetyScore", None)
        except Exception as e:
            logging.debug(f"실시간 KPI 데이터 조회 실패 (무시): {e}")
        
        response_data = {
            "success": True,
            "mongodb_connected": mongodb_connected,  # MongoDB 연결 상태 (실제 상태 반영)
            "kpi": {
                "helmet": 0,
                "vest": 0,
                "fall": 0,
                "total": 0,
                "total_workers": realtime_total_workers,  # 실시간 작업자 수 추가
                "safety_score": realtime_safety_score  # 실시간 안전 점수 추가
            },
            "chart_data": chart_data,
            "message": "통계 조회 성공 (MongoDB 미연결)"
        }
        
        # 결과 캐싱 (MongoDB 미연결 상태도 캐싱)
        stats_cache.put(cache_key, response_data)
        return web.json_response(response_data)
    
    except Exception as e:
        logging.error(f"통계 조회 중 오류: {e}")
        response = create_standard_response(
            status="error",
            message=f"통계 조회 실패: {str(e)}",
            error_code="STATISTICS_ERROR"
        )
        
        # 에러 발생 시에도 캐싱 (짧은 TTL로 재시도 방지)
        cache_key = f"stats_{request.query_string}"
        stats_cache.put(cache_key, response, ttl=5.0)  # 5초만 캐싱
        return web.json_response(response, status=500)

# --- 프론트엔드 API 핸들러들 ---
async def api_stats_handler(request: web.Request):
    """통계 데이터 API"""
    global storage_manager
    try:
        if storage_manager is None:
            return web.json_response({
                'success': False,
                'error': '데이터 저장소가 초기화되지 않았습니다.'
            }, status=500)
        
        today = datetime.date.today()
        stats = storage_manager.get_daily_statistics(today)
        
        # 전체 등록된 사람 수
        all_persons = storage_manager.get_all_persons()
        
        # 오늘 출입 기록 수
        today_logs = storage_manager.get_access_logs(start_date=today, end_date=today)
        
        # 오늘 출입 이미지 수
        today_images = storage_manager.get_daily_access_images(access_date=today)
        
        return web.json_response({
            'success': True,
            'data': {
                'total_images': stats['total_images'] if stats else 0,
                'total_persons': len(all_persons),
                'today_access_count': len(today_logs),
                'today_images_count': len(today_images),
                'matched_count': len([log for log in today_logs if log.get('person_id')])
            }
        })
    except Exception as e:
        logging.error(f"통계 조회 오류: {e}", exc_info=True)
        return web.json_response({'success': False, 'error': str(e)}, status=500)


async def api_persons_get_handler(request: web.Request):
    """등록된 사람 목록 API (GET)"""
    global storage_manager
    try:
        if storage_manager is None:
            return web.json_response({
                'success': False,
                'error': '데이터 저장소가 초기화되지 않았습니다.'
            }, status=500)
        
        persons = storage_manager.get_all_persons()
        return web.json_response({'success': True, 'data': persons})
    except Exception as e:
        logging.error(f"사람 목록 조회 오류: {e}", exc_info=True)
        return web.json_response({'success': False, 'error': str(e)}, status=500)


async def api_persons_post_handler(request: web.Request):
    """사람 등록 API (POST)"""
    global storage_manager
    try:
        if storage_manager is None:
            return web.json_response({
                'success': False,
                'error': '데이터 저장소가 초기화되지 않았습니다.'
            }, status=500)
        
        # multipart/form-data 파싱
        data = await request.post()
        
        person_id = data.get('person_id')
        name = data.get('name')
        department = data.get('department', '')
        position = data.get('position', '')
        phone = data.get('phone', '')
        email = data.get('email', '')
        use_captured_image = data.get('use_captured_image') == 'true'
        captured_image_path = data.get('captured_image_path')
        
        if not person_id or not name:
            return web.json_response({
                'success': False,
                'error': '사람 ID와 이름은 필수입니다'
            }, status=400)
        
        # 이미지 처리
        registered_image_path = None
        if use_captured_image and captured_image_path:
            # 촬영된 이미지 사용
            source_path = Path(captured_image_path)
            if not source_path.exists():
                return web.json_response({
                    'success': False,
                    'error': '촬영된 이미지를 찾을 수 없습니다'
                }, status=400)
            registered_image_path = str(source_path)
        else:
            # 파일 업로드 사용
            if 'image' not in data:
                return web.json_response({
                    'success': False,
                    'error': '이미지 파일이 없습니다'
                }, status=400)
            
            file = data['image']
            if not hasattr(file, 'filename') or not file.filename:
                return web.json_response({
                    'success': False,
                    'error': '파일이 선택되지 않았습니다'
                }, status=400)
            
            # 이미지 저장 경로 설정 (프로젝트 루트/images 폴더)
            base_dir = Path(__file__).parent.parent.parent
            images_folder = base_dir / 'images'
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # 파일 저장
            # 파일명 안전하게 처리 (Windows 경로 문제 방지)
            import re
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', file.filename)
            filename = f"{person_id}_{safe_filename}"
            filepath = images_folder / filename
            
            # 파일 내용 읽기 및 저장
            file_content = file.file.read()
            with open(filepath, 'wb') as f:
                f.write(file_content)
            
            registered_image_path = str(filepath)
        
        # 사람 등록
        record_id = storage_manager.register_person(
            person_id=person_id,
            name=name,
            registered_image_path=registered_image_path,
            face_encoding_path=None,  # TODO: 얼굴 인코딩 저장 기능 추가
            department=department,
            position=position,
            phone=phone,
            email=email
        )
        
        if record_id:
            return web.json_response({'success': True, 'data': {'id': record_id}})
        else:
            return web.json_response({
                'success': False,
                'error': '등록 실패 (이미 존재하는 ID일 수 있습니다)'
            }, status=400)
            
    except Exception as e:
        logging.error(f"사람 등록 오류: {e}", exc_info=True)
        return web.json_response({'success': False, 'error': str(e)}, status=500)


async def api_access_logs_handler(request: web.Request):
    """출입 기록 조회 API"""
    global storage_manager
    try:
        if storage_manager is None:
            return web.json_response({
                'success': False,
                'error': '데이터 저장소가 초기화되지 않았습니다.'
            }, status=500)
        
        start_date_str = request.query.get('start_date')
        end_date_str = request.query.get('end_date')
        camera_id = request.query.get('camera_id')
        limit = int(request.query.get('limit', 100))
        
        start_date = None
        end_date = None
        if start_date_str:
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if end_date_str:
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        logs = storage_manager.get_access_logs(
            start_date=start_date,
            end_date=end_date,
            camera_id=camera_id,
            limit=limit
        )
        
        return web.json_response({'success': True, 'data': logs})
    except Exception as e:
        logging.error(f"출입 기록 조회 오류: {e}", exc_info=True)
        return web.json_response({'success': False, 'error': str(e)}, status=500)


async def api_daily_images_handler(request: web.Request):
    """일일 출입 이미지 조회 API"""
    global storage_manager
    try:
        if storage_manager is None:
            return web.json_response({
                'success': False,
                'error': '데이터 저장소가 초기화되지 않았습니다.'
            }, status=500)
        
        person_id = request.query.get('person_id', type=int)
        date_str = request.query.get('date')
        
        access_date = datetime.date.today()
        if date_str:
            access_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        
        images = storage_manager.get_daily_access_images(
            person_id=person_id,
            access_date=access_date
        )
        
        return web.json_response({'success': True, 'data': images})
    except Exception as e:
        logging.error(f"일일 이미지 조회 오류: {e}", exc_info=True)
        return web.json_response({'success': False, 'error': str(e)}, status=500)


# 백그라운드 임베딩 작업을 위한 전역 변수
_embedding_thread_lock = threading.Lock()
_embedding_thread_running = False
_main_event_loop = None  # 메인 이벤트 루프 참조 저장
_processing_workers = set()  # 현재 처리 중인 작업자 집합 (중복 처리 방지)


def check_complete_image_set(worker_folder: Path) -> bool:
    """작업자 폴더에 정면, 왼쪽, 오른쪽 이미지가 모두 있는지 확인"""
    if not worker_folder.exists():
        return False
    
    required_patterns = ['capture_정면_', 'capture_왼쪽_', 'capture_오른쪽_']
    found_patterns = set()
    
    for image_file in worker_folder.glob('*.jpg'):
        filename = image_file.name
        for pattern in required_patterns:
            if pattern in filename:
                found_patterns.add(pattern)
                break
    
    return len(found_patterns) == len(required_patterns)


def run_background_embedding(base_dir: Path):
    """백그라운드에서 임베딩 작업을 실행하는 함수"""
    global _embedding_thread_running
    
    def _run_embedding():
        global _embedding_thread_running, _main_event_loop
        try:
            logging.info("[EMBEDDING] 백그라운드 임베딩 작업 시작")
            
            # build_database.py 스크립트 경로
            build_database_script = base_dir / 'face' / 'scripts' / 'build_database.py'
            
            if not build_database_script.exists():
                logging.warning(f"[EMBEDDING] build_database.py를 찾을 수 없습니다: {build_database_script}")
                return
            
            # 환경 변수 설정
            env = os.environ.copy()
            pythonpath_list = [
                str(base_dir / 'src' / 'backend'),
                str(base_dir / 'src'),
                str(base_dir),
            ]
            if 'PYTHONPATH' in env:
                pythonpath_list.append(env['PYTHONPATH'])
            env['PYTHONPATH'] = os.pathsep.join(pythonpath_list)
            
            # build_database.py 실행
            script_dir = base_dir / 'face' / 'scripts'
            result = subprocess.run(
                [sys.executable, str(build_database_script)],
                cwd=str(script_dir),
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logging.info("[EMBEDDING] ✅ 백그라운드 임베딩 작업 완료 (PPE 합성 + 증강 + 임베딩 모두 완료)")
                
                # stdout 출력에서 통계 정보 추출
                embedding_stats = {
                    'new_images': 0,
                    'new_embeddings': 0,
                    'total_embeddings': 0,
                    'total_persons': 0
                }
                
                if result.stdout:
                    # 중요한 정보만 추출하여 로그에 기록
                    stdout_lines = result.stdout.split('\n')
                    for line in stdout_lines:
                        if any(keyword in line for keyword in ['처리한 새 이미지', '새로 추가된 임베딩', '인덱스 총 임베딩', '처리 완료', '완료']):
                            logging.info(f"[EMBEDDING] {line.strip()}")
                        
                        # 통계 정보 파싱
                        if '처리한 새 이미지' in line:
                            try:
                                import re
                                match = re.search(r'(\d+)개', line)
                                if match:
                                    embedding_stats['new_images'] = int(match.group(1))
                            except:
                                pass
                        if '새로 추가된 임베딩' in line:
                            try:
                                import re
                                match = re.search(r'(\d+)개', line)
                                if match:
                                    embedding_stats['new_embeddings'] = int(match.group(1))
                            except:
                                pass
                        if '인덱스 총 임베딩' in line:
                            try:
                                import re
                                match = re.search(r'(\d+)개', line)
                                if match:
                                    embedding_stats['total_embeddings'] = int(match.group(1))
                            except:
                                pass
                        if '인덱스 총 인물' in line:
                            try:
                                import re
                                match = re.search(r'(\d+)명', line)
                                if match:
                                    embedding_stats['total_persons'] = int(match.group(1))
                            except:
                                pass
                
                # 프론트엔드에 완료 알림 전송
                try:
                    if _main_event_loop and not _main_event_loop.is_closed():
                        # WebSocket으로 알림 전송
                        notification = {
                            'type': 'embedding_complete',
                            'status': 'success',
                            'message': '임베딩 작업이 완료되었습니다.',
                            'stats': embedding_stats,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # 이벤트 루프에서 코루틴 실행
                        asyncio.run_coroutine_threadsafe(
                            _send_embedding_notification(notification),
                            _main_event_loop
                        )
                        logging.info(f"[EMBEDDING] 프론트엔드 알림 전송: {notification}")
                except Exception as e:
                    logging.warning(f"[EMBEDDING] 프론트엔드 알림 전송 실패: {e}")
            else:
                logging.warning(f"[EMBEDDING] ⚠️ 임베딩 작업 실패 (코드: {result.returncode})")
                if result.stderr:
                    logging.warning(f"[EMBEDDING] 오류 출력: {result.stderr[:500]}")
                if result.stdout:
                    logging.warning(f"[EMBEDDING] stdout 출력: {result.stdout[:500]}")
                
                # 실패 알림도 전송
                try:
                    if _main_event_loop and not _main_event_loop.is_closed():
                        notification = {
                            'type': 'embedding_complete',
                            'status': 'error',
                            'message': f'임베딩 작업이 실패했습니다. (코드: {result.returncode})',
                            'timestamp': datetime.now().isoformat()
                        }
                        asyncio.run_coroutine_threadsafe(
                            _send_embedding_notification(notification),
                            _main_event_loop
                        )
                except Exception as e:
                    logging.warning(f"[EMBEDDING] 실패 알림 전송 실패: {e}")
        except Exception as e:
            logging.error(f"[EMBEDDING] 백그라운드 임베딩 작업 오류: {e}", exc_info=True)
        finally:
            with _embedding_thread_lock:
                _embedding_thread_running = False
            # 처리 완료 후 작업자 제거는 다음 촬영 시 자동으로 재확인됨
            # (임베딩 완료 후 _embedding_thread_running이 False가 되면
            #  다음 촬영에서 check_complete_image_set이 다시 실행됨)
            logging.info("[EMBEDDING] 백그라운드 임베딩 스레드 종료")
    
    # 중복 실행 방지
    with _embedding_thread_lock:
        if _embedding_thread_running:
            logging.info("[EMBEDDING] 이미 임베딩 작업이 실행 중입니다. 건너뜁니다.")
            return
        _embedding_thread_running = True
    
    # 백그라운드 스레드로 실행
    thread = threading.Thread(target=_run_embedding, daemon=True)
    thread.start()
    logging.info("[EMBEDDING] 백그라운드 임베딩 스레드 시작")


def get_worker_info_by_id(worker_id: str) -> Tuple[Optional[str], Optional[int]]:
    """worker_id로 작업자 정보 조회 (registered_persons.json + MongoDB)"""
    if not worker_id:
        return None, None
    
    try:
        # 1. registered_persons.json에서 조회
        if storage_manager:
            persons = storage_manager.get_all_persons()
            for p in persons:
                if str(p.get('person_id')) == str(worker_id) or str(p.get('id')) == str(worker_id):
                    person_name = p.get('name')
                    person_id = p.get('id')
                    logging.info(f"[CAPTURE] registered_persons.json에서 작업자 정보 조회 성공: {person_name} (ID: {person_id}, worker_id: {worker_id})")
                    return person_name, person_id
        
        # 2. MongoDB에서 조회
        try:
            from database import get_database
            db = get_database()
            if db and db.is_connected():
                workers = db.get_all_workers(active_only=True)
                for worker in workers:
                    # worker_id 또는 workerId 필드 확인
                    w_id = str(worker.get('worker_id', '')) or str(worker.get('workerId', ''))
                    if w_id == str(worker_id):
                        person_name = worker.get('name')
                        # MongoDB의 _id를 person_id로 사용 (또는 worker_id를 그대로 사용)
                        person_id = worker.get('_id') or worker.get('worker_id')
                        logging.info(f"[CAPTURE] MongoDB에서 작업자 정보 조회 성공: {person_name} (worker_id: {worker_id})")
                        return person_name, person_id
        except Exception as db_error:
            logging.debug(f"[CAPTURE] MongoDB 작업자 조회 실패 (무시): {db_error}")
        
        logging.warning(f"[CAPTURE] worker_id로 작업자 정보를 찾을 수 없음: {worker_id}")
        return None, None
    except Exception as e:
        logging.warning(f"[CAPTURE] worker_id로 작업자 정보 조회 중 오류: {e}")
        return None, None


async def api_capture_stream_handler(request: web.Request):
    """웹 스트림에서 이미지 캡처 및 매칭 API"""
    global storage_manager
    try:
        if storage_manager is None:
            return web.json_response({
                'success': False,
                'error': '데이터 저장소가 초기화되지 않았습니다.'
            }, status=500)
        
        # multipart/form-data 파싱
        data = await request.post()
        
        if 'image' not in data:
            return web.json_response({
                'success': False,
                'error': '이미지 파일이 없습니다'
            }, status=400)
        
        file = data['image']
        if not hasattr(file, 'filename') or not file.filename:
            return web.json_response({
                'success': False,
                'error': '파일명이 없습니다'
            }, status=400)
        
        camera_id = int(data.get('camera_id', 0))
        camera_name = data.get('camera_name', 'CAM001')
        location = data.get('location', '정문')
        step = data.get('step', 'unknown')
        worker_id = data.get('worker_id') or data.get('workerId') or data.get('worker_code')  # 입력받은 작업자 ID
        if worker_id and worker_id == 'UNKNOWN':
            worker_id = None  # UNKNOWN은 None으로 처리
        logging.info(f"[CAPTURE] 촬영 요청: step={step}, worker_id={worker_id}, camera_id={camera_id}")
        
        # 이미지 저장 경로 설정 (프로젝트 루트/images 폴더)
        base_dir = Path(__file__).parent.parent.parent
        images_folder = base_dir / 'images'
        images_folder.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장 (임시 파일명으로 먼저 저장, 얼굴 인식 후 최종 파일명으로 변경)
        timestamp = datetime.now()
        step_names = {
            'front': '정면',
            'left': '왼쪽',
            'right': '오른쪽',
            'unknown': ''
        }
        step_name = step_names.get(step, '')
        
        # 임시 파일명으로 저장
        temp_filename = f"temp_{timestamp.strftime('%Y%m%d_%H%M%S')}_{step_name}.jpg"
        temp_image_path = images_folder / temp_filename
        
        # 파일 내용 읽기 및 저장
        file_content = file.file.read()
        with open(temp_image_path, 'wb') as f:
            f.write(file_content)
        
        # 얼굴 검출 및 인식
        person_id = None
        person_name = '미등록 사용자'
        match_confidence = None
        matched = False
        status = 'normal'
        face_locations = []
        
        # 얼굴 검출 및 인식 수행
        if safety_system_instance:
            try:
                import cv2
                import numpy as np
                from ai_processors import _process_face_recognition
                from utils import find_best_match_faiss
                import config
                
                # 이미지 읽기
                img = cv2.imread(str(temp_image_path))
                if img is not None:
                    # 얼굴 감지 (YOLO face model 사용)
                    face_model = getattr(safety_system_instance, 'face_model', None)
                    if face_model:
                        try:
                            # YOLO 얼굴 감지
                            results = face_model(img, verbose=False)
                            face_boxes = []
                            if results and len(results) > 0:
                                for result in results:
                                    boxes = result.boxes
                                    if boxes is not None:
                                        for box in boxes:
                                            # 얼굴 클래스만 필터링 (클래스 ID 확인 필요)
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                            conf = float(box.conf[0])
                                            logging.debug(f"[CAPTURE] 얼굴 감지 시도: conf={conf:.3f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
                                            if conf > 0.5:  # 신뢰도 임계값
                                                face_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                                                face_locations.append((int(y1), int(x2), int(y2), int(x1)))
                                                logging.info(f"[CAPTURE] ✅ 얼굴 감지 성공: conf={conf:.3f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
                            
                            if not face_boxes:
                                logging.warning(f"[CAPTURE] ⚠️ 얼굴 감지 실패: 신뢰도 0.5 이상인 얼굴이 없음")
                            
                            # 얼굴이 감지된 경우 인식 시도
                            if face_boxes:
                                # 첫 번째 얼굴만 처리
                                x1, y1, x2, y2 = face_boxes[0]
                                person_img = img[y1:y2, x1:x2]
                                
                                # 얼굴 인식 수행
                                face_analyzer = getattr(safety_system_instance, 'face_analyzer', None)
                                face_database = getattr(safety_system_instance, 'face_database', None)
                                fast_recognizer = getattr(safety_system_instance, 'fast_recognizer', None)
                                use_adaface = getattr(safety_system_instance, 'use_adaface', False)
                                adaface_model_path = getattr(safety_system_instance, 'adaface_model_path', None)
                                face_uses_trt = getattr(safety_system_instance, 'face_uses_trt', False)
                                
                                if face_analyzer and face_database:
                                    try:
                                        person_name, similarity_score, embedding, face_bbox = _process_face_recognition(
                                            person_img_for_detection=person_img,
                                            person_id_text='capture',
                                            face_model=face_model,
                                            face_analyzer=face_analyzer,
                                            face_database=face_database,
                                            use_adaface=use_adaface,
                                            adaface_model_path=adaface_model_path,
                                            fast_recognizer=fast_recognizer,
                                            pre_detected_face=None,
                                            original_frame=img,
                                            face_uses_trt=face_uses_trt
                                        )
                                        
                                        if person_name and person_name != "Unknown":
                                            matched = True
                                            match_confidence = float(similarity_score) if similarity_score else None
                                            status = 'normal'
                                            
                                            # person_id 찾기 (registered_persons.json에서)
                                            persons = storage_manager.get_all_persons()
                                            for p in persons:
                                                if p.get('name') == person_name or p.get('person_id') == person_name:
                                                    person_id = p.get('id')
                                                    break
                                        else:
                                            status = 'warning'
                                            person_name = '미등록 사용자'
                                    except Exception as e:
                                        logging.warning(f"얼굴 인식 실패: {e}")
                                        status = 'warning'
                                        person_name = '미등록 사용자'
                                else:
                                    status = 'warning'
                                    person_name = '미등록 사용자'
                            else:
                                status = 'warning'
                                person_name = '미등록 사용자'
                        except Exception as e:
                            logging.warning(f"얼굴 감지 실패: {e}")
                            status = 'warning'
                            person_name = '미등록 사용자'
                    else:
                        status = 'warning'
                        person_name = '미등록 사용자'
            except Exception as e:
                logging.warning(f"얼굴 인식 처리 중 오류: {e}")
                status = 'warning'
                person_name = '미등록 사용자'
        else:
            status = 'warning'
            person_name = '미등록 사용자'
        
        # 얼굴 인식 실패 시 worker_id로 작업자 정보 조회 (모든 경우에 대해)
        if person_name == '미등록 사용자' and worker_id:
            found_name, found_id = get_worker_info_by_id(worker_id)
            if found_name:
                person_name = found_name
                person_id = found_id
                matched = True
                status = 'normal'
                logging.info(f"[CAPTURE] ✅ worker_id로 작업자 정보 조회 성공: {person_name} (ID: {person_id}, worker_id: {worker_id})")
        
        # 얼굴 인식 후 최종 저장 경로 설정 (face/data/new_faces/{작업자이름}/)
        # new_faces 폴더에 작업자 이름 폴더 자동 생성
        new_faces_base = base_dir / 'face' / 'data' / 'new_faces'
        new_faces_base.mkdir(parents=True, exist_ok=True)
        
        # 작업자 이름으로 폴더 생성 (한글 이름 그대로 사용)
        if person_name and person_name != '미등록 사용자':
            worker_folder = new_faces_base / person_name
        else:
            worker_folder = new_faces_base / '미등록'
        worker_folder.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성 (작업자 이름 포함)
        # person_name에서 파일명에 사용할 수 없는 문자 제거 (공백, 특수문자 등)
        safe_person_name = person_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        safe_person_name = ''.join(c for c in safe_person_name if c.isalnum() or c in ('_', '-', '.'))
        
        if step_name:
            if person_name and person_name != '미등록 사용자':
                filename = f"capture_{step_name}_{safe_person_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                filename = f"capture_{step_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            if person_name and person_name != '미등록 사용자':
                filename = f"capture_{safe_person_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                filename = f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # 최종 경로로 파일 이동 (new_faces/{작업자이름}/)
        image_path = worker_folder / filename
        try:
            if temp_image_path.exists():
                if temp_image_path != image_path:
                    # 파일 이동 (rename은 같은 파일시스템 내에서만 작동)
                    import shutil
                    shutil.move(str(temp_image_path), str(image_path))
                    logging.info(f"[CAPTURE] 파일 저장: face/data/new_faces/{worker_folder.name}/{filename} (작업자: {person_name})")
                else:
                    logging.info(f"[CAPTURE] 파일 이미 존재: {filename}")
        except Exception as e:
            logging.warning(f"[CAPTURE] 파일 이동 실패, 임시 파일명 유지: {e}")
            image_path = temp_image_path  # 이동 실패 시 임시 파일 경로 사용
        
        # 일일 출입 이미지 저장 (파일명 변경 후)
        if person_id and person_name and person_name != '미등록 사용자':
            try:
                storage_manager.save_daily_access_image(
                    person_id=person_id,
                    person_name=person_name,
                    image_path=str(image_path),
                    access_date=timestamp.date()
                )
            except Exception as e:
                logging.warning(f"일일 출입 이미지 저장 실패: {e}")
        
        # 출입 기록 저장 (aivis-front와 동일한 로직)
        try:
            if storage_manager:
                record_id = storage_manager.save_access_log(
                    image_path=str(image_path),
                    camera_id=camera_name,
                    location=location,
                    detected_person_count=len(face_locations) if face_locations else 0,
                    status=status,
                    notes='웹에서 캡처',
                    timestamp=timestamp,
                    person_id=person_id,
                    person_name=person_name,
                    match_confidence=match_confidence,
                    worker_id=worker_id  # 입력받은 작업자 ID 추가
                )
            else:
                record_id = None
        except Exception:
            record_id = None
        
        # 3장(정면, 왼쪽, 오른쪽)이 모두 들어왔는지 확인 후 자동 처리
        # 작업자 정보가 있고, new_faces 폴더에 저장된 경우에만 확인
        if person_name and person_name != '미등록 사용자' and worker_folder.exists():
            try:
                global _processing_workers
                
                # 3장이 모두 있는지 확인
                is_complete = check_complete_image_set(worker_folder)
                
                if is_complete:
                    # 이미 처리 중인 작업자는 건너뜀
                    if person_name in _processing_workers:
                        logging.info(f"[CAPTURE] 작업자 '{person_name}'는 이미 처리 중입니다. 건너뜁니다.")
                    elif _embedding_thread_running:
                        logging.info(f"[CAPTURE] 다른 작업자의 임베딩이 진행 중입니다. 대기 후 자동 처리됩니다.")
                    else:
                        _processing_workers.add(person_name)
                        logging.info(f"[CAPTURE] ✅ 3장 모두 완료! 자동 처리 시작: 작업자={person_name}")
                        logging.info(f"[CAPTURE] 처리 순서: 1) new_faces → data/images 이동, 2) PPE 합성, 3) 증강, 4) 임베딩")
                        
                        # 백그라운드에서 임베딩 작업 시작 (자동으로 PPE 합성, 증강, 임베딩 수행)
                        try:
                            run_background_embedding(base_dir)
                            # 처리 완료 후 작업자 제거는 임베딩 완료 시점에 수행
                            # (임베딩 스레드가 완료되면 _embedding_thread_running이 False가 되므로
                            #  다음 촬영 시 자동으로 재확인됨)
                        except Exception as e:
                            logging.warning(f"[CAPTURE] 백그라운드 임베딩 시작 실패: {e}")
                            _processing_workers.discard(person_name)
                else:
                    # 아직 3장이 모두 들어오지 않음
                    current_images = list(worker_folder.glob('*.jpg'))
                    found_patterns = []
                    for img in current_images:
                        if 'capture_정면_' in img.name:
                            found_patterns.append('정면')
                        elif 'capture_왼쪽_' in img.name:
                            found_patterns.append('왼쪽')
                        elif 'capture_오른쪽_' in img.name:
                            found_patterns.append('오른쪽')
                    logging.info(f"[CAPTURE] 작업자 '{person_name}': 현재 {len(current_images)}장 저장됨 ({', '.join(found_patterns) if found_patterns else '없음'}) - 3장 필요: 정면, 왼쪽, 오른쪽")
            except Exception as e:
                logging.warning(f"[CAPTURE] 이미지 세트 확인 중 오류: {e}")
        
        return web.json_response({
            'success': True,
            'image_path': str(image_path),
            'absolute_path': str(image_path.resolve()),
            'filename': filename,
            'data': {
                'image_path': str(image_path),
                'record_id': record_id,
                'has_face': True if face_locations else False,
                'face_count': len(face_locations) if face_locations else 0,
                'matched': matched,
                'person_name': str(person_name) if person_name else '미등록 사용자',
                'match_confidence': float(match_confidence) if match_confidence is not None else None,
                'timestamp': timestamp.isoformat()
            },
            'message': '이미지 저장 완료'
        })
        
    except Exception as e:
        logging.error(f"이미지 캡처 오류: {e}", exc_info=True)
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)


async def api_detect_face_position_handler(request: web.Request):
    """얼굴 위치 감지 API (자동 촬영용)"""
    global safety_system_instance
    try:
        # multipart/form-data 파싱
        data = await request.post()
        
        if 'image' not in data:
            return web.json_response({
                'success': False,
                'error': '이미지 파일이 없습니다'
            }, status=400)
        
        file = data['image']
        
        # 임시 파일로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file_content = file.file.read()
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # OpenCV로 이미지 읽기
            img = cv2.imread(tmp_path)
            if img is None:
                return web.json_response({
                    'success': False,
                    'error': '이미지를 읽을 수 없습니다.'
                }, status=500)
            
            img_height, img_width = img.shape[:2]
            
            # SafetySystem이 초기화되지 않았거나 얼굴 모델이 없으면 기본 응답 반환
            if safety_system_instance is None or safety_system_instance.face_model is None:
                logging.warning("얼굴 검출 모델이 없습니다. 기본 응답 반환")
                return web.json_response({
                    'success': True,
                    'has_face': True,
                    'face_detected': True,
                    'should_capture': False,
                    'face_size_info': {
                        'current_width': 0,
                        'current_height': 0,
                        'threshold': 150
                    },
                    'message': '얼굴 검출 모델이 초기화되지 않았습니다.'
                })
            
            # YOLO 얼굴 감지 모델 사용
            face_model = safety_system_instance.face_model
            
            # 얼굴 검출 수행 (GPU 사용)
            face_kwargs = {'conf': config.Thresholds.FACE_DETECTION_CONFIDENCE, 'verbose': False}
            if hasattr(safety_system_instance, 'device_face') and 'cuda' in str(safety_system_instance.device_face):
                face_kwargs['device'] = safety_system_instance.device_face
            yolo_results = face_model(img, **face_kwargs)
            
            # 얼굴이 검출되지 않은 경우
            if not yolo_results or len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
                return web.json_response({
                    'success': True,
                    'has_face': False,
                    'face_detected': False
                })
            
            # 첫 번째 얼굴 선택 (confidence가 가장 높은 것)
            boxes_with_conf = []
            for box in yolo_results[0].boxes:
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                boxes_with_conf.append((bbox, conf))
            
            # confidence 기준으로 정렬
            boxes_with_conf.sort(key=lambda x: x[1], reverse=True)
            
            if not boxes_with_conf:
                return web.json_response({
                    'success': True,
                    'has_face': False,
                    'face_detected': False
                })
            
            # 첫 번째 얼굴의 위치 사용
            biggest_bbox, biggest_conf = boxes_with_conf[0]
            fx1, fy1, fx2, fy2 = int(biggest_bbox[0]), int(biggest_bbox[1]), int(biggest_bbox[2]), int(biggest_bbox[3])
            
            # 얼굴 중심점 계산
            face_center_x = (fx1 + fx2) / 2
            face_center_y = (fy1 + fy2) / 2
            
            # 얼굴 크기 계산
            face_width = fx2 - fx1
            face_height = fy2 - fy1
            
            # 이미지 중심점
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            
            # 가이드라인 영역 (중앙 60% 영역)
            guide_width = img_width * 0.6
            guide_height = img_height * 0.6
            guide_left = img_center_x - guide_width / 2
            guide_right = img_center_x + guide_width / 2
            guide_top = img_center_y - guide_height / 2
            guide_bottom = img_center_y + guide_height / 2
            
            # 얼굴이 가이드라인 영역 안에 있는지 확인
            face_in_guide = (
                face_center_x >= guide_left and 
                face_center_x <= guide_right and
                face_center_y >= guide_top and
                face_center_y <= guide_bottom
            )
            
            # 얼굴 크기가 적절한지 확인 (이미지의 15%~35% 정도)
            face_size_ratio = (face_width * face_height) / (img_width * img_height)
            face_size_ok = 0.15 <= face_size_ratio <= 0.35
            
            # 적절한 얼굴 크기 계산 (이미지의 20-25% 정도를 목표로)
            target_face_size_ratio = 0.20  # 20% 목표
            target_face_width = img_width * (target_face_size_ratio ** 0.5) * 1.2  # 약 1.2배 (세로가 더 길므로)
            target_face_height = img_width * (target_face_size_ratio ** 0.5) * 1.5  # 약 1.5배
            
            # 현재 얼굴 크기와 목표 크기 비교
            current_face_area = face_width * face_height
            target_face_area = target_face_width * target_face_height
            size_difference_ratio = current_face_area / target_face_area if target_face_area > 0 else 1.0
            
            # 중앙에서 얼굴 중심까지의 거리 (정규화)
            distance_from_center_x = abs(face_center_x - img_center_x) / img_width
            distance_from_center_y = abs(face_center_y - img_center_y) / img_height
            distance_from_center = (distance_from_center_x + distance_from_center_y) / 2
            
            # 촬영 조건: 얼굴이 가이드라인 안에 있고, 크기가 적절하며, 중앙에 가까움
            should_capture = face_in_guide and face_size_ok and distance_from_center < 0.15
            
            # 가이드라인 크기 계산 (얼굴 크기에 맞춤)
            # 얼굴이 작으면 큰 가이드라인, 크면 작은 가이드라인
            if size_difference_ratio < 0.8:
                # 얼굴이 작음 - 큰 가이드라인 표시 (가까이 가세요)
                guide_width_ratio = target_face_width / img_width * 1.3
                guide_height_ratio = target_face_height / img_height * 1.3
            elif size_difference_ratio > 1.3:
                # 얼굴이 큼 - 작은 가이드라인 표시 (멀리 가세요)
                guide_width_ratio = target_face_width / img_width * 0.8
                guide_height_ratio = target_face_height / img_height * 0.8
            else:
                # 적절한 크기 - 현재 얼굴 크기에 맞춤
                guide_width_ratio = face_width / img_width * 1.1
                guide_height_ratio = face_height / img_height * 1.1
            
            guide_width_px = img_width * guide_width_ratio
            guide_height_px = img_height * guide_height_ratio
            
            
            response_data = {
                'success': True,
                'has_face': True,
                'face_detected': True,
                'face_location': {
                    'top': int(fy1),
                    'right': int(fx2),
                    'bottom': int(fy2),
                    'left': int(fx1),
                    'center_x': int(face_center_x),
                    'center_y': int(face_center_y)
                },
                'image_size': {
                    'width': int(img_width),
                    'height': int(img_height)
                },
                'guide_area': {
                    'left': int(guide_left),
                    'right': int(guide_right),
                    'top': int(guide_top),
                    'bottom': int(guide_bottom)
                },
                'face_in_guide': face_in_guide,
                'face_size_ok': face_size_ok,
                'face_size_ratio': float(face_size_ratio),
                'distance_from_center': float(distance_from_center),
                'should_capture': should_capture,
                'face_size_info': {
                    'current_width': int(face_width),
                    'current_height': int(face_height),
                    'current_area': int(current_face_area),
                    'target_width': int(target_face_width),
                    'target_height': int(target_face_height),
                    'target_area': int(target_face_area),
                    'size_difference_ratio': float(size_difference_ratio),
                    'is_too_small': size_difference_ratio < 0.8,
                    'is_too_large': size_difference_ratio > 1.3
                },
                'guide_size': {
                    'width_px': int(guide_width_px),
                    'height_px': int(guide_height_px),
                    'width_ratio': float(guide_width_ratio),
                    'height_ratio': float(guide_height_ratio),
                    'center_x': int(img_center_x),
                    'center_y': int(img_center_y)
                }
            }
            
            return web.json_response(response_data)
            
        finally:
            # 임시 파일 삭제
            try:
                os.remove(tmp_path)
            except:
                pass
                
    except Exception as e:
        logging.error(f"얼굴 위치 감지 오류: {e}", exc_info=True)
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

# --- CORS 설정 및 라우트 등록 ---
def create_app():
    app = web.Application()

    # CORS 설정
    # 환경 변수에서 허용된 도메인 가져오기 (쉼표로 구분)
    allowed_origins_str = os.getenv('ALLOWED_ORIGINS', '')
    if allowed_origins_str:
        # 환경 변수가 있으면 해당 도메인만 허용
        allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]
        logging.info(f"[CORS] 허용된 도메인: {allowed_origins}")
        
        # 각 도메인별로 CORS 설정
        cors = aiohttp_cors.setup(app, defaults={
            origin: aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            ) for origin in allowed_origins
        })
    else:
        # 환경 변수가 없으면 개발 모드 (모든 도메인 허용, 경고 로그)
        logging.warning("[CORS] ⚠️ ALLOWED_ORIGINS 환경 변수가 설정되지 않았습니다. 모든 도메인 허용 (개발 모드)")
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

    # Rate Limiting 미들웨어 추가
    @web.middleware
    async def rate_limit_middleware(request, handler):
        """Rate Limiting 미들웨어"""
        # Rate Limiting 제외 경로 목록
        excluded_paths = [
            '/api/health',  # Health check
            '/api/violations/stats',  # 통계 조회 (실시간 업데이트)
            '/api/statistics',  # 통계 조회 (실시간 업데이트)
            '/api/violations',  # 위반 사항 조회 (실시간 업데이트)
            '/api/camera-results',  # 카메라 결과 (실시간 업데이트)
            '/api/model-results',  # 모델 결과 (실시간 업데이트)
            '/api/fps',  # FPS 추적 (실시간 업데이트)
            '/api/performance',  # 성능 통계 (실시간 업데이트)
            '/api/gpu',  # GPU 사용량 (실시간 업데이트)
            '/api/stats',  # 시스템 통계 (실시간 업데이트)
            '/stream',  # MJPEG 스트림
            '/api/stream',  # MJPEG 스트림
            '/video_feed',  # 비디오 피드
            '/api/camera-preview',  # 카메라 프리뷰
        ]
        
        # Health check, WebSocket, 실시간 API는 Rate Limiting 제외
        if (request.path == '/api/health' or 
            request.path.startswith('/ws') or
            any(request.path.startswith(path) for path in excluded_paths)):
            return await handler(request)
        
        # 클라이언트 IP 가져오기
        client_ip = request.remote
        if not client_ip:
            # X-Forwarded-For 헤더 확인 (리버스 프록시 사용 시)
            forwarded_for = request.headers.get('X-Forwarded-For', '')
            if forwarded_for:
                client_ip = forwarded_for.split(',')[0].strip()
            else:
                client_ip = 'unknown'
        
        # Rate Limit 확인
        if not rate_limiter.is_allowed(client_ip):
            remaining = rate_limiter.get_remaining(client_ip)
            logging.warning(f"[Rate Limit] IP {client_ip}의 요청이 제한되었습니다. (경로: {request.path}, 남은 요청: {remaining})")
            return web.json_response({
                'error': 'Rate limit exceeded',
                'error_code': 'RATE_LIMIT_EXCEEDED',
                'message': f'요청 한도가 초과되었습니다. 잠시 후 다시 시도해주세요.',
                'retry_after': int(RATE_LIMIT_PERIOD)
            }, status=429, headers={
                'Retry-After': str(int(RATE_LIMIT_PERIOD)),
                'X-RateLimit-Limit': str(RATE_LIMIT_MAX_CALLS),
                'X-RateLimit-Remaining': str(remaining)
            })
        
        # Rate Limit 헤더 추가
        response = await handler(request)
        remaining = rate_limiter.get_remaining(client_ip)
        response.headers['X-RateLimit-Limit'] = str(RATE_LIMIT_MAX_CALLS)
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        
        return response

    # 요청 통계 미들웨어 추가 (final과 동일)
    @web.middleware
    async def stats_middleware(request, handler):
        # Health check는 즉시 응답 (프레임 처리와 완전 분리)
        if request.path == '/api/health':
            return await handler(request)
        
        # WebSocket 연결은 미들웨어를 통과하되 통계는 건너뜀
        if request.path.startswith('/ws'):
            # WebSocket 연결은 직접 핸들러로 전달 (통계 측정 없음)
            return await handler(request)
        
        start_time = time.time()
        with stats_lock:
            system_stats["total_requests"] += 1
        
        response = None
        try:
            response = await handler(request)
            
            # 스트림 연결도 지속적이므로 응답 시간 측정 제외
            if request.path.startswith('/stream') or request.path.startswith('/video_feed'):
                # 스트림 연결이므로 응답 시간 로깅 생략 (연결 종료까지 기다리면 안됨)
                return response
            
            # 일반 HTTP 요청: 응답 시간 로깅 및 통계 업데이트
            response_time = time.time() - start_time
            with stats_lock:
                system_stats["response_times"].append(response_time)
                # 최대 개수 유지
                if len(system_stats["response_times"]) > MAX_RESPONSE_TIMES:
                    system_stats["response_times"] = system_stats["response_times"][-MAX_RESPONSE_TIMES:]
            if response_time > 1.0:  # 1초 이상 걸린 요청만 로깅
                logging.warning(f"느린 응답: {request.path} - {response_time:.2f}초")
            
            # 응답 반환 (중요!)
            return response
            
        except (ConnectionResetError, ConnectionError, OSError, asyncio.CancelledError) as e:
            # 클라이언트 연결 종료는 정상적인 동작이므로 DEBUG 레벨로만 로깅
            # 스트림 연결의 경우 특히 자주 발생할 수 있음
            # 스트림/비디오 피드 경로는 핸들러에서 이미 처리했으므로 여기서는 무시
            # 연결 종료는 에러로 간주하지 않음 (통계에 포함하지 않음)
            if not (request.path.startswith('/stream') or request.path.startswith('/video_feed')):
                logging.debug(f"클라이언트 연결 종료: {request.path}")
            raise
        except web.HTTPNotFound:
            # 404 오류는 경고 레벨로만 로깅 (자주 발생할 수 있음)
            with stats_lock:
                system_stats["error_count"] += 1
            logging.warning(f"요청 경로를 찾을 수 없습니다: {request.path}")
            raise
        except Exception as e:
            with stats_lock:
                system_stats["error_count"] += 1
            logging.error(f"[미들웨어] 요청 처리 중 오류: {request.path} - {e}", exc_info=True)
            # 미들웨어에서 예외가 발생해도 응답을 반환하도록 수정
            try:
                error_response = create_standard_response(
                    status="error",
                    message=f"요청 처리 중 오류가 발생했습니다: {str(e)}",
                    error_code="MIDDLEWARE_ERROR"
                )
                return web.json_response(error_response, status=500)
            except Exception as response_error:
                logging.error(f"[미들웨어] 응답 생성 실패: {response_error}", exc_info=True)
                # 최후의 수단: 간단한 텍스트 응답
                return web.Response(text=f"Internal Server Error: {str(e)}", status=500)
    
    # Rate Limiting 미들웨어를 먼저 등록 (가장 먼저 실행)
    app.middlewares.append(rate_limit_middleware)
    app.middlewares.append(stats_middleware)
    
    # 대시보드 HTML 서빙 (루트 경로는 먼저 등록)
    async def dashboard_handler(request: web.Request):
        """AIVIS 루트 경로 - 프론트엔드로 리다이렉트"""
        try:
            # 프론트엔드 주소로 리다이렉트
            frontend_url = "https://localhost:5173"
            logging.info(f"[루트] 프론트엔드로 리다이렉트: {frontend_url}")
            return web.Response(
                status=302,
                headers={'Location': frontend_url}
            )
        except Exception as e:
            logging.error(f"[루트] 리다이렉트 오류: {e}", exc_info=True)
            error_html = f"""<!DOCTYPE html>
<html><head><title>AIVIS Backend API</title></head>
<body>
<h1>AIVIS Backend API Server</h1>
<p>백엔드 API 서버가 실행 중입니다.</p>
<p>프론트엔드: <a href="https://localhost:5173">https://localhost:5173</a></p>
<p>API 문서: <a href="/api/health">/api/health</a></p>
</body></html>"""
            return web.Response(text=error_html, content_type='text/html')
    
    # 루트 경로와 /dashboard 경로 모두 대시보드 제공 (먼저 등록)
    app.router.add_get("/", dashboard_handler)
    app.router.add_get("/dashboard", dashboard_handler)
    
    # 웹소켓 엔드포인트 (클라이언트용 - 프레임 수신 및 처리)
    app.router.add_get("/ws", websocket_handler)
    
    # 대시보드 전용 웹소켓 엔드포인트 (데이터만 받기)
    app.router.add_get("/ws/dashboard", dashboard_websocket_handler)

    # MJPEG 스트림 엔드포인트 (개선: 부드러운 스트리밍을 위한 버퍼링 및 최적화)
    async def mjpeg_stream(request: web.Request):
        """MJPEG 스트림 핸들러 - 원본 또는 처리된 비디오 스트림"""
        cam_id = 0
        processed = False  # 기본값: 원본 영상
        try:
            global latest_frames, frame_lock
            
            # cam_id 파라미터 안전하게 파싱
            try:
                cam_id = int(request.query.get('cam_id', '0'))
                # processed 파라미터 확인 (상세 화면용 처리된 영상)
                processed = request.query.get('processed', 'false').lower() == 'true'
            except (ValueError, TypeError) as e:
                cam_id = 0
                logging.warning(f"잘못된 cam_id 파라미터, 기본값 0 사용: {e}")
            
            # 원본 프레임을 사용하는 경우 get_camera_buffer 함수 미리 import
            get_camera_buffer = None
            if not processed:
                try:
                    from camera_worker import get_camera_buffer as _get_camera_buffer
                    get_camera_buffer = _get_camera_buffer
                except ImportError as e:
                    logging.error(f"[스트림] get_camera_buffer import 실패: {e}", exc_info=True)
                    return web.Response(
                        status=500,
                        text=f"Failed to import get_camera_buffer: {str(e)}",
                        content_type='text/plain'
                    )
            
            response = web.StreamResponse(
                status=200,
                reason='OK',
                headers={
                    'Content-Type': 'multipart/x-mixed-replace; boundary=--jpgboundary',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'Connection': 'keep-alive',  # 연결 유지
                    'X-Accel-Buffering': 'no'  # nginx 버퍼링 비활성화
                }
            )
            
            # 응답 준비 시도 (클라이언트가 이미 연결을 끊었을 수 있음)
            try:
                await response.prepare(request)
            except (ConnectionResetError, ConnectionError, OSError, asyncio.CancelledError) as e:
                # 클라이언트가 연결을 끊은 경우 정상적인 동작이므로 DEBUG 레벨로만 로깅
                return response  # 빈 응답 반환
            except Exception as e:
                # 예상치 못한 오류만 WARNING으로 로깅
                logging.warning(f"스트림 CAM-{cam_id} 연결 준비 중 예상치 못한 오류: {e}", exc_info=True)
                return response  # 빈 응답 반환
        except Exception as e:
            # 함수 시작 부분에서 발생하는 모든 예외 처리
            logging.error(f"[스트림] 핸들러 초기화 오류 (CAM-{cam_id}): {e}", exc_info=True)
            try:
                return web.Response(
                    status=500, 
                    text=f"Stream initialization error: {str(e)}",
                    content_type='text/plain'
                )
            except:
                # 응답 생성도 실패하면 빈 응답 반환
                return web.Response(status=500)

        # 프레임이 없을 때 보여줄 기본 이미지 생성 (한 번만)
        try:
            import numpy as np
            default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(default_frame, "Waiting for camera feed...", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, default_frame_bytes = cv2.imencode('.jpg', default_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if default_frame_bytes is None:
                raise ValueError("기본 프레임 인코딩 실패")
            default_frame_bytes = default_frame_bytes.tobytes()
        except Exception as e:
            logging.error(f"기본 프레임 생성 실패 (CAM-{cam_id}): {e}", exc_info=True)
            # 최소한의 에러 응답 생성
            error_msg = b"Stream Error"
            return web.Response(
                status=500,
                body=error_msg,
                content_type='text/plain',
                headers={'Cache-Control': 'no-cache'}
            )

        # 스트리밍 최적화 변수 (단순화된 로직)
        last_frame_bytes = None  # 마지막 프레임 버퍼
        target_fps = 30  # 30 FPS (버벅거림 방지, 안정성 우선)
        frame_interval = 1.0 / target_fps  # 약 0.0222초

        try:
            while True:
                frame_bytes = None
                
                try:
                    if processed:
                        # 처리된 프레임 사용 (상세 화면용)
                        with frame_lock:
                            frame_bytes = latest_frames.get(cam_id)
                            if not frame_bytes:
                                # 처리된 프레임이 없으면 원본 프레임으로 폴백
                                logging.warning(f"[스트림] 처리된 프레임 없음 (CAM-{cam_id}), 원본 프레임으로 폴백. latest_frames 키: {list(latest_frames.keys())}")
                                if get_camera_buffer is not None:
                                    try:
                                        buffer = get_camera_buffer(cam_id)
                                        if buffer and buffer.get("latest_frame") is not None:
                                            frame = buffer["latest_frame"]
                                            ret, buffer_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                            if ret:
                                                frame_bytes = buffer_jpeg.tobytes()
                                    except Exception as e:
                                        logging.warning(f"폴백 원본 프레임 가져오기 실패 (CAM-{cam_id}): {e}")
                            else:
                                # 처리된 프레임이 있으면 그대로 사용
                                pass
                    else:
                        # 원본 프레임 사용 (메인 화면용)
                        if get_camera_buffer is not None:
                            try:
                                buffer = get_camera_buffer(cam_id)
                                if buffer and buffer.get("latest_frame") is not None:
                                    # 원본 프레임을 JPEG로 인코딩
                                    frame = buffer["latest_frame"]
                                    frame_w = buffer.get("frame_width", frame.shape[1])
                                    frame_h = buffer.get("frame_height", frame.shape[0])
                                    ret, buffer_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                    if ret:
                                        frame_bytes = buffer_jpeg.tobytes()
                            except Exception as e:
                                # 실패하면 처리된 프레임으로 폴백
                                with frame_lock:
                                    frame_bytes = latest_frames.get(cam_id)
                        else:
                            # get_camera_buffer가 없으면 처리된 프레임으로 폴백
                            with frame_lock:
                                frame_bytes = latest_frames.get(cam_id)
                except Exception as e:
                    logging.warning(f"[스트림] 프레임 가져오기 오류 (CAM-{cam_id}): {e}", exc_info=True)
                    # 오류 발생 시 기본 프레임 사용
                    frame_bytes = None

                # 프레임 전송: 최신 프레임이 있으면 사용, 없으면 이전 프레임 유지
                if frame_bytes:
                    last_frame_bytes = frame_bytes
                elif processed and not last_frame_bytes:
                    # 처리된 프레임을 요청했는데 없고, 이전 프레임도 없으면 원본 프레임으로 즉시 폴백
                    if get_camera_buffer is not None:
                        try:
                            buffer = get_camera_buffer(cam_id)
                            if buffer and buffer.get("latest_frame") is not None:
                                frame = buffer["latest_frame"]
                                frame_w = buffer.get("frame_width", frame.shape[1])
                                frame_h = buffer.get("frame_height", frame.shape[0])
                                ret, buffer_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                if ret:
                                    frame_bytes = buffer_jpeg.tobytes()
                                    last_frame_bytes = frame_bytes
                        except Exception as e:
                            logging.debug(f"폴백 원본 프레임 처리 실패 (CAM-{cam_id}): {e}")
                
                # 전송할 프레임 결정
                send_frame = last_frame_bytes if last_frame_bytes else default_frame_bytes
                
                try:
                    await response.write(
                        b'--jpgboundary\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(send_frame)).encode() + b'\r\n\r\n' +
                        send_frame + b'\r\n'
                    )
                except (ConnectionResetError, ConnectionError, OSError, asyncio.CancelledError):
                    break
                except Exception as e:
                    logging.warning(f"스트림 CAM-{cam_id} 프레임 전송 중 오류: {e}")
                    break

                # 일정한 프레임 간격 유지
                await asyncio.sleep(frame_interval)
                
        except (ConnectionResetError, ConnectionError, OSError, asyncio.CancelledError) as e:
            # 클라이언트 연결 종료는 정상적인 동작이므로 DEBUG 레벨로만 로깅
            logging.debug(f"스트림 CAM-{cam_id} 클라이언트 연결 종료: {e}")
        except Exception as e:
            # 예상치 못한 오류만 WARNING으로 로깅
            logging.warning(f"스트림 CAM-{cam_id} 처리 중 예상치 못한 오류: {e}", exc_info=True)
        finally:
            # 연결 종료 시 정리 작업 (필요한 경우)
            try:
                if not response._eof_sent:
                    await response.write_eof()
            except:
                pass
        
        # 명시적으로 response 반환 (Missing return statement 에러 방지)
        return response

    # MJPEG 스트림 엔드포인트 (대시보드용)
    app.router.add_get("/stream", mjpeg_stream)
    app.router.add_get("/api/stream", mjpeg_stream)  # 프론트엔드 호환성
    
    # 카메라 썸네일 엔드포인트 (대시보드용)
    async def camera_preview_handler(request: web.Request):
        """카메라 썸네일 반환 (대시보드용)"""
        try:
            from camera_worker import get_camera_thumbnail
            
            cam_id = int(request.query.get('cam_id', 0))
            thumbnail_bytes = get_camera_thumbnail(cam_id)
            
            if thumbnail_bytes:
                return web.Response(
                    body=thumbnail_bytes,
                    content_type='image/jpeg',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    }
                )
            else:
                # 썸네일이 아직 준비되지 않았을 때 204 No Content 반환 (빈 응답)
                return web.Response(status=204)
        except ValueError as e:
            logging.warning(f"썸네일 조회 오류 (잘못된 cam_id): {e}")
            return web.Response(status=400, text=f"잘못된 카메라 ID: {e}")
        except Exception as e:
            logging.error(f"썸네일 조회 오류: {e}", exc_info=True)
            return web.Response(status=500, text=str(e))
    
    app.router.add_get("/api/camera-preview", camera_preview_handler)
    
    # API 엔드포인트 (final과 동일)
    app.router.add_get('/api/status', api_status_handler)
    app.router.add_get('/api/cameras', api_cameras_handler)
    app.router.add_get('/api/model-results', api_model_results_handler)
    app.router.add_get('/api/camera-results', api_camera_results_handler)  # HTTP 폴링용
    app.router.add_get('/api/health', api_health_handler)
    app.router.add_get('/api/violations', api_violations_handler)
    app.router.add_get('/api/violations/stats', api_statistics_handler)  # 프론트엔드 호환성
    app.router.add_put('/api/violations/update-status', api_update_violation_status_handler)  # 위반 사항 상태 업데이트
    app.router.add_get('/api/workers', api_workers_handler)
    app.router.add_post('/api/workers', api_workers_post_handler)  # 작업자 생성
    app.router.add_put('/api/workers/{workerId}', api_workers_put_handler)  # 작업자 업데이트
    app.router.add_delete('/api/workers/{workerId}', api_workers_delete_handler)  # 작업자 삭제
    app.router.add_get('/api/statistics', api_statistics_handler)
    app.router.add_get('/api/fps', api_fps_handler)  # 실시간 FPS 추적 API
    app.router.add_get('/api/performance', api_performance_handler)  # 파이프라인별 성능 통계 API
    app.router.add_get('/api/gpu', api_gpu_handler)  # GPU 사용량 모니터링 API
    
    # 프론트엔드 API 엔드포인트
    app.router.add_get('/api/stats', api_stats_handler)
    app.router.add_get('/api/persons', api_persons_get_handler)
    app.router.add_post('/api/persons', api_persons_post_handler)
    app.router.add_get('/api/access-logs', api_access_logs_handler)
    app.router.add_get('/api/daily-images', api_daily_images_handler)
    app.router.add_post('/api/capture-stream', api_capture_stream_handler)
    app.router.add_post('/api/detect-face-position', api_detect_face_position_handler)
    app.router.add_post('/api/frontend-logs', api_frontend_logs_handler)  # 프론트엔드 로그 수집
    app.router.add_get('/api/violation-image', api_violation_image_handler)  # 위반 이미지 제공

    # CORS 적용
    for route in list(app.router.routes()):
        cors.add(route)

    return app

# --- 메인 함수 ---
def main():
    global safety_system_instance, storage_manager

    setup_logging()
    
    # CPU 스레드 수 최적화 (M4 Pro 14코어 활용)
    import torch
    optimal_threads = min(8, os.cpu_count() or 8)  # M4 Pro: 8-10 스레드가 최적
    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(2)  # 연산 간 병렬화
    logging.info(f"✅ CPU 스레드 최적화: {optimal_threads} 스레드 (총 {os.cpu_count()} 코어)")
    
    # StorageManager 초기화
    try:
        storage_manager = LocalStorageManager()
        logging.info("StorageManager 초기화 완료")
    except Exception as e:
        logging.error(f"StorageManager 초기화 실패: {e}", exc_info=True)
        storage_manager = None

    # 웹 애플리케이션 먼저 생성 (health check가 작동하도록)
    app = create_app()
    
    # SafetySystem 초기화를 백그라운드에서 비동기로 실행 (재시도 로직 포함)
    async def initialize_safety_system():
        # global safety_system_instance  <-- 제거: state 모듈을 직접 사용
        import state # state 모듈 임포트
        
        max_retries = 3
        retry_count = 0
        
        logging.info("=" * 60)
        logging.info("SafetySystem 초기화 시작")
        logging.info("=" * 60)
        
        # 모델 파일 경로 확인 및 로깅
        import config
        logging.info(f"모델 파일 경로 확인:")
        logging.info(f"  - Violation 모델: {config.Paths.YOLO_VIOLATION_MODEL}")
        logging.info(f"  - Pose 모델: {config.Paths.YOLO_POSE_MODEL}")
        logging.info(f"  - Face 모델: {config.Paths.YOLO_FACE_MODEL}")
        
        # 모델 파일 존재 여부 확인
        import os
        violation_exists = os.path.exists(config.Paths.YOLO_VIOLATION_MODEL)
        pose_exists = os.path.exists(config.Paths.YOLO_POSE_MODEL)
        face_exists = os.path.exists(config.Paths.YOLO_FACE_MODEL)
        
        logging.info(f"모델 파일 존재 여부:")
        logging.info(f"  - Violation 모델: {'✅ 존재' if violation_exists else '❌ 없음'}")
        logging.info(f"  - Pose 모델: {'✅ 존재' if pose_exists else '❌ 없음'}")
        logging.info(f"  - Face 모델: {'✅ 존재' if face_exists else '❌ 없음 (선택적)'}")
        
        if not violation_exists or not pose_exists:
            missing = []
            if not violation_exists:
                missing.append(f"Violation 모델: {config.Paths.YOLO_VIOLATION_MODEL}")
            if not pose_exists:
                missing.append(f"Pose 모델: {config.Paths.YOLO_POSE_MODEL}")
            logging.critical("=" * 60)
            logging.critical("❌ 필수 모델 파일이 없습니다!")
            for m in missing:
                logging.critical(f"  - {m}")
            logging.critical("=" * 60)
            with safety_system_lock:
                state.safety_system_instance = None
            return
        
        while retry_count < max_retries:
            try:
                logging.info(f"SafetySystem 초기화 시작... (시도 {retry_count + 1}/{max_retries})")
                init_start_time = time.time()
                new_safety_system = await asyncio.to_thread(core.SafetySystem)
                init_elapsed = time.time() - init_start_time
                logging.info(f"SafetySystem 인스턴스 생성 완료 (소요 시간: {init_elapsed:.2f}초)")
                
                # 락을 사용하여 전역 변수에 안전하게 할당 (멀티스레드 안전성)
                with safety_system_lock:
                    state.safety_system_instance = new_safety_system
                    logging.info(f"🔒 SafetySystem 할당 완료: safety_system_instance={state.safety_system_instance is not None}")
                
                # 모델 초기화 확인 (락 해제 후에도 값이 유지되는지 확인)
                with safety_system_lock:
                    temp_check = state.safety_system_instance
                logging.info(f"🔍 할당 후 확인: temp_check={temp_check is not None}")
                
                if temp_check and temp_check.violation_model and temp_check.pose_model:
                    logging.info("=" * 60)
                    logging.info("✅ SafetySystem 초기화 완료 - 모든 모델 로드됨")
                    logging.info(f"   - Violation 모델: {temp_check.violation_model is not None}")
                    logging.info(f"   - Pose 모델: {temp_check.pose_model is not None}")
                    if temp_check.face_model and temp_check.face_analyzer:
                        logging.info("✅ 얼굴 인식 모델도 로드됨")
                        logging.info(f"   - Face 모델: {temp_check.face_model is not None}")
                        logging.info(f"   - Face Analyzer: {temp_check.face_analyzer is not None}")
                    logging.info("=" * 60)
                    # 최종 확인: 락을 사용하여 다시 한 번 확인
                    with safety_system_lock:
                        final_check = state.safety_system_instance
                        logging.info(f"✅ 최종 확인: safety_system_instance={final_check is not None}")
                    
                    # 모델 Warmup (GPU 예열) - 초기화 직후 실행
                    logging.info("🔥 모델 Warmup 시작 (GPU 초기화 및 최적화)...")
                    try:
                        # 더미 프레임 생성 (832x832, 검은 화면)
                        dummy_frame = np.zeros((832, 832, 3), dtype=np.uint8)
                        
                        # YOLO 모델 Warmup (Half Precision 사용)
                        if temp_check.violation_model:
                            logging.info("  - YOLO Violation Warmup...")
                            temp_check.violation_model(dummy_frame, verbose=False, half=True, device=temp_check.device)
                            
                        if temp_check.pose_model:
                            logging.info("  - YOLO Pose Warmup...")
                            temp_check.pose_model(dummy_frame, verbose=False, half=True, device=temp_check.device)
                            
                        # 얼굴 인식 Warmup
                        if temp_check.face_model:
                            logging.info("  - Face Detection Warmup...")
                            temp_check.face_model(dummy_frame, verbose=False, device=temp_check.device_face)
                            
                        logging.info("✅ 모델 Warmup 완료 - 이제 최대 속도로 처리됩니다.")
                    except Exception as e:
                        logging.warning(f"⚠️ 모델 Warmup 중 오류 (무시됨): {e}")

                    break
                else:
                    missing_models = []
                    if not temp_check:
                        missing_models.append("SafetySystem 인스턴스")
                    if not temp_check or not temp_check.violation_model:
                        missing_models.append("Violation 모델")
                    if not temp_check or not temp_check.pose_model:
                        missing_models.append("Pose 모델")
                    raise RuntimeError(f"필수 모델이 로드되지 않았습니다: {', '.join(missing_models)}")
                    
            except Exception as e:
                retry_count += 1
                logging.error(f"SafetySystem 초기화 실패 (시도 {retry_count}/{max_retries}): {e}", exc_info=True)
                
                if retry_count < max_retries:
                    wait_time = retry_count * 5  # 5초, 10초 대기
                    logging.info(f"{wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                else:
                    logging.critical("=" * 60)
                    logging.critical("SafetySystem 초기화 최종 실패 - 서버는 계속 실행되지만 AI 기능이 비활성화됩니다.")
                    logging.critical("=" * 60)
                    with safety_system_lock:
                        safety_system_instance = None  # 실패 상태 명시

    # 포트 설정 (환경 변수 또는 기본값 사용, 기본값 8081)
    port = int(os.getenv('SERVER_PORT', '8081'))

    # 메인 서버 시작 (대시보드 포함, 단일 포트에서 모든 서비스 제공)
    async def start_server():
        # 메인 이벤트 루프 참조 저장 (임베딩 알림용)
        global _main_event_loop
        _main_event_loop = asyncio.get_event_loop()
        logging.info("[SERVER] 메인 이벤트 루프 참조 저장 완료")
        
        # 종료 신호 처리 (Windows 호환)
        shutdown_event = asyncio.Event()
        
        def signal_handler(sig, frame):
            logging.info("종료 신호 수신 (Ctrl+C). 서버 종료 중...")
            # 이벤트 루프에서 종료 이벤트 설정
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(shutdown_event.set)
            except RuntimeError:
                # 이벤트 루프가 없으면 직접 종료
                shutdown_event.set()
        
        # Windows에서는 CTRL_C_EVENT와 CTRL_BREAK_EVENT 처리
        if os.name == 'nt':  # Windows
            # Windows에서는 signal.CTRL_C_EVENT를 사용
            signal.signal(signal.SIGINT, signal_handler)
            # SIGBREAK도 처리 (Ctrl+Break)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, signal_handler)
        else:
            # Unix/Linux에서는 SIGINT와 SIGTERM 모두 처리
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        try:
            # 환경 변수 확인 로깅
            env_port = os.getenv('SERVER_PORT', '기본값(8081)')
            logging.info(f"환경 변수 SERVER_PORT: {env_port}")
            logging.info(f"서버 시작 중... (포트: {port})")
            runner = web.AppRunner(app)
            await runner.setup()

            # SSL 컨텍스트 생성 (개발 환경용 자체 서명 인증서)
            ssl_context = None
            enable_ssl = os.getenv('ENABLE_SSL', 'true').lower() == 'true'

            if enable_ssl:
                try:
                    import ssl
                    import subprocess
                    # 자체 서명 인증서 생성을 위한 임시 SSL context (개발 환경용)
                    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    # SSL 인증서 경로 (없으면 None으로 유지하여 HTTP만 사용)
                    cert_file = os.path.join(os.path.dirname(__file__), 'cert.pem')
                    key_file = os.path.join(os.path.dirname(__file__), 'key.pem')

                    if os.path.exists(cert_file) and os.path.exists(key_file):
                        ssl_context.load_cert_chain(cert_file, key_file)
                        logging.info("✅ SSL 인증서 로드 완료 - HTTPS/WSS 활성화")
                    else:
                        # 인증서가 없으면 자동 생성 시도
                        logging.warning("⚠️ SSL 인증서 파일이 없습니다. 자동 생성 시도...")
                        try:
                            # OpenSSL이 설치되어 있는지 확인
                            result = subprocess.run(['openssl', 'version'], capture_output=True, text=True, timeout=5)
                            if result.returncode == 0:
                                # 인증서 생성
                                logging.info("OpenSSL 발견. 인증서 생성 중...")
                                cert_dir = os.path.dirname(__file__)
                                subprocess.run([
                                    'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
                                    '-keyout', key_file,
                                    '-out', cert_file,
                                    '-days', '365',
                                    '-nodes',
                                    '-subj', '/C=KR/ST=Seoul/L=Seoul/O=AIVIS/OU=Development/CN=localhost'
                                ], cwd=cert_dir, check=True, timeout=30)
                                
                                if os.path.exists(cert_file) and os.path.exists(key_file):
                                    ssl_context.load_cert_chain(cert_file, key_file)
                                    logging.info("✅ SSL 인증서 자동 생성 완료 - HTTPS/WSS 활성화")
                                    logging.info("⚠️ 주의: 자체 서명 인증서입니다. 브라우저에서 '고급' -> 'localhost로 이동'을 클릭하여 신뢰하세요.")
                                else:
                                    logging.warning("⚠️ 인증서 생성 실패. HTTP/WS만 사용합니다.")
                                    ssl_context = None
                            else:
                                logging.warning("⚠️ OpenSSL을 찾을 수 없습니다. HTTP/WS만 사용합니다.")
                                logging.warning(f"   인증서 수동 생성 방법: ")
                                logging.warning(f"   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
                                ssl_context = None
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                            logging.warning(f"⚠️ 인증서 자동 생성 실패: {e}. HTTP/WS만 사용합니다.")
                            logging.warning(f"   인증서 수동 생성 방법: ")
                            logging.warning(f"   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
                            logging.warning(f"   또는 generate_ssl_cert.bat 스크립트 실행")
                            ssl_context = None
                except Exception as e:
                    logging.warning(f"⚠️ SSL 설정 실패: {e}. HTTP/WS만 사용합니다.")
                    ssl_context = None

            # 포트 바인딩 시도
            try:
                site = web.TCPSite(runner, '0.0.0.0', port, ssl_context=ssl_context)
                await site.start()
                logging.info(f"포트 {port} 바인딩 성공")
            except OSError as e:
                if "Address already in use" in str(e) or "address already in use" in str(e).lower():
                    logging.error(f"포트 {port}가 이미 사용 중입니다. 다른 포트를 사용하거나 기존 프로세스를 종료하세요.")
                    logging.error(f"포트 사용 확인: lsof -i :{port} 또는 netstat -an | grep {port}")
                else:
                    logging.error(f"포트 바인딩 실패: {e}")
                raise
            
            # 서버가 완전히 시작될 때까지 짧은 대기 (health check가 즉시 응답 가능하도록)
            await asyncio.sleep(1)

            # SSL 활성화 여부에 따라 프로토콜 표시
            http_protocol = "https" if ssl_context else "http"
            ws_protocol = "wss" if ssl_context else "ws"

            logging.info(f"✅ 서버 시작 완료: {http_protocol}://0.0.0.0:{port}")
            logging.info(f"  - 대시보드: {http_protocol}://0.0.0.0:{port}/")
            logging.info(f"  - WebSocket (클라이언트): {ws_protocol}://0.0.0.0:{port}/ws")
            logging.info(f"  - WebSocket (대시보드): {ws_protocol}://0.0.0.0:{port}/ws/dashboard")
            logging.info(f"  - MJPEG Stream: {http_protocol}://0.0.0.0:{port}/stream")
            logging.info(f"  - Health Check: {http_protocol}://0.0.0.0:{port}/api/health")
            logging.info("Health check 엔드포인트가 즉시 사용 가능합니다 (503 응답 = 초기화 중, 200 응답 = 준비 완료)")
            
            # 초기 GPU 상태 출력
            if torch.cuda.is_available():
                logging.info("초기 GPU 상태 확인 중...")
                log_gpu_optimization_recommendations(stats_lock, system_stats, _DEFAULT_FACE_WORKERS, _DEFAULT_YOLO_WORKERS)
        except Exception as e:
            logging.critical(f"서버 시작 실패: {e}", exc_info=True)
            raise

        # SafetySystem 초기화를 백그라운드에서 시작 (서버는 이미 실행 중)
        # 즉시 시작하여 모델 로딩 시간 최소화
        init_task = asyncio.create_task(initialize_safety_system())
        
        # 백그라운드 카메라 워커 시작 (SafetySystem 초기화 후)
        async def start_camera_workers():
            """SafetySystem 초기화 완료 후 카메라 워커 시작"""
            # SafetySystem 초기화 대기 (락을 사용하여 안전하게 확인)
            max_wait = 300  # 최대 5분 대기
            wait_count = 0
            
            # state 모듈을 통해 상태 확인 (변수 동기화 보장)
            import state
            
            while wait_count < max_wait:
                with safety_system_lock:
                    # state.safety_system_instance를 확인해야 정확함
                    if state.safety_system_instance is not None:
                        break
                await asyncio.sleep(1)
                wait_count += 1
                if wait_count % 10 == 0:
                    logging.info(f"카메라 워커 시작 대기 중... ({wait_count}초 경과)")
            
            # 최종 확인 (락 사용)
            with safety_system_lock:
                final_check = state.safety_system_instance
                logging.info(f"🎥 카메라 워커 시작 전 확인: safety_system_instance={final_check is not None}")
                if final_check is None:
                    logging.error("SafetySystem 초기화 실패 - 카메라 워커를 시작할 수 없습니다")
                    return
            
            logging.info("✅ SafetySystem 초기화 완료 - 카메라 워커 시작")
            
            # MongoDB 연결 확인
            db_service = None
            try:
                from database import get_database
                db = get_database()
                if db and db.is_connected():
                    # DatabaseService 래퍼에서 실제 서비스 가져오기
                    if hasattr(db, 'db_service'):
                        db_service = db.db_service
                    logging.info("✅ MongoDB 연결 확인 - 위반 사항 저장 가능")
                else:
                    logging.warning("⚠️ MongoDB 연결 실패 - 위반 사항 저장 불가")
            except Exception as e:
                logging.warning(f"⚠️ MongoDB 연결 확인 실패: {e}")
            
            # 카메라 소스 설정 (환경 변수 또는 기본값)
            # Mac에서는 사용 가능한 카메라만 시작
            import platform
            camera_sources = {}
            
            # OpenCV 경고 억제 (한 번만 설정)
            os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
            import warnings
            warnings.filterwarnings('ignore')
            
            # 카메라 0 설정
            cam0_source = os.getenv('CAMERA_0_SOURCE', '0')
            if platform.system() == 'Darwin':
                # Mac에서 카메라 0 확인
                test_cap = cv2.VideoCapture(int(cam0_source), cv2.CAP_AVFOUNDATION)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    if ret:
                        camera_sources[0] = cam0_source
                        logging.info(f"✅ Camera 0 사용 가능: {cam0_source}")
                    else:
                        logging.warning(f"⚠️ Camera 0 ({cam0_source}) 열렸지만 프레임 읽기 실패 (권한 문제일 수 있음)")
                else:
                    logging.warning(f"⚠️ Camera 0 ({cam0_source}) 사용 불가")
                    logging.warning(f"   → Mac 카메라 권한이 필요할 수 있습니다 (시스템 설정 > 개인 정보 보호 및 보안 > 카메라)")
                test_cap.release()
            else:
                camera_sources[0] = cam0_source
            
            # 카메라 1 설정 (2대 카메라 강제 사용)
            cam1_source = os.getenv('CAMERA_1_SOURCE', '1')
            if platform.system() == 'Darwin':
                # Mac에서 카메라 1 확인
                test_cap = cv2.VideoCapture(int(cam1_source), cv2.CAP_AVFOUNDATION)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    if ret:
                        camera_sources[1] = cam1_source
                        logging.info(f"✅ Camera 1 사용 가능: {cam1_source}")
                    else:
                        logging.warning(f"⚠️ Camera 1 ({cam1_source}) 열렸지만 프레임 읽기 실패 (권한 문제일 수 있음)")
                        logging.warning(f"   → 카메라 2대 설정 모드: Camera 1이 필요합니다")
                else:
                    logging.warning(f"⚠️ Camera 1 ({cam1_source}) 사용 불가")
                    logging.warning(f"   → 카메라 2대 설정 모드: Camera 1이 필요합니다")
                test_cap.release()
            else:
                camera_sources[1] = cam1_source
            
            # 사용 가능한 카메라 확인 및 로깅
            if not camera_sources:
                logging.error("=" * 60)
                logging.error("❌ 사용 가능한 카메라가 없습니다!")
                logging.error("=" * 60)
                if platform.system() == 'Darwin':
                    logging.error("")
                    logging.error("🔒 Mac 카메라 권한 문제일 수 있습니다!")
                    logging.error("")
                    logging.error("💡 해결 방법:")
                    logging.error("   1. 시스템 설정 열기: 􀆔 (Command) + Space → '시스템 설정'")
                    logging.error("   2. 개인 정보 보호 및 보안 > 카메라로 이동")
                    logging.error("   3. 다음 앱들에 카메라 권한 부여:")
                    logging.error("      - Python")
                    logging.error("      - Terminal (터미널)")
                    logging.error("      - 또는 실행 중인 IDE (VS Code, PyCharm 등)")
                    logging.error("")
                    logging.error("   4. 권한 부여 후 서버를 재시작하세요")
                    logging.error("")
                    logging.error("   또는 터미널에서 다음 명령으로 확인:")
                    logging.error("   python3 check_cameras_mac.py")
                    logging.error("")
                else:
                    logging.error("💡 카메라가 연결되어 있는지 확인하거나 check_cameras_mac.py를 실행하세요")
                logging.error("=" * 60)
                logging.error("❌ 사용 가능한 카메라가 없어 카메라 워커를 시작할 수 없습니다.")
                return
            elif len(camera_sources) < 2:
                logging.warning("=" * 60)
                logging.warning(f"⚠️ 카메라 1대만 사용 가능합니다 (권장: 2대)")
                logging.warning(f"   현재 사용 가능한 카메라: {len(camera_sources)}개")
                if 0 in camera_sources:
                    logging.warning(f"   - Camera 0: {camera_sources[0]} ✅")
                else:
                    logging.warning(f"   - Camera 0: ❌ 사용 불가")
                if 1 in camera_sources:
                    logging.warning(f"   - Camera 1: {camera_sources[1]} ✅")
                else:
                    logging.warning(f"   - Camera 1: ❌ 사용 불가")
                logging.warning("   → 사용 가능한 카메라만 시작합니다")
                logging.warning("=" * 60)
            else:
                logging.info(f"✅ 사용 가능한 카메라: {len(camera_sources)}개")
                if 0 in camera_sources:
                    logging.info(f"   - Camera 0: {camera_sources[0]} ✅")
                if 1 in camera_sources:
                    logging.info(f"   - Camera 1: {camera_sources[1]} ✅")
            
            # 카메라 워커 시작 (사용 가능한 카메라만 시작)
            
            try:
                from camera_worker import camera_worker
                
                # 락을 사용하여 safety_system_instance를 안전하게 읽기
                with safety_system_lock:
                    # state 모듈에서 직접 읽어야 최신 값을 가져올 수 있음 (from ... import로 가져온 변수는 업데이트 안 됨)
                    current_safety_system = state.safety_system_instance
                    is_none = current_safety_system is None
                    logging.info(f"🎥 카메라 워커 시작 ({len(camera_sources)}대): safety_system_instance={not is_none}, 타입={type(current_safety_system).__name__ if current_safety_system is not None else 'None'}")
                
                if current_safety_system is None:
                    logging.error("❌ SafetySystem이 None입니다! 카메라 워커를 시작할 수 없습니다.")
                    return
                
                camera_list = ", ".join([f"Camera {cam_id}={source}" for cam_id, source in camera_sources.items()])
                logging.info(f"✅ 카메라 {len(camera_sources)}대 설정 완료: {camera_list}")
                for cam_id, source in camera_sources.items():
                    logging.info(f"🎥 Camera {cam_id} 워커 시작 (소스: {source}, SafetySystem 전달: {current_safety_system is not None})")
                    asyncio.create_task(
                        camera_worker(
                            cam_id=cam_id,
                            camera_source=source,
                            safety_system=current_safety_system,  # 락으로 읽은 값 사용
                            storage_manager=storage_manager,
                            db_service=db_service,
                            fps=60.0  # 60 FPS (성능 최대화, 프레임 우선 처리)
                        )
                    )
                
                logging.info("✅ 모든 카메라 워커 시작 완료")
                
                # 적응형 워커 관리자 백그라운드 작업 시작 (10초마다 워커 수 조정)
                async def adaptive_worker_adjustment_task():
                    """적응형 워커 조정 백그라운드 작업"""
                    try:
                        from state import adaptive_worker_manager, update_worker_executors
                        if adaptive_worker_manager:
                            # 초기 대기 (서버 안정화 대기)
                            await asyncio.sleep(30.0)
                            while True:
                                await asyncio.sleep(30.0)  # 30초마다 조정 (Executor 교체 빈도 감소)
                                try:
                                    update_worker_executors()
                                except Exception as e:
                                    logging.warning(f"적응형 워커 조정 실패 (무시): {e}")
                    except Exception as e:
                        logging.warning(f"적응형 워커 조정 작업 시작 실패: {e}")
                
                asyncio.create_task(adaptive_worker_adjustment_task())
                logging.info("✅ 적응형 워커 관리자 백그라운드 작업 시작 (30초마다 자동 조정)")
                
                # 위반 사항 배치 처리 백그라운드 태스크 시작
                try:
                    from violation_batch_processor import process_violation_batch
                    asyncio.create_task(process_violation_batch())
                    logging.info("✅ 위반 사항 배치 처리 백그라운드 작업 시작")
                except ImportError as e:
                    logging.warning(f"배치 처리 모듈 import 실패: {e}")
                except Exception as e:
                    logging.warning(f"배치 처리 작업 시작 실패: {e}")
                    
            except ImportError as e:
                logging.error(f"카메라 워커 모듈 import 실패: {e}")
            except Exception as e:
                logging.error(f"카메라 워커 시작 오류: {e}", exc_info=True)
        
        # 카메라 워커 시작 태스크
        asyncio.create_task(start_camera_workers())
        
        # 초기화 상태 모니터링 (주기적으로 확인)
        async def monitor_initialization():
            """SafetySystem 초기화 상태 모니터링"""
            check_count = 0
            while True:
                await asyncio.sleep(5)  # 5초마다 확인
                check_count += 1
                
                if safety_system_instance is None:
                    if check_count <= 6:  # 처음 30초 동안만 상세 로깅
                        logging.warning(f"SafetySystem 초기화 대기 중... ({check_count * 5}초 경과)")
                    elif check_count % 6 == 0:  # 30초마다 로깅
                        logging.warning(f"SafetySystem 초기화가 {check_count * 5}초 이상 걸리고 있습니다. 모델 파일을 확인하세요.")
                elif safety_system_instance.violation_model is None or safety_system_instance.pose_model is None:
                    logging.warning("일부 모델이 로드되지 않았습니다. 모델 파일 경로를 확인하세요.")
                    break  # 모니터링 종료
                else:
                    # 초기화 완료 확인
                    logging.info("✅ SafetySystem 초기화 완료 확인 - AI 처리 가능")
                    break  # 모니터링 종료
        
        asyncio.create_task(monitor_initialization())
        
        # GPU 사용량 주기적 모니터링 (60초마다)
        async def monitor_gpu_usage():
            """주기적으로 GPU 사용량 체크 및 최적화 권장사항 출력"""
            await asyncio.sleep(60)  # 서버 시작 후 60초 대기
            while True:
                try:
                    log_gpu_optimization_recommendations(stats_lock, system_stats, _DEFAULT_FACE_WORKERS, _DEFAULT_YOLO_WORKERS)
                except Exception as e:
                    logging.warning(f"GPU 모니터링 중 오류: {e}")
                await asyncio.sleep(300)  # 5분마다 체크
        
        asyncio.create_task(monitor_gpu_usage())
        
        # 주기적으로 죽은 WebSocket 연결 정리 (30초마다)
        async def cleanup_dead_websockets():
            """주기적으로 죽은 WebSocket 연결 정리"""
            while True:
                await asyncio.sleep(30)  # 30초마다
                try:
                    dead_connections = []
                    # 복사본 사용하여 동시 수정 방지
                    for ws in list(connected_websockets):
                        if ws.closed:
                            dead_connections.append(ws)
                    
                    for ws in dead_connections:
                        connected_websockets.discard(ws)
                        dashboard_websockets.discard(ws)
                    
                    if dead_connections:
                        logging.debug(f"죽은 WebSocket 연결 {len(dead_connections)}개 정리 완료")
                except Exception as e:
                    logging.warning(f"WebSocket 정리 중 오류: {e}", exc_info=True)
        
        # WebSocket 정리 태스크 시작
        asyncio.create_task(cleanup_dead_websockets())
        
        # 주기적으로 대시보드에 모델 결과 브로드캐스트 (1초마다)
        async def periodic_broadcast():
            """주기적으로 대시보드에 모델 결과 전송"""
            while True:
                try:
                    await asyncio.sleep(max(0.05, DASHBOARD_BROADCAST_INTERVAL))
                    await broadcast_model_results()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.debug(f"주기적 브로드캐스트 오류: {e}")
                    await asyncio.sleep(1.0)  # 오류 시 잠시 대기 후 재시도
        
        # 주기적 브로드캐스트 태스크 시작
        asyncio.create_task(periodic_broadcast())

        # 종료 신호 대기 (Ctrl+C로 종료 가능)
        try:
            await shutdown_event.wait()
            logging.info("종료 신호 수신. 서버 종료 시작...")
        except asyncio.CancelledError:
            pass
        
        # 서버 종료
        logging.info("서버 종료 중...")
        try:
            await runner.cleanup()
            logging.info("서버 종료 완료")
        except Exception as e:
            logging.warning(f"서버 종료 중 오류: {e}")

    # 이벤트 루프에서 서버 시작 (Windows Ctrl+C 호환)
    try:
        # Windows에서 Ctrl+C 처리를 위해 ProactorEventLoop 사용
        if os.name == 'nt':  # Windows
            # Windows에서는 ProactorEventLoop가 기본값이지만 명시적으로 설정
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(start_server())
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt 수신. 서버 종료 중...")
            finally:
                loop.close()
        else:
            asyncio.run(start_server())
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt 수신. 서버 종료 중...")
    except Exception as e:
        logging.critical(f"서버 실행 중 치명적 오류: {e}", exc_info=True)
        raise
    finally:
        # 모든 스레드 풀 안전하게 종료
        logging.info("스레드 풀 종료 중...")
        try:
            face_recognition_executor.shutdown(wait=False)  # wait=False로 빠른 종료
        except Exception as e:
            logging.warning(f"face_recognition_executor 종료 중 오류: {e}")
        try:
            yolo_executor.shutdown(wait=False)
        except Exception as e:
            logging.warning(f"yolo_executor 종료 중 오류: {e}")
        try:
            dangerous_behavior_executor.shutdown(wait=False)
        except Exception as e:
            logging.warning(f"dangerous_behavior_executor 종료 중 오류: {e}")
        try:
            frame_processing_executor.shutdown(wait=False)
        except Exception as e:
            logging.warning(f"frame_processing_executor 종료 중 오류: {e}")
        logging.info("모든 리소스 정리 완료")

if __name__ == "__main__":
    main()
