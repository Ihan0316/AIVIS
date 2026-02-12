"""
적응형 워커 관리 시스템
GPU 사용률 및 지연 시간을 모니터링하여 워커 수를 자동 조정
"""
import time
import logging
import threading
from collections import deque
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    avg_processing_time_ms: float = 0.0
    queue_size: int = 0
    gpu_utilization: float = 0.0  # 추정 GPU 사용률 (0-100)
    latency_ms: float = 0.0
    fps: float = 0.0


class AdaptiveWorkerManager:
    """GPU 사용률 및 지연 시간 기반 워커 수 자동 조정"""
    
    def __init__(
        self,
        initial_face_workers: int = 8,
        initial_yolo_workers: int = 10,
        initial_danger_workers: int = 6,
        initial_frame_workers: int = 12,
        min_workers: int = 2,
        max_workers: int = 30,
        adjustment_interval: float = 10.0  # 10초마다 조정
    ):
        self.current_face_workers = initial_face_workers
        self.current_yolo_workers = initial_yolo_workers
        self.current_danger_workers = initial_danger_workers
        self.current_frame_workers = initial_frame_workers
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        
        self.adjustment_interval = adjustment_interval if adjustment_interval > 0 else 30.0
        self.last_adjustment_time = time.time()
        
        # 워커 수 변경 임계값 (너무 자주 변경 방지)
        self.min_change_threshold = 3  # 최소 3개 이상 차이날 때만 변경 (Executor 교체 빈도 감소)
        
        # 성능 메트릭 히스토리
        self.processing_times: deque = deque(maxlen=100)
        self.queue_sizes: deque = deque(maxlen=100)
        self.latencies: deque = deque(maxlen=100)
        
        self.lock = threading.Lock()
        
        # MPS 감지
        self.is_mps = hasattr(torch, "backends") and torch.backends.mps.is_available()
        
        logger.info(f"✅ 적응형 워커 관리자 초기화: Face={initial_face_workers}, YOLO={initial_yolo_workers}, Danger={initial_danger_workers}, Frame={initial_frame_workers}")
    
    def update_metrics(
        self,
        processing_time_ms: float,
        queue_size: int = 0,
        latency_ms: float = 0.0,
        fps: float = 0.0
    ):
        """성능 메트릭 업데이트"""
        with self.lock:
            if processing_time_ms > 0:
                self.processing_times.append(processing_time_ms)
            if queue_size >= 0:
                self.queue_sizes.append(queue_size)
            if latency_ms > 0:
                self.latencies.append(latency_ms)
    
    def _estimate_gpu_utilization(self) -> float:
        """GPU 사용률 추정 (MPS는 처리 시간과 큐 크기로 추정)"""
        if not self.processing_times:
            return 0.0
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        avg_queue_size = sum(self.queue_sizes) / len(self.queue_sizes) if self.queue_sizes else 0
        
        # 처리 시간이 짧고 큐가 비어있으면 GPU 사용률 낮음
        # 처리 시간이 길고 큐가 쌓이면 GPU 사용률 높음 (부하)
        if avg_processing_time < 50:  # 50ms 미만: 빠른 처리
            if avg_queue_size < 2:
                return 30.0  # GPU 여유
            else:
                return 60.0  # 중간 부하
        elif avg_processing_time < 100:  # 100ms 미만
            if avg_queue_size < 5:
                return 60.0  # 중간 부하
            else:
                return 80.0  # 높은 부하
        else:  # 100ms 이상: 느린 처리
            if avg_queue_size > 5:
                return 95.0  # 매우 높은 부하
            else:
                return 75.0  # 높은 부하
    
    def _calculate_optimal_workers(
        self,
        current_workers: int,
        gpu_util: float,
        avg_latency_ms: float,
        avg_queue_size: float,
        target_latency_ms: float = 100.0
    ) -> int:
        """최적 워커 수 계산"""
        # GPU 사용률이 낮고 지연이 높으면 워커 증가
        # GPU 사용률이 높고 지연이 낮으면 워커 감소
        
        # 워커 수 변경 임계값 적용 (너무 자주 변경 방지)
        if gpu_util < 50 and avg_latency_ms > target_latency_ms * 1.5:
            # GPU 여유 + 지연 높음: 워커 증가
            new_workers = min(self.max_workers, current_workers + 2)
            # 최소 변경 임계값 체크
            if abs(new_workers - current_workers) >= self.min_change_threshold:
                logger.info(f"📈 워커 증가: {current_workers} -> {new_workers} (GPU 사용률: {gpu_util:.1f}%, 지연: {avg_latency_ms:.1f}ms)")
                return new_workers
        elif gpu_util < 50 and avg_queue_size > 5:  # 큐 임계값 증가 (3 -> 5)
            # GPU 여유 + 큐 쌓임: 워커 증가
            new_workers = min(self.max_workers, current_workers + 1)
            # 최소 변경 임계값 체크
            if abs(new_workers - current_workers) >= self.min_change_threshold:
                logger.info(f"📈 워커 증가: {current_workers} -> {new_workers} (GPU 사용률: {gpu_util:.1f}%, 큐 크기: {avg_queue_size:.1f})")
                return new_workers
        elif gpu_util > 85 and avg_latency_ms < target_latency_ms * 0.8:
            # GPU 과부하 + 지연 낮음: 워커 감소
            new_workers = max(self.min_workers, current_workers - 1)
            # 최소 변경 임계값 체크
            if abs(new_workers - current_workers) >= self.min_change_threshold:
                logger.info(f"📉 워커 감소: {current_workers} -> {new_workers} (GPU 사용률: {gpu_util:.1f}%, 지연: {avg_latency_ms:.1f}ms)")
                return new_workers
        elif gpu_util > 90:
            # GPU 매우 과부하: 워커 감소
            new_workers = max(self.min_workers, current_workers - 2)
            # 최소 변경 임계값 체크
            if abs(new_workers - current_workers) >= self.min_change_threshold:
                logger.info(f"📉 워커 감소: {current_workers} -> {new_workers} (GPU 사용률: {gpu_util:.1f}%)")
                return new_workers
        
        # 현재 상태 유지
        return current_workers
    
    def adjust_workers(self) -> Tuple[int, int, int, int]:
        """워커 수 자동 조정"""
        current_time = time.time()
        
        # 조정 간격 체크
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return (
                self.current_face_workers,
                self.current_yolo_workers,
                self.current_danger_workers,
                self.current_frame_workers
            )
        
        with self.lock:
            if not self.processing_times:
                return (
                    self.current_face_workers,
                    self.current_yolo_workers,
                    self.current_danger_workers,
                    self.current_frame_workers
                )
            
            # 평균 메트릭 계산
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            avg_queue_size = sum(self.queue_sizes) / len(self.queue_sizes) if self.queue_sizes else 0
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else avg_processing_time
            
            # GPU 사용률 추정
            gpu_util = self._estimate_gpu_utilization()
            
            # 각 워커 타입별 조정
            # YOLO 워커: GPU 부하에 가장 민감
            new_yolo_workers = self._calculate_optimal_workers(
                self.current_yolo_workers,
                gpu_util,
                avg_latency,
                avg_queue_size,
                target_latency_ms=100.0
            )
            
            # Face 워커: YOLO 워커의 70-80% 수준
            new_face_workers = self._calculate_optimal_workers(
                self.current_face_workers,
                gpu_util * 0.8,  # Face는 GPU 부하가 약간 낮음
                avg_latency,
                avg_queue_size,
                target_latency_ms=150.0
            )
            
            # Danger 워커: YOLO 워커의 50-60% 수준
            new_danger_workers = max(
                self.min_workers,
                int(new_yolo_workers * 0.6)
            )
            
            # Frame 워커: CPU 기반이므로 큐 크기에 따라 조정
            if avg_queue_size > 5:
                new_frame_workers = min(self.max_workers, self.current_frame_workers + 2)
            elif avg_queue_size < 2:
                new_frame_workers = max(self.min_workers, self.current_frame_workers - 1)
            else:
                new_frame_workers = self.current_frame_workers
            
            # 변경사항 적용
            changed = False
            if new_yolo_workers != self.current_yolo_workers:
                self.current_yolo_workers = new_yolo_workers
                changed = True
            if new_face_workers != self.current_face_workers:
                self.current_face_workers = new_face_workers
                changed = True
            if new_danger_workers != self.current_danger_workers:
                self.current_danger_workers = new_danger_workers
                changed = True
            if new_frame_workers != self.current_frame_workers:
                self.current_frame_workers = new_frame_workers
                changed = True
            
            if changed:
                logger.info(
                    f"🔄 워커 수 자동 조정 완료: "
                    f"Face={self.current_face_workers}, "
                    f"YOLO={self.current_yolo_workers}, "
                    f"Danger={self.current_danger_workers}, "
                    f"Frame={self.current_frame_workers} "
                    f"(GPU 사용률: {gpu_util:.1f}%, 지연: {avg_latency:.1f}ms, 큐: {avg_queue_size:.1f})"
                )
            
            self.last_adjustment_time = current_time
            
            return (
                self.current_face_workers,
                self.current_yolo_workers,
                self.current_danger_workers,
                self.current_frame_workers
            )
    
    def get_current_workers(self) -> Tuple[int, int, int, int]:
        """현재 워커 수 반환"""
        with self.lock:
            return (
                self.current_face_workers,
                self.current_yolo_workers,
                self.current_danger_workers,
                self.current_frame_workers
            )
    
    def get_metrics(self) -> PerformanceMetrics:
        """현재 성능 메트릭 반환"""
        with self.lock:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
            avg_queue_size = sum(self.queue_sizes) / len(self.queue_sizes) if self.queue_sizes else 0
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else avg_processing_time
            gpu_util = self._estimate_gpu_utilization()
            
            return PerformanceMetrics(
                avg_processing_time_ms=avg_processing_time,
                queue_size=int(avg_queue_size),
                gpu_utilization=gpu_util,
                latency_ms=avg_latency,
                fps=1000.0 / avg_processing_time if avg_processing_time > 0 else 0.0
            )


# 전역 적응형 워커 관리자 인스턴스
_adaptive_worker_manager: Optional[AdaptiveWorkerManager] = None
_manager_lock = threading.Lock()


def get_adaptive_worker_manager() -> Optional[AdaptiveWorkerManager]:
    """적응형 워커 관리자 인스턴스 가져오기"""
    return _adaptive_worker_manager


def initialize_adaptive_worker_manager(
    initial_face_workers: int = 8,
    initial_yolo_workers: int = 10,
    initial_danger_workers: int = 6,
    initial_frame_workers: int = 12
) -> AdaptiveWorkerManager:
    """적응형 워커 관리자 초기화"""
    global _adaptive_worker_manager
    with _manager_lock:
        if _adaptive_worker_manager is None:
            _adaptive_worker_manager = AdaptiveWorkerManager(
                initial_face_workers=initial_face_workers,
                initial_yolo_workers=initial_yolo_workers,
                initial_danger_workers=initial_danger_workers,
                initial_frame_workers=initial_frame_workers
            )
        return _adaptive_worker_manager

