"""
적응형 성능 최적화 시스템
FPS 기반 동적 프레임 스킵, 해상도 조정, 배치 크기 최적화
"""
import time
import random
from collections import deque
from typing import Deque, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    fps: float = 0.0
    processing_time_ms: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    skip_ratio: float = 0.0


class AdaptiveFrameSkipper:
    """시스템 부하에 따른 동적 프레임 스킵"""

    def __init__(self, target_fps: float = 30.0, window_size: int = 30):
        self.target_fps = target_fps
        self.window_size = window_size
        self.fps_history: Deque[float] = deque(maxlen=window_size)
        self.skip_ratio = 0.0
        self.min_skip_ratio = 0.0
        self.max_skip_ratio = 0.5  # 최대 50% 스킵
        self.last_update_time = time.time()

    def update_fps(self, current_fps: float):
        """현재 FPS 업데이트 및 스킵 비율 조정"""
        self.fps_history.append(current_fps)

        # 1초에 한 번만 조정
        now = time.time()
        if now - self.last_update_time < 1.0:
            return

        self.last_update_time = now

        # 평균 FPS 계산
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

        # 스킵 비율 동적 조정
        if avg_fps < self.target_fps * 0.7:  # 70% 미만
            # FPS가 너무 낮음 - 스킵 증가
            self.skip_ratio = min(self.max_skip_ratio, self.skip_ratio + 0.05)
            logger.warning(f"FPS too low ({avg_fps:.1f}/{self.target_fps}), increasing skip ratio to {self.skip_ratio:.2f}")

        elif avg_fps < self.target_fps * 0.85:  # 85% 미만
            # 약간 낮음 - 조금씩 스킵 증가
            self.skip_ratio = min(self.max_skip_ratio, self.skip_ratio + 0.02)

        elif avg_fps > self.target_fps * 0.95:  # 95% 이상
            # FPS가 충분함 - 스킵 감소
            self.skip_ratio = max(self.min_skip_ratio, self.skip_ratio - 0.02)

    def should_skip(self) -> bool:
        """현재 프레임을 스킵해야 하는지 판단"""
        if self.skip_ratio <= 0.0:
            return False

        # 확률적으로 스킵 (랜덤성 추가로 균일한 스킵)
        return random.random() < self.skip_ratio

    def get_metrics(self) -> dict:
        """현재 메트릭 반환"""
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        return {
            'target_fps': self.target_fps,
            'current_fps': avg_fps,
            'skip_ratio': self.skip_ratio,
            'frames_tracked': len(self.fps_history)
        }


class AdaptiveResolution:
    """동적 해상도 조정 (GPU 메모리/성능 기반)"""

    def __init__(self, initial_size: Tuple[int, int] = (1024, 1024)):
        self.current_size = initial_size
        self.min_size = (640, 640)
        self.max_size = (1280, 1280)
        self.available_sizes = [
            (640, 640),
            (832, 832),
            (1024, 1024),
            (1280, 1280)
        ]
        self.performance_history: Deque[float] = deque(maxlen=30)

    def update_performance(self, processing_time_ms: float, target_time_ms: float = 33.0):
        """처리 시간에 따라 해상도 조정"""
        self.performance_history.append(processing_time_ms)

        # 10프레임마다 조정
        if len(self.performance_history) < 10:
            return

        avg_time = sum(self.performance_history) / len(self.performance_history)

        # 처리 시간이 너무 느림 - 해상도 낮춤
        if avg_time > target_time_ms * 1.2:
            self._decrease_resolution()

        # 처리 시간이 충분히 빠름 - 해상도 높임
        elif avg_time < target_time_ms * 0.7:
            self._increase_resolution()

    def _decrease_resolution(self):
        """해상도 낮추기"""
        current_idx = self.available_sizes.index(self.current_size)
        if current_idx > 0:
            new_size = self.available_sizes[current_idx - 1]
            logger.info(f"Decreasing resolution: {self.current_size} -> {new_size}")
            self.current_size = new_size

    def _increase_resolution(self):
        """해상도 높이기"""
        current_idx = self.available_sizes.index(self.current_size)
        if current_idx < len(self.available_sizes) - 1:
            new_size = self.available_sizes[current_idx + 1]
            logger.info(f"Increasing resolution: {self.current_size} -> {new_size}")
            self.current_size = new_size

    def get_current_size(self) -> Tuple[int, int]:
        """현재 해상도 반환"""
        return self.current_size


class PerformanceMonitor:
    """종합 성능 모니터"""

    def __init__(self):
        self.frame_times: Deque[float] = deque(maxlen=100)
        self.last_frame_time = time.time()
        self.total_frames = 0
        self.skipped_frames = 0

    def start_frame(self):
        """프레임 처리 시작"""
        self.last_frame_time = time.time()

    def end_frame(self, skipped: bool = False):
        """프레임 처리 종료"""
        elapsed = time.time() - self.last_frame_time
        self.frame_times.append(elapsed)
        self.total_frames += 1

        if skipped:
            self.skipped_frames += 1

    def get_fps(self) -> float:
        """현재 FPS 계산"""
        if not self.frame_times:
            return 0.0

        avg_time = sum(self.frame_times) / len(self.frame_times)
        if avg_time == 0:
            return 0.0

        return 1.0 / avg_time

    def get_avg_processing_time_ms(self) -> float:
        """평균 처리 시간 (밀리초)"""
        if not self.frame_times:
            return 0.0

        return (sum(self.frame_times) / len(self.frame_times)) * 1000

    def get_skip_rate(self) -> float:
        """스킵 비율"""
        if self.total_frames == 0:
            return 0.0

        return self.skipped_frames / self.total_frames

    def get_metrics(self) -> PerformanceMetrics:
        """종합 메트릭"""
        return PerformanceMetrics(
            fps=self.get_fps(),
            processing_time_ms=self.get_avg_processing_time_ms(),
            skip_ratio=self.get_skip_rate()
        )

    def reset(self):
        """메트릭 리셋"""
        self.frame_times.clear()
        self.total_frames = 0
        self.skipped_frames = 0


class AdaptiveOptimizer:
    """통합 적응형 최적화 시스템"""

    def __init__(self, target_fps: float = 30.0, initial_size: Tuple[int, int] = (1024, 1024)):
        self.frame_skipper = AdaptiveFrameSkipper(target_fps)
        self.resolution_adjuster = AdaptiveResolution(initial_size)
        self.performance_monitor = PerformanceMonitor()

    def should_process_frame(self) -> bool:
        """프레임을 처리해야 하는지 판단"""
        should_skip = self.frame_skipper.should_skip()

        if should_skip:
            self.performance_monitor.end_frame(skipped=True)

        return not should_skip

    def start_frame_processing(self):
        """프레임 처리 시작"""
        self.performance_monitor.start_frame()

    def end_frame_processing(self):
        """프레임 처리 종료 및 메트릭 업데이트"""
        self.performance_monitor.end_frame(skipped=False)

        # FPS 업데이트
        current_fps = self.performance_monitor.get_fps()
        self.frame_skipper.update_fps(current_fps)

        # 해상도 조정
        avg_time_ms = self.performance_monitor.get_avg_processing_time_ms()
        target_time_ms = 1000 / self.frame_skipper.target_fps
        self.resolution_adjuster.update_performance(avg_time_ms, target_time_ms)

    def get_optimal_resolution(self) -> Tuple[int, int]:
        """현재 최적 해상도 반환"""
        return self.resolution_adjuster.get_current_size()

    def get_performance_metrics(self) -> dict:
        """종합 성능 메트릭"""
        metrics = self.performance_monitor.get_metrics()
        return {
            'fps': round(metrics.fps, 1),
            'processing_time_ms': round(metrics.processing_time_ms, 1),
            'skip_ratio': round(metrics.skip_ratio, 2),
            'current_resolution': self.get_optimal_resolution(),
            'target_fps': self.frame_skipper.target_fps
        }
