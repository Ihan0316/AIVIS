# pipeline_manager.py - 2단계 파이프라인 관리
"""
최적화된 2단계 파이프라인 관리 모듈

Step 1: 상시 가동 (매 프레임)
- YOLO Pose (with Tracking): 사람 감지 + 추적 + 쓰러짐 판단
- YOLO PPE: 헬멧/조끼 미착용 감지

Step 2: 조건부 가동 (이벤트 발생 시)
- YOLO Face + AdaFace: 신원 확인
- Trigger 조건:
  - 조건 A: 위반 사항 발생 시 (쓰러짐 OR 안전장비 미착용)
  - 조건 B: 새로운 사람 등장 시 (Track ID가 처음 생성되었을 때)
  - 조건 C: 주기적 확인 (해당 ID에 대해 1초에 1번만)
"""
import logging
import time
from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict
import threading
import numpy as np
import cv2

import config


class PersonTracker:
    """
    사람별 추적 상태 관리
    Track ID별로 얼굴 인식 상태, 위반 상태 등을 관리
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # Track ID별 상태 관리
        # {cam_id: {track_id: PersonState}}
        self._person_states: Dict[int, Dict[int, 'PersonState']] = defaultdict(dict)
        
        # 최근에 처음 등장한 Track ID (새로운 사람)
        # {cam_id: {track_id: first_seen_time}}
        self._new_tracks: Dict[int, Dict[int, float]] = defaultdict(dict)
        
        # Track ID별 마지막 얼굴 인식 시간
        # {cam_id: {track_id: last_face_recognition_time}}
        self._last_face_recognition: Dict[int, Dict[int, float]] = defaultdict(dict)
        
        # 얼굴 인식 주기 (초)
        self.face_recognition_interval = 1.0  # 1초에 1번
        
        # 새로운 사람 판단 시간 (초) - 이 시간 내에 등장한 Track ID는 새로운 사람으로 간주
        self.new_person_threshold = 2.0
        
        # 만료 시간 (초) - 이 시간 동안 업데이트되지 않은 Track ID는 삭제
        self.expiry_time = 30.0
    
    def update_track(self, cam_id: int, track_id: int, 
                     person_box: Tuple[int, int, int, int],
                     has_violation: bool = False,
                     violation_types: Optional[List[str]] = None) -> 'PersonState':
        """
        Track ID 상태 업데이트
        
        Args:
            cam_id: 카메라 ID
            track_id: 추적 ID
            person_box: 사람 바운딩 박스 (x1, y1, x2, y2)
            has_violation: 위반 사항 발생 여부
            violation_types: 위반 유형 리스트
            
        Returns:
            PersonState: 업데이트된 사람 상태
        """
        with self.lock:
            now = time.time()
            
            # 기존 상태 확인 또는 새로 생성
            if track_id not in self._person_states[cam_id]:
                # 새로운 Track ID
                state = PersonState(track_id, person_box, now)
                self._person_states[cam_id][track_id] = state
                self._new_tracks[cam_id][track_id] = now
            else:
                state = self._person_states[cam_id][track_id]
                state.update(person_box, now)
            
            # 위반 상태 업데이트
            if has_violation and violation_types:
                state.add_violation(violation_types, now)
            
            return state
    
    def should_recognize_face(self, cam_id: int, track_id: int, 
                              has_violation: bool = False) -> Tuple[bool, str]:
        """
        해당 Track ID에 대해 얼굴 인식을 수행해야 하는지 판단
        
        Args:
            cam_id: 카메라 ID
            track_id: 추적 ID
            has_violation: 위반 사항 발생 여부
            
        Returns:
            (should_recognize, reason): 얼굴 인식 수행 여부와 이유
        """
        with self.lock:
            now = time.time()
            
            # Track ID가 없으면 인식하지 않음
            if track_id not in self._person_states[cam_id]:
                return False, "unknown_track"
            
            state = self._person_states[cam_id][track_id]
            
            # 조건 A: 위반 사항 발생 시 (쓰러짐 OR 안전장비 미착용)
            if has_violation:
                # 위반 발생 시에는 무조건 얼굴 인식 시도 (누가 위반했는지 기록)
                self._last_face_recognition[cam_id][track_id] = now
                return True, "violation_detected"
            
            # 조건 B: 새로운 사람 등장 시 (Track ID가 처음 생성되었을 때)
            if track_id in self._new_tracks[cam_id]:
                first_seen = self._new_tracks[cam_id][track_id]
                if now - first_seen < self.new_person_threshold:
                    # 새로운 사람이고 아직 얼굴 인식이 안 된 경우
                    if state.recognized_name is None:
                        self._last_face_recognition[cam_id][track_id] = now
                        return True, "new_person"
            
            # 조건 C: 주기적 확인 (해당 ID에 대해 1초에 1번만)
            last_recognition = self._last_face_recognition[cam_id].get(track_id, 0)
            if now - last_recognition >= self.face_recognition_interval:
                # 아직 인식되지 않은 경우에만 주기적 인식
                if state.recognized_name is None or state.recognition_confidence < 0.5:
                    self._last_face_recognition[cam_id][track_id] = now
                    return True, "periodic_check"
            
            return False, "no_need"
    
    def set_recognized(self, cam_id: int, track_id: int, 
                       name: str, confidence: float) -> None:
        """
        얼굴 인식 결과 저장
        
        Args:
            cam_id: 카메라 ID
            track_id: 추적 ID
            name: 인식된 이름
            confidence: 인식 신뢰도
        """
        with self.lock:
            if track_id in self._person_states[cam_id]:
                state = self._person_states[cam_id][track_id]
                state.set_recognized(name, confidence)
                
                # 인식 성공 시 새로운 사람 목록에서 제거
                if name != "Unknown" and track_id in self._new_tracks[cam_id]:
                    del self._new_tracks[cam_id][track_id]
    
    def get_state(self, cam_id: int, track_id: int) -> Optional['PersonState']:
        """Track ID의 상태 가져오기"""
        with self.lock:
            return self._person_states[cam_id].get(track_id)
    
    def get_all_states(self, cam_id: int) -> Dict[int, 'PersonState']:
        """해당 카메라의 모든 사람 상태 가져오기"""
        with self.lock:
            return dict(self._person_states[cam_id])
    
    def cleanup_expired(self, cam_id: int) -> None:
        """만료된 Track ID 정리"""
        with self.lock:
            now = time.time()
            expired_tracks = []
            
            for track_id, state in self._person_states[cam_id].items():
                if now - state.last_update > self.expiry_time:
                    expired_tracks.append(track_id)
            
            for track_id in expired_tracks:
                del self._person_states[cam_id][track_id]
                if track_id in self._new_tracks[cam_id]:
                    del self._new_tracks[cam_id][track_id]
                if track_id in self._last_face_recognition[cam_id]:
                    del self._last_face_recognition[cam_id][track_id]
            
            # 만료된 Track ID 정리 완료 (로그 생략)


class PersonState:
    """
    개별 사람의 상태 정보
    """
    
    def __init__(self, track_id: int, person_box: Tuple[int, int, int, int], 
                 first_seen: float):
        self.track_id = track_id
        self.person_box = person_box
        self.first_seen = first_seen
        self.last_update = first_seen
        
        # 얼굴 인식 결과
        self.recognized_name: Optional[str] = None
        self.recognition_confidence: float = 0.0
        self.face_embedding: Optional[np.ndarray] = None
        self.face_bbox: Optional[Tuple[int, int, int, int]] = None
        
        # 위반 상태
        self.current_violations: List[str] = []
        self.last_violation_time: float = 0.0
        self.violation_history: List[Tuple[float, List[str]]] = []
        
        # 위험 행동 상태
        self.is_fallen: bool = False
        self.fall_start_time: Optional[float] = None
    
    def update(self, person_box: Tuple[int, int, int, int], timestamp: float) -> None:
        """상태 업데이트"""
        self.person_box = person_box
        self.last_update = timestamp
    
    def add_violation(self, violation_types: List[str], timestamp: float) -> None:
        """위반 상태 추가"""
        self.current_violations = violation_types
        self.last_violation_time = timestamp
        self.violation_history.append((timestamp, violation_types))
        
        # 히스토리 크기 제한 (최근 100개만 유지)
        if len(self.violation_history) > 100:
            self.violation_history = self.violation_history[-100:]
    
    def set_recognized(self, name: str, confidence: float) -> None:
        """얼굴 인식 결과 설정"""
        self.recognized_name = name
        self.recognition_confidence = confidence
    
    def set_fall_status(self, is_fallen: bool, timestamp: float) -> None:
        """넘어짐 상태 설정"""
        if is_fallen and not self.is_fallen:
            # 넘어짐 시작
            self.is_fallen = True
            self.fall_start_time = timestamp
        elif not is_fallen:
            # 넘어짐 해제
            self.is_fallen = False
            self.fall_start_time = None


class FaceRecognitionQueue:
    """
    얼굴 인식 작업 큐 관리
    우선순위: 위반 발생 > 새로운 사람 > 주기적 확인
    """
    
    def __init__(self, max_queue_size: int = 10):
        self.lock = threading.Lock()
        self.max_queue_size = max_queue_size
        
        # 우선순위별 큐
        # {cam_id: {priority: [(track_id, person_crop, face_obj, reason), ...]}}
        self._queues: Dict[int, Dict[int, List[Tuple]]] = defaultdict(
            lambda: {0: [], 1: [], 2: []}  # 0: 최고 우선순위, 1: 중간, 2: 낮음
        )
        
        # 우선순위 매핑
        self.priority_map = {
            "violation_detected": 0,  # 최고 우선순위
            "new_person": 1,          # 중간 우선순위
            "periodic_check": 2       # 낮은 우선순위
        }
    
    def add_task(self, cam_id: int, track_id: int, 
                 person_crop: np.ndarray, 
                 face_obj: Optional[Any],
                 reason: str) -> bool:
        """
        얼굴 인식 작업 추가
        
        Args:
            cam_id: 카메라 ID
            track_id: 추적 ID
            person_crop: 사람 영역 크롭 이미지
            face_obj: 감지된 얼굴 객체 (있으면)
            reason: 인식 이유
            
        Returns:
            bool: 큐에 추가되었는지 여부
        """
        with self.lock:
            priority = self.priority_map.get(reason, 2)
            queue = self._queues[cam_id][priority]
            
            # 이미 같은 track_id가 큐에 있으면 추가하지 않음
            for item in queue:
                if item[0] == track_id:
                    return False
            
            # 큐 크기 제한
            if len(queue) >= self.max_queue_size:
                # 우선순위가 낮은 작업부터 제거
                if priority < 2 and len(self._queues[cam_id][2]) > 0:
                    self._queues[cam_id][2].pop(0)
                elif priority < 1 and len(self._queues[cam_id][1]) > 0:
                    self._queues[cam_id][1].pop(0)
                else:
                    queue.pop(0)  # 같은 우선순위에서 가장 오래된 작업 제거
            
            queue.append((track_id, person_crop, face_obj, reason))
            return True
    
    def get_next_task(self, cam_id: int) -> Optional[Tuple[int, np.ndarray, Any, str]]:
        """
        다음 얼굴 인식 작업 가져오기 (우선순위 순)
        
        Returns:
            (track_id, person_crop, face_obj, reason) 또는 None
        """
        with self.lock:
            for priority in [0, 1, 2]:
                queue = self._queues[cam_id][priority]
                if queue:
                    return queue.pop(0)
            return None
    
    def get_queue_size(self, cam_id: int) -> int:
        """큐 크기 반환"""
        with self.lock:
            total = 0
            for priority in [0, 1, 2]:
                total += len(self._queues[cam_id][priority])
            return total
    
    def clear(self, cam_id: int) -> None:
        """큐 비우기"""
        with self.lock:
            for priority in [0, 1, 2]:
                self._queues[cam_id][priority].clear()


# 전역 인스턴스
person_tracker = PersonTracker()
face_recognition_queue = FaceRecognitionQueue()


def get_person_crop(frame: np.ndarray, person_box: Tuple[int, int, int, int], 
                    padding: float = 0.1) -> Optional[np.ndarray]:
    """
    프레임에서 사람 영역 크롭
    
    Args:
        frame: 원본 프레임
        person_box: 사람 바운딩 박스 (x1, y1, x2, y2)
        padding: 패딩 비율 (기본 10%)
        
    Returns:
        크롭된 이미지 또는 None
    """
    try:
        x1, y1, x2, y2 = person_box
        h, w = frame.shape[:2]
        
        # 패딩 적용
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        # 경계 클리핑
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2].copy()
    except Exception as e:
        logging.debug(f"사람 영역 크롭 실패: {e}")
        return None


def should_run_face_detection_for_frame(cam_id: int, 
                                        has_violations: bool,
                                        new_track_ids: List[int],
                                        unrecognized_track_ids: List[int]) -> bool:
    """
    해당 프레임에서 얼굴 감지를 실행해야 하는지 판단
    
    Args:
        cam_id: 카메라 ID
        has_violations: 위반 사항 발생 여부
        new_track_ids: 새로운 Track ID 리스트
        unrecognized_track_ids: 아직 인식되지 않은 Track ID 리스트
        
    Returns:
        bool: 얼굴 감지 실행 여부
    """
    # 조건 A: 위반 사항 발생 시
    if has_violations:
        return True
    
    # 조건 B: 새로운 사람 등장 시
    if new_track_ids:
        return True
    
    # 조건 C: 인식되지 않은 사람이 있고 주기적 확인 필요 시
    if unrecognized_track_ids:
        return True
    
    return False

