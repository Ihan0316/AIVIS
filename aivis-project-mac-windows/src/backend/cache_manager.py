# -*- coding: utf-8 -*-
"""
캐시 관리 모듈
메모리 누수 방지를 위한 LRU 캐시 및 TTL 캐시 구현
"""
import time
import threading
from typing import Dict, Optional, Any, List, Union
from collections import OrderedDict


class LRUCache:
    """
    스레드 안전한 LRU (Least Recently Used) 캐시
    
    최근에 사용된 항목을 유지하고, 오래된 항목을 자동으로 제거합니다.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: 최대 캐시 크기
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        캐시에서 값을 가져옵니다.
        접근한 항목은 최근 사용 목록의 맨 앞으로 이동합니다.
        
        Args:
            key: 캐시 키
        
        Returns:
            캐시된 값 또는 None
        """
        with self.lock:
            if key in self.cache:
                # 최근 사용 목록의 맨 앞으로 이동
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: Any, value: Any) -> None:
        """
        캐시에 값을 저장합니다.
        최대 크기를 초과하면 가장 오래된 항목을 제거합니다.
        
        Args:
            key: 캐시 키
            value: 저장할 값
        """
        with self.lock:
            if key in self.cache:
                # 기존 항목 업데이트 및 최근 사용 목록의 맨 앞으로 이동
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거 (FIFO)
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def remove(self, key: Any) -> bool:
        """
        캐시에서 항목을 제거합니다.
        
        Args:
            key: 제거할 캐시 키
        
        Returns:
            제거 성공 여부
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """캐시를 모두 비웁니다."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """현재 캐시 크기를 반환합니다."""
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[Any]:
        """캐시의 모든 키를 반환합니다."""
        with self.lock:
            return list(self.cache.keys())


class TTLCache:
    """
    스레드 안전한 TTL (Time To Live) 캐시
    
    각 항목에 만료 시간을 설정하고, 만료된 항목을 자동으로 제거합니다.
    """
    
    def __init__(self, default_ttl: float = 60.0):
        """
        Args:
            default_ttl: 기본 TTL (초)
        """
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        캐시에서 값을 가져옵니다.
        만료된 항목은 None을 반환하고 캐시에서 제거합니다.
        
        Args:
            key: 캐시 키
        
        Returns:
            캐시된 값 또는 None (만료된 경우)
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            current_time = time.time()
            
            # 만료 확인
            if current_time > entry['expires_at']:
                del self.cache[key]
                return None
            
            return entry['value']
    
    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        캐시에 값을 저장합니다.
        
        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: TTL (초), None이면 기본값 사용
        """
        with self.lock:
            if ttl is None:
                ttl = self.default_ttl
            
            self.cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl
            }
    
    def remove(self, key: Any) -> bool:
        """
        캐시에서 항목을 제거합니다.
        
        Args:
            key: 제거할 캐시 키
        
        Returns:
            제거 성공 여부
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear_expired(self) -> int:
        """
        만료된 항목을 모두 제거합니다.
        
        Returns:
            제거된 항목 수
        """
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time > entry['expires_at']
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def clear(self) -> None:
        """캐시를 모두 비웁니다."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """현재 캐시 크기를 반환합니다."""
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[Any]:
        """
        캐시의 모든 키를 반환합니다 (만료된 항목 제외).
        주의: 이 메서드는 만료된 항목을 자동으로 제거합니다.
        """
        with self.lock:
            current_time = time.time()
            # 만료된 항목 제거
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time > entry['expires_at']
            ]
            for key in expired_keys:
                del self.cache[key]
            return list(self.cache.keys())


class IdentityCache:
    """
    얼굴 인식 결과 캐시 (LRU + TTL 혼합)
    
    각 카메라별로 최근 인식 결과를 관리합니다.
    """
    
    def __init__(self, max_items_per_cam: int = 30, ttl: float = 1.2):
        """
        Args:
            max_items_per_cam: 카메라당 최대 항목 수
            ttl: 항목 만료 시간 (초)
        """
        self.max_items_per_cam = max_items_per_cam
        self.ttl = ttl
        self.caches: Dict[int, List[Dict[str, Any]]] = {}
        self.lock = threading.Lock()
    
    def add(self, cam_id: int, entry: Dict[str, Any]) -> None:
        """
        캐시에 항목을 추가합니다.
        
        Args:
            cam_id: 카메라 ID
            entry: 캐시 항목 (box, name, score, ts 포함)
        """
        with self.lock:
            if cam_id not in self.caches:
                self.caches[cam_id] = []
            
            cache = self.caches[cam_id]
            
            # 타임스탬프 추가 (없으면 현재 시간)
            if 'ts' not in entry:
                entry['ts'] = time.time()
            
            # 새 항목 추가
            cache.append(entry)
            
            # 최대 크기 초과 시 오래된 항목 제거
            if len(cache) > self.max_items_per_cam:
                cache.pop(0)
    
    def get_recent(self, cam_id: int, max_age: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        최근 항목을 가져옵니다.
        
        Args:
            cam_id: 카메라 ID
            max_age: 최대 나이 (초), None이면 TTL 사용
        
        Returns:
            최근 항목 리스트
        """
        with self.lock:
            if cam_id not in self.caches:
                return []
            
            cache = self.caches[cam_id]
            current_time = time.time()
            
            if max_age is None:
                max_age = self.ttl
            
            # 만료된 항목 제거 및 최근 항목 반환
            valid_items = [
                item for item in cache
                if current_time - item.get('ts', 0) <= max_age
            ]
            
            # 만료된 항목 제거
            self.caches[cam_id] = valid_items
            
            return valid_items
    
    def clear(self, cam_id: Optional[int] = None) -> None:
        """
        캐시를 비웁니다.
        
        Args:
            cam_id: 카메라 ID, None이면 모든 캐시 비움
        """
        with self.lock:
            if cam_id is None:
                self.caches.clear()
            elif cam_id in self.caches:
                del self.caches[cam_id]
    
    def get(self, cam_id: int) -> Optional[Dict[Any, Dict[str, Any]]]:
        """
        카메라별 캐시를 딕셔너리 형태로 반환합니다 (tracker_id -> entry).
        
        Args:
            cam_id: 카메라 ID
        
        Returns:
            딕셔너리 형태의 캐시 또는 None
        """
        with self.lock:
            if cam_id not in self.caches:
                return None
            
            cache = self.caches[cam_id]
            current_time = time.time()
            
            # 만료된 항목 제거 및 딕셔너리로 변환
            result = {}
            valid_items = []
            
            for item in cache:
                if current_time - item.get('ts', 0) <= self.ttl:
                    valid_items.append(item)
                    # tracker_id가 있으면 키로 사용, 없으면 box를 키로 사용
                    tracker_id = item.get('tracker_id')
                    if tracker_id is not None:
                        result[tracker_id] = item
                    else:
                        # box를 키로 사용 (튜플로 변환)
                        box = item.get('box', (0, 0, 0, 0))
                        if isinstance(box, (list, tuple)) and len(box) == 4:
                            result[tuple(box)] = item
            
            # 만료된 항목 제거
            self.caches[cam_id] = valid_items
            
            return result if result else None
    
    def size(self, cam_id: Optional[int] = None) -> int:
        """
        캐시 크기를 반환합니다.
        
        Args:
            cam_id: 카메라 ID, None이면 전체 크기
        
        Returns:
            캐시 크기
        """
        with self.lock:
            if cam_id is None:
                return sum(len(cache) for cache in self.caches.values())
            return len(self.caches.get(cam_id, []))

