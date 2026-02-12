# -*- coding: utf-8 -*-
"""
로컬 파일 시스템 기반 데이터 관리 모듈
JSON 파일을 사용하여 데이터를 저장하고 관리합니다.
"""

import json
import os
import sys
import datetime
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# 로깅 설정 (한글 깨짐 방지)
logger = logging.getLogger(__name__)


class LocalStorageManager:
    """로컬 파일 시스템 기반 데이터 관리 클래스"""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Args:
            data_dir: 데이터 파일 저장 디렉토리 (None이면 프로젝트 루트의 data 폴더 사용)
        """
        if data_dir is None:
            # 프로젝트 루트의 data 폴더 사용 (src/backend에서 두 단계 위)
            base_dir = Path(__file__).parent.parent.parent
            # 프로젝트 루트에 data 폴더 생성
            data_dir = base_dir / 'data'
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 파일 경로
        self.persons_file = self.data_dir / 'registered_persons.json'
        self.logs_file = self.data_dir / 'access_logs.json'
        self.daily_images_file = self.data_dir / 'daily_access_images.json'
        self.statistics_file = self.data_dir / 'daily_statistics.json'
        
        # 초기화
        self._initialize_files()
    
    def _initialize_files(self) -> None:
        """데이터 파일 초기화"""
        files: Dict[Path, Union[List[Any], Dict[str, Any]]] = {
            self.persons_file: [],
            self.logs_file: [],
            self.daily_images_file: [],
            self.statistics_file: {}
        }
        
        for file_path, default_data in files.items():
            if not file_path.exists():
                self._save_json(file_path, default_data)
                logger.info(f"데이터 파일 생성: {file_path}")
    
    def _load_json(self, file_path: Path) -> Union[List[Any], Dict[str, Any]]:
        """JSON 파일 로드"""
        try:
            if not file_path.exists():
                return [] if 'statistics' not in str(file_path) else {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data: Any = json.load(f)
                if isinstance(data, (list, dict)):
                    return data
                return [] if 'statistics' not in str(file_path) else {}
        except Exception as e:
            logger.error(f"JSON 파일 로드 오류 ({file_path}): {e}")
            return [] if 'statistics' not in str(file_path) else {}
    
    def _save_json(self, file_path: Path, data: Union[List[Any], Dict[str, Any]]) -> None:
        """JSON 파일 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"JSON 파일 저장 오류 ({file_path}): {e}")
            raise
    
    def register_person(self, person_id: str, name: str, registered_image_path: str,
                       face_encoding_path: Optional[str] = None, department: Optional[str] = None,
                       position: Optional[str] = None, phone: Optional[str] = None, email: Optional[str] = None) -> Optional[int]:
        """사람 등록"""
        try:
            persons_data = self._load_json(self.persons_file)
            if not isinstance(persons_data, list):
                persons: List[Dict[str, Any]] = []
            else:
                persons = persons_data
            
            # 중복 확인
            if any(isinstance(p, dict) and p.get('person_id') == person_id for p in persons):
                logger.warning(f"이미 등록된 사람 ID: {person_id}")
                return None
            
            # 새 사람 추가
            person_data = {
                'id': len(persons) + 1,
                'person_id': person_id,
                'name': name,
                'department': department,
                'position': position,
                'phone': phone,
                'email': email,
                'registered_image_path': registered_image_path,
                'face_encoding_path': face_encoding_path,
                'status': 'active',
                'registered_at': datetime.datetime.now().isoformat(),
                'updated_at': datetime.datetime.now().isoformat()
            }
            
            persons.append(person_data)
            self._save_json(self.persons_file, persons)
            
            person_id_value: int = person_data['id']
            logger.info(f"사람 등록 완료: ID={person_id_value}, 이름={name}")
            return person_id_value
            
        except Exception as e:
            logger.error(f"사람 등록 오류: {e}")
            return None
    
    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """사람 정보 조회"""
        try:
            persons_data = self._load_json(self.persons_file)
            if not isinstance(persons_data, list):
                return None
            persons = persons_data
            for person in persons:
                if isinstance(person, dict) and person.get('person_id') == person_id and person.get('status') == 'active':
                    return person
            return None
        except Exception as e:
            logger.error(f"사람 정보 조회 오류: {e}")
            return None
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """모든 등록된 사람 조회"""
        try:
            persons_data = self._load_json(self.persons_file)
            if not isinstance(persons_data, list):
                return []
            persons = persons_data
            return [p for p in persons if isinstance(p, dict) and p.get('status') == 'active']
        except Exception as e:
            logger.error(f"사람 목록 조회 오류: {e}")
            return []
    
    def save_access_log(self, image_path: str, camera_id: Optional[str] = None,
                       location: Optional[str] = None, detected_person_count: int = 0,
                       status: str = 'normal', notes: Optional[str] = None,
                       timestamp: Optional[datetime.datetime] = None,
                       person_id: Optional[int] = None, person_name: Optional[str] = None,
                       match_confidence: Optional[float] = None,
                       worker_id: Optional[str] = None) -> Optional[int]:
        """출입 기록 저장 (JSON 파일 + MongoDB) - aivis-front와 동일한 로직"""
        try:
            if timestamp is None:
                timestamp = datetime.datetime.now()
            
            # JSON 파일에 저장 (aivis-front와 동일)
            logs_data = self._load_json(self.logs_file)
            if not isinstance(logs_data, list):
                logs: List[Dict[str, Any]] = []
            else:
                logs = logs_data
            
            log_data = {
                'id': len(logs) + 1,
                'timestamp': timestamp.isoformat(),
                'image_path': image_path,
                'camera_id': camera_id,
                'location': location,
                'detected_person_count': detected_person_count,
                'status': status,
                'notes': notes,
                'person_id': person_id,
                'person_name': person_name,
                'match_confidence': match_confidence,
                'worker_id': worker_id,  # 입력받은 작업자 ID 추가
                'created_at': datetime.datetime.now().isoformat(),
                'updated_at': datetime.datetime.now().isoformat()
            }
            
            logs.append(log_data)
            self._save_json(self.logs_file, logs)
            
            # 일일 통계 업데이트 (aivis-front와 동일)
            self._update_daily_statistics(timestamp.date())
            
            log_id_value: int = log_data['id']
            logger.info(f"출입 기록 저장 완료: ID={log_id_value}")
            return log_id_value
            
        except Exception as e:
            logger.error(f"출입 기록 저장 오류: {e}")
            return None
    
    def _update_daily_statistics(self, date: datetime.date) -> None:
        """일일 통계 업데이트"""
        try:
            stats_data = self._load_json(self.statistics_file)
            if not isinstance(stats_data, dict):
                stats: Dict[str, Any] = {}
            else:
                stats = stats_data
            date_str = date.isoformat()
            
            if date_str not in stats:
                stats[date_str] = {
                    'date': date_str,
                    'total_images': 0,
                    'total_entries': 0,
                    'total_exits': 0,
                    'status': 'pending'
                }
            
            if isinstance(stats[date_str], dict):
                stats[date_str]['total_images'] = stats[date_str].get('total_images', 0) + 1
                stats[date_str]['updated_at'] = datetime.datetime.now().isoformat()
            
            self._save_json(self.statistics_file, stats)
            
        except Exception as e:
            logger.error(f"일일 통계 업데이트 오류: {e}")
    
    def get_access_logs(
        self, 
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        camera_id: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """출입 기록 조회"""
        try:
            logs_data = self._load_json(self.logs_file)
            if not isinstance(logs_data, list):
                return []
            logs = logs_data
            
            # 필터링
            filtered_logs: List[Dict[str, Any]] = []
            for log in logs:
                if not isinstance(log, dict):
                    continue
                try:
                    log_date = datetime.datetime.fromisoformat(log.get('timestamp', '')).date()
                except (ValueError, AttributeError):
                    continue
                
                if start_date and log_date < start_date:
                    continue
                if end_date and log_date > end_date:
                    continue
                if camera_id and log.get('camera_id') != camera_id:
                    continue
                
                filtered_logs.append(log)
            
            # 정렬 및 제한
            filtered_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return filtered_logs[:limit]
            
        except Exception as e:
            logger.error(f"출입 기록 조회 오류: {e}")
            return []
    
    def get_daily_statistics(self, date: Optional[datetime.date] = None) -> Optional[Dict[str, Any]]:
        """일일 통계 조회"""
        try:
            if date is None:
                date = datetime.date.today()
            
            stats_data = self._load_json(self.statistics_file)
            if not isinstance(stats_data, dict):
                return None
            stats = stats_data
            date_str = date.isoformat()
            
            result = stats.get(date_str)
            if isinstance(result, dict):
                return result
            return None
            
        except Exception as e:
            logger.error(f"일일 통계 조회 오류: {e}")
            return None
    
    def save_daily_access_image(self, person_id: int, person_name: str,
                                image_path: str, access_log_id: Optional[int] = None,
                                access_date: Optional[datetime.date] = None) -> Optional[int]:
        """일일 출입 이미지 저장"""
        try:
            if access_date is None:
                access_date = datetime.date.today()
            
            images_data = self._load_json(self.daily_images_file)
            if not isinstance(images_data, list):
                images: List[Dict[str, Any]] = []
            else:
                images = images_data
            
            # 중복 확인
            for img in images:
                if isinstance(img, dict) and (img.get('person_id') == person_id and
                    img.get('access_date') == access_date.isoformat() and
                    img.get('image_path') == image_path):
                    logger.info("이미 존재하는 기록")
                    return img.get('id')
            
            # 새 이미지 추가
            image_data = {
                'id': len(images) + 1,
                'person_id': person_id,
                'person_name': person_name,
                'access_date': access_date.isoformat(),
                'image_path': image_path,
                'access_log_id': access_log_id,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            images.append(image_data)
            self._save_json(self.daily_images_file, images)
            
            image_id_value: int = image_data['id']
            logger.info(f"일일 출입 이미지 저장 완료: ID={image_id_value}")
            return image_id_value
            
        except Exception as e:
            logger.error(f"일일 출입 이미지 저장 오류: {e}")
            return None
    
    def get_daily_access_images(
        self, 
        person_id: Optional[int] = None,
        access_date: Optional[datetime.date] = None
    ) -> List[Dict[str, Any]]:
        """일일 출입 이미지 조회"""
        try:
            if access_date is None:
                access_date = datetime.date.today()
            
            images_data = self._load_json(self.daily_images_file)
            if not isinstance(images_data, list):
                return []
            images = images_data
            
            # 필터링
            filtered_images: List[Dict[str, Any]] = []
            for img in images:
                if not isinstance(img, dict):
                    continue
                try:
                    img_date = datetime.date.fromisoformat(img.get('access_date', ''))
                except (ValueError, AttributeError):
                    continue
                
                if img_date != access_date:
                    continue
                if person_id and img.get('person_id') != person_id:
                    continue
                
                filtered_images.append(img)
            
            # 정렬
            filtered_images.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return filtered_images
            
        except Exception as e:
            logger.error(f"일일 출입 이미지 조회 오류: {e}")
            return []

