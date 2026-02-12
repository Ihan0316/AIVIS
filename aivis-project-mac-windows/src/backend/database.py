# -*- coding: utf-8 -*-
"""
MongoDB 데이터베이스 연결 모듈
새 프론트엔드와의 호환성을 위한 래퍼
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_db_instance = None
_violation_service = None
_worker_service = None

class DatabaseWrapper:
    """DatabaseService를 백엔드 API와 호환되도록 래핑"""
    
    def __init__(self, db_service: Any, violation_service: Any, worker_service: Any) -> None:
        self.db_service = db_service
        self._violation_service = violation_service
        self.worker_service = worker_service
    
    @property
    def violation_service(self):
        """ViolationService 접근자"""
        return self._violation_service
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        if self.db_service is None:
            return False
        return self.db_service.is_connected()
    
    def get_violations(
        self, 
        worker_name: Optional[str] = None, 
        camera_id: Optional[int] = None, 
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """위반 사항 조회 (백엔드 API 호환)"""
        if not self.is_connected():
            logger.warning("[DB 조회] DB 연결되지 않음")
            return []
        
        try:
            # timestamp 변환 (MongoDB 쿼리 최적화를 위해)
            start_timestamp = None
            end_timestamp = None
            if start_time:
                start_timestamp = int(start_time.timestamp() * 1000)
            if end_time:
                end_timestamp = int(end_time.timestamp() * 1000)
            
            # worker_name을 worker_id로 변환 (필요시)
            worker_id = None
            if worker_name:
                # worker_name이 worker_id일 수도 있으므로 둘 다 확인
                worker_id = worker_name
            
            # MongoDB 쿼리로 직접 필터링 (메모리 필터링 대신)
            result = self.violation_service.get_violations(
                limit=limit,
                offset=0,
                status=None,
                cam_id=camera_id,  # camera_id를 cam_id로 변환
                worker_id=worker_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp
            )
            violations = result.get('violations', [])
            
            logger.debug(f"[DB 조회] MongoDB 쿼리 결과: {len(violations)}건 (limit={limit}, cam_id={camera_id}, worker_id={worker_id}, start={start_timestamp}, end={end_timestamp})")
            
            # event_type 필터링 (MongoDB 쿼리에서 지원하지 않으므로 메모리에서 필터링)
            if event_type:
                filtered = []
                for v in violations:
                    violation_type = v.get('violation_type') or v.get('type', '')
                    if event_type in violation_type:
                        filtered.append(v)
                violations = filtered
                logger.debug(f"[DB 조회] event_type 필터링 후: {len(violations)}건")
            
            # worker_name 필터링 (정확한 매칭이 필요한 경우)
            if worker_name:
                filtered = []
                for v in violations:
                    v_worker_name = v.get('worker_name', '')
                    v_worker_id = v.get('worker_id', '')
                    if v_worker_name == worker_name or v_worker_id == worker_name:
                        filtered.append(v)
                violations = filtered
                logger.debug(f"[DB 조회] worker_name 필터링 후: {len(violations)}건")
            
            return violations[:limit]
        except Exception as e:
            logger.error(f"[DB 조회] 위반 사항 조회 오류: {e}", exc_info=True)
            return []
    
    def get_all_workers(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """작업자 목록 조회 (백엔드 API 호환)"""
        if not self.is_connected():
            return []
        
        try:
            if self.worker_service is None:
                return []
            workers = self.worker_service.get_all_workers()
            # active_only 필터링 (현재는 모든 작업자 반환)
            if isinstance(workers, list):
                return workers
            return []
        except Exception as e:
            logger.error(f"작업자 조회 오류: {e}", exc_info=True)
            return []
    
    def get_violation_statistics(self, days: int = 7, start_timestamp: Optional[int] = None, end_timestamp: Optional[int] = None) -> Dict[str, Any]:
        """위반 사항 통계 조회 (백엔드 API 호환)"""
        # 기본 chart_data 생성 (7주 데이터)
        default_chart_data = []
        for i in range(7):
            default_chart_data.append({
                "week": i,
                "violations": 0,
                "helmet_violations": 0,
                "fall_detections": 0,
                "completed": 0
            })
        
        if not self.is_connected():
            return {
                "total_violations": 0,
                "period_days": days,
                "camera_stats": {},
                "worker_stats": {},
                "event_type_stats": {},
                "chart_data": default_chart_data  # 기본 차트 데이터 포함
            }
        
        try:
            # start_timestamp와 end_timestamp가 제공되면 시간 필터링된 통계 조회
            stats = self.violation_service.get_violation_stats(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
            kpi = stats.get('kpi', {})
            chart_data = stats.get('chart_data', [])
            
            # chart_data가 비어있거나 형식이 잘못된 경우 기본값 사용
            if not chart_data or not isinstance(chart_data, list):
                logger.warning(f"chart_data가 비어있거나 잘못된 형식입니다. 기본값 사용. chart_data: {chart_data}")
                chart_data = default_chart_data
            elif len(chart_data) < 7:
                # 부족한 주차 데이터를 0으로 채움
                for i in range(len(chart_data), 7):
                    chart_data.append({
                        "week": i,
                        "violations": 0,
                        "helmet_violations": 0,
                        "fall_detections": 0,
                        "completed": 0
                    })
            elif len(chart_data) > 7:
                # 7개 초과면 최근 7개만 사용
                chart_data = chart_data[-7:]
            
            logger.debug(f"[통계] chart_data 반환: {len(chart_data)}개 주차, 샘플: {chart_data[0] if chart_data else 'None'}")
            
            return {
                "total_violations": kpi.get('total', 0),
                "period_days": days,
                "camera_stats": {},
                "worker_stats": {},
                "event_type_stats": {
                    "helmet": kpi.get('helmet', 0),
                    "vest": kpi.get('vest', 0),
                    "fall": kpi.get('fall', 0)
                },
                "chart_data": chart_data  # 차트 데이터 포함
            }
        except Exception as e:
            logger.error(f"통계 조회 오류: {e}", exc_info=True)
            # 에러 발생 시 기본값 반환
            return {
                "total_violations": 0,
                "period_days": days,
                "camera_stats": {},
                "worker_stats": {},
                "event_type_stats": {},
                "chart_data": default_chart_data  # 기본 차트 데이터 포함
            }

class SimpleMongoDBService:
    """간단한 MongoDB 서비스 (pymongo 직접 사용)"""
    
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self._connected = False
    
    def connect(self) -> bool:
        """MongoDB 연결 (연결 풀 최적화)"""
        try:
            from pymongo import MongoClient, WriteConcern
            from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
            
            # 연결 풀 설정 최적화
            max_pool_size = int(os.getenv('MONGO_MAX_POOL_SIZE', '50'))
            min_pool_size = int(os.getenv('MONGO_MIN_POOL_SIZE', '10'))
            max_idle_time_ms = int(os.getenv('MONGO_MAX_IDLE_TIME_MS', '30000'))
            
            # MongoClient 생성 (WriteConcern은 컬렉션 레벨에서 설정)
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=max_pool_size,
                minPoolSize=min_pool_size,
                maxIdleTimeMS=max_idle_time_ms
            )
            
            # 연결 테스트
            if self.client is not None:
                self.client.admin.command('ping')
                self.db = self.client[self.db_name]
                self._connected = True
            else:
                return False
            
            # 인덱스 생성
            self._create_indexes()
            
            logger.info(f"✅ MongoDB 연결 성공: {self.db_name} ({self.mongo_uri})")
            return True
        except ImportError:
            logger.warning("⚠️  pymongo가 설치되지 않았습니다. 설치: pip install pymongo")
            return False
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.warning(f"⚠️  MongoDB 서버 연결 실패: {e}")
            logger.warning(f"⚠️  MongoDB 서버가 실행 중인지 확인하세요: {self.mongo_uri}")
            return False
        except Exception as e:
            logger.error(f"⚠️  MongoDB 연결 오류: {e}", exc_info=True)
            return False
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        if not self._connected or self.client is None:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            self._connected = False
            return False
    
    def _create_indexes(self) -> None:
        """필요한 인덱스 생성 (최적화된 복합 인덱스)"""
        try:
            if self.db is None:
                return
            # violation 컬렉션 인덱스
            violations = self.db['violation']
            
            # 기존 인덱스 제거 (선택적, 마이그레이션 시)
            # violations.drop_indexes()  # 주의: 기존 인덱스 모두 삭제
            
            # 1. timestamp 인덱스 (가장 중요한 쿼리 필드)
            violations.create_index([('timestamp', -1)], name='idx_timestamp')
            
            # 2. 복합 인덱스: (cam_id, timestamp) - 카메라별 최신 위반 조회
            violations.create_index([('cam_id', 1), ('timestamp', -1)], name='idx_cam_timestamp')
            
            # 3. 복합 인덱스: (worker_id, timestamp, type) - 작업자별 위반 조회 및 중복 체크
            violations.create_index([('worker_id', 1), ('timestamp', -1), ('type', 1)], name='idx_worker_timestamp_type')
            
            # 4. 복합 인덱스: (status, timestamp) - 상태별 조회
            violations.create_index([('status', 1), ('timestamp', -1)], name='idx_status_timestamp')
            
            # 5. TTL 인덱스: 90일 후 자동 삭제 (선택적)
            ttl_days = int(os.getenv('VIOLATION_TTL_DAYS', '90'))
            if ttl_days > 0:
                try:
                    violations.create_index([('timestamp', 1)], expireAfterSeconds=ttl_days * 24 * 3600, name='idx_ttl')
                    logger.info(f"MongoDB: TTL 인덱스 생성 완료 ({ttl_days}일 후 자동 삭제)")
                except Exception as ttl_error:
                    logger.warning(f"MongoDB: TTL 인덱스 생성 실패: {ttl_error}")
            
            # 하위 호환성 인덱스 (기존 코드 호환)
            violations.create_index([('violation_datetime', -1)], name='idx_violation_datetime')
            violations.create_index([('worker_id', 1)], name='idx_worker_id')
            violations.create_index([('violation_type', 1)], name='idx_violation_type')
            violations.create_index([('status', 1)], name='idx_status')
            violations.create_index([('worker_id', 1), ('violation_datetime', -1)], name='idx_worker_datetime')
            
            # 고유 인덱스: 중복 방지 (worker_id + type + timestamp_ms 조합)
            # 같은 작업자의 같은 위반이 같은 밀리초에 발생하는 경우를 방지
            try:
                # 고유 인덱스 생성 (중복 저장 방지)
                violations.create_index(
                    [('worker_id', 1), ('type', 1), ('timestamp', 1)],
                    unique=True,
                    name='idx_unique_worker_type_timestamp',
                    partialFilterExpression={'timestamp': {'$exists': True}}
                )
                logger.info("MongoDB: 고유 인덱스 생성 완료 (worker_id + type + timestamp)")
            except Exception as index_error:
                # 이미 존재하는 인덱스이거나 중복 데이터가 있어서 실패할 수 있음
                logger.warning(f"MongoDB: 고유 인덱스 생성 실패 (이미 존재하거나 중복 데이터 있음): {index_error}")
                # 일반 인덱스로 대체
                try:
                    violations.create_index(
                        [('worker_id', 1), ('type', 1), ('timestamp', -1)],
                        name='idx_worker_type_timestamp'
                    )
                    logger.info("MongoDB: 일반 인덱스 생성 완료 (worker_id + type + timestamp)")
                except Exception as fallback_error:
                    logger.warning(f"MongoDB: 일반 인덱스 생성도 실패: {fallback_error}")
            
            # worker 컬렉션 인덱스
            workers = self.db['worker']
            
            # workerId가 null인 기존 레코드 정리 (인덱스 생성 전)
            try:
                # workerId가 null이거나 빈 문자열인 레코드에 임시 값 할당
                workers.update_many(
                    {'$or': [{'workerId': None}, {'workerId': ''}, {'workerId': {'$exists': False}}]},
                    {'$set': {'workerId': f'unknown_{datetime.now().timestamp()}'}}
                )
                logger.info("MongoDB: workerId가 null인 레코드 정리 완료")
            except Exception as cleanup_error:
                logger.warning(f"MongoDB workerId 정리 실패 (계속 진행): {cleanup_error}")
            
            # unique 인덱스 생성 (sparse 옵션으로 null 값 허용)
            try:
                workers.create_index([('workerId', 1)], unique=True, sparse=True)
            except Exception as unique_error:
                # unique 인덱스 생성 실패 시 일반 인덱스로 생성
                logger.warning(f"MongoDB workerId unique 인덱스 생성 실패, 일반 인덱스로 생성: {unique_error}")
                try:
                    workers.create_index([('workerId', 1)])
                except Exception as normal_error:
                    logger.warning(f"MongoDB workerId 인덱스 생성 실패: {normal_error}")
            
            workers.create_index([('workerName', 1)])
            
            # face 컬렉션 인덱스 (얼굴 인식 데이터)
            faces = self.db['face']
            # workerId unique 인덱스 (sparse 옵션으로 null 값 허용)
            try:
                faces.create_index([('workerId', 1)], unique=True, sparse=True, name='idx_face_workerId')
            except Exception as unique_error:
                logger.warning(f"MongoDB face workerId unique 인덱스 생성 실패, 일반 인덱스로 생성: {unique_error}")
                try:
                    faces.create_index([('workerId', 1)], name='idx_face_workerId')
                except Exception as normal_error:
                    logger.warning(f"MongoDB face workerId 인덱스 생성 실패: {normal_error}")
            
            # workerName 인덱스
            faces.create_index([('workerName', 1)], name='idx_face_workerName')
            
            # created_at 인덱스 (최신 얼굴 데이터 조회용)
            faces.create_index([('created_at', -1)], name='idx_face_created_at')
            
            # updated_at 인덱스
            faces.create_index([('updated_at', -1)], name='idx_face_updated_at')
            
            logger.info("MongoDB: face 컬렉션 인덱스 생성 완료")
            
            logger.debug("MongoDB 인덱스 생성 완료")
        except Exception as e:
            logger.warning(f"MongoDB 인덱스 생성 실패: {e}")
    
    def get_violations_collection(self) -> Any:
        """violation 컬렉션 가져오기"""
        if not self.is_connected() or self.db is None:
            return None
        return self.db['violation']
    
    
    def get_workers_collection(self) -> Any:
        """worker 컬렉션 가져오기"""
        if not self.is_connected() or self.db is None:
            return None
        return self.db['worker']
    
    def get_faces_collection(self) -> Any:
        """face 컬렉션 가져오기"""
        if not self.is_connected() or self.db is None:
            return None
        return self.db['face']
    
    def close(self) -> None:
        """연결 종료"""
        if self.client:
            self.client.close()
            self._connected = False

class SimpleViolationService:
    """간단한 위반 사항 서비스"""
    
    def __init__(self, db_service: SimpleMongoDBService):
        self.db_service = db_service
    
    def get_violations(
        self, 
        limit: int = 1000, 
        offset: int = 0, 
        status: Optional[str] = None,
        cam_id: Optional[int] = None,
        worker_id: Optional[str] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """위반 사항 조회 (MongoDB 쿼리 최적화)"""
        if not self.db_service.is_connected():
            logger.warning("[DB 조회] DB 서비스가 연결되지 않음")
            return {'violations': []}
        
        try:
            collection = self.db_service.get_violations_collection()
            if collection is None:
                logger.warning("[DB 조회] violations 컬렉션을 가져올 수 없음")
                return {'violations': []}
            
            # MongoDB 쿼리로 필터링 (메모리 필터링 대신)
            query = {}
            if status:
                query['status'] = status
            if cam_id is not None:
                query['cam_id'] = cam_id  # 표준 필드명 사용
            if worker_id:
                query['worker_id'] = worker_id
            if start_timestamp is not None or end_timestamp is not None:
                timestamp_query = {}
                if start_timestamp is not None:
                    timestamp_query['$gte'] = start_timestamp
                if end_timestamp is not None:
                    timestamp_query['$lte'] = end_timestamp
                if timestamp_query:
                    query['timestamp'] = timestamp_query
            
            logger.debug(f"[DB 조회] MongoDB 쿼리: {query}, limit={limit}, offset={offset}")
            
            # 전체 문서 수 확인 (디버깅용)
            total_count = collection.count_documents(query)
            logger.debug(f"[DB 조회] 쿼리 매칭 문서 수: {total_count}개")
            
            # timestamp 인덱스 활용하여 정렬 (violation_datetime보다 빠름)
            cursor = collection.find(query).sort('timestamp', -1).skip(offset)
            # limit이 None이면 모든 데이터 반환 (limit 적용 안 함)
            if limit is not None:
                cursor = cursor.limit(limit)
            violations = list(cursor)
            
            logger.info(f"[DB 조회] MongoDB에서 {len(violations)}개 위반 사항 조회 완료 (총 {total_count}개 중)")
            
            # ObjectId를 문자열로 변환 및 실제 DB 데이터 구조 확인
            for v in violations:
                if '_id' in v:
                    v['_id'] = str(v['_id'])
            
            # 실제 DB 데이터 구조 로깅 (디버깅용)
            if violations:
                sample = violations[0]
                logger.info(f"[DB 조회] 실제 DB 데이터 샘플 - 모든 필드: {list(sample.keys())}")
                logger.info(f"[DB 조회] 실제 DB 데이터 샘플 - 값: worker_id={sample.get('worker_id')}, worker_name={sample.get('worker_name')}, "
                          f"type={sample.get('type')}, violation_type={sample.get('violation_type')}, "
                          f"timestamp={sample.get('timestamp')}, violation_datetime={sample.get('violation_datetime')}, "
                          f"cam_id={sample.get('cam_id')}, camera_id={sample.get('camera_id')}, "
                          f"work_zone={sample.get('work_zone')}, status={sample.get('status')}")
            
            return {'violations': violations}
        except Exception as e:
            logger.error(f"[DB 조회] 위반 사항 조회 오류: {e}", exc_info=True)
            return {'violations': []}
    
    def get_violation_stats(self, start_timestamp: Optional[int] = None, end_timestamp: Optional[int] = None) -> Dict[str, Any]:
        """위반 사항 통계 조회 (aggregate()로 최적화)"""
        if not self.db_service.is_connected():
            return {'kpi': {}, 'chart_data': []}
        
        try:
            collection = self.db_service.get_violations_collection()
            if collection is None:
                return {'kpi': {}, 'chart_data': []}
            
            # 시간 필터링 조건 추가 (get_violations와 동일하게 timestamp만 사용)
            time_match = {}
            if start_timestamp is not None or end_timestamp is not None:
                # timestamp 필터만 사용 (get_violations와 동일)
                timestamp_query = {}
                if start_timestamp is not None:
                    timestamp_query['$gte'] = start_timestamp
                if end_timestamp is not None:
                    timestamp_query['$lte'] = end_timestamp
                
                if timestamp_query:
                    time_match = {'timestamp': timestamp_query}
                    logger.debug(f"[통계] 시간 필터: start_ts={start_timestamp}, end_ts={end_timestamp} (timestamp만 사용)")
            
            # aggregate()로 한 번에 집계 (count_documents() 여러 번 호출 대신)
            # 시간 필터가 있으면 먼저 매칭
            match_stage = {}
            if time_match:
                match_stage = {'$match': time_match}
            
            pipeline = []
            if match_stage:
                pipeline.append(match_stage)
            
            # $facet 안의 각 파이프라인은 이전 단계의 결과를 받아서 처리하므로,
            # 시간 필터가 이미 적용된 데이터에서 타입별로 필터링합니다.
            pipeline.append({
                    '$facet': {
                        'total': [{'$count': 'count'}],
                        'helmet': [
                            {'$match': {'$or': [
                                {'type': {'$regex': '안전모|헬멧|helmet|hard.*hat|hardhat', '$options': 'i'}},
                                {'violation_type': {'$regex': '안전모|헬멧|helmet|hard.*hat|hardhat', '$options': 'i'}}
                            ]}},
                            {'$count': 'count'}
                        ],
                        'vest': [
                            {'$match': {'$or': [
                                {'type': {'$regex': '안전조끼|조끼|vest|safety.*vest|reflective', '$options': 'i'}},
                                {'violation_type': {'$regex': '안전조끼|조끼|vest|safety.*vest|reflective', '$options': 'i'}}
                            ]}},
                            {'$count': 'count'}
                        ],
                        'fall': [
                            {'$match': {'$or': [
                                {'type': {'$regex': '넘어짐|낙상|fall|fallen|trip|slip', '$options': 'i'}},
                                {'violation_type': {'$regex': '넘어짐|낙상|fall|fallen|trip|slip', '$options': 'i'}}
                            ]}},
                            {'$count': 'count'}
                        ],
                        'completed': [
                            {'$match': {'status': 'done'}},
                            {'$count': 'count'}
                        ]
                    }
            })
            
            # 디버깅: 파이프라인 로깅
            if start_timestamp or end_timestamp:
                logger.info(f"[통계] 집계 파이프라인 실행: start_ts={start_timestamp}, end_ts={end_timestamp}, time_match={time_match}")
            
            result = list(collection.aggregate(pipeline))
            if result:
                stats = result[0]
                total = stats['total'][0]['count'] if stats['total'] else 0
                helmet = stats['helmet'][0]['count'] if stats['helmet'] else 0
                vest = stats['vest'][0]['count'] if stats['vest'] else 0
                fall = stats['fall'][0]['count'] if stats['fall'] else 0
                completed = stats['completed'][0]['count'] if stats['completed'] else 0
                logger.info(f"[통계] 집계 결과: total={total}, helmet={helmet}, vest={vest}, fall={fall}, completed={completed}, start_ts={start_timestamp}, end_ts={end_timestamp}")
                
                # 디버깅: 실제 데이터 샘플 확인 (시간 필터가 적용된 경우)
                if start_timestamp or end_timestamp:
                    try:
                        sample_query = time_match if time_match else {}
                        sample_docs = list(collection.find(sample_query).limit(10))
                        if sample_docs:
                            logger.info(f"[통계] 디버깅 - 시간 필터 적용된 샘플 데이터 ({len(sample_docs)}개):")
                            # 타입별 분류 확인
                            type_counts = {}
                            for doc in sample_docs:
                                type_val = doc.get('type') or doc.get('violation_type') or ''
                                type_counts[type_val] = type_counts.get(type_val, 0) + 1
                                logger.info(f"  - type={doc.get('type')}, violation_type={doc.get('violation_type')}, timestamp={doc.get('timestamp')}")
                            logger.info(f"[통계] 타입별 개수: {type_counts}")
                            
                            # 정규식 테스트
                            import re
                            helmet_pattern = re.compile('안전모|헬멧|helmet|hard.*hat|hardhat', re.IGNORECASE)
                            vest_pattern = re.compile('안전조끼|조끼|vest|safety.*vest|reflective', re.IGNORECASE)
                            fall_pattern = re.compile('넘어짐|낙상|fall|fallen|trip|slip', re.IGNORECASE)
                            
                            helmet_matches = [t for t in type_counts.keys() if helmet_pattern.search(t)]
                            vest_matches = [t for t in type_counts.keys() if vest_pattern.search(t)]
                            fall_matches = [t for t in type_counts.keys() if fall_pattern.search(t)]
                            
                            logger.info(f"[통계] 정규식 매칭 테스트:")
                            logger.info(f"  - 안전모 매칭: {helmet_matches}")
                            logger.info(f"  - 안전조끼 매칭: {vest_matches}")
                            logger.info(f"  - 넘어짐 매칭: {fall_matches}")
                    except Exception as e:
                        logger.debug(f"[통계] 샘플 데이터 조회 실패: {e}")
            else:
                total = helmet = vest = fall = completed = 0
                logger.warning(f"[통계] 집계 결과 없음: start_ts={start_timestamp}, end_ts={end_timestamp}")
            
            kpi = {
                'total': total,
                'helmet': helmet,
                'vest': vest,
                'fall': fall
            }
            
            # 주간 차트 데이터 생성 (최근 7주) - MongoDB aggregate()로 최적화
            chart_data: List[Any] = []
            from datetime import datetime, timedelta
            
            try:
                today = datetime.now()
                # 이번 주의 시작일 (월요일)
                days_since_monday = today.weekday()
                this_week_start = today - timedelta(days=days_since_monday)
                this_week_start = this_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # MongoDB aggregate()로 주차별 통계 계산 (메모리 필터링 대신)
                for week_idx in range(6, -1, -1):  # 6주 전부터 이번 주까지 (총 7주)
                    week_start = this_week_start - timedelta(weeks=week_idx)
                    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
                    
                    # timestamp 범위 계산 (밀리초)
                    week_start_ts = int(week_start.timestamp() * 1000)
                    week_end_ts = int(week_end.timestamp() * 1000)
                    
                    # 해당 주의 통계를 한 번에 계산 (timestamp 우선 사용)
                    week_pipeline = [
                        {
                            '$match': {
                                '$or': [
                                    # timestamp 필드가 있으면 사용 (우선)
                                    {'timestamp': {'$gte': week_start_ts, '$lte': week_end_ts}},
                                    # 하위 호환: timestamp가 없고 violation_datetime만 있는 경우
                                    {
                                        'timestamp': {'$exists': False},
                                        'violation_datetime': {
                                            '$gte': week_start.strftime('%Y-%m-%d'),
                                            '$lte': week_end.strftime('%Y-%m-%d 23:59:59')
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            '$group': {
                                '_id': None,
                                'violations': {'$sum': 1},
                                'helmet': {
                                    '$sum': {
                                        '$cond': [
                                            {'$or': [
                                                {'$regexMatch': {'input': {'$ifNull': ['$type', '']}, 'regex': '안전모|헬멧|helmet|hard.*hat|hardhat', 'options': 'i'}},
                                                {'$regexMatch': {'input': {'$ifNull': ['$violation_type', '']}, 'regex': '안전모|헬멧|helmet|hard.*hat|hardhat', 'options': 'i'}}
                                            ]},
                                            1,
                                            0
                                        ]
                                    }
                                },
                                'vest': {
                                    '$sum': {
                                        '$cond': [
                                            {'$or': [
                                                {'$regexMatch': {'input': {'$ifNull': ['$type', '']}, 'regex': '안전조끼|조끼|vest|safety.*vest|reflective', 'options': 'i'}},
                                                {'$regexMatch': {'input': {'$ifNull': ['$violation_type', '']}, 'regex': '안전조끼|조끼|vest|safety.*vest|reflective', 'options': 'i'}}
                                            ]},
                                            1,
                                            0
                                        ]
                                    }
                                },
                                'fall': {
                                    '$sum': {
                                        '$cond': [
                                            {'$or': [
                                                {'$regexMatch': {'input': {'$ifNull': ['$type', '']}, 'regex': '넘어짐|낙상|fall|fallen|trip|slip', 'options': 'i'}},
                                                {'$regexMatch': {'input': {'$ifNull': ['$violation_type', '']}, 'regex': '넘어짐|낙상|fall|fallen|trip|slip', 'options': 'i'}}
                                            ]},
                                            1,
                                            0
                                        ]
                                    }
                                },
                                'completed': {
                                    '$sum': {
                                        '$cond': [{'$eq': ['$status', 'done']}, 1, 0]
                                    }
                                }
                            }
                        }
                    ]
                    
                    week_result = list(collection.aggregate(week_pipeline))
                    if week_result:
                        week_stats = week_result[0]
                        week_violations = week_stats.get('violations', 0)
                        week_helmet = week_stats.get('helmet', 0)
                        week_fall = week_stats.get('fall', 0)
                        week_completed = week_stats.get('completed', 0)
                    else:
                        week_violations = week_helmet = week_fall = week_completed = 0
                    
                    chart_data.append({
                        'week': week_idx,
                        'violations': week_violations,
                        'helmet_violations': week_helmet,
                        'fall_detections': week_fall,
                        'completed': week_completed
                    })
                
                logger.info(f"[통계] 주간 차트 데이터 생성 완료: {len(chart_data)}개 주차, 총 위반: {sum(w['violations'] for w in chart_data)}건")
            except Exception as chart_error:
                logger.error(f"주간 차트 데이터 생성 오류: {chart_error}", exc_info=True)
                # 에러 발생 시 기본값으로 7주 데이터 생성
                for i in range(7):
                    chart_data.append({
                        'week': i,
                        'violations': 0,
                        'helmet_violations': 0,
                        'fall_detections': 0,
                        'completed': 0
                    })
            
            return {'kpi': kpi, 'chart_data': chart_data}
        except Exception as e:
            logger.error(f"통계 조회 오류: {e}", exc_info=True)
            return {'kpi': {}, 'chart_data': []}

class SimpleWorkerService:
    """간단한 작업자 서비스"""
    
    def __init__(self, db_service: SimpleMongoDBService):
        self.db_service = db_service
    
    def get_all_workers(self) -> List[Dict[str, Any]]:
        """작업자 목록 조회"""
        if not self.db_service.is_connected():
            logger.warning("작업자 조회: MongoDB 연결되지 않음")
            return []
        
        try:
            collection = self.db_service.get_workers_collection()
            if collection is None:
                logger.warning("작업자 조회: worker 컬렉션을 가져올 수 없음")
                return []
            
            # 모든 작업자 조회
            all_workers = list(collection.find({}))
            logger.info(f"작업자 조회: DB에서 총 {len(all_workers)}개 레코드 조회됨")
            
            # 디버깅: 샘플 레코드 상세 로깅
            if all_workers:
                # 처음 3개 레코드 샘플 로깅
                for i, sample in enumerate(all_workers[:3]):
                    worker_id = sample.get('workerId') or sample.get('worker_id', 'N/A')
                    name = sample.get('name') or sample.get('workerName', 'N/A')
                    logger.info(f"  샘플[{i+1}]: workerId={worker_id}, name={name}, keys={list(sample.keys())}")
            else:
                logger.warning("작업자 조회: DB에 레코드가 없습니다!")
            
            # ObjectId를 문자열로 변환 및 필터링
            # 실제 등록된 작업자는 name 또는 workerName 필드가 있어야 함
            workers = []
            filtered_count = 0
            for w in all_workers:
                if '_id' in w:
                    w['_id'] = str(w['_id'])
                
                # 필드명 정규화: worker_id -> workerId (프론트엔드 호환)
                # DB에 worker_id만 있는 경우를 대비하여 명확하게 매핑
                if 'worker_id' in w and w['worker_id']:
                    worker_id_value = str(w['worker_id']).strip()
                    # workerId가 없거나 unknown_으로 시작하는 경우 worker_id로 덮어쓰기
                    if not w.get('workerId') or (w.get('workerId') and str(w.get('workerId', '')).startswith('unknown_')):
                        w['workerId'] = worker_id_value
                        logger.debug(f"작업자 필드 매핑: worker_id={worker_id_value} -> workerId={w['workerId']}")
                elif 'worker_id' in w and 'workerId' not in w:
                    # worker_id가 있지만 workerId가 없는 경우
                    w['workerId'] = w['worker_id']
                
                # name -> workerName 매핑
                if 'name' in w and 'workerName' not in w:
                    w['workerName'] = w['name']
                
                # workerId 확인 (매핑 후)
                worker_id = w.get('workerId') or w.get('worker_id', '')
                
                # name 또는 workerName 확인 (실제 등록된 작업자 여부 판단)
                worker_name = w.get('name') or w.get('workerName') or ''
                
                # 필터링 조건:
                # 1. unknown_으로 시작하는 workerId이면서 name/workerName이 없는 경우만 제외
                #    (인덱스 생성을 위한 임시 레코드)
                # 2. name/workerName이 있으면 실제 작업자로 간주하여 포함
                # 3. name/workerName이 전혀 없는 빈 레코드는 제외
                
                # 이름이 없는 경우 필터링
                if not worker_name or worker_name.strip() == '':
                    if worker_id and worker_id.startswith('unknown_'):
                        # unknown_으로 시작하고 이름도 없는 경우 = 임시 레코드
                        filtered_count += 1
                        logger.debug(f"작업자 필터링: 임시 레코드 제외 - workerId={worker_id}")
                        continue
                    else:
                        # 이름이 없고 unknown_으로 시작하지 않는 경우도 빈 레코드로 간주
                        filtered_count += 1
                        logger.debug(f"작업자 필터링: 이름 없는 빈 레코드 제외 - workerId={worker_id}")
                        continue
                
                # 이름이 있는 경우는 실제 작업자로 간주하여 포함
                if worker_id and worker_id.startswith('unknown_'):
                    logger.debug(f"작업자 포함: unknown_으로 시작하지만 이름 있음 - workerId={worker_id}, name={worker_name}")
                
                workers.append(w)
            
            # 중복 제거: 같은 workerId를 가진 레코드 중 하나만 유지
            # (가장 최근에 업데이트된 레코드 또는 _id가 있는 레코드 우선)
            seen_worker_ids = {}
            unique_workers = []
            duplicate_count = 0
            
            for w in workers:
                worker_id = w.get('workerId') or w.get('worker_id', '')
                if worker_id:
                    if worker_id in seen_worker_ids:
                        duplicate_count += 1
                        logger.debug(f"작업자 중복 제거: workerId={worker_id} (이미 존재)")
                        # 기존 레코드 유지 (첫 번째 것)
                        continue
                    seen_worker_ids[worker_id] = True
                # workerId가 없는 경우는 _id로 구분 (모두 포함)
                unique_workers.append(w)
            
            if duplicate_count > 0:
                logger.info(f"작업자 조회: {duplicate_count}개 중복 레코드 제거됨")
            
            if filtered_count > 0:
                logger.info(f"작업자 조회: {filtered_count}개 임시 레코드 필터링됨")
            
            # 작업자 ID 기준 내림차순 정렬 (숫자로 변환하여 정렬)
            def get_sort_key(worker):
                worker_id = worker.get('workerId') or worker.get('worker_id', '')
                if not worker_id:
                    return -1  # workerId가 없는 경우 맨 뒤로
                
                # 숫자로 변환 시도
                try:
                    # unknown_으로 시작하는 경우 처리
                    if worker_id.startswith('unknown_'):
                        return -1  # unknown_은 맨 뒤로
                    return int(worker_id)
                except (ValueError, TypeError):
                    # 숫자가 아닌 경우 문자열로 비교 (맨 뒤로)
                    return -1
            
            unique_workers.sort(key=get_sort_key, reverse=False)  # 내림차순 (큰 숫자부터)
            
            logger.info(f"작업자 조회: 최종 {len(unique_workers)}개 작업자 반환 (중복 제거 후, ID 내림차순 정렬)")
            return unique_workers
        except Exception as e:
            logger.error(f"작업자 조회 오류: {e}", exc_info=True)
            return []

def get_database():
    """MongoDB 데이터베이스 인스턴스 가져오기"""
    global _db_instance, _violation_service, _worker_service
    
    # 기존 인스턴스가 있으면 연결 상태 확인
    if _db_instance is not None:
        if _db_instance.is_connected():
            return _db_instance
        else:
            # 연결이 끊어진 경우 재연결 시도
            logger.warning("[MongoDB] 기존 연결이 끊어짐 - 재연결 시도")
            _db_instance = None
            _violation_service = None
            _worker_service = None
    
    # 새로 연결 시도
    if _db_instance is None:
        try:
            # 환경 변수에서 MongoDB 설정 가져오기
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            mongo_db_name = os.getenv('MONGO_DB_NAME', 'aivis')
            
            logger.info(f"[MongoDB] 연결 시도: {mongo_uri} (DB: {mongo_db_name})")
            
            # 간단한 MongoDB 서비스 생성
            db_service = SimpleMongoDBService(mongo_uri, mongo_db_name)
            
            if db_service.connect():
                # 서비스 초기화
                _violation_service = SimpleViolationService(db_service)
                _worker_service = SimpleWorkerService(db_service)
                
                # 래퍼 생성 (호환성 유지)
                _db_instance = DatabaseWrapper(db_service, _violation_service, _worker_service)
                logger.info("✅ MongoDB 연결 및 초기화 완료")
            else:
                logger.warning("⚠️  MongoDB 연결 실패 - MongoDB가 실행 중인지 확인하세요")
                logger.warning(f"⚠️  연결 URI: {mongo_uri}")
                logger.warning(f"⚠️  DB 이름: {mongo_db_name}")
                _db_instance = None
                
        except ImportError as import_err:
            logger.warning("=" * 80)
            logger.warning("⚠️  pymongo가 설치되지 않았습니다")
            logger.warning("⚠️  설치: pip install pymongo")
            logger.warning("=" * 80)
            _db_instance = None
        except Exception as e:
            logger.warning("=" * 80)
            logger.warning("⚠️  MongoDB 연결 오류")
            logger.warning(f"⚠️  오류: {e}")
            logger.warning("⚠️  MongoDB 기능이 비활성화됩니다.")
            logger.warning("⚠️  MongoDB 서버가 실행 중인지 확인하세요 (mongodb://localhost:27017/)")
            logger.warning("⚠️  pymongo가 설치되어 있는지 확인하세요 (pip install pymongo)")
            logger.warning("=" * 80)
            logger.error(f"MongoDB 연결 오류 상세: {e}", exc_info=True)
            _db_instance = None
    
    return _db_instance

