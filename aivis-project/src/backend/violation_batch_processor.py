"""
위반 사항 배치 처리 모듈
DB 부하를 줄이기 위해 배치로 저장
"""
import asyncio
import logging
import time
import queue
from datetime import datetime
from typing import List, Dict, Any
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from state import (
    violation_batch_queue,
    violation_batch_lock,
    VIOLATION_BATCH_SIZE,
    VIOLATION_BATCH_INTERVAL,
    violation_last_saved
)


async def process_violation_batch():
    """배치 큐에서 위반 사항을 모아서 일괄 저장"""
    batch_items: List[Dict[str, Any]] = []
    last_batch_time = time.time()
    last_log_time = time.time()
    
    logging.info("[배치 처리] 배치 처리 태스크 시작")
    
    while True:
        try:
            # 큐에서 항목 가져오기 (비동기로 실행하여 이벤트 루프 블로킹 방지)
            item = None
            try:
                # queue.Queue.get()은 동기 함수이므로 run_in_executor로 비동기 실행
                loop = asyncio.get_event_loop()
                item = await loop.run_in_executor(None, lambda: violation_batch_queue.get(timeout=0.5))
                batch_items.append(item)
                logging.debug(f"[배치 처리] 큐에서 항목 가져옴 (현재 배치: {len(batch_items)}개)")
            except queue.Empty:
                # 큐가 비어있음 (정상)
                pass
            except Exception as e:
                logging.debug(f"[배치 처리] 큐에서 항목 가져오기 실패: {e}")
            
            current_time = time.time()
            should_flush = False
            
            # 배치 크기 또는 시간 간격 체크
            if len(batch_items) >= VIOLATION_BATCH_SIZE:
                should_flush = True
                logging.info(f"[배치 처리] 배치 크기 도달: {len(batch_items)}개")
            elif len(batch_items) > 0 and (current_time - last_batch_time) >= VIOLATION_BATCH_INTERVAL:
                should_flush = True
                logging.info(f"[배치 처리] 시간 간격 도달: {len(batch_items)}개 (간격: {current_time - last_batch_time:.1f}초)")
            
            if should_flush:
                await flush_batch(batch_items)
                batch_items = []
                last_batch_time = current_time
            
            # 주기적으로 큐 상태 로깅 (10초마다)
            if current_time - last_log_time >= 10.0:
                queue_size = violation_batch_queue.qsize()
                if queue_size > 0 or len(batch_items) > 0:
                    logging.info(f"[배치 처리] 큐 상태: 큐 크기={queue_size}, 현재 배치={len(batch_items)}개")
                last_log_time = current_time
            
            # 짧은 대기 (CPU 사용량 최소화)
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logging.error(f"[배치 처리] 오류: {e}", exc_info=True)
            await asyncio.sleep(1)


async def flush_batch(batch_items: List[Dict[str, Any]]):
    """배치 항목들을 MongoDB에 일괄 저장"""
    if not batch_items:
        return
    
    try:
        # 모든 항목에서 DB 서비스 가져오기 (첫 번째 항목 사용)
        db_service = batch_items[0].get('db_service')
        if not db_service or not db_service.is_connected():
            logging.warning("[배치 처리] DB 서비스가 연결되지 않음")
            return
        
        violations_collection = db_service.get_violations_collection()
        if violations_collection is None:
            logging.warning("[배치 처리] violations 컬렉션을 가져올 수 없음")
            return
        
        # 배치 항목들을 위반 데이터로 변환
        violation_docs: List[Dict[str, Any]] = []
        seen_keys = set()  # 배치 내 중복 체크용
        
        for item in batch_items:
            violations = item.get('violations', [])
            cam_id = item.get('cam_id', 0)
            recognized_faces = item.get('recognized_faces', [])
            
            # 얼굴 정보 매핑
            face_to_worker = {}
            face_to_confidence = {}
            for face in recognized_faces:
                name = face.get("name", "")
                worker_id = face.get("worker_id", "")
                confidence = face.get("confidence", 0.0)
                if name and name != "Unknown" and name != "알 수 없음":
                    face_to_worker[name] = worker_id if worker_id else name
                    face_to_confidence[name] = confidence
            
            current_time = datetime.now()
            violation_datetime = current_time.strftime('%Y-%m-%d %H:%M:%S')
            timestamp_ms = int(current_time.timestamp() * 1000)
            
            # worker 테이블에서 이름으로 workerId 조회하는 함수 (동기 함수, 캐싱 포함)
            worker_name_cache = {}  # 배치 내에서 worker_name -> worker_id 캐싱
            
            def get_worker_id_from_name(worker_name: str) -> str:
                """worker_name으로 worker 테이블에서 숫자 workerId 조회 (캐싱 포함)"""
                if not worker_name or worker_name == "Unknown" or worker_name == "알 수 없음":
                    return "0"
                
                # 캐시 확인
                if worker_name in worker_name_cache:
                    return worker_name_cache[worker_name]
                
                try:
                    workers_collection = db_service.get_workers_collection()
                    # MongoDB Collection 객체는 bool()로 직접 체크할 수 없으므로 is None으로 비교
                    if workers_collection is None:
                        worker_name_cache[worker_name] = "0"
                        return "0"
                    
                    # name 또는 workerName으로 조회 (대소문자 구분)
                    worker = workers_collection.find_one({
                        '$or': [
                            {'name': worker_name},
                            {'workerName': worker_name}
                        ]
                    })
                    
                    if worker:
                        # workerId 반환 (숫자 또는 문자열)
                        worker_id = worker.get('workerId') or worker.get('worker_id')
                        if worker_id:
                            result = str(worker_id)
                            worker_name_cache[worker_name] = result
                            return result
                except Exception as e:
                    logging.warning(f"[배치 처리] worker 조회 실패 (worker_name={worker_name}): {e}")
                
                worker_name_cache[worker_name] = "0"
                return "0"
            
            for violation in violations:
                worker_name = violation.get("worker", "Unknown")
                
                # 1차: face_to_worker에서 찾기
                worker_id = face_to_worker.get(worker_name, None)
                
                # 2차: worker_id가 없거나 worker_name과 같거나 숫자가 아니면 worker 테이블에서 조회
                # worker_id가 숫자인지 확인 (문자열이어도 숫자로 변환 가능한지 체크)
                is_numeric_id = False
                if worker_id:
                    try:
                        # 숫자로 변환 가능한지 확인
                        int(worker_id)
                        is_numeric_id = True
                    except (ValueError, TypeError):
                        is_numeric_id = False
                
                if not worker_id or worker_id == worker_name or not is_numeric_id:
                    # worker_name이 "Unknown" 또는 "알 수 없음"인 경우 특별 처리
                    if worker_name == "Unknown" or worker_name == "알 수 없음":
                        worker_id = "0"
                    else:
                        # worker 테이블에서 조회 (동기 함수이므로 run_in_executor 사용, 캐싱으로 중복 조회 방지)
                        try:
                            loop = asyncio.get_event_loop()
                            worker_id = await loop.run_in_executor(
                                None, 
                                lambda wn=worker_name: get_worker_id_from_name(wn)
                            )
                        except Exception as e:
                            logging.warning(f"[배치 처리] worker 조회 중 오류 (worker_name={worker_name}): {e}")
                            worker_id = "0"
                
                violation_types = violation.get("violations", [])
                
                is_face_recognized = worker_name in face_to_worker and worker_name != "Unknown" and worker_name != "알 수 없음"
                face_recognition_status = "recognized" if is_face_recognized else ("unrecognized" if worker_name != "Unknown" else "no_face")
                recognized_confidence = face_to_confidence.get(worker_name, 0.0) if is_face_recognized else None
                
                for violation_type in violation_types:
                    if not violation_type:
                        continue
                    
                    # 중복 체크 키 생성 (초 단위로 그룹화하여 같은 초 내의 같은 위반은 하나만 저장)
                    # 저장 빈도를 줄이기 위해 초 단위로 그룹화
                    timestamp_sec = timestamp_ms // 1000
                    duplicate_key = f"{worker_id}_{violation_type}_{cam_id}_{timestamp_sec}"
                    if duplicate_key in seen_keys:
                        logging.debug(f"[배치 처리] 배치 내 중복 항목 건너뜀 (초 단위): {duplicate_key}")
                        continue
                    seen_keys.add(duplicate_key)
                    
                    # DB에 이미 존재하는지 확인 (초 단위 범위로 체크하여 같은 초 내의 같은 위반은 하나만 저장)
                    # timestamp_ms를 초 단위 범위로 변환 (같은 초 내의 모든 밀리초 포함)
                    timestamp_sec_start = timestamp_sec * 1000
                    timestamp_sec_end = timestamp_sec_start + 999
                    existing = violations_collection.find_one({
                        'worker_id': worker_id,
                        'type': violation_type,
                        'timestamp': {'$gte': timestamp_sec_start, '$lte': timestamp_sec_end}
                    })
                    if existing:
                        logging.debug(f"[배치 처리] DB에 이미 존재하는 항목 건너뜀 (초 단위): worker_id={worker_id}, type={violation_type}, timestamp_sec={timestamp_sec}")
                        continue
                    
                    # severity 결정
                    severity = "high"
                    if "안전모" in violation_type or "helmet" in violation_type.lower():
                        severity = "high"
                    elif "안전조끼" in violation_type or "vest" in violation_type.lower():
                        severity = "medium"
                    elif "넘어짐" in violation_type or "fall" in violation_type.lower():
                        severity = "critical"
                    else:
                        severity = violation.get("severity", "medium")
                    
                    # 이미지 경로
                    image_path = violation.get("image_path", "")
                    if not image_path:
                        image_filename = f"violation_{current_time.strftime('%Y%m%d_%H%M%S')}_{worker_id}.jpg"
                        image_path = f"/images/{image_filename}"
                    
                    # work_zone 처리 (없으면 cam_id 기반으로 생성)
                    work_zone = violation.get("work_zone", "")
                    if not work_zone:
                        area_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                        work_zone = area_map.get(cam_id, f"A-{cam_id+1}")
                    
                    # 표준화된 스키마 (프론트엔드 기준 필드명 사용)
                    violation_doc = {
                        # 표준 필드 (프론트엔드 우선 필드명)
                        'timestamp': timestamp_ms,  # 숫자 타임스탬프 (인덱스용)
                        'violation_datetime': violation_datetime,  # 문자열 형식 날짜시간 (프론트엔드 호환)
                        'cam_id': cam_id,  # 표준 필드명
                        'worker_id': worker_id,
                        'worker_name': worker_name,
                        'type': violation_type,  # 표준 필드명
                        'severity': severity,
                        'status': 'new',
                        'image_path': image_path,
                        'work_zone': work_zone,  # cam_id 기반으로 생성된 값 또는 기존 값
                        'processing_time': None,
                        
                        # 얼굴 인식 상태
                        'is_face_recognized': is_face_recognized,
                        'face_recognition_status': face_recognition_status,
                        'recognized_confidence': recognized_confidence
                    }
                    
                    violation_docs.append(violation_doc)
        
        if not violation_docs:
            logging.debug("[배치 처리] 저장할 위반 사항이 없음 (중복 제거 후)")
            return
        
        # bulk_write로 일괄 삽입 (고유 인덱스로 중복 방지)
        try:
            from pymongo import InsertOne
            
            # InsertOne으로 일괄 삽입 (고유 인덱스가 중복을 자동으로 방지)
            operations = []
            for doc in violation_docs:
                operations.append(InsertOne(doc))
            
            if operations:
                result = violations_collection.bulk_write(operations, ordered=False)
                saved_count = result.inserted_count
                logging.info(f"[배치 처리] 일괄 저장 완료: {saved_count}개 저장, 총 {len(operations)}개 처리")
                
                # 저장된 데이터 샘플 로깅 (디버깅용)
                if saved_count > 0:
                    # 최근 저장된 문서 하나 가져오기
                    try:
                        sample_doc = violations_collection.find_one(
                            {'timestamp': {'$gte': timestamp_ms - 1000}},
                            sort=[('timestamp', -1)]
                        )
                        if sample_doc:
                            logging.debug(f"[배치 처리] 저장된 데이터 샘플: worker_id={sample_doc.get('worker_id')}, type={sample_doc.get('type')}, timestamp={sample_doc.get('timestamp')}, cam_id={sample_doc.get('cam_id')}")
                    except Exception as e:
                        logging.debug(f"[배치 처리] 샘플 데이터 조회 실패: {e}")
        
        except BulkWriteError as e:
            # 일부 실패해도 계속 진행 (중복 키 오류는 정상)
            saved_count = e.details.get('nInserted', 0)
            duplicate_errors = sum(1 for err in e.details.get('writeErrors', []) if err.get('code') == 11000)
            if duplicate_errors > 0:
                logging.debug(f"[배치 처리] {duplicate_errors}개 중복 항목 건너뜀 (정상), {saved_count}개 저장됨")
            else:
                logging.warning(f"[배치 처리] 일부 저장 실패: {saved_count}개 저장됨, 오류: {e.details}")
        except Exception as e:
            logging.error(f"[배치 처리] 일괄 저장 오류: {e}", exc_info=True)
    
    except Exception as e:
        logging.error(f"[배치 처리] 배치 플러시 오류: {e}", exc_info=True)

