# -*- coding: utf-8 -*-
"""
3개월치 가상 위반 데이터 생성 스크립트
MongoDB violation 컬렉션에 데이터 삽입
"""

import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

# MongoDB 연결
def get_mongo_client():
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        mongo_db_name = os.getenv('MONGO_DB_NAME', 'aivis')
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # 연결 테스트
        
        print(f"✅ MongoDB 연결 성공: {mongo_uri} (DB: {mongo_db_name})")
        return client[mongo_db_name]
    except Exception as e:
        print(f"❌ MongoDB 연결 실패: {e}")
        return None


def generate_fake_violations(days: int = 90) -> List[Dict[str, Any]]:
    """
    가상 위반 데이터 생성
    
    Args:
        days: 생성할 기간 (일수), 기본 90일 (3개월)
    
    Returns:
        생성된 위반 데이터 리스트
    """
    
    # 작업자 목록 (실제 DB에 있는 작업자 또는 가상 작업자)
    workers = [
        {"worker_id": "1", "worker_name": "유승원"},
        {"worker_id": "2", "worker_name": "조이한"},
        {"worker_id": "3", "worker_name": "김철수"},
        {"worker_id": "4", "worker_name": "이영희"},
        {"worker_id": "5", "worker_name": "박민수"},
        {"worker_id": "unknown", "worker_name": "알수없음"},
    ]
    
    # 위반 유형
    violation_types = [
        {"type": "안전모", "violation_type": "안전모 미착용"},
        {"type": "안전조끼", "violation_type": "안전조끼 미착용"},
        {"type": "넘어짐", "violation_type": "넘어짐 감지"},
        {"type": "안전모, 안전조끼", "violation_type": "안전모, 안전조끼 미착용"},
    ]
    
    # 위반 유형별 가중치 (발생 빈도)
    violation_weights = [40, 35, 10, 15]  # 안전모 > 안전조끼 > 복합 > 넘어짐
    
    # 카메라 ID
    cam_ids = [0, 1]
    
    # 작업 구역
    work_zones = ["A-1", "A-2", "B-1", "B-2"]
    
    # 상태
    statuses = ["pending", "done", "done", "done"]  # done이 더 많음
    
    violations = []
    
    # 현재 시간 기준으로 과거 데이터 생성
    now = datetime.now()
    start_date = now - timedelta(days=days)
    
    # 일별 위반 수 범위 (주말은 적게)
    weekday_violations = (5, 20)  # 평일: 5~20건
    weekend_violations = (1, 8)   # 주말: 1~8건
    
    print(f"📅 데이터 생성 기간: {start_date.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')} ({days}일)")
    
    current_date = start_date
    total_violations = 0
    
    while current_date <= now:
        # 주말/평일 구분
        is_weekend = current_date.weekday() >= 5
        min_v, max_v = weekend_violations if is_weekend else weekday_violations
        
        # 해당 일의 위반 수 결정
        daily_violations = random.randint(min_v, max_v)
        
        for _ in range(daily_violations):
            # 근무 시간 내 랜덤 시간 (08:00 ~ 18:00)
            hour = random.randint(8, 17)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            violation_time = current_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
            timestamp_ms = int(violation_time.timestamp() * 1000)
            
            # 랜덤 선택
            worker = random.choice(workers)
            violation = random.choices(violation_types, weights=violation_weights)[0]
            cam_id = random.choice(cam_ids)
            work_zone = random.choice(work_zones)
            status = random.choice(statuses)
            
            # 위반 데이터 생성
            violation_data = {
                "timestamp": timestamp_ms,
                "violation_datetime": violation_time.strftime('%Y-%m-%d %H:%M:%S'),
                "worker_id": worker["worker_id"],
                "worker_name": worker["worker_name"],
                "type": violation["type"],
                "violation_type": violation["violation_type"],
                "cam_id": cam_id,
                "camera_id": cam_id,  # 호환성
                "work_zone": work_zone,
                "status": status,
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "image_path": f"logs/{violation_time.strftime('%Y%m%d_%H%M%S')}_CAM{cam_id}_{worker['worker_name']}_{violation['type']}.jpg",
                "created_at": violation_time.isoformat(),
                "is_fake": True  # 가상 데이터 표시
            }
            
            violations.append(violation_data)
            total_violations += 1
        
        current_date += timedelta(days=1)
    
    print(f"✅ 총 {total_violations}개 가상 위반 데이터 생성 완료")
    return violations


def insert_violations(db, violations: List[Dict[str, Any]]) -> int:
    """
    MongoDB에 위반 데이터 삽입
    
    Args:
        db: MongoDB 데이터베이스 객체
        violations: 삽입할 위반 데이터 리스트
    
    Returns:
        삽입된 문서 수
    """
    if not violations:
        print("⚠️ 삽입할 데이터가 없습니다")
        return 0
    
    try:
        collection = db['violation']
        
        # 기존 가상 데이터 삭제 (is_fake=True)
        deleted = collection.delete_many({"is_fake": True})
        if deleted.deleted_count > 0:
            print(f"🗑️ 기존 가상 데이터 {deleted.deleted_count}개 삭제")
        
        # 새 데이터 삽입
        result = collection.insert_many(violations)
        inserted_count = len(result.inserted_ids)
        
        print(f"✅ {inserted_count}개 문서 삽입 완료")
        return inserted_count
    except Exception as e:
        print(f"❌ 데이터 삽입 실패: {e}")
        return 0


def print_statistics(violations: List[Dict[str, Any]]):
    """통계 출력"""
    print("\n" + "=" * 50)
    print("📊 생성된 데이터 통계")
    print("=" * 50)
    
    # 위반 유형별 통계
    type_counts = {}
    for v in violations:
        t = v.get('type', 'Unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\n🔹 위반 유형별:")
    for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {t}: {count}건")
    
    # 작업자별 통계
    worker_counts = {}
    for v in violations:
        w = v.get('worker_name', 'Unknown')
        worker_counts[w] = worker_counts.get(w, 0) + 1
    
    print("\n🔹 작업자별:")
    for w, count in sorted(worker_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {w}: {count}건")
    
    # 상태별 통계
    status_counts = {}
    for v in violations:
        s = v.get('status', 'Unknown')
        status_counts[s] = status_counts.get(s, 0) + 1
    
    print("\n🔹 상태별:")
    for s, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {s}: {count}건")
    
    # 카메라별 통계
    cam_counts = {}
    for v in violations:
        c = v.get('cam_id', 'Unknown')
        cam_counts[c] = cam_counts.get(c, 0) + 1
    
    print("\n🔹 카메라별:")
    for c, count in sorted(cam_counts.items(), key=lambda x: str(x)):
        print(f"   - CAM-{c}: {count}건")
    
    print("\n" + "=" * 50)


def main():
    print("=" * 60)
    print("🔧 3개월치 가상 위반 데이터 생성 스크립트")
    print("=" * 60)
    
    # MongoDB 연결
    db = get_mongo_client()
    if db is None:
        print("❌ MongoDB 연결 실패. 스크립트를 종료합니다.")
        return
    
    # 가상 데이터 생성 (3개월 = 90일)
    violations = generate_fake_violations(days=90)
    
    # 통계 출력
    print_statistics(violations)
    
    # 사용자 확인
    print("\n⚠️ 위 데이터를 MongoDB에 삽입하시겠습니까?")
    print("   (기존 가상 데이터는 삭제됩니다)")
    confirm = input("   [y/N]: ").strip().lower()
    
    if confirm == 'y':
        inserted = insert_violations(db, violations)
        print(f"\n✅ 완료! {inserted}개 문서가 삽입되었습니다.")
    else:
        print("\n❌ 취소되었습니다.")


if __name__ == "__main__":
    main()

