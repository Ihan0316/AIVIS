#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB 가상 데이터 생성 스크립트
1년치 샘플 데이터 생성
"""

import os
import sys
from datetime import datetime, timedelta
import random
from collections import defaultdict

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    print("❌ pymongo가 설치되지 않았습니다.")
    print("설치: pip install pymongo")
    sys.exit(1)


def connect_mongodb():
    """MongoDB 연결"""
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'aivis')
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[db_name]
        print(f"✅ MongoDB 연결 성공: {mongo_uri} (DB: {db_name})")
        return client, db
    except Exception as e:
        print(f"❌ MongoDB 연결 실패: {e}")
        sys.exit(1)


def generate_workers(db, count=50):
    """작업자 데이터 생성"""
    print("\n" + "="*80)
    print("👥 WORKER 데이터 생성")
    print("="*80)
    
    collection = db['worker']
    
    # 기존 데이터 확인
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  기존 작업자 데이터가 {existing_count}건 있습니다.")
        response = input("기존 데이터를 삭제하고 새로 생성하시겠습니까? (yes/no): ")
        if response.lower() != 'yes':
            print("작업자 데이터 생성 취소")
            return existing_count
        collection.delete_many({})
        print("✅ 기존 데이터 삭제 완료")
    
    # 팀 목록
    teams = ['A팀', 'B팀', 'C팀', 'D팀']
    roles = ['worker', 'manager']
    
    # 작업자 데이터 생성
    workers = []
    base_worker_id = 1000
    
    # 각 팀별로 작업자 생성
    for team in teams:
        # 매니저 1명
        manager_id = f"{base_worker_id}"
        workers.append({
            'workerId': manager_id,
            'workerName': f"{team} 매니저",
            'team': team,
            'role': 'manager',
            'contact': f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
            'blood_type': random.choice(['A', 'B', 'AB', 'O'])
        })
        base_worker_id += 1
        
        # 일반 작업자들 (팀당 약 10-12명)
        worker_count = random.randint(10, 12)
        for i in range(worker_count):
            worker_id = f"{base_worker_id}"
            workers.append({
                'workerId': worker_id,
                'workerName': f"{team} 작업자{i+1}",
                'team': team,
                'role': 'worker',
                'contact': f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
                'blood_type': random.choice(['A', 'B', 'AB', 'O'])
            })
            base_worker_id += 1
    
    # 데이터 삽입
    if workers:
        result = collection.insert_many(workers)
        print(f"✅ 작업자 데이터 생성 완료: {len(result.inserted_ids)}건")
        return len(result.inserted_ids)
    
    return 0


def generate_violations(db, start_date=None, end_date=None):
    """위반 데이터 생성 (1년치)"""
    print("\n" + "="*80)
    print("📊 VIOLATION 데이터 생성 (1년치)")
    print("="*80)
    
    collection = db['violation']
    
    # 기존 데이터 확인
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  기존 위반 데이터가 {existing_count}건 있습니다.")
        response = input("기존 데이터를 삭제하고 새로 생성하시겠습니까? (yes/no): ")
        if response.lower() != 'yes':
            print("위반 데이터 생성 취소")
            return existing_count
        collection.delete_many({})
        print("✅ 기존 데이터 삭제 완료")
    
    # 날짜 범위 설정
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    print(f"📅 생성 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    # 작업자 목록 가져오기
    workers_collection = db['worker']
    workers = list(workers_collection.find({'role': 'worker'}))
    
    if not workers:
        print("⚠️  작업자 데이터가 없습니다. 먼저 작업자 데이터를 생성하세요.")
        return 0
    
    print(f"👥 작업자 수: {len(workers)}명")
    
    # 위반 유형
    violation_types = [
        '안전모 미착용',
        '안전조끼 미착용',
        '넘어짐'
    ]
    
    # 위반 유형별 심각도
    severity_map = {
        '안전모 미착용': 'high',
        '안전조끼 미착용': 'medium',
        '넘어짐': 'critical'
    }
    
    # 카메라 ID와 구역 매핑
    area_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    # 위반 데이터 생성
    violations = []
    current_date = start_date
    
    # 일별 위반 건수 (평균적으로 하루에 5-15건)
    total_days = (end_date - start_date).days
    print(f"📊 총 {total_days}일치 데이터 생성 중...")
    
    violation_id = 0
    batch_size = 1000
    
    while current_date < end_date:
        # 하루에 생성할 위반 건수 (랜덤)
        daily_count = random.randint(3, 12)
        
        for _ in range(daily_count):
            # 랜덤 시간 생성 (오전 6시 ~ 오후 8시)
            hour = random.randint(6, 20)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            violation_time = current_date.replace(hour=hour, minute=minute, second=second)
            timestamp_ms = int(violation_time.timestamp() * 1000)
            
            # 랜덤 작업자 선택
            worker = random.choice(workers)
            worker_id = worker.get('workerId') or worker.get('worker_id', '')
            worker_name = worker.get('workerName') or worker.get('name', 'Unknown')
            
            # 랜덤 위반 유형
            violation_type = random.choice(violation_types)
            severity = severity_map.get(violation_type, 'medium')
            
            # 랜덤 카메라 ID
            cam_id = random.randint(0, 3)
            work_zone = area_map.get(cam_id, f"A-{cam_id+1}")
            
            # 상태 (대부분 new, 일부 done)
            status = 'new' if random.random() > 0.3 else 'done'
            
            # 이미지 경로
            image_path = f"/images/violation_{violation_time.strftime('%Y%m%d_%H%M%S')}_{worker_id}.jpg"
            
            violation_doc = {
                'timestamp': timestamp_ms,
                'cam_id': cam_id,
                'worker_id': worker_id,
                'worker_name': worker_name,
                'type': violation_type,
                'severity': severity,
                'status': status,
                'image_path': image_path,
                'work_zone': work_zone,
                'processing_time': random.randint(30, 300) if status == 'done' else None,
                'is_face_recognized': random.random() > 0.2,  # 80% 인식 성공
                'face_recognition_status': 'recognized' if random.random() > 0.2 else 'unrecognized',
                'recognized_confidence': round(random.uniform(0.7, 0.99), 3) if random.random() > 0.2 else None
            }
            
            violations.append(violation_doc)
            violation_id += 1
            
            # 배치로 저장
            if len(violations) >= batch_size:
                try:
                    collection.insert_many(violations)
                    print(f"  ✅ {violation_id}건 저장 완료...")
                    violations = []
                except Exception as e:
                    print(f"  ⚠️  저장 오류: {e}")
                    violations = []
        
        # 다음 날로 이동
        current_date += timedelta(days=1)
        
        # 진행 상황 출력 (매 30일마다)
        if (current_date - start_date).days % 30 == 0:
            print(f"  📅 진행: {current_date.strftime('%Y-%m-%d')} ({violation_id}건 생성됨)")
    
    # 남은 데이터 저장
    if violations:
        try:
            collection.insert_many(violations)
            print(f"  ✅ 최종 {len(violations)}건 저장 완료")
        except Exception as e:
            print(f"  ⚠️  저장 오류: {e}")
    
    total_count = collection.count_documents({})
    print(f"\n✅ 위반 데이터 생성 완료: 총 {total_count}건")
    return total_count


def generate_faces(db):
    """얼굴 데이터 생성 (선택적)"""
    print("\n" + "="*80)
    print("👤 FACE 데이터 생성")
    print("="*80)
    
    collection = db['face']
    
    # 기존 데이터 확인
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  기존 얼굴 데이터가 {existing_count}건 있습니다.")
        response = input("기존 데이터를 삭제하고 새로 생성하시겠습니까? (yes/no): ")
        if response.lower() != 'yes':
            print("얼굴 데이터 생성 취소")
            return existing_count
        collection.delete_many({})
        print("✅ 기존 데이터 삭제 완료")
    
    # 작업자 목록 가져오기
    workers_collection = db['worker']
    workers = list(workers_collection.find({}))
    
    if not workers:
        print("⚠️  작업자 데이터가 없습니다. 먼저 작업자 데이터를 생성하세요.")
        return 0
    
    print(f"👥 작업자 수: {len(workers)}명")
    
    # 얼굴 데이터 생성
    faces = []
    timestamp_ms = int(datetime.now().timestamp() * 1000)
    
    for worker in workers:
        worker_id = worker.get('workerId') or worker.get('worker_id', '')
        worker_name = worker.get('workerName') or worker.get('name', 'Unknown')
        
        # 가상 임베딩 생성 (512차원, 0~1 사이 값)
        embedding = [random.uniform(0, 1) for _ in range(512)]
        
        # 이미지 경로
        image_path = f"/images/face_{worker_id}.jpg"
        
        face_doc = {
            'workerId': worker_id,
            'workerName': worker_name,
            'embedding': embedding,
            'image_path': image_path,
            'created_at': timestamp_ms,
            'updated_at': timestamp_ms
        }
        
        faces.append(face_doc)
    
    # 데이터 삽입
    if faces:
        result = collection.insert_many(faces)
        print(f"✅ 얼굴 데이터 생성 완료: {len(result.inserted_ids)}건")
        return len(result.inserted_ids)
    
    return 0


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MongoDB 가상 데이터 생성')
    parser.add_argument('--workers', action='store_true', help='작업자 데이터 생성')
    parser.add_argument('--violations', action='store_true', help='위반 데이터 생성 (1년치)')
    parser.add_argument('--faces', action='store_true', help='얼굴 데이터 생성')
    parser.add_argument('--all', action='store_true', help='모든 데이터 생성')
    parser.add_argument('--yes', action='store_true', help='확인 없이 자동 실행')
    args = parser.parse_args()
    
    print("="*80)
    print("MongoDB 가상 데이터 생성 스크립트")
    print("="*80)
    
    if not args.yes:
        print("\n⚠️  이 스크립트는 기존 데이터를 삭제하고 새로 생성할 수 있습니다.")
        response = input("계속하시겠습니까? (yes/no): ")
        if response.lower() != 'yes':
            print("취소되었습니다.")
            return
    
    client, db = connect_mongodb()
    
    try:
        total_workers = 0
        total_violations = 0
        total_faces = 0
        
        # 작업자 데이터 생성
        if args.all or args.workers:
            total_workers = generate_workers(db)
        
        # 위반 데이터 생성
        if args.all or args.violations:
            total_violations = generate_violations(db)
        
        # 얼굴 데이터 생성
        if args.all or args.faces:
            total_faces = generate_faces(db)
        
        print("\n" + "="*80)
        print("✅ 데이터 생성 완료")
        print("="*80)
        print(f"작업자: {total_workers}건")
        print(f"위반: {total_violations}건")
        print(f"얼굴: {total_faces}건")
        print("="*80)
        
    finally:
        client.close()


if __name__ == '__main__':
    main()

