"""
오늘 날짜 기준 위반 데이터 생성 스크립트
기존 DB 형식을 참고하여 오늘 날짜로만 생성
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))

import random
from datetime import datetime, timedelta
from pymongo import MongoClient

# MongoDB 연결 설정
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('MONGODB_DB_NAME', 'aivis')

# 위반 유형
VIOLATION_TYPES = ['안전모', '안전조끼', '넘어짐']

# 작업 구역
AREAS = ['A-1', 'A-2', 'B-1', 'B-2']

def generate_today_violations():
    """오늘 날짜 기준 위반 데이터 생성"""
    
    # MongoDB 연결
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    violations_collection = db['violation']
    workers_collection = db['worker']
    
    print(f"MongoDB 연결: {MONGO_URI}/{DB_NAME}")
    
    # 기존 violation 데이터 형식 확인
    sample = violations_collection.find_one()
    if sample:
        print("\n기존 데이터 형식 확인:")
        print(f"  필드: {list(sample.keys())}")
    
    # worker 컬렉션에서 작업자 목록 가져오기
    workers_cursor = workers_collection.find({})
    workers_list = list(workers_cursor)
    
    if not workers_list:
        print("❌ worker 컬렉션에 작업자가 없습니다!")
        print("기본 작업자로 진행합니다: 정준성, 조이한, 유승원")
        WORKERS = [
            {'name': '정준성', 'worker_id': '정준성', 'team': 'A팀'},
            {'name': '조이한', 'worker_id': '조이한', 'team': 'B팀'},
            {'name': '유승원', 'worker_id': '유승원', 'team': 'B팀'}
        ]
    else:
        WORKERS = workers_list
        print(f"\n✅ worker 컬렉션에서 {len(WORKERS)}명의 작업자를 가져왔습니다:")
        for w in WORKERS:
            print(f"   - {w.get('name', 'N/A')} (ID: {w.get('worker_id', 'N/A')}, 팀: {w.get('team', 'N/A')})")
    
    # 오늘 날짜의 기존 데이터 삭제
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    delete_result = violations_collection.delete_many({
        'violation_datetime': {'$regex': f'^{today_str}'}
    })
    print(f"\n기존 오늘({today_str}) 데이터 삭제: {delete_result.deleted_count}건")
    
    # 오늘 하루에 랜덤하게 10~30건의 위반 생성
    daily_violations = random.randint(10, 30)
    
    records = []
    current_time = datetime.now()
    
    print(f"\n오늘({today_str}) 데이터 생성 시작...")
    
    # 오늘 00:00부터 현재 시간까지
    start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    for i in range(daily_violations):
        # 랜덤 작업자 선택
        worker_data = random.choice(WORKERS)
        worker_name = worker_data.get('name', worker_data.get('worker_id', 'Unknown'))
        worker_id = worker_data.get('worker_id', worker_name)
        worker_team = worker_data.get('team', 'A팀')
        
        # 랜덤 위반 유형 (1개만)
        violation_type = random.choice(VIOLATION_TYPES)
        
        # 랜덤 시간 (00:00 ~ 현재 시간)
        random_seconds = random.randint(0, int((current_time - start_of_day).total_seconds()))
        violation_time = start_of_day + timedelta(seconds=random_seconds)
        
        # 랜덤 구역
        area = random.choice(AREAS)
        
        # 랜덤 카메라
        cam_id = random.randint(0, 3)
        
        # 상태: 대부분 new, 일부 done
        status = random.choice(['new', 'new', 'new', 'done'])  # 75% new
        
        # 위반 유형별 심각도 설정
        if violation_type == '넘어짐':
            severity = 'critical'
        elif violation_type == '안전모':
            severity = 'high'
        else:
            severity = 'medium'
        
        # 작업 구역 (A, B 형식)
        work_zone = area.split('-')[0] if '-' in area else area
        
        # 타임스탬프 (밀리초)
        timestamp_ms = int(violation_time.timestamp() * 1000)
        
        # 이미지 경로 생성
        image_filename = f"{violation_time.strftime('%Y%m%d_%H%M%S')}_CAM{cam_id}_{worker_name}_{violation_type}.jpg"
        image_path = f"C:\\Users\\ihan\\Desktop\\aivis-project-mac-windows\\logs\\{image_filename}"
        
        record = {
            'timestamp': timestamp_ms,
            'violation_datetime': violation_time.strftime('%Y-%m-%d %H:%M:%S'),
            'cam_id': cam_id,
            'worker_id': str(worker_id),
            'worker_name': worker_name,
            'type': violation_type,
            'severity': severity,
            'status': status,
            'image_path': image_path,
            'work_zone': work_zone,
            'processing_time': None,
            'is_face_recognized': True,
            'face_recognition_status': 'recognized',
            'recognized_confidence': round(random.uniform(0.5, 0.9), 2),
            'is_test_data': True  # 테스트 데이터 표시
        }
        
        records.append(record)
    
    print(f"총 {len(records)}건의 테스트 데이터 생성")
    
    # MongoDB에 저장
    if records:
        result = violations_collection.insert_many(records)
        print(f"✅ MongoDB에 {len(result.inserted_ids)}건 저장 완료!")
    
    # 통계 출력
    print("\n=== 생성된 데이터 통계 ===")
    
    # 작업자별 통계
    print("\n[작업자별 위반 건수]")
    worker_stats = {}
    for r in records:
        worker_id = r['worker_id']
        worker_stats[worker_id] = worker_stats.get(worker_id, 0) + 1
    
    for worker_data in WORKERS:
        worker_id = worker_data.get('worker_id', worker_data.get('name', 'Unknown'))
        worker_name = worker_data.get('name', worker_id)
        count = worker_stats.get(str(worker_id), 0)
        print(f"  - {worker_name}: {count}건")
    
    # 위반 유형별 통계
    print("\n[위반 유형별 건수]")
    for vtype in VIOLATION_TYPES:
        count = len([r for r in records if r['type'] == vtype])
        print(f"  - {vtype}: {count}건")
    
    # 상태별 통계
    print("\n[상태별 건수]")
    status_stats = {}
    for r in records:
        status = r['status']
        status_stats[status] = status_stats.get(status, 0) + 1
    for status, count in status_stats.items():
        print(f"  - {status}: {count}건")
    
    client.close()
    print("\n✅ 완료!")

if __name__ == "__main__":
    generate_today_violations()

