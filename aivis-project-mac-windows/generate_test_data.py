"""
3개월치 테스트 위반 데이터 생성 스크립트
MongoDB worker 컬렉션에서 작업자 목록을 가져와서 랜덤 생성
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))

import asyncio
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

def generate_test_data():
    """3개월치 테스트 데이터 생성"""
    
    # MongoDB 연결
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    violations_collection = db['violation']
    workers_collection = db['worker']
    
    print(f"MongoDB 연결: {MONGO_URI}/{DB_NAME}")
    
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
        print(f"✅ worker 컬렉션에서 {len(WORKERS)}명의 작업자를 가져왔습니다:")
        for w in WORKERS:
            print(f"   - {w.get('name', 'N/A')} (ID: {w.get('worker_id', 'N/A')}, 팀: {w.get('team', 'N/A')})")
    
    # 기존 테스트 데이터 삭제 (선택적)
    # violations_collection.delete_many({'is_test_data': True})
    
    # 3개월 전부터 오늘까지
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    records = []
    current_date = start_date
    
    print(f"\n데이터 생성 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    while current_date <= end_date:
        # 하루에 랜덤하게 5~20건의 위반 생성
        daily_violations = random.randint(5, 20)
        
        # 주말에는 위반이 적음
        if current_date.weekday() >= 5:  # 토, 일
            daily_violations = random.randint(0, 5)
        
        for _ in range(daily_violations):
            # 랜덤 작업자 선택 (worker 컬렉션에서)
            worker_data = random.choice(WORKERS)
            worker_name = worker_data.get('name', worker_data.get('worker_id', 'Unknown'))
            worker_id = worker_data.get('worker_id', worker_name)
            worker_team = worker_data.get('team', 'A팀')
            
            # 랜덤 위반 유형 (1~2개)
            num_violations = random.randint(1, 2)
            violation_types = random.sample(VIOLATION_TYPES, min(num_violations, len(VIOLATION_TYPES)))
            
            # 랜덤 시간 (08:00 ~ 18:00 근무시간)
            hour = random.randint(8, 17)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            violation_time = current_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
            
            # 랜덤 구역
            area = random.choice(AREAS)
            
            # 랜덤 카메라
            cam_id = random.randint(0, 3)
            
            # 상태: 대부분 done, 일부 new
            if current_date.date() == end_date.date():
                status = random.choice(['new', 'new', 'done'])  # 오늘은 new가 많음
            else:
                status = 'done'  # 과거 데이터는 done
            
            # 위반 유형별 심각도 설정
            violation_type = violation_types[0] if violation_types else '위반'
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
            
            # 이미지 경로 생성 (테스트 데이터용)
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
        
        current_date += timedelta(days=1)
    
    print(f"총 {len(records)}건의 테스트 데이터 생성")
    
    # MongoDB에 저장
    if records:
        result = violations_collection.insert_many(records)
        print(f"✅ MongoDB에 {len(result.inserted_ids)}건 저장 완료!")
    
    # 통계 출력
    print("\n=== 생성된 데이터 통계 ===")
    
    # 작업자별 통계
    print("\n[작업자별 위반 건수]")
    for worker_data in WORKERS:
        worker_name = worker_data.get('name', worker_data.get('worker_id', 'Unknown'))
        worker_id = worker_data.get('worker_id', worker_name)
        count = len([r for r in records if r['worker_id'] == worker_id])
        print(f"  - {worker_name}: {count}건")
    
    # 위반 유형별 통계
    print("\n[위반 유형별 건수]")
    for vtype in VIOLATION_TYPES:
        count = len([r for r in records if r['type'] == vtype])
        print(f"  - {vtype}: {count}건")
    
    # 월별 통계
    print("\n[월별 위반 건수]")
    monthly_stats = {}
    for r in records:
        month_key = r['created_at'].strftime('%Y-%m')
        monthly_stats[month_key] = monthly_stats.get(month_key, 0) + 1
    
    for month, count in sorted(monthly_stats.items()):
        print(f"  - {month}: {count}건")
    
    client.close()
    print("\n✅ 완료!")

if __name__ == "__main__":
    generate_test_data()

