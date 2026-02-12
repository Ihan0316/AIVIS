"""
MongoDB 스키마 마이그레이션 스크립트
기존 데이터에 가이드 스키마 필드 추가
"""
import sys
import os
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    print("❌ pymongo가 설치되지 않았습니다.")
    print("설치: pip install pymongo")
    sys.exit(1)


def migrate_violations():
    """violation 컬렉션 마이그레이션"""
    try:
        # MongoDB 연결
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        db_name = os.getenv('MONGO_DB_NAME', 'aivis')
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        
        # 연결 테스트
        client.admin.command('ping')
        print("✅ MongoDB 연결 성공")
        
        violations_collection = db['violation']
        
        # 마이그레이션 대상 문서 찾기
        query = {
            '$or': [
                {'timestamp': {'$exists': False}},
                {'cam_id': {'$exists': False}},
                {'worker_name': {'$exists': False}},
                {'type': {'$exists': False}},
                {'severity': {'$exists': False}},
                {'is_face_recognized': {'$exists': False}},
                {'face_recognition_status': {'$exists': False}},
                {'recognized_confidence': {'$exists': False}}
            ]
        }
        
        total = violations_collection.count_documents(query)
        print(f"\n📊 마이그레이션 대상: {total}건")
        
        if total == 0:
            print("✅ 모든 문서가 이미 마이그레이션되었습니다.")
            return
        
        # 마이그레이션 실행
        updated = 0
        skipped = 0
        
        for doc in violations_collection.find(query):
            try:
                update_fields = {}
                
                # timestamp 추가 (violation_datetime에서 변환)
                if 'timestamp' not in doc:
                    if 'violation_datetime' in doc:
                        try:
                            dt_str = doc['violation_datetime']
                            if isinstance(dt_str, str):
                                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                                update_fields['timestamp'] = int(dt.timestamp() * 1000)
                        except:
                            # 현재 시간 사용
                            update_fields['timestamp'] = int(datetime.now().timestamp() * 1000)
                    else:
                        update_fields['timestamp'] = int(datetime.now().timestamp() * 1000)
                
                # cam_id 추가 (camera_id에서 복사)
                if 'cam_id' not in doc:
                    if 'camera_id' in doc:
                        update_fields['cam_id'] = doc['camera_id']
                    else:
                        update_fields['cam_id'] = 0
                
                # worker_name 추가 (worker_id와 동일하게 설정 또는 조회)
                if 'worker_name' not in doc or not doc.get('worker_name'):
                    worker_id = doc.get('worker_id', '')
                    if worker_id:
                        # worker 컬렉션에서 이름 조회
                        workers_collection = db['worker']
                        worker = workers_collection.find_one({'worker_id': worker_id})
                        if worker and worker.get('name'):
                            update_fields['worker_name'] = worker['name']
                        else:
                            update_fields['worker_name'] = worker_id
                    else:
                        update_fields['worker_name'] = 'Unknown'
                
                # type 추가 (violation_type에서 복사)
                if 'type' not in doc:
                    if 'violation_type' in doc:
                        update_fields['type'] = doc['violation_type']
                    else:
                        update_fields['type'] = 'Unknown'
                
                # severity 추가 (위반 유형에 따라 결정)
                if 'severity' not in doc:
                    violation_type = doc.get('type') or doc.get('violation_type', '')
                    if "안전모" in violation_type or "helmet" in violation_type.lower():
                        update_fields['severity'] = "high"
                    elif "안전조끼" in violation_type or "vest" in violation_type.lower():
                        update_fields['severity'] = "medium"
                    elif "넘어짐" in violation_type or "fall" in violation_type.lower():
                        update_fields['severity'] = "critical"
                    else:
                        update_fields['severity'] = "medium"
                
                # 얼굴 인식 상태 필드 추가
                if 'is_face_recognized' not in doc:
                    worker_name = doc.get('worker_name', '') or update_fields.get('worker_name', '')
                    worker_id = doc.get('worker_id', '')
                    # worker_name이 "Unknown"이 아니고 worker_id가 있으면 인식된 것으로 간주
                    is_recognized = (worker_name and worker_name != "Unknown" and 
                                   worker_name != "알 수 없음" and worker_name != "unknown" and
                                   worker_id and worker_id != "unknown")
                    update_fields['is_face_recognized'] = is_recognized
                
                if 'face_recognition_status' not in doc:
                    worker_name = doc.get('worker_name', '') or update_fields.get('worker_name', '')
                    if not worker_name or worker_name == "Unknown" or worker_name == "알 수 없음":
                        update_fields['face_recognition_status'] = "no_face"
                    elif update_fields.get('is_face_recognized', False) or doc.get('is_face_recognized', False):
                        update_fields['face_recognition_status'] = "recognized"
                    else:
                        update_fields['face_recognition_status'] = "unrecognized"
                
                if 'recognized_confidence' not in doc:
                    # 기존 데이터에는 신뢰도 정보가 없으므로 null로 설정
                    update_fields['recognized_confidence'] = None
                
                # 업데이트 실행
                if update_fields:
                    violations_collection.update_one(
                        {'_id': doc['_id']},
                        {'$set': update_fields}
                    )
                    updated += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"⚠️ 문서 업데이트 오류 (_id: {doc.get('_id')}): {e}")
                skipped += 1
        
        print(f"\n✅ 마이그레이션 완료:")
        print(f"   - 업데이트: {updated}건")
        print(f"   - 건너뜀: {skipped}건")
        
        # 인덱스 생성
        print("\n📑 인덱스 생성 중...")
        violations_collection.create_index([('timestamp', -1)])
        violations_collection.create_index([('cam_id', 1), ('timestamp', -1)])
        violations_collection.create_index([('worker_id', 1), ('timestamp', 1), ('type', 1)], unique=False)
        violations_collection.create_index([('type', 1)])
        violations_collection.create_index([('severity', 1)])
        violations_collection.create_index([('is_face_recognized', 1)])
        violations_collection.create_index([('face_recognition_status', 1)])
        print("✅ 인덱스 생성 완료")
        
    except Exception as e:
        print(f"❌ 마이그레이션 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    print("=" * 60)
    print("MongoDB 스키마 마이그레이션")
    print("=" * 60)
    print()
    print("기존 violation 문서에 가이드 스키마 필드 추가:")
    print("  - timestamp (밀리초)")
    print("  - cam_id")
    print("  - worker_name")
    print("  - type")
    print("  - severity")
    print()
    
    response = input("마이그레이션을 시작하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        sys.exit(0)
    
    migrate_violations()
    
    print()
    print("=" * 60)
    print("마이그레이션 완료!")
    print("=" * 60)

