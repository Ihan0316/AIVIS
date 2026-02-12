#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB 데이터 확인 및 분석 스크립트
중복 데이터 및 불필요한 데이터 확인
"""

import os
import sys
from datetime import datetime
from collections import defaultdict
from pprint import pprint

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


def check_violations_collection(db):
    """violation 컬렉션 확인"""
    print("\n" + "="*80)
    print("📊 VIOLATION 컬렉션 분석")
    print("="*80)
    
    collection = db['violation']
    total_count = collection.count_documents({})
    print(f"\n총 문서 수: {total_count}건")
    
    if total_count == 0:
        print("⚠️  데이터가 없습니다.")
        return
    
    # 샘플 데이터 확인
    print("\n📋 샘플 데이터 (최근 3건):")
    sample_docs = list(collection.find().sort('timestamp', -1).limit(3))
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n--- 샘플 {i} ---")
        print(f"  _id: {doc.get('_id')}")
        print(f"  worker_id: {doc.get('worker_id')}")
        print(f"  worker_name: {doc.get('worker_name')}")
        print(f"  type: {doc.get('type')}")
        print(f"  violation_type: {doc.get('violation_type')}")
        print(f"  cam_id: {doc.get('cam_id')}")
        print(f"  camera_id: {doc.get('camera_id')}")
        print(f"  timestamp: {doc.get('timestamp')}")
        print(f"  violation_datetime: {doc.get('violation_datetime')}")
        print(f"  status: {doc.get('status')}")
        print(f"  모든 필드: {list(doc.keys())}")
    
    # 필드 사용 현황 분석
    print("\n📊 필드 사용 현황:")
    field_stats = {}
    all_docs = collection.find({})
    for doc in all_docs:
        for key in doc.keys():
            if key != '_id':
                if key not in field_stats:
                    field_stats[key] = 0
                field_stats[key] += 1
    
    for field, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_count) * 100
        print(f"  {field}: {count}건 ({percentage:.1f}%)")
    
    # 중복 데이터 확인 (worker_id + type + timestamp 조합)
    print("\n🔍 중복 데이터 확인:")
    duplicates = defaultdict(list)
    all_docs = collection.find({})
    for doc in all_docs:
        worker_id = doc.get('worker_id', '')
        violation_type = doc.get('type') or doc.get('violation_type', '')
        timestamp = doc.get('timestamp')
        
        # timestamp를 초 단위로 정규화 (밀리초인 경우)
        if timestamp:
            if isinstance(timestamp, (int, float)):
                # 밀리초를 초로 변환 (같은 초 내 중복 체크)
                timestamp_sec = int(timestamp / 1000) if timestamp > 1e12 else int(timestamp)
            else:
                timestamp_sec = str(timestamp)
        else:
            timestamp_sec = None
        
        key = (worker_id, violation_type, timestamp_sec)
        duplicates[key].append(doc.get('_id'))
    
    duplicate_count = 0
    for key, ids in duplicates.items():
        if len(ids) > 1:
            duplicate_count += len(ids) - 1
            print(f"  중복 발견: worker_id={key[0]}, type={key[1]}, timestamp={key[2]}")
            print(f"    중복 개수: {len(ids)}건 (IDs: {ids[:5]})")
    
    if duplicate_count == 0:
        print("  ✅ 중복 데이터 없음")
    else:
        print(f"\n  ⚠️  총 중복 데이터: {duplicate_count}건")
    
    # 불필요한 필드 확인 (하위 호환 필드)
    print("\n🔍 하위 호환 필드 확인:")
    compatibility_fields = {
        'violation_type': 'type',
        'camera_id': 'cam_id',
        'violation_datetime': 'timestamp'
    }
    
    for old_field, new_field in compatibility_fields.items():
        old_count = collection.count_documents({old_field: {'$exists': True}})
        new_count = collection.count_documents({new_field: {'$exists': True}})
        print(f"  {old_field} (구 필드): {old_count}건")
        print(f"  {new_field} (신 필드): {new_count}건")
        if old_count > 0 and new_count > 0:
            print(f"    ⚠️  두 필드 모두 사용 중 (마이그레이션 필요)")
    
    # 빈 값 또는 null 값 확인
    print("\n🔍 빈 값 또는 null 값 확인:")
    empty_fields = {}
    all_docs = collection.find({})
    for doc in all_docs:
        for key, value in doc.items():
            if key == '_id':
                continue
            if value is None or value == '' or (isinstance(value, list) and len(value) == 0):
                if key not in empty_fields:
                    empty_fields[key] = 0
                empty_fields[key] += 1
    
    if empty_fields:
        for field, count in sorted(empty_fields.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_count) * 100
            print(f"  {field}: {count}건 ({percentage:.1f}%)")
    else:
        print("  ✅ 빈 값 없음")


def check_workers_collection(db):
    """worker 컬렉션 확인"""
    print("\n" + "="*80)
    print("👥 WORKER 컬렉션 분석")
    print("="*80)
    
    collection = db['worker']
    total_count = collection.count_documents({})
    print(f"\n총 문서 수: {total_count}건")
    
    if total_count == 0:
        print("⚠️  데이터가 없습니다.")
        return
    
    # 샘플 데이터 확인
    print("\n📋 샘플 데이터 (최근 3건):")
    sample_docs = list(collection.find().limit(3))
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n--- 샘플 {i} ---")
        print(f"  _id: {doc.get('_id')}")
        print(f"  workerId: {doc.get('workerId')}")
        print(f"  worker_id: {doc.get('worker_id')}")
        print(f"  workerName: {doc.get('workerName')}")
        print(f"  name: {doc.get('name')}")
        print(f"  team: {doc.get('team')}")
        print(f"  role: {doc.get('role')}")
        print(f"  모든 필드: {list(doc.keys())}")
    
    # 중복 workerId 확인
    print("\n🔍 중복 workerId 확인:")
    worker_ids = defaultdict(list)
    all_docs = collection.find({})
    for doc in all_docs:
        worker_id = doc.get('workerId') or doc.get('worker_id', '')
        if worker_id:
            worker_ids[worker_id].append(doc.get('_id'))
    
    duplicate_count = 0
    for worker_id, ids in worker_ids.items():
        if len(ids) > 1:
            duplicate_count += len(ids) - 1
            print(f"  중복 발견: workerId={worker_id}")
            print(f"    중복 개수: {len(ids)}건 (IDs: {ids})")
    
    if duplicate_count == 0:
        print("  ✅ 중복 workerId 없음")
    else:
        print(f"\n  ⚠️  총 중복 데이터: {duplicate_count}건")
    
    # 임시 레코드 확인 (unknown_으로 시작하는 workerId)
    print("\n🔍 임시 레코드 확인:")
    temp_records = list(collection.find({
        'workerId': {'$regex': '^unknown_'}
    }))
    print(f"  임시 레코드: {len(temp_records)}건")
    if temp_records:
        print("  ⚠️  unknown_으로 시작하는 임시 레코드가 있습니다.")
        for record in temp_records[:5]:
            print(f"    - workerId: {record.get('workerId')}, name: {record.get('name') or record.get('workerName')}")
    
    # 이름 없는 레코드 확인
    print("\n🔍 이름 없는 레코드 확인:")
    no_name_records = list(collection.find({
        '$or': [
            {'name': {'$exists': False}},
            {'name': ''},
            {'name': None},
            {'workerName': {'$exists': False}},
            {'workerName': ''},
            {'workerName': None}
        ]
    }))
    print(f"  이름 없는 레코드: {len(no_name_records)}건")
    if no_name_records:
        print("  ⚠️  이름이 없는 레코드가 있습니다.")
        for record in no_name_records[:5]:
            print(f"    - _id: {record.get('_id')}, workerId: {record.get('workerId')}")


def check_access_logs_collection(db):
    """access_log 컬렉션 확인"""
    print("\n" + "="*80)
    print("🚪 ACCESS_LOG 컬렉션 분석")
    print("="*80)
    
    collection = db['access_log']
    total_count = collection.count_documents({})
    print(f"\n총 문서 수: {total_count}건")
    
    if total_count == 0:
        print("⚠️  데이터가 없습니다.")
        return
    
    # 샘플 데이터 확인
    print("\n📋 샘플 데이터 (최근 3건):")
    sample_docs = list(collection.find().sort('timestamp', -1).limit(3))
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n--- 샘플 {i} ---")
        print(f"  _id: {doc.get('_id')}")
        print(f"  camera_id: {doc.get('camera_id')}")
        print(f"  person_id: {doc.get('person_id')}")
        print(f"  timestamp: {doc.get('timestamp')}")
        print(f"  status: {doc.get('status')}")
        print(f"  모든 필드: {list(doc.keys())}")


def check_faces_collection(db):
    """face 컬렉션 확인"""
    print("\n" + "="*80)
    print("👤 FACE 컬렉션 분석")
    print("="*80)
    
    collection = db['face']
    total_count = collection.count_documents({})
    print(f"\n총 문서 수: {total_count}건")
    
    if total_count == 0:
        print("⚠️  데이터가 없습니다.")
        return
    
    # 필드 사용 현황 확인
    print("\n📊 필드 사용 현황:")
    field_counts = defaultdict(int)
    for doc in collection.find():
        for key in doc.keys():
            if key != '_id':
                field_counts[key] += 1
    
    for field, count in sorted(field_counts.items()):
        percentage = (count / total_count) * 100
        print(f"  - {field}: {count}건 ({percentage:.1f}%)")
    
    # 샘플 데이터 확인
    print("\n📋 샘플 데이터 (최근 3건):")
    sample_docs = list(collection.find().sort('created_at', -1).limit(3))
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n--- 샘플 {i} ---")
        print(f"  _id: {doc.get('_id')}")
        print(f"  workerId: {doc.get('workerId')}")
        print(f"  workerName: {doc.get('workerName')}")
        print(f"  image_path: {doc.get('image_path')}")
        print(f"  embedding: {'있음' if doc.get('embedding') else '없음'} (길이: {len(doc.get('embedding', [])) if doc.get('embedding') else 0})")
        print(f"  created_at: {doc.get('created_at')}")
        print(f"  updated_at: {doc.get('updated_at')}")
        print(f"  모든 필드: {list(doc.keys())}")


def main():
    """메인 함수"""
    print("="*80)
    print("MongoDB 데이터 확인 및 분석")
    print("="*80)
    
    client, db = connect_mongodb()
    
    try:
        # 컬렉션 목록 확인
        print("\n📚 컬렉션 목록:")
        collections = db.list_collection_names()
        for col_name in collections:
            count = db[col_name].count_documents({})
            print(f"  - {col_name}: {count}건")
        
        # 각 컬렉션 분석
        check_violations_collection(db)
        check_workers_collection(db)
        check_access_logs_collection(db)
        check_faces_collection(db)
        
        print("\n" + "="*80)
        print("✅ 분석 완료")
        print("="*80)
        
    finally:
        client.close()


if __name__ == '__main__':
    main()

