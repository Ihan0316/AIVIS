#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB 데이터 정리 스크립트
중복 데이터 제거 및 불필요한 필드 정리
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
    from bson import ObjectId
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


def remove_duplicate_violations(db, dry_run=True):
    """중복 위반 데이터 제거"""
    print("\n" + "="*80)
    print("🔍 중복 위반 데이터 제거")
    print("="*80)
    
    collection = db['violation']
    
    # 중복 그룹 찾기
    duplicates = defaultdict(list)
    all_docs = collection.find({})
    
    for doc in all_docs:
        worker_id = doc.get('worker_id', '')
        violation_type = doc.get('type') or doc.get('violation_type', '')
        timestamp = doc.get('timestamp')
        
        # timestamp를 초 단위로 정규화
        if timestamp:
            if isinstance(timestamp, (int, float)):
                timestamp_sec = int(timestamp / 1000) if timestamp > 1e12 else int(timestamp)
            else:
                timestamp_sec = str(timestamp)
        else:
            timestamp_sec = None
        
        key = (worker_id, violation_type, timestamp_sec)
        duplicates[key].append({
            '_id': doc.get('_id'),
            'timestamp': timestamp,
            'created_at': doc.get('_id').generation_time if hasattr(doc.get('_id'), 'generation_time') else None
        })
    
    # 중복 그룹 필터링 (2개 이상인 것만)
    duplicate_groups = {k: v for k, v in duplicates.items() if len(v) > 1}
    
    if not duplicate_groups:
        print("✅ 중복 데이터 없음")
        return 0
    
    print(f"\n📊 중복 그룹 수: {len(duplicate_groups)}개")
    
    total_to_remove = 0
    ids_to_remove = []
    
    for key, docs in duplicate_groups.items():
        # timestamp가 가장 큰 것(최신)을 제외하고 나머지 제거
        # 또는 _id가 가장 큰 것(최신)을 제외
        sorted_docs = sorted(docs, key=lambda x: (
            x['timestamp'] if x['timestamp'] else 0,
            str(x['_id'])
        ), reverse=True)
        
        # 첫 번째(최신)는 유지, 나머지는 제거 대상
        for doc in sorted_docs[1:]:
            ids_to_remove.append(doc['_id'])
            total_to_remove += 1
    
    print(f"🗑️  제거 대상: {total_to_remove}건")
    
    if dry_run:
        print("\n⚠️  DRY RUN 모드 - 실제로 제거하지 않습니다.")
        print("제거할 문서 ID 샘플 (최대 10개):")
        for doc_id in ids_to_remove[:10]:
            doc = collection.find_one({'_id': doc_id})
            if doc:
                print(f"  - {doc_id}: worker_id={doc.get('worker_id')}, type={doc.get('type')}, timestamp={doc.get('timestamp')}")
    else:
        if ids_to_remove:
            result = collection.delete_many({'_id': {'$in': ids_to_remove}})
            print(f"✅ {result.deleted_count}건의 중복 데이터 제거 완료")
            return result.deleted_count
    
    return total_to_remove


def cleanup_compatibility_fields(db, dry_run=True):
    """하위 호환 필드 정리"""
    print("\n" + "="*80)
    print("🧹 하위 호환 필드 정리")
    print("="*80)
    
    collection = db['violation']
    total_count = collection.count_documents({})
    
    # 하위 호환 필드 제거 (신 필드가 있는 경우)
    fields_to_remove = {
        'violation_type': 'type',
        'camera_id': 'cam_id',
        'violation_datetime': 'timestamp'  # violation_datetime은 유지 (문자열 형식 필요할 수 있음)
    }
    
    updated_count = 0
    
    for old_field, new_field in fields_to_remove.items():
        if old_field == 'violation_datetime':
            # violation_datetime은 유지 (문자열 형식이 필요할 수 있음)
            continue
            
        # 신 필드가 있고 구 필드도 있는 문서 찾기
        query = {
            new_field: {'$exists': True},
            old_field: {'$exists': True}
        }
        
        count = collection.count_documents(query)
        print(f"\n{old_field} → {new_field}: {count}건")
        
        if count > 0:
            if dry_run:
                print(f"  ⚠️  DRY RUN: {count}건의 {old_field} 필드가 제거될 예정입니다.")
            else:
                result = collection.update_many(
                    query,
                    {'$unset': {old_field: ''}}
                )
                print(f"  ✅ {result.modified_count}건의 {old_field} 필드 제거 완료")
                updated_count += result.modified_count
    
    return updated_count


def fix_worker_ids(db, dry_run=True):
    """Worker ID 정리"""
    print("\n" + "="*80)
    print("👥 Worker ID 정리")
    print("="*80)
    
    collection = db['worker']
    
    # unknown_으로 시작하는 workerId를 worker_id로 업데이트
    query = {'workerId': {'$regex': '^unknown_'}}
    temp_workers = list(collection.find(query))
    
    print(f"임시 workerId 레코드: {len(temp_workers)}건")
    
    if not temp_workers:
        print("✅ 정리할 레코드 없음")
        return 0
    
    updated_count = 0
    
    for worker in temp_workers:
        worker_id = worker.get('worker_id')
        if worker_id:
            if dry_run:
                print(f"  ⚠️  DRY RUN: workerId '{worker.get('workerId')}' → '{worker_id}'로 변경 예정")
            else:
                result = collection.update_one(
                    {'_id': worker['_id']},
                    {'$set': {'workerId': str(worker_id)}}
                )
                if result.modified_count > 0:
                    updated_count += 1
                    print(f"  ✅ workerId 업데이트: {worker.get('workerId')} → {worker_id}")
    
    return updated_count


def fix_empty_worker_names(db, dry_run=True):
    """빈 workerName 정리"""
    print("\n" + "="*80)
    print("👤 빈 Worker Name 정리")
    print("="*80)
    
    collection = db['worker']
    
    # name이 있지만 workerName이 없는 경우
    query = {
        'name': {'$exists': True, '$ne': None, '$ne': ''},
        '$or': [
            {'workerName': {'$exists': False}},
            {'workerName': None},
            {'workerName': ''}
        ]
    }
    
    workers_to_fix = list(collection.find(query))
    print(f"정리 대상: {len(workers_to_fix)}건")
    
    if not workers_to_fix:
        print("✅ 정리할 레코드 없음")
        return 0
    
    updated_count = 0
    
    for worker in workers_to_fix:
        name = worker.get('name')
        if name:
            if dry_run:
                print(f"  ⚠️  DRY RUN: workerName을 '{name}'로 설정 예정 (workerId: {worker.get('workerId')})")
            else:
                result = collection.update_one(
                    {'_id': worker['_id']},
                    {'$set': {'workerName': name}}
                )
                if result.modified_count > 0:
                    updated_count += 1
                    print(f"  ✅ workerName 업데이트: {worker.get('workerId')} → '{name}'")
    
    return updated_count


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MongoDB 데이터 정리')
    parser.add_argument('--execute', action='store_true', help='실제로 데이터를 수정합니다 (기본값: dry-run)')
    parser.add_argument('--yes', action='store_true', help='확인 없이 자동 실행')
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    print("="*80)
    print("MongoDB 데이터 정리 스크립트")
    print("="*80)
    
    if dry_run:
        print("\n⚠️  DRY RUN 모드 - 실제로 데이터를 수정하지 않습니다.")
        print("실제로 수정하려면 --execute 플래그를 사용하세요.")
    else:
        print("\n⚠️  EXECUTE 모드 - 실제로 데이터를 수정합니다!")
        if not args.yes:
            response = input("계속하시겠습니까? (yes/no): ")
            if response.lower() != 'yes':
                print("취소되었습니다.")
                return
    
    client, db = connect_mongodb()
    
    try:
        # 중복 데이터 제거
        removed_duplicates = remove_duplicate_violations(db, dry_run=dry_run)
        
        # 하위 호환 필드 정리
        cleaned_fields = cleanup_compatibility_fields(db, dry_run=dry_run)
        
        # Worker ID 정리
        fixed_workers = fix_worker_ids(db, dry_run=dry_run)
        
        # 빈 workerName 정리
        fixed_names = fix_empty_worker_names(db, dry_run=dry_run)
        
        print("\n" + "="*80)
        print("✅ 정리 완료")
        print("="*80)
        print(f"제거된 중복 데이터: {removed_duplicates}건")
        print(f"정리된 필드: {cleaned_fields}건")
        print(f"수정된 Worker ID: {fixed_workers}건")
        print(f"수정된 Worker Name: {fixed_names}건")
        
    finally:
        client.close()


if __name__ == '__main__':
    main()

