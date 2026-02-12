#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
오늘 날짜의 violation 데이터 중 최신 10건만 남기고 나머지 삭제하는 스크립트
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    print("❌ pymongo가 설치되지 않았습니다.")
    print("설치: pip install pymongo")
    sys.exit(1)


def cleanup_today_violations(auto_confirm=False):
    """오늘 날짜의 violation 데이터 중 최신 10건만 남기고 나머지 삭제"""
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
        
        # 오늘 날짜 계산 (00:00:00 ~ 23:59:59)
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # timestamp 기준 (밀리초)
        today_start_ts = int(today_start.timestamp() * 1000)
        today_end_ts = int(today_end.timestamp() * 1000)
        
        # 오늘 날짜 문자열 (YYYY-MM-DD)
        today_str = today_start.strftime('%Y-%m-%d')
        
        print(f"\n📅 오늘 날짜: {today_str}")
        print(f"   시간 범위: {today_start.strftime('%Y-%m-%d %H:%M:%S')} ~ {today_end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 오늘 날짜의 데이터 조회 쿼리
        # timestamp 또는 violation_datetime이 오늘 날짜인 데이터
        query = {
            '$or': [
                {
                    'timestamp': {
                        '$gte': today_start_ts,
                        '$lte': today_end_ts
                    }
                },
                {
                    'timestamp': {'$exists': False},
                    'violation_datetime': {'$regex': f'^{today_str}'}
                }
            ]
        }
        
        # 오늘 날짜의 전체 데이터 개수 확인
        total_today = violations_collection.count_documents(query)
        print(f"\n📊 오늘 날짜의 violation 데이터: {total_today}건")
        
        if total_today <= 10:
            print("✅ 오늘 날짜의 데이터가 10건 이하이므로 삭제할 데이터가 없습니다.")
            return
        
        # 최신 10건의 _id 가져오기 (timestamp 또는 violation_datetime 기준 정렬)
        # timestamp가 있으면 timestamp 기준, 없으면 violation_datetime 기준
        keep_docs = list(
            violations_collection.find(query)
            .sort([
                ('timestamp', -1),  # timestamp 내림차순 (최신순)
                ('violation_datetime', -1)  # violation_datetime 내림차순 (최신순)
            ])
            .limit(10)
        )
        
        keep_ids = [doc['_id'] for doc in keep_docs]
        print(f"✅ 유지할 최신 10건의 _id: {len(keep_ids)}개")
        
        # 삭제할 데이터 개수
        delete_count = total_today - 10
        print(f"🗑️  삭제할 데이터: {delete_count}건")
        
        # 확인 메시지
        if not auto_confirm:
            print(f"\n⚠️  정말로 오늘 날짜의 violation 데이터 중 최신 10건을 제외한 {delete_count}건을 삭제하시겠습니까?")
            print("   (다른 날짜의 데이터는 건드리지 않습니다)")
            try:
                confirm = input("   삭제하려면 'yes'를 입력하세요: ")
                if confirm.lower() != 'yes':
                    print("❌ 삭제가 취소되었습니다.")
                    return
            except EOFError:
                print("❌ 대화형 입력을 받을 수 없습니다. --yes 플래그를 사용하세요.")
                return
        else:
            print(f"\n⚠️  오늘 날짜의 violation 데이터 중 최신 10건을 제외한 {delete_count}건을 삭제합니다.")
            print("   (다른 날짜의 데이터는 건드리지 않습니다)")
        
        # 삭제 쿼리: 오늘 날짜이면서 keep_ids에 포함되지 않은 데이터
        delete_query = {
            '$and': [
                query,  # 오늘 날짜 조건
                {'_id': {'$nin': keep_ids}}  # 유지할 _id 제외
            ]
        }
        
        # 삭제 실행
        delete_result = violations_collection.delete_many(delete_query)
        deleted_count = delete_result.deleted_count
        
        print(f"\n✅ 삭제 완료: {deleted_count}건 삭제됨")
        print(f"   남은 오늘 날짜 데이터: {total_today - deleted_count}건")
        
        # 최종 확인
        remaining_count = violations_collection.count_documents(query)
        print(f"   최종 확인: {remaining_count}건 (예상: 10건)")
        
        if remaining_count == 10:
            print("✅ 정확히 10건이 남았습니다.")
        else:
            print(f"⚠️  예상과 다릅니다. (예상: 10건, 실제: {remaining_count}건)")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'client' in locals():
            client.close()
            print("\n✅ MongoDB 연결 종료")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='오늘 날짜의 violation 데이터 중 최신 10건만 남기고 나머지 삭제')
    parser.add_argument('--yes', action='store_true', help='확인 없이 자동으로 삭제 실행')
    args = parser.parse_args()
    
    cleanup_today_violations(auto_confirm=args.yes)

