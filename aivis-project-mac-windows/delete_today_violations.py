from pymongo import MongoClient
from datetime import datetime

client = MongoClient('mongodb://localhost:27017')
db = client['aivis']

today = datetime.now()
today_str = today.strftime('%Y-%m-%d')

print(f'오늘 날짜: {today_str}')

# 전체 violation 개수
total = db['violation'].count_documents({})
print(f'전체 violation: {total}건')

# 날짜별 개수 확인
pipeline = [
    {'$project': {'date': {'$substr': ['$violation_datetime', 0, 10]}}},
    {'$group': {'_id': '$date', 'count': {'$sum': 1}}},
    {'$sort': {'_id': -1}},
    {'$limit': 10}
]
result = list(db['violation'].aggregate(pipeline))
print('\n최근 날짜별 개수:')
for r in result:
    print(f'  {r["_id"]}: {r["count"]}건')

# 오늘 날짜 삭제
today_count = db['violation'].count_documents({'violation_datetime': {'$regex': f'^{today_str}'}})
print(f'\n오늘({today_str}) 데이터: {today_count}건')

if today_count > 0:
    result = db['violation'].delete_many({'violation_datetime': {'$regex': f'^{today_str}'}})
    print(f'삭제 완료: {result.deleted_count}건')
else:
    print('삭제할 데이터가 없습니다.')

client.close()

