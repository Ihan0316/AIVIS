import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))
from mongodb_storage import MongoDBStorage
import asyncio

async def check():
    storage = MongoDBStorage()
    await storage.initialize()
    
    # 작업자 목록 확인
    workers = await storage.get_all_workers()
    print('=== 등록된 작업자 (MongoDB) ===')
    for w in workers:
        name = w.get('name', 'N/A')
        worker_id = w.get('worker_id', 'N/A')
        print(f"  - {name} (ID: {worker_id})")
    
    # 최근 위반 데이터에서 worker 확인
    violations = await storage.get_violations(limit=50)
    print('\n=== 최근 위반 데이터의 작업자 ===')
    workers_in_violations = set()
    for v in violations:
        worker = v.get('worker_id') or v.get('worker_name') or v.get('name')
        if worker:
            workers_in_violations.add(worker)
    for w in workers_in_violations:
        print(f'  - {w}')
    
    print(f'\n총 {len(violations)}개 위반 데이터 확인됨')

asyncio.run(check())

