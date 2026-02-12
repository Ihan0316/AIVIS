import sys
sys.path.insert(0, 'src/backend')

import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis
import utils

# 모델 로드
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# FAISS 로드
index = faiss.read_index('face/data/face_index.faiss')
labels = np.load('face/data/face_index.faiss.labels.npy', allow_pickle=True)
face_database = (index, labels)

# 테스트 이미지
test_path = 'logs/20251205_172425_CAM0_알수없음_안전조끼.jpg'
img = cv2.imdecode(np.fromfile(test_path, dtype=np.uint8), cv2.IMREAD_COLOR)
print(f'이미지 크기: {img.shape}')

# 얼굴 검출
faces = app.get(img)
print(f'검출된 얼굴 수: {len(faces)}')

if faces:
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        det_score = face.det_score
        print(f'\n얼굴 {i+1}: bbox={bbox}, det_score={det_score:.3f}')
        
        # 임베딩 추출 및 정규화
        emb = face.embedding / np.linalg.norm(face.embedding)
        
        # FAISS 검색
        emb_query = emb.reshape(1, -1).astype(np.float32)
        dist, idx = index.search(emb_query, 3)
        
        print(f'  Top-3: {[(str(labels[idx[0][j]]), f"{dist[0][j]:.4f}") for j in range(3)]}')
        
        # utils.find_best_match_faiss 호출
        name, similarity = utils.find_best_match_faiss(emb, face_database, 0.55)
        print(f'  최종: {name}, 유사도={similarity:.4f}')
else:
    print('얼굴 미검출!')

