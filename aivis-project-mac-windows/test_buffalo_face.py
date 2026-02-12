# buffalo_l 얼굴 감지 테스트
import cv2
import numpy as np
import time

# InsightFace 로드
from insightface.app import FaceAnalysis

print("🦬 buffalo_l 테스트 시작...")

# 1. 모델 초기화
face_analyzer = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
face_analyzer.prepare(ctx_id=1, det_size=(640, 640))  # GPU 1 (서버와 동일)
print("✅ buffalo_l 초기화 완료")

# 2. 카메라로 테스트
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 카메라에서 프레임을 가져올 수 없습니다")
    exit(1)

print(f"📷 프레임 크기: {frame.shape}")

# 3. 얼굴 감지
start = time.time()
faces = face_analyzer.get(frame)
elapsed = time.time() - start

print(f"🔍 감지된 얼굴: {len(faces)}개 ({elapsed*1000:.1f}ms)")

for i, face in enumerate(faces):
    bbox = face.bbox.astype(int)
    print(f"  얼굴 {i+1}: 박스={bbox}, det_score={face.det_score:.3f}")
    if face.embedding is not None:
        print(f"    임베딩: shape={face.embedding.shape}, norm={np.linalg.norm(face.embedding):.3f}")
    else:
        print(f"    ❌ 임베딩 없음!")

# 4. 결과 이미지 저장
for face in faces:
    bbox = face.bbox.astype(int)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

cv2.imwrite("test_buffalo_result.jpg", frame)
print(f"✅ 결과 저장: test_buffalo_result.jpg")

