"""
얼굴 인식 디버그 테스트
- 등록 이미지로 인식 테스트
- 임계값 분석
"""

import os
import sys
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_FILE = os.path.join(PROJECT_ROOT, "face", "data", "face_index.faiss")
FAISS_LABELS_FILE = os.path.join(PROJECT_ROOT, "face", "data", "face_index.faiss.labels.npy")

# 테스트 이미지 경로
TEST_IMAGES = {
    "유승원": os.path.join(PROJECT_ROOT, "face", "image", "유승원", "유승원정면.jpg"),
    "정준성": os.path.join(PROJECT_ROOT, "face", "image", "정준성", "정준성정면1.jpg"),
    "조이한": os.path.join(PROJECT_ROOT, "face", "image", "조이한", "조이한정면.jpg"),
}


def load_faiss_index():
    """FAISS 인덱스 로드"""
    print("\n" + "=" * 60)
    print("📊 FAISS 인덱스 로드")
    print("=" * 60)
    
    index = faiss.read_index(FAISS_INDEX_FILE)
    labels = np.load(FAISS_LABELS_FILE, allow_pickle=True)
    
    print(f"✅ 인덱스 크기: {index.ntotal}")
    print(f"✅ 임베딩 차원: {index.d}")
    print(f"✅ 레이블 수: {len(labels)}")
    
    # 인물별 통계
    unique, counts = np.unique(labels, return_counts=True)
    print("\n📋 인물별 임베딩 수:")
    for name, count in zip(unique, counts):
        print(f"   - {name}: {count}개")
    
    return index, labels


def test_recognition(face_analyzer, index, labels):
    """얼굴 인식 테스트"""
    print("\n" + "=" * 60)
    print("🔍 얼굴 인식 테스트")
    print("=" * 60)
    
    results = []
    
    for name, img_path in TEST_IMAGES.items():
        print(f"\n👤 테스트: {name}")
        print(f"   이미지: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"   ❌ 이미지 파일 없음!")
            continue
        
        # 이미지 로드
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"   ❌ 이미지 로드 실패!")
            continue
        
        print(f"   이미지 크기: {img.shape}")
        
        # 얼굴 검출 및 임베딩 추출
        faces = face_analyzer.get(img)
        
        if not faces:
            print(f"   ❌ 얼굴 미검출!")
            continue
        
        face = faces[0]
        embedding = face.embedding
        
        # 정규화
        embedding = embedding / np.linalg.norm(embedding)
        
        print(f"   ✅ 얼굴 검출됨")
        print(f"   임베딩 norm: {np.linalg.norm(embedding):.4f}")
        
        # FAISS 검색 (Inner Product = Cosine Similarity for normalized vectors)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        k = 5  # Top-5 검색
        
        distances, indices = index.search(embedding, k)
        
        print(f"\n   🔎 Top-5 검색 결과:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            matched_name = labels[idx]
            # Inner Product (IP) 거리 -> 코사인 유사도
            # IP = cos(theta) for normalized vectors
            similarity = dist  # 이미 코사인 유사도
            print(f"      {i+1}. {matched_name}: 유사도 {similarity:.4f} (거리 {dist:.4f})")
        
        # 최종 판정
        best_idx = indices[0][0]
        best_dist = distances[0][0]
        best_name = labels[best_idx]
        
        # 일반적인 임계값들
        thresholds = {
            "엄격 (0.4)": 0.4,
            "보통 (0.35)": 0.35,
            "느슨 (0.3)": 0.3,
            "매우 느슨 (0.25)": 0.25,
        }
        
        print(f"\n   📊 임계값별 판정:")
        for th_name, threshold in thresholds.items():
            if best_dist >= threshold:
                status = f"✅ {best_name} (유사도 {best_dist:.4f} >= {threshold})"
            else:
                status = f"❌ 알수없음 (유사도 {best_dist:.4f} < {threshold})"
            print(f"      {th_name}: {status}")
        
        results.append({
            "expected": name,
            "predicted": best_name,
            "similarity": best_dist,
            "correct": name == best_name
        })
    
    return results


def analyze_database_quality(index, labels):
    """데이터베이스 품질 분석"""
    print("\n" + "=" * 60)
    print("📈 데이터베이스 품질 분석")
    print("=" * 60)
    
    unique_labels = list(set(labels))
    
    # 각 인물별 임베딩 추출 및 분석
    for person in sorted(unique_labels):
        indices = [i for i, l in enumerate(labels) if l == person]
        embeddings = np.array([index.reconstruct(i) for i in indices])
        
        # 센트로이드 계산
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # 클래스 내 유사도
        similarities = []
        for emb in embeddings:
            emb = emb / np.linalg.norm(emb)
            sim = np.dot(emb, centroid)
            similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        print(f"\n👤 {person}:")
        print(f"   임베딩 수: {len(indices)}")
        print(f"   중심과의 유사도: 평균 {avg_sim:.4f}, 최소 {min_sim:.4f}, 최대 {max_sim:.4f}")
    
    # 클래스 간 분리도
    print("\n🔍 클래스 간 센트로이드 유사도 (낮을수록 분리 잘됨):")
    centroids = {}
    for person in sorted(unique_labels):
        indices = [i for i, l in enumerate(labels) if l == person]
        embeddings = np.array([index.reconstruct(i) for i in indices])
        centroid = np.mean(embeddings, axis=0)
        centroids[person] = centroid / np.linalg.norm(centroid)
    
    persons = sorted(unique_labels)
    for i, p1 in enumerate(persons):
        for p2 in persons[i+1:]:
            sim = np.dot(centroids[p1], centroids[p2])
            print(f"   {p1} ↔ {p2}: {sim:.4f}")


def main():
    print("=" * 60)
    print("🔬 얼굴 인식 디버그 테스트")
    print("=" * 60)
    
    # buffalo_l 모델 로드
    print("\n🦬 buffalo_l 모델 로딩 중...")
    face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    print("✅ 모델 로드 완료!")
    
    # FAISS 인덱스 로드
    index, labels = load_faiss_index()
    
    # 데이터베이스 품질 분석
    analyze_database_quality(index, labels)
    
    # 얼굴 인식 테스트
    results = test_recognition(face_analyzer, index, labels)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    
    print(f"\n정확도: {correct}/{total} ({100*correct/total:.1f}%)")
    
    for r in results:
        status = "✅" if r["correct"] else "❌"
        print(f"   {status} {r['expected']}: 예측={r['predicted']}, 유사도={r['similarity']:.4f}")


if __name__ == "__main__":
    main()

