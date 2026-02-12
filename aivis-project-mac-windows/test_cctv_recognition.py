"""
실시간 CCTV 환경 시뮬레이션 테스트
- 저화질, 작은 얼굴, 다양한 조명 조건
"""

import os
import sys
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_FILE = os.path.join(PROJECT_ROOT, "face", "data", "face_index.faiss")
FAISS_LABELS_FILE = os.path.join(PROJECT_ROOT, "face", "data", "face_index.faiss.labels.npy")

# 테스트 이미지
TEST_IMAGE = os.path.join(PROJECT_ROOT, "face", "image", "유승원", "유승원정면.jpg")


def simulate_cctv_conditions(img):
    """CCTV 환경 시뮬레이션"""
    results = []
    
    # 원본
    results.append(("원본 (고화질)", img.copy()))
    
    # 1. 저해상도 시뮬레이션 (640x480)
    h, w = img.shape[:2]
    small = cv2.resize(img, (640, 480))
    results.append(("640x480 리사이즈", small))
    
    # 2. 매우 작은 얼굴 (320x240)
    tiny = cv2.resize(img, (320, 240))
    results.append(("320x240 리사이즈", tiny))
    
    # 3. 어두운 환경
    dark = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)
    results.append(("어두운 환경", dark))
    
    # 4. 밝은 환경 (과노출)
    bright = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
    results.append(("밝은 환경", bright))
    
    # 5. 흐릿한 이미지 (움직임 블러)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    results.append(("블러 (7x7)", blur))
    
    # 6. 노이즈 추가
    noise = img.copy().astype(np.float32)
    noise += np.random.normal(0, 20, img.shape)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    results.append(("노이즈 추가", noise))
    
    # 7. 저해상도 + 어두움 (최악의 경우)
    worst = cv2.resize(img, (320, 240))
    worst = cv2.convertScaleAbs(worst, alpha=0.5, beta=-30)
    results.append(("최악 (작은+어두움)", worst))
    
    return results


def test_recognition(face_analyzer, index, labels, img, condition_name):
    """인식 테스트"""
    faces = face_analyzer.get(img)
    
    if not faces:
        return {
            "condition": condition_name,
            "detected": False,
            "similarity": 0.0,
            "result": "얼굴 미검출"
        }
    
    face = faces[0]
    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.reshape(1, -1).astype(np.float32)
    
    # FAISS 검색
    k = 3
    distances, indices = index.search(embedding, k)
    best_dist = distances[0][0]
    second_dist = distances[0][1] if k > 1 else 0
    best_name = labels[indices[0][0]]
    second_name = labels[indices[0][1]] if k > 1 else ""
    
    gap = best_dist - second_dist
    
    # 인식 결과 판정 (백엔드 로직과 동일)
    if best_dist >= 0.70:
        result = f"✅ {best_name} (고유사도)"
    elif best_dist >= 0.60 and gap >= 0.05:
        result = f"✅ {best_name} (중유사도)"
    elif best_dist >= 0.55 and gap >= 0.08:
        result = f"✅ {best_name} (저유사도)"
    else:
        result = f"❌ Unknown (유사도={best_dist:.3f}, 차이={gap:.3f})"
    
    return {
        "condition": condition_name,
        "detected": True,
        "similarity": best_dist,
        "gap": gap,
        "best_match": best_name,
        "second_match": second_name,
        "result": result
    }


def main():
    print("=" * 70)
    print("📹 CCTV 환경 시뮬레이션 테스트")
    print("=" * 70)
    
    # 모델 로드
    print("\n🦬 buffalo_l 모델 로딩 중...")
    face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    print("✅ 모델 로드 완료!")
    
    # FAISS 로드
    index = faiss.read_index(FAISS_INDEX_FILE)
    labels = np.load(FAISS_LABELS_FILE, allow_pickle=True)
    print(f"✅ FAISS 인덱스 로드: {index.ntotal}개 임베딩")
    
    # 테스트 이미지 로드
    img = cv2.imdecode(np.fromfile(TEST_IMAGE, dtype=np.uint8), cv2.IMREAD_COLOR)
    print(f"✅ 테스트 이미지: {TEST_IMAGE}")
    print(f"   원본 크기: {img.shape}")
    
    # CCTV 환경 시뮬레이션
    conditions = simulate_cctv_conditions(img)
    
    print("\n" + "=" * 70)
    print("📊 테스트 결과")
    print("=" * 70)
    
    for condition_name, condition_img in conditions:
        result = test_recognition(face_analyzer, index, labels, condition_img, condition_name)
        
        print(f"\n📷 {result['condition']}:")
        if result['detected']:
            print(f"   유사도: {result['similarity']:.4f}")
            print(f"   1위: {result['best_match']}, 2위: {result['second_match']}")
            print(f"   차이: {result['gap']:.4f}")
            print(f"   결과: {result['result']}")
        else:
            print(f"   결과: {result['result']}")
    
    # 임계값 권장 사항
    print("\n" + "=" * 70)
    print("💡 권장 사항")
    print("=" * 70)
    
    # 통계 분석
    detected_results = [r for name, img in conditions for r in [test_recognition(face_analyzer, index, labels, img, name)] if r['detected']]
    if detected_results:
        sims = [r['similarity'] for r in detected_results]
        avg_sim = np.mean(sims)
        min_sim = np.min(sims)
        
        print(f"\n유사도 통계: 평균={avg_sim:.4f}, 최소={min_sim:.4f}")
        
        if min_sim < 0.55:
            print(f"\n⚠️  최소 유사도({min_sim:.3f})가 현재 임계값(0.55)보다 낮습니다!")
            print(f"   권장 임계값: {min_sim - 0.05:.2f} ~ {avg_sim - 0.1:.2f}")
            print(f"\n   환경변수로 조정하세요:")
            print(f"   set SIMILARITY_THRESHOLD={max(0.30, min_sim - 0.05):.2f}")


if __name__ == "__main__":
    main()

