#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buffalo_l 기반 얼굴 임베딩 데이터베이스 생성 스크립트
- InsightFace buffalo_l 모델 사용 (얼굴 감지 + 임베딩 통합)
- 다양한 조명/각도 증강 포함
"""

import os
import sys
import cv2
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

# ============================================================
# 설정
# ============================================================
DB_PATH = os.path.join(parent_dir, "data", "images")  # 얼굴 이미지 폴더
FAISS_INDEX_FILE = os.path.join(project_root, "src", "backend", "face_index.faiss")
FAISS_LABELS_FILE = FAISS_INDEX_FILE + ".labels.npy"
AUGMENTED_IMAGES_DIR = os.path.join(parent_dir, "data", "augmented_buffalo")

# 증강 모드: "full" (23종), "balanced" (10종), "fast" (7종)
AUGMENTATION_MODE = "full"
SAVE_AUGMENTED_IMAGES = True

# 품질 체크
ENABLE_CLAHE = True
MIN_BLUR_VARIANCE = 40.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 235.0
MIN_FACE_SIZE = 30  # 최소 얼굴 크기 (픽셀)

# 대표 임베딩 설정
USE_REPRESENTATIVE_EMBEDDING = False  # False = 모든 임베딩 저장 (인식률 향상)
TOP_N_PER_PERSON = 15


def is_low_quality(img: np.ndarray) -> bool:
    """품질 체크: 블러/밝기 기준"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = float(np.mean(gray))
        if var < MIN_BLUR_VARIANCE:
            return True
        if mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS:
            return True
        return False
    except Exception:
        return False


def apply_clahe_bgr(img: np.ndarray) -> np.ndarray:
    """Y 채널 CLAHE로 조명 표준화"""
    if img is None or img.size == 0:
        return img
    try:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    except Exception:
        return img


def create_augmentations(img: np.ndarray, mode: str = "full") -> list:
    """
    이미지 증강 생성
    Returns: [(augmented_img, label), ...]
    """
    augmented = []
    h, w = img.shape[:2]
    
    # 기본 증강 (항상 포함)
    augmented.append((img, "original"))
    augmented.append((cv2.flip(img, 1), "flip"))
    
    # 90도 회전 (넘어진 사람 인식용)
    augmented.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), "rotate_90_cw"))
    augmented.append((cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), "rotate_90_ccw"))
    
    if mode in ["full", "balanced"]:
        # 밝기 증강
        augmented.append((cv2.convertScaleAbs(img, alpha=1.0, beta=30), "bright_+30"))
        augmented.append((cv2.convertScaleAbs(img, alpha=1.0, beta=-30), "bright_-30"))
        
        # 대비 증강
        augmented.append((cv2.convertScaleAbs(img, alpha=1.2, beta=0), "contrast_+1.2"))
        augmented.append((cv2.convertScaleAbs(img, alpha=0.8, beta=0), "contrast_0.8"))
        
        # 샤프닝
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        augmented.append((cv2.filter2D(img, -1, kernel), "sharpen"))
        
        # 히스토그램 평활화
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        augmented.append((cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR), "hist_equal"))
    
    if mode == "full":
        # 추가 밝기
        augmented.append((cv2.convertScaleAbs(img, alpha=1.0, beta=45), "bright_+45"))
        augmented.append((cv2.convertScaleAbs(img, alpha=1.0, beta=-45), "bright_-45"))
        
        # 블러
        augmented.append((cv2.GaussianBlur(img, (3, 3), 0), "blur"))
        
        # 카메라 해상도 시뮬레이션
        if h > 400 or w > 400:
            for target, label in [(640, "cam_sim_640"), (480, "cam_sim_480")]:
                scale = target / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
                augmented.append((up, label))
        
        # ===== 추가 조명 증강 =====
        
        # 감마 보정 (실내/실외 조명 차이)
        for gamma_val, gamma_label in [(0.7, "gamma_0.7"), (1.5, "gamma_1.5"), (2.0, "gamma_2.0")]:
            inv_gamma = 1.0 / gamma_val
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            augmented.append((cv2.LUT(img, table), gamma_label))
        
        # 색온도 변경
        # 따뜻한 (백열등)
        warm = img.copy()
        warm[:,:,2] = np.clip(warm[:,:,2] * 1.1, 0, 255).astype(np.uint8)
        warm[:,:,0] = np.clip(warm[:,:,0] * 0.9, 0, 255).astype(np.uint8)
        augmented.append((warm, "color_warm"))
        
        # 차가운 (형광등)
        cool = img.copy()
        cool[:,:,0] = np.clip(cool[:,:,0] * 1.15, 0, 255).astype(np.uint8)
        cool[:,:,2] = np.clip(cool[:,:,2] * 0.85, 0, 255).astype(np.uint8)
        augmented.append((cool, "color_cool"))
        
        # 그림자 합성
        # 왼쪽 그림자
        shadow_left = img.copy().astype(np.float32)
        for col in range(w // 2):
            factor = 0.5 + (col / (w // 2)) * 0.5
            shadow_left[:, col] = shadow_left[:, col] * factor
        augmented.append((np.clip(shadow_left, 0, 255).astype(np.uint8), "shadow_left"))
        
        # 오른쪽 그림자
        shadow_right = img.copy().astype(np.float32)
        for col in range(w // 2, w):
            factor = 1.0 - ((col - w // 2) / (w // 2)) * 0.5
            shadow_right[:, col] = shadow_right[:, col] * factor
        augmented.append((np.clip(shadow_right, 0, 255).astype(np.uint8), "shadow_right"))
        
        # 노이즈 추가 (저조도 시뮬레이션)
        noise_img = img.copy().astype(np.float32)
        noise = np.random.normal(0, 15, noise_img.shape).astype(np.float32)
        augmented.append((np.clip(noise_img + noise, 0, 255).astype(np.uint8), "noise_low_light"))
        
        noise_heavy = img.copy().astype(np.float32)
        noise2 = np.random.normal(0, 25, noise_heavy.shape).astype(np.float32)
        augmented.append((np.clip(noise_heavy + noise2, 0, 255).astype(np.uint8), "noise_very_low"))
    
    return augmented


def build_database():
    """buffalo_l을 사용한 얼굴 임베딩 데이터베이스 생성"""
    print("=" * 60)
    print("🦬 buffalo_l 기반 얼굴 임베딩 데이터베이스 생성")
    print("=" * 60)
    print(f"이미지 폴더: {DB_PATH}")
    print(f"증강 모드: {AUGMENTATION_MODE}")
    print(f"CLAHE 적용: {ENABLE_CLAHE}")
    print("-" * 60)
    
    # buffalo_l 모델 로드
    print("\n🦬 buffalo_l 모델 로딩 중...")
    try:
        from insightface.app import FaceAnalysis
        face_analyzer = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ buffalo_l 모델 로드 완료!")
    except Exception as e:
        print(f"❌ buffalo_l 모델 로드 실패: {e}")
        return
    
    # 이미지 폴더 확인
    if not os.path.exists(DB_PATH):
        print(f"❌ 이미지 폴더가 없습니다: {DB_PATH}")
        return
    
    # 인물별 폴더 탐색
    person_folders = [f for f in os.listdir(DB_PATH) 
                      if os.path.isdir(os.path.join(DB_PATH, f))]
    
    if not person_folders:
        print(f"❌ 인물 폴더가 없습니다: {DB_PATH}")
        return
    
    print(f"\n📁 발견된 인물: {len(person_folders)}명")
    for pf in person_folders:
        print(f"   - {pf}")
    
    # 임베딩 저장용
    all_embeddings = []
    all_labels = []
    stats = {"total_images": 0, "total_faces": 0, "failed": 0}
    
    # 증강 이미지 저장 폴더
    if SAVE_AUGMENTED_IMAGES:
        os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)
    
    # 각 인물별 처리
    for person_name in person_folders:
        person_path = os.path.join(DB_PATH, person_name)
        print(f"\n👤 처리 중: {person_name}")
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(person_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"   ⚠️ 이미지 없음")
            continue
        
        print(f"   📷 이미지 수: {len(image_files)}")
        person_embeddings = []
        
        # 증강 이미지 저장 폴더
        if SAVE_AUGMENTED_IMAGES:
            person_aug_dir = os.path.join(AUGMENTED_IMAGES_DIR, person_name)
            os.makedirs(person_aug_dir, exist_ok=True)
        
        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            stats["total_images"] += 1
            
            # 이미지 로드
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"      ❌ 로드 실패: {img_file}")
                stats["failed"] += 1
                continue
            
            # 품질 체크
            if is_low_quality(img):
                print(f"      ⚠️ 저품질 스킵: {img_file}")
                continue
            
            # CLAHE 적용
            if ENABLE_CLAHE:
                img = apply_clahe_bgr(img)
            
            # 증강 생성
            augmented_images = create_augmentations(img, AUGMENTATION_MODE)
            
            # 각 증강 이미지에서 얼굴 감지 및 임베딩 추출
            for aug_img, aug_label in augmented_images:
                try:
                    # buffalo_l로 얼굴 감지 + 임베딩 추출
                    faces = face_analyzer.get(aug_img)
                    
                    if not faces:
                        continue
                    
                    # 가장 큰 얼굴 선택
                    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    
                    # 얼굴 크기 체크
                    face_w = best_face.bbox[2] - best_face.bbox[0]
                    face_h = best_face.bbox[3] - best_face.bbox[1]
                    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                        continue
                    
                    # 임베딩 추출
                    embedding = best_face.embedding
                    if embedding is None:
                        continue
                    
                    # 정규화
                    embedding = embedding / np.linalg.norm(embedding)
                    person_embeddings.append(embedding)
                    stats["total_faces"] += 1
                    
                    # 증강 이미지 저장
                    if SAVE_AUGMENTED_IMAGES:
                        base_name = os.path.splitext(img_file)[0]
                        aug_path = os.path.join(person_aug_dir, f"{base_name}_{aug_label}.jpg")
                        cv2.imencode('.jpg', aug_img)[1].tofile(aug_path)
                        
                except Exception as e:
                    continue
        
        print(f"   ✅ 추출된 임베딩: {len(person_embeddings)}개")
        
        # 대표 임베딩 또는 전체 저장
        if person_embeddings:
            if USE_REPRESENTATIVE_EMBEDDING and len(person_embeddings) > TOP_N_PER_PERSON:
                # 센트로이드 + Top-N
                centroid = np.mean(person_embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                
                # 센트로이드와 가장 유사한 N개 선택
                similarities = [np.dot(centroid, emb) for emb in person_embeddings]
                top_indices = np.argsort(similarities)[-TOP_N_PER_PERSON:]
                
                all_embeddings.append(centroid)
                all_labels.append(person_name)
                
                for idx in top_indices:
                    all_embeddings.append(person_embeddings[idx])
                    all_labels.append(person_name)
            else:
                # 전체 저장
                for emb in person_embeddings:
                    all_embeddings.append(emb)
                    all_labels.append(person_name)
    
    # FAISS 인덱스 생성
    print("\n" + "=" * 60)
    print("📊 FAISS 인덱스 생성")
    print("=" * 60)
    
    if not all_embeddings:
        print("❌ 임베딩이 없습니다!")
        return
    
    embeddings_array = np.array(all_embeddings).astype('float32')
    labels_array = np.array(all_labels)
    
    # L2 정규화 (코사인 유사도용)
    faiss.normalize_L2(embeddings_array)
    
    # Inner Product 인덱스 (정규화된 벡터에서 코사인 유사도와 동일)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    # 저장
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(FAISS_LABELS_FILE, labels_array)
    
    # 결과 출력
    print(f"\n✅ 데이터베이스 생성 완료!")
    print("-" * 60)
    print(f"총 이미지: {stats['total_images']}개")
    print(f"총 얼굴 감지: {stats['total_faces']}개")
    print(f"실패: {stats['failed']}개")
    print(f"인물 수: {len(set(all_labels))}명")
    print(f"총 임베딩: {index.ntotal}개")
    print("-" * 60)
    print(f"저장된 파일:")
    print(f"  - {FAISS_INDEX_FILE}")
    print(f"  - {FAISS_LABELS_FILE}")
    if SAVE_AUGMENTED_IMAGES:
        print(f"  - 증강 이미지: {AUGMENTED_IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    build_database()

