# 얼굴 인식 시스템 가이드

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [전체 재구축](#전체-재구축)
3. [API 키 설정](#api-키-설정)
4. [임베딩 설정](#임베딩-설정)
5. [최적화 가이드](#최적화-가이드)
6. [CCTV 환경 최적화](#cctv-환경-최적화)
7. [문제 해결](#문제-해결)

---

## 🚀 빠른 시작

### 전체 재구축 (한 번에 실행)

```powershell
cd face
.\rebuild_all.bat
```

이 스크립트는 다음을 자동으로 실행합니다:
1. **face/data 폴더 정리** - 기존 데이터 삭제
2. **원본 이미지 복사** - face/image → face/data/images
3. **나노바나나 PPE 합성** - Gemini API를 사용한 보호구 합성
4. **증강 및 임베딩 생성** - 12가지 증강 + FAISS 인덱스 생성

### 예상 소요 시간
- **PPE 합성**: 약 10-15분 (API 호출 시간에 따라 다름)
- **증강 및 임베딩**: 약 20-30분 (이미지 수에 따라 다름)
- **총 예상 시간**: 약 30-45분

---

## 📁 폴더 구조

```
face/
├── image/              # 원본 이미지 (사람별 폴더)
│   ├── donghun/
│   ├── donghyeon/
│   └── ...
├── data/               # 처리된 데이터
│   ├── images/         # 복사된 이미지 + PPE 합성본
│   ├── augmented/      # 증강된 이미지들
│   ├── embeddings/     # 원본 임베딩 (백업용)
│   ├── face_index.faiss           # FAISS 인덱스
│   └── face_index.faiss.labels.npy # 레이블 파일
├── scripts/            # 스크립트
│   ├── rebuild_face_database.py  # 전체 재구축
│   ├── ppe_synthesis_and_embedding.py  # PPE 합성
│   └── build_database.py        # 임베딩 생성
├── nanobanana/         # PPE 합성 프롬프트
│   └── prompts/
│       ├── helmet_only.txt
│       ├── vest_only.txt
│       ├── mask_only.txt
│       ├── helmet_vest.txt
│       └── ppe_ko.txt
└── .env                # Gemini API 키 설정
```

---

## 🔑 API 키 설정

### 1. Gemini API 키 발급
1. [Google AI Studio](https://aistudio.google.com/app/apikey)에 접속
2. "Create API Key" 클릭
3. API 키 복사

### 2. .env 파일 생성
`face/.env` 파일을 생성하고 다음 내용을 추가:

```
GEMINI_API_KEY=여기에_발급받은_API_키_입력
```

### 3. 확인
```powershell
cd face
type .env
```

---

## ⚙️ 임베딩 설정

### 현재 설정 (최고 성능)

#### 얼굴 탐지 해상도
- **det_size**: (832, 832) - 최고 해상도
- **효과**: 작은 얼굴도 정확히 감지, CCTV 환경 최적화

#### 데이터 증강 모드
- **AUGMENTATION_MODE**: "full" - 12가지 증강
- **증강 종류**:
  1. 원본 이미지
  2. 좌우 반전
  3. 밝기 증가 (25)
  4. 밝기 증가 (45)
  5. 밝기 감소 (-25)
  6. 밝기 감소 (-45)
  7. 대비 증가 (1.15)
  8. 대비 감소 (0.85)
  9. 가우시안 블러
  10. 샤프닝
  11. 리사이즈 시뮬레이션
  12. 히스토그램 균등화

#### 얼굴 품질 필터링
- **min_quality_score**: 0.25 - CCTV 환경 최적화
- **min_face_size**: 30 - 작은 얼굴도 포함

### 예상 인식률
- **일반 환경**: 92-97%
- **CCTV 환경**: 88-95%

### 임베딩 특징
- **차원**: 512차원 벡터
- **정규화**: L2 정규화 적용
- **검색 방식**: FAISS IndexFlatIP (완전 검색, 100% 정확도)
- **검색 속도**: 1-2ms (1,000개 기준)

---

## 🎯 최적화 가이드

### PPE 합성 최적화

#### 적용된 최적화
- ✅ **병렬 처리**: 최대 6개 이미지 동시 처리
- ✅ **이미지 크기 최적화**: 1280x1280 이하로 자동 리사이즈
- ✅ **재시도 개선**: 3회 재시도
- ✅ **전략적 배분**: 얼굴 인식 최적화
  - helmet_only: 4장 (얼굴 보임 ✅)
  - vest_only: 3장 (얼굴 보임 ✅)
  - mask_only: 2장 (얼굴 일부 가림 ⚠️)
  - helmet_vest: 1장 (얼굴 보임 ✅)

#### 성능 개선 효과
- **이전**: 53개 이미지 처리에 11분 16초
- **최적화 후**: 약 3-4분
- **개선율**: 약 65-70% 단축

### 증강 모드 선택

| 모드 | 증강 개수 | 예상 시간 | CCTV 인식률 | 추천 용도 |
|------|-----------|-----------|-------------|-----------|
| **full** ⭐ | 12가지 | 7-8분 | 85-92% | 최고 품질 (기본값) |
| **balanced** | 8가지 | 4-5분 | 83-90% | 일반 사용 |
| **fast** | 5가지 | 2-3분 | 80-87% | 빠른 처리 |

### 설정 변경 방법

#### 증강 모드 변경
`scripts/build_database.py` 파일에서:
```python
AUGMENTATION_MODE = "full"  # "full", "balanced", "fast" 중 선택
```

#### 인식 임계값 조정
`src/backend/config.py` 파일에서:
```python
class Thresholds:
    SIMILARITY: float = 0.36  # 낮추면 인식률 향상, 높이면 오인식 감소
    MIN_FACE_SIZE: int = 20
```

---

## 📹 CCTV 환경 최적화

### CCTV 환경 특성
1. **먼 거리 촬영**: 작은 얼굴도 감지 필요
2. **저해상도**: CCTV 화질 한계
3. **조명 변화**: 낮/밤, 실내/외 조명 차이
4. **다양한 각도**: 위에서 아래로 촬영
5. **마스크/안전모**: 보호구 착용 상태

### 적용된 최적화

#### 1. 얼굴 탐지 크기 증가
- **det_size**: (832, 832) - 최고 해상도
- **효과**: 더 먼 거리의 작은 얼굴도 감지 가능

#### 2. CCTV 환경 맞춤 데이터 증강
- **샤프닝**: 저해상도 이미지 선명도 향상
- **리사이즈 시뮬레이션**: CCTV 저해상도 환경 시뮬레이션
- **히스토그램 균등화**: 조명 불균일 대응

#### 3. 작은 얼굴 감지 강화
- **min_face_size**: 30
- **효과**: 먼 거리에서 찍힌 작은 얼굴도 포함

#### 4. 품질 필터링 조정
- **min_quality_score**: 0.25
- **효과**: CCTV 저해상도 이미지도 포함하여 인식률 향상

### CCTV 환경별 최적 설정

#### 실내 CCTV (조명 좋음)
```python
SIMILARITY = 0.40
MIN_FACE_SIZE = 30
FACE_DETECTION_CONFIDENCE = 0.6
```

#### 실외 CCTV (조명 변화 큼)
```python
SIMILARITY = 0.38
MIN_FACE_SIZE = 25
FACE_DETECTION_CONFIDENCE = 0.5
```

#### 저해상도 CCTV
```python
SIMILARITY = 0.35
MIN_FACE_SIZE = 20
FACE_DETECTION_CONFIDENCE = 0.4
```

---

## 🔧 성능 튜닝 가이드

### 인식률이 낮을 때
1. `SIMILARITY` 값 감소: 0.36 → 0.35
2. `min_quality_score` 감소: 0.25 → 0.20
3. 더 많은 이미지 등록 (사람당 8-10장)
4. 증강 모드를 `full`로 설정

### 오인식이 많을 때
1. `SIMILARITY` 값 증가: 0.36 → 0.42
2. `min_quality_score` 증가: 0.25 → 0.30
3. 이미지 품질 향상 (더 명확한 사진 사용)

### 처리 속도가 느릴 때
1. 증강 모드를 `balanced` 또는 `fast`로 변경
2. `FACE_DETECTION_INTERVAL` 증가: 2 → 3
3. `MAX_FACES_PER_FRAME` 감소: 10 → 5

---

## 📊 임베딩 데이터베이스 정보

### 기본 통계
- **총 임베딩 개수**: 약 1,000-1,500개 (이미지 수에 따라 다름)
- **등록된 인물 수**: 10명
- **인물당 평균 임베딩**: 약 100-150개

### 데이터 증강 효과
- **원본 이미지당 임베딩**: 평균 12개 (full 모드)
- **PPE 합성본 포함**: 원본 + 합성본 모두 증강
- **임베딩 비율**: 약 12배 (증강 포함)

### FAISS 인덱스
- **타입**: IndexFlatIP (Inner Product)
- **특징**: 완전 검색, 100% 정확도
- **검색 시간**: 1-2ms (1,000개 기준)
- **용도**: 실시간 얼굴 인식

---

## 🐛 문제 해결

### API 키 오류
```
❌ 오류: GEMINI_API_KEY 환경 변수를 찾을 수 없습니다.
```
**해결**: `face/.env` 파일에 API 키가 올바르게 설정되어 있는지 확인

### 폴더 삭제 오류
```
OSError: [WinError 145] 디렉터리가 비어 있지 않습니다
```
**해결**: 파일이 사용 중일 수 있습니다. 잠시 후 다시 시도하거나 프로세스를 종료

### OpenMP 라이브러리 충돌
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```
**해결**: 환경 변수 `KMP_DUPLICATE_LIB_OK=TRUE` 설정 (스크립트에 자동 포함)

### 임베딩 생성 실패
- 모델 파일이 `model/` 폴더에 있는지 확인
- conda 환경이 제대로 활성화되었는지 확인
- GPU 메모리가 충분한지 확인

---

## 📝 사용 예시

### 새 작업자 등록
1. `face/image/새작업자이름/` 폴더 생성
2. 얼굴 사진 3-5장 추가
3. `.\rebuild_all.bat` 실행

### 기존 작업자 사진 추가
1. `face/image/기존작업자이름/` 폴더에 사진 추가
2. `.\rebuild_all.bat` 실행 (전체 재구축)

### PPE 합성만 실행
```powershell
cd face\scripts
python ppe_synthesis_and_embedding.py
```

### 임베딩만 재생성
```powershell
cd face\scripts
python build_database.py
```

---

## ⚠️ 주의사항

1. **처리 시간**: full 모드는 balanced 모드보다 약 60% 더 오래 걸립니다.
2. **오인식 가능성**: SIMILARITY를 낮추면 인식률은 향상되지만 오인식 가능성이 약간 증가할 수 있습니다.
3. **메모리 사용**: 12가지 증강으로 더 많은 메모리를 사용할 수 있습니다.
4. **API 제한**: 병렬 처리를 너무 높이면 (8 이상) API 제한에 걸릴 수 있습니다.
5. **API 키 보안**: `.env` 파일은 절대 공개하지 마세요.

---

## ✅ 확인 사항

### 재구축 완료 후 확인
```powershell
cd face\data
dir face_index.faiss
dir face_index.faiss.labels.npy
dir embeddings\face_embeddings.npy
```

### 생성된 파일
- ✅ `face/data/face_index.faiss` - FAISS 인덱스
- ✅ `face/data/face_index.faiss.labels.npy` - 레이블 파일
- ✅ `face/data/embeddings/face_embeddings.npy` - 원본 임베딩 (백업용)
- ✅ `face/data/augmented/` - 증강된 이미지들
- ✅ `face/data/images/` - PPE 합성 이미지 포함

---

**최종 업데이트**: 2025-01-27  
**버전**: 2.0
