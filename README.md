# AIVIS (AI Video Intelligence & Safety)

AI 기반 **영상 감시**, **PPE(안전장비) 위반 감지**, **얼굴 인식**을 수행하는 통합 시스템입니다.  
이 저장소는 AIVIS 관련 **두 가지 프로젝트 변형**을 포함합니다.

---

## 목차

- [저장소 구성](#저장소-구성)
- [기술 스택](#기술-스택)
- [사전 요구사항](#사전-요구사항)
- [프로젝트별 실행 방법](#프로젝트별-실행-방법)
- [환경 변수 및 설정](#환경-변수-및-설정)
- [모델 파일 안내](#모델-파일-안내)
- [Git 저장소 이름 변경](#git-저장소-이름-변경)
- [문제 해결](#문제-해결)

---

## 저장소 구성

| 폴더 | 대상 OS | 설명 |
|------|--------|------|
| **`aivis-project/`** | Mac / Linux 중심 | 메인 AIVIS 프로젝트. FastAPI 백엔드, 적응형 워커/최적화, MongoDB 연동. |
| **`aivis-project-mac-windows/`** | Mac / Windows | 동일 기능의 변형. Windows용 배치 파일(.bat), 프론트 컴포넌트(Camera/Events 등) 포함. |

두 프로젝트 모두 **백엔드(Python) + 프론트엔드(React/Vite)** 구조이며, **AdaFace** 기반 얼굴 인식과 **YOLO** 기반 PPE/포즈/얼굴 검출을 사용합니다.

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| **백엔드** | Python 3.10+, FastAPI, aiohttp, WebSocket |
| **AI/ML** | AdaFace(얼굴 임베딩), YOLO(v8/v11) – PPE / Pose / Face, FAISS(얼굴 검색) |
| **프론트엔드** | React 19, Vite 7, Chart.js, jsPDF 등 |
| **DB** | MongoDB |
| **기타** | ONNX Runtime, TensorRT(선택), python-dotenv |

---

## 사전 요구사항

- **Python 3.10 이상** (가상환경 권장)
- **Node.js LTS** (프론트엔드 빌드/실행)
- **MongoDB** (기본 포트 27017)
- **Windows에서 InsightFace/일부 패키지 사용 시**: C++ Build Tools(Visual Studio Build Tools, “Desktop development with C++” 워크로드)

---

## 프로젝트별 실행 방법

### 1. aivis-project (Mac / Linux)

#### 백엔드

```bash
cd aivis-project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd src/backend && python main.py
```

- 백엔드 기본 포트: **8080**

#### 프론트엔드

```bash
cd aivis-project/aivis-front/frontend
npm install
npm run dev
```

- 프론트 기본 주소: **http://localhost:5173**

#### 한 번에 실행 (Mac)

```bash
cd aivis-project
./start_aivis.sh
```

종료: `./stop_aivis.sh`

---

### 2. aivis-project-mac-windows (Windows 권장)

#### 방법 A: 배치 파일로 실행 (권장)

- **백엔드만**: `start_backend.bat`
- **프론트만**: `start_frontend.bat`
- **한 번에**: `start_aivis.bat`
- **종료**: `stop_aivis.bat`

#### 방법 B: 수동 실행

**터미널 1 – 백엔드**

```batch
cd aivis-project-mac-windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd src\backend && python main.py
```

**터미널 2 – 프론트엔드**

```batch
cd aivis-project-mac-windows\aivis-front\frontend
npm install
npm run dev
```

- 접속: **http://localhost:5173**

자세한 실행 옵션·배치 파일 설명은 **`aivis-project-mac-windows/README.md`** 를 참고하세요.

---

## 환경 변수 및 설정

각 프로젝트의 **백엔드·프론트엔드** 루트에 `.env` 파일을 두고 사용할 수 있습니다. (저장소에는 포함되지 않음)

### 백엔드 예시 (`src/backend/.env` 또는 프로젝트 루트)

```env
# MongoDB
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=aivis

# 모델 경로 (기본값 사용 시 생략 가능)
# MODEL_BASE_DIR=경로
# ADAFACE_MODEL=경로

# 얼굴 인식 임계값 (선택)
# SIMILARITY_THRESHOLD=0.30
# SIMILARITY_DIFF_THRESHOLD=0.12

# YOLO 신뢰도 (선택)
# YOLO_CONFIDENCE=0.08
# POSE_CONFIDENCE=0.08
```

### 프론트엔드

- API 베이스 URL 등은 `aivis-front/frontend` 내 설정 또는 `.env`에서 조정합니다.

---

## 모델 파일 안내

용량이 큰 **모델 파일은 이 저장소에 포함되어 있지 않습니다.**  
아래 디렉터리에 직접 넣어 사용해야 합니다.

| 경로 | 용도 |
|------|------|
| `aivis-project/model/` | YOLO PPE/Pose/Face (.pt, .engine, .onnx), AdaFace (.ckpt, .onnx) |
| `aivis-project-mac-windows/model/` | 위와 동일 (해당 프로젝트용) |
| `*/models/` | InsightFace 등 추가 ONNX 모델 (필요 시) |

**필요한 모델 예시**

- PPE: `Yolo11n_PPE1.pt` / `.engine` / `.onnx`
- Pose: `yolo11n-pose.pt` / `.engine` / `.onnx`
- Face: `yolov8n-face.pt` / `.engine` / `.onnx`, AdaFace `adaface_ir50_ms1mv2.ckpt` 또는 `.onnx`

AdaFace 소스는 **AdaFace** 공식 저장소를 clone 후 `AdaFace/` 폴더에 두거나, 서브모듈로 추가해 사용할 수 있습니다.

---

## Git 저장소 이름 변경

GitHub에서 저장소 이름을 바꾼 뒤, 로컬에서 원격 주소만 새 이름으로 맞추면 됩니다.

1. **GitHub**  
   저장소 → **Settings** → **General** → **Repository name**에서 이름 변경 후 **Rename**.

2. **로컬에서 remote URL 변경**

```bash
cd "/Users/ihanjo/Downloads/무제 폴더"
git remote set-url origin https://github.com/Ihan0316/새저장소이름.git
```

예: 새 이름이 `aivis` 라면  
`git remote set-url origin https://github.com/Ihan0316/aivis.git`

---

## 문제 해결

### HTTP 500 / Push 실패

- 대용량 파일(모델, 동영상, 대량 이미지)이 커밋에 포함되면 GitHub push에서 500/타임아웃이 날 수 있습니다.
- 루트 `.gitignore`에 `model/`, `*.pt`, `*.onnx`, `face/**/*.jpg` 등이 포함되어 있으므로, **새로 추가하는 대용량 파일은 커밋하지 마세요.**

### 백엔드가 뜨지 않음

- MongoDB 실행 여부: `netstat -ano | findstr :27017` (Windows), `lsof -i :27017` (Mac/Linux)
- 포트 8080 사용 여부 확인
- 가상환경 활성화 후 `pip install -r requirements.txt` 재실행

### 프론트엔드가 뜨지 않음

- `node --version`으로 Node 설치 확인
- `aivis-front/frontend`에서 `npm install` 후 `npm run dev`

### 카메라 미연결

- USB/드라이버 확인
- 백엔드 로그에서 카메라 인덱스 확인 (config에서 카메라 설정 조정)

---

## 라이선스 및 사용

이 프로젝트는 내부/개인 사용 목적으로 제공됩니다.  
각 하위 프로젝트의 상세 문서는 해당 폴더의 **README.md**를 참고하세요.
