# 무제 폴더 (AIVIS 프로젝트)

이 저장소는 **AIVIS** (AI 기반 영상 감시/얼굴 인식) 관련 프로젝트들을 포함합니다.

## 구성

| 폴더 | 설명 |
|------|------|
| `aivis-project/` | 메인 AIVIS 프로젝트 (Python 백엔드 + React/Vite 프론트엔드) |
| `aivis-project-mac-windows/` | Mac/Windows용 AIVIS 변형 버전 |

## aivis-project 개요

- **백엔드**: FastAPI, 얼굴 인식(AdaFace), YOLO 기반 PPE/포즈/얼굴 검출
- **프론트엔드**: `aivis-front/frontend/` — Vite + React
- **데이터**: `data/`, `face/` — 접근 로그, 등록 인물, 얼굴 이미지
- **모델**: `model/`, `models/` — ONNX/TensorRT/ PyTorch 모델 파일

### 실행 방법 (aivis-project)

1. **백엔드**
   ```bash
   cd aivis-project
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cd src/backend && python main.py
   ```

2. **프론트엔드**
   ```bash
   cd aivis-project/aivis-front/frontend
   npm install
   npm run dev
   ```

각 하위 프로젝트의 상세 설정은 해당 폴더의 `README.md` 또는 설정 파일을 참고하세요.
