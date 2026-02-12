# AIVIS - AI Safety Monitoring

## 🚀 Quick Start (최종 정리 완료!)

### 방법 1: 개별 실행 (권장)

**PowerShell 또는 명령 프롬프트 2개 열기**

#### 창 1: 백엔드
```batch
START_BACKEND.bat
```

#### 창 2: 프론트엔드
```batch
START_FRONTEND.bat
```

### 방법 2: 한 번에 실행

```batch
START_ALL.bat
```

### 브라우저 접속
```
http://localhost:5173
```

### 종료
```batch
STOP_ALL.bat
```

---

## Files

- `START_BACKEND.bat` - 백엔드 서버 시작 (포트 8080)
- `START_FRONTEND.bat` - 프론트엔드 시작 (포트 5173)
- `START_ALL.bat` - 백엔드 + 프론트엔드 한 번에 시작
- `STOP_ALL.bat` - 모든 서버 종료
- `PROJECT_ARCHITECTURE.md` - 프로젝트 구조 문서
- `프론트백엔드_연결_가이드.md` - 연결 가이드
- `백엔드_카메라_처리_이유.md` - 아키텍처 설명

---

## Features

- Real-time AI detection (Face, PPE, Pose)
- 30 FPS video streaming
- Real-time dashboard (0.5s updates)
- Camera switching (97% stable)

---

**Just run 2 bat files! 🚀**

1. **Python 3.10 이상**
   - [Python 다운로드](https://www.python.org/downloads/)
   - 설치 시 "Add Python to PATH" 체크

2. **Node.js (LTS 버전)**
   - [Node.js 다운로드](https://nodejs.org/)
   - 설치 시 "Add to PATH" 체크
   - 설치 후 터미널 재시작

3. **MongoDB**
   - [MongoDB 다운로드](https://www.mongodb.com/try/download/community)
   - 기본 포트: 27017
   - 설치 가이드: `.\install_mongodb.bat` 실행
   - 서비스 시작: `.\start_mongodb.bat` 실행

4. **C++ Build Tools** (InsightFace 사용 시)
   - [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - "Desktop development with C++" 워크로드 선택
   - 설치 후 재부팅

## 📁 프로젝트 구조

```
aivis-project/
├── src/backend/              # 기존 백엔드 (aiohttp, 포트 8080)
│   ├── main.py              # 메인 서버
│   ├── config.py            # 설정
│   ├── core.py              # AI 모델 처리
│   └── database.py          # MongoDB 연결
├── aivis-front-Final/        # 새 프론트엔드
│   └── aivis-front/
│       ├── frontend/         # React + Vite
│       └── backend/          # Flask 백엔드 (선택사항)
├── model/                    # YOLO 모델 파일
├── models/                   # InsightFace 모델
├── face/data/               # FAISS 인덱스 파일
├── data/                     # 데이터 파일 (JSON)
└── images/                   # 이미지 저장 폴더
```

## 🛠️ 배치 파일

- `start-backend.bat` - 백엔드 서버 시작
- `start-frontend-final.bat` - 새 프론트엔드 시작
- `kill-all.bat` - 모든 프로세스 종료
- `check_cameras.bat` - 카메라 진단 도구
- `얼굴인식_개선_실행.bat` - 얼굴 인식 임베딩 재생성

자세한 내용은 `배치_파일_사용_가이드.md` 참고

## 🔧 환경 변수

`.env` 파일 생성 (선택사항):
```env
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=aivis
GOOGLE_API_KEY=your_api_key_here  # 얼굴 인식 개선 시 필요
```

## 📚 주요 기능

- **PPE 감지**: 안전모, 안전조끼, 마스크 감지
- **위험 행동 감지**: 낙상, 위험 자세 감지
- **얼굴 인식**: FAISS 기반 실시간 얼굴 인식
- **실시간 스트리밍**: WebSocket 기반 실시간 비디오 스트리밍
- **통계 및 리포트**: 위반 사항 통계 및 리포트 생성

## 🐛 문제 해결

### 백엔드가 시작되지 않음
- MongoDB가 실행 중인지 확인: `netstat -ano | findstr :27017`
- 가상환경이 활성화되었는지 확인
- 포트 8080이 사용 중인지 확인

### 프론트엔드가 시작되지 않음
- Node.js가 설치되어 있는지 확인: `node --version`
- `node_modules` 폴더 확인: 없으면 `npm install` 실행
- 포트 5173이 사용 중인지 확인

### 카메라가 감지되지 않음
- `.\check_cameras.bat` 실행하여 카메라 확인
- 브라우저에서 카메라 권한 확인
- USB 연결 확인

## 📝 참고 문서

- `START_HERE.md` - 시작 가이드
- `SYSTEM_ARCHITECTURE.md` - 시스템 아키텍처
- `TECH_STACK.md` - 기술 스택
- `docs/CHANGELOG.md` - 변경 이력
- `docs/OPTIMIZATION.md` - 최적화 가이드

## 📄 라이선스

이 프로젝트는 내부 사용을 위한 것입니다.