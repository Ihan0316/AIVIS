# 프론트엔드 (Frontend)

React + Vite 기반 프론트엔드 애플리케이션

## 구조

```
frontend/
├── src/
│   ├── components/     # React 컴포넌트 (향후 분리 예정)
│   ├── services/       # API 서비스
│   │   └── api.js      # API 호출 모듈
│   ├── utils/          # 유틸리티
│   │   ├── utils.js    # 유틸 함수
│   │   └── translations.js  # 번역
│   ├── hooks/          # 커스텀 훅 (향후 추가 예정)
│   ├── App.jsx         # 메인 앱 컴포넌트
│   ├── App.css         # 스타일
│   ├── main.jsx        # 진입점
│   └── index.css       # 글로벌 스타일
├── public/             # 정적 파일
├── package.json        # npm 의존성
└── vite.config.js      # Vite 설정
```

## 실행 방법

```bash
# 프론트엔드 디렉토리로 이동
cd frontend

# 의존성 설치 (최초 1회)
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 미리보기
npm run preview
```

## 주요 기능

- 대시보드 (통계, KPI)
- 작업자 관리 (CRUD)
- 위반 사항 관리
- 실시간 카메라 스트리밍
- 얼굴 인식 등록

## 기술 스택

- React 19
- Vite
- Chart.js (차트)
- jsPDF (PDF 생성)
- XLSX (Excel 내보내기)

