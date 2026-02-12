@echo off
chcp 65001 >nul
echo ======================================================================
echo 얼굴 임베딩 데이터베이스 전체 재구축 (통합 스크립트)
echo ======================================================================
echo.
echo 실행 순서:
echo   1. face/data 폴더 정리 (기존 데이터 삭제)
echo   2. 원본 이미지 복사 (face/image -^> face/data/images)
echo   3. 나노바나나 PPE 합성 (Gemini API 사용)
echo   4. 증강 및 임베딩 생성 (12가지 증강 + FAISS 인덱스)
echo.
echo ======================================================================
echo.

cd /d "%~dp0"

REM conda 환경 확인
set CONDA_BASE=%LOCALAPPDATA%\anaconda3
if exist "%CONDA_BASE%\python.exe" (
    echo [INFO] Conda Python 사용: %CONDA_BASE%\python.exe
    set PYTHON_CMD=%CONDA_BASE%\python.exe
) else (
    echo [INFO] 기본 Python 사용
    set PYTHON_CMD=python
)

REM OpenMP 환경 변수 설정
set KMP_DUPLICATE_LIB_OK=TRUE

REM 스크립트 실행 (PPE 합성 포함, 자동 모드)
echo [INFO] 전체 재구축 스크립트 실행 중...
echo [INFO] - face/data 폴더 정리
echo [INFO] - 원본 이미지 복사
echo [INFO] - 나노바나나 PPE 합성
echo [INFO] - 증강 및 임베딩 생성
echo.
"%PYTHON_CMD%" scripts\rebuild_face_database.py --ppe --auto

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo [OK] 전체 재구축 완료!
    echo ======================================================================
    echo.
    echo 생성된 파일:
    echo   - face/data/face_index.faiss
    echo   - face/data/face_index.faiss.labels.npy
    echo   - face/data/embeddings/face_embeddings.npy
    echo   - face/data/augmented/ (증강 이미지)
    echo   - face/data/images/ (PPE 합성 이미지 포함)
    echo.
) else (
    echo.
    echo ======================================================================
    echo [ERROR] 오류 발생 (종료 코드: %ERRORLEVEL%)
    echo ======================================================================
    echo.
    echo 문제 해결:
    echo   1. .env 파일에 GEMINI_API_KEY가 설정되어 있는지 확인
    echo   2. face/image 폴더에 원본 이미지가 있는지 확인
    echo   3. 로그를 확인하여 구체적인 오류 원인 파악
    echo.
)

pause

