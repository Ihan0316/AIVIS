@echo off
REM AIVIS 프로젝트 Windows 환경 설정 스크립트
REM Python 가상환경 생성 및 의존성 설치

setlocal enabledelayedexpansion

echo ========================================
echo   AIVIS Windows 환경 설정
echo ========================================
echo.

REM Change to project root
cd /d "%~dp0"

REM 1. Python 확인
echo [1/5] Python 확인 중...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python이 설치되어 있지 않습니다!
    echo [INFO] Python 3.10 이상을 설치해주세요.
    pause
    exit /b 1
)
python --version
echo [OK] Python 확인 완료
echo.

REM 2. CUDA 확인
echo [2/5] CUDA 확인 중...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch가 설치되어 있지 않거나 CUDA를 확인할 수 없습니다.
    echo [INFO] setup_windows.bat 실행 후 CUDA PyTorch를 설치하세요:
    echo [INFO]   pip uninstall torch torchvision
    echo [INFO]   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
echo.

REM 3. 가상환경 확인 및 생성
echo [3/5] 가상환경 확인 중...
if exist "venv" (
    if exist "venv\bin" (
        echo [WARNING] Mac/Linux 가상환경이 감지되었습니다.
        echo [INFO] Windows용 가상환경으로 재생성합니다...
        rmdir /s /q venv 2>nul
        if exist "venv" (
            echo [ERROR] 가상환경을 자동으로 제거할 수 없습니다.
            echo [INFO] 수동으로 제거해주세요:
            echo [INFO]   1. 모든 프로세스 종료
            echo [INFO]   2. rmdir /s /q venv 실행
            pause
            exit /b 1
        )
        python -m venv venv
        echo [OK] Windows용 가상환경 생성 완료
    ) else if exist "venv\Scripts" (
        echo [INFO] 가상환경이 이미 존재합니다 (Windows 호환)
    ) else (
        echo [WARNING] 가상환경이 손상된 것으로 보입니다.
        echo [INFO] 가상환경을 재생성합니다...
        rmdir /s /q venv
        python -m venv venv
        echo [OK] 가상환경 재생성 완료
    )
) else (
    echo [INFO] 가상환경 생성 중...
    python -m venv venv
    echo [OK] 가상환경 생성 완료
)
echo.

REM 4. 가상환경 활성화 및 pip 업그레이드
echo [4/7] pip 업그레이드 중...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo [OK] pip 업그레이드 완료
echo.

REM 5. PyTorch CUDA 버전 설치 (CUDA 12.1)
echo [5/7] PyTorch CUDA 버전 설치 중...
echo [INFO] 기존 PyTorch 제거 중...
python -m pip uninstall torch torchvision -y >nul 2>&1
echo [INFO] PyTorch CUDA 12.1 버전 설치 중 (이 작업은 시간이 걸릴 수 있습니다)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch CUDA 설치에 실패했습니다. CPU 버전으로 설치를 시도합니다...
    python -m pip install torch torchvision
)
echo [OK] PyTorch 설치 완료
echo.

REM 6. ONNX Runtime GPU 버전 설치
echo [6/7] ONNX Runtime GPU 버전 설치 중...
echo [INFO] 기존 ONNX Runtime 제거 중...
python -m pip uninstall onnxruntime onnxruntime-gpu -y >nul 2>&1
echo [INFO] ONNX Runtime GPU 버전 설치 중...
python -m pip install onnxruntime-gpu
if %errorlevel% neq 0 (
    echo [WARNING] ONNX Runtime GPU 설치에 실패했습니다. CPU 버전으로 설치를 시도합니다...
    python -m pip install onnxruntime
)
echo [OK] ONNX Runtime 설치 완료
echo.

REM 7. 나머지 의존성 설치
echo [7/8] 나머지 Python 의존성 설치 중...
echo [INFO] 이 작업은 시간이 걸릴 수 있습니다...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] 의존성 설치에 실패했습니다.
    pause
    exit /b 1
)
echo [OK] Python 의존성 설치 완료
echo.

REM 8. PyTorch CUDA 재설치 (requirements.txt 설치 중 CPU 버전이 설치되었을 수 있음)
echo [8/9] PyTorch CUDA 재확인 및 재설치 중...
python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'Current PyTorch: {torch.__version__}'); print(f'CUDA Available: {cuda_available}')" 2>nul
python -c "import torch; exit(0 if torch.cuda.is_available() and 'cu' in torch.__version__ else 1)" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch CUDA 버전이 아닙니다. CUDA 버전으로 재설치합니다...
    python -m pip uninstall torch torchvision -y >nul 2>&1
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if %errorlevel% neq 0 (
        echo [WARNING] PyTorch CUDA 재설치에 실패했습니다.
    ) else (
        echo [OK] PyTorch CUDA 재설치 완료
    )
) else (
    echo [OK] PyTorch CUDA 버전이 올바르게 설치되어 있습니다.
)
echo.

REM 9. CUDA 패키지 확인
echo [8/8] CUDA 패키지 확인 중...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch CUDA 확인 실패
)
python -c "import onnxruntime; providers = onnxruntime.get_available_providers(); print(f'ONNX Runtime Providers: {providers}'); print('CUDA Available:' if 'CUDAExecutionProvider' in providers else 'CUDA Not Available')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] ONNX Runtime 확인 실패
)
echo.

REM 10. Frontend 의존성 확인
if exist "aivis-front\frontend" (
    echo [BONUS] Frontend 의존성 확인 중...
    cd aivis-front\frontend
    if not exist "node_modules" (
        echo [INFO] Frontend 의존성 설치 중...
        call npm install
        if %errorlevel% neq 0 (
            echo [WARNING] Frontend 의존성 설치에 실패했습니다.
        ) else (
            echo [OK] Frontend 의존성 설치 완료
        )
    ) else (
        echo [INFO] Frontend 의존성이 이미 설치되어 있습니다.
    )
    cd ..\..
    echo.
)

echo ========================================
echo   환경 설정 완료!
echo ========================================
echo.
echo [INFO] 다음 명령어로 프로젝트를 실행할 수 있습니다:
echo [INFO]   start_aivis.bat        - 백엔드와 프론트엔드 모두 실행
echo [INFO]   start_backend.bat     - 백엔드만 실행
echo [INFO]   start_frontend.bat    - 프론트엔드만 실행
echo.
pause

