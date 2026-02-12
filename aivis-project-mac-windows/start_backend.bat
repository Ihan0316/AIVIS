@echo off
REM AIVIS Backend Server Launcher for Windows (CUDA Optimized)
REM Backend (Port: 8081)

echo ========================================
echo   AIVIS Backend Server
echo   - Port: 8081
echo ========================================
echo.

REM Change to project root
cd /d "%~dp0"

REM Clean up port
echo [INFO] Cleaning up port 8081...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /R /C:":8081 " ^| findstr "LISTENING" 2^>nul') do (
    echo [INFO] Killing process on port 8081 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM Clean up log files
if exist "logs\*.log" del /q "logs\*.log" >nul 2>&1
if exist "logs\*.csv" del /q "logs\*.csv" >nul 2>&1
echo [OK] Port and log cleanup completed
echo.

REM Optimization environment variables
set MODEL_INPUT_WIDTH=512
set MODEL_INPUT_HEIGHT=384
set POSE_INTERVAL=3
set DETECTION_INTERVAL=3
set FACE_DETECTION_INTERVAL=5
set MAX_WORKERS=16
set BATCH_SIZE=1
set ENABLE_HALF_PRECISION=1
echo [OPTIMIZATION] GPU optimized settings applied

REM NVIDIA CUDA Libraries for ONNX Runtime GPU (buffalo_l)
set NVIDIA_LIBS=%USERPROFILE%\anaconda3\envs\aivis-gpu\Lib\site-packages\nvidia
set PATH=%NVIDIA_LIBS%\cublas\bin;%NVIDIA_LIBS%\cudnn\bin;%NVIDIA_LIBS%\cufft\bin;%NVIDIA_LIBS%\cusparse\bin;%NVIDIA_LIBS%\cusolver\bin;%NVIDIA_LIBS%\curand\bin;%NVIDIA_LIBS%\cuda_runtime\bin;%NVIDIA_LIBS%\cuda_nvrtc\bin;%NVIDIA_LIBS%\nvjitlink\bin;%PATH%
echo [CUDA] NVIDIA libraries added to PATH

REM Activate conda environment using conda.bat (most reliable method)
echo [CONDA] Activating aivis-gpu environment...
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate aivis-gpu
if %errorlevel% neq 0 (
    echo [ERROR] Conda activation failed!
    echo [INFO] Trying alternative method...
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" aivis-gpu
)
echo [OK] Conda environment activated
echo.

REM Verify onnxruntime is available
echo [CHECK] Verifying onnxruntime...
python -c "import onnxruntime; print('[OK] onnxruntime:', onnxruntime.__version__)"
if %errorlevel% neq 0 (
    echo [ERROR] onnxruntime not available!
    pause
    exit /b 1
)
echo.

cd src\backend
if not exist "main.py" (
    echo [ERROR] main.py not found!
    pause
    exit /b 1
)

echo [DEBUG] Current directory: %CD%
echo [INFO] Starting Backend Server...
echo.

REM Execute Python script
python main.py
if %errorlevel% neq 0 (
    echo [ERROR] Backend server exited with error code: %errorlevel%
)
pause
