@echo off
REM AIVIS Frontend Server Launcher for Windows
REM Frontend (Port: 5173)

setlocal enabledelayedexpansion

echo ========================================
echo   AIVIS Frontend Server
echo   - Port: 5173
echo ========================================
echo.

REM Change to project root
cd /d "%~dp0"
echo [DEBUG] Current directory: %CD%
echo [DEBUG] Script directory: %~dp0

REM Clean up port (중복 실행 방지)
echo [INFO] Cleaning up port 5173...
echo [DEBUG] Checking port 5173...
set PORT_FOUND=0
set KILLED_COUNT=0
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5173" ^| findstr "LISTENING"') do (
    set PORT_FOUND=1
    set PID=%%a
    echo [INFO] Killing process at port 5173 ^(PID: !PID!^)
    taskkill /F /PID !PID! >nul 2>&1
    if !errorlevel! equ 0 (
        set /a KILLED_COUNT+=1
        echo [DEBUG] Successfully killed process !PID!
    ) else (
        echo [WARNING] Failed to kill process !PID!
    )
)
if !PORT_FOUND!==0 (
    echo [DEBUG] No process found at port 5173
) else (
    echo [INFO] Killed !KILLED_COUNT! process^(es^) at port 5173
)
timeout /t 2 /nobreak >nul
echo [DEBUG] Port cleanup completed

REM 추가 확인: node 프로세스가 여전히 실행 중인지 확인 (중복 실행 방지)
echo [DEBUG] Checking for remaining node processes...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq node.exe" /FO LIST 2^>nul ^| findstr "PID:"') do (
    echo [WARNING] Node.js process still running ^(PID: %%a^)
    echo [INFO] Waiting for processes to terminate...
    timeout /t 1 /nobreak >nul
)

REM Check for frontend directory
echo [DEBUG] Checking for frontend directory...
if not exist "aivis-front\frontend" (
    echo [ERROR] Frontend directory not found: aivis-front\frontend
    echo [DEBUG] Current directory: %CD%
    pause
    exit /b 1
)
echo [DEBUG] Frontend directory found: %CD%\aivis-front\frontend

cd aivis-front\frontend
echo [DEBUG] Changed to frontend directory: %CD%

echo [INFO] Starting Frontend Server...
echo.
echo ========================================
echo   Vite Development Server
echo ========================================
echo.
echo [INFO] Vite will display the server URL below when ready...
echo [INFO] Expected URLs:
echo [INFO]   - Local: https://localhost:5173
echo [INFO]   - Network: https://0.0.0.0:5173
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

REM Start frontend (Vite will automatically display the URL)
REM --host 0.0.0.0 옵션으로 모든 네트워크 인터페이스에서 접근 가능
echo [DEBUG] Starting Vite dev server...
echo [DEBUG] Command: npm run dev -- --host 0.0.0.0 --port 5173
echo [DEBUG] Working directory: %CD%
echo [DEBUG] Checking npm and node...
call npm --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] npm is not available!
    echo [INFO] Please install Node.js and npm
    pause
    exit /b 1
)
call node --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] node is not available!
    echo [INFO] Please install Node.js
    pause
    exit /b 1
)
echo [DEBUG] npm and node are available
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo [WARNING] node_modules not found!
    echo [INFO] Installing dependencies...
    call npm install
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

npm run dev -- --host 0.0.0.0 --port 5173

if %errorlevel% neq 0 (
    echo [ERROR] Frontend server exited with error code: %errorlevel%
    echo [INFO] Please check the error messages above
)
pause

