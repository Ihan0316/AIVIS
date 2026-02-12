@echo off
REM AIVIS 전체 시스템 실행 스크립트 (Windows)
REM Backend와 Frontend를 모두 실행합니다.

setlocal enabledelayedexpansion

echo ========================================
echo   AIVIS 전체 시스템 실행
echo ========================================
echo.

REM Change to project root
cd /d "%~dp0"
echo [DEBUG] Current directory: %CD%
echo [DEBUG] Script directory: %~dp0

REM Clean up ports
echo [INFO] Cleaning up ports...
echo [DEBUG] Checking port 8081...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8081" ^| findstr "LISTENING"') do (
    echo [INFO] Killing process at port 8081 ^(PID: %%a^)
    taskkill /F /PID %%a >nul 2>&1
)
echo [DEBUG] Checking port 5173...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5173" ^| findstr "LISTENING"') do (
    echo [INFO] Killing process at port 5173 ^(PID: %%a^)
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo [OK] Port cleanup completed

REM 1. Start Backend in new window using VBScript
echo [INFO] Starting Backend Server in new window...
set SCRIPT_DIR=%~dp0
set BACKEND_BAT=!SCRIPT_DIR!start_backend.bat
set VBS_SCRIPT=%TEMP%\start_backend_%RANDOM%.vbs

echo [DEBUG] SCRIPT_DIR: !SCRIPT_DIR!
echo [DEBUG] BACKEND_BAT: !BACKEND_BAT!
echo [DEBUG] VBS_SCRIPT: !VBS_SCRIPT!

if not exist "!BACKEND_BAT!" (
    echo [ERROR] Backend script not found: !BACKEND_BAT!
    pause
    exit /b 1
)

REM Create VBScript to start backend in new window
(
    echo Set WshShell = CreateObject^("WScript.Shell"^)
    echo WshShell.Run "cmd /k ""!BACKEND_BAT!""", 1, False
    echo Set WshShell = Nothing
) > "!VBS_SCRIPT!"

if exist "!VBS_SCRIPT!" (
    echo [DEBUG] VBScript created, executing...
    cscript //nologo "!VBS_SCRIPT!"
    if !errorlevel! equ 0 (
        del "!VBS_SCRIPT!" >nul 2>&1
        timeout /t 2 /nobreak >nul
        echo [OK] Backend started in new window
    ) else (
        echo [ERROR] Failed to execute VBScript ^(error code: !errorlevel!^)
        del "!VBS_SCRIPT!" >nul 2>&1
        echo [INFO] Trying alternative method...
        start "AIVIS Backend" cmd /k "!BACKEND_BAT!"
        timeout /t 2 /nobreak >nul
        echo [OK] Backend started using alternative method
    )
) else (
    echo [ERROR] Failed to create VBScript launcher
    echo [INFO] Trying alternative method...
    start "AIVIS Backend" cmd /k "!BACKEND_BAT!"
    timeout /t 2 /nobreak >nul
    echo [OK] Backend started using alternative method
)
echo.

REM Wait for backend to start
echo [INFO] Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

REM 2. Start Frontend in new window using VBScript
echo [INFO] Starting Frontend Server in new window...

REM Check if frontend is already running (중복 실행 방지)
set FRONTEND_RUNNING=0
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5173" ^| findstr "LISTENING"') do (
    set FRONTEND_RUNNING=1
    echo [WARNING] Frontend is already running at port 5173 ^(PID: %%a^)
    echo [INFO] Skipping frontend startup to avoid duplicate instances
)

if !FRONTEND_RUNNING! equ 1 (
    echo [OK] Frontend is already running, skipping startup
) else (
    set FRONTEND_BAT=!SCRIPT_DIR!start_frontend.bat
    set VBS_SCRIPT=%TEMP%\start_frontend_%RANDOM%.vbs
    
    echo [DEBUG] FRONTEND_BAT: !FRONTEND_BAT!
    echo [DEBUG] VBS_SCRIPT: !VBS_SCRIPT!
    
    if not exist "!FRONTEND_BAT!" (
        echo [ERROR] Frontend script not found: !FRONTEND_BAT!
        pause
        exit /b 1
    )
    
    REM Create VBScript to start frontend in new window (한 번만 실행)
    (
        echo Set WshShell = CreateObject^("WScript.Shell"^)
        echo WshShell.Run "cmd /k ""!FRONTEND_BAT!""", 1, False
        echo Set WshShell = Nothing
    ) > "!VBS_SCRIPT!"
    
    if exist "!VBS_SCRIPT!" (
        echo [DEBUG] VBScript created, executing...
        cscript //nologo "!VBS_SCRIPT!"
        if !errorlevel! equ 0 (
            del "!VBS_SCRIPT!" >nul 2>&1
            timeout /t 2 /nobreak >nul
            echo [OK] Frontend started in new window
        ) else (
            echo [ERROR] Failed to execute VBScript ^(error code: !errorlevel!^)
            del "!VBS_SCRIPT!" >nul 2>&1
            echo [INFO] Trying alternative method...
            start "AIVIS Frontend" cmd /k "!FRONTEND_BAT!"
            timeout /t 2 /nobreak >nul
            echo [OK] Frontend started using alternative method
        )
    ) else (
        echo [ERROR] Failed to create VBScript launcher
        echo [INFO] Trying alternative method...
        start "AIVIS Frontend" cmd /k "!FRONTEND_BAT!"
        timeout /t 2 /nobreak >nul
        echo [OK] Frontend started using alternative method
    )
)
echo.

echo ========================================
echo   시스템 실행 완료!
echo ========================================
echo.
echo [INFO] Backend: http://localhost:8081
echo [INFO] Frontend: https://localhost:5173
echo.
echo [INFO] 각 서버는 별도 창에서 실행 중입니다.
echo [INFO] 서버를 중지하려면 stop_aivis.bat를 실행하세요.
echo.
pause

