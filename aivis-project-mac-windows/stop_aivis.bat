@echo off
REM AIVIS 전체 시스템 중지 스크립트 (Windows)
REM Backend와 Frontend를 모두 중지합니다.

setlocal enabledelayedexpansion

echo ========================================
echo   AIVIS 시스템 중지
echo ========================================
echo.

REM Function to kill process on port
:kill_port
set port=%~1
set found=false
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /R /C:":%port% " ^| findstr "LISTENING"') do (
    echo [INFO] Killing process on port %port% (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
    set found=true
)
if not "!found!"=="true" (
    echo [INFO] No process found on port %port%
)
timeout /t 1 /nobreak >nul
goto :eof

REM Kill Backend (Port 8081)
echo [INFO] Stopping Backend Server (Port 8081)...
call :kill_port 8081

REM Kill Frontend (Port 5173)
echo [INFO] Stopping Frontend Server (Port 5173)...
call :kill_port 5173

REM Additional cleanup: Kill Python processes related to AIVIS
echo [INFO] Cleaning up Python processes...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr "PID:"') do (
    set pid=%%a
    set pid=!pid:PID: =!
    for /f "tokens=1" %%b in ('wmic process where "ProcessId=!pid!" get CommandLine /format:list ^| findstr "main.py"') do (
        echo [INFO] Killing Python process (PID: !pid!)
        taskkill /F /PID !pid! >nul 2>&1
    )
)

REM Additional cleanup: Kill Node processes related to Vite
echo [INFO] Cleaning up Node processes...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq node.exe" /FO LIST ^| findstr "PID:"') do (
    set pid=%%a
    set pid=!pid:PID: =!
    for /f "tokens=1" %%b in ('wmic process where "ProcessId=!pid!" get CommandLine /format:list ^| findstr "vite"') do (
        echo [INFO] Killing Node process (PID: !pid!)
        taskkill /F /PID !pid! >nul 2>&1
    )
)

timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo   시스템 중지 완료!
echo ========================================
echo.
pause

