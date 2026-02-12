#!/bin/bash

# AIVIS Integrated Launcher for macOS (MPS Optimized)
# Backend (Port: 8081)
# Frontend (Port: 5173)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  AIVIS Integrated Launcher (MPS Optimized)"
echo "  - Backend (Port: 8081)"
echo "  - Frontend (Port: 5173)"
echo "========================================"
echo ""

# Change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}[INFO]${NC} Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

# 0. Clean up ports
echo -e "${BLUE}[INFO]${NC} Cleaning up ports..."
kill_port 8081
kill_port 8080
kill_port 5173
sleep 1

# 1. Find Python environment
PYTHON_EXE=""
VENV_ACTIVATED=false

# Check for conda
if command -v conda &> /dev/null; then
    echo -e "${GREEN}[ENV]${NC} Found Conda"
    
    # Initialize conda if needed
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        eval "$(conda shell.bash hook)"
    fi
    
    # Try to use conda base environment
    PYTHON_EXE="$(conda info --base)/bin/python"
    
    if [ -f "$PYTHON_EXE" ]; then
        echo -e "${GREEN}[ENV]${NC} Using Conda Python: $PYTHON_EXE"
        
        # Verify MPS availability
        echo -e "${BLUE}[INFO]${NC} Verifying MPS support..."
        "$PYTHON_EXE" -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')" 2>/dev/null || echo -e "${YELLOW}[WARNING]${NC} Could not verify MPS support"
    else
        echo -e "${YELLOW}[WARNING]${NC} Conda Python not found, trying venv..."
    fi
fi

# Check for venv (Mac-compatible only)
if [ -z "$PYTHON_EXE" ] && [ -d "venv" ]; then
    if [ -f "venv/bin/python" ]; then
        echo -e "${GREEN}[ENV]${NC} Found local venv (Mac-compatible)"
        source venv/bin/activate
        PYTHON_EXE="python"
        VENV_ACTIVATED=true
        
        # Verify MPS availability
        echo -e "${BLUE}[INFO]${NC} Verifying MPS support..."
        python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')" 2>/dev/null || echo -e "${YELLOW}[WARNING]${NC} Could not verify MPS support"
    elif [ -d "venv/Scripts" ] && [ ! -d "venv/bin" ]; then
        echo -e "${RED}[ERROR]${NC} Windows virtual environment detected!"
        echo -e "${YELLOW}[INFO]${NC} Please recreate venv for Mac:"
        echo -e "${YELLOW}[INFO]${NC}   rm -rf venv && python3 -m venv venv"
        echo -e "${YELLOW}[INFO]${NC}   ./setup_mac.sh"
        exit 1
    fi
fi

# Check for system Python (fallback)
if [ -z "$PYTHON_EXE" ] && command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
    echo -e "${YELLOW}[WARNING]${NC} Using system Python (venv/conda not found)"
    echo -e "${YELLOW}[WARNING]${NC} MPS might not work properly without proper environment"
fi

# Final check
if [ -z "$PYTHON_EXE" ]; then
    echo -e "${RED}[ERROR]${NC} No Python environment found!"
    echo -e "${YELLOW}[INFO]${NC} Please create a venv: python3 -m venv venv"
    exit 1
fi

# Start services
# 2. Start Backend
echo -e "${BLUE}[INFO]${NC} Starting Backend Server..."

# 최적화 환경변수 설정 (MPS/CPU 최대 활용)
export MODEL_INPUT_WIDTH=512
export MODEL_INPUT_HEIGHT=384
export POSE_INTERVAL=2  # 3 -> 2 (넘어짐 감지 정확도 향상)
export DETECTION_INTERVAL=1
export MAX_WORKERS=12
echo -e "${GREEN}[OPTIMIZATION]${NC} 최적화 설정:"
echo -e "${GREEN}[OPTIMIZATION]${NC}   - 모델 입력 해상도: ${MODEL_INPUT_WIDTH}x${MODEL_INPUT_HEIGHT}"
echo -e "${GREEN}[OPTIMIZATION]${NC}   - Pose 실행 간격: ${POSE_INTERVAL}프레임마다 (넘어짐 감지 최적화)"
echo -e "${GREEN}[OPTIMIZATION]${NC}   - 최대 워커 수: ${MAX_WORKERS}"

cd src/backend

# Try to start in new terminal window, fallback to background
if osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR/src/backend' && $PYTHON_EXE main.py\"" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} Backend started in new terminal window"
    BACKEND_PID=""
else
    # Fallback: run in background
    echo -e "${YELLOW}[INFO]${NC} Running backend in background (logs: backend.log)..."
    nohup $PYTHON_EXE main.py > ../backend.log 2>&1 &
    BACKEND_PID=$!
    echo -e "${GREEN}[OK]${NC} Backend started (PID: $BACKEND_PID)"
fi

cd "$SCRIPT_DIR"

# Wait a bit for backend to start
sleep 3

# 3. Start Frontend
echo -e "${BLUE}[INFO]${NC} Starting Frontend Server..."
if [ -d "aivis-front/frontend" ]; then
    cd aivis-front/frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}[INFO]${NC} Installing frontend dependencies..."
        npm install
    fi
    
    # Try to start in new terminal window, fallback to background
    if osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR/aivis-front/frontend' && npm run dev -- --port 5173\"" > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} Frontend started in new terminal window"
        FRONTEND_PID=""
    else
        # Fallback: run in background
        echo -e "${YELLOW}[INFO]${NC} Running frontend in background (logs: frontend.log)..."
        nohup npm run dev -- --port 5173 > ../frontend.log 2>&1 &
        FRONTEND_PID=$!
        echo -e "${GREEN}[OK]${NC} Frontend started (PID: $FRONTEND_PID)"
    fi
    
    cd "$SCRIPT_DIR"
else
    echo -e "${RED}[ERROR]${NC} Frontend directory not found: aivis-front/frontend"
    FRONTEND_PID=""
fi

echo ""
echo "========================================"
echo "  All services started!"
echo "  - Backend: http://localhost:8081"
echo "  - Frontend: http://localhost:5173"
echo "========================================"
echo ""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}[INFO]${NC} Stopping services..."
    kill_port 8081
    kill_port 5173
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}[OK]${NC} All services stopped"
    exit 0
}

# Wait for user interrupt
trap cleanup INT TERM

# Keep script running if services are in background
if [ ! -z "$BACKEND_PID" ] || [ ! -z "$FRONTEND_PID" ]; then
    echo -e "${GREEN}Services are running. Press Ctrl+C to stop all services${NC}"
    # Wait for background processes
    wait
else
    echo -e "${GREEN}Services are running in separate terminal windows.${NC}"
    echo -e "${YELLOW}To stop services, close the terminal windows or run: ./stop_aivis.sh${NC}"
    # Keep script alive
    while true; do
        sleep 1
    done
fi

