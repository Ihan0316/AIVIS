#!/bin/bash

# AIVIS Service Stopper for macOS
# Stops all AIVIS services (Backend and Frontend)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  AIVIS Service Stopper"
echo "========================================"
echo ""

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}[INFO]${NC} Stopping process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 1
        return 0
    else
        echo -e "${BLUE}[INFO]${NC} No process found on port $port"
        return 1
    fi
}

# Stop services
echo -e "${BLUE}[INFO]${NC} Stopping AIVIS services..."

stopped_any=false

if kill_port 8081; then
    stopped_any=true
fi

if kill_port 8080; then
    stopped_any=true
fi

if kill_port 5173; then
    stopped_any=true
fi

if [ "$stopped_any" = true ]; then
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} All services stopped!"
else
    echo ""
    echo -e "${BLUE}[INFO]${NC} No running services found"
fi

echo ""

