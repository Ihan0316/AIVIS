#!/bin/bash

# AIVIS macOS Setup Script
# Sets up Python environment and installs dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  AIVIS macOS Setup Script"
echo "========================================"
echo ""

# Change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo -e "${BLUE}[INFO]${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3 is not installed!"
    echo -e "${YELLOW}[INFO]${NC} Please install Python3: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}[OK]${NC} $PYTHON_VERSION"

# Check if venv exists and is Mac-compatible
if [ -d "venv" ]; then
    if [ -d "venv/Scripts" ] && [ ! -d "venv/bin" ]; then
        # Windows venv detected, need to recreate for Mac
        echo -e "${YELLOW}[WARNING]${NC} Windows virtual environment detected"
        echo -e "${BLUE}[INFO]${NC} Removing Windows venv and creating Mac-compatible venv..."
        
        # Try multiple methods to remove venv
        # Method 1: Remove write protection
        chmod -R u+w venv 2>/dev/null || true
        
        # Method 2: Try rm -rf first
        if rm -rf venv 2>/dev/null; then
            echo -e "${GREEN}[OK]${NC} Removed Windows venv"
        else
            # Method 3: Try find -delete
            if find venv -delete 2>/dev/null; then
                echo -e "${GREEN}[OK]${NC} Removed Windows venv (using find)"
            else
                # Method 4: Try with sudo
                echo -e "${YELLOW}[WARNING]${NC} Standard removal failed, trying with elevated permissions..."
                if sudo rm -rf venv 2>/dev/null; then
                    echo -e "${GREEN}[OK]${NC} Removed with elevated permissions"
                else
                    echo -e "${RED}[ERROR]${NC} Could not remove venv automatically"
                    echo -e "${YELLOW}[INFO]${NC} Please manually remove it:"
                    echo -e "${YELLOW}[INFO]${NC}   1. Close any processes using venv"
                    echo -e "${YELLOW}[INFO]${NC}   2. Run: rm -rf venv"
                    echo -e "${YELLOW}[INFO]${NC}   3. Or: sudo rm -rf venv"
                    exit 1
                fi
            fi
        fi
        
        python3 -m venv venv
        echo -e "${GREEN}[OK]${NC} Mac-compatible virtual environment created"
    elif [ -d "venv/bin" ]; then
        echo -e "${BLUE}[INFO]${NC} Virtual environment already exists (Mac-compatible)"
    else
        # Corrupted venv, recreate
        echo -e "${YELLOW}[WARNING]${NC} Virtual environment appears corrupted"
        echo -e "${BLUE}[INFO]${NC} Recreating virtual environment..."
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}[OK]${NC} Virtual environment recreated"
    fi
else
    echo -e "${BLUE}[INFO]${NC} Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}[OK]${NC} Virtual environment created"
fi

# Activate venv
echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    # Fallback for Windows venv (shouldn't happen after above check)
    source venv/Scripts/activate
else
    echo -e "${RED}[ERROR]${NC} Cannot find venv activation script!"
    echo -e "${YELLOW}[INFO]${NC} Please recreate venv: rm -rf venv && python3 -m venv venv"
    exit 1
fi

# Upgrade pip
echo -e "${BLUE}[INFO]${NC} Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo -e "${BLUE}[INFO]${NC} Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}[OK]${NC} Python dependencies installed"
else
    echo -e "${YELLOW}[WARNING]${NC} requirements.txt not found"
fi

# Check for frontend
if [ -d "aivis-front/frontend" ]; then
    echo -e "${BLUE}[INFO]${NC} Installing frontend dependencies..."
    cd aivis-front/frontend
    
    if command -v npm &> /dev/null; then
        # Check for rollup native module issue (npm optional dependencies bug)
        if [ -d "node_modules" ] && [ ! -d "node_modules/@rollup/rollup-darwin-arm64" ] && [ -d "node_modules/rollup" ]; then
            echo -e "${YELLOW}[WARNING]${NC} Detected rollup native module issue (npm optional dependencies bug)"
            echo -e "${BLUE}[INFO]${NC} Cleaning up and reinstalling dependencies..."
            rm -rf node_modules package-lock.json
        fi
        
        if [ ! -d "node_modules" ]; then
            npm install
            echo -e "${GREEN}[OK]${NC} Frontend dependencies installed"
        else
            echo -e "${BLUE}[INFO]${NC} Frontend dependencies already installed"
            # Verify rollup is properly installed
            if [ ! -d "node_modules/@rollup/rollup-darwin-arm64" ] && [ -d "node_modules/rollup" ]; then
                echo -e "${YELLOW}[WARNING]${NC} Rollup native module missing, reinstalling..."
                rm -rf node_modules package-lock.json
                npm install
            fi
        fi
        
        # Fix executable permissions for node_modules/.bin scripts
        if [ -d "node_modules/.bin" ]; then
            chmod +x node_modules/.bin/* 2>/dev/null && echo -e "${GREEN}[OK]${NC} Fixed executable permissions for npm scripts" || true
        fi
    else
        echo -e "${YELLOW}[WARNING]${NC} npm is not installed. Frontend dependencies not installed."
        echo -e "${YELLOW}[INFO]${NC} Install Node.js: brew install node"
    fi
    
    cd "$SCRIPT_DIR"
else
    echo -e "${YELLOW}[WARNING]${NC} Frontend directory not found"
fi

# Verify PyTorch and MPS
echo ""
echo -e "${BLUE}[INFO]${NC} Verifying PyTorch and MPS support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if hasattr(torch.backends, 'mps'):
    mps_available = torch.backends.mps.is_available()
    print(f'MPS Available: {mps_available}')
    if mps_available:
        print('✅ MPS (Metal GPU) support is enabled!')
    else:
        print('⚠️  MPS is not available (may need macOS 12.3+ and Apple Silicon)')
else:
    print('⚠️  MPS backend not found in PyTorch')
" || echo -e "${YELLOW}[WARNING]${NC} Could not verify PyTorch installation"

echo ""
echo "========================================"
echo -e "${GREEN}Setup completed!${NC}"
echo ""
echo "To start AIVIS, run:"
echo "  ./start_aivis.sh"
echo ""
echo "To stop AIVIS, run:"
echo "  ./stop_aivis.sh"
echo "========================================"

