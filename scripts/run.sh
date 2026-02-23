#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found!"
    print_info "Please run: ./scripts/install.sh"
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    print_info "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate

# Check if config exists
if [ ! -f "config/config.json" ]; then
    print_error "config/config.json not found!"
    print_info "Please run: ./scripts/install.sh"
    exit 1
fi

# Get port from env or default
PORT=${FACE_VIEWER_PORT:-5050}
HOST=${FACE_VIEWER_HOST:-0.0.0.0}
DEBUG=${DEBUG:-0}

# Export environment variables for the app
export FACE_VIEWER_PORT=$PORT
export FACE_VIEWER_HOST=$HOST
export FACE_VIEWER_DEBUG=$DEBUG

# Print startup info
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Face Gallery${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Server: ${BLUE}http://${HOST}:${PORT}${NC}"
echo -e "Debug mode: ${BLUE}${DEBUG}${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Change to frontend directory and run the app
cd frontend
python app.py
