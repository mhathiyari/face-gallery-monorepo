#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Face Gallery Setup Verification${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

ERRORS=0

# Check directory structure
print_info "Checking directory structure..."
for dir in backend frontend docker scripts config data; do
    if [ -d "$dir" ]; then
        print_success "Directory exists: $dir/"
    else
        print_error "Missing directory: $dir/"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check essential files
print_info "Checking essential files..."
FILES=(
    "docker-compose.yml"
    "docker-compose.cpu.yml"
    ".env.example"
    "scripts/install.sh"
    "scripts/run.sh"
    "config/config.example.json"
    "backend/requirements.txt"
    "frontend/requirements.txt"
    "frontend/app.py"
    "frontend/config_loader.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "File exists: $file"
    else
        print_error "Missing file: $file"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check if config.json exists
print_info "Checking configuration..."
if [ -f "config/config.json" ]; then
    print_success "config.json exists"
else
    print_warning "config.json not found (will use config.example.json)"
    print_info "Run: cp config/config.example.json config/config.json"
fi

if [ -f ".env" ]; then
    print_success ".env exists"
else
    print_warning ".env not found (optional for manual install)"
    print_info "Run: cp .env.example .env"
fi
echo ""

# Check data directories
print_info "Checking data directories..."
for dir in data/photos data/collections data/uploads; do
    if [ -d "$dir" ]; then
        print_success "Data directory exists: $dir/"
    else
        print_warning "Data directory missing: $dir/ (will be created on first run)"
    fi
done
echo ""

# Check Python installation (if not using Docker)
print_info "Checking Python..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python installed: $PYTHON_VERSION"

    # Check version
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
        print_success "Python version OK (3.11+)"
    else
        print_warning "Python 3.11+ recommended, found $PYTHON_VERSION"
    fi
else
    print_warning "Python not found (OK if using Docker)"
fi
echo ""

# Check Docker installation
print_info "Checking Docker..."
if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
    print_success "Docker installed: $DOCKER_VERSION"

    if docker compose version >/dev/null 2>&1; then
        print_success "Docker Compose available"
    else
        print_warning "Docker Compose not found"
    fi
else
    print_warning "Docker not found (OK if using manual install)"
fi
echo ""

# Check GPU
print_info "Checking GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_success "NVIDIA GPU detected: $GPU_NAME"

    # Check NVIDIA Container Toolkit
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 echo "GPU OK" >/dev/null 2>&1; then
        print_success "Docker GPU access OK"
    else
        print_warning "Docker GPU access not available (install NVIDIA Container Toolkit)"
    fi
else
    print_warning "No NVIDIA GPU detected (will use CPU mode)"
fi
echo ""

# Check virtual environment (for manual install)
if [ -d ".venv" ]; then
    print_info "Checking virtual environment..."
    print_success "Virtual environment exists"

    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate

        # Check key packages
        if python -c "import face_search" 2>/dev/null; then
            print_success "Backend package (face_search) installed"
        else
            print_warning "Backend package not installed"
            print_info "Run: cd backend && pip install -e ."
        fi

        if python -c "import flask" 2>/dev/null; then
            print_success "Flask installed"
        else
            print_error "Flask not installed"
            ERRORS=$((ERRORS + 1))
        fi

        deactivate
    fi
    echo ""
fi

# Summary
echo -e "${GREEN}========================================${NC}"
if [ $ERRORS -eq 0 ]; then
    print_success "All critical checks passed!"
    echo ""
    echo "Next steps:"
    echo "1. Configure: cp config/config.example.json config/config.json"
    echo "2. Docker: docker-compose up"
    echo "   OR Manual: ./scripts/run.sh"
else
    print_error "Found $ERRORS error(s). Please fix before running."
fi
echo -e "${GREEN}========================================${NC}"
echo ""
