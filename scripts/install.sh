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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_info "Checking Python version..."

    if command_exists python3.11; then
        PYTHON_CMD=python3.11
    elif command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON_CMD=python3
        else
            print_error "Python 3.11+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3.11+ not found. Please install Python 3.11 or higher."
        exit 1
    fi

    print_success "Found Python: $($PYTHON_CMD --version)"
}

# Check for GPU
check_gpu() {
    print_info "Checking for GPU support..."

    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
        HAS_GPU=true
    else
        print_warning "No NVIDIA GPU detected. Will use CPU mode (slower)."
        HAS_GPU=false
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."

    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists. Skipping creation."
    else
        $PYTHON_CMD -m venv .venv
        print_success "Virtual environment created."
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source .venv/bin/activate
}

# Install dependencies
install_deps() {
    print_info "Installing backend dependencies..."
    pip install --upgrade pip setuptools wheel

    # Try minimal requirements first (more flexible versions)
    if [ -f "backend/requirements-minimal.txt" ]; then
        print_info "Using flexible dependency versions..."
        pip install -r backend/requirements-minimal.txt
    else
        pip install -r backend/requirements.txt
    fi

    if [ "$HAS_GPU" = true ]; then
        print_info "Installing GPU-accelerated packages..."
        pip install onnxruntime-gpu
        # Optionally install faiss-gpu
        read -p "Install FAISS GPU version for faster search? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install faiss-gpu
        fi
    else
        print_info "Installing CPU-only packages..."
        pip install onnxruntime
    fi

    print_info "Installing backend package..."
    cd backend
    pip install -e .
    cd ..

    print_info "Installing frontend dependencies..."
    pip install -r frontend/requirements.txt
    pip install -r frontend/requirements-face-search.txt

    print_success "All dependencies installed."
}

# Create necessary directories
create_dirs() {
    print_info "Creating necessary directories..."
    mkdir -p data/photos
    mkdir -p data/collections
    mkdir -p data/uploads
    print_success "Directories created."
}

# Create config file
create_config() {
    if [ -f "config/config.json" ]; then
        print_warning "config.json already exists. Skipping."
    else
        print_info "Creating default config.json..."
        cat > config/config.json << EOF
{
  "photos_dir": "./data/photos",
  "collections_dir": "./data/collections",
  "uploads_dir": "./data/uploads",
  "backend_root": "./backend",
  "port": 5050,
  "host": "0.0.0.0",
  "debug": false,
  "model": {
    "name": "buffalo_l",
    "device": "auto"
  },
  "clustering": {
    "eps": 0.6,
    "min_samples": 2
  }
}
EOF
        print_success "config.json created in config/ directory."
    fi
}

# Create .env file
create_env() {
    if [ -f ".env" ]; then
        print_warning ".env already exists. Skipping."
    else
        print_info "Creating .env file..."
        cat > .env << EOF
# Face Gallery Configuration

# Directories
PHOTOS_DIR=./data/photos
COLLECTIONS_DIR=./data/collections
UPLOADS_DIR=./data/uploads

# Server settings
FACE_VIEWER_PORT=5050
FACE_VIEWER_HOST=0.0.0.0
DEBUG=0

# GPU settings (for Docker)
GPU_COUNT=1
EOF
        print_success ".env file created."
    fi
}

# Verify installation
verify_install() {
    print_info "Verifying installation..."

    source .venv/bin/activate

    if $PYTHON_CMD backend/verify_setup.py; then
        print_success "Installation verified successfully!"
    else
        print_error "Installation verification failed."
        exit 1
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Installation Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Edit the configuration (optional):"
    echo -e "   ${BLUE}nano config/config.json${NC}"
    echo ""
    echo "2. Place your photos in the photos directory:"
    echo -e "   ${BLUE}cp -r /path/to/your/photos/* data/photos/${NC}"
    echo ""
    echo "3. Start the application:"
    echo -e "   ${BLUE}./scripts/run.sh${NC}"
    echo ""
    echo "4. Open your browser to:"
    echo -e "   ${BLUE}http://localhost:5050${NC}"
    echo ""
    echo "For Docker users:"
    echo -e "   ${BLUE}docker-compose up${NC}  (GPU)"
    echo -e "   ${BLUE}docker-compose -f docker-compose.cpu.yml up${NC}  (CPU)"
    echo ""
}

# Main installation flow
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Face Gallery Installation${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Check prerequisites
    check_python
    check_gpu

    # Setup
    create_venv
    activate_venv
    install_deps
    create_dirs
    create_config
    create_env

    # Verify
    verify_install

    # Done
    print_next_steps
}

# Run main
main
