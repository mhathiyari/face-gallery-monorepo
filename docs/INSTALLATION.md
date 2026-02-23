# Installation Guide

Detailed installation instructions for all platforms and methods.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Docker Installation](#docker-installation)
3. [Manual Installation](#manual-installation)
4. [Platform-Specific Notes](#platform-specific-notes)
5. [Verification](#verification)

---

## System Requirements

### Minimum
- **OS**: Linux, macOS, or Windows (WSL2)
- **CPU**: 4+ cores
- **RAM**: 4GB
- **Storage**: 10GB free space
- **Python**: 3.11+ (for manual install)

### Recommended
- **OS**: Linux (Ubuntu 22.04+)
- **CPU**: 8+ cores
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8 or 12.x
- **Storage**: 20GB+ free space

---

## Docker Installation

### Prerequisites

1. **Install Docker**

**Ubuntu/Debian:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in
```

**macOS:**
- Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- Install and start Docker Desktop

**Windows:**
- Install WSL2: `wsl --install`
- Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Enable WSL2 backend in Docker settings

2. **Install NVIDIA Container Toolkit (for GPU)**

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Test GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Install Face Gallery

```bash
# 1. Clone repository
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery

# 2. Configure
cp .env.example .env
nano .env  # Edit configuration

# 3. Build and run
docker-compose up
```

**First run will take time** (downloads base images, installs dependencies).

### Docker Configuration

Edit `.env`:
```bash
# Point to your photos
PHOTOS_DIR=/home/youruser/Photos

# Output directory
COLLECTIONS_DIR=./data/collections

# Port
FACE_VIEWER_PORT=5050

# GPU count
GPU_COUNT=1
```

### Docker Commands

```bash
# Start (GPU)
docker-compose up

# Start (CPU only)
docker-compose -f docker-compose.cpu.yml up

# Start in background
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up
```

---

## Manual Installation

### Ubuntu/Debian

```bash
# 1. Install Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-venv
sudo apt install -y build-essential git wget

# 2. Install system dependencies
sudo apt install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1

# 3. Install CUDA (for GPU, optional)
# Follow: https://developer.nvidia.com/cuda-downloads

# 4. Clone and install Face Gallery
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery
./scripts/install.sh

# 5. Run
./scripts/run.sh
```

### macOS

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.11
brew install python@3.11

# 3. Clone and install
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery
./scripts/install.sh

# 4. Run
./scripts/run.sh
```

**Note**: macOS users with Apple Silicon (M1/M2/M3) can use MPS acceleration:
```json
// config/config.json
{
  "model": {
    "device": "mps"
  }
}
```

### Windows (WSL2)

```bash
# 1. Install WSL2
wsl --install -d Ubuntu-22.04

# 2. Inside WSL, follow Ubuntu instructions above
cd ~
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery
./scripts/install.sh
./scripts/run.sh

# 3. Access from Windows browser
# http://localhost:5050
```

---

## Platform-Specific Notes

### Linux

**GPU Setup:**
```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install:
sudo apt install nvidia-driver-535  # Or latest version
sudo reboot
```

**CUDA Setup:**
```bash
# Check CUDA
nvcc --version

# If not found:
# Download from https://developer.nvidia.com/cuda-downloads
# Follow installation instructions
```

### macOS

**Apple Silicon (M1/M2/M3):**
- Use `device: "mps"` in config for GPU acceleration
- Install via Homebrew for best compatibility
- FAISS may have issues; use CPU mode if needed

**Intel Mac:**
- Use `device: "cpu"` (no GPU support)
- Consider using Docker for easier dependency management

### Windows

**Native Windows:**
- Not officially supported
- Use WSL2 (recommended) or Docker Desktop

**WSL2:**
- GPU support requires WSL2 + CUDA toolkit in WSL
- Follow [NVIDIA CUDA on WSL2 guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

---

## Verification

### Check Installation

```bash
# Activate environment
source .venv/bin/activate

# Run verification script
python backend/verify_setup.py
```

Expected output:
```
✓ Python 3.11.x
✓ NumPy 1.26.x
✓ PyTorch 2.x.x
✓ FAISS
✓ InsightFace
✓ Device: cuda (or cpu/mps)
✓ All systems ready!
```

### Check GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should print: `CUDA available: True` (if GPU setup correctly)

### Test Processing

```bash
# Test with sample images
python backend/examples/sort_images_by_person.py \
  /path/to/test/photos \
  /tmp/test-output
```

---

## Troubleshooting

### Python Version Issues

```bash
# Check version
python3 --version

# If wrong version, use python3.11 explicitly
python3.11 -m venv .venv
```

### CUDA/GPU Issues

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA-enabled packages
pip install --force-reinstall torch torchvision
pip install onnxruntime-gpu
```

### InsightFace Installation Fails

```bash
# Install build dependencies
sudo apt install -y build-essential python3-dev

# Install separately
pip install insightface onnxruntime
```

### FAISS Installation Fails

```bash
# Try CPU version first
pip install faiss-cpu

# For GPU (requires CUDA)
pip install faiss-gpu
```

### Memory Issues During Install

```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

### Docker Build Fails

```bash
# Clean Docker cache
docker system prune -a

# Build with more memory
docker-compose build --memory=4g

# Check disk space
df -h
```

---

## Next Steps

After successful installation:

1. **Configure**: Edit `config/config.json` (see [config/README.md](../config/README.md))
2. **Add Photos**: Copy photos to `data/photos/`
3. **Start**: Run `./scripts/run.sh` or `docker-compose up`
4. **Use**: Open http://localhost:5050

See [QUICKSTART.md](../QUICKSTART.md) for usage guide.

---

## Getting Help

- [GitHub Issues](https://github.com/yourusername/face-gallery/issues)
- [Discussions](https://github.com/yourusername/face-gallery/discussions)
- [Main README](../README.md)
