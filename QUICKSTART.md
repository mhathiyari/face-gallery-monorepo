# Quick Start Guide

Get Face Gallery running in 5 minutes!

## Prerequisites

Choose your installation method:

**Docker (Easiest):**
- Docker installed
- NVIDIA Container Toolkit (for GPU)

**Manual:**
- Python 3.11+
- 4GB+ RAM

## Step-by-Step

### Method 1: Docker with GPU (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery

# 2. Configure
cp .env.example .env
nano .env  # Edit PHOTOS_DIR to point to your photos

# 3. Run
docker-compose up

# 4. Open browser
# Go to http://localhost:5050
```

Done! The app is now running.

### Method 2: Docker CPU-Only

```bash
# Same as above, but use:
docker-compose -f docker-compose.cpu.yml up
```

### Method 3: Manual Install

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery

# 2. Run installer
./scripts/install.sh

# 3. Start the app
./scripts/run.sh

# 4. Open browser
# Go to http://localhost:5050
```

## First Run

Once the app is running:

### 1. Add Your Photos

**Docker users:**
- Set `PHOTOS_DIR=/path/to/photos` in `.env`
- Restart: `docker-compose restart`

**Manual install users:**
```bash
cp -r ~/Pictures/MyPhotos/* data/photos/
```

### 2. Sort Photos by Person

In the web UI:
1. Toggle **"Owner Mode"** switch (top right)
2. Set **Input Folder**: `/app/data/photos` (Docker) or `./data/photos` (manual)
3. Set **Output Folder**: `/app/data/collections/my-collection`
4. Click **"Start Sorting"**

Wait for processing to complete (progress shown in UI).

### 3. Browse Results

1. Click **"Add Collection"**
2. Enter path: `/app/data/collections/my-collection`
3. Click the collection to activate it
4. Browse the people grid!

### 4. Search by Photo

1. Click **"Search by Image"**
2. Upload a photo of a person
3. View all matching photos

## Common Issues

### "NVIDIA GPU not detected" (Docker)

Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Test:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### "Port 5050 already in use"

Change the port in `.env`:
```bash
FACE_VIEWER_PORT=8080
```

Or directly:
```bash
FACE_VIEWER_PORT=8080 docker-compose up
```

### "No module named 'face_search'"

Backend not installed. Run:
```bash
source .venv/bin/activate
cd backend
pip install -e .
```

### Slow processing

1. Check if GPU is being used:
   - Docker: `docker logs face-gallery | grep -i gpu`
   - Manual: Check startup logs for "Using device: cuda"

2. If using CPU, lower batch size in `config/config.json`:
   ```json
   {"indexing": {"batch_size": 8}}
   ```

## Next Steps

- Read the [full README](README.md)
- Configure advanced settings: [config/README.md](config/README.md)
- Enable Google Drive sharing (optional)
- Adjust clustering sensitivity for better results

## Getting Help

- Check [Troubleshooting](README.md#troubleshooting) in main README
- Open an [issue](https://github.com/yourusername/face-gallery/issues)
- Join [discussions](https://github.com/yourusername/face-gallery/discussions)

---

**Tip:** Start with a small folder of photos (~100 images) to test before processing your entire collection!
