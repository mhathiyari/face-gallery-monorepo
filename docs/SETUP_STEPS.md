# Setup Steps - Complete Guide

This document provides step-by-step instructions for setting up and using Face Gallery.

## Overview

Face Gallery is now structured as a monorepo containing:
- **Backend**: Face recognition engine (face-search-system)
- **Frontend**: Web UI (face_search_viewer)
- **Docker**: Containerized deployment
- **Scripts**: Automated installation and running

## Quick Decision Tree

**Do you have Docker installed?**
- ✅ Yes → Use Docker method (easiest)
- ❌ No → Use Manual Install method

**Do you have an NVIDIA GPU?**
- ✅ Yes → Use GPU configuration (faster)
- ❌ No → Use CPU configuration (slower but works)

---

## Method 1: Docker + GPU (Fastest & Easiest)

### Prerequisites
- Docker installed
- NVIDIA GPU
- NVIDIA Container Toolkit installed

### Steps

1. **Get the code:**
   ```bash
   git clone https://github.com/yourusername/face-gallery.git
   cd face-gallery
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   nano .env
   ```

   Edit these lines:
   ```bash
   PHOTOS_DIR=/home/youruser/Photos  # Your actual photo folder
   COLLECTIONS_DIR=./data/collections
   FACE_VIEWER_PORT=5050
   ```

3. **Start the application:**
   ```bash
   docker-compose up
   ```

   First run will take 10-15 minutes (downloads images, installs packages).

4. **Access the UI:**
   - Open browser: http://localhost:5050
   - You should see the Face Gallery interface

5. **Process your photos:**
   - Toggle "Owner Mode" (top right)
   - Input folder: `/app/data/photos`
   - Output folder: `/app/data/collections/my-photos`
   - Click "Start Sorting"

6. **View results:**
   - Click "Add Collection"
   - Path: `/app/data/collections/my-photos`
   - Click to activate
   - Browse your organized photos!

---

## Method 2: Docker CPU-Only

Same as Method 1, but use:
```bash
docker-compose -f docker-compose.cpu.yml up
```

**Note**: Processing will be 5-10x slower without GPU.

---

## Method 3: Manual Install (Advanced)

### Prerequisites
- Python 3.11 or higher
- Git
- 4GB+ RAM
- NVIDIA GPU (optional, for speed)

### Steps

1. **Get the code:**
   ```bash
   git clone https://github.com/yourusername/face-gallery.git
   cd face-gallery
   ```

2. **Run installer:**
   ```bash
   ./scripts/install.sh
   ```

   This will:
   - Check Python version
   - Detect GPU
   - Create virtual environment
   - Install all dependencies
   - Create config files
   - Verify installation

   Takes 5-10 minutes.

3. **Add your photos:**
   ```bash
   cp -r ~/Pictures/MyPhotos/* data/photos/
   ```

4. **Start the application:**
   ```bash
   ./scripts/run.sh
   ```

5. **Access the UI:**
   - Open browser: http://localhost:5050

6. **Process photos:**
   - Toggle "Owner Mode"
   - Input folder: `./data/photos`
   - Output folder: `./data/collections/my-photos`
   - Click "Start Sorting"

7. **View results:**
   - Click "Add Collection"
   - Path: `/full/path/to/face-gallery/data/collections/my-photos`
   - Click to activate
   - Browse!

---

## Configuration Options

### Basic Configuration

Edit `config/config.json`:

```json
{
  "paths": {
    "photos_dir": "./data/photos",
    "collections_dir": "./data/collections"
  },
  "model": {
    "device": "auto",
    "name": "buffalo_l"
  },
  "clustering": {
    "eps": 0.6,
    "min_samples": 2
  }
}
```

### Important Settings

**Device (model.device)**:
- `"auto"`: Automatically use GPU if available
- `"cuda"`: Force NVIDIA GPU
- `"cpu"`: Force CPU (slow but always works)
- `"mps"`: Apple Silicon GPU (M1/M2/M3)

**Clustering Epsilon (clustering.eps)**:
- `0.4`: Very strict (more, smaller clusters)
- `0.6`: Balanced (recommended default)
- `0.8`: Loose (fewer, larger clusters)

**Model Size (model.name)**:
- `"buffalo_l"`: Large (best accuracy, slower)
- `"buffalo_m"`: Medium (balanced)
- `"buffalo_s"`: Small (fastest, less accurate)

---

## Common Workflows

### Workflow 1: First Time Setup

1. Install (Docker or manual)
2. Add a small test folder (~50 photos) to `data/photos/test`
3. Run sorting on test folder
4. Verify results look good
5. Adjust `eps` in config if needed
6. Process full photo collection

### Workflow 2: Adding New Photos

1. Copy new photos to `data/photos/new-batch/`
2. Run sorting to new output folder
3. Add collection in UI
4. Browse and label people

### Workflow 3: Re-clustering Existing Collection

1. Export faces from existing collection
2. Adjust `eps` in config
3. Re-run clustering with new settings
4. Compare results

### Workflow 4: Search for Specific Person

1. Have an existing processed collection active
2. Click "Search by Image"
3. Upload clear photo of person's face
4. View all matching photos
5. Add label if needed

---

## Performance Tuning

### For Speed

```json
{
  "model": {
    "device": "cuda",
    "name": "buffalo_l"
  },
  "indexing": {
    "batch_size": 64
  }
}
```

### For Memory Efficiency

```json
{
  "model": {
    "device": "cuda",
    "name": "buffalo_s"
  },
  "indexing": {
    "batch_size": 16
  }
}
```

### For CPU-Only Systems

```json
{
  "model": {
    "device": "cpu",
    "name": "buffalo_m"
  },
  "indexing": {
    "batch_size": 8
  }
}
```

---

## Troubleshooting by Symptom

### App won't start

**Check if port is in use:**
```bash
# Linux/Mac
lsof -i :5050

# Change port if needed
FACE_VIEWER_PORT=8080 ./scripts/run.sh
```

**Check Python version:**
```bash
python3 --version  # Should be 3.11+
```

### GPU not detected

**Docker:**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install NVIDIA Container Toolkit
```

**Manual:**
```bash
# Check GPU
nvidia-smi

# Check PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory

Lower batch size:
```json
{"indexing": {"batch_size": 8}}
```

### Processing is slow

1. Verify GPU is being used (check logs)
2. Increase batch size if you have memory
3. Use smaller model if needed

### Too many clusters (people split up)

Increase epsilon:
```json
{"clustering": {"eps": 0.7}}
```

### Too few clusters (different people grouped together)

Decrease epsilon:
```json
{"clustering": {"eps": 0.5}}
```

---

## File Structure Reference

```
face-gallery/
├── backend/                 # Face recognition engine
│   ├── src/face_search/    # Core library
│   ├── examples/           # Example scripts
│   │   └── sort_images_by_person.py  # Main sorting script
│   ├── requirements.txt    # Backend dependencies
│   └── verify_setup.py     # Installation check
│
├── frontend/               # Web UI
│   ├── app.py             # Flask application
│   ├── static/            # HTML/CSS/JS
│   ├── requirements.txt   # Frontend dependencies
│   └── config.json        # Runtime config (created)
│
├── docker/                 # Docker files
│   ├── Dockerfile         # GPU image
│   └── Dockerfile.cpu     # CPU image
│
├── scripts/                # Helper scripts
│   ├── install.sh         # Install everything
│   └── run.sh            # Start the app
│
├── config/                 # Configuration
│   ├── config.example.json # Example config
│   ├── config.json        # Your config (created)
│   └── README.md          # Config documentation
│
├── data/                   # Your data
│   ├── photos/            # Put photos here
│   ├── collections/       # Processed collections
│   └── uploads/           # Temporary uploads
│
├── docker-compose.yml      # Docker GPU setup
├── docker-compose.cpu.yml  # Docker CPU setup
├── .env                    # Environment config (create from .env.example)
├── README.md              # Main documentation
├── QUICKSTART.md          # Quick start guide
└── SETUP_STEPS.md         # This file
```

---

## Next Steps After Setup

1. **Label people**: Click on person folders and add names
2. **Search**: Use image search to find specific people
3. **Share** (optional): Configure Google Drive integration
4. **Backup**: Back up your `data/collections/` folder
5. **Tune**: Adjust clustering parameters for better results

---

## Getting Help

1. Check [QUICKSTART.md](QUICKSTART.md) for common issues
2. Read [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed install help
3. See [config/README.md](config/README.md) for config options
4. Open an [issue](https://github.com/yourusername/face-gallery/issues) if stuck

---

**Remember**: Start small! Test with 50-100 photos before processing thousands.
