# Face Gallery

> Self-hosted face recognition photo organizer with GPU acceleration

Automatically sort your photo collection by the people in them. Fast, private, and runs entirely on your own hardware.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)

## Features

- **Smart Face Recognition** - Powered by InsightFace and FAISS
- **Automatic Clustering** - Groups photos by person automatically
- **Web Interface** - Browse and manage your photo collections
- **GPU Accelerated** - Fast processing with CUDA support
- **Privacy First** - Everything runs locally, your photos never leave your machine
- **Easy Setup** - Docker or simple install script
- **Search by Photo** - Find all photos of a person using a sample image

## Quick Start

### Option 1: Docker (Recommended)

**With GPU (fastest):**
```bash
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery
cp .env.example .env
# Edit .env to set your photo paths
docker-compose up
```

**CPU only:**
```bash
docker-compose -f docker-compose.cpu.yml up
```

Open http://localhost:5050 in your browser.

### Option 2: Manual Install

**Requirements:**
- Python 3.11+
- NVIDIA GPU with CUDA (optional, but recommended)

**Install:**
```bash
git clone https://github.com/yourusername/face-gallery.git
cd face-gallery
./scripts/install.sh
```

**Run:**
```bash
./scripts/run.sh
```

Open http://localhost:5050 in your browser.

## What Does It Do?

1. **Index your photos** - Scans your photo folder and detects faces
2. **Cluster by person** - Automatically groups faces of the same person
3. **Browse results** - Web UI to view collections organized by person
4. **Search** - Find all photos containing a specific person
5. **Label & Share** - Add names and optionally share via Google Drive

## Project Structure

```
face-gallery/
├── backend/              # Face detection & recognition engine
│   ├── src/face_search/  # Core library
│   ├── examples/         # Example scripts
│   └── tests/            # Test suite
├── frontend/             # Web UI (Flask app)
│   ├── app.py           # Main application
│   └── static/          # HTML/CSS/JS
├── docker/               # Docker configurations
│   ├── Dockerfile       # GPU-enabled image
│   └── Dockerfile.cpu   # CPU-only image
├── scripts/              # Installation & run scripts
│   ├── install.sh       # One-command installer
│   └── run.sh           # Start the application
├── config/               # Configuration files
│   ├── config.example.json
│   └── README.md        # Config documentation
└── data/                 # Your data (created on setup)
    ├── photos/          # Source photos go here
    ├── collections/     # Processed collections stored here
    └── uploads/         # Temporary uploads
```

## Installation Methods Comparison

| Method | Difficulty | GPU Support | Best For |
|--------|-----------|-------------|----------|
| Docker + GPU | Easy | ✅ Yes | Most users with NVIDIA GPU |
| Docker CPU | Easy | ❌ No | Testing, small collections |
| Manual Install | Medium | ✅ Yes | Developers, custom setups |

## Configuration

### Basic Configuration

Edit `config/config.json`:

```json
{
  "paths": {
    "photos_dir": "./data/photos",
    "collections_dir": "./data/collections"
  },
  "server": {
    "port": 5050
  },
  "model": {
    "device": "auto"
  }
}
```

See [config/README.md](config/README.md) for full configuration options.

### Environment Variables

Use `.env` file for Docker or export in shell:

```bash
PHOTOS_DIR=/path/to/your/photos
COLLECTIONS_DIR=/path/to/output
FACE_VIEWER_PORT=5050
```

## Usage Guide

### 1. Add Photos

**Manual install:**
```bash
cp -r /path/to/your/photos/* data/photos/
```

**Docker:**
```bash
# Set PHOTOS_DIR in .env to point to your photos
```

### 2. Run Face Sorting

**Via Web UI:**
- Toggle "Owner Mode"
- Set input folder path
- Click "Start Sorting"

**Via Command Line:**
```bash
source .venv/bin/activate
python backend/examples/sort_images_by_person.py \
  /path/to/photos \
  /path/to/output
```

### 3. Browse Results

- Open web UI at http://localhost:5050
- Click "Add Collection" and select your sorted folder
- Browse people and photos
- Toggle between face crops and full images

### 4. Search by Photo

- Click "Search by Image"
- Upload a photo of a person
- View all matching photos

## Advanced Features

### Adjust Clustering Sensitivity

Edit `config/config.json`:

```json
{
  "clustering": {
    "eps": 0.6,
    "min_samples": 2
  }
}
```

- Lower `eps` (0.4): Stricter matching, more clusters
- Higher `eps` (0.8): Looser matching, fewer clusters

### Google Drive Sharing

1. Get OAuth credentials from Google Cloud Console
2. Save as `credentials.json` in project root
3. Enable in config:

```json
{
  "drive": {
    "enabled": true,
    "credentials_path": "./credentials.json"
  }
}
```

4. Use "Upload to Drive" feature in web UI

### Batch Processing Large Collections

For better performance with large collections:

```json
{
  "indexing": {
    "batch_size": 64,
    "enable_deduplication": true,
    "checkpoint_interval": 100
  }
}
```

## Performance

| Hardware | Processing Speed | Recommended Batch Size |
|----------|-----------------|------------------------|
| NVIDIA RTX 3060+ | 800-1200 faces/sec | 64 |
| NVIDIA GTX 1060 | 400-600 faces/sec | 32 |
| CPU (8 cores) | 50-100 faces/sec | 16 |

## Troubleshooting

### GPU not detected

Check Docker GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If that fails, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Out of memory

Lower batch size in config:
```json
{"indexing": {"batch_size": 16}}
```

### Too many/few clusters

Adjust `eps` in clustering config:
- Too many separate clusters? Increase `eps` to 0.7 or 0.8
- People grouped together incorrectly? Decrease `eps` to 0.5 or 0.4

### Docker build fails

Make sure you have enough disk space (10GB+ recommended) and try:
```bash
docker system prune -a
docker-compose build --no-cache
```

## Development

### Running Tests

```bash
source .venv/bin/activate
cd backend
pytest
```

### Backend Development

```bash
cd backend
pip install -e .
# Make changes
pytest
```

### Frontend Development

```bash
cd frontend
export FACE_VIEWER_DEBUG=1
python app.py
```

## Architecture

- **Backend**: Python face recognition engine using InsightFace (ArcFace embeddings) and FAISS (vector similarity search)
- **Frontend**: Flask web application with simple HTML/JS interface
- **Storage**: SQLite for metadata, FAISS for vector indexes
- **Clustering**: DBSCAN algorithm for grouping faces

## Requirements

- Python 3.11+
- 4GB+ RAM (8GB+ recommended)
- NVIDIA GPU with CUDA support (optional but recommended)
- Docker 20.10+ with NVIDIA Container Toolkit (for Docker install)

## License

MIT License - see LICENSE file for details

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Flask](https://flask.palletsprojects.com/) - Web framework

## Roadmap

- [ ] Multi-user support
- [ ] Mobile app
- [ ] Video support
- [ ] Advanced search filters
- [ ] Export collections as albums
- [ ] Cloud deployment guides

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/face-gallery/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/face-gallery/discussions)

## Privacy & Security

Face Gallery is designed with privacy in mind:
- All processing happens locally on your machine
- No data is sent to external services (unless you enable Drive sharing)
- Your photos never leave your control
- No telemetry or tracking

---

**Made with ❤️ for the privacy-conscious photographer**
