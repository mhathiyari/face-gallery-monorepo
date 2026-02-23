# Face Gallery

> Self-hosted face recognition photo organizer with GPU acceleration

Automatically sort your photo collection by the people in them. Fast, private, and runs entirely on your own hardware.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)

## Features

- **Smart Face Recognition** — Powered by InsightFace and FAISS
- **Automatic Clustering** — Groups photos by person automatically
- **Web Interface** — Browse and manage your photo collections
- **GPU Accelerated** — Fast processing with CUDA / Apple Silicon (MPS) support
- **Privacy First** — Everything runs locally, your photos never leave your machine
- **Search by Photo** — Find all photos of a person using a sample image
- **Google Drive Sharing** — Optionally share collections via Google Drive

## Quick Start

### Option 1: Docker (Recommended)

**With GPU (fastest):**
```bash
git clone https://github.com/mhathiyari/face-gallery-monorepo.git
cd face-gallery-monorepo
cp .env.example .env
# Edit .env to set your photo paths
docker compose up
```

**CPU only:**
```bash
docker compose -f docker-compose.cpu.yml up
```

Open http://localhost:5050 in your browser.

### Option 2: Manual Install

**Requirements:** Python 3.11+, NVIDIA GPU with CUDA or Apple Silicon (optional but recommended)

```bash
git clone https://github.com/mhathiyari/face-gallery-monorepo.git
cd face-gallery-monorepo
make install
make run
```

Or step by step:

```bash
pip install -e backend/[cpu]       # or backend/[gpu] for CUDA
pip install -r frontend/requirements.txt
python frontend/app.py
```

Open http://localhost:5050 in your browser.

## Usage

### 1. Sort Photos by Person

**Via command line:**
```bash
python backend/examples/sort_images_by_person.py \
  /path/to/photos \
  /path/to/output
```

**Via web UI:**
- Toggle "Owner Mode" on
- Set input and output folder paths
- Click "Start Sorting"

### 2. Browse Results

- Open http://localhost:5050
- Click "Add Collection" and enter the path to your sorted folder
- Browse people cards with face thumbnails
- Click a person to see all their photos
- Toggle between cropped faces and full images

### 3. Search by Photo

- Upload a photo of a person
- View all matching photos across the collection

## Project Structure

```
face-gallery-monorepo/
├── backend/                # Face detection & recognition engine
│   ├── src/face_search/    # Core library (models, search, storage)
│   ├── examples/           # CLI scripts (sort_images_by_person.py)
│   ├── tests/              # Unit & integration tests
│   └── pyproject.toml      # Backend package config
├── frontend/               # Web UI (Flask)
│   ├── app.py              # Main application
│   ├── config_loader.py    # Configuration system
│   ├── static/             # HTML / CSS / JS
│   └── requirements.txt    # Frontend dependencies
├── config/                 # Configuration
│   ├── config.example.json # Example config (copy to config.json)
│   └── README.md           # Config documentation
├── docker/                 # Docker images
│   ├── Dockerfile          # GPU-enabled image
│   └── Dockerfile.cpu      # CPU-only image
├── scripts/                # Automation
│   ├── install.sh          # One-command installer
│   ├── run.sh              # Start the application
│   └── verify.sh           # Verify installation
├── tests/                  # Root-level integration tests
│   └── test_frontend_smoke.py
├── docs/                   # Setup & installation guides
├── data/                   # Runtime data (gitignored)
│   ├── photos/
│   ├── collections/
│   └── uploads/
├── Makefile                # Dev workflow targets
├── requirements-dev.txt    # Dev dependencies (pytest, ruff, black, mypy)
├── docker-compose.yml      # GPU Docker Compose
└── docker-compose.cpu.yml  # CPU Docker Compose
```

## Configuration

Copy the example config and customize:

```bash
cp config/config.example.json config/config.json
```

Key settings:

```json
{
  "paths": {
    "photos_dir": "./data/photos",
    "collections_dir": "./data/collections"
  },
  "server": { "port": 5050 },
  "model": { "device": "auto" },
  "clustering": { "eps": 0.6, "min_samples": 2 }
}
```

- **`model.device`**: `"auto"` (detects GPU/MPS), `"cuda"`, `"mps"`, or `"cpu"`
- **`clustering.eps`**: Lower (0.4) = stricter matching, higher (0.8) = looser
- **`security.root_dir`**: Root directory for file access (default `"~"`)

See [config/README.md](config/README.md) for all options.

## Development

### Makefile Targets

```bash
make install         # Install backend + frontend + dev deps
make test            # Run all tests
make test-backend    # Backend unit tests only
make test-frontend   # Frontend smoke tests only
make lint            # Run ruff + black checks
make run             # Start the Flask server
make docker-gpu      # Build & run GPU Docker image
make docker-cpu      # Build & run CPU Docker image
```

### Running Tests

```bash
# All tests
make test

# Backend unit tests
cd backend && python -m pytest tests/unit/ -v

# Frontend smoke tests
python -m pytest tests/test_frontend_smoke.py -v
```

### Local Development

```bash
# Install in editable mode
pip install -e backend/[dev,cpu]
pip install -r frontend/requirements.txt
pip install -r requirements-dev.txt

# Run with debug mode
python frontend/app.py
```

## Architecture

- **Backend**: InsightFace (ArcFace embeddings) + FAISS (vector similarity search)
- **Frontend**: Flask web app with vanilla HTML/CSS/JS
- **Storage**: SQLite for metadata, FAISS indexes for embeddings
- **Clustering**: DBSCAN for grouping faces by identity
- **Representative Selection**: Automatic best-face selection using pose filtering and confidence scoring

## Performance

| Hardware | Processing Speed | Recommended Batch Size |
|----------|-----------------|------------------------|
| NVIDIA RTX 3060+ | 800–1200 faces/sec | 64 |
| Apple M1/M2 (MPS) | 200–400 faces/sec | 32 |
| CPU (8 cores) | 50–100 faces/sec | 16 |

## Troubleshooting

**GPU not detected:** Check NVIDIA Container Toolkit is installed, or that MPS is available on Apple Silicon.

**Out of memory:** Lower `indexing.batch_size` to 16 in config.

**Too many/few clusters:** Adjust `clustering.eps` — increase to merge more, decrease to split.

## Privacy & Security

- All processing happens locally on your machine
- No data is sent to external services (unless you enable Drive sharing)
- No telemetry or tracking

## License

MIT License — see [LICENSE](LICENSE) for details.

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) — Face recognition models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Flask](https://flask.palletsprojects.com/) — Web framework
