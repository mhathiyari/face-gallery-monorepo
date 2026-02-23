# Configuration Guide

This directory contains the configuration files for Face Gallery.

## Quick Start

1. Copy the example config:
   ```bash
   cp config.example.json config.json
   ```

2. Edit `config.json` with your settings.

## Configuration Options

### Paths

- **photos_dir**: Where your source photos are located
- **collections_dir**: Where processed face collections are stored
- **uploads_dir**: Temporary upload directory for web UI
- **backend_root**: Path to the backend (face-search-system) code

### Server

- **host**: Server bind address (default: 0.0.0.0)
- **port**: Port to run the web server on (default: 5050)
- **debug**: Enable Flask debug mode (default: false)

### Model

- **name**: InsightFace model variant
  - `buffalo_l`: Large model (best accuracy)
  - `buffalo_m`: Medium model (balanced)
  - `buffalo_s`: Small model (fastest)

- **device**: Computation device
  - `auto`: Automatically detect (GPU if available, else CPU)
  - `cuda`: Force NVIDIA GPU
  - `cpu`: Force CPU
  - `mps`: Apple Silicon GPU (macOS)

- **min_confidence**: Minimum confidence for face detection (0.0-1.0, default: 0.5)

### Clustering

- **eps**: Distance threshold for clustering
  - Lower (0.4): Stricter matching, more clusters
  - Medium (0.6): Recommended default
  - Higher (0.8): Looser matching, fewer clusters

- **min_samples**: Minimum faces needed to form a cluster (default: 2)

### Indexing

- **batch_size**: Images to process at once
  - Lower (8-16): Less GPU memory, slower
  - Medium (32): Good balance
  - Higher (64+): More GPU memory, faster

- **enable_deduplication**: Skip duplicate faces (default: true)
- **checkpoint_interval**: Save progress every N images (default: 100)

### Search

- **max_results**: Maximum search results to return (default: 10)
- **min_similarity**: Minimum similarity threshold (0.0-1.0, default: 0.8)

### Drive (Optional)

Configure Google Drive integration for sharing:

- **credentials_path**: Path to OAuth credentials.json
- **token_path**: Where to store the access token
- **enabled**: Enable/disable Drive features (default: false)

## Environment Variables

You can override config values with environment variables:

- `FACE_VIEWER_PORT`: Override server port
- `FACE_VIEWER_HOST`: Override server host
- `FACE_VIEWER_DEBUG`: Enable debug mode (1/true)
- `PHOTOS_DIR`: Override photos directory
- `COLLECTIONS_DIR`: Override collections directory

## Docker Configuration

When using Docker, mount your config:

```yaml
volumes:
  - ./config:/app/config:ro
```

Or use environment variables in `.env`:

```bash
PHOTOS_DIR=/path/to/photos
COLLECTIONS_DIR=/path/to/collections
FACE_VIEWER_PORT=5050
```

## Examples

### Minimal Config (CPU, fast setup)
```json
{
  "paths": {
    "photos_dir": "./data/photos"
  },
  "model": {
    "device": "cpu"
  }
}
```

### GPU Config (best performance)
```json
{
  "paths": {
    "photos_dir": "/mnt/photos"
  },
  "model": {
    "device": "cuda",
    "name": "buffalo_l"
  },
  "indexing": {
    "batch_size": 64
  }
}
```

### Strict Matching (fewer, cleaner clusters)
```json
{
  "clustering": {
    "eps": 0.4,
    "min_samples": 3
  }
}
```

## Troubleshooting

### GPU not detected
Set `"device": "cuda"` explicitly in config.

### Out of memory errors
Lower `batch_size` in indexing config.

### Too many/few clusters
Adjust `eps` in clustering config:
- Too many clusters → increase eps (e.g., 0.7)
- Too few clusters → decrease eps (e.g., 0.5)
