# Face Search System

> Offline, GPU-accelerated face recognition system - An open-source alternative to AWS Rekognition

Fast, accurate face detection, recognition, and clustering using InsightFace and FAISS. **Sort your photos by person, find similar faces, or build face search into your applications.**

## 🚀 Quick Start - Sort Images by Person

**Want to organize your photos by the people in them? Here's how:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the sorting script
python examples/sort_images_by_person.py /path/to/your/photos /path/to/output

# Done! Your photos are now organized by person in the output folder.
```

**Example:**
```bash
python examples/sort_images_by_person.py ~/Photos/party ~/Photos/sorted
```

**Output structure:**
```
sorted/
├── person_000/          # All images of person 1
│   ├── IMG_001.jpg
│   ├── IMG_005.jpg
│   └── IMG_012.jpg
├── person_001/          # All images of person 2
│   ├── IMG_002.jpg
│   └── IMG_008.jpg
├── person_002/          # All images of person 3
│   └── ...
└── unmatched/           # Single faces or unclear matches
    └── IMG_099.jpg
```

## ✨ Features

- **🔍 Face Detection** - Detect faces in images with high accuracy
- **🎯 Face Recognition** - Generate 512-dimensional embeddings for each face
- **🔗 Face Clustering** - Automatically group faces of the same person
- **⚡ Fast Indexing** - Process 800+ faces per second
- **💾 Persistent Collections** - Save and reload face databases
- **🔎 Similarity Search** - Find similar faces in your collection
- **📊 AWS Rekognition-Compatible** - Familiar API format
- **🎨 Deduplication** - Automatically skip duplicate faces
- **♻️ Checkpoint/Resume** - Resume interrupted indexing jobs

## 📋 Requirements

- Python 3.11+
- CUDA-capable GPU (optional, but recommended for speed)
  - Falls back to CPU if GPU not available
- 4GB+ RAM recommended

## 🔧 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd face-search-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter InsightFace installation issues:
```bash
pip install insightface onnxruntime
```

For GPU support:
```bash
pip install onnxruntime-gpu
pip install faiss-gpu  # Instead of faiss-cpu
```

### 3. Verify installation

```bash
python verify_setup.py
```

You should see:
```
✓ Python 3.11.9
✓ NumPy 1.26.4
✓ PyTorch 2.x.x
✓ FAISS
✓ InsightFace
✓ All systems ready!
```

## 📖 Usage Examples

### Example 1: Sort Images by Person (Simplest)

```python
from pathlib import Path
from face_search.models import InsightFaceModel
from face_search.storage import Collection
from face_search.indexer import BatchIndexer
from face_search.search import cluster_faces

# 1. Load model
model = InsightFaceModel(model_name='buffalo_l', device='auto')

# 2. Create collection
collection = Collection(
    collection_id='my_photos',
    collections_root=Path('./collections'),
    embedding_dim=512
)

# 3. Index your photos
indexer = BatchIndexer(model=model, collection=collection)
stats = indexer.index_folder('/path/to/photos')

print(f"Indexed {stats.total_faces_indexed} faces from {stats.total_images} images")

# 4. Cluster faces by person
all_faces = [...]  # Get faces from collection (see examples/sort_images_by_person.py)
clusters = cluster_faces(all_faces, eps=0.6, min_samples=2)

print(f"Found {len(clusters)} different people")
```

### Example 2: Find Similar Faces

```python
from face_search.search import SearchEngine

# Create search engine
engine = SearchEngine(model=model, collection=collection)

# Search for faces similar to a query image
results = engine.search_by_image(
    image_path='query.jpg',
    max_results=10,
    min_similarity=0.8
)

# Print results
for rank, result in enumerate(results[0], 1):
    print(f"{rank}. {result.face.image_path} (similarity: {result.similarity:.2%})")
```

### Example 3: Compare Two Images

```python
# Compare faces between two images
comparison = engine.compare_faces(
    source_image='person1.jpg',
    target_image='person2.jpg'
)

match = comparison['best_match']
print(f"Similarity: {match['similarity']:.2%}")

if match['similarity'] > 0.8:
    print("✅ Same person!")
else:
    print("❌ Different people")
```

### Example 4: Batch Processing with Progress

```python
from face_search.indexer import BatchIndexer

indexer = BatchIndexer(
    model=model,
    collection=collection,
    batch_size=32,  # Process 32 images at a time
    enable_deduplication=True  # Skip duplicate faces
)

# Index with automatic checkpoint/resume
stats = indexer.index_folder(
    folder_path='/large/photo/library',
    recursive=True,
    min_confidence=0.7  # Only keep confident detections
)

print(f"""
Indexing Complete:
  Images: {stats.total_images}
  Faces: {stats.total_faces_indexed}
  Speed: {stats.faces_per_second:.1f} faces/sec
  Duplicates skipped: {stats.duplicates_skipped}
""")
```

## 🏗️ Architecture

```
face-search-system/
├── src/face_search/
│   ├── models/              # Face detection & embedding models
│   │   ├── base.py         # Abstract model interface
│   │   └── insightface_model.py  # InsightFace implementation
│   ├── storage/            # Data persistence
│   │   ├── faiss_index.py  # Vector similarity search
│   │   ├── metadata_db.py  # SQLite metadata storage
│   │   └── collection_manager.py  # Collection management
│   ├── indexer.py          # Batch face indexing
│   ├── face.py             # Data classes
│   ├── deduplication.py    # Duplicate detection
│   └── search/             # Search engine
│       ├── engine.py       # High-level search API
│       ├── results.py      # Result formatting
│       └── clustering.py   # Face clustering (DBSCAN)
├── tests/
│   ├── unit/               # Unit tests (198+ tests)
│   └── integration/        # Integration tests
├── examples/               # Example scripts
└── collections/            # Stored face databases (auto-created)
```

## 🎯 How It Works

1. **Face Detection** - InsightFace detects faces and facial landmarks
2. **Embedding Generation** - Each face → 512-dimensional vector
3. **FAISS Indexing** - Fast similarity search using vector index
4. **Metadata Storage** - SQLite stores face locations, confidence, etc.
5. **Clustering** - DBSCAN groups similar faces (same person)
6. **Search** - Find similar faces in milliseconds

## 🔬 Technical Details

### Models
- **Face Detection:** RetinaFace (from InsightFace)
- **Face Recognition:** ArcFace (512-dimensional embeddings)
- **Similarity Metric:** L2 distance (Euclidean)

### Performance
- **Indexing Speed:** 800+ faces/second (measured)
- **Search Latency:** <100ms for 10k faces (target)
- **Embedding Dimension:** 512
- **Index Type:** FAISS Flat or IVF for large collections

### Storage
- **Vector Index:** FAISS (Facebook AI Similarity Search)
- **Metadata:** SQLite with soft deletes
- **Synchronization:** Atomic operations with file locking

## 📊 API Reference

### Core Classes

#### `InsightFaceModel`
Face detection and embedding model.

```python
model = InsightFaceModel(
    model_name='buffalo_l',  # Model variant
    device='auto'            # 'auto', 'cuda', 'cpu', 'mps'
)

# Detect faces
faces = model.detect_faces('image.jpg', min_confidence=0.5)
```

#### `Collection`
Manages a face collection (FAISS index + metadata).

```python
collection = Collection(
    collection_id='my_collection',
    collections_root=Path('./collections'),
    embedding_dim=512
)

# Add a face
face_id = collection.add_face(
    image_path='photo.jpg',
    bbox={'x1': 10, 'y1': 20, 'x2': 100, 'y2': 150},
    embedding=embedding_vector,
    confidence=0.95
)

# Search
results = collection.search(query_embedding, k=10)

# Save/load
collection.save()
```

#### `BatchIndexer`
Batch process image folders.

```python
indexer = BatchIndexer(
    model=model,
    collection=collection,
    batch_size=32,
    enable_deduplication=True
)

stats = indexer.index_folder('photos/')
```

#### `SearchEngine`
High-level search interface.

```python
engine = SearchEngine(model=model, collection=collection)

# Search by image
results = engine.search_by_image('query.jpg', max_results=10)

# Search by face ID
results = engine.search_by_face_id(face_id=123, max_results=10)

# Compare two images
comparison = engine.compare_faces('img1.jpg', 'img2.jpg')
```

#### Face Clustering

```python
from face_search.search import cluster_faces, find_representative_faces

# Cluster faces by identity
clusters = cluster_faces(
    faces=face_list,
    eps=0.6,          # Distance threshold
    min_samples=2     # Minimum cluster size
)

# Find best representative for each cluster
for cluster_id, cluster_faces in clusters.items():
    rep = find_representative_faces(
        cluster_faces,
        method='highest_confidence',  # or 'centroid', 'highest_quality'
        n_representatives=1
    )[0]
    print(f"Person {cluster_id}: {rep.image_path}")
```

## 🎨 Advanced Usage

### Adjusting Clustering Sensitivity

The `eps` parameter controls how strict face matching is:

```python
# Strict matching (fewer, smaller clusters)
clusters = cluster_faces(faces, eps=0.4, min_samples=2)

# Loose matching (more faces grouped together)
clusters = cluster_faces(faces, eps=0.8, min_samples=2)

# Recommended: 0.6 for general photos
clusters = cluster_faces(faces, eps=0.6, min_samples=2)
```

### Custom Result Formatting

```python
from face_search.search import format_results_aws_style, format_results_simple

# AWS Rekognition format
aws_results = format_results_aws_style(results)

# Simple format
simple_results = format_results_simple(results, max_results=10)

# Export to JSON
from face_search.search import export_results_json
export_results_json(results, 'results.json', format_style='simple')
```

### Filtering Results

```python
from face_search.search import filter_results

# Only high-confidence matches
filtered = filter_results(
    results,
    min_similarity=0.8,
    min_confidence=0.9
)

# Exclude specific images
filtered = filter_results(
    results,
    exclude_images=['selfie.jpg', 'group_photo.jpg']
)
```

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests (slower)
pytest tests/integration/

# With coverage
pytest --cov=face_search --cov-report=html
```

**Test Statistics:**
- Unit tests: 198+ tests, 100% passing
- Integration tests: 10 tests
- Total coverage: ~85%

## ⚙️ Configuration

### Adjusting Performance

```python
# For large collections (>100k faces), use IVF index
from face_search.storage import FAISSIndex, IndexType

index = FAISSIndex(
    dim=512,
    device='cuda',
    index_type=IndexType.IVF,  # Faster for large collections
    metric='L2'
)
```

### GPU Memory Management

```python
# Monitor GPU memory
stats = collection.index.get_stats()
print(f"GPU memory: {stats.get('gpu_memory_mb', 0)}MB")

# For limited GPU memory, use smaller batches
indexer = BatchIndexer(
    model=model,
    collection=collection,
    batch_size=8  # Smaller batches use less memory
)
```

## 🐛 Troubleshooting

### "No module named 'insightface'"
```bash
pip install insightface onnxruntime
```

### "CUDA out of memory"
- Reduce `batch_size` in BatchIndexer
- Use CPU mode: `device='cpu'`
- Close other GPU applications

### "No faces detected"
- Check image quality (resolution, lighting)
- Lower `min_confidence` threshold
- Verify faces are clearly visible

### FAISS crashes on macOS
- Known issue with faiss-cpu on macOS
- Use faiss-gpu or run on Linux/Windows
- Indexing works fine; search has platform issues

### Clustering too many/few groups
- Adjust `eps` parameter:
  - Too many clusters? Increase `eps` (e.g., 0.8)
  - Too few clusters? Decrease `eps` (e.g., 0.4)

## 📝 Project Status

**✅ Completed (Week 1-2):**
- ✅ Core face detection and recognition
- ✅ FAISS vector indexing
- ✅ Metadata storage and management
- ✅ Batch indexing pipeline
- ✅ Face search engine
- ✅ Face clustering (DBSCAN)
- ✅ Deduplication
- ✅ Checkpoint/resume
- ✅ Comprehensive testing (198+ unit tests)
- ✅ Integration testing
- ✅ Working examples

**⏳ In Progress (Week 3):**
- High-level Python API (FaceSearchClient)
- Command-line interface (CLI)
- Package installation (pip install)

**🔮 Planned (Week 4):**
- FastAPI web interface
- GPU optimization
- Extended documentation
- Performance benchmarks

## 📚 Documentation

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Detailed development roadmap
- [examples/](examples/) - Working code examples
- [tests/](tests/) - Comprehensive test suite

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

## 🙏 Acknowledgments

- **InsightFace** - Face detection and recognition models
- **FAISS** - Vector similarity search library
- **PyTorch** - Deep learning framework

---

**Built with ❤️ for the computer vision community**
