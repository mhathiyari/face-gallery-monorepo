"""Tests for batch indexer."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from face_search.indexer import BatchIndexer, is_image_file
from face_search.models import InsightFaceModel
from face_search.storage import Collection


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test images."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_images(temp_dir):
    """Create temporary test images."""
    image_paths = []

    for i in range(5):
        image_path = temp_dir / f'test_image_{i}.jpg'
        # Create a simple test image
        img = Image.new('RGB', (200, 200), color=(i * 50, 100, 150))
        img.save(image_path)
        image_paths.append(image_path)

    return image_paths


@pytest.fixture
def temp_collection(temp_dir):
    """Create a temporary collection for testing."""
    collection = Collection(
        collection_id='test_indexer',
        collections_root=temp_dir / 'collections',
        embedding_dim=512
    )
    yield collection
    # Cleanup will happen with temp_dir


@pytest.fixture
def mock_model():
    """Create a mock model for testing (to avoid GPU requirements)."""
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
            self.embedding_dim = 512

        def detect_faces(self, image_path, min_confidence=0.5):
            """Mock face detection that returns fake faces."""
            from face_search.models.base import DetectedFace, BoundingBox

            # Return 0-2 random faces
            num_faces = np.random.randint(0, 3)
            faces = []

            for i in range(num_faces):
                face = DetectedFace(
                    bbox=BoundingBox(
                        x1=10 + i * 50,
                        y1=20 + i * 50,
                        x2=100 + i * 50,
                        y2=150 + i * 50
                    ),
                    confidence=0.95,
                    embedding=np.random.randn(512).astype(np.float32),
                    landmarks=np.array([
                        [30, 40], [70, 40], [50, 70], [40, 100], [60, 100]
                    ], dtype=np.float32)
                )
                faces.append(face)

            return faces

        def get_embedding(self, face_image):
            """Mock embedding generation."""
            return np.random.randn(512).astype(np.float32)

        def get_embeddings_batch(self, face_images, batch_size=32):
            """Mock batch embedding generation."""
            return np.random.randn(len(face_images), 512).astype(np.float32)

    return MockModel()


class TestImageFileDetection:
    """Tests for image file detection."""

    def test_is_image_file_valid(self):
        """Test detecting valid image files."""
        assert is_image_file(Path('image.jpg')) is True
        assert is_image_file(Path('image.jpeg')) is True
        assert is_image_file(Path('image.png')) is True
        assert is_image_file(Path('image.JPG')) is True

    def test_is_image_file_invalid(self):
        """Test rejecting non-image files."""
        assert is_image_file(Path('document.pdf')) is False
        assert is_image_file(Path('video.mp4')) is False
        assert is_image_file(Path('text.txt')) is False


class TestBatchIndexer:
    """Tests for BatchIndexer."""

    def test_create_indexer(self, mock_model, temp_collection):
        """Test creating a batch indexer."""
        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            batch_size=32
        )

        assert indexer.model == mock_model
        assert indexer.collection == temp_collection
        assert indexer.batch_size == 32

    def test_index_single_image(self, mock_model, temp_collection, temp_images):
        """Test indexing a single image."""
        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection
        )

        result = indexer.index_image(temp_images[0], min_confidence=0.5)

        assert result.image_path == temp_images[0]
        assert result.faces_found >= 0
        # Faces indexed might be less due to deduplication

    def test_index_folder(self, mock_model, temp_collection, temp_images):
        """Test indexing a folder of images."""
        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            batch_size=4
        )

        stats = indexer.index_folder(
            folder_path=temp_images[0].parent,
            recursive=False,
            min_confidence=0.5
        )

        assert stats.total_images == len(temp_images)
        assert stats.processing_time > 0
        # Can't assert exact numbers due to random mock

    def test_index_folder_recursive(self, mock_model, temp_collection, temp_dir):
        """Test recursive folder indexing."""
        # Create subdirectory with images
        subdir = temp_dir / 'subdir'
        subdir.mkdir()

        for i in range(3):
            img = Image.new('RGB', (200, 200), color=(i * 80, 120, 180))
            img.save(subdir / f'sub_image_{i}.jpg')

        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection
        )

        # Non-recursive should find 0 images in temp_dir
        stats = indexer.index_folder(
            folder_path=temp_dir,
            recursive=False
        )
        assert stats.total_images == 0

        # Recursive should find 3 images
        indexer2 = BatchIndexer(
            model=mock_model,
            collection=temp_collection
        )
        stats = indexer2.index_folder(
            folder_path=temp_dir,
            recursive=True
        )
        assert stats.total_images == 3

    def test_deduplication(self, mock_model, temp_collection, temp_images):
        """Test face deduplication."""
        # With deduplication
        indexer_with_dedup = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            enable_deduplication=True
        )

        # Without deduplication
        indexer_no_dedup = BatchIndexer(
            model=mock_model,
            collection=Collection(
                collection_id='test_no_dedup',
                collections_root=temp_collection.collection_path.parent,
                embedding_dim=512
            ),
            enable_deduplication=False
        )

        # Index with deduplication
        stats_with = indexer_with_dedup.index_folder(temp_images[0].parent)

        # Index without deduplication (new collection)
        stats_without = indexer_no_dedup.index_folder(temp_images[0].parent)

        # With dedup might have fewer or equal faces indexed
        # (can't guarantee more duplicates due to random mock)
        assert stats_with.total_images == stats_without.total_images

    def test_checkpoint_save_load(self, mock_model, temp_collection, temp_images, temp_dir):
        """Test checkpoint saving and loading."""
        checkpoint_path = temp_dir / 'checkpoint.json'

        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            checkpoint_interval=2,
            checkpoint_path=checkpoint_path
        )

        # Index some images
        indexer.index_folder(temp_images[0].parent)

        # Checkpoint should exist
        assert checkpoint_path.exists()

        # Create new indexer and resume
        indexer2 = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            checkpoint_path=checkpoint_path
        )

        # Load checkpoint
        indexer2._load_checkpoint()

        # Should have processed images recorded
        assert len(indexer2._processed_images) > 0

    def test_resume_from_checkpoint(self, mock_model, temp_collection, temp_images, temp_dir):
        """Test resuming indexing from checkpoint."""
        checkpoint_path = temp_dir / 'checkpoint.json'

        # First indexer - process half the images
        indexer1 = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            checkpoint_path=checkpoint_path
        )

        # Process first 2 images and mark as processed
        for img in temp_images[:2]:
            indexer1.index_image(img)
            indexer1._processed_images.add(str(img.resolve()))

        indexer1._save_checkpoint()

        # Second indexer - resume
        indexer2 = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            checkpoint_path=checkpoint_path
        )

        stats = indexer2.index_folder(
            folder_path=temp_images[0].parent,
            resume=True
        )

        # Should only process remaining images (3 out of 5)
        assert stats.total_images == 3

    def test_progress_callback(self, mock_model, temp_collection, temp_images):
        """Test progress callback."""
        results = []

        def callback(result):
            results.append(result)

        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection
        )

        indexer.index_folder(
            folder_path=temp_images[0].parent,
            progress_callback=callback
        )

        # Callback should have been called for each image
        assert len(results) == len(temp_images)

    def test_get_stats(self, mock_model, temp_collection, temp_images):
        """Test getting statistics."""
        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection
        )

        stats = indexer.index_folder(temp_images[0].parent)

        # Get stats
        retrieved_stats = indexer.get_stats()

        assert retrieved_stats.total_images == stats.total_images
        assert retrieved_stats.processing_time == stats.processing_time

    def test_clear_checkpoint(self, mock_model, temp_collection, temp_dir):
        """Test clearing checkpoint."""
        checkpoint_path = temp_dir / 'checkpoint.json'

        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            checkpoint_path=checkpoint_path
        )

        # Save checkpoint
        indexer._save_checkpoint()
        assert checkpoint_path.exists()

        # Clear checkpoint
        indexer.clear_checkpoint()
        assert not checkpoint_path.exists()
        assert len(indexer._processed_images) == 0

    def test_error_handling(self, temp_collection, temp_dir):
        """Test error handling for invalid images."""
        # Create a model that validates images
        class ValidatingModel:
            def __init__(self):
                self.device = 'cpu'
                self.embedding_dim = 512

            def detect_faces(self, image_path, min_confidence=0.5):
                # Try to actually load the image
                from PIL import Image
                Image.open(image_path)  # This will raise if not valid
                return []

        # Create a non-image file
        invalid_file = temp_dir / 'not_an_image.jpg'
        invalid_file.write_text('This is not an image')

        indexer = BatchIndexer(
            model=ValidatingModel(),
            collection=temp_collection
        )

        # Should handle error gracefully
        result = indexer.index_image(invalid_file)

        assert result.faces_found == 0
        assert len(result.errors) > 0

    def test_min_confidence_filtering(self, temp_collection, temp_dir):
        """Test minimum confidence filtering."""
        class ConfidenceModel:
            """Mock model with controllable confidence."""
            def __init__(self):
                self.device = 'cpu'
                self.embedding_dim = 512

            def detect_faces(self, image_path, min_confidence=0.5):
                from face_search.models.base import DetectedFace, BoundingBox

                # Return faces with varying confidence
                faces = []
                for conf in [0.3, 0.6, 0.9]:
                    if conf >= min_confidence:
                        face = DetectedFace(
                            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
                            confidence=conf,
                            embedding=np.random.randn(512).astype(np.float32),
                            landmarks=np.array([
                                [30, 40], [70, 40], [50, 70], [40, 100], [60, 100]
                            ], dtype=np.float32)
                        )
                        faces.append(face)
                return faces

        # Create test image
        test_img = temp_dir / 'test.jpg'
        Image.new('RGB', (200, 200), color='red').save(test_img)

        # Index with different thresholds
        model = ConfidenceModel()
        indexer = BatchIndexer(model=model, collection=temp_collection)

        # With min_confidence=0.5, should get 2 faces (0.6, 0.9)
        result = indexer.index_image(test_img, min_confidence=0.5)
        assert result.faces_found == 2

    def test_batch_indexing(self, mock_model, temp_collection, temp_images):
        """Test batch indexing multiple images."""
        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection,
            batch_size=2
        )

        results = indexer.index_images_batch(
            image_paths=temp_images[:3],
            min_confidence=0.5,
            show_progress=False
        )

        assert len(results) == 3
        for result in results:
            assert result.image_path in temp_images[:3]


class TestBatchIndexingStats:
    """Tests for batch indexing statistics."""

    def test_stats_accumulation(self, mock_model, temp_collection, temp_images):
        """Test that stats accumulate correctly."""
        indexer = BatchIndexer(
            model=mock_model,
            collection=temp_collection
        )

        stats = indexer.index_folder(temp_images[0].parent)

        # Check basic stats
        assert stats.total_images > 0
        assert stats.total_faces_found >= 0
        assert stats.processing_time > 0

        # Check calculated properties
        assert 0.0 <= stats.success_rate <= 1.0
        assert stats.faces_per_image >= 0
        assert stats.images_per_second >= 0
