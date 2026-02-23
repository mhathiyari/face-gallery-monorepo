"""Tests for deduplication utilities."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

from face_search.deduplication import (
    compute_image_hash,
    compute_perceptual_distance,
    bbox_overlap,
    compute_embedding_similarity,
    is_duplicate_face,
    ImageHashCache,
    FaceDeduplicator
)
from face_search.face import BoundingBox, FaceDetection


@pytest.fixture
def temp_image():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(f.name)
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_image2():
    """Create a second temporary test image (different)."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a different test image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(f.name)
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestImageHashing:
    """Tests for image hashing functions."""

    def test_compute_file_hash(self, temp_image):
        """Test computing file-based hash."""
        hash1 = compute_image_hash(temp_image, use_perceptual=False)

        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Same file should produce same hash
        hash2 = compute_image_hash(temp_image, use_perceptual=False)
        assert hash1 == hash2

    def test_compute_perceptual_hash(self, temp_image):
        """Test computing perceptual hash."""
        hash1 = compute_image_hash(temp_image, use_perceptual=True)

        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Same file should produce same hash
        hash2 = compute_image_hash(temp_image, use_perceptual=True)
        assert hash1 == hash2

    def test_different_images_different_hashes(self, temp_image, temp_image2):
        """Test that different images produce different hashes."""
        hash1 = compute_image_hash(temp_image, use_perceptual=False)
        hash2 = compute_image_hash(temp_image2, use_perceptual=False)

        assert hash1 != hash2

    def test_hash_nonexistent_file(self):
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            compute_image_hash('/nonexistent/path/image.jpg')

    def test_perceptual_distance(self):
        """Test computing perceptual distance."""
        # Same hash = distance 0
        hash1 = "abc123"
        distance = compute_perceptual_distance(hash1, hash1)
        assert distance == 0

        # Different hashes
        hash2 = "abc124"  # 1 bit different
        distance = compute_perceptual_distance(hash1, hash2)
        assert distance > 0


class TestBoundingBoxOverlap:
    """Tests for bounding box overlap."""

    def test_overlapping_boxes(self):
        """Test IoU with overlapping boxes."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        bbox2 = BoundingBox(x1=50, y1=50, x2=150, y2=150)

        iou = bbox_overlap(bbox1, bbox2)
        assert 0.14 < iou < 0.15  # Should be ~0.143

    def test_no_overlap(self):
        """Test IoU with no overlap."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        bbox2 = BoundingBox(x1=100, y1=100, x2=150, y2=150)

        iou = bbox_overlap(bbox1, bbox2)
        assert iou == 0.0

    def test_complete_overlap(self):
        """Test IoU with identical boxes."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        bbox2 = BoundingBox(x1=0, y1=0, x2=100, y2=100)

        iou = bbox_overlap(bbox1, bbox2)
        assert iou == 1.0


class TestEmbeddingSimilarity:
    """Tests for embedding similarity."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical embeddings."""
        emb = np.random.randn(512).astype(np.float32)

        similarity = compute_embedding_similarity(emb, emb, metric='cosine')
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_different(self):
        """Test cosine similarity with different embeddings."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)

        similarity = compute_embedding_similarity(emb1, emb2, metric='cosine')
        assert -1.0 <= similarity <= 1.0

    def test_euclidean_distance_identical(self):
        """Test Euclidean distance with identical embeddings."""
        emb = np.random.randn(512).astype(np.float32)

        distance = compute_embedding_similarity(emb, emb, metric='euclidean')
        assert abs(distance) < 1e-6

    def test_euclidean_distance_different(self):
        """Test Euclidean distance with different embeddings."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)

        distance = compute_embedding_similarity(emb1, emb2, metric='euclidean')
        assert distance > 0

    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(256).astype(np.float32)

        with pytest.raises(ValueError, match="shapes don't match"):
            compute_embedding_similarity(emb1, emb2)

    def test_unknown_metric(self):
        """Test error on unknown metric."""
        emb = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_embedding_similarity(emb, emb, metric='invalid')


class TestFaceDuplication:
    """Tests for face duplication detection."""

    def test_duplicate_same_bbox_and_embedding(self, temp_image):
        """Test detecting duplicates with same bbox and embedding."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512).astype(np.float32)

        face1 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )

        face2 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )

        assert is_duplicate_face(face1, face2) is True

    def test_not_duplicate_different_bbox(self, temp_image):
        """Test not detecting duplicates with different bbox."""
        embedding = np.random.randn(512).astype(np.float32)

        face1 = FaceDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )

        face2 = FaceDetection(
            bbox=BoundingBox(x1=200, y1=300, x2=400, y2=500),  # No overlap
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )

        assert is_duplicate_face(face1, face2) is False

    def test_not_duplicate_different_embedding(self, temp_image):
        """Test not detecting duplicates with different embeddings."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)

        face1 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=np.random.randn(512).astype(np.float32),
            image_path=temp_image
        )

        face2 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=np.random.randn(512).astype(np.float32),  # Different
            image_path=temp_image
        )

        assert is_duplicate_face(face1, face2) is False

    def test_not_duplicate_different_image(self, temp_image, temp_image2):
        """Test not detecting duplicates from different images."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512).astype(np.float32)

        face1 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )

        face2 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image2  # Different image
        )

        # With check_same_image=True (default)
        assert is_duplicate_face(face1, face2, check_same_image=True) is False

        # With check_same_image=False
        assert is_duplicate_face(face1, face2, check_same_image=False) is True

    def test_duplicate_no_embedding(self, temp_image):
        """Test duplicate detection without embeddings (bbox only)."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)

        face1 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=None,
            image_path=temp_image
        )

        face2 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=None,
            image_path=temp_image
        )

        # Should detect as duplicate based on bbox alone
        assert is_duplicate_face(face1, face2) is True


class TestImageHashCache:
    """Tests for ImageHashCache."""

    def test_cache_basic(self, temp_image):
        """Test basic cache functionality."""
        cache = ImageHashCache()

        # First call - cache miss
        hash1 = cache.get_hash(temp_image)
        assert cache._misses == 1
        assert cache._hits == 0

        # Second call - cache hit
        hash2 = cache.get_hash(temp_image)
        assert cache._misses == 1
        assert cache._hits == 1

        assert hash1 == hash2

    def test_cache_hit_rate(self, temp_image):
        """Test cache hit rate calculation."""
        cache = ImageHashCache()

        # Get hash 3 times
        cache.get_hash(temp_image)
        cache.get_hash(temp_image)
        cache.get_hash(temp_image)

        # 1 miss, 2 hits = 66.7% hit rate
        assert abs(cache.hit_rate - 0.667) < 0.01

    def test_is_duplicate_image(self, temp_image, temp_image2):
        """Test duplicate image detection."""
        cache = ImageHashCache(use_perceptual=False)

        # Same image
        assert cache.is_duplicate_image(temp_image, temp_image) is True

        # Different images
        assert cache.is_duplicate_image(temp_image, temp_image2) is False

    def test_cache_clear(self, temp_image):
        """Test clearing cache."""
        cache = ImageHashCache()

        cache.get_hash(temp_image)
        assert cache.size == 1

        cache.clear()
        assert cache.size == 0
        assert cache._hits == 0
        assert cache._misses == 0


class TestFaceDeduplicator:
    """Tests for FaceDeduplicator."""

    def test_deduplicator_basic(self, temp_image):
        """Test basic deduplicator functionality."""
        dedup = FaceDeduplicator(window_size=10)

        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512).astype(np.float32)

        face1 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )

        # First face is not a duplicate
        assert dedup.is_duplicate(face1) is False
        dedup.add(face1)

        # Same face is a duplicate
        face2 = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )
        assert dedup.is_duplicate(face2) is True

    def test_deduplicator_window_size(self, temp_image):
        """Test deduplicator window size limit."""
        dedup = FaceDeduplicator(window_size=2)

        # Add 3 faces
        for i in range(3):
            face = FaceDetection(
                bbox=BoundingBox(x1=i*10, y1=i*10, x2=i*10+50, y2=i*10+50),
                confidence=0.95,
                embedding=np.random.randn(512).astype(np.float32),
                image_path=temp_image
            )
            dedup.add(face)

        # Window should only keep last 2
        assert len(dedup._faces) == 2

    def test_deduplicator_clear(self, temp_image):
        """Test clearing deduplicator."""
        dedup = FaceDeduplicator()

        face = FaceDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.95,
            embedding=np.random.randn(512).astype(np.float32),
            image_path=temp_image
        )

        dedup.add(face)
        assert len(dedup._faces) == 1

        dedup.clear()
        assert len(dedup._faces) == 0
        assert dedup.duplicates_found == 0

    def test_deduplicator_count(self, temp_image):
        """Test duplicate counting."""
        dedup = FaceDeduplicator()

        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512).astype(np.float32)

        face = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            image_path=temp_image
        )

        dedup.add(face)

        # Check same face 3 times
        for _ in range(3):
            duplicate_face = FaceDetection(
                bbox=bbox,
                confidence=0.95,
                embedding=embedding,
                image_path=temp_image
            )
            dedup.is_duplicate(duplicate_face)

        assert dedup.duplicates_found == 3
