"""Tests for face data classes."""

import pytest
import numpy as np
from pathlib import Path

from face_search.face import (
    BoundingBox,
    FaceQuality,
    Landmarks,
    FaceDetection,
    SearchResult,
    IndexingResult,
    BatchIndexingStats
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_create_bbox(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)

        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 150

    def test_bbox_properties(self):
        """Test bounding box properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)

        assert bbox.width == 90
        assert bbox.height == 130
        assert bbox.area == 90 * 130
        assert bbox.center == (55, 85)

    def test_bbox_to_dict(self):
        """Test converting bbox to dictionary."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        bbox_dict = bbox.to_dict()

        assert bbox_dict == {
            'x1': 10,
            'y1': 20,
            'x2': 100,
            'y2': 150
        }

    def test_bbox_from_dict(self):
        """Test creating bbox from dictionary."""
        bbox_dict = {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 150}
        bbox = BoundingBox.from_dict(bbox_dict)

        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 150

    def test_bbox_overlap_iou(self):
        """Test IoU calculation."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        bbox2 = BoundingBox(x1=50, y1=50, x2=150, y2=150)

        # Overlapping boxes
        iou = bbox1.overlap_iou(bbox2)
        assert 0.14 < iou < 0.15  # Should be ~0.143

    def test_bbox_no_overlap(self):
        """Test IoU with no overlap."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        bbox2 = BoundingBox(x1=100, y1=100, x2=150, y2=150)

        iou = bbox1.overlap_iou(bbox2)
        assert iou == 0.0

    def test_bbox_complete_overlap(self):
        """Test IoU with complete overlap."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        bbox2 = BoundingBox(x1=0, y1=0, x2=100, y2=100)

        iou = bbox1.overlap_iou(bbox2)
        assert iou == 1.0


class TestFaceQuality:
    """Tests for FaceQuality dataclass."""

    def test_create_quality(self):
        """Test creating face quality."""
        quality = FaceQuality(
            sharpness=0.8,
            brightness=0.6,
            size=0.7,
            frontal=0.9
        )

        assert quality.sharpness == 0.8
        assert quality.brightness == 0.6
        assert quality.size == 0.7
        assert quality.frontal == 0.9

    def test_auto_compute_overall(self):
        """Test automatic overall quality computation."""
        quality = FaceQuality(
            sharpness=1.0,
            brightness=1.0,
            size=1.0,
            frontal=1.0
        )

        # Should be weighted average
        expected = 1.0 * 0.3 + 1.0 * 0.2 + 1.0 * 0.2 + 1.0 * 0.3
        assert quality.overall == expected


class TestLandmarks:
    """Tests for Landmarks dataclass."""

    def test_create_landmarks(self):
        """Test creating landmarks."""
        landmarks = Landmarks(
            left_eye=(10, 20),
            right_eye=(30, 20),
            nose=(20, 35),
            left_mouth=(15, 50),
            right_mouth=(25, 50)
        )

        assert landmarks.left_eye == (10, 20)
        assert landmarks.right_eye == (30, 20)

    def test_landmarks_from_array(self):
        """Test creating landmarks from array."""
        arr = np.array([
            [10, 20],  # left_eye
            [30, 20],  # right_eye
            [20, 35],  # nose
            [15, 50],  # left_mouth
            [25, 50]   # right_mouth
        ], dtype=np.float32)

        landmarks = Landmarks.from_array(arr)

        assert landmarks.left_eye == (10, 20)
        assert landmarks.right_eye == (30, 20)
        assert landmarks.nose == (20, 35)

    def test_landmarks_to_array(self):
        """Test converting landmarks to array."""
        landmarks = Landmarks(
            left_eye=(10, 20),
            right_eye=(30, 20),
            nose=(20, 35),
            left_mouth=(15, 50),
            right_mouth=(25, 50)
        )

        arr = landmarks.to_array()

        assert arr.shape == (5, 2)
        assert np.array_equal(arr[0], [10, 20])
        assert np.array_equal(arr[1], [30, 20])

    def test_landmarks_invalid_array(self):
        """Test error on invalid array shape."""
        arr = np.array([[10, 20], [30, 20]])  # Wrong shape

        with pytest.raises(ValueError, match="Expected landmarks shape"):
            Landmarks.from_array(arr)


class TestFaceDetection:
    """Tests for FaceDetection dataclass."""

    def test_create_face_detection(self):
        """Test creating face detection."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512).astype(np.float32)

        face = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding
        )

        assert face.bbox == bbox
        assert face.confidence == 0.95
        assert np.array_equal(face.embedding, embedding)

    def test_face_detection_validation(self):
        """Test validation of face detection."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)

        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between"):
            FaceDetection(bbox=bbox, confidence=1.5)

        # Invalid embedding type
        with pytest.raises(TypeError, match="Embedding must be numpy array"):
            FaceDetection(bbox=bbox, confidence=0.9, embedding=[1, 2, 3])

        # Invalid embedding dimension
        embedding_2d = np.random.randn(512, 2).astype(np.float32)
        with pytest.raises(ValueError, match="Embedding must be 1-dimensional"):
            FaceDetection(bbox=bbox, confidence=0.9, embedding=embedding_2d)

    def test_is_indexed(self):
        """Test is_indexed property."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        face = FaceDetection(bbox=bbox, confidence=0.95)

        assert not face.is_indexed

        face.face_id = 1
        face.embedding_index = 0

        assert face.is_indexed

    def test_face_to_dict(self):
        """Test converting face to dictionary."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512).astype(np.float32)
        quality = FaceQuality(sharpness=0.8, brightness=0.6, size=0.7, frontal=0.9)

        face = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            quality=quality,
            image_path=Path('/path/to/image.jpg'),
            face_id=1,
            embedding_index=0
        )

        face_dict = face.to_dict()

        assert face_dict['confidence'] == 0.95
        assert face_dict['has_embedding'] is True
        assert face_dict['embedding_dim'] == 512
        assert face_dict['face_id'] == 1
        assert face_dict['embedding_index'] == 0
        assert face_dict['is_indexed'] is True
        assert face_dict['quality']['overall'] == quality.overall


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating search result."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        face = FaceDetection(bbox=bbox, confidence=0.95)

        result = SearchResult(
            face=face,
            distance=0.5,
            rank=1
        )

        assert result.face == face
        assert result.distance == 0.5
        assert result.rank == 1

    def test_similarity_computation(self):
        """Test automatic similarity computation."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        face = FaceDetection(bbox=bbox, confidence=0.95)

        result = SearchResult(face=face, distance=0.5)

        # similarity = 1 / (1 + distance)
        expected_similarity = 1.0 / (1.0 + 0.5)
        assert abs(result.similarity - expected_similarity) < 1e-6

    def test_search_result_to_dict(self):
        """Test converting search result to dict."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        face = FaceDetection(bbox=bbox, confidence=0.95)

        result = SearchResult(face=face, distance=0.5, rank=2)
        result_dict = result.to_dict()

        assert result_dict['rank'] == 2
        assert result_dict['distance'] == 0.5
        assert 'similarity' in result_dict
        assert 'face' in result_dict


class TestIndexingResult:
    """Tests for IndexingResult dataclass."""

    def test_create_indexing_result(self):
        """Test creating indexing result."""
        result = IndexingResult(
            image_path=Path('/path/to/image.jpg'),
            faces_found=3,
            faces_indexed=2,
            face_ids=[1, 2]
        )

        assert result.faces_found == 3
        assert result.faces_indexed == 2
        assert len(result.face_ids) == 2

    def test_success_property(self):
        """Test success property."""
        # Successful indexing
        result = IndexingResult(
            image_path=Path('/path/to/image.jpg'),
            faces_found=2,
            faces_indexed=2,
            face_ids=[1, 2]
        )
        assert result.success is True

        # Failed indexing (no faces indexed)
        result = IndexingResult(
            image_path=Path('/path/to/image.jpg'),
            faces_found=0,
            faces_indexed=0
        )
        assert result.success is False

        # Partial success (has errors)
        result = IndexingResult(
            image_path=Path('/path/to/image.jpg'),
            faces_found=2,
            faces_indexed=1,
            errors=['One face failed']
        )
        assert result.success is False

    def test_partial_success_property(self):
        """Test partial_success property."""
        result = IndexingResult(
            image_path=Path('/path/to/image.jpg'),
            faces_found=2,
            faces_indexed=1,
            errors=['One face failed']
        )
        assert result.partial_success is True


class TestBatchIndexingStats:
    """Tests for BatchIndexingStats dataclass."""

    def test_create_stats(self):
        """Test creating batch stats."""
        stats = BatchIndexingStats(
            total_images=100,
            successful_images=95,
            failed_images=5,
            total_faces_found=250,
            total_faces_indexed=240,
            processing_time=60.0
        )

        assert stats.total_images == 100
        assert stats.successful_images == 95
        assert stats.total_faces_indexed == 240

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = BatchIndexingStats(
            total_images=100,
            successful_images=95
        )

        assert stats.success_rate == 0.95

    def test_faces_per_image(self):
        """Test faces per image calculation."""
        stats = BatchIndexingStats(
            total_images=100,
            total_faces_found=250
        )

        assert stats.faces_per_image == 2.5

    def test_processing_speed(self):
        """Test processing speed calculations."""
        stats = BatchIndexingStats(
            total_images=100,
            total_faces_indexed=250,
            processing_time=50.0
        )

        assert stats.images_per_second == 2.0
        assert stats.faces_per_second == 5.0

    def test_stats_to_dict(self):
        """Test converting stats to dict."""
        stats = BatchIndexingStats(
            total_images=100,
            successful_images=95,
            total_faces_indexed=250,
            processing_time=60.0
        )

        stats_dict = stats.to_dict()

        assert stats_dict['total_images'] == 100
        assert stats_dict['successful_images'] == 95
        assert 'success_rate' in stats_dict
        assert 'faces_per_second' in stats_dict
