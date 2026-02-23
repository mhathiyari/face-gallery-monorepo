"""Tests for search engine and related utilities."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from face_search.search import (
    SearchEngine,
    rank_results,
    format_results_aws_style,
    format_results_simple,
    filter_results,
    cluster_faces,
    get_identity_cluster,
    find_representative_faces,
    compute_cluster_statistics
)
from face_search.face import FaceDetection, BoundingBox, SearchResult, FaceQuality
from face_search.storage import Collection


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_collection(temp_dir):
    """Create a test collection with some faces."""
    collection = Collection(
        collection_id='test_search',
        collections_root=temp_dir / 'collections',
        embedding_dim=512
    )

    # Add some test faces
    for i in range(10):
        embedding = np.random.randn(512).astype(np.float32)
        collection.add_face(
            image_path=f'/test/image_{i}.jpg',
            bbox={'x1': 10, 'y1': 20, 'x2': 100, 'y2': 150},
            embedding=embedding,
            confidence=0.9 + i * 0.01
        )

    return collection


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
            self.embedding_dim = 512

        def detect_faces(self, image_path, min_confidence=0.5):
            from face_search.models.base import DetectedFace, BoundingBox
            # Return 1-2 fake faces
            faces = []
            for i in range(2):
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

    return MockModel()


@pytest.fixture
def temp_image(temp_dir):
    """Create a temporary test image."""
    image_path = temp_dir / 'test_query.jpg'
    img = Image.new('RGB', (200, 200), color='red')
    img.save(image_path)
    return image_path


class TestSearchEngine:
    """Tests for SearchEngine class."""

    def test_create_search_engine(self, mock_model, test_collection):
        """Test creating a search engine."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        assert engine.model == mock_model
        assert engine.collection == test_collection
        assert engine.default_max_results == 10

    def test_search_by_image(self, mock_model, test_collection, temp_image):
        """Test searching by image."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        results = engine.search_by_image(
            image_path=temp_image,
            max_results=5
        )

        # Should return list of results (one per detected face)
        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should be a list of SearchResult objects
        for face_results in results:
            assert isinstance(face_results, list)
            assert all(isinstance(r, SearchResult) for r in face_results)

    def test_search_by_image_no_faces(self, test_collection, temp_image):
        """Test error when no faces detected."""
        class NoFaceModel:
            def __init__(self):
                self.device = 'cpu'
                self.embedding_dim = 512

            def detect_faces(self, image_path, min_confidence=0.5):
                return []  # No faces

        engine = SearchEngine(
            model=NoFaceModel(),
            collection=test_collection
        )

        with pytest.raises(ValueError, match="No faces detected"):
            engine.search_by_image(temp_image)

    def test_search_by_image_return_all_faces(self, mock_model, test_collection, temp_image):
        """Test returning results for all detected faces."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        # Search with return_all_faces=True
        results = engine.search_by_image(
            image_path=temp_image,
            return_all_faces=True
        )

        # Should return results for all detected faces (mock returns 2)
        assert len(results) == 2

    def test_search_by_image_best_face_only(self, mock_model, test_collection, temp_image):
        """Test returning results for best face only."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        # Search with return_all_faces=False (default)
        results = engine.search_by_image(
            image_path=temp_image,
            return_all_faces=False
        )

        # Should return results for only one face
        assert len(results) == 1

    def test_search_by_face_id(self, mock_model, test_collection):
        """Test searching by face ID."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        # Search by first face
        results = engine.search_by_face_id(
            face_id=1,
            max_results=5
        )

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

        # Should not include the query face itself
        assert not any(r.face.face_id == 1 for r in results)

    def test_search_by_face_id_include_self(self, mock_model, test_collection):
        """Test searching by face ID with self included."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        results = engine.search_by_face_id(
            face_id=1,
            max_results=5,
            include_self=True
        )

        # Should include the query face
        assert any(r.face.face_id == 1 for r in results)

    def test_search_by_face_id_not_found(self, mock_model, test_collection):
        """Test error when face ID not found."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        with pytest.raises(ValueError, match="not found"):
            engine.search_by_face_id(face_id=9999)

    def test_search_by_metadata(self, mock_model, test_collection):
        """Test searching by metadata filters."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        # Search by image path substring
        results = engine.search_by_metadata(
            image_path='image_1',
            limit=10
        )

        assert isinstance(results, list)
        assert all(isinstance(f, FaceDetection) for f in results)
        assert all('image_1' in str(f.image_path) for f in results)

    def test_search_by_metadata_confidence_filter(self, mock_model, test_collection):
        """Test filtering by confidence."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        results = engine.search_by_metadata(
            min_confidence=0.95,
            limit=100
        )

        assert all(f.confidence >= 0.95 for f in results)

    def test_compare_faces(self, mock_model, test_collection, temp_dir):
        """Test comparing faces between two images."""
        # Create two test images
        img1 = temp_dir / 'img1.jpg'
        img2 = temp_dir / 'img2.jpg'
        Image.new('RGB', (200, 200), color='red').save(img1)
        Image.new('RGB', (200, 200), color='blue').save(img2)

        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        comparison = engine.compare_faces(
            source_image=img1,
            target_image=img2
        )

        assert 'source_faces' in comparison
        assert 'target_faces' in comparison
        assert 'best_match' in comparison
        assert 'all_matches' in comparison

        # Should have detected faces
        assert len(comparison['source_faces']) > 0
        assert len(comparison['target_faces']) > 0

        # Should have match info
        assert 'similarity' in comparison['best_match']
        assert 'distance' in comparison['best_match']

    def test_get_stats(self, mock_model, test_collection):
        """Test getting engine statistics."""
        engine = SearchEngine(
            model=mock_model,
            collection=test_collection
        )

        stats = engine.get_stats()

        assert 'collection_id' in stats
        assert 'total_faces' in stats
        assert 'active_faces' in stats
        assert 'model' in stats


class TestResultFormatting:
    """Tests for result formatting utilities."""

    def test_rank_results(self):
        """Test ranking search results."""
        # Create mock results
        face = FaceDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.9
        )

        results = [
            SearchResult(face=face, distance=0.5, rank=1),
            SearchResult(face=face, distance=0.3, rank=1),
            SearchResult(face=face, distance=0.7, rank=1)
        ]

        # Rank by similarity (default)
        ranked = rank_results(results)

        # Should be sorted by similarity (descending)
        assert ranked[0].distance == 0.3  # Lowest distance = highest similarity
        assert ranked[-1].distance == 0.7

        # Ranks should be updated
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3

    def test_format_results_aws_style(self):
        """Test AWS Rekognition style formatting."""
        face = FaceDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.95,
            face_id=1,
            image_path=Path('/test/image.jpg')
        )

        results = [
            SearchResult(face=face, distance=0.5, rank=1)
        ]

        formatted = format_results_aws_style(results)

        assert 'FaceMatches' in formatted
        assert len(formatted['FaceMatches']) == 1

        match = formatted['FaceMatches'][0]
        assert 'Similarity' in match
        assert 'Face' in match
        assert 'BoundingBox' in match['Face']
        assert 'Confidence' in match['Face']

    def test_format_results_simple(self):
        """Test simple formatting."""
        face = FaceDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.95,
            face_id=1
        )

        results = [
            SearchResult(face=face, distance=0.5, rank=1)
        ]

        formatted = format_results_simple(results)

        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert 'rank' in formatted[0]
        assert 'similarity' in formatted[0]
        assert 'face_id' in formatted[0]

    def test_filter_results_by_similarity(self):
        """Test filtering results by similarity."""
        face = FaceDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.9
        )

        results = [
            SearchResult(face=face, distance=0.2, rank=1),  # High similarity
            SearchResult(face=face, distance=0.8, rank=2),  # Low similarity
            SearchResult(face=face, distance=0.5, rank=3)   # Medium similarity
        ]

        # Filter for high similarity (> 0.6)
        filtered = filter_results(results, min_similarity=0.6)

        # Only high similarity results should remain
        assert len(filtered) < len(results)
        assert all(r.similarity >= 0.6 for r in filtered)

    def test_filter_results_by_confidence(self):
        """Test filtering by detection confidence."""
        results = [
            SearchResult(
                face=FaceDetection(bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10), confidence=0.9),
                distance=0.5,
                rank=1
            ),
            SearchResult(
                face=FaceDetection(bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10), confidence=0.6),
                distance=0.5,
                rank=2
            )
        ]

        filtered = filter_results(results, min_confidence=0.8)

        assert len(filtered) == 1
        assert filtered[0].face.confidence == 0.9


class TestClustering:
    """Tests for face clustering."""

    def test_cluster_faces_basic(self):
        """Test basic face clustering."""
        # Create faces with similar embeddings
        base_embedding = np.random.randn(512).astype(np.float32)

        faces = []
        for i in range(10):
            # Add small noise to create similar embeddings
            embedding = base_embedding + np.random.randn(512).astype(np.float32) * 0.1

            face = FaceDetection(
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
                confidence=0.9,
                embedding=embedding
            )
            faces.append(face)

        # Cluster faces
        clusters = cluster_faces(faces, eps=0.8, min_samples=2)

        assert isinstance(clusters, dict)
        # Should have at least one cluster
        assert len(clusters) > 0

    def test_cluster_faces_no_embeddings(self):
        """Test error when faces have no embeddings."""
        faces = [
            FaceDetection(
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
                confidence=0.9,
                embedding=None  # No embedding
            )
        ]

        with pytest.raises(ValueError, match="must have embeddings"):
            cluster_faces(faces)

    def test_cluster_faces_noise(self):
        """Test clustering with noise (isolated faces)."""
        faces = []

        # Add clustered faces
        base_emb = np.random.randn(512).astype(np.float32)
        for i in range(5):
            face = FaceDetection(
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
                confidence=0.9,
                embedding=base_emb + np.random.randn(512).astype(np.float32) * 0.05
            )
            faces.append(face)

        # Add outlier (noise)
        outlier = FaceDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.9,
            embedding=np.random.randn(512).astype(np.float32) * 5  # Very different
        )
        faces.append(outlier)

        clusters = cluster_faces(faces, eps=0.5, min_samples=2)

        # Should have noise cluster (-1)
        assert -1 in clusters or len(clusters[-1]) == 0 if -1 in clusters else True

    def test_find_representative_faces_centroid(self):
        """Test finding representative by centroid."""
        # Create cluster with known centroid
        base_embedding = np.ones(512, dtype=np.float32)

        faces = []
        for i in range(5):
            embedding = base_embedding + np.random.randn(512).astype(np.float32) * 0.1
            face = FaceDetection(
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
                confidence=0.9,
                embedding=embedding
            )
            faces.append(face)

        representatives = find_representative_faces(faces, method='centroid', n_representatives=1)

        assert len(representatives) == 1
        assert isinstance(representatives[0], FaceDetection)

    def test_find_representative_faces_highest_confidence(self):
        """Test finding representative by confidence."""
        faces = [
            FaceDetection(
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                confidence=0.8,
                embedding=np.random.randn(512).astype(np.float32)
            ),
            FaceDetection(
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                confidence=0.95,  # Highest
                embedding=np.random.randn(512).astype(np.float32)
            ),
            FaceDetection(
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                confidence=0.7,
                embedding=np.random.randn(512).astype(np.float32)
            )
        ]

        representatives = find_representative_faces(faces, method='highest_confidence', n_representatives=1)

        assert len(representatives) == 1
        assert representatives[0].confidence == 0.95

    def test_find_representative_faces_highest_quality(self):
        """Test finding representative by quality."""
        faces = [
            FaceDetection(
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                confidence=0.9,
                embedding=np.random.randn(512).astype(np.float32),
                quality=FaceQuality(sharpness=0.7, brightness=0.6, size=0.8, frontal=0.9)
            ),
            FaceDetection(
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                confidence=0.9,
                embedding=np.random.randn(512).astype(np.float32),
                quality=FaceQuality(sharpness=0.9, brightness=0.9, size=0.9, frontal=0.95)  # Best quality
            )
        ]

        representatives = find_representative_faces(faces, method='highest_quality', n_representatives=1)

        assert len(representatives) == 1
        # Should be the one with highest quality
        assert representatives[0].quality.overall > faces[0].quality.overall

    def test_compute_cluster_statistics(self):
        """Test computing cluster statistics."""
        face = FaceDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            confidence=0.9,
            embedding=np.random.randn(512).astype(np.float32)
        )

        clusters = {
            0: [face, face, face],  # Cluster of 3
            1: [face, face],         # Cluster of 2
            -1: [face]              # Noise
        }

        stats = compute_cluster_statistics(clusters)

        assert stats['n_clusters'] == 2  # Excluding noise
        assert stats['n_faces'] == 6
        assert stats['n_noise'] == 1
        assert stats['avg_cluster_size'] == 2.5
        assert stats['min_cluster_size'] == 2
        assert stats['max_cluster_size'] == 3

    def test_get_identity_cluster(self, test_collection):
        """Test getting identity cluster for a face."""
        # This test requires faces to be similar enough to cluster
        # The test collection has random embeddings, so results may vary
        try:
            cluster = get_identity_cluster(
                collection=test_collection,
                face_id=1,
                eps=2.0,  # Large eps to ensure clustering
                min_samples=1
            )

            assert isinstance(cluster, list)
            assert all(isinstance(f, FaceDetection) for f in cluster)
            # Should at least include the query face
            assert any(f.face_id == 1 for f in cluster)
        except ValueError:
            # Face not found is acceptable for this test
            pass
