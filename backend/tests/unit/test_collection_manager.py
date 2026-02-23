"""Unit tests for collection manager.

Tests cover:
- Collection creation and loading
- Adding faces (single and batch)
- Searching faces
- Synchronization verification
- Index rebuilding
- Locking mechanism
"""

import pytest
import numpy as np
from pathlib import Path
import time
import threading

from face_search.storage import (
    Collection,
    SyncError,
    CollectionLock,
    collection_lock,
    LockTimeout
)
from face_search.models import ModelVersion


# Test fixtures

@pytest.fixture
def collections_root(tmp_path):
    """Create temporary collections root directory."""
    return tmp_path / "collections"


@pytest.fixture
def sample_embedding():
    """Create a sample embedding."""
    np.random.seed(42)
    return np.random.randn(512).astype(np.float32)


@pytest.fixture
def sample_embeddings():
    """Create multiple sample embeddings."""
    np.random.seed(42)
    return np.random.randn(10, 512).astype(np.float32)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return {'x1': 10.0, 'y1': 20.0, 'x2': 100.0, 'y2': 150.0}


@pytest.fixture
def collection(collections_root):
    """Create a test collection."""
    return Collection(
        collection_id="test_collection",
        collections_root=collections_root,
        embedding_dim=512,
        device='cpu'
    )


# Tests for Collection Lock

class TestCollectionLock:
    """Tests for collection locking mechanism."""

    def test_lock_creation(self, tmp_path):
        """Test lock file creation."""
        lock_path = tmp_path / "test.lock"
        lock = CollectionLock(lock_path, timeout=5.0)

        assert not lock.is_locked()

        lock.acquire()
        assert lock.is_locked()
        assert lock_path.exists()

        lock.release()
        assert not lock.is_locked()
        assert not lock_path.exists()

    def test_lock_context_manager(self, tmp_path):
        """Test lock as context manager."""
        lock_path = tmp_path / "test.lock"

        with CollectionLock(lock_path):
            assert lock_path.exists()

        assert not lock_path.exists()

    def test_lock_timeout(self, tmp_path):
        """Test lock timeout."""
        lock_path = tmp_path / "test.lock"
        lock1 = CollectionLock(lock_path, timeout=0.5)

        # Acquire first lock
        lock1.acquire()

        # Try to acquire second lock (should timeout)
        lock2 = CollectionLock(lock_path, timeout=0.5)
        with pytest.raises(LockTimeout):
            lock2.acquire()

        lock1.release()

    def test_collection_lock_helper(self, tmp_path):
        """Test collection_lock helper function."""
        collection_path = tmp_path / "collection"
        collection_path.mkdir()

        with collection_lock(collection_path):
            lock_file = collection_path / ".lock"
            assert lock_file.exists()

        assert not lock_file.exists()


# Tests for Collection Creation

class TestCollectionCreation:
    """Tests for collection creation."""

    def test_create_new_collection(self, collections_root):
        """Test creating a new collection."""
        col = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )

        assert col.collection_id == "test"
        assert col.embedding_dim == 512
        assert col.ntotal == 0
        assert col.collection_path.exists()
        assert col.config_path.exists()

    def test_collection_directory_structure(self, collections_root):
        """Test collection directory structure is created."""
        col = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )

        assert col.collection_path.exists()
        assert col.config_path.exists()
        assert col.db_path.exists()
        # index.faiss doesn't exist until first save

    def test_create_with_model_version(self, collections_root):
        """Test creating collection with model version."""
        model_version = ModelVersion(
            model_type="InsightFace",
            model_name="buffalo_l",
            embedding_dim=512
        )

        col = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512,
            model_version=model_version
        )

        assert col.model_version == model_version
        assert col.config['model_version'] is not None


# Tests for Collection Loading

class TestCollectionLoading:
    """Tests for loading existing collections."""

    def test_load_existing_collection(self, collections_root, sample_embedding, sample_bbox):
        """Test loading an existing collection."""
        # Create collection and add face
        col1 = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )
        col1.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            confidence=0.95
        )
        col1.save()

        # Load collection
        col2 = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )

        assert col2.ntotal == 1
        assert col2.index.ntotal == 1

    def test_load_wrong_dimension_fails(self, collections_root):
        """Test loading collection with wrong dimension fails."""
        # Create with 512 dim
        col1 = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )
        col1.save()

        # Try to load with 256 dim
        with pytest.raises(ValueError, match="dimension mismatch"):
            Collection(
                collection_id="test",
                collections_root=collections_root,
                embedding_dim=256
            )


# Tests for Adding Faces

class TestAddFaces:
    """Tests for adding faces to collection."""

    def test_add_single_face(self, collection, sample_embedding, sample_bbox):
        """Test adding a single face."""
        face_id = collection.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            confidence=0.95
        )

        assert isinstance(face_id, int)
        assert collection.ntotal == 1
        assert collection.index.ntotal == 1

    def test_add_multiple_faces(self, collection, sample_bbox):
        """Test adding multiple faces."""
        for i in range(5):
            embedding = np.random.randn(512).astype(np.float32)
            collection.add_face(
                image_path=f"/path/to/image{i}.jpg",
                bbox=sample_bbox,
                embedding=embedding,
                confidence=0.9 + i * 0.01
            )

        assert collection.ntotal == 5
        assert collection.index.ntotal == 5

    def test_add_wrong_dimension_fails(self, collection, sample_bbox):
        """Test adding face with wrong dimension fails."""
        wrong_embedding = np.random.randn(256).astype(np.float32)

        with pytest.raises(ValueError, match="dimension"):
            collection.add_face(
                image_path="/path/to/image.jpg",
                bbox=sample_bbox,
                embedding=wrong_embedding,
                confidence=0.95
            )

    def test_add_faces_batch(self, collection, sample_embeddings, sample_bbox):
        """Test adding faces in batch."""
        faces = []
        for i in range(10):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.9 + i * 0.01
            })

        face_ids = collection.add_faces_batch(faces, sample_embeddings)

        assert len(face_ids) == 10
        assert collection.ntotal == 10
        assert collection.index.ntotal == 10

    def test_add_batch_mismatched_length_fails(self, collection, sample_embeddings, sample_bbox):
        """Test adding batch with mismatched lengths fails."""
        faces = [
            {'image_path': "/path/to/image.jpg", 'bbox': sample_bbox, 'confidence': 0.95}
        ]

        with pytest.raises(ValueError, match="doesn't match"):
            collection.add_faces_batch(faces, sample_embeddings)  # 1 face, 10 embeddings


# Tests for Searching

class TestSearch:
    """Tests for searching faces."""

    def test_search_similar_faces(self, collection, sample_embeddings, sample_bbox):
        """Test searching for similar faces."""
        # Add faces
        faces = []
        for i in range(10):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.95
            })
        collection.add_faces_batch(faces, sample_embeddings)

        # Search with first embedding
        query = sample_embeddings[0]
        results = collection.search(query, k=5)

        assert len(results) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # First result should be the query itself (distance ~0)
        assert results[0][1] < 1e-5

    def test_search_with_threshold(self, collection, sample_embeddings, sample_bbox):
        """Test searching with distance threshold."""
        # Add faces
        faces = []
        for i in range(10):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.95
            })
        collection.add_faces_batch(faces, sample_embeddings)

        # Search with very low threshold
        query = sample_embeddings[0]
        results = collection.search(query, k=10, threshold=0.01)

        # Should only return very similar faces
        assert len(results) <= 10
        assert all(dist < 0.01 for _, dist in results)

    def test_search_by_face_id(self, collection, sample_embeddings, sample_bbox):
        """Test searching by face ID."""
        # Add faces
        faces = []
        face_ids = []
        for i in range(10):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.95
            })
        face_ids = collection.add_faces_batch(faces, sample_embeddings)

        # Search by first face
        results = collection.search_by_face_id(face_ids[0], k=5)

        assert len(results) <= 5
        # Results should not include the query face itself
        assert all(face.face_id != face_ids[0] for face, _ in results)

    def test_search_nonexistent_face_fails(self, collection):
        """Test searching by non-existent face ID fails."""
        with pytest.raises(ValueError, match="not found"):
            collection.search_by_face_id(9999, k=5)

    def test_search_wrong_dimension_fails(self, collection, sample_bbox):
        """Test searching with wrong dimension fails."""
        # Add a face first
        embedding = np.random.randn(512).astype(np.float32)
        collection.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=embedding,
            confidence=0.95
        )

        # Try to search with wrong dimension
        wrong_query = np.random.randn(256).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            collection.search(wrong_query, k=5)


# Tests for Deletion

class TestDeletion:
    """Tests for deleting faces."""

    def test_delete_face(self, collection, sample_embedding, sample_bbox):
        """Test deleting a face."""
        # Add face
        face_id = collection.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            confidence=0.95
        )

        assert collection.ntotal == 1

        # Delete face
        result = collection.delete_face(face_id)
        assert result is True
        assert collection.ntotal == 0

    def test_deleted_face_not_in_search(self, collection, sample_embeddings, sample_bbox):
        """Test deleted faces don't appear in search results."""
        # Add faces
        faces = []
        for i in range(5):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.95
            })
        face_ids = collection.add_faces_batch(faces, sample_embeddings[:5])

        # Delete one face
        collection.delete_face(face_ids[2])

        # Search should not return deleted face
        query = sample_embeddings[2]
        results = collection.search(query, k=10)

        # Should have 4 results (not including deleted face)
        assert len(results) == 4
        assert all(face.face_id != face_ids[2] for face, _ in results)


# Tests for Index Rebuild

class TestIndexRebuild:
    """Tests for index rebuilding."""

    def test_rebuild_index(self, collection, sample_embeddings, sample_bbox):
        """Test rebuilding index after deletions."""
        # Add faces
        faces = []
        for i in range(10):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.95
            })
        face_ids = collection.add_faces_batch(faces, sample_embeddings)

        # Delete some faces
        collection.delete_face(face_ids[2])
        collection.delete_face(face_ids[5])
        collection.delete_face(face_ids[7])

        # Before rebuild: index has 10, active has 7
        assert collection.index.ntotal == 10
        assert collection.ntotal == 7

        # Rebuild
        collection.rebuild_index()

        # After rebuild: both should have 7
        assert collection.index.ntotal == 7
        assert collection.ntotal == 7

    def test_rebuild_empty_collection(self, collection):
        """Test rebuilding empty collection."""
        collection.rebuild_index()
        assert collection.index.ntotal == 0
        assert collection.ntotal == 0


# Tests for Synchronization

class TestSynchronization:
    """Tests for synchronization between index and database."""

    def test_verify_sync_on_load(self, collections_root, sample_embedding, sample_bbox):
        """Test synchronization is verified on load."""
        # Create collection and add face
        col1 = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )
        col1.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            confidence=0.95
        )
        col1.save()

        # Manually corrupt by deleting from database
        col1.metadata.mark_deleted(1)

        # Try to load (should detect sync error)
        with pytest.raises(SyncError):
            Collection(
                collection_id="test",
                collections_root=collections_root,
                embedding_dim=512
            )


# Tests for Save/Load

class TestSaveLoad:
    """Tests for saving and loading collections."""

    def test_save_and_load(self, collections_root, sample_embeddings, sample_bbox):
        """Test saving and loading collection."""
        # Create and populate collection
        col1 = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )

        faces = []
        for i in range(10):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.95
            })
        col1.add_faces_batch(faces, sample_embeddings)
        col1.save()

        # Load collection
        col2 = Collection(
            collection_id="test",
            collections_root=collections_root,
            embedding_dim=512
        )

        assert col2.ntotal == 10
        assert col2.index.ntotal == 10

        # Search should work
        query = sample_embeddings[0]
        results = col2.search(query, k=5)
        assert len(results) == 5


# Tests for Statistics

class TestStatistics:
    """Tests for collection statistics."""

    def test_get_stats(self, collection, sample_embeddings, sample_bbox):
        """Test getting collection statistics."""
        # Add faces
        faces = []
        for i in range(10):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.95
            })
        face_ids = collection.add_faces_batch(faces, sample_embeddings)

        # Delete some
        collection.delete_face(face_ids[0])
        collection.delete_face(face_ids[5])

        stats = collection.get_stats()
        assert stats['collection_id'] == 'test_collection'
        assert stats['total_faces'] == 10
        assert stats['active_faces'] == 8
        assert stats['deleted_faces'] == 2
        assert stats['embedding_dim'] == 512

    def test_repr(self, collection):
        """Test string representation."""
        repr_str = repr(collection)
        assert 'Collection' in repr_str
        assert 'test_collection' in repr_str

    def test_len(self, collection, sample_embedding, sample_bbox):
        """Test __len__ method."""
        assert len(collection) == 0

        collection.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            confidence=0.95
        )

        assert len(collection) == 1


# Integration Tests

class TestIntegration:
    """Integration tests for collection manager."""

    def test_full_workflow(self, collections_root, sample_bbox):
        """Test complete workflow."""
        # Create collection
        col = Collection(
            collection_id="workflow_test",
            collections_root=collections_root,
            embedding_dim=512
        )

        # Add faces
        np.random.seed(42)
        embeddings = np.random.randn(20, 512).astype(np.float32)
        faces = []
        for i in range(20):
            faces.append({
                'image_path': f"/path/to/image{i}.jpg",
                'bbox': sample_bbox,
                'confidence': 0.9 + (i % 10) * 0.01
            })
        face_ids = col.add_faces_batch(faces, embeddings)

        # Search
        query = embeddings[0]
        results = col.search(query, k=10)
        assert len(results) == 10

        # Delete some faces
        for i in [3, 7, 11, 15]:
            col.delete_face(face_ids[i])

        # Search again (deleted faces won't appear in results)
        results2 = col.search(query, k=10)
        assert len(results2) <= 10  # May be fewer if deleted faces were nearest neighbors
        assert col.ntotal == 16

        # Rebuild index
        col.rebuild_index()
        assert col.index.ntotal == 16

        # Save
        col.save()

        # Load in new instance
        col2 = Collection(
            collection_id="workflow_test",
            collections_root=collections_root,
            embedding_dim=512
        )
        assert col2.ntotal == 16

        # Search in loaded collection
        results3 = col2.search(query, k=10)
        assert len(results3) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
