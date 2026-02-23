"""Unit tests for metadata database.

Tests cover:
- Database creation
- CRUD operations
- Soft delete functionality
- Index synchronization
- Transaction handling
- Query performance
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from face_search.storage import MetadataStore, Face


# Test fixtures

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_metadata.db"
    return str(db_path)


@pytest.fixture
def metadata_store(temp_db):
    """Create a MetadataStore instance."""
    return MetadataStore(db_path=temp_db, collection_id="test_collection")


@pytest.fixture
def sample_embedding():
    """Create a sample embedding."""
    np.random.seed(42)
    return np.random.randn(512).astype(np.float32)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return {'x1': 10.0, 'y1': 20.0, 'x2': 100.0, 'y2': 150.0}


# Tests for Database Creation

class TestDatabaseCreation:
    """Tests for database creation and initialization."""

    def test_database_creation(self, temp_db):
        """Test database file is created."""
        store = MetadataStore(db_path=temp_db, collection_id="test")
        assert Path(temp_db).exists()

    def test_database_directory_creation(self, tmp_path):
        """Test database directory is created if missing."""
        db_path = tmp_path / "subdir" / "nested" / "metadata.db"
        store = MetadataStore(db_path=str(db_path), collection_id="test")
        assert db_path.exists()
        assert db_path.parent.exists()

    def test_multiple_stores_same_db(self, temp_db):
        """Test multiple stores can access same database."""
        store1 = MetadataStore(db_path=temp_db, collection_id="collection1")
        store2 = MetadataStore(db_path=temp_db, collection_id="collection2")

        assert store1.db_path == store2.db_path
        assert store1.collection_id != store2.collection_id


# Tests for Adding Faces

class TestAddFace:
    """Tests for adding face records."""

    def test_add_single_face(self, metadata_store, sample_embedding, sample_bbox):
        """Test adding a single face."""
        face_id = metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        assert isinstance(face_id, int)
        assert face_id > 0

    def test_add_multiple_faces(self, metadata_store, sample_embedding, sample_bbox):
        """Test adding multiple faces."""
        face_ids = []
        for i in range(5):
            face_id = metadata_store.add_face(
                image_path=f"/path/to/image{i}.jpg",
                bbox=sample_bbox,
                embedding=sample_embedding,
                embedding_index=i,
                confidence=0.9 + i * 0.01
            )
            face_ids.append(face_id)

        assert len(face_ids) == 5
        assert len(set(face_ids)) == 5  # All unique

    def test_add_face_increments_id(self, metadata_store, sample_embedding, sample_bbox):
        """Test face IDs increment correctly."""
        id1 = metadata_store.add_face(
            image_path="/path/to/image1.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )
        id2 = metadata_store.add_face(
            image_path="/path/to/image2.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=1,
            confidence=0.96
        )

        assert id2 > id1

    def test_embedding_serialization(self, metadata_store, sample_bbox):
        """Test embedding is correctly serialized and deserialized."""
        original_embedding = np.random.randn(512).astype(np.float32)

        face_id = metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=original_embedding,
            embedding_index=0,
            confidence=0.95
        )

        face = metadata_store.get_face_by_id(face_id)
        retrieved_embedding = face.get_embedding()

        np.testing.assert_array_equal(original_embedding, retrieved_embedding)


# Tests for Retrieving Faces

class TestGetFace:
    """Tests for retrieving face records."""

    def test_get_face_by_id(self, metadata_store, sample_embedding, sample_bbox):
        """Test getting face by ID."""
        face_id = metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        face = metadata_store.get_face_by_id(face_id)
        assert face is not None
        assert face.face_id == face_id
        assert face.image_path == "/path/to/image.jpg"
        assert face.confidence == 0.95

    def test_get_nonexistent_face(self, metadata_store):
        """Test getting non-existent face returns None."""
        face = metadata_store.get_face_by_id(9999)
        assert face is None

    def test_get_face_by_index(self, metadata_store, sample_embedding, sample_bbox):
        """Test getting face by embedding index."""
        face_id = metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=42,
            confidence=0.95
        )

        face = metadata_store.get_face_by_index(42)
        assert face is not None
        assert face.face_id == face_id
        assert face.embedding_index == 42

    def test_get_faces_by_image(self, metadata_store, sample_embedding, sample_bbox):
        """Test getting all faces from an image."""
        image_path = "/path/to/image.jpg"

        # Add multiple faces from same image
        for i in range(3):
            metadata_store.add_face(
                image_path=image_path,
                bbox=sample_bbox,
                embedding=sample_embedding,
                embedding_index=i,
                confidence=0.9 + i * 0.01
            )

        faces = metadata_store.get_faces_by_image(image_path)
        assert len(faces) == 3
        assert all(f.image_path == image_path for f in faces)

    def test_get_faces_by_image_different_images(self, metadata_store, sample_embedding, sample_bbox):
        """Test getting faces only returns from specific image."""
        metadata_store.add_face(
            image_path="/path/to/image1.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )
        metadata_store.add_face(
            image_path="/path/to/image2.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=1,
            confidence=0.96
        )

        faces = metadata_store.get_faces_by_image("/path/to/image1.jpg")
        assert len(faces) == 1
        assert faces[0].image_path == "/path/to/image1.jpg"


# Tests for Soft Delete

class TestSoftDelete:
    """Tests for soft delete functionality."""

    def test_mark_deleted(self, metadata_store, sample_embedding, sample_bbox):
        """Test marking face as deleted."""
        face_id = metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        # Mark as deleted
        result = metadata_store.mark_deleted(face_id)
        assert result is True

        # Verify deleted flag is set
        face = metadata_store.get_face_by_id(face_id)
        assert face.deleted is True

    def test_mark_nonexistent_face_deleted(self, metadata_store):
        """Test marking non-existent face returns False."""
        result = metadata_store.mark_deleted(9999)
        assert result is False

    def test_deleted_face_not_in_active_count(self, metadata_store, sample_embedding, sample_bbox):
        """Test deleted faces don't count as active."""
        # Add 3 faces
        for i in range(3):
            metadata_store.add_face(
                image_path=f"/path/to/image{i}.jpg",
                bbox=sample_bbox,
                embedding=sample_embedding,
                embedding_index=i,
                confidence=0.95
            )

        assert metadata_store.count_active_faces() == 3
        assert metadata_store.count_total_faces() == 3

        # Delete one
        metadata_store.mark_deleted(1)

        assert metadata_store.count_active_faces() == 2
        assert metadata_store.count_total_faces() == 3

    def test_get_deleted_faces(self, metadata_store, sample_embedding, sample_bbox):
        """Test getting deleted faces."""
        # Add faces
        id1 = metadata_store.add_face(
            image_path="/path/to/image1.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )
        id2 = metadata_store.add_face(
            image_path="/path/to/image2.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=1,
            confidence=0.96
        )

        # Delete one
        metadata_store.mark_deleted(id1)

        deleted_faces = metadata_store.get_deleted_faces()
        assert len(deleted_faces) == 1
        assert deleted_faces[0].face_id == id1


# Tests for Permanent Delete

class TestPermanentDelete:
    """Tests for permanent deletion."""

    def test_delete_face(self, metadata_store, sample_embedding, sample_bbox):
        """Test permanently deleting a face."""
        face_id = metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        assert metadata_store.count_total_faces() == 1

        # Permanently delete
        result = metadata_store.delete_face(face_id)
        assert result is True

        # Verify face is gone
        assert metadata_store.count_total_faces() == 0
        assert metadata_store.get_face_by_id(face_id) is None

    def test_delete_nonexistent_face(self, metadata_store):
        """Test deleting non-existent face returns False."""
        result = metadata_store.delete_face(9999)
        assert result is False


# Tests for Collection Isolation

class TestCollectionIsolation:
    """Tests for collection isolation."""

    def test_different_collections_isolated(self, temp_db, sample_embedding, sample_bbox):
        """Test different collections are isolated."""
        store1 = MetadataStore(db_path=temp_db, collection_id="collection1")
        store2 = MetadataStore(db_path=temp_db, collection_id="collection2")

        # Add face to collection1
        store1.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        # Add face to collection2
        store2.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.96
        )

        assert store1.count_total_faces() == 1
        assert store2.count_total_faces() == 1

    def test_get_face_respects_collection(self, temp_db, sample_embedding, sample_bbox):
        """Test face retrieval respects collection ID."""
        store1 = MetadataStore(db_path=temp_db, collection_id="collection1")
        store2 = MetadataStore(db_path=temp_db, collection_id="collection2")

        # Add face to collection1
        face_id = store1.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        # Try to get from collection2
        face = store2.get_face_by_id(face_id)
        assert face is None

        # Get from collection1
        face = store1.get_face_by_id(face_id)
        assert face is not None


# Tests for Index Rebuild Support

class TestIndexRebuild:
    """Tests for index rebuild support."""

    def test_rebuild_index_mapping(self, metadata_store, sample_embedding, sample_bbox):
        """Test creating index mapping for rebuild."""
        # Add 5 faces
        for i in range(5):
            metadata_store.add_face(
                image_path=f"/path/to/image{i}.jpg",
                bbox=sample_bbox,
                embedding=sample_embedding,
                embedding_index=i,
                confidence=0.95
            )

        # Delete face with index 2
        metadata_store.mark_deleted(3)  # Face ID 3 has index 2

        # Get rebuild mapping
        mapping = metadata_store.rebuild_index_mapping()

        # Should have 4 active faces (indices 0, 1, 3, 4 -> new indices 0, 1, 2, 3)
        assert len(mapping) == 4
        assert mapping[0] == 0
        assert mapping[1] == 1
        assert mapping[3] == 2
        assert mapping[4] == 3

    def test_update_indices_after_rebuild(self, metadata_store, sample_embedding, sample_bbox):
        """Test updating indices after rebuild."""
        # Add faces
        for i in range(3):
            metadata_store.add_face(
                image_path=f"/path/to/image{i}.jpg",
                bbox=sample_bbox,
                embedding=sample_embedding,
                embedding_index=i * 10,  # Indices: 0, 10, 20
                confidence=0.95
            )

        # Create mapping (simulating rebuild)
        mapping = {0: 0, 10: 1, 20: 2}

        # Update indices
        metadata_store.update_indices_after_rebuild(mapping)

        # Verify new indices
        face0 = metadata_store.get_face_by_index(0)
        face1 = metadata_store.get_face_by_index(1)
        face2 = metadata_store.get_face_by_index(2)

        assert face0 is not None
        assert face1 is not None
        assert face2 is not None


# Tests for Statistics

class TestStatistics:
    """Tests for database statistics."""

    def test_get_stats(self, metadata_store, sample_embedding, sample_bbox):
        """Test getting database statistics."""
        # Add some faces
        for i in range(5):
            metadata_store.add_face(
                image_path=f"/path/to/image{i}.jpg",
                bbox=sample_bbox,
                embedding=sample_embedding,
                embedding_index=i,
                confidence=0.95
            )

        # Delete one
        metadata_store.mark_deleted(1)

        stats = metadata_store.get_stats()
        assert stats['collection_id'] == 'test_collection'
        assert stats['total_faces'] == 5
        assert stats['active_faces'] == 4
        assert stats['deleted_faces'] == 1

    def test_repr(self, metadata_store, sample_embedding, sample_bbox):
        """Test string representation."""
        metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        repr_str = repr(metadata_store)
        assert 'MetadataStore' in repr_str
        assert 'test_collection' in repr_str


# Tests for Face Model

class TestFaceModel:
    """Tests for Face model."""

    def test_face_to_dict(self, metadata_store, sample_embedding, sample_bbox):
        """Test Face to_dict conversion."""
        face_id = metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=sample_embedding,
            embedding_index=0,
            confidence=0.95
        )

        face = metadata_store.get_face_by_id(face_id)
        face_dict = face.to_dict()

        assert face_dict['face_id'] == face_id
        assert face_dict['collection_id'] == 'test_collection'
        assert face_dict['image_path'] == '/path/to/image.jpg'
        assert face_dict['bbox'] == sample_bbox
        assert face_dict['confidence'] == 0.95
        assert face_dict['deleted'] is False

    def test_get_embedding_by_index(self, metadata_store, sample_bbox):
        """Test getting embedding by index."""
        original_embedding = np.random.randn(512).astype(np.float32)

        metadata_store.add_face(
            image_path="/path/to/image.jpg",
            bbox=sample_bbox,
            embedding=original_embedding,
            embedding_index=42,
            confidence=0.95
        )

        retrieved_embedding = metadata_store.get_embedding_by_index(42)
        np.testing.assert_array_equal(original_embedding, retrieved_embedding)

    def test_get_embedding_by_nonexistent_index(self, metadata_store):
        """Test getting embedding by non-existent index."""
        embedding = metadata_store.get_embedding_by_index(9999)
        assert embedding is None


# Integration Tests

class TestIntegration:
    """Integration tests for metadata database."""

    def test_full_workflow(self, metadata_store, sample_bbox):
        """Test complete workflow."""
        embeddings = [np.random.randn(512).astype(np.float32) for _ in range(10)]

        # Add faces
        face_ids = []
        for i, emb in enumerate(embeddings):
            face_id = metadata_store.add_face(
                image_path=f"/path/to/image{i}.jpg",
                bbox=sample_bbox,
                embedding=emb,
                embedding_index=i,
                confidence=0.9 + i * 0.01
            )
            face_ids.append(face_id)

        # Verify count
        assert metadata_store.count_active_faces() == 10

        # Delete some faces
        metadata_store.mark_deleted(face_ids[2])
        metadata_store.mark_deleted(face_ids[5])

        # Verify counts
        assert metadata_store.count_active_faces() == 8
        assert metadata_store.count_total_faces() == 10

        # Get rebuild mapping
        mapping = metadata_store.rebuild_index_mapping()
        assert len(mapping) == 8

        # Verify deleted faces
        deleted = metadata_store.get_deleted_faces()
        assert len(deleted) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
