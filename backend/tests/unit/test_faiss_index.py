"""Unit tests for FAISS index manager.

Tests cover:
- Index creation (Flat and IVF)
- Adding vectors
- Searching for similar vectors
- Save/load functionality
- Error handling
- GPU/CPU device management
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from face_search.storage import FAISSIndex, IndexType


# Test fixtures

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    # Create 100 embeddings of dimension 512
    return np.random.randn(100, 512).astype(np.float32)


@pytest.fixture
def small_embeddings():
    """Create small set of embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 128).astype(np.float32)


@pytest.fixture
def flat_index():
    """Create a Flat FAISS index."""
    return FAISSIndex(dim=512, device='cpu', index_type=IndexType.FLAT)


@pytest.fixture
def ivf_index():
    """Create an IVF FAISS index."""
    return FAISSIndex(dim=512, device='cpu', index_type=IndexType.IVF, nlist=10)


# Tests for Index Creation

class TestIndexCreation:
    """Tests for FAISS index creation."""

    def test_flat_index_creation(self):
        """Test Flat index creation."""
        index = FAISSIndex(dim=512, device='cpu', index_type=IndexType.FLAT)
        assert index.dim == 512
        assert index.device == 'cpu'
        assert index.index_type == IndexType.FLAT
        assert index.ntotal == 0
        assert index.is_trained  # Flat index is always trained

    def test_ivf_index_creation(self):
        """Test IVF index creation."""
        index = FAISSIndex(dim=512, device='cpu', index_type=IndexType.IVF, nlist=10)
        assert index.dim == 512
        assert index.index_type == IndexType.IVF
        assert not index.is_trained  # IVF index needs training

    def test_auto_device_selection(self):
        """Test automatic device selection."""
        index = FAISSIndex(dim=512, device='auto')
        assert index.device in ['cpu', 'cuda', 'mps']

    def test_different_dimensions(self):
        """Test creating indices with different dimensions."""
        for dim in [128, 256, 512, 1024]:
            index = FAISSIndex(dim=dim, device='cpu')
            assert index.dim == dim

    def test_different_metrics(self):
        """Test creating indices with different metrics."""
        # L2 metric
        index_l2 = FAISSIndex(dim=512, device='cpu', metric='L2')
        assert index_l2.metric == 'L2'

        # Inner Product metric
        index_ip = FAISSIndex(dim=512, device='cpu', metric='IP')
        assert index_ip.metric == 'IP'


# Tests for Adding Vectors

class TestAddVectors:
    """Tests for adding vectors to the index."""

    def test_add_vectors_flat(self, flat_index, sample_embeddings):
        """Test adding vectors to Flat index."""
        flat_index.add(sample_embeddings)
        assert flat_index.ntotal == 100

    def test_add_vectors_ivf(self, ivf_index, sample_embeddings):
        """Test adding vectors to IVF index (includes training)."""
        assert not ivf_index.is_trained
        ivf_index.add(sample_embeddings)
        assert ivf_index.is_trained
        assert ivf_index.ntotal == 100

    def test_add_multiple_batches(self, flat_index):
        """Test adding vectors in multiple batches."""
        batch1 = np.random.randn(50, 512).astype(np.float32)
        batch2 = np.random.randn(30, 512).astype(np.float32)

        flat_index.add(batch1)
        assert flat_index.ntotal == 50

        flat_index.add(batch2)
        assert flat_index.ntotal == 80

    def test_add_wrong_dimension(self, flat_index):
        """Test adding vectors with wrong dimension."""
        wrong_embeddings = np.random.randn(10, 256).astype(np.float32)
        with pytest.raises(ValueError, match="doesn't match"):
            flat_index.add(wrong_embeddings)

    def test_add_wrong_shape(self, flat_index):
        """Test adding vectors with wrong shape."""
        wrong_shape = np.random.randn(512).astype(np.float32)  # 1D instead of 2D
        with pytest.raises(ValueError, match="Expected 2D array"):
            flat_index.add(wrong_shape)

    def test_add_auto_convert_dtype(self, flat_index):
        """Test automatic conversion to float32."""
        embeddings_float64 = np.random.randn(10, 512).astype(np.float64)
        flat_index.add(embeddings_float64)
        assert flat_index.ntotal == 10


# Tests for Search

class TestSearch:
    """Tests for searching in the index."""

    def test_search_flat(self, flat_index, sample_embeddings):
        """Test searching in Flat index."""
        flat_index.add(sample_embeddings)

        # Search with first embedding
        query = sample_embeddings[0:1]
        distances, indices = flat_index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        # First result should be the query itself (distance ~0)
        assert indices[0, 0] == 0
        assert distances[0, 0] < 1e-5

    def test_search_ivf(self, ivf_index, sample_embeddings):
        """Test searching in IVF index."""
        ivf_index.add(sample_embeddings)

        query = sample_embeddings[0:1]
        distances, indices = ivf_index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)

    def test_search_1d_query(self, flat_index, sample_embeddings):
        """Test searching with 1D query."""
        flat_index.add(sample_embeddings)

        # 1D query (single vector)
        query = sample_embeddings[0]
        distances, indices = flat_index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)

    def test_search_multiple_queries(self, flat_index, sample_embeddings):
        """Test searching with multiple queries."""
        flat_index.add(sample_embeddings)

        # Multiple queries
        queries = sample_embeddings[0:3]
        distances, indices = flat_index.search(queries, k=5)

        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)

    def test_search_without_distances(self, flat_index, sample_embeddings):
        """Test searching without returning distances."""
        flat_index.add(sample_embeddings)

        query = sample_embeddings[0:1]
        indices = flat_index.search(query, k=5, return_distances=False)

        assert isinstance(indices, np.ndarray)
        assert indices.shape == (1, 5)

    def test_search_empty_index(self, flat_index):
        """Test searching in empty index."""
        query = np.random.randn(1, 512).astype(np.float32)
        with pytest.raises(ValueError, match="empty index"):
            flat_index.search(query, k=5)

    def test_search_k_larger_than_index(self, flat_index, small_embeddings):
        """Test searching with k larger than index size."""
        flat_index_small = FAISSIndex(dim=128, device='cpu')
        flat_index_small.add(small_embeddings)

        query = small_embeddings[0:1]
        # Request more neighbors than exist
        distances, indices = flat_index_small.search(query, k=100)

        # Should return only available vectors
        assert distances.shape == (1, 10)
        assert indices.shape == (1, 10)

    def test_search_wrong_dimension(self, flat_index, sample_embeddings):
        """Test searching with wrong dimension."""
        flat_index.add(sample_embeddings)

        wrong_query = np.random.randn(1, 256).astype(np.float32)
        with pytest.raises(ValueError, match="doesn't match"):
            flat_index.search(wrong_query, k=5)


# Tests for Save/Load

class TestSaveLoad:
    """Tests for saving and loading indices."""

    def test_save_flat_index(self, flat_index, sample_embeddings, tmp_path):
        """Test saving Flat index."""
        flat_index.add(sample_embeddings)

        filepath = tmp_path / "test_index.faiss"
        flat_index.save(filepath)

        assert filepath.exists()

    def test_load_flat_index(self, flat_index, sample_embeddings, tmp_path):
        """Test loading Flat index."""
        flat_index.add(sample_embeddings)

        filepath = tmp_path / "test_index.faiss"
        flat_index.save(filepath)

        # Create new index and load
        new_index = FAISSIndex(dim=512, device='cpu')
        new_index.load(filepath)

        assert new_index.ntotal == 100
        assert new_index.is_trained

        # Test search on loaded index
        query = sample_embeddings[0:1]
        distances, indices = new_index.search(query, k=5)
        assert indices[0, 0] == 0

    def test_save_ivf_index(self, ivf_index, sample_embeddings, tmp_path):
        """Test saving IVF index."""
        ivf_index.add(sample_embeddings)

        filepath = tmp_path / "test_ivf_index.faiss"
        ivf_index.save(filepath)

        assert filepath.exists()

    def test_load_ivf_index(self, ivf_index, sample_embeddings, tmp_path):
        """Test loading IVF index."""
        ivf_index.add(sample_embeddings)

        filepath = tmp_path / "test_ivf_index.faiss"
        ivf_index.save(filepath)

        # Create new index and load
        new_index = FAISSIndex(dim=512, device='cpu', index_type=IndexType.IVF)
        new_index.load(filepath)

        assert new_index.ntotal == 100
        assert new_index.is_trained

    def test_load_nonexistent_file(self, flat_index):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            flat_index.load("nonexistent_index.faiss")

    def test_load_wrong_dimension(self, sample_embeddings, tmp_path):
        """Test loading index with wrong dimension."""
        # Create and save index with dim=512
        index1 = FAISSIndex(dim=512, device='cpu')
        index1.add(sample_embeddings)

        filepath = tmp_path / "test_index.faiss"
        index1.save(filepath)

        # Try to load with wrong dimension
        index2 = FAISSIndex(dim=256, device='cpu')
        with pytest.raises(ValueError, match="doesn't match"):
            index2.load(filepath)

    def test_save_creates_directory(self, flat_index, sample_embeddings, tmp_path):
        """Test that save creates parent directories."""
        flat_index.add(sample_embeddings)

        # Save to nested directory
        filepath = tmp_path / "subdir" / "nested" / "index.faiss"
        flat_index.save(filepath)

        assert filepath.exists()
        assert filepath.parent.exists()


# Tests for Index Operations

class TestIndexOperations:
    """Tests for various index operations."""

    def test_reset_index(self, flat_index, sample_embeddings):
        """Test resetting index."""
        flat_index.add(sample_embeddings)
        assert flat_index.ntotal == 100

        flat_index.reset()
        assert flat_index.ntotal == 0

    def test_get_stats(self, flat_index, sample_embeddings):
        """Test getting index statistics."""
        flat_index.add(sample_embeddings)

        stats = flat_index.get_stats()
        assert stats['ntotal'] == 100
        assert stats['dim'] == 512
        assert stats['index_type'] == 'Flat'
        assert stats['device'] == 'cpu'
        assert stats['metric'] == 'L2'
        assert stats['is_trained'] is True

    def test_len(self, flat_index, sample_embeddings):
        """Test __len__ method."""
        assert len(flat_index) == 0

        flat_index.add(sample_embeddings)
        assert len(flat_index) == 100

    def test_repr(self, flat_index):
        """Test string representation."""
        repr_str = repr(flat_index)
        assert 'FAISSIndex' in repr_str
        assert 'dim=512' in repr_str
        assert 'type=Flat' in repr_str


# Tests for Optimal Index Selection

class TestOptimalIndexSelection:
    """Tests for optimal index type selection."""

    def test_small_dataset_uses_flat(self):
        """Test that small datasets use Flat index."""
        index_type = FAISSIndex.get_optimal_index_type(n_vectors=1000, dim=512)
        assert index_type == IndexType.FLAT

    def test_large_dataset_uses_ivf(self):
        """Test that large datasets use IVF index."""
        index_type = FAISSIndex.get_optimal_index_type(n_vectors=50000, dim=512)
        assert index_type == IndexType.IVF

    def test_threshold_at_10000(self):
        """Test threshold at 10,000 vectors."""
        # Just below threshold
        index_type1 = FAISSIndex.get_optimal_index_type(n_vectors=9999, dim=512)
        assert index_type1 == IndexType.FLAT

        # At threshold
        index_type2 = FAISSIndex.get_optimal_index_type(n_vectors=10000, dim=512)
        assert index_type2 == IndexType.IVF


# Integration Tests

class TestIntegration:
    """Integration tests for FAISS index."""

    def test_full_workflow(self, sample_embeddings, tmp_path):
        """Test complete workflow: create -> add -> search -> save -> load."""
        # Create index
        index1 = FAISSIndex(dim=512, device='cpu', index_type=IndexType.FLAT)

        # Add vectors
        index1.add(sample_embeddings)
        assert index1.ntotal == 100

        # Search
        query = sample_embeddings[0:1]
        distances1, indices1 = index1.search(query, k=5)
        assert indices1[0, 0] == 0

        # Save
        filepath = tmp_path / "workflow_index.faiss"
        index1.save(filepath)

        # Load into new index
        index2 = FAISSIndex(dim=512, device='cpu')
        index2.load(filepath)

        # Search in loaded index
        distances2, indices2 = index2.search(query, k=5)

        # Results should be identical
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_allclose(distances1, distances2, rtol=1e-5)

    def test_incremental_indexing(self, tmp_path):
        """Test adding vectors incrementally and saving."""
        index = FAISSIndex(dim=128, device='cpu')

        # Add in batches
        for i in range(5):
            batch = np.random.randn(20, 128).astype(np.float32)
            index.add(batch)

        assert index.ntotal == 100

        # Save and reload
        filepath = tmp_path / "incremental_index.faiss"
        index.save(filepath)

        new_index = FAISSIndex(dim=128, device='cpu')
        new_index.load(filepath)
        assert new_index.ntotal == 100

    def test_different_index_types_same_data(self, sample_embeddings):
        """Test that different index types work with same data."""
        # Flat index
        flat_index = FAISSIndex(dim=512, device='cpu', index_type=IndexType.FLAT)
        flat_index.add(sample_embeddings)

        # IVF index
        ivf_index = FAISSIndex(dim=512, device='cpu', index_type=IndexType.IVF, nlist=10)
        ivf_index.add(sample_embeddings)

        # Both should have same number of vectors
        assert flat_index.ntotal == ivf_index.ntotal == 100

        # Search should work in both
        query = sample_embeddings[0:1]
        distances_flat, indices_flat = flat_index.search(query, k=5)
        distances_ivf, indices_ivf = ivf_index.search(query, k=5)

        # Results should be similar (exact for first result)
        assert indices_flat[0, 0] == indices_ivf[0, 0] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
