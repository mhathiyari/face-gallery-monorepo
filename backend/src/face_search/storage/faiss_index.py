"""FAISS-based vector similarity search index.

This module provides a high-level interface to FAISS for fast similarity search
of face embeddings. It supports both CPU and GPU indices, automatic index type
selection, and efficient save/load operations.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """FAISS index types."""
    FLAT = "Flat"  # Exact search (IndexFlatL2)
    IVF = "IVF"    # Approximate search with inverted file index


class FAISSIndex:
    """FAISS-based similarity search index for face embeddings.

    This class provides a high-level interface to FAISS with automatic device
    selection, index type optimization, and save/load functionality.

    Attributes:
        dim: Dimension of the embedding vectors
        device: Device to use ('cuda', 'mps', 'cpu', 'auto')
        index_type: Type of FAISS index (Flat or IVF)
        metric: Distance metric ('L2' or 'IP' for inner product)
    """

    def __init__(
        self,
        dim: int,
        device: str = 'auto',
        index_type: Optional[IndexType] = None,
        metric: str = 'L2',
        nlist: int = 100
    ):
        """Initialize FAISS index.

        Args:
            dim: Dimension of the embedding vectors
            device: Device to use ('cuda', 'mps', 'cpu', 'auto')
            index_type: Type of index (None for automatic selection)
            metric: Distance metric ('L2' or 'IP')
            nlist: Number of clusters for IVF index (default: 100)
        """
        self.dim = dim
        self.metric = metric
        self.nlist = nlist
        self._index = None
        self._is_trained = False
        self._index_type = index_type

        # Import FAISS
        try:
            import faiss
            self.faiss = faiss
        except ImportError as e:
            raise ImportError(
                "FAISS is not installed. "
                "Install it with: pip install faiss-cpu or pip install faiss-gpu"
            ) from e

        # Select device
        self._device = self._select_device(device)
        self._use_gpu = self._device in ['cuda', 'mps']

        # Create index
        self._create_index(index_type)

        logger.info(
            f"FAISS index created: dim={dim}, type={self._index_type.value}, "
            f"device={self._device}, metric={metric}"
        )

    def _select_device(self, device: str) -> str:
        """Select the best available device.

        Args:
            device: Requested device ('cuda', 'mps', 'cpu', 'auto')

        Returns:
            Selected device string
        """
        if device != 'auto':
            return device

        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Note: FAISS doesn't directly support MPS, so we'll use CPU
                logger.info("MPS detected but FAISS doesn't support it directly, using CPU")
                return 'cpu'
        except ImportError:
            pass

        return 'cpu'

    def _create_index(self, index_type: Optional[IndexType] = None):
        """Create a FAISS index.

        Args:
            index_type: Type of index to create (None for automatic selection)
        """
        if index_type is None:
            # Default to Flat index (exact search)
            index_type = IndexType.FLAT

        self._index_type = index_type

        if index_type == IndexType.FLAT:
            # Create exact search index (IndexFlatL2)
            if self.metric == 'L2':
                self._index = self.faiss.IndexFlatL2(self.dim)
            else:  # Inner Product
                self._index = self.faiss.IndexFlatIP(self.dim)
            self._is_trained = True  # Flat index doesn't need training

        elif index_type == IndexType.IVF:
            # Create approximate search index with IVF
            if self.metric == 'L2':
                quantizer = self.faiss.IndexFlatL2(self.dim)
                self._index = self.faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
            else:  # Inner Product
                quantizer = self.faiss.IndexFlatIP(self.dim)
                self._index = self.faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
            self._is_trained = False  # IVF index needs training

        # Move to GPU if available and requested
        if self._use_gpu and self._device == 'cuda':
            try:
                res = self.faiss.StandardGpuResources()
                self._index = self.faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("Index moved to GPU (CUDA)")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}, using CPU")
                self._use_gpu = False
                self._device = 'cpu'

    @staticmethod
    def get_optimal_index_type(n_vectors: int, dim: int) -> IndexType:
        """Determine optimal index type based on dataset size.

        Args:
            n_vectors: Number of vectors in the dataset
            dim: Dimension of the vectors

        Returns:
            Recommended IndexType

        Notes:
            - Flat (exact): < 10,000 vectors
            - IVF (approximate): >= 10,000 vectors
        """
        if n_vectors < 10000:
            return IndexType.FLAT
        else:
            return IndexType.IVF

    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index.

        Args:
            embeddings: Array of embeddings, shape (N, dim)

        Raises:
            ValueError: If embeddings have wrong shape or index not trained
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.dim}"
            )

        # Ensure float32 type (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Train index if needed (IVF indices)
        if not self._is_trained:
            if embeddings.shape[0] < self.nlist:
                logger.warning(
                    f"Training IVF index with {embeddings.shape[0]} vectors, "
                    f"but nlist={self.nlist}. Consider using fewer clusters or more data."
                )
            logger.info(f"Training IVF index with {embeddings.shape[0]} vectors...")
            self._index.train(embeddings)
            self._is_trained = True
            logger.info("IVF index training complete")

        # Add vectors
        self._index.add(embeddings)
        logger.debug(f"Added {embeddings.shape[0]} vectors to index")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        return_distances: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            query: Query embedding(s), shape (dim,) or (N, dim)
            k: Number of nearest neighbors to return
            return_distances: Whether to return distances

        Returns:
            If return_distances=True: (distances, indices) both shape (N, k)
            If return_distances=False: indices only, shape (N, k)

        Raises:
            ValueError: If query has wrong shape or index is empty
        """
        if self.ntotal == 0:
            raise ValueError("Cannot search in empty index")

        # Handle 1D query (single vector)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2:
            raise ValueError(f"Expected 1D or 2D query, got shape {query.shape}")

        if query.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension {query.shape[1]} doesn't match "
                f"index dimension {self.dim}"
            )

        # Ensure float32 type
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        # Ensure k doesn't exceed index size
        k = min(k, self.ntotal)

        # Search
        distances, indices = self._index.search(query, k)

        if return_distances:
            return distances, indices
        else:
            return indices

    def remove(self, ids: np.ndarray) -> None:
        """Remove vectors by IDs (only for some index types).

        Args:
            ids: Array of IDs to remove

        Raises:
            NotImplementedError: If index type doesn't support removal
        """
        if not hasattr(self._index, 'remove_ids'):
            raise NotImplementedError(
                f"Index type {self._index_type.value} doesn't support removal. "
                "Consider rebuilding the index instead."
            )

        # Convert to IDSelector
        sel = self.faiss.IDSelectorBatch(ids.astype(np.int64))
        self._index.remove_ids(sel)
        logger.debug(f"Removed {len(ids)} vectors from index")

    def reset(self) -> None:
        """Reset the index (remove all vectors)."""
        self._index.reset()
        if self._index_type == IndexType.IVF:
            self._is_trained = False
        logger.debug("Index reset")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save index to disk.

        For GPU indices, automatically converts to CPU before saving.

        Args:
            filepath: Path to save the index
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert GPU index to CPU for saving
        if self._use_gpu and self._device == 'cuda':
            try:
                cpu_index = self.faiss.index_gpu_to_cpu(self._index)
                self.faiss.write_index(cpu_index, str(filepath))
                logger.info(f"Saved GPU index to {filepath} (converted to CPU)")
            except Exception as e:
                logger.error(f"Failed to save GPU index: {e}")
                raise
        else:
            self.faiss.write_index(self._index, str(filepath))
            logger.info(f"Saved index to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """Load index from disk.

        Automatically converts to GPU if requested device is CUDA.

        Args:
            filepath: Path to the saved index

        Raises:
            FileNotFoundError: If index file doesn't exist
            ValueError: If loaded index has wrong dimension
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")

        # Load index
        loaded_index = self.faiss.read_index(str(filepath))

        # Verify dimension
        if loaded_index.d != self.dim:
            raise ValueError(
                f"Loaded index dimension {loaded_index.d} doesn't match "
                f"expected dimension {self.dim}"
            )

        # Convert to GPU if needed
        if self._use_gpu and self._device == 'cuda':
            try:
                res = self.faiss.StandardGpuResources()
                self._index = self.faiss.index_cpu_to_gpu(res, 0, loaded_index)
                logger.info(f"Loaded index from {filepath} and moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move loaded index to GPU: {e}, using CPU")
                self._index = loaded_index
                self._use_gpu = False
                self._device = 'cpu'
        else:
            self._index = loaded_index
            logger.info(f"Loaded index from {filepath}")

        # Update training status
        self._is_trained = True

    @property
    def ntotal(self) -> int:
        """Get total number of vectors in the index."""
        return self._index.ntotal

    @property
    def is_trained(self) -> bool:
        """Check if index is trained."""
        return self._is_trained

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def index_type(self) -> IndexType:
        """Get index type."""
        return self._index_type

    def get_stats(self) -> dict:
        """Get index statistics.

        Returns:
            Dictionary with index stats
        """
        return {
            'ntotal': self.ntotal,
            'dim': self.dim,
            'index_type': self._index_type.value,
            'device': self._device,
            'metric': self.metric,
            'is_trained': self._is_trained,
            'nlist': self.nlist if self._index_type == IndexType.IVF else None
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FAISSIndex(dim={self.dim}, type={self._index_type.value}, "
            f"device={self._device}, ntotal={self.ntotal})"
        )

    def __len__(self) -> int:
        """Get number of vectors in index."""
        return self.ntotal
