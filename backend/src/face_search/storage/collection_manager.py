"""Collection manager for face search system.

This module provides the Collection class which manages a complete face collection,
including FAISS index, metadata database, and synchronization between them.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import logging
import numpy as np

from .faiss_index import FAISSIndex, IndexType
from .metadata_db import MetadataStore, Face
from .collection_lock import collection_lock, LockTimeout
from ..models import ModelVersion

logger = logging.getLogger(__name__)


class SyncError(Exception):
    """Raised when FAISS index and metadata are out of sync."""
    pass


class Collection:
    """Face collection manager.

    Manages a complete face collection including:
    - FAISS vector index for similarity search
    - SQLite metadata database for face information
    - Configuration and model version tracking
    - Synchronization between index and database

    Directory structure:
        collection_dir/
            ├── index.faiss         # FAISS index
            ├── metadata.db         # SQLite database
            ├── config.json         # Collection configuration
            └── .lock              # Lock file for concurrent access

    Attributes:
        collection_id: Unique identifier for this collection
        collection_path: Path to collection directory
        embedding_dim: Dimension of face embeddings
    """

    def __init__(
        self,
        collection_id: str,
        collections_root: Path,
        embedding_dim: int = 512,
        model_version: Optional[ModelVersion] = None,
        device: str = 'auto'
    ):
        """Initialize or load a collection.

        Args:
            collection_id: Unique collection identifier
            collections_root: Root directory for all collections
            embedding_dim: Dimension of face embeddings
            model_version: Model version for compatibility checking
            device: Device for FAISS index ('auto', 'cpu', 'cuda', 'mps')
        """
        self.collection_id = collection_id
        self.collection_path = Path(collections_root) / collection_id
        self.embedding_dim = embedding_dim
        self.model_version = model_version
        self.device = device

        # Create collection directory
        self.collection_path.mkdir(parents=True, exist_ok=True)

        # Initialize paths
        self.index_path = self.collection_path / "index.faiss"
        self.db_path = self.collection_path / "metadata.db"
        self.config_path = self.collection_path / "config.json"

        # Initialize components (will be set in _load_or_create)
        self.index: Optional[FAISSIndex] = None
        self.metadata: Optional[MetadataStore] = None
        self.config: Dict[str, Any] = {}

        # Load or create collection
        self._load_or_create()

        logger.info(f"Collection '{collection_id}' ready: {self.ntotal} faces")

    def _load_or_create(self):
        """Load existing collection or create new one."""
        if self.config_path.exists():
            # Load existing collection
            self._load()
        else:
            # Create new collection
            self._create()

    def _create(self):
        """Create a new collection."""
        logger.info(f"Creating new collection: {self.collection_id}")

        # Create FAISS index
        self.index = FAISSIndex(
            dim=self.embedding_dim,
            device=self.device,
            index_type=IndexType.FLAT  # Start with Flat, upgrade to IVF later
        )

        # Create metadata database
        self.metadata = MetadataStore(
            db_path=str(self.db_path),
            collection_id=self.collection_id
        )

        # Create configuration
        self.config = {
            'collection_id': self.collection_id,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'created_at': self.metadata.get_stats()['active_faces'],  # Placeholder
            'model_version': self.model_version.to_dict() if self.model_version else None
        }

        # Save configuration
        self._save_config()

        logger.info(f"Collection created: {self.collection_id}")

    def _load(self):
        """Load existing collection."""
        logger.info(f"Loading collection: {self.collection_id}")

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Verify embedding dimension
        if self.config['embedding_dim'] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {self.config['embedding_dim']}"
            )

        # Load FAISS index
        self.index = FAISSIndex(
            dim=self.embedding_dim,
            device=self.device
        )
        if self.index_path.exists():
            self.index.load(self.index_path)

        # Load metadata database
        self.metadata = MetadataStore(
            db_path=str(self.db_path),
            collection_id=self.collection_id
        )

        # Verify synchronization
        self._verify_sync()

        logger.info(f"Collection loaded: {self.collection_id}")

    def _save_config(self):
        """Save collection configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _verify_sync(self):
        """Verify FAISS index and metadata are synchronized.

        Raises:
            SyncError: If index and metadata are out of sync
        """
        index_count = self.index.ntotal
        db_count = self.metadata.count_active_faces()

        if index_count != db_count:
            raise SyncError(
                f"Collection out of sync: FAISS has {index_count} vectors, "
                f"database has {db_count} active faces. Run rebuild_index()."
            )

    def add_face(
        self,
        image_path: str,
        bbox: Dict[str, float],
        embedding: np.ndarray,
        confidence: float,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None
    ) -> int:
        """Add a face to the collection (atomic operation).

        Args:
            image_path: Path to source image
            bbox: Bounding box dictionary {x1, y1, x2, y2}
            embedding: Face embedding vector
            confidence: Detection confidence score
            yaw: Face yaw angle (left-right rotation)
            pitch: Face pitch angle (up-down tilt)
            roll: Face roll angle (head tilt)

        Returns:
            Face ID

        Raises:
            ValueError: If embedding dimension doesn't match
        """
        # Validate embedding dimension
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} doesn't match "
                f"collection dimension {self.embedding_dim}"
            )

        # Use lock for atomic operation
        with collection_lock(self.collection_path):
            # Add to FAISS index
            embedding_2d = embedding.reshape(1, -1)
            self.index.add(embedding_2d)
            embedding_index = self.index.ntotal - 1  # Last added index

            # Add to metadata database
            try:
                face_id = self.metadata.add_face(
                    image_path=image_path,
                    bbox=bbox,
                    embedding=embedding,
                    embedding_index=embedding_index,
                    confidence=confidence,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll
                )
            except Exception as e:
                # Rollback is difficult with FAISS, log error
                logger.error(
                    f"Failed to add face to database after adding to index. "
                    f"Collection may be out of sync: {e}"
                )
                raise

            logger.debug(f"Added face {face_id} to collection {self.collection_id}")
            return face_id

    def add_faces_batch(
        self,
        faces: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[int]:
        """Add multiple faces in batch (more efficient).

        Args:
            faces: List of face dictionaries with 'image_path', 'bbox', 'confidence',
                   and optionally 'yaw', 'pitch', 'roll'
            embeddings: Array of embeddings, shape (N, dim)

        Returns:
            List of face IDs

        Raises:
            ValueError: If number of faces doesn't match embeddings
        """
        if len(faces) != embeddings.shape[0]:
            raise ValueError(
                f"Number of faces ({len(faces)}) doesn't match "
                f"number of embeddings ({embeddings.shape[0]})"
            )

        face_ids = []

        with collection_lock(self.collection_path):
            # Get starting index
            start_index = self.index.ntotal

            # Add all embeddings to FAISS
            self.index.add(embeddings)

            # Add to metadata database
            for i, face_data in enumerate(faces):
                embedding_index = start_index + i
                face_id = self.metadata.add_face(
                    image_path=face_data['image_path'],
                    bbox=face_data['bbox'],
                    embedding=embeddings[i],
                    embedding_index=embedding_index,
                    confidence=face_data['confidence'],
                    yaw=face_data.get('yaw'),
                    pitch=face_data.get('pitch'),
                    roll=face_data.get('roll')
                )
                face_ids.append(face_id)

        logger.info(f"Added {len(face_ids)} faces to collection {self.collection_id}")
        return face_ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[Face, float]]:
        """Search for similar faces.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Optional distance threshold (only return if distance < threshold)

        Returns:
            List of (Face, distance) tuples, sorted by similarity

        Raises:
            ValueError: If query embedding dimension doesn't match
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query dimension {query_embedding.shape[0]} doesn't match "
                f"collection dimension {self.embedding_dim}"
            )

        # Search FAISS index
        query_2d = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_2d, k=k)

        # Get metadata for results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip invalid indices
            if idx < 0:
                continue

            # Apply threshold
            if threshold is not None and dist > threshold:
                continue

            # Get face metadata
            face = self.metadata.get_face_by_index(int(idx))
            if face and not face.deleted:
                results.append((face, float(dist)))

        return results

    def search_by_face_id(
        self,
        face_id: int,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[Face, float]]:
        """Search for faces similar to a specific face in the collection.

        Args:
            face_id: ID of the face to use as query
            k: Number of results to return
            threshold: Optional distance threshold

        Returns:
            List of (Face, distance) tuples

        Raises:
            ValueError: If face not found
        """
        # Get face embedding
        face = self.metadata.get_face_by_id(face_id)
        if not face:
            raise ValueError(f"Face {face_id} not found in collection")

        query_embedding = face.get_embedding()
        return self.search(query_embedding, k=k + 1, threshold=threshold)[1:]  # Skip self

    def delete_face(self, face_id: int) -> bool:
        """Mark a face as deleted (soft delete).

        Args:
            face_id: Face ID to delete

        Returns:
            True if successful
        """
        with collection_lock(self.collection_path):
            result = self.metadata.mark_deleted(face_id)
            if result:
                logger.info(f"Marked face {face_id} as deleted")
            return result

    def rebuild_index(self):
        """Rebuild FAISS index, removing deleted faces.

        This is an expensive operation that should be done periodically
        to reclaim space and improve performance.
        """
        logger.info(f"Rebuilding index for collection {self.collection_id}")

        with collection_lock(self.collection_path, timeout=120.0):  # Longer timeout
            # Get all active faces
            active_faces = self.metadata.get_all_faces(include_deleted=False)

            if not active_faces:
                # No active faces, create empty index
                self.index.reset()
                logger.info("Index rebuilt (empty)")
                return

            # Extract embeddings
            embeddings = []
            for face in active_faces:
                embeddings.append(face.get_embedding())

            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Create new index
            old_index_type = self.index.index_type
            new_index = FAISSIndex(
                dim=self.embedding_dim,
                device=self.device,
                index_type=old_index_type
            )

            # Add embeddings
            new_index.add(embeddings_array)

            # Update embedding indices in database
            index_mapping = {}
            for new_idx, face in enumerate(active_faces):
                old_idx = face.embedding_index
                index_mapping[old_idx] = new_idx

            self.metadata.update_indices_after_rebuild(index_mapping)

            # Replace old index
            self.index = new_index

            # Save
            self.save()

            logger.info(
                f"Index rebuilt: {len(active_faces)} active faces, "
                f"removed {len(index_mapping) - len(active_faces)} deleted faces"
            )

    def save(self):
        """Save collection to disk."""
        with collection_lock(self.collection_path):
            # Save FAISS index
            self.index.save(self.index_path)

            # Database auto-saves
            # Save configuration
            self._save_config()

            logger.debug(f"Collection saved: {self.collection_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        db_stats = self.metadata.get_stats()
        index_stats = self.index.get_stats()

        return {
            'collection_id': self.collection_id,
            'total_faces': db_stats['total_faces'],
            'active_faces': db_stats['active_faces'],
            'deleted_faces': db_stats['deleted_faces'],
            'index_type': index_stats['index_type'],
            'device': index_stats['device'],
            'embedding_dim': self.embedding_dim,
            'synced': self.index.ntotal == db_stats['active_faces']
        }

    @property
    def ntotal(self) -> int:
        """Get total number of active faces."""
        return self.metadata.count_active_faces()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Collection(id='{self.collection_id}', "
            f"faces={self.ntotal}, "
            f"dim={self.embedding_dim})"
        )

    def __len__(self) -> int:
        """Get number of active faces."""
        return self.ntotal
