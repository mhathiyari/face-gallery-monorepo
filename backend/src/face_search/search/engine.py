"""Search engine for face similarity search.

This module provides the SearchEngine class for searching faces by image,
face ID, or metadata filters. It integrates with the Collection and Model
to provide high-level search functionality.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Callable
import logging
import numpy as np

from ..models import BaseFaceModel
from ..storage import Collection
from ..face import FaceDetection, SearchResult, BoundingBox, Landmarks
from .results import format_results_aws_style, rank_results

logger = logging.getLogger(__name__)


class SearchEngine:
    """High-level search engine for face similarity search.

    Provides methods for searching faces by:
    - Image file (detect faces and search for each)
    - Face ID (search for similar faces to a known face)
    - Metadata filters (filter by image path, confidence, etc.)

    Usage:
        engine = SearchEngine(
            model=face_model,
            collection=collection
        )

        # Search by image
        results = engine.search_by_image(
            image_path='query.jpg',
            max_results=10,
            min_similarity=0.8
        )

        # Search by face ID
        results = engine.search_by_face_id(
            face_id=123,
            max_results=10
        )
    """

    def __init__(
        self,
        model: BaseFaceModel,
        collection: Collection,
        default_max_results: int = 10,
        default_min_confidence: float = 0.5
    ):
        """Initialize search engine.

        Args:
            model: Face detection/embedding model
            collection: Face collection to search
            default_max_results: Default maximum number of results
            default_min_confidence: Default minimum detection confidence
        """
        self.model = model
        self.collection = collection
        self.default_max_results = default_max_results
        self.default_min_confidence = default_min_confidence

        logger.info(
            f"SearchEngine initialized for collection '{collection.collection_id}' "
            f"with {collection.ntotal} faces"
        )

    def search_by_image(
        self,
        image_path: Union[str, Path],
        max_results: Optional[int] = None,
        min_similarity: Optional[float] = None,
        min_confidence: Optional[float] = None,
        return_all_faces: bool = False
    ) -> List[List[SearchResult]]:
        """Search for faces similar to those in an image.

        Detects all faces in the query image and searches for similar faces
        for each detected face.

        Args:
            image_path: Path to query image
            max_results: Maximum results per face (default: default_max_results)
            min_similarity: Minimum similarity score (0-1, higher is more similar)
            min_confidence: Minimum detection confidence (default: default_min_confidence)
            return_all_faces: If False, only return results for the highest confidence face

        Returns:
            List of search results, one list per detected face
            Each inner list contains SearchResult objects ranked by similarity

        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If no faces detected in image
        """
        image_path = Path(image_path)
        max_results = max_results or self.default_max_results
        min_confidence = min_confidence or self.default_min_confidence

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Detect faces in query image
        logger.info(f"Detecting faces in {image_path}")
        detected_faces = self.model.detect_faces(
            str(image_path),
            min_confidence=min_confidence
        )

        if not detected_faces:
            raise ValueError(f"No faces detected in {image_path}")

        logger.info(f"Found {len(detected_faces)} face(s) in query image")

        # Convert to FaceDetection objects
        query_faces = []
        for df in detected_faces:
            face = FaceDetection(
                bbox=BoundingBox(
                    x1=df.bbox.x1,
                    y1=df.bbox.y1,
                    x2=df.bbox.x2,
                    y2=df.bbox.y2
                ),
                confidence=df.confidence,
                embedding=df.embedding,
                landmarks=Landmarks.from_array(df.landmarks) if df.landmarks is not None else None,
                image_path=image_path
            )
            query_faces.append(face)

        # If only returning results for best face
        if not return_all_faces:
            # Find face with highest confidence
            best_face = max(query_faces, key=lambda f: f.confidence)
            query_faces = [best_face]

        # Search for each detected face
        all_results = []
        for i, face in enumerate(query_faces):
            logger.debug(
                f"Searching for face {i+1}/{len(query_faces)} "
                f"(confidence: {face.confidence:.2f})"
            )

            results = self._search_by_embedding(
                embedding=face.embedding,
                max_results=max_results,
                min_similarity=min_similarity,
                query_face=face
            )

            all_results.append(results)

        return all_results

    def search_by_face_id(
        self,
        face_id: int,
        max_results: Optional[int] = None,
        min_similarity: Optional[float] = None,
        include_self: bool = False
    ) -> List[SearchResult]:
        """Search for faces similar to a specific face in the collection.

        Args:
            face_id: ID of the face to use as query
            max_results: Maximum number of results
            min_similarity: Minimum similarity score (0-1)
            include_self: If True, include the query face in results

        Returns:
            List of SearchResult objects ranked by similarity

        Raises:
            ValueError: If face ID not found
        """
        max_results = max_results or self.default_max_results

        logger.info(f"Searching for faces similar to face_id={face_id}")

        # Get face from collection
        face_record = self.collection.metadata.get_face_by_id(face_id)
        if not face_record:
            raise ValueError(f"Face ID {face_id} not found in collection")

        # Get embedding
        embedding = face_record.get_embedding()

        # Create FaceDetection for query face
        query_face = FaceDetection(
            bbox=BoundingBox.from_dict(face_record.bbox if isinstance(face_record.bbox, dict) else eval(face_record.bbox)),
            confidence=face_record.confidence,
            embedding=embedding,
            image_path=Path(face_record.image_path),
            face_id=face_record.face_id,
            embedding_index=face_record.embedding_index
        )

        # Search
        results = self._search_by_embedding(
            embedding=embedding,
            max_results=max_results + 1 if not include_self else max_results,
            min_similarity=min_similarity,
            query_face=query_face
        )

        # Remove self from results if requested
        if not include_self:
            results = [r for r in results if r.face.face_id != face_id]

        return results[:max_results]

    def search_by_metadata(
        self,
        image_path: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        limit: int = 100
    ) -> List[FaceDetection]:
        """Search faces by metadata filters.

        Args:
            image_path: Filter by image path (exact match or substring)
            min_confidence: Minimum detection confidence
            max_confidence: Maximum detection confidence
            limit: Maximum number of results

        Returns:
            List of FaceDetection objects matching filters
        """
        logger.info(f"Searching by metadata: image_path={image_path}, confidence=[{min_confidence}, {max_confidence}]")

        # Get all active faces
        all_faces = self.collection.metadata.get_all_faces(include_deleted=False)

        # Apply filters
        filtered_faces = []
        for face_record in all_faces:
            # Filter by image path
            if image_path is not None:
                if image_path not in face_record.image_path:
                    continue

            # Filter by confidence
            if min_confidence is not None and face_record.confidence < min_confidence:
                continue
            if max_confidence is not None and face_record.confidence > max_confidence:
                continue

            # Convert to FaceDetection
            face = FaceDetection(
                bbox=BoundingBox.from_dict(face_record.bbox if isinstance(face_record.bbox, dict) else eval(face_record.bbox)),
                confidence=face_record.confidence,
                embedding=face_record.get_embedding(),
                image_path=Path(face_record.image_path),
                face_id=face_record.face_id,
                embedding_index=face_record.embedding_index
            )

            filtered_faces.append(face)

            # Check limit
            if len(filtered_faces) >= limit:
                break

        logger.info(f"Found {len(filtered_faces)} faces matching filters")
        return filtered_faces

    def compare_faces(
        self,
        source_image: Union[str, Path],
        target_image: Union[str, Path],
        min_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compare faces between two images.

        Detects faces in both images and finds the best match between them.

        Args:
            source_image: Path to source image
            target_image: Path to target image
            min_confidence: Minimum detection confidence

        Returns:
            Dictionary with comparison results:
            {
                'source_faces': List[FaceDetection],
                'target_faces': List[FaceDetection],
                'best_match': {
                    'source_face_idx': int,
                    'target_face_idx': int,
                    'similarity': float,
                    'distance': float
                },
                'all_matches': List[Dict] (all pairwise comparisons)
            }

        Raises:
            FileNotFoundError: If images don't exist
            ValueError: If no faces detected
        """
        min_confidence = min_confidence or self.default_min_confidence

        # Detect faces in both images
        source_faces = self.model.detect_faces(str(source_image), min_confidence)
        target_faces = self.model.detect_faces(str(target_image), min_confidence)

        if not source_faces:
            raise ValueError(f"No faces detected in source image: {source_image}")
        if not target_faces:
            raise ValueError(f"No faces detected in target image: {target_image}")

        # Compare all pairs
        all_matches = []
        best_match = None
        best_similarity = -1

        for i, src_face in enumerate(source_faces):
            for j, tgt_face in enumerate(target_faces):
                # Compute similarity
                similarity = self._compute_similarity(
                    src_face.embedding,
                    tgt_face.embedding
                )
                distance = self._compute_distance(
                    src_face.embedding,
                    tgt_face.embedding
                )

                match = {
                    'source_face_idx': i,
                    'target_face_idx': j,
                    'similarity': similarity,
                    'distance': distance
                }
                all_matches.append(match)

                # Track best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = match

        # Convert to FaceDetection objects
        source_face_detections = [
            FaceDetection(
                bbox=BoundingBox(x1=f.bbox.x1, y1=f.bbox.y1, x2=f.bbox.x2, y2=f.bbox.y2),
                confidence=f.confidence,
                embedding=f.embedding,
                image_path=Path(source_image)
            )
            for f in source_faces
        ]

        target_face_detections = [
            FaceDetection(
                bbox=BoundingBox(x1=f.bbox.x1, y1=f.bbox.y1, x2=f.bbox.x2, y2=f.bbox.y2),
                confidence=f.confidence,
                embedding=f.embedding,
                image_path=Path(target_image)
            )
            for f in target_faces
        ]

        return {
            'source_faces': source_face_detections,
            'target_faces': target_face_detections,
            'best_match': best_match,
            'all_matches': all_matches
        }

    def _search_by_embedding(
        self,
        embedding: np.ndarray,
        max_results: int,
        min_similarity: Optional[float] = None,
        query_face: Optional[FaceDetection] = None
    ) -> List[SearchResult]:
        """Internal method to search by embedding vector.

        Args:
            embedding: Query embedding vector
            max_results: Maximum number of results
            min_similarity: Minimum similarity score (0-1)
            query_face: Optional query face for reference

        Returns:
            List of SearchResult objects ranked by similarity
        """
        # Convert similarity to distance threshold
        # similarity = 1 / (1 + distance)
        # distance = 1/similarity - 1
        distance_threshold = None
        if min_similarity is not None:
            if min_similarity <= 0 or min_similarity > 1:
                raise ValueError(f"min_similarity must be in (0, 1], got {min_similarity}")
            distance_threshold = (1.0 / min_similarity) - 1.0

        # Search in collection
        search_results = self.collection.search(
            query_embedding=embedding,
            k=max_results,
            threshold=distance_threshold
        )

        # Convert to SearchResult objects
        results = []
        for rank, (face_record, distance) in enumerate(search_results, start=1):
            # Convert to FaceDetection
            face = FaceDetection(
                bbox=BoundingBox.from_dict(face_record.bbox if isinstance(face_record.bbox, dict) else eval(face_record.bbox)),
                confidence=face_record.confidence,
                embedding=face_record.get_embedding(),
                image_path=Path(face_record.image_path),
                face_id=face_record.face_id,
                embedding_index=face_record.embedding_index
            )

            # Create SearchResult
            result = SearchResult(
                face=face,
                distance=distance,
                rank=rank
            )
            results.append(result)

        logger.debug(f"Found {len(results)} results")
        return results

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute L2 distance between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            L2 distance (lower is more similar)
        """
        return float(np.linalg.norm(emb1 - emb2))

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics.

        Returns:
            Dictionary with statistics
        """
        collection_stats = self.collection.get_stats()

        return {
            'collection_id': self.collection.collection_id,
            'total_faces': collection_stats['total_faces'],
            'active_faces': collection_stats['active_faces'],
            'embedding_dim': self.collection.embedding_dim,
            'model': self.model.__class__.__name__,
            'device': self.model.device
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SearchEngine("
            f"collection='{self.collection.collection_id}', "
            f"faces={self.collection.ntotal}, "
            f"model={self.model.__class__.__name__})"
        )
