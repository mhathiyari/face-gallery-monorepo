"""Deduplication utilities for face search system.

This module provides functions for detecting and handling duplicate images and faces:
- Image hashing for exact/near-duplicate image detection
- Face deduplication based on bounding box overlap and embedding similarity
- Efficient duplicate checking during batch indexing
"""

import hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
from PIL import Image
import logging

from .face import BoundingBox, FaceDetection

logger = logging.getLogger(__name__)


def compute_image_hash(
    image_path: Union[str, Path],
    hash_algorithm: str = 'md5',
    use_perceptual: bool = False,
    thumbnail_size: Tuple[int, int] = (8, 8)
) -> str:
    """Compute hash of an image for duplicate detection.

    Args:
        image_path: Path to image file
        hash_algorithm: Hash algorithm ('md5', 'sha256', etc.)
        use_perceptual: If True, use perceptual hashing (detects similar images)
                       If False, use file-based hashing (detects exact duplicates)
        thumbnail_size: Size for perceptual hash thumbnail (only if use_perceptual=True)

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If image doesn't exist
        IOError: If image can't be read
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        if use_perceptual:
            # Perceptual hash: hash of downsampled grayscale image
            # This detects visually similar images (crops, resizes, etc.)
            img = Image.open(image_path)

            # Convert to grayscale and resize
            img = img.convert('L').resize(thumbnail_size, Image.LANCZOS)

            # Get pixel data
            pixels = np.array(img).flatten()

            # Compute average
            avg = pixels.mean()

            # Create binary hash: 1 if pixel > average, 0 otherwise
            binary = ''.join('1' if p > avg else '0' for p in pixels)

            # Convert binary string to hex
            hash_value = hex(int(binary, 2))[2:]

        else:
            # File-based hash: hash of raw file bytes
            # This only detects exact duplicate files
            hasher = hashlib.new(hash_algorithm)

            with open(image_path, 'rb') as f:
                # Read in chunks for memory efficiency
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)

            hash_value = hasher.hexdigest()

        logger.debug(f"Computed {'perceptual' if use_perceptual else 'file'} hash for {image_path}: {hash_value[:16]}...")
        return hash_value

    except Exception as e:
        logger.error(f"Error computing hash for {image_path}: {e}")
        raise IOError(f"Failed to compute hash: {e}")


def compute_perceptual_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two perceptual hashes.

    Args:
        hash1: First hash (hex string)
        hash2: Second hash (hex string)

    Returns:
        Hamming distance (number of differing bits)
    """
    # Convert hex to binary
    try:
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        # Count differing bits
        return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
    except ValueError:
        logger.warning(f"Invalid hash format: {hash1[:8]}... or {hash2[:8]}...")
        return float('inf')


def bbox_overlap(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculate overlap between two bounding boxes using IoU.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Intersection over Union (IoU) score between 0 and 1
    """
    return bbox1.overlap_iou(bbox2)


def compute_embedding_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    metric: str = 'cosine'
) -> float:
    """Compute similarity between two face embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Similarity score (higher means more similar)
        - For cosine: 0-1 (1 = identical)
        - For euclidean: L2 distance (lower is more similar)
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}"
        )

    if metric == 'cosine':
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    elif metric == 'euclidean':
        # L2 distance (lower is better)
        return float(np.linalg.norm(embedding1 - embedding2))

    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")


def is_duplicate_face(
    face1: FaceDetection,
    face2: FaceDetection,
    bbox_threshold: float = 0.5,
    embedding_threshold: float = 0.95,
    check_same_image: bool = True
) -> bool:
    """Check if two face detections are duplicates.

    Two faces are considered duplicates if:
    1. They're from the same image (optional check)
    2. Their bounding boxes have significant overlap (IoU > threshold)
    3. Their embeddings are very similar (cosine similarity > threshold)

    Args:
        face1: First face detection
        face2: Second face detection
        bbox_threshold: Minimum IoU to consider faces overlapping (0-1)
        embedding_threshold: Minimum cosine similarity to consider faces same (0-1)
        check_same_image: If True, only check faces from same image

    Returns:
        True if faces are duplicates, False otherwise
    """
    # Check if from same image
    if check_same_image:
        if face1.image_path != face2.image_path:
            return False

        if face1.image_path is None or face2.image_path is None:
            logger.warning("Faces have no image_path, skipping same-image check")
            return False

    # Check bounding box overlap
    iou = bbox_overlap(face1.bbox, face2.bbox)
    if iou < bbox_threshold:
        return False

    # Check embedding similarity (if both have embeddings)
    if face1.embedding is not None and face2.embedding is not None:
        similarity = compute_embedding_similarity(
            face1.embedding,
            face2.embedding,
            metric='cosine'
        )

        if similarity < embedding_threshold:
            return False

        logger.debug(
            f"Duplicate detected: IoU={iou:.3f}, similarity={similarity:.3f}"
        )
        return True

    else:
        # If no embeddings, only use bbox overlap
        logger.debug(f"Duplicate detected (bbox only): IoU={iou:.3f}")
        return True


class ImageHashCache:
    """Cache for image hashes to avoid recomputing.

    Usage:
        cache = ImageHashCache()
        hash1 = cache.get_hash('/path/to/image.jpg')
        hash2 = cache.get_hash('/path/to/image.jpg')  # Retrieved from cache
    """

    def __init__(
        self,
        use_perceptual: bool = False,
        perceptual_threshold: int = 5
    ):
        """Initialize hash cache.

        Args:
            use_perceptual: Use perceptual hashing
            perceptual_threshold: Max Hamming distance for perceptual duplicates
        """
        self._cache: dict[Path, str] = {}
        self.use_perceptual = use_perceptual
        self.perceptual_threshold = perceptual_threshold
        self._hits = 0
        self._misses = 0

    def get_hash(self, image_path: Union[str, Path]) -> str:
        """Get hash for image (from cache or compute).

        Args:
            image_path: Path to image

        Returns:
            Image hash string
        """
        image_path = Path(image_path)

        if image_path in self._cache:
            self._hits += 1
            return self._cache[image_path]

        self._misses += 1
        hash_value = compute_image_hash(
            image_path,
            use_perceptual=self.use_perceptual
        )
        self._cache[image_path] = hash_value
        return hash_value

    def is_duplicate_image(
        self,
        image_path1: Union[str, Path],
        image_path2: Union[str, Path]
    ) -> bool:
        """Check if two images are duplicates.

        Args:
            image_path1: First image path
            image_path2: Second image path

        Returns:
            True if images are duplicates
        """
        hash1 = self.get_hash(image_path1)
        hash2 = self.get_hash(image_path2)

        if self.use_perceptual:
            distance = compute_perceptual_distance(hash1, hash2)
            return distance <= self.perceptual_threshold
        else:
            return hash1 == hash2

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ImageHashCache(size={self.size}, "
            f"hits={self._hits}, misses={self._misses}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


class FaceDeduplicator:
    """Efficient face deduplication for batch processing.

    Maintains a sliding window of recent faces to check for duplicates
    without comparing against all previously seen faces.

    Usage:
        dedup = FaceDeduplicator(window_size=100)

        for face in faces:
            if not dedup.is_duplicate(face):
                # Process non-duplicate face
                dedup.add(face)
    """

    def __init__(
        self,
        window_size: int = 1000,
        bbox_threshold: float = 0.5,
        embedding_threshold: float = 0.95
    ):
        """Initialize deduplicator.

        Args:
            window_size: Number of recent faces to check against
            bbox_threshold: Minimum IoU for duplicate detection
            embedding_threshold: Minimum cosine similarity for duplicate detection
        """
        self.window_size = window_size
        self.bbox_threshold = bbox_threshold
        self.embedding_threshold = embedding_threshold

        self._faces: List[FaceDetection] = []
        self._duplicates_found = 0

    def add(self, face: FaceDetection):
        """Add a face to the deduplication window.

        Args:
            face: Face detection to add
        """
        self._faces.append(face)

        # Keep only recent faces
        if len(self._faces) > self.window_size:
            self._faces.pop(0)

    def is_duplicate(self, face: FaceDetection) -> bool:
        """Check if face is a duplicate of any recent face.

        Args:
            face: Face to check

        Returns:
            True if face is a duplicate
        """
        for existing_face in reversed(self._faces):  # Check recent faces first
            if is_duplicate_face(
                face,
                existing_face,
                bbox_threshold=self.bbox_threshold,
                embedding_threshold=self.embedding_threshold,
                check_same_image=True
            ):
                self._duplicates_found += 1
                return True

        return False

    def clear(self):
        """Clear the deduplication window."""
        self._faces.clear()
        self._duplicates_found = 0

    @property
    def duplicates_found(self) -> int:
        """Get number of duplicates found."""
        return self._duplicates_found

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FaceDeduplicator(window_size={self.window_size}, "
            f"current_size={len(self._faces)}, "
            f"duplicates_found={self._duplicates_found})"
        )
