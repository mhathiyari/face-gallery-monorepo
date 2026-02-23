"""Face data classes for the face search system.

This module defines the core data structures for representing faces,
detections, and search results throughout the system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box for a detected face.

    Attributes:
        x1: Left x coordinate
        y1: Top y coordinate
        x2: Right x coordinate
        y2: Bottom y coordinate
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Get center point (x, y) of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format.

        Returns:
            Dictionary with x1, y1, x2, y2 keys
        """
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        """Create BoundingBox from dictionary.

        Args:
            data: Dictionary with x1, y1, x2, y2 keys

        Returns:
            BoundingBox instance
        """
        return cls(
            x1=data['x1'],
            y1=data['y1'],
            x2=data['x2'],
            y2=data['y2']
        )

    def overlap_iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union (IoU) with another box.

        Args:
            other: Another BoundingBox

        Returns:
            IoU score between 0 and 1
        """
        # Calculate intersection
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def __repr__(self) -> str:
        """String representation."""
        return f"BoundingBox(x1={self.x1:.1f}, y1={self.y1:.1f}, x2={self.x2:.1f}, y2={self.y2:.1f})"


@dataclass
class FaceQuality:
    """Quality metrics for a detected face.

    Attributes:
        sharpness: Image sharpness score (0-1, higher is better)
        brightness: Brightness score (0-1, 0.5 is ideal)
        size: Relative size score (0-1, larger is better)
        frontal: How frontal the face is (0-1, higher is better)
        overall: Overall quality score (0-1)
    """
    sharpness: float = 0.0
    brightness: float = 0.0
    size: float = 0.0
    frontal: float = 1.0  # Default to frontal if not computed
    overall: float = 0.0

    def __post_init__(self):
        """Compute overall quality if not provided."""
        if self.overall == 0.0:
            # Weighted average of quality metrics
            self.overall = (
                self.sharpness * 0.3 +
                self.brightness * 0.2 +
                self.size * 0.2 +
                self.frontal * 0.3
            )


@dataclass
class Landmarks:
    """Facial landmarks (keypoints).

    Attributes:
        left_eye: Left eye coordinates (x, y)
        right_eye: Right eye coordinates (x, y)
        nose: Nose tip coordinates (x, y)
        left_mouth: Left mouth corner coordinates (x, y)
        right_mouth: Right mouth corner coordinates (x, y)
    """
    left_eye: Optional[tuple[float, float]] = None
    right_eye: Optional[tuple[float, float]] = None
    nose: Optional[tuple[float, float]] = None
    left_mouth: Optional[tuple[float, float]] = None
    right_mouth: Optional[tuple[float, float]] = None

    @classmethod
    def from_array(cls, landmarks: np.ndarray) -> 'Landmarks':
        """Create Landmarks from numpy array.

        Args:
            landmarks: Array of shape (5, 2) with [left_eye, right_eye, nose, left_mouth, right_mouth]

        Returns:
            Landmarks instance
        """
        if landmarks.shape != (5, 2):
            raise ValueError(f"Expected landmarks shape (5, 2), got {landmarks.shape}")

        return cls(
            left_eye=tuple(landmarks[0]),
            right_eye=tuple(landmarks[1]),
            nose=tuple(landmarks[2]),
            left_mouth=tuple(landmarks[3]),
            right_mouth=tuple(landmarks[4])
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            Array of shape (5, 2)
        """
        return np.array([
            self.left_eye or (0, 0),
            self.right_eye or (0, 0),
            self.nose or (0, 0),
            self.left_mouth or (0, 0),
            self.right_mouth or (0, 0)
        ], dtype=np.float32)


@dataclass
class FaceDetection:
    """A detected face with all metadata.

    This is the primary data structure for faces in the indexing pipeline.

    Attributes:
        bbox: Bounding box of the face
        confidence: Detection confidence (0-1)
        embedding: Face embedding vector (512-dim)
        landmarks: Facial landmarks (optional)
        quality: Face quality metrics (optional)
        image_path: Path to source image
        face_id: Database face ID (set after indexing)
        embedding_index: Index in FAISS (set after indexing)
        yaw: Face yaw angle - left/right rotation (optional)
        pitch: Face pitch angle - up/down tilt (optional)
        roll: Face roll angle - head tilt (optional)
    """
    bbox: BoundingBox
    confidence: float
    embedding: Optional[np.ndarray] = None
    landmarks: Optional[Landmarks] = None
    quality: Optional[FaceQuality] = None
    image_path: Optional[Path] = None
    face_id: Optional[int] = None
    embedding_index: Optional[int] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    def __post_init__(self):
        """Validate data after initialization."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

        if self.embedding is not None:
            if not isinstance(self.embedding, np.ndarray):
                raise TypeError(f"Embedding must be numpy array, got {type(self.embedding)}")
            if self.embedding.ndim != 1:
                raise ValueError(f"Embedding must be 1-dimensional, got shape {self.embedding.shape}")

    @property
    def is_indexed(self) -> bool:
        """Check if face has been indexed."""
        return self.face_id is not None and self.embedding_index is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation (without embedding data)
        """
        return {
            'bbox': self.bbox.to_dict(),
            'confidence': self.confidence,
            'has_embedding': self.embedding is not None,
            'embedding_dim': self.embedding.shape[0] if self.embedding is not None else None,
            'has_landmarks': self.landmarks is not None,
            'quality': {
                'overall': self.quality.overall,
                'sharpness': self.quality.sharpness,
                'brightness': self.quality.brightness,
                'size': self.quality.size,
                'frontal': self.quality.frontal
            } if self.quality else None,
            'image_path': str(self.image_path) if self.image_path else None,
            'face_id': self.face_id,
            'embedding_index': self.embedding_index,
            'is_indexed': self.is_indexed
        }

    def __repr__(self) -> str:
        """String representation."""
        indexed = " [INDEXED]" if self.is_indexed else ""
        quality_str = f", quality={self.quality.overall:.2f}" if self.quality else ""
        return (
            f"FaceDetection(confidence={self.confidence:.2f}, "
            f"bbox={self.bbox}{quality_str}{indexed})"
        )


@dataclass
class SearchResult:
    """Result from a face search query.

    Attributes:
        face: The matched face
        distance: L2 distance from query (lower is more similar)
        similarity: Similarity score 0-1 (higher is more similar)
        rank: Rank in search results (1-indexed)
    """
    face: FaceDetection
    distance: float
    similarity: float = field(init=False)
    rank: int = 1

    def __post_init__(self):
        """Compute similarity from distance."""
        # Convert L2 distance to similarity score
        # Using formula: similarity = 1 / (1 + distance)
        self.similarity = 1.0 / (1.0 + self.distance)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            'rank': self.rank,
            'distance': self.distance,
            'similarity': self.similarity,
            'confidence': self.face.confidence,
            'face': self.face.to_dict()
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SearchResult(rank={self.rank}, "
            f"similarity={self.similarity:.3f}, "
            f"distance={self.distance:.3f})"
        )


@dataclass
class IndexingResult:
    """Result from indexing an image.

    Attributes:
        image_path: Path to the indexed image
        faces_found: Number of faces detected
        faces_indexed: Number of faces successfully indexed
        face_ids: List of database face IDs
        errors: List of error messages (if any)
    """
    image_path: Path
    faces_found: int = 0
    faces_indexed: int = 0
    face_ids: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if indexing was successful."""
        return self.faces_indexed > 0 and len(self.errors) == 0

    @property
    def partial_success(self) -> bool:
        """Check if indexing had partial success."""
        return self.faces_indexed > 0 and len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            'image_path': str(self.image_path),
            'faces_found': self.faces_found,
            'faces_indexed': self.faces_indexed,
            'face_ids': self.face_ids,
            'errors': self.errors,
            'success': self.success,
            'partial_success': self.partial_success
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "✓" if self.success else "⚠" if self.partial_success else "✗"
        return (
            f"IndexingResult({status} {self.faces_indexed}/{self.faces_found} faces, "
            f"{len(self.errors)} errors)"
        )


@dataclass
class BatchIndexingStats:
    """Statistics from a batch indexing operation.

    Attributes:
        total_images: Total images processed
        successful_images: Images with at least one face indexed
        failed_images: Images that failed completely
        total_faces_found: Total faces detected
        total_faces_indexed: Total faces successfully indexed
        duplicates_skipped: Number of duplicate faces skipped
        errors: List of error messages
        processing_time: Total processing time in seconds
    """
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    total_faces_found: int = 0
    total_faces_indexed: int = 0
    duplicates_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_images / self.total_images if self.total_images > 0 else 0.0

    @property
    def faces_per_image(self) -> float:
        """Average faces per image."""
        return self.total_faces_found / self.total_images if self.total_images > 0 else 0.0

    @property
    def images_per_second(self) -> float:
        """Processing speed in images/second."""
        return self.total_images / self.processing_time if self.processing_time > 0 else 0.0

    @property
    def faces_per_second(self) -> float:
        """Processing speed in faces/second."""
        return self.total_faces_indexed / self.processing_time if self.processing_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            'total_images': self.total_images,
            'successful_images': self.successful_images,
            'failed_images': self.failed_images,
            'total_faces_found': self.total_faces_found,
            'total_faces_indexed': self.total_faces_indexed,
            'duplicates_skipped': self.duplicates_skipped,
            'success_rate': self.success_rate,
            'faces_per_image': self.faces_per_image,
            'images_per_second': self.images_per_second,
            'faces_per_second': self.faces_per_second,
            'processing_time': self.processing_time,
            'errors_count': len(self.errors)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BatchIndexingStats("
            f"images={self.successful_images}/{self.total_images}, "
            f"faces={self.total_faces_indexed}, "
            f"speed={self.faces_per_second:.1f} faces/sec, "
            f"errors={len(self.errors)})"
        )
