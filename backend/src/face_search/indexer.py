"""Batch face indexer for processing image folders.

This module provides the BatchIndexer class for efficiently processing
large collections of images, detecting faces, and adding them to collections.
"""

from pathlib import Path
from typing import List, Optional, Union, Iterator, Callable
import logging
import time
import json
from tqdm import tqdm
import numpy as np

from .models import BaseFaceModel
from .storage import Collection
from .face import (
    FaceDetection,
    BoundingBox,
    Landmarks,
    FaceQuality,
    IndexingResult,
    BatchIndexingStats
)
from .deduplication import FaceDeduplicator, ImageHashCache

logger = logging.getLogger(__name__)


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def is_image_file(path: Path) -> bool:
    """Check if file is a supported image format.

    Args:
        path: Path to check

    Returns:
        True if file is a supported image
    """
    return path.suffix.lower() in IMAGE_EXTENSIONS


class BatchIndexer:
    """Batch indexer for efficiently processing image folders.

    Features:
    - Batch face detection and embedding generation
    - Automatic deduplication
    - Progress tracking with tqdm
    - Error handling and logging
    - Checkpoint/resume support
    - Configurable batch sizes for GPU efficiency

    Usage:
        indexer = BatchIndexer(
            model=face_model,
            collection=collection,
            batch_size=32
        )

        stats = indexer.index_folder(
            folder_path='./images',
            recursive=True,
            min_confidence=0.5
        )

        print(f"Indexed {stats.total_faces_indexed} faces from {stats.total_images} images")
    """

    def __init__(
        self,
        model: BaseFaceModel,
        collection: Collection,
        batch_size: int = 32,
        enable_deduplication: bool = True,
        dedup_window_size: int = 1000,
        checkpoint_interval: int = 100,
        checkpoint_path: Optional[Path] = None
    ):
        """Initialize batch indexer.

        Args:
            model: Face detection/embedding model
            collection: Collection to add faces to
            batch_size: Batch size for embedding generation
            enable_deduplication: Enable duplicate face detection
            dedup_window_size: Size of deduplication window
            checkpoint_interval: Save checkpoint every N images
            checkpoint_path: Path to save checkpoints (default: collection_dir/checkpoint.json)
        """
        self.model = model
        self.collection = collection
        self.batch_size = batch_size
        self.enable_deduplication = enable_deduplication

        # Initialize deduplication
        self.deduplicator = FaceDeduplicator(
            window_size=dedup_window_size
        ) if enable_deduplication else None

        self.image_hash_cache = ImageHashCache(use_perceptual=False)

        # Checkpoint settings
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path or (
            collection.collection_path / "indexing_checkpoint.json"
        )

        # State
        self._processed_images: set[str] = set()
        self._stats = BatchIndexingStats()

        logger.info(
            f"BatchIndexer initialized: batch_size={batch_size}, "
            f"dedup={enable_deduplication}"
        )

    def index_folder(
        self,
        folder_path: Union[str, Path],
        recursive: bool = True,
        min_confidence: float = 0.5,
        resume: bool = False,
        progress_callback: Optional[Callable[[IndexingResult], None]] = None
    ) -> BatchIndexingStats:
        """Index all images in a folder.

        Args:
            folder_path: Path to folder containing images
            recursive: Recursively process subdirectories
            min_confidence: Minimum face detection confidence
            resume: Resume from checkpoint if available
            progress_callback: Optional callback called after each image

        Returns:
            BatchIndexingStats with processing statistics

        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if not folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        # Load checkpoint if resuming
        if resume:
            self._load_checkpoint()
            logger.info(f"Resuming from checkpoint: {len(self._processed_images)} images already processed")

        # Find all image files
        logger.info(f"Scanning for images in {folder_path}...")
        image_paths = self._find_images(folder_path, recursive)

        # Filter out already processed images
        if resume:
            image_paths = [
                p for p in image_paths
                if str(p.resolve()) not in self._processed_images
            ]

        logger.info(f"Found {len(image_paths)} images to process")

        if not image_paths:
            logger.warning("No images to process")
            return self._stats

        # Reset stats for this run
        start_time = time.time()
        self._stats = BatchIndexingStats()

        # Process images with progress bar
        with tqdm(total=len(image_paths), desc="Indexing images", unit="img") as pbar:
            for i, image_path in enumerate(image_paths):
                try:
                    # Index single image
                    result = self.index_image(
                        image_path,
                        min_confidence=min_confidence
                    )

                    # Update stats
                    self._update_stats(result)

                    # Mark as processed
                    self._processed_images.add(str(image_path.resolve()))

                    # Call progress callback
                    if progress_callback:
                        progress_callback(result)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'faces': self._stats.total_faces_indexed,
                        'duplicates': self._stats.duplicates_skipped,
                        'errors': len(self._stats.errors)
                    })

                    # Save checkpoint periodically
                    if (i + 1) % self.checkpoint_interval == 0:
                        self._save_checkpoint()

                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    self._stats.errors.append(f"{image_path}: {str(e)}")
                    self._stats.failed_images += 1
                    pbar.update(1)

        # Final stats
        self._stats.processing_time = time.time() - start_time
        self._save_checkpoint()  # Save final checkpoint

        logger.info(
            f"Indexing complete: {self._stats.total_faces_indexed} faces from "
            f"{self._stats.successful_images} images in {self._stats.processing_time:.1f}s "
            f"({self._stats.faces_per_second:.1f} faces/sec)"
        )

        return self._stats

    def index_image(
        self,
        image_path: Union[str, Path],
        min_confidence: float = 0.5
    ) -> IndexingResult:
        """Index a single image.

        Args:
            image_path: Path to image file
            min_confidence: Minimum face detection confidence

        Returns:
            IndexingResult with details of indexing operation
        """
        image_path = Path(image_path)
        result = IndexingResult(image_path=image_path)

        try:
            # Detect faces
            detected_faces = self.model.detect_faces(
                str(image_path),
                min_confidence=min_confidence
            )

            result.faces_found = len(detected_faces)

            if not detected_faces:
                logger.debug(f"No faces found in {image_path}")
                return result

            # Convert to FaceDetection objects
            face_detections = []
            for df in detected_faces:
                # Create FaceDetection from model output
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
                    image_path=image_path,
                    yaw=df.yaw,
                    pitch=df.pitch,
                    roll=df.roll
                )

                # Check for duplicates
                if self.deduplicator and self.deduplicator.is_duplicate(face):
                    logger.debug(f"Skipping duplicate face in {image_path}")
                    continue

                face_detections.append(face)

                # Add to deduplicator window
                if self.deduplicator:
                    self.deduplicator.add(face)

            # Batch add faces to collection
            if face_detections:
                face_ids = self._add_faces_batch(face_detections)
                result.faces_indexed = len(face_ids)
                result.face_ids = face_ids

        except Exception as e:
            logger.error(f"Error indexing {image_path}: {e}")
            result.errors.append(str(e))

        return result

    def index_images_batch(
        self,
        image_paths: List[Union[str, Path]],
        min_confidence: float = 0.5,
        show_progress: bool = True
    ) -> List[IndexingResult]:
        """Index multiple images.

        Args:
            image_paths: List of image paths
            min_confidence: Minimum face detection confidence
            show_progress: Show progress bar

        Returns:
            List of IndexingResult objects
        """
        results = []

        iterator = tqdm(image_paths, desc="Indexing") if show_progress else image_paths

        for image_path in iterator:
            result = self.index_image(image_path, min_confidence)
            results.append(result)

        return results

    def _add_faces_batch(self, faces: List[FaceDetection]) -> List[int]:
        """Add multiple faces to collection.

        Args:
            faces: List of face detections

        Returns:
            List of face IDs
        """
        # Prepare data for batch add
        faces_data = []
        embeddings = []

        for face in faces:
            faces_data.append({
                'image_path': str(face.image_path),
                'bbox': face.bbox.to_dict(),
                'confidence': face.confidence,
                'yaw': face.yaw,
                'pitch': face.pitch,
                'roll': face.roll
            })
            embeddings.append(face.embedding)

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Batch add to collection
        face_ids = self.collection.add_faces_batch(faces_data, embeddings_array)

        return face_ids

    def _find_images(
        self,
        folder_path: Path,
        recursive: bool = True
    ) -> List[Path]:
        """Find all image files in folder.

        Args:
            folder_path: Folder to search
            recursive: Search subdirectories

        Returns:
            List of image file paths
        """
        if recursive:
            # Recursively find all images
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(folder_path.rglob(f'*{ext}'))
                images.extend(folder_path.rglob(f'*{ext.upper()}'))
        else:
            # Only top level
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(folder_path.glob(f'*{ext}'))
                images.extend(folder_path.glob(f'*{ext.upper()}'))

        # Sort for consistent ordering
        return sorted(images)

    def _update_stats(self, result: IndexingResult):
        """Update batch statistics from indexing result.

        Args:
            result: IndexingResult to process
        """
        self._stats.total_images += 1
        self._stats.total_faces_found += result.faces_found
        self._stats.total_faces_indexed += result.faces_indexed

        if result.success:
            self._stats.successful_images += 1
        elif result.faces_indexed == 0:
            self._stats.failed_images += 1

        if result.errors:
            self._stats.errors.extend(result.errors)

        # Update duplicates count
        if self.deduplicator:
            self._stats.duplicates_skipped = self.deduplicator.duplicates_found

    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        try:
            checkpoint_data = {
                'processed_images': list(self._processed_images),
                'stats': self._stats.to_dict(),
                'timestamp': time.time()
            }

            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.debug(f"Checkpoint saved: {len(self._processed_images)} images")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Load checkpoint from disk."""
        try:
            if not self.checkpoint_path.exists():
                logger.debug("No checkpoint found")
                return

            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            self._processed_images = set(checkpoint_data.get('processed_images', []))

            logger.info(f"Checkpoint loaded: {len(self._processed_images)} images already processed")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    def clear_checkpoint(self):
        """Clear checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint cleared")

        self._processed_images.clear()

    def get_stats(self) -> BatchIndexingStats:
        """Get current statistics.

        Returns:
            Current BatchIndexingStats
        """
        return self._stats

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BatchIndexer("
            f"batch_size={self.batch_size}, "
            f"dedup={self.enable_deduplication}, "
            f"processed={len(self._processed_images)} images)"
        )
