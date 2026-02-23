"""Utilities for loading and managing test datasets.

This module provides helper functions for working with test datasets like CelebA
for integration testing and benchmarking.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import random
from PIL import Image
import tempfile
import shutil


def create_synthetic_dataset(
    output_dir: Path,
    num_images: int = 100,
    num_identities: int = 20,
    faces_per_identity: int = 5,
    image_size: Tuple[int, int] = (200, 200)
) -> Path:
    """Create a synthetic face dataset for testing.

    Creates a simple synthetic dataset with multiple images per identity.
    Each identity gets a unique color, and faces are created as colored rectangles.

    Args:
        output_dir: Directory to save images
        num_images: Total number of images to generate
        num_identities: Number of unique identities
        faces_per_identity: Average faces per identity
        image_size: Size of generated images

    Returns:
        Path to the created dataset directory
    """
    dataset_dir = output_dir / 'synthetic_faces'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Generate colors for each identity
    colors = []
    for i in range(num_identities):
        # Generate distinct colors
        hue = (i * 360 // num_identities) % 360
        # Convert HSV to RGB (simplified)
        r = int(255 * (1 - abs((hue / 60) % 2 - 1)))
        g = int(255 * (1 - abs(((hue - 120) / 60) % 2 - 1)))
        b = int(255 * (1 - abs(((hue - 240) / 60) % 2 - 1)))
        colors.append((max(100, r), max(100, g), max(100, b)))

    # Create images
    image_count = 0
    for identity_id in range(num_identities):
        # Number of images for this identity
        n_images = min(faces_per_identity, num_images - image_count)

        for img_idx in range(n_images):
            # Create image with identity's color
            img = Image.new('RGB', image_size, color=colors[identity_id])

            # Save image
            image_path = dataset_dir / f'identity_{identity_id:03d}_image_{img_idx:03d}.jpg'
            img.save(image_path)

            image_count += 1
            if image_count >= num_images:
                break

        if image_count >= num_images:
            break

    return dataset_dir


def find_test_images(
    directory: Path,
    max_images: Optional[int] = None,
    shuffle: bool = True,
    extensions: Optional[List[str]] = None
) -> List[Path]:
    """Find image files in a directory.

    Args:
        directory: Directory to search
        max_images: Maximum number of images to return
        shuffle: Whether to shuffle the results
        extensions: List of file extensions to include (default: common image formats)

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # Convert to lowercase for case-insensitive matching
    extensions = [ext.lower() for ext in extensions]

    # Find all images
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f'*{ext}'))
        images.extend(directory.rglob(f'*{ext.upper()}'))

    # Remove duplicates and sort
    images = sorted(set(images))

    # Shuffle if requested
    if shuffle:
        random.shuffle(images)

    # Limit number of images
    if max_images is not None:
        images = images[:max_images]

    return images


def get_identity_from_filename(image_path: Path) -> Optional[str]:
    """Extract identity/person ID from filename.

    Supports common naming conventions:
    - identity_001_image_002.jpg -> identity_001
    - person_john_doe_01.jpg -> person_john_doe
    - 0001_02.jpg -> 0001

    Args:
        image_path: Path to image file

    Returns:
        Identity string or None if not extractable
    """
    filename = image_path.stem  # Filename without extension

    # Try common patterns
    parts = filename.split('_')

    if len(parts) >= 2:
        # Check if first part looks like identity ID
        if parts[0] in ['identity', 'person', 'id']:
            return '_'.join(parts[:2])  # e.g., identity_001
        elif parts[0].isdigit():
            return parts[0]  # e.g., 0001

    # Fallback: use first part
    return parts[0] if parts else None


def group_images_by_identity(
    images: List[Path]
) -> dict[str, List[Path]]:
    """Group images by identity/person.

    Args:
        images: List of image paths

    Returns:
        Dictionary mapping identity to list of image paths
    """
    groups = {}

    for image_path in images:
        identity = get_identity_from_filename(image_path)
        if identity:
            if identity not in groups:
                groups[identity] = []
            groups[identity].append(image_path)

    return groups


def create_train_test_split(
    images: List[Path],
    test_ratio: float = 0.2,
    min_images_per_identity: int = 2
) -> Tuple[List[Path], List[Path]]:
    """Split images into train and test sets.

    Ensures each identity appears in both train and test sets if possible.

    Args:
        images: List of image paths
        test_ratio: Ratio of images to use for testing
        min_images_per_identity: Minimum images needed per identity for splitting

    Returns:
        Tuple of (train_images, test_images)
    """
    # Group by identity
    groups = group_images_by_identity(images)

    train_images = []
    test_images = []

    for identity, identity_images in groups.items():
        if len(identity_images) < min_images_per_identity:
            # Too few images, add all to train
            train_images.extend(identity_images)
        else:
            # Split this identity's images
            n_test = max(1, int(len(identity_images) * test_ratio))
            shuffled = identity_images.copy()
            random.shuffle(shuffled)

            test_images.extend(shuffled[:n_test])
            train_images.extend(shuffled[n_test:])

    return train_images, test_images


class TempDataset:
    """Context manager for temporary test datasets.

    Usage:
        with TempDataset(num_images=100) as dataset_dir:
            # Use dataset_dir for testing
            images = find_test_images(dataset_dir)
        # Dataset is automatically cleaned up
    """

    def __init__(
        self,
        num_images: int = 100,
        num_identities: int = 20
    ):
        """Initialize temporary dataset.

        Args:
            num_images: Number of images to generate
            num_identities: Number of unique identities
        """
        self.num_images = num_images
        self.num_identities = num_identities
        self.temp_dir = None
        self.dataset_dir = None

    def __enter__(self) -> Path:
        """Create temporary dataset."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dataset_dir = create_synthetic_dataset(
            self.temp_dir,
            num_images=self.num_images,
            num_identities=self.num_identities
        )
        return self.dataset_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary dataset."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def validate_dataset(directory: Path) -> dict:
    """Validate a dataset and return statistics.

    Args:
        directory: Dataset directory

    Returns:
        Dictionary with validation results and statistics
    """
    images = find_test_images(directory, shuffle=False)
    groups = group_images_by_identity(images)

    # Compute statistics
    images_per_identity = [len(imgs) for imgs in groups.values()]

    return {
        'total_images': len(images),
        'total_identities': len(groups),
        'avg_images_per_identity': sum(images_per_identity) / len(images_per_identity) if images_per_identity else 0,
        'min_images_per_identity': min(images_per_identity) if images_per_identity else 0,
        'max_images_per_identity': max(images_per_identity) if images_per_identity else 0,
        'identities': list(groups.keys())
    }
