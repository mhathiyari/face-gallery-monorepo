"""Face clustering utilities using DBSCAN and other algorithms.

This module provides functions for clustering faces based on embedding similarity,
useful for grouping faces of the same person or finding identity clusters.
"""

from typing import List, Dict, Optional, Set, Tuple, Any
import numpy as np
import logging
from collections import defaultdict

from ..face import FaceDetection
from ..storage import Collection

logger = logging.getLogger(__name__)


def cluster_faces(
    faces: List[FaceDetection],
    eps: float = 0.6,
    min_samples: int = 2,
    metric: str = 'euclidean'
) -> Dict[int, List[FaceDetection]]:
    """Cluster faces using DBSCAN algorithm.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
    faces based on embedding similarity. Faces that are close in embedding space
    are grouped together.

    Args:
        faces: List of face detections with embeddings
        eps: Maximum distance between two samples to be considered neighbors
             For L2 distance on normalized embeddings, typical values: 0.4-0.8
        min_samples: Minimum number of samples in a neighborhood for a point
                    to be considered a core point
        metric: Distance metric ('euclidean' or 'cosine')

    Returns:
        Dictionary mapping cluster_id to list of faces
        cluster_id = -1 represents noise (faces that don't belong to any cluster)

    Raises:
        ValueError: If faces have no embeddings or inconsistent embedding dimensions
    """
    if not faces:
        return {}

    # Validate embeddings
    embeddings = []
    for face in faces:
        if face.embedding is None:
            raise ValueError("All faces must have embeddings for clustering")
        embeddings.append(face.embedding)

    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Verify consistent dimensions
    if len(set(emb.shape[0] for emb in embeddings)) > 1:
        raise ValueError("All embeddings must have the same dimension")

    logger.info(f"Clustering {len(faces)} faces with eps={eps}, min_samples={min_samples}")

    # Run DBSCAN clustering
    labels = _dbscan(embeddings_array, eps=eps, min_samples=min_samples, metric=metric)

    # Group faces by cluster
    clusters = defaultdict(list)
    for face, label in zip(faces, labels):
        clusters[int(label)].append(face)

    logger.info(
        f"Found {len([k for k in clusters.keys() if k != -1])} clusters "
        f"(noise: {len(clusters.get(-1, []))} faces)"
    )

    return dict(clusters)


def _dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Simple DBSCAN implementation.

    Args:
        X: Array of embeddings, shape (n_samples, n_features)
        eps: Maximum distance between neighbors
        min_samples: Minimum cluster size
        metric: Distance metric

    Returns:
        Array of cluster labels, shape (n_samples,)
        Label -1 indicates noise
    """
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1, dtype=int)
    cluster_id = 0

    # Compute pairwise distances
    if metric == 'euclidean':
        distances = _pairwise_euclidean(X)
    elif metric == 'cosine':
        distances = _pairwise_cosine_distance(X)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Track visited points
    visited = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        if visited[i]:
            continue

        visited[i] = True

        # Find neighbors
        neighbors = np.where(distances[i] <= eps)[0]

        if len(neighbors) < min_samples:
            # Mark as noise
            labels[i] = -1
        else:
            # Start new cluster
            labels[i] = cluster_id
            _expand_cluster(i, neighbors, cluster_id, labels, visited, distances, eps, min_samples)
            cluster_id += 1

    return labels


def _expand_cluster(
    point_idx: int,
    neighbors: np.ndarray,
    cluster_id: int,
    labels: np.ndarray,
    visited: np.ndarray,
    distances: np.ndarray,
    eps: float,
    min_samples: int
):
    """Expand cluster by adding density-reachable points."""
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True

            # Find neighbors of this neighbor
            neighbor_neighbors = np.where(distances[neighbor_idx] <= eps)[0]

            if len(neighbor_neighbors) >= min_samples:
                # Add new neighbors to expand search
                neighbors = np.unique(np.concatenate([neighbors, neighbor_neighbors]))

        # Add to cluster if not yet assigned
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id

        i += 1


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances.

    Args:
        X: Array of shape (n_samples, n_features)

    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    # Using the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    xx = np.sum(X ** 2, axis=1)[:, np.newaxis]
    yy = xx.T
    xy = np.dot(X, X.T)
    distances = np.sqrt(np.maximum(xx + yy - 2 * xy, 0))
    return distances


def _pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distances.

    Args:
        X: Array of shape (n_samples, n_features)

    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    # Normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / np.maximum(norms, 1e-10)

    # Compute cosine similarity
    similarity = np.dot(X_normalized, X_normalized.T)

    # Convert to distance (1 - similarity)
    distance = 1 - similarity

    return distance


def get_identity_cluster(
    collection: Collection,
    face_id: int,
    eps: float = 0.6,
    min_samples: int = 2
) -> List[FaceDetection]:
    """Get cluster of faces representing the same identity as a given face.

    Args:
        collection: Face collection
        face_id: ID of the face to find cluster for
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        List of faces in the same identity cluster

    Raises:
        ValueError: If face_id not found
    """
    logger.info(f"Finding identity cluster for face_id={face_id}")

    # Get all faces from collection
    all_face_records = collection.metadata.get_all_faces(include_deleted=False)

    if not all_face_records:
        return []

    # Convert to FaceDetection objects
    from ..face import BoundingBox
    from pathlib import Path

    faces = []
    query_face_idx = None

    for i, record in enumerate(all_face_records):
        face = FaceDetection(
            bbox=BoundingBox.from_dict(record.bbox if isinstance(record.bbox, dict) else eval(record.bbox)),
            confidence=record.confidence,
            embedding=record.get_embedding(),
            image_path=Path(record.image_path),
            face_id=record.face_id,
            embedding_index=record.embedding_index
        )
        faces.append(face)

        if record.face_id == face_id:
            query_face_idx = i

    if query_face_idx is None:
        raise ValueError(f"Face {face_id} not found in collection")

    # Cluster all faces
    clusters = cluster_faces(faces, eps=eps, min_samples=min_samples)

    # Find which cluster contains the query face
    query_cluster_id = None
    for cluster_id, faces_in_cluster in clusters.items():
        if any(f.face_id == face_id for f in faces_in_cluster):
            query_cluster_id = cluster_id
            break

    if query_cluster_id is None or query_cluster_id == -1:
        # Face is noise, return just the face itself
        logger.info(f"Face {face_id} is not part of any cluster (noise)")
        return [faces[query_face_idx]]

    # Return all faces in the cluster
    faces_in_cluster = clusters[query_cluster_id]
    logger.info(f"Found {len(faces_in_cluster)} faces in identity cluster for face {face_id}")

    return faces_in_cluster


def cluster_by_image(
    faces: List[FaceDetection]
) -> Dict[str, List[FaceDetection]]:
    """Group faces by their source image.

    Args:
        faces: List of face detections

    Returns:
        Dictionary mapping image path to list of faces from that image
    """
    clusters = defaultdict(list)

    for face in faces:
        image_key = str(face.image_path) if face.image_path else 'unknown'
        clusters[image_key].append(face)

    return dict(clusters)


def find_representative_faces(
    cluster: List[FaceDetection],
    method: str = 'centroid',
    n_representatives: int = 1
) -> List[FaceDetection]:
    """Find representative face(s) for a cluster.

    Args:
        cluster: List of faces in cluster
        method: Method to find representatives:
                - 'centroid': Face closest to cluster centroid
                - 'highest_quality': Face with highest quality score
                - 'highest_confidence': Face with highest detection confidence
        n_representatives: Number of representatives to return

    Returns:
        List of representative faces

    Raises:
        ValueError: If cluster is empty or method is unknown
    """
    if not cluster:
        raise ValueError("Cannot find representative of empty cluster")

    if method == 'centroid':
        # Compute cluster centroid
        embeddings = np.array([f.embedding for f in cluster])
        centroid = embeddings.mean(axis=0)

        # Find faces closest to centroid
        distances = [np.linalg.norm(f.embedding - centroid) for f in cluster]
        sorted_indices = np.argsort(distances)

        return [cluster[i] for i in sorted_indices[:n_representatives]]

    elif method == 'highest_quality':
        # Sort by quality score
        faces_with_quality = [f for f in cluster if f.quality is not None]
        if not faces_with_quality:
            # Fallback to highest confidence if no quality scores
            return find_representative_faces(cluster, method='highest_confidence', n_representatives=n_representatives)

        sorted_faces = sorted(faces_with_quality, key=lambda f: f.quality.overall, reverse=True)
        return sorted_faces[:n_representatives]

    elif method == 'highest_confidence':
        # Sort by detection confidence
        sorted_faces = sorted(cluster, key=lambda f: f.confidence, reverse=True)
        return sorted_faces[:n_representatives]

    else:
        raise ValueError(f"Unknown method: {method}")


def select_best_representative(
    faces: List[FaceDetection],
    prefer_frontal: bool = True,
    frontal_yaw_threshold: float = 30.0,
    min_frontal_confidence: float = 0.75
) -> Tuple[FaceDetection, str]:
    """Select the best representative face from a cluster with pose-based filtering.

    This function prioritizes frontal faces with high confidence. If no frontal faces
    are available, it falls back to highest confidence overall.

    Args:
        faces: List of faces in the cluster
        prefer_frontal: Whether to prefer frontal faces (default: True)
        frontal_yaw_threshold: Max absolute yaw for frontal classification (default: 30°)
        min_frontal_confidence: Minimum confidence for a "good" frontal face (default: 0.75)

    Returns:
        Tuple of (selected_face, selection_reason)
        selection_reason can be:
        - 'frontal_high_confidence': Frontal face with confidence > threshold
        - 'frontal_best_available': Best frontal face (may have lower confidence)
        - 'highest_confidence_no_frontal': No frontal faces, picked highest confidence
        - 'highest_confidence_no_pose': No pose data available, picked highest confidence

    Raises:
        ValueError: If faces list is empty

    Example:
        >>> face, reason = select_best_representative(cluster_faces)
        >>> print(f"Selected {face.image_path.name} ({reason})")
    """
    if not faces:
        raise ValueError("Cannot select representative from empty list")

    # Check if any faces have pose data
    faces_with_pose = [f for f in faces if f.yaw is not None]

    if not faces_with_pose or not prefer_frontal:
        # No pose data or not preferring frontal - fall back to highest confidence
        best_face = max(faces, key=lambda f: f.confidence)
        reason = 'highest_confidence_no_pose' if not faces_with_pose else 'highest_confidence_no_frontal'
        return best_face, reason

    # Separate frontal faces
    frontal_faces = [f for f in faces_with_pose if abs(f.yaw) < frontal_yaw_threshold]

    if frontal_faces:
        # Found frontal faces - pick the one with highest confidence
        best_frontal = max(frontal_faces, key=lambda f: f.confidence)

        if best_frontal.confidence >= min_frontal_confidence:
            return best_frontal, 'frontal_high_confidence'
        else:
            return best_frontal, 'frontal_best_available'
    else:
        # No frontal faces - fall back to highest confidence overall
        best_face = max(faces, key=lambda f: f.confidence)
        return best_face, 'highest_confidence_no_frontal'


def compute_cluster_statistics(
    clusters: Dict[int, List[FaceDetection]]
) -> Dict[str, Any]:
    """Compute statistics about face clusters.

    Args:
        clusters: Dictionary mapping cluster_id to list of faces

    Returns:
        Dictionary with statistics
    """
    # Separate noise from real clusters
    real_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise = clusters.get(-1, [])

    if not real_clusters:
        return {
            'n_clusters': 0,
            'n_faces': sum(len(v) for v in clusters.values()),
            'n_noise': len(noise),
            'avg_cluster_size': 0.0,
            'min_cluster_size': 0,
            'max_cluster_size': 0
        }

    cluster_sizes = [len(v) for v in real_clusters.values()]

    return {
        'n_clusters': len(real_clusters),
        'n_faces': sum(len(v) for v in clusters.values()),
        'n_noise': len(noise),
        'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes),
        'min_cluster_size': min(cluster_sizes),
        'max_cluster_size': max(cluster_sizes),
        'cluster_size_distribution': {
            'small (2-5)': len([s for s in cluster_sizes if 2 <= s <= 5]),
            'medium (6-20)': len([s for s in cluster_sizes if 6 <= s <= 20]),
            'large (21+)': len([s for s in cluster_sizes if s >= 21])
        }
    }


def calculate_cluster_quality_metrics(
    faces: List[FaceDetection],
    frontal_yaw_threshold: float = 30.0,
    profile_yaw_threshold: float = 60.0
) -> Dict[str, Any]:
    """Calculate quality metrics for a cluster of faces.

    This function analyzes pose angles and confidence scores to assess cluster quality.
    Useful for identifying bad clusters (e.g., all side profiles) before filtering.

    Args:
        faces: List of faces in the cluster
        frontal_yaw_threshold: Max absolute yaw for frontal face (default: 30°)
        profile_yaw_threshold: Min absolute yaw for side profile (default: 60°)

    Returns:
        Dictionary with quality metrics:
        - total_faces: Total number of faces
        - frontal_faces: Count of frontal faces (|yaw| < threshold)
        - side_profile_faces: Count of side profiles (|yaw| > threshold)
        - faces_with_pose: Count of faces with pose data
        - avg_confidence: Average detection confidence
        - avg_yaw: Average yaw angle (if available)
        - yaw_variance: Variance in yaw angles (high = inconsistent)
        - has_good_frontal: Boolean if any frontal face with confidence > 0.75

    Example:
        >>> metrics = calculate_cluster_quality_metrics(faces)
        >>> if metrics['frontal_faces'] == 0:
        >>>     print("Bad cluster - no frontal faces!")
    """
    if not faces:
        return {
            'total_faces': 0,
            'frontal_faces': 0,
            'side_profile_faces': 0,
            'faces_with_pose': 0,
            'avg_confidence': 0.0,
            'avg_yaw': None,
            'yaw_variance': None,
            'has_good_frontal': False
        }

    total_faces = len(faces)

    # Collect pose and confidence data
    yaw_angles = []
    confidences = []
    frontal_count = 0
    profile_count = 0
    has_good_frontal = False

    for face in faces:
        confidences.append(face.confidence)

        # Check for pose data
        if face.yaw is not None:
            yaw = face.yaw
            yaw_angles.append(yaw)

            # Classify as frontal or profile
            abs_yaw = abs(yaw)
            if abs_yaw < frontal_yaw_threshold:
                frontal_count += 1
                # Check if this is a "good" frontal
                if face.confidence > 0.75:
                    has_good_frontal = True
            elif abs_yaw > profile_yaw_threshold:
                profile_count += 1

    # Calculate statistics
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    avg_yaw = None
    yaw_variance = None
    if yaw_angles:
        avg_yaw = sum(yaw_angles) / len(yaw_angles)
        yaw_variance = np.var(yaw_angles)

    return {
        'total_faces': total_faces,
        'frontal_faces': frontal_count,
        'side_profile_faces': profile_count,
        'faces_with_pose': len(yaw_angles),
        'avg_confidence': float(avg_confidence),
        'avg_yaw': float(avg_yaw) if avg_yaw is not None else None,
        'yaw_variance': float(yaw_variance) if yaw_variance is not None else None,
        'has_good_frontal': has_good_frontal
    }


def merge_clusters(
    clusters: Dict[int, List[FaceDetection]],
    similarity_threshold: float = 0.9
) -> Dict[int, List[FaceDetection]]:
    """Merge similar clusters based on representative similarity.

    Args:
        clusters: Dictionary mapping cluster_id to list of faces
        similarity_threshold: Minimum cosine similarity to merge clusters

    Returns:
        Dictionary with merged clusters
    """
    # Remove noise cluster
    real_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise = clusters.get(-1, [])

    if len(real_clusters) <= 1:
        return clusters

    # Find representative for each cluster
    cluster_ids = list(real_clusters.keys())
    representatives = {}

    for cluster_id in cluster_ids:
        rep = find_representative_faces(real_clusters[cluster_id], method='centroid', n_representatives=1)[0]
        representatives[cluster_id] = rep

    # Compute pairwise similarities
    merged = {}
    cluster_mapping = {}  # Old cluster_id -> new cluster_id

    new_cluster_id = 0
    for i, cluster_id_1 in enumerate(cluster_ids):
        if cluster_id_1 in cluster_mapping:
            continue  # Already merged

        # Start new merged cluster
        merged_faces = list(real_clusters[cluster_id_1])
        cluster_mapping[cluster_id_1] = new_cluster_id

        # Check similarity with remaining clusters
        for cluster_id_2 in cluster_ids[i+1:]:
            if cluster_id_2 in cluster_mapping:
                continue

            # Compute similarity between representatives
            emb1 = representatives[cluster_id_1].embedding
            emb2 = representatives[cluster_id_2].embedding

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            if similarity >= similarity_threshold:
                # Merge clusters
                merged_faces.extend(real_clusters[cluster_id_2])
                cluster_mapping[cluster_id_2] = new_cluster_id

        merged[new_cluster_id] = merged_faces
        new_cluster_id += 1

    # Add noise back
    if noise:
        merged[-1] = noise

    logger.info(f"Merged {len(real_clusters)} clusters into {len(merged) - (1 if noise else 0)} clusters")

    return merged
