"""Result formatting and ranking utilities.

This module provides utilities for formatting search results in various styles,
including AWS Rekognition-compatible format, and for ranking/sorting results.
"""

from typing import List, Dict, Any, Optional, Callable
import json
from pathlib import Path

from ..face import SearchResult, FaceDetection


def rank_results(
    results: List[SearchResult],
    key: Optional[Callable[[SearchResult], float]] = None,
    reverse: bool = True
) -> List[SearchResult]:
    """Rank search results by a custom key.

    Args:
        results: List of search results
        key: Function to extract ranking key (default: similarity score)
        reverse: If True, sort in descending order (default for similarity)

    Returns:
        Ranked list of SearchResult objects with updated rank fields
    """
    if key is None:
        key = lambda r: r.similarity

    # Sort results
    sorted_results = sorted(results, key=key, reverse=reverse)

    # Update rank numbers
    for rank, result in enumerate(sorted_results, start=1):
        result.rank = rank

    return sorted_results


def format_results_aws_style(
    results: List[SearchResult],
    include_embeddings: bool = False
) -> Dict[str, Any]:
    """Format search results in AWS Rekognition style.

    Args:
        results: List of search results
        include_embeddings: If True, include embedding vectors

    Returns:
        Dictionary in AWS Rekognition format with:
        - FaceMatches: List of face matches
        - SearchedFaceId: ID of queried face (if available)
        - FaceModelVersion: Model version (if available)
    """
    face_matches = []

    for result in results:
        face = result.face

        # Build face match object
        face_match = {
            'Similarity': result.similarity * 100,  # AWS uses 0-100
            'Face': {
                'FaceId': face.face_id if face.face_id is not None else 'unknown',
                'BoundingBox': {
                    'Width': (face.bbox.x2 - face.bbox.x1),
                    'Height': (face.bbox.y2 - face.bbox.y1),
                    'Left': face.bbox.x1,
                    'Top': face.bbox.y1
                },
                'ImageId': str(face.image_path) if face.image_path else 'unknown',
                'Confidence': face.confidence * 100,  # AWS uses 0-100
            }
        }

        # Add optional fields
        if face.quality:
            face_match['Face']['Quality'] = {
                'Brightness': face.quality.brightness * 100,
                'Sharpness': face.quality.sharpness * 100
            }

        if face.landmarks:
            face_match['Face']['Landmarks'] = [
                {'Type': 'eyeLeft', 'X': face.landmarks.left_eye[0], 'Y': face.landmarks.left_eye[1]},
                {'Type': 'eyeRight', 'X': face.landmarks.right_eye[0], 'Y': face.landmarks.right_eye[1]},
                {'Type': 'nose', 'X': face.landmarks.nose[0], 'Y': face.landmarks.nose[1]},
                {'Type': 'mouthLeft', 'X': face.landmarks.left_mouth[0], 'Y': face.landmarks.left_mouth[1]},
                {'Type': 'mouthRight', 'X': face.landmarks.right_mouth[0], 'Y': face.landmarks.right_mouth[1]},
            ] if all([
                face.landmarks.left_eye,
                face.landmarks.right_eye,
                face.landmarks.nose,
                face.landmarks.left_mouth,
                face.landmarks.right_mouth
            ]) else []

        if include_embeddings and face.embedding is not None:
            face_match['Face']['Embedding'] = face.embedding.tolist()

        face_matches.append(face_match)

    # Build response
    response = {
        'FaceMatches': face_matches,
        'FaceModelVersion': '1.0'  # Could be from model version
    }

    return response


def format_results_simple(
    results: List[SearchResult],
    max_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Format search results in simple dictionary format.

    Args:
        results: List of search results
        max_results: Maximum results to include

    Returns:
        List of dictionaries with simplified result data
    """
    formatted = []

    for result in results[:max_results] if max_results else results:
        formatted.append({
            'rank': result.rank,
            'similarity': round(result.similarity, 4),
            'distance': round(result.distance, 4),
            'face_id': result.face.face_id,
            'image_path': str(result.face.image_path) if result.face.image_path else None,
            'confidence': round(result.face.confidence, 4),
            'bbox': result.face.bbox.to_dict()
        })

    return formatted


def format_results_detailed(
    results: List[SearchResult],
    max_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Format search results with detailed information.

    Args:
        results: List of search results
        max_results: Maximum results to include

    Returns:
        List of dictionaries with detailed result data
    """
    formatted = []

    for result in results[:max_results] if max_results else results:
        face_dict = result.face.to_dict()

        formatted.append({
            'rank': result.rank,
            'similarity': round(result.similarity, 4),
            'distance': round(result.distance, 4),
            'face': face_dict
        })

    return formatted


def group_results_by_image(
    results: List[SearchResult]
) -> Dict[str, List[SearchResult]]:
    """Group search results by source image.

    Args:
        results: List of search results

    Returns:
        Dictionary mapping image path to list of results from that image
    """
    grouped = {}

    for result in results:
        image_path = str(result.face.image_path) if result.face.image_path else 'unknown'

        if image_path not in grouped:
            grouped[image_path] = []

        grouped[image_path].append(result)

    return grouped


def filter_results(
    results: List[SearchResult],
    min_similarity: Optional[float] = None,
    max_distance: Optional[float] = None,
    min_confidence: Optional[float] = None,
    exclude_images: Optional[List[str]] = None
) -> List[SearchResult]:
    """Filter search results by various criteria.

    Args:
        results: List of search results
        min_similarity: Minimum similarity score (0-1)
        max_distance: Maximum L2 distance
        min_confidence: Minimum detection confidence (0-1)
        exclude_images: List of image paths to exclude

    Returns:
        Filtered list of SearchResult objects
    """
    filtered = []

    for result in results:
        # Filter by similarity
        if min_similarity is not None and result.similarity < min_similarity:
            continue

        # Filter by distance
        if max_distance is not None and result.distance > max_distance:
            continue

        # Filter by confidence
        if min_confidence is not None and result.face.confidence < min_confidence:
            continue

        # Filter by image path
        if exclude_images and result.face.image_path:
            if str(result.face.image_path) in exclude_images:
                continue

        filtered.append(result)

    return filtered


def get_top_matches(
    results: List[SearchResult],
    n: int = 1,
    min_similarity: Optional[float] = None
) -> List[SearchResult]:
    """Get top N matches from search results.

    Args:
        results: List of search results
        n: Number of top results to return
        min_similarity: Optional minimum similarity threshold

    Returns:
        Top N SearchResult objects
    """
    # Filter by minimum similarity if specified
    if min_similarity is not None:
        results = filter_results(results, min_similarity=min_similarity)

    # Return top N
    return results[:n]


def compute_result_statistics(
    results: List[SearchResult]
) -> Dict[str, Any]:
    """Compute statistics about search results.

    Args:
        results: List of search results

    Returns:
        Dictionary with statistics
    """
    if not results:
        return {
            'count': 0,
            'avg_similarity': 0.0,
            'avg_distance': 0.0,
            'avg_confidence': 0.0
        }

    similarities = [r.similarity for r in results]
    distances = [r.distance for r in results]
    confidences = [r.face.confidence for r in results]

    # Group by image
    images = set(str(r.face.image_path) for r in results if r.face.image_path)

    return {
        'count': len(results),
        'unique_images': len(images),
        'avg_similarity': sum(similarities) / len(similarities),
        'min_similarity': min(similarities),
        'max_similarity': max(similarities),
        'avg_distance': sum(distances) / len(distances),
        'min_distance': min(distances),
        'max_distance': max(distances),
        'avg_confidence': sum(confidences) / len(confidences),
        'min_confidence': min(confidences),
        'max_confidence': max(confidences)
    }


def export_results_json(
    results: List[SearchResult],
    output_path: Path,
    format_style: str = 'simple'
) -> None:
    """Export search results to JSON file.

    Args:
        results: List of search results
        output_path: Path to output JSON file
        format_style: Format style ('simple', 'detailed', or 'aws')
    """
    if format_style == 'simple':
        data = format_results_simple(results)
    elif format_style == 'detailed':
        data = format_results_detailed(results)
    elif format_style == 'aws':
        data = format_results_aws_style(results)
    else:
        raise ValueError(f"Unknown format style: {format_style}")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def deduplicate_results(
    results: List[SearchResult],
    key: str = 'face_id'
) -> List[SearchResult]:
    """Remove duplicate results based on a key.

    Args:
        results: List of search results
        key: Key to use for deduplication ('face_id', 'image_path')

    Returns:
        Deduplicated list of SearchResult objects (keeps first occurrence)
    """
    seen = set()
    deduped = []

    for result in results:
        if key == 'face_id':
            identifier = result.face.face_id
        elif key == 'image_path':
            identifier = str(result.face.image_path) if result.face.image_path else None
        else:
            raise ValueError(f"Unknown deduplication key: {key}")

        if identifier not in seen:
            seen.add(identifier)
            deduped.append(result)

    return deduped
