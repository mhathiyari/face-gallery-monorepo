"""Search module for face similarity search.

This module provides high-level search functionality including:
- SearchEngine for querying faces by image, face ID, or metadata
- Result formatting utilities (AWS Rekognition style, simple, detailed)
- Face clustering using DBSCAN algorithm
- Result ranking and filtering

Usage:
    from face_search.search import SearchEngine, cluster_faces, format_results_aws_style

    # Create search engine
    engine = SearchEngine(model=model, collection=collection)

    # Search by image
    results = engine.search_by_image('query.jpg', max_results=10)

    # Format results
    formatted = format_results_aws_style(results[0])

    # Cluster faces
    clusters = cluster_faces(faces, eps=0.6, min_samples=2)
"""

from .engine import SearchEngine
from .results import (
    rank_results,
    format_results_aws_style,
    format_results_simple,
    format_results_detailed,
    group_results_by_image,
    filter_results,
    get_top_matches,
    compute_result_statistics,
    export_results_json,
    deduplicate_results
)
from .clustering import (
    cluster_faces,
    get_identity_cluster,
    cluster_by_image,
    find_representative_faces,
    select_best_representative,
    compute_cluster_statistics,
    calculate_cluster_quality_metrics,
    merge_clusters
)

__all__ = [
    # Engine
    'SearchEngine',

    # Result formatting
    'rank_results',
    'format_results_aws_style',
    'format_results_simple',
    'format_results_detailed',
    'group_results_by_image',
    'filter_results',
    'get_top_matches',
    'compute_result_statistics',
    'export_results_json',
    'deduplicate_results',

    # Clustering
    'cluster_faces',
    'get_identity_cluster',
    'cluster_by_image',
    'find_representative_faces',
    'select_best_representative',
    'compute_cluster_statistics',
    'calculate_cluster_quality_metrics',
    'merge_clusters',
]
