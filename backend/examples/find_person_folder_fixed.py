#!/usr/bin/env python3
"""
Find which person folder a query image belongs to (FIXED for unnormalized embeddings).

Uses cosine similarity instead of L2 distance conversion.

Usage:
    python examples/find_person_folder_fixed.py /path/to/query_image.jpg /path/to/sorted_output
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_search.models import InsightFaceModel
from face_search.storage import Collection
from face_search.storage.collection_manager import SyncError


def find_person_folder(
    query_image_path: str,
    sorted_output_folder: str,
    top_k: int = 20,
    min_cosine_similarity: float = 0.6
):
    """Find which person folder a query image belongs to.

    Args:
        query_image_path: Path to query image containing a person
        sorted_output_folder: Path to output folder from sort_images_by_person.py
        top_k: Number of similar faces to retrieve
        min_cosine_similarity: Minimum cosine similarity threshold (0-1, higher = stricter)

    Returns:
        dict with 'person_folder', 'confidence', and 'matches'
    """
    query_path = Path(query_image_path)
    output_path = Path(sorted_output_folder)

    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    collections_path = output_path / 'collections'
    if not collections_path.exists():
        raise FileNotFoundError(
            f"Collections folder not found: {collections_path}\n"
            f"Make sure you've run sort_images_by_person.py first!"
        )

    print("🔍 Face Recognition - Find Person Folder")
    print("=" * 60)

    # Load face detection model
    print("\n📦 Loading face detection model...")
    model = InsightFaceModel(model_name='buffalo_l', device='auto')
    print(f"✅ Model loaded on device: {model.device}")

    # Load collection
    print("\n💾 Loading face collection...")
    try:
        collection = Collection(
            collection_id='image_sorter',
            collections_root=collections_path,
            embedding_dim=512
        )
    except SyncError:
        print("⚠️  Collection out of sync, rebuilding FAISS index...")
        # Manually rebuild by initializing components
        from face_search.storage.metadata_db import MetadataStore
        from face_search.storage.faiss_index import FAISSIndex
        import numpy as np

        collection_path = collections_path / 'image_sorter'
        metadata_path = collection_path / 'metadata.db'
        index_path = collection_path / 'index.faiss'

        # Load metadata
        metadata = MetadataStore(str(metadata_path), 'image_sorter')

        # Create new index
        faiss_index = FAISSIndex(dim=512)

        # Add all active faces to index
        all_faces = metadata.get_all_faces(include_deleted=False)
        embeddings = [face.get_embedding() for face in all_faces]

        if embeddings:
            embeddings_array = np.array(embeddings)
            faiss_index.add(embeddings_array)
            faiss_index.save(str(index_path))
            print(f"✅ Index rebuilt with {len(embeddings)} faces")

        # Now load the collection normally (should work now)
        try:
            collection = Collection(
                collection_id='image_sorter',
                collections_root=collections_path,
                embedding_dim=512
            )
        except SyncError as e2:
            # If still failing, something is wrong
            raise RuntimeError(f"Failed to load collection even after rebuild: {e2}")

    print(f"✅ Collection loaded: {collection.ntotal} faces indexed")

    # Detect face in query image
    print(f"\n📸 Detecting face in query image: {query_path.name}")
    faces = model.detect_faces(str(query_path), min_confidence=0.3)

    if not faces:
        print("❌ No face detected in query image")
        return None

    print(f"✅ Found {len(faces)} face(s), using the first one")

    query_embedding = faces[0].embedding
    query_norm = np.linalg.norm(query_embedding)

    # Search using FAISS for top-K nearest neighbors (by L2 distance)
    print(f"\n🔎 Searching for {top_k} nearest faces...")
    results = collection.search(query_embedding, k=top_k)

    # Compute cosine similarity for each result and group by cluster_id
    print("\n📊 Grouping matches by cluster_id...")
    cluster_votes = defaultdict(list)  # cluster_id -> list of (similarity, face_record)

    for face_record, l2_dist in results:
        stored_embedding = face_record.get_embedding()

        # Compute cosine similarity
        cosine_sim = np.dot(query_embedding, stored_embedding) / (
            query_norm * np.linalg.norm(stored_embedding)
        )

        if cosine_sim >= min_cosine_similarity:
            cluster_id = face_record.cluster_id
            if cluster_id is not None:
                cluster_votes[cluster_id].append((cosine_sim, face_record))

    if not cluster_votes:
        print(f"❌ No similar faces found (cosine similarity threshold: {min_cosine_similarity:.2f})")
        print(f"   Try lowering --min-similarity (currently {min_cosine_similarity:.2f})")
        return None

    # Find best cluster by vote count and similarity
    cluster_scores = {}
    for cluster_id, votes in cluster_votes.items():
        count = len(votes)
        avg_similarity = sum(s for s, _ in votes) / count
        max_similarity = max(s for s, _ in votes)
        cluster_scores[cluster_id] = {
            'count': count,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'score': count * avg_similarity,
            'votes': votes
        }

    best_cluster_id = max(cluster_scores.items(), key=lambda x: x[1]['score'])[0]
    best_cluster = cluster_scores[best_cluster_id]

    print(f"✅ Best match: cluster_id={best_cluster_id} with {best_cluster['count']} faces")

    # Map cluster to person folder
    print("\n📁 Finding person folder for cluster...")
    folder_votes = defaultdict(list)  # folder -> list of (similarity, image_path)

    for similarity, face_record in best_cluster['votes']:
        img_path = Path(face_record.image_path)

        # Find which person folder this image is in
        for person_folder in output_path.glob('person_*'):
            if person_folder.is_dir():
                for sorted_img in person_folder.glob('*'):
                    if sorted_img.stem in img_path.stem or img_path.stem in sorted_img.stem:
                        folder_votes[person_folder.name].append(
                            (similarity, str(img_path))
                        )
                        break

    # Check unmatched folder
    unmatched_folder = output_path / 'unmatched'
    if unmatched_folder.exists():
        for similarity, face_record in best_cluster['votes']:
            img_path = Path(face_record.image_path)
            for sorted_img in unmatched_folder.glob('*'):
                if sorted_img.stem in img_path.stem or img_path.stem in sorted_img.stem:
                    folder_votes['unmatched'].append(
                        (similarity, str(img_path))
                    )
                    break

    if not folder_votes:
        print("⚠️  Matches found but couldn't map to person folders")
        print("   This might happen if the sorted folder structure was modified")
        return None

    # Determine best match by folder with most votes and highest avg similarity
    folder_scores = {}
    for folder, votes in folder_votes.items():
        count = len(votes)
        avg_similarity = sum(s for s, _ in votes) / count
        max_similarity = max(s for s, _ in votes)
        folder_scores[folder] = {
            'count': count,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'score': count * avg_similarity  # Combined score
        }

    # Sort by combined score
    best_folder = max(folder_scores.items(), key=lambda x: x[1]['score'])
    folder_name, scores = best_folder

    # Display results
    print("\n" + "=" * 60)
    print("🎯 RESULTS")
    print("=" * 60)
    print(f"\n✅ Person belongs to: {folder_name}/")
    print(f"   Cluster ID: {best_cluster_id}")
    print(f"   Confidence: {scores['avg_similarity']:.2%}")
    print(f"   Matches: {scores['count']} faces (from cluster with {best_cluster['count']} total faces)")
    print(f"   Max similarity: {scores['max_similarity']:.2%}")

    # Show top 3 folders
    print(f"\n📋 Top matches:")
    sorted_folders = sorted(
        folder_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:3]

    for i, (folder, stats) in enumerate(sorted_folders, 1):
        print(f"   {i}. {folder:20s} - "
              f"{stats['count']} matches, "
              f"{stats['avg_similarity']:.1%} avg similarity")

    # Show sample matching images
    print(f"\n🖼️  Sample matching images from {folder_name}:")
    top_matches = sorted(folder_votes[folder_name], key=lambda x: x[0], reverse=True)[:3]
    for sim, img_path in top_matches:
        print(f"   {Path(img_path).name} (similarity: {sim:.2%})")

    # Load and display representative metadata if available
    rep_file = output_path / folder_name / '.representative.json'
    if rep_file.exists():
        try:
            import json
            with open(rep_file, 'r') as f:
                rep_data = json.load(f)

            print(f"\n👤 Representative Face Info:")
            print(f"   Image: {rep_data['image_name']}")
            print(f"   Confidence: {rep_data['confidence']:.2%}")
            if rep_data.get('yaw') is not None:
                print(f"   Pose: yaw={rep_data['yaw']:.1f}°, pitch={rep_data.get('pitch', 0):.1f}°, roll={rep_data.get('roll', 0):.1f}°")
            print(f"   Selection: {rep_data.get('selection_reason', 'unknown')}")

            # Display cluster quality warnings
            cluster_stats = rep_data.get('cluster_stats', {})
            if cluster_stats:
                print(f"\n📊 Cluster Quality:")
                print(f"   Total faces: {cluster_stats.get('total_faces', 0)}")
                print(f"   Frontal: {cluster_stats.get('frontal_faces', 0)}, "
                      f"Profile: {cluster_stats.get('side_profile_faces', 0)}")
                print(f"   Avg confidence: {cluster_stats.get('avg_confidence', 0):.2%}")

                # Warnings for potentially bad clusters
                if cluster_stats.get('frontal_faces', 0) == 0 and cluster_stats.get('faces_with_pose', 0) > 0:
                    print(f"   ⚠️  WARNING: No frontal faces detected - cluster may be low quality")
                if cluster_stats.get('yaw_variance') and cluster_stats['yaw_variance'] > 500:
                    print(f"   ⚠️  WARNING: High pose variance ({cluster_stats['yaw_variance']:.0f}) - inconsistent cluster")

        except Exception as e:
            print(f"\n⚠️  Could not read representative metadata: {e}")

    return {
        'person_folder': folder_name,
        'confidence': scores['avg_similarity'],
        'match_count': scores['count'],
        'max_similarity': scores['max_similarity'],
        'all_scores': folder_scores
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python find_person_folder_fixed.py <query_image> <sorted_output_folder>")
        print("\nExample:")
        print("  python examples/find_person_folder_fixed.py ~/query.jpg ~/Downloads/Sath_payari_ghat_sorted2")
        print("\nOptional arguments:")
        print("  --top-k N          Number of similar faces to check (default: 20)")
        print("  --min-similarity S  Minimum cosine similarity 0-1 (default: 0.6)")
        sys.exit(1)

    query_image = sys.argv[1]
    sorted_folder = sys.argv[2]

    # Parse optional arguments
    top_k = 20
    min_similarity = 0.6

    for i, arg in enumerate(sys.argv[3:], start=3):
        if arg == '--top-k' and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
        elif arg == '--min-similarity' and i + 1 < len(sys.argv):
            min_similarity = float(sys.argv[i + 1])

    result = find_person_folder(
        query_image,
        sorted_folder,
        top_k=top_k,
        min_cosine_similarity=min_similarity
    )
