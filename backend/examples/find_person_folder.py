#!/usr/bin/env python3
"""
Find which person folder a query image belongs to.

This script uses face similarity search to identify which person folder
from sort_images_by_person.py a given query image belongs to.

Usage:
    python examples/find_person_folder.py /path/to/query_image.jpg /path/to/sorted_output
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_search.models import InsightFaceModel
from face_search.storage import Collection
from face_search.search import SearchEngine


def find_person_folder(
    query_image_path: str,
    sorted_output_folder: str,
    top_k: int = 10,
    min_similarity: float = 0.5
):
    """Find which person folder a query image belongs to.

    Args:
        query_image_path: Path to query image containing a person
        sorted_output_folder: Path to output folder from sort_images_by_person.py
        top_k: Number of similar faces to retrieve
        min_similarity: Minimum similarity threshold (0-1, higher = stricter)

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

    # First, check if we need to rebuild the index
    from face_search.storage.collection_manager import SyncError
    from face_search.storage import MetadataStore
    import faiss

    collection_path = collections_path / 'image_sorter'
    index_path = collection_path / 'index.faiss'

    # Check if index needs rebuilding
    needs_rebuild = False
    if not index_path.exists():
        print(f"⚠️  FAISS index not found, will rebuild...")
        needs_rebuild = True
    else:
        # Check sync
        try:
            index = faiss.read_index(str(index_path))
            db = MetadataStore(collection_path / 'metadata.db', collection_id='image_sorter')
            face_count = db.count_faces()
            if index.ntotal != face_count:
                print(f"⚠️  Index out of sync ({index.ntotal} vs {face_count} faces), will rebuild...")
                needs_rebuild = True
        except Exception:
            needs_rebuild = True

    if needs_rebuild:
        # Manually rebuild before loading Collection
        print(f"   Rebuilding index...")
        db = MetadataStore(collection_path / 'metadata.db', collection_id='image_sorter')
        all_faces = db.get_all_faces(include_deleted=False)

        if not all_faces:
            raise ValueError("No faces found in database!")

        # Get embeddings
        import numpy as np
        import pickle
        embeddings = []
        for face in all_faces:
            emb = pickle.loads(face.embedding)
            embeddings.append(emb)

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Create and save index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        faiss.write_index(index, str(index_path))

        print(f"✅ Index rebuilt: {index.ntotal} faces")

    # Now load collection normally
    collection = Collection(
        collection_id='image_sorter',
        collections_root=collections_path,
        embedding_dim=512
    )
    print(f"✅ Collection loaded: {collection.ntotal} faces indexed")

    # Create search engine
    print("\n🔎 Creating search engine...")
    search_engine = SearchEngine(
        model=model,
        collection=collection,
        default_max_results=top_k
    )

    # Search for similar faces
    print(f"\n📸 Searching for similar faces in query image: {query_path.name}")
    results = search_engine.search_by_image(
        image_path=query_path,
        max_results=top_k,
        min_similarity=min_similarity,
        return_all_faces=False  # Only return results for the best face in query
    )

    if not results:
        print("❌ No face detected in query image")
        return None

    # Get results for the first (or only) face in query image
    # results is List[List[SearchResult]], so results[0] is List[SearchResult]
    face_results = results[0]

    if not face_results:
        print(f"❌ No similar faces found (similarity threshold: {min_similarity})")
        print(f"   Try lowering min_similarity or check if the person exists in the collection")
        return None

    print(f"✅ Found {len(face_results)} similar faces")

    # Map image paths to their person folders
    print("\n📊 Analyzing matches...")
    folder_votes = defaultdict(list)  # folder -> list of (similarity, image_path)

    for match in face_results:
        # Get the image path from the match
        img_path = Path(match.face.image_path)

        # Find which person folder this image is in
        # Images were copied to person_XXX folders in sorted_output_folder
        for person_folder in output_path.glob('person_*'):
            if person_folder.is_dir():
                # Check if any image in this folder matches
                for sorted_img in person_folder.glob('*'):
                    if sorted_img.stem in img_path.stem or img_path.stem in sorted_img.stem:
                        folder_votes[person_folder.name].append(
                            (match.similarity, str(img_path))
                        )
                        break

    # Also check unmatched folder
    unmatched_folder = output_path / 'unmatched'
    if unmatched_folder.exists():
        for match in face_results:
            img_path = Path(match.face.image_path)
            for sorted_img in unmatched_folder.glob('*'):
                if sorted_img.stem in img_path.stem or img_path.stem in sorted_img.stem:
                    folder_votes['unmatched'].append(
                        (match.similarity, str(img_path))
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
    print(f"   Confidence: {scores['avg_similarity']:.2%}")
    print(f"   Matches: {scores['count']} faces")
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

    return {
        'person_folder': folder_name,
        'confidence': scores['avg_similarity'],
        'match_count': scores['count'],
        'max_similarity': scores['max_similarity'],
        'all_scores': folder_scores
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python find_person_folder.py <query_image> <sorted_output_folder>")
        print("\nExample:")
        print("  python examples/find_person_folder.py ~/query.jpg ~/Downloads/Sath_payari_ghat_sorted2")
        print("\nOptional arguments:")
        print("  --top-k N          Number of similar faces to check (default: 10)")
        print("  --min-similarity S  Minimum similarity threshold 0-1 (default: 0.5)")
        sys.exit(1)

    query_image = sys.argv[1]
    sorted_folder = sys.argv[2]

    # Parse optional arguments
    top_k = 10
    min_similarity = 0.5

    for i, arg in enumerate(sys.argv[3:], start=3):
        if arg == '--top-k' and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
        elif arg == '--min-similarity' and i + 1 < len(sys.argv):
            min_similarity = float(sys.argv[i + 1])

    result = find_person_folder(
        query_image,
        sorted_folder,
        top_k=top_k,
        min_similarity=min_similarity
    )
