"""
Importable high-level operations for Face Gallery.

These functions wrap the logic from backend/examples/ scripts into proper
importable functions with relative imports. They can be called directly
from Python code (CLI, web UI) or via the `face-gallery` CLI.

Print output is preserved for CLI users; when called from the web UI's
JobManager, stdout is captured automatically.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .face import BoundingBox, FaceDetection
from .indexer import BatchIndexer
from .models import InsightFaceModel
from .search import (
    calculate_cluster_quality_metrics,
    cluster_faces,
    select_best_representative,
)
from .storage import Collection
from .storage.collection_manager import SyncError


def sort_images_by_person(
    input_folder: str,
    output_folder: str,
    *,
    model_name: str = "buffalo_l",
    device: str = "auto",
    min_confidence: float = 0.5,
    eps: float = 0.5,
    min_samples: int = 2,
    metric: str = "cosine",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Sort images by detected person/identity.

    Args:
        input_folder: Folder containing images to sort.
        output_folder: Folder where sorted images will be organized.
        model_name: InsightFace model name.
        device: Compute device ('auto', 'cuda', 'mps', 'cpu').
        min_confidence: Minimum face detection confidence (0-1).
        eps: DBSCAN epsilon. For cosine: 0.3 (strict) to 0.6 (lenient).
        min_samples: Minimum faces to form a cluster.
        metric: Distance metric ('cosine' or 'euclidean').
        batch_size: Batch size for face indexing.

    Returns:
        Dict with keys: persons (int), images_sorted (int), noise_count (int).
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"Error: Input folder not found: {input_path}")
        return {"persons": 0, "images_sorted": 0, "noise_count": 0}

    output_path.mkdir(parents=True, exist_ok=True)

    print("Face-based Image Sorting")
    print("=" * 50)

    # Step 1: Initialize face detection model
    print("\nLoading face detection model...")
    try:
        model = InsightFaceModel(model_name=model_name, device=device)
        print(f"Model loaded on device: {model.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"persons": 0, "images_sorted": 0, "noise_count": 0}

    # Step 2: Create collection for storing faces
    print("\nCreating face collection...")
    collection = Collection(
        collection_id="image_sorter",
        collections_root=output_path / "collections",
        embedding_dim=512,
    )
    print("Collection created")

    # Step 3: Index all faces in the input folder
    print(f"\nScanning and indexing faces in: {input_path}")
    indexer = BatchIndexer(
        model=model,
        collection=collection,
        batch_size=batch_size,
        enable_deduplication=True,
    )

    stats = indexer.index_folder(
        folder_path=input_path,
        recursive=True,
        min_confidence=min_confidence,
    )

    print(f"\nIndexing Results:")
    print(f"   Images processed: {stats.total_images}")
    print(f"   Faces found: {stats.total_faces_found}")
    print(f"   Faces indexed: {stats.total_faces_indexed}")
    print(f"   Duplicates skipped: {stats.duplicates_skipped}")
    print(f"   Processing time: {stats.processing_time:.2f}s")
    print(f"   Speed: {stats.faces_per_second:.1f} faces/sec")

    if stats.total_faces_indexed == 0:
        print("\nNo faces found. Make sure your images contain visible faces.")
        return {"persons": 0, "images_sorted": 0, "noise_count": 0}

    # Step 4: Get all indexed faces
    print("\nClustering faces by identity...")
    all_faces = []
    for face_record in collection.metadata.get_all_faces(include_deleted=False):
        face = FaceDetection(
            bbox=BoundingBox.from_dict(
                eval(face_record.bbox)
                if isinstance(face_record.bbox, str)
                else face_record.bbox
            ),
            confidence=face_record.confidence,
            embedding=face_record.get_embedding(),
            image_path=Path(face_record.image_path),
            face_id=face_record.face_id,
            yaw=face_record.yaw,
            pitch=face_record.pitch,
            roll=face_record.roll,
        )
        all_faces.append(face)

    # Step 5: Cluster faces
    clusters = cluster_faces(
        faces=all_faces,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )

    real_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise_faces = clusters.get(-1, [])

    print(f"Found {len(real_clusters)} identities")
    print(f"   Noise (unmatched faces): {len(noise_faces)}")

    # Step 5.5: Save cluster assignments to database
    print("\nSaving cluster assignments to database...")
    face_cluster_map = {}
    for cluster_id, faces_in_cluster in clusters.items():
        for face in faces_in_cluster:
            face_cluster_map[face.face_id] = cluster_id

    updated = collection.metadata.update_cluster_assignments(face_cluster_map)
    print(f"Saved {updated} cluster assignments")

    # Step 6: Organize images by cluster
    print("\nOrganizing images into folders...")
    sorted_count = 0

    for cluster_id, faces_in_cluster in real_clusters.items():
        person_folder = output_path / f"person_{cluster_id:03d}"
        person_folder.mkdir(exist_ok=True)

        images_in_cluster = set(str(f.image_path) for f in faces_in_cluster)

        for img_path in images_in_cluster:
            src = Path(img_path)
            if src.exists():
                dst = person_folder / src.name
                counter = 1
                while dst.exists():
                    dst = person_folder / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1
                shutil.copy2(src, dst)
                sorted_count += 1

        print(
            f"   Person {cluster_id}: {len(faces_in_cluster)} faces, "
            f"{len(images_in_cluster)} images -> {person_folder.name}/"
        )

    # Handle noise
    if noise_faces:
        noise_folder = output_path / "unmatched"
        noise_folder.mkdir(exist_ok=True)

        noise_images = set(str(f.image_path) for f in noise_faces)
        for img_path in noise_images:
            src = Path(img_path)
            if src.exists():
                dst = noise_folder / src.name
                counter = 1
                while dst.exists():
                    dst = noise_folder / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1
                shutil.copy2(src, dst)

        print(
            f"   Unmatched: {len(noise_faces)} faces, "
            f"{len(noise_images)} images -> unmatched/"
        )

    print(f"\nDone! Sorted {sorted_count} images into {len(real_clusters)} person folders")
    print(f"Results saved to: {output_path}")

    # Step 7: Generate .representative.json for each person folder
    print("\nGenerating representative metadata files...")
    for cluster_id, faces_in_cluster in real_clusters.items():
        cluster_stats = calculate_cluster_quality_metrics(faces_in_cluster)
        rep, selection_reason = select_best_representative(
            faces_in_cluster, prefer_frontal=True
        )

        rep_metadata = {
            "face_id": rep.face_id,
            "image_name": rep.image_path.name,
            "image_path": str(rep.image_path),
            "bbox": {
                "x1": rep.bbox.x1,
                "y1": rep.bbox.y1,
                "x2": rep.bbox.x2,
                "y2": rep.bbox.y2,
            },
            "confidence": float(rep.confidence),
            "yaw": float(rep.yaw) if rep.yaw is not None else None,
            "pitch": float(rep.pitch) if rep.pitch is not None else None,
            "roll": float(rep.roll) if rep.roll is not None else None,
            "selection_reason": selection_reason,
            "cluster_stats": cluster_stats,
        }

        person_folder = output_path / f"person_{cluster_id:03d}"
        rep_file = person_folder / ".representative.json"
        with open(rep_file, "w") as f:
            json.dump(rep_metadata, f, indent=2)

        print(
            f"   Person {cluster_id}: {rep.image_path.name} "
            f"(confidence: {rep.confidence:.2f}, {selection_reason})"
        )

    print(f"Generated {len(real_clusters)} representative metadata files")

    return {
        "persons": len(real_clusters),
        "images_sorted": sorted_count,
        "noise_count": len(noise_faces),
    }


def find_person_folder(
    query_image_path: str,
    sorted_output_folder: str,
    *,
    top_k: int = 20,
    min_cosine_similarity: float = 0.6,
    model_name: str = "buffalo_l",
    device: str = "auto",
) -> Optional[Dict[str, Any]]:
    """Find which person folder a query image belongs to.

    Args:
        query_image_path: Path to query image containing a person.
        sorted_output_folder: Path to output folder from sort_images_by_person.
        top_k: Number of similar faces to retrieve.
        min_cosine_similarity: Minimum cosine similarity threshold (0-1).
        model_name: InsightFace model name.
        device: Compute device.

    Returns:
        Dict with 'person_folder', 'confidence', 'match_count',
        'max_similarity', 'all_scores', or None if no match found.
    """
    query_path = Path(query_image_path)
    output_path = Path(sorted_output_folder)

    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    collections_path = output_path / "collections"
    if not collections_path.exists():
        raise FileNotFoundError(
            f"Collections folder not found: {collections_path}\n"
            f"Make sure you've run sort first!"
        )

    print("Face Recognition - Find Person Folder")
    print("=" * 60)

    # Load face detection model
    print("\nLoading face detection model...")
    model = InsightFaceModel(model_name=model_name, device=device)
    print(f"Model loaded on device: {model.device}")

    # Load collection
    print("\nLoading face collection...")
    try:
        collection = Collection(
            collection_id="image_sorter",
            collections_root=collections_path,
            embedding_dim=512,
        )
    except SyncError:
        print("Collection out of sync, rebuilding FAISS index...")
        from .storage.faiss_index import FAISSIndex
        from .storage.metadata_db import MetadataStore

        collection_path = collections_path / "image_sorter"
        metadata_path = collection_path / "metadata.db"
        index_path = collection_path / "index.faiss"

        metadata = MetadataStore(str(metadata_path), "image_sorter")
        faiss_index = FAISSIndex(dim=512)

        all_faces_records = metadata.get_all_faces(include_deleted=False)
        embeddings = [face.get_embedding() for face in all_faces_records]

        if embeddings:
            embeddings_array = np.array(embeddings)
            faiss_index.add(embeddings_array)
            faiss_index.save(str(index_path))
            print(f"Index rebuilt with {len(embeddings)} faces")

        try:
            collection = Collection(
                collection_id="image_sorter",
                collections_root=collections_path,
                embedding_dim=512,
            )
        except SyncError as e2:
            raise RuntimeError(
                f"Failed to load collection even after rebuild: {e2}"
            ) from e2

    print(f"Collection loaded: {collection.ntotal} faces indexed")

    # Detect face in query image
    print(f"\nDetecting face in query image: {query_path.name}")
    faces = model.detect_faces(str(query_path), min_confidence=0.3)

    if not faces:
        print("No face detected in query image")
        return None

    print(f"Found {len(faces)} face(s), using the first one")

    query_embedding = faces[0].embedding
    query_norm = np.linalg.norm(query_embedding)

    # Search for nearest neighbors
    print(f"\nSearching for {top_k} nearest faces...")
    results = collection.search(query_embedding, k=top_k)

    # Compute cosine similarity and group by cluster_id
    print("\nGrouping matches by cluster_id...")
    cluster_votes: Dict[int, list] = defaultdict(list)

    for face_record, l2_dist in results:
        stored_embedding = face_record.get_embedding()
        cosine_sim = np.dot(query_embedding, stored_embedding) / (
            query_norm * np.linalg.norm(stored_embedding)
        )

        if cosine_sim >= min_cosine_similarity:
            cluster_id = face_record.cluster_id
            if cluster_id is not None:
                cluster_votes[cluster_id].append((cosine_sim, face_record))

    if not cluster_votes:
        print(
            f"No similar faces found (cosine similarity threshold: "
            f"{min_cosine_similarity:.2f})"
        )
        return None

    # Find best cluster
    cluster_scores: Dict[int, Dict[str, Any]] = {}
    for cluster_id, votes in cluster_votes.items():
        count = len(votes)
        avg_similarity = sum(s for s, _ in votes) / count
        max_similarity = max(s for s, _ in votes)
        cluster_scores[cluster_id] = {
            "count": count,
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "score": count * avg_similarity,
            "votes": votes,
        }

    best_cluster_id = max(
        cluster_scores.items(), key=lambda x: x[1]["score"]
    )[0]
    best_cluster = cluster_scores[best_cluster_id]

    print(
        f"Best match: cluster_id={best_cluster_id} "
        f"with {best_cluster['count']} faces"
    )

    # Map cluster to person folder
    print("\nFinding person folder for cluster...")
    folder_votes: Dict[str, list] = defaultdict(list)

    for similarity, face_record in best_cluster["votes"]:
        img_path = Path(face_record.image_path)

        for person_folder in output_path.glob("person_*"):
            if person_folder.is_dir():
                for sorted_img in person_folder.glob("*"):
                    if (
                        sorted_img.stem in img_path.stem
                        or img_path.stem in sorted_img.stem
                    ):
                        folder_votes[person_folder.name].append(
                            (similarity, str(img_path))
                        )
                        break

    # Check unmatched folder
    unmatched_folder = output_path / "unmatched"
    if unmatched_folder.exists():
        for similarity, face_record in best_cluster["votes"]:
            img_path = Path(face_record.image_path)
            for sorted_img in unmatched_folder.glob("*"):
                if (
                    sorted_img.stem in img_path.stem
                    or img_path.stem in sorted_img.stem
                ):
                    folder_votes["unmatched"].append(
                        (similarity, str(img_path))
                    )
                    break

    if not folder_votes:
        print("Matches found but couldn't map to person folders")
        return None

    # Determine best match
    folder_scores: Dict[str, Dict[str, Any]] = {}
    for folder, votes in folder_votes.items():
        count = len(votes)
        avg_similarity = sum(s for s, _ in votes) / count
        max_similarity = max(s for s, _ in votes)
        folder_scores[folder] = {
            "count": count,
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "score": count * avg_similarity,
        }

    best_folder = max(folder_scores.items(), key=lambda x: x[1]["score"])
    folder_name, scores = best_folder

    print(f"\nPerson belongs to: {folder_name}/")
    print(f"   Confidence: {scores['avg_similarity']:.2%}")
    print(f"   Matches: {scores['count']} faces")

    return {
        "person_folder": folder_name,
        "confidence": scores["avg_similarity"],
        "match_count": scores["count"],
        "max_similarity": scores["max_similarity"],
        "all_scores": folder_scores,
    }
