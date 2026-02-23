#!/usr/bin/env python3
"""
Simple example: Sort images in a folder by person/identity.

This script demonstrates using the face search system to:
1. Index all faces in a folder
2. Cluster faces by identity (same person)
3. Copy/organize images into folders by person

Usage:
    python examples/sort_images_by_person.py /path/to/photos /path/to/output
"""

import sys
from pathlib import Path
import shutil
import json
# Clustering parameters
metric_param = 'cosine'  # Distance metric: 'cosine' (recommended) or 'euclidean'
eps_param = 0.5  # For cosine: 0.3 (strict) to 0.6 (lenient). For euclidean: 15-25
min_samples_param = 2  # Minimum faces to form a cluster
# Add src to path so we can import without installing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_search.models import InsightFaceModel
from face_search.storage import Collection
from face_search.indexer import BatchIndexer
from face_search.search import (
    cluster_faces,
    find_representative_faces,
    select_best_representative,
    calculate_cluster_quality_metrics
)


def sort_images_by_person(input_folder: str, output_folder: str, min_confidence: float = 0.5):
    """Sort images by detected person/identity.

    Args:
        input_folder: Folder containing images to sort
        output_folder: Folder where sorted images will be organized
        min_confidence: Minimum face detection confidence (0-1)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"❌ Error: Input folder not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print("🚀 Face-based Image Sorting")
    print("=" * 50)

    # Step 1: Initialize face detection model
    print("\n📦 Loading face detection model...")
    try:
        model = InsightFaceModel(model_name='buffalo_l', device='auto')
        print(f"✅ Model loaded on device: {model.device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure InsightFace is installed: pip install insightface")
        return

    # Step 2: Create collection for storing faces
    print("\n💾 Creating face collection...")
    collection = Collection(
        collection_id='image_sorter',
        collections_root=output_path / 'collections',
        embedding_dim=512
    )
    print(f"✅ Collection created")

    # Step 3: Index all faces in the input folder
    print(f"\n🔍 Scanning and indexing faces in: {input_path}")
    indexer = BatchIndexer(
        model=model,
        collection=collection,
        batch_size=16,
        enable_deduplication=True
    )

    stats = indexer.index_folder(
        folder_path=input_path,
        recursive=True,
        min_confidence=min_confidence
    )

    print(f"\n📊 Indexing Results:")
    print(f"   Images processed: {stats.total_images}")
    print(f"   Faces found: {stats.total_faces_found}")
    print(f"   Faces indexed: {stats.total_faces_indexed}")
    print(f"   Duplicates skipped: {stats.duplicates_skipped}")
    print(f"   Processing time: {stats.processing_time:.2f}s")
    print(f"   Speed: {stats.faces_per_second:.1f} faces/sec")

    if stats.total_faces_indexed == 0:
        print("\n⚠️  No faces found. Make sure your images contain visible faces.")
        return

    # Step 4: Get all indexed faces
    print("\n🔗 Clustering faces by identity...")
    all_faces = []
    for face_record in collection.metadata.get_all_faces(include_deleted=False):
        from face_search.face import FaceDetection, BoundingBox

        face = FaceDetection(
            bbox=BoundingBox.from_dict(eval(face_record.bbox) if isinstance(face_record.bbox, str) else face_record.bbox),
            confidence=face_record.confidence,
            embedding=face_record.get_embedding(),
            image_path=Path(face_record.image_path),
            face_id=face_record.face_id,
            yaw=face_record.yaw,
            pitch=face_record.pitch,
            roll=face_record.roll
        )
        all_faces.append(face)

    # Step 5: Cluster faces by identity
    # eps: smaller = more strict
    # min_samples: minimum faces to form a cluster
    # metric: 'cosine' is scale-invariant and works better for unnormalized embeddings
    clusters = cluster_faces(
        faces=all_faces,
        eps=eps_param,  # Adjust at top of file
        min_samples=min_samples_param,  # Adjust at top of file
        metric=metric_param  # Adjust at top of file
    )

    # Remove noise cluster
    real_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise_faces = clusters.get(-1, [])

    print(f"✅ Found {len(real_clusters)} identities")
    print(f"   Noise (unmatched faces): {len(noise_faces)}")

    # Step 5.5: Save cluster assignments to database
    print(f"\n💾 Saving cluster assignments to database...")
    face_cluster_map = {}
    for cluster_id, faces_in_cluster in clusters.items():
        for face in faces_in_cluster:
            face_cluster_map[face.face_id] = cluster_id

    updated = collection.metadata.update_cluster_assignments(face_cluster_map)
    print(f"✅ Saved {updated} cluster assignments")

    # Step 6: Organize images by cluster
    print(f"\n📁 Organizing images into folders...")

    sorted_count = 0

    # Create folders for each person/cluster
    for cluster_id, faces_in_cluster in real_clusters.items():
        person_folder = output_path / f"person_{cluster_id:03d}"
        person_folder.mkdir(exist_ok=True)

        # Get unique images in this cluster
        images_in_cluster = set(str(f.image_path) for f in faces_in_cluster)

        # Copy images to person folder
        for img_path in images_in_cluster:
            src = Path(img_path)
            if src.exists():
                dst = person_folder / src.name

                # Handle duplicate filenames
                counter = 1
                while dst.exists():
                    dst = person_folder / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1

                shutil.copy2(src, dst)
                sorted_count += 1

        print(f"   Person {cluster_id}: {len(faces_in_cluster)} faces, {len(images_in_cluster)} images -> {person_folder.name}/")

    # Handle noise (single faces or unmatched)
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

        print(f"   Unmatched: {len(noise_faces)} faces, {len(noise_images)} images -> unmatched/")

    print(f"\n✅ Done! Sorted {sorted_count} images into {len(real_clusters)} person folders")
    print(f"📂 Results saved to: {output_path}")

    # Step 7: Generate .representative.json for each person folder
    print(f"\n📄 Generating representative metadata files...")
    for cluster_id, faces_in_cluster in real_clusters.items():
        # Calculate cluster quality metrics
        cluster_stats = calculate_cluster_quality_metrics(faces_in_cluster)

        # Select best representative using pose filtering
        rep, selection_reason = select_best_representative(faces_in_cluster, prefer_frontal=True)

        # Create representative metadata
        rep_metadata = {
            'face_id': rep.face_id,
            'image_name': rep.image_path.name,
            'image_path': str(rep.image_path),
            'bbox': {
                'x1': rep.bbox.x1,
                'y1': rep.bbox.y1,
                'x2': rep.bbox.x2,
                'y2': rep.bbox.y2
            },
            'confidence': float(rep.confidence),
            'yaw': float(rep.yaw) if rep.yaw is not None else None,
            'pitch': float(rep.pitch) if rep.pitch is not None else None,
            'roll': float(rep.roll) if rep.roll is not None else None,
            'selection_reason': selection_reason,
            'cluster_stats': cluster_stats
        }

        # Save to person folder
        person_folder = output_path / f"person_{cluster_id:03d}"
        rep_file = person_folder / '.representative.json'

        with open(rep_file, 'w') as f:
            json.dump(rep_metadata, f, indent=2)

        print(f"   Person {cluster_id}: {rep.image_path.name} (confidence: {rep.confidence:.2f}, {selection_reason})")

    print(f"✅ Generated {len(real_clusters)} representative metadata files")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sort_images_by_person.py <input_folder> <output_folder>")
        print("\nExample:")
        print("  python sort_images_by_person.py ~/Photos/party ~/Photos/sorted")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    sort_images_by_person(input_folder, output_folder)
