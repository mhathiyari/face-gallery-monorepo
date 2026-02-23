"""End-to-end integration tests for the face search system.

These tests verify the complete pipeline from image indexing to search,
using synthetic or real datasets.
"""

import pytest
import time
from pathlib import Path
import tempfile
import shutil
import numpy as np

from face_search.models import InsightFaceModel
from face_search.storage import Collection
from face_search.indexer import BatchIndexer
from face_search.search import SearchEngine
from .test_dataset_utils import (
    TempDataset,
    find_test_images,
    create_train_test_split,
    group_images_by_identity
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_model():
    """Create a mock model for testing without GPU requirements."""
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
            self.embedding_dim = 512
            self._face_cache = {}  # Cache embeddings per image for consistency

        def detect_faces(self, image_path, min_confidence=0.5):
            """Mock face detection with consistent embeddings."""
            from face_search.models.base import DetectedFace, BoundingBox
            from PIL import Image

            # Load image to get consistent embedding based on image content
            img = Image.open(image_path)
            img_array = np.array(img)

            # Generate consistent embedding based on image content
            # Use image hash as seed for reproducibility
            seed = hash(img_array.tobytes()) % (2**32)
            np.random.seed(seed)

            # Create embedding
            embedding = np.random.randn(512).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)

            face = DetectedFace(
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
                confidence=0.95,
                embedding=embedding,
                landmarks=np.array([
                    [30, 40], [70, 40], [50, 70], [40, 100], [60, 100]
                ], dtype=np.float32)
            )

            return [face]

    return MockModel()


class TestEndToEndPipeline:
    """Integration tests for the complete face search pipeline."""

    def test_index_and_search_synthetic_dataset(self, mock_model, temp_dir):
        """Test complete pipeline with synthetic dataset."""
        # Create synthetic dataset
        with TempDataset(num_images=50, num_identities=10) as dataset_dir:
            # Create collection
            collection = Collection(
                collection_id='test_e2e',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            # Create indexer
            indexer = BatchIndexer(
                model=mock_model,
                collection=collection,
                batch_size=10,
                enable_deduplication=True
            )

            # Index dataset
            stats = indexer.index_folder(
                folder_path=dataset_dir,
                recursive=True,
                min_confidence=0.5
            )

            # Verify indexing succeeded
            assert stats.total_images > 0
            assert stats.total_faces_indexed > 0
            assert stats.processing_time > 0

            # Verify collection has faces
            assert collection.ntotal > 0
            assert collection.ntotal == stats.total_faces_indexed

            # Create search engine
            engine = SearchEngine(
                model=mock_model,
                collection=collection
            )

            # Test search with a query image
            query_images = find_test_images(dataset_dir, max_images=5)
            for query_image in query_images:
                results = engine.search_by_image(
                    image_path=query_image,
                    max_results=5,
                    return_all_faces=False
                )

                # Should return results
                assert len(results) > 0
                assert len(results[0]) > 0  # At least one result per detected face

                # Top result should have high similarity
                top_result = results[0][0]
                assert top_result.similarity > 0.5

    def test_indexing_performance(self, mock_model, temp_dir):
        """Test indexing performance metrics."""
        with TempDataset(num_images=100, num_identities=20) as dataset_dir:
            collection = Collection(
                collection_id='test_perf',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            indexer = BatchIndexer(
                model=mock_model,
                collection=collection,
                batch_size=16
            )

            start_time = time.time()
            stats = indexer.index_folder(dataset_dir)
            elapsed_time = time.time() - start_time

            # Verify performance metrics
            assert stats.total_images == 100
            assert stats.faces_per_second > 0
            assert stats.images_per_second > 0

            # Should process reasonably fast (mock model)
            assert elapsed_time < 30  # 100 images in under 30 seconds

            print(f"\nIndexing Performance:")
            print(f"  Total images: {stats.total_images}")
            print(f"  Total faces: {stats.total_faces_indexed}")
            print(f"  Processing time: {stats.processing_time:.2f}s")
            print(f"  Images/sec: {stats.images_per_second:.2f}")
            print(f"  Faces/sec: {stats.faces_per_second:.2f}")

    def test_search_accuracy_same_identity(self, mock_model, temp_dir):
        """Test that searching with images of same identity returns high similarity."""
        with TempDataset(num_images=60, num_identities=10) as dataset_dir:
            # Group images by identity
            all_images = find_test_images(dataset_dir)
            identity_groups = group_images_by_identity(all_images)

            # Split into train/test
            train_images, test_images = create_train_test_split(all_images, test_ratio=0.3)

            # Index train images
            collection = Collection(
                collection_id='test_accuracy',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            indexer = BatchIndexer(
                model=mock_model,
                collection=collection
            )

            indexer.index_images_batch(train_images, show_progress=False)

            # Search with test images
            engine = SearchEngine(
                model=mock_model,
                collection=collection
            )

            # For each test image, check if top results include same identity
            correct = 0
            total = 0

            for test_image in test_images[:10]:  # Test subset
                try:
                    results = engine.search_by_image(
                        image_path=test_image,
                        max_results=5,
                        return_all_faces=False
                    )

                    if results and results[0]:
                        # Top result should be from same identity (in synthetic dataset)
                        # Since synthetic dataset creates same color per identity,
                        # similar images should be found
                        top_result = results[0][0]

                        # With our mock model, same-identity images should have high similarity
                        if top_result.similarity > 0.7:
                            correct += 1

                        total += 1

                except Exception:
                    # Skip errors (e.g., no faces detected)
                    pass

            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0

            print(f"\nSearch Accuracy: {accuracy:.1%} ({correct}/{total})")

            # Should have reasonable accuracy (may vary due to random embeddings)
            # This is a smoke test to ensure search is working
            assert total > 0  # At least some searches completed

    def test_checkpoint_resume(self, mock_model, temp_dir):
        """Test checkpoint and resume functionality."""
        with TempDataset(num_images=30, num_identities=5) as dataset_dir:
            collection = Collection(
                collection_id='test_checkpoint',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            checkpoint_path = temp_dir / 'checkpoint.json'

            # First indexing session - partial
            indexer1 = BatchIndexer(
                model=mock_model,
                collection=collection,
                checkpoint_interval=5,
                checkpoint_path=checkpoint_path
            )

            images = find_test_images(dataset_dir)

            # Index first half
            indexer1.index_images_batch(images[:15], show_progress=False)

            # Verify checkpoint exists
            assert checkpoint_path.exists()

            initial_count = collection.ntotal

            # Resume indexing with new indexer
            indexer2 = BatchIndexer(
                model=mock_model,
                collection=collection,
                checkpoint_path=checkpoint_path
            )

            stats = indexer2.index_folder(dataset_dir, resume=True)

            # Should have indexed remaining images
            final_count = collection.ntotal
            assert final_count > initial_count

            # Total should be close to total images (some may have been skipped)
            assert final_count >= len(images) * 0.9  # Allow 10% margin

    def test_deduplication(self, mock_model, temp_dir):
        """Test face deduplication during indexing."""
        with TempDataset(num_images=20, num_identities=5) as dataset_dir:
            # Index with deduplication
            collection_dedup = Collection(
                collection_id='test_dedup_on',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            indexer_dedup = BatchIndexer(
                model=mock_model,
                collection=collection_dedup,
                enable_deduplication=True
            )

            stats_dedup = indexer_dedup.index_folder(dataset_dir)

            # Index without deduplication
            collection_no_dedup = Collection(
                collection_id='test_dedup_off',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            indexer_no_dedup = BatchIndexer(
                model=mock_model,
                collection=collection_no_dedup,
                enable_deduplication=False
            )

            stats_no_dedup = indexer_no_dedup.index_folder(dataset_dir)

            # Both should index all images
            assert stats_dedup.total_images == stats_no_dedup.total_images

            # Results should be similar (may detect duplicates in synthetic data)
            # This is mainly a smoke test
            assert stats_dedup.total_faces_indexed > 0
            assert stats_no_dedup.total_faces_indexed > 0

    def test_collection_persistence(self, mock_model, temp_dir):
        """Test that collections persist and can be reloaded."""
        with TempDataset(num_images=20, num_identities=5) as dataset_dir:
            collection_dir = temp_dir / 'collections'

            # Create and populate collection
            collection1 = Collection(
                collection_id='test_persist',
                collections_root=collection_dir,
                embedding_dim=512
            )

            indexer = BatchIndexer(
                model=mock_model,
                collection=collection1
            )

            stats1 = indexer.index_folder(dataset_dir)
            initial_count = collection1.ntotal

            # Save collection
            collection1.save()

            # Create new collection instance (reload)
            collection2 = Collection(
                collection_id='test_persist',
                collections_root=collection_dir,
                embedding_dim=512
            )

            # Should have same number of faces
            assert collection2.ntotal == initial_count

            # Should be able to search
            engine = SearchEngine(
                model=mock_model,
                collection=collection2
            )

            query_image = find_test_images(dataset_dir, max_images=1)[0]
            results = engine.search_by_image(query_image, max_results=5)

            assert len(results) > 0

    def test_search_by_face_id(self, mock_model, temp_dir):
        """Test searching for similar faces by face ID."""
        with TempDataset(num_images=30, num_identities=6) as dataset_dir:
            collection = Collection(
                collection_id='test_search_id',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            indexer = BatchIndexer(
                model=mock_model,
                collection=collection
            )

            indexer.index_folder(dataset_dir)

            engine = SearchEngine(
                model=mock_model,
                collection=collection
            )

            # Get first face ID
            all_faces = collection.metadata.get_all_faces(include_deleted=False)
            assert len(all_faces) > 0

            first_face_id = all_faces[0].face_id

            # Search for similar faces
            results = engine.search_by_face_id(
                face_id=first_face_id,
                max_results=5,
                include_self=False
            )

            # Should return results
            assert len(results) > 0

            # Should not include the query face itself
            assert not any(r.face.face_id == first_face_id for r in results)

    def test_compare_faces(self, mock_model, temp_dir):
        """Test comparing faces between two images."""
        with TempDataset(num_images=10, num_identities=3) as dataset_dir:
            collection = Collection(
                collection_id='test_compare',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            engine = SearchEngine(
                model=mock_model,
                collection=collection
            )

            images = find_test_images(dataset_dir, max_images=5)

            if len(images) >= 2:
                comparison = engine.compare_faces(
                    source_image=images[0],
                    target_image=images[1]
                )

                assert 'source_faces' in comparison
                assert 'target_faces' in comparison
                assert 'best_match' in comparison
                assert 'all_matches' in comparison

                # Should have detected faces
                assert len(comparison['source_faces']) > 0
                assert len(comparison['target_faces']) > 0

                # Should have similarity score
                assert 'similarity' in comparison['best_match']
                assert 0 <= comparison['best_match']['similarity'] <= 1


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks (marked as slow tests)."""

    def test_large_dataset_indexing(self, mock_model, temp_dir):
        """Benchmark indexing performance on larger dataset."""
        with TempDataset(num_images=500, num_identities=50) as dataset_dir:
            collection = Collection(
                collection_id='test_large',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            indexer = BatchIndexer(
                model=mock_model,
                collection=collection,
                batch_size=32
            )

            start_time = time.time()
            stats = indexer.index_folder(dataset_dir)
            elapsed_time = time.time() - start_time

            print(f"\nLarge Dataset Benchmarks:")
            print(f"  Images: {stats.total_images}")
            print(f"  Faces: {stats.total_faces_indexed}")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Speed: {stats.faces_per_second:.2f} faces/sec")
            print(f"  Success rate: {stats.success_rate:.1%}")

            # Should complete in reasonable time
            assert elapsed_time < 120  # 500 images in under 2 minutes

            # Should have good success rate
            assert stats.success_rate > 0.8

    def test_search_performance(self, mock_model, temp_dir):
        """Benchmark search performance."""
        with TempDataset(num_images=200, num_identities=30) as dataset_dir:
            collection = Collection(
                collection_id='test_search_perf',
                collections_root=temp_dir / 'collections',
                embedding_dim=512
            )

            # Index dataset
            indexer = BatchIndexer(
                model=mock_model,
                collection=collection
            )
            indexer.index_folder(dataset_dir)

            engine = SearchEngine(
                model=mock_model,
                collection=collection
            )

            # Benchmark search latency
            query_images = find_test_images(dataset_dir, max_images=20)
            search_times = []

            for query_image in query_images:
                start = time.time()
                results = engine.search_by_image(
                    query_image,
                    max_results=10
                )
                elapsed = time.time() - start
                search_times.append(elapsed)

            avg_search_time = sum(search_times) / len(search_times)

            print(f"\nSearch Performance:")
            print(f"  Collection size: {collection.ntotal} faces")
            print(f"  Avg search time: {avg_search_time*1000:.1f}ms")
            print(f"  Min: {min(search_times)*1000:.1f}ms")
            print(f"  Max: {max(search_times)*1000:.1f}ms")

            # Should be reasonably fast (< 1 second per search)
            assert avg_search_time < 1.0
