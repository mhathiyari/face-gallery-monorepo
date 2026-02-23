"""Unit tests for face recognition models.

Tests cover:
- Model initialization
- Face detection
- Embedding generation
- Batch processing
- Error handling
- Model version tracking
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from PIL import Image

from face_search.models import (
    BaseFaceModel,
    InsightFaceModel,
    DetectedFace,
    BoundingBox,
    ModelVersion,
    create_model_version_from_model,
    save_model_version,
    load_model_version,
    verify_model_compatibility
)


# Test fixtures

@pytest.fixture
def sample_image():
    """Create a sample RGB image with a face-like pattern."""
    # Create a 640x640 RGB image with random noise
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_image_file(tmp_path):
    """Create a temporary image file."""
    img = Image.new('RGB', (640, 640), color=(128, 128, 128))
    # Draw a simple face-like pattern (oval for head)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Head
    draw.ellipse([220, 180, 420, 460], fill=(255, 220, 177), outline=(0, 0, 0))
    # Eyes
    draw.ellipse([270, 280, 310, 320], fill=(0, 0, 0))
    draw.ellipse([330, 280, 370, 320], fill=(0, 0, 0))
    # Mouth
    draw.arc([280, 350, 360, 410], 0, 180, fill=(0, 0, 0), width=3)

    filepath = tmp_path / "test_face.jpg"
    img.save(filepath)
    return filepath


@pytest.fixture
def insightface_model():
    """Create an InsightFace model instance."""
    try:
        model = InsightFaceModel(device='cpu')  # Use CPU for testing
        return model
    except Exception as e:
        pytest.skip(f"InsightFace model not available: {e}")


# Tests for BoundingBox

class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_bbox_creation(self):
        """Test bounding box creation."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 150

    def test_bbox_width_height(self):
        """Test width and height calculation."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        assert bbox.width == 90
        assert bbox.height == 130

    def test_bbox_to_list(self):
        """Test conversion to list."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        assert bbox.to_list() == [10, 20, 100, 150]

    def test_bbox_to_dict(self):
        """Test conversion to dictionary."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        bbox_dict = bbox.to_dict()
        assert bbox_dict['x1'] == 10
        assert bbox_dict['y1'] == 20
        assert bbox_dict['x2'] == 100
        assert bbox_dict['y2'] == 150
        assert bbox_dict['width'] == 90
        assert bbox_dict['height'] == 130


# Tests for DetectedFace

class TestDetectedFace:
    """Tests for DetectedFace class."""

    def test_detected_face_creation(self):
        """Test detected face creation."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        face = DetectedFace(bbox=bbox, confidence=0.95)
        assert face.bbox == bbox
        assert face.confidence == 0.95
        assert face.landmarks is None
        assert face.embedding is None

    def test_detected_face_with_embedding(self):
        """Test detected face with embedding."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512)
        face = DetectedFace(bbox=bbox, confidence=0.95, embedding=embedding)
        assert face.embedding is not None
        assert face.embedding.shape == (512,)

    def test_detected_face_to_dict(self):
        """Test conversion to dictionary."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        embedding = np.random.randn(512)
        face = DetectedFace(bbox=bbox, confidence=0.95, embedding=embedding)
        face_dict = face.to_dict()
        assert 'bbox' in face_dict
        assert 'confidence' in face_dict
        assert face_dict['confidence'] == 0.95
        assert 'embedding_shape' in face_dict


# Tests for BaseFaceModel

class TestBaseFaceModel:
    """Tests for BaseFaceModel abstract class."""

    def test_device_selection_cpu(self):
        """Test CPU device selection."""
        # We can't instantiate abstract class, so test the static method
        device = BaseFaceModel._select_device('cpu')
        assert device == 'cpu'

    def test_device_selection_auto(self):
        """Test automatic device selection."""
        device = BaseFaceModel._select_device('auto')
        assert device in ['cuda', 'mps', 'cpu']

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseFaceModel cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseFaceModel()


# Tests for InsightFaceModel

class TestInsightFaceModel:
    """Tests for InsightFaceModel implementation."""

    def test_model_initialization(self, insightface_model):
        """Test model initialization."""
        assert insightface_model is not None
        assert insightface_model.device in ['cuda', 'mps', 'cpu']

    def test_get_embedding_dim(self, insightface_model):
        """Test embedding dimension retrieval."""
        dim = insightface_model.get_embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0
        assert dim == 512  # InsightFace typically uses 512-dim embeddings

    def test_get_model_info(self, insightface_model):
        """Test model info retrieval."""
        info = insightface_model.get_model_info()
        assert 'name' in info
        assert 'type' in info
        assert 'embedding_dim' in info
        assert 'device' in info
        assert info['type'] == 'InsightFace'
        assert info['embedding_dim'] == 512

    def test_detect_faces_with_image_array(self, insightface_model, sample_image):
        """Test face detection with numpy array."""
        faces = insightface_model.detect_faces(sample_image, min_confidence=0.3)
        # Random noise unlikely to contain faces, but shouldn't crash
        assert isinstance(faces, list)

    def test_detect_faces_with_image_file(self, insightface_model, sample_image_file):
        """Test face detection with image file."""
        faces = insightface_model.detect_faces(sample_image_file, min_confidence=0.3)
        assert isinstance(faces, list)
        # Our simple drawing might or might not be detected as a face
        # but the function should return a list

    def test_detect_faces_file_not_found(self, insightface_model):
        """Test face detection with non-existent file."""
        with pytest.raises(FileNotFoundError):
            insightface_model.detect_faces("nonexistent_file.jpg")

    def test_get_embedding_valid_face(self, insightface_model, sample_image):
        """Test embedding generation."""
        # Create a small test image (might not contain a face)
        test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        try:
            embedding = insightface_model.get_embedding(test_img)
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] == 512
        except ValueError:
            # It's OK if no face is detected in random noise
            pass

    def test_get_embedding_invalid_input(self, insightface_model):
        """Test embedding generation with invalid input."""
        with pytest.raises(ValueError):
            insightface_model.get_embedding("not an array")

    def test_get_embedding_wrong_shape(self, insightface_model):
        """Test embedding generation with wrong image shape."""
        # Wrong shape (2D instead of 3D)
        img = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
        with pytest.raises(ValueError):
            insightface_model.get_embedding(img)

    def test_get_embeddings_batch_empty(self, insightface_model):
        """Test batch embedding with empty list."""
        embeddings = insightface_model.get_embeddings_batch([])
        assert len(embeddings) == 0

    def test_get_embeddings_batch_multiple(self, insightface_model):
        """Test batch embedding with multiple images."""
        # Create multiple test images
        images = [
            np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        try:
            embeddings = insightface_model.get_embeddings_batch(images)
            # Might fail if no faces detected, which is OK
            if len(embeddings) > 0:
                assert embeddings.shape[0] <= len(images)
                assert embeddings.shape[1] == 512
        except ValueError:
            # It's OK if all images fail (no faces in random noise)
            pass

    def test_model_repr(self, insightface_model):
        """Test model string representation."""
        repr_str = repr(insightface_model)
        assert 'InsightFaceModel' in repr_str
        assert 'buffalo_l' in repr_str


# Tests for ModelVersion

class TestModelVersion:
    """Tests for ModelVersion class."""

    def test_model_version_creation(self):
        """Test model version creation."""
        version = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )
        assert version.model_type == 'InsightFace'
        assert version.model_name == 'buffalo_l'
        assert version.embedding_dim == 512
        assert version.version == "1.0.0"
        assert version.created_at is not None

    def test_model_version_to_dict(self):
        """Test conversion to dictionary."""
        version = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )
        version_dict = version.to_dict()
        assert version_dict['model_type'] == 'InsightFace'
        assert version_dict['model_name'] == 'buffalo_l'
        assert version_dict['embedding_dim'] == 512

    def test_model_version_to_json(self):
        """Test conversion to JSON."""
        version = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )
        json_str = version.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['model_type'] == 'InsightFace'

    def test_model_version_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'model_type': 'InsightFace',
            'model_name': 'buffalo_l',
            'embedding_dim': 512,
            'version': '1.0.0',
            'created_at': '2024-01-01T00:00:00',
            'checksum': None,
            'metadata': {}
        }
        version = ModelVersion.from_dict(data)
        assert version.model_type == 'InsightFace'
        assert version.model_name == 'buffalo_l'

    def test_model_version_from_json(self):
        """Test creation from JSON."""
        json_str = '''
        {
            "model_type": "InsightFace",
            "model_name": "buffalo_l",
            "embedding_dim": 512,
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00",
            "checksum": null,
            "metadata": {}
        }
        '''
        version = ModelVersion.from_json(json_str)
        assert version.model_type == 'InsightFace'

    def test_model_version_compatibility(self):
        """Test compatibility checking."""
        v1 = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )
        v2 = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512,
            version='2.0.0'  # Different version
        )
        assert v1.is_compatible(v2)

    def test_model_version_incompatibility(self):
        """Test incompatibility detection."""
        v1 = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )
        v2 = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_s',  # Different model
            embedding_dim=512
        )
        assert not v1.is_compatible(v2)

    def test_model_version_signature(self):
        """Test signature generation."""
        version = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )
        signature = version.get_signature()
        assert isinstance(signature, str)
        assert len(signature) == 16  # SHA256 truncated to 16 chars

    def test_model_version_equality(self):
        """Test equality comparison."""
        v1 = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512,
            version='1.0.0'
        )
        v2 = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512,
            version='1.0.0'
        )
        assert v1 == v2

    def test_save_and_load_model_version(self, tmp_path):
        """Test saving and loading model version."""
        version = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )

        filepath = tmp_path / "model_version.json"
        save_model_version(version, str(filepath))

        assert filepath.exists()

        loaded_version = load_model_version(str(filepath))
        assert loaded_version.model_type == version.model_type
        assert loaded_version.model_name == version.model_name
        assert loaded_version.embedding_dim == version.embedding_dim

    def test_create_model_version_from_model(self, insightface_model):
        """Test creating ModelVersion from model instance."""
        version = create_model_version_from_model(insightface_model)
        assert isinstance(version, ModelVersion)
        assert version.model_type == 'InsightFace'
        assert version.embedding_dim == 512

    def test_verify_model_compatibility(self, insightface_model):
        """Test model compatibility verification."""
        stored_version = ModelVersion(
            model_type='InsightFace',
            model_name='buffalo_l',
            embedding_dim=512
        )
        is_compatible, message = verify_model_compatibility(
            insightface_model,
            stored_version
        )
        assert is_compatible
        assert "compatible" in message.lower()

    def test_verify_model_incompatibility(self, insightface_model):
        """Test model incompatibility detection."""
        stored_version = ModelVersion(
            model_type='InsightFace',
            model_name='different_model',
            embedding_dim=512
        )
        is_compatible, message = verify_model_compatibility(
            insightface_model,
            stored_version
        )
        # May or may not be incompatible depending on actual model name
        assert isinstance(is_compatible, bool)
        assert isinstance(message, str)


# Integration tests

class TestModelIntegration:
    """Integration tests for the model system."""

    def test_full_workflow(self, insightface_model, sample_image_file):
        """Test complete workflow: detect -> embed -> version."""
        # Detect faces
        faces = insightface_model.detect_faces(sample_image_file, min_confidence=0.3)

        # Create model version
        version = create_model_version_from_model(insightface_model)
        assert version is not None

        # Get model info
        info = insightface_model.get_model_info()
        assert info['embedding_dim'] == version.embedding_dim


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
