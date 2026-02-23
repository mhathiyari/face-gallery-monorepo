"""
GPU/MPS Setup Verification Tests

CRITICAL: Run these tests BEFORE building anything!
This ensures all GPU/MPS acceleration is properly configured.
"""

import pytest
import torch
import numpy as np


def test_python_version():
    """Verify Python version is 3.11+"""
    import sys
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def test_numpy_version():
    """Verify NumPy is installed and compatible"""
    assert np.__version__.startswith("1."), f"NumPy 1.x required, got {np.__version__}"
    print(f"✓ NumPy version: {np.__version__}")

    # Test basic NumPy operation
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.sum() == 15
    print(f"✓ NumPy basic operations work")


def test_pytorch_installation():
    """Verify PyTorch is installed"""
    print(f"✓ PyTorch version: {torch.__version__}")
    assert torch.__version__, "PyTorch not installed"


def test_mps_available():
    """
    Verify MPS (Metal Performance Shaders) is available on Apple Silicon

    MPS is the GPU acceleration backend for Apple Silicon Macs.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available (not on Apple Silicon Mac)")

    print(f"✓ MPS (Metal GPU) available: {torch.backends.mps.is_available()}")
    print(f"✓ MPS built: {torch.backends.mps.is_built()}")

    assert torch.backends.mps.is_available(), "MPS should be available on Apple Silicon"
    assert torch.backends.mps.is_built(), "MPS should be built"


def test_mps_tensor_operations():
    """Test basic tensor operations on MPS device"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = torch.device("mps")

    # Create tensor on MPS
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)

    # Perform operation
    z = torch.matmul(x, y)

    assert z.device.type == "mps"
    assert z.shape == (100, 100)
    print(f"✓ MPS tensor operations work: {z.shape} tensor on {z.device}")


def test_faiss_installation():
    """Verify FAISS is installed and working"""
    import faiss

    print(f"✓ FAISS version: {faiss.__version__}")

    # Test basic FAISS operations
    dimension = 128
    index = faiss.IndexFlatL2(dimension)

    # Add some vectors
    vectors = np.random.random((100, dimension)).astype('float32')
    index.add(vectors)

    # Search
    query = np.random.random((5, dimension)).astype('float32')
    distances, indices = index.search(query, k=10)

    assert index.ntotal == 100, f"Expected 100 vectors, got {index.ntotal}"
    assert distances.shape == (5, 10), f"Expected shape (5, 10), got {distances.shape}"
    print(f"✓ FAISS operations work: indexed {index.ntotal} vectors")


def test_insightface_installation():
    """Verify InsightFace is installed"""
    try:
        from insightface.app import FaceAnalysis
        print("✓ InsightFace imported successfully")

        # Note: We don't initialize the full model here to keep tests fast
        # Full model initialization tested in integration tests

    except ImportError as e:
        pytest.fail(f"InsightFace import failed: {e}")


def test_onnxruntime_installation():
    """Verify ONNX Runtime is installed"""
    import onnxruntime as ort

    print(f"✓ ONNX Runtime version: {ort.__version__}")

    # Check available execution providers
    providers = ort.get_available_providers()
    print(f"✓ ONNX Runtime providers: {providers}")

    # On Mac, we expect CPUExecutionProvider at minimum
    assert 'CPUExecutionProvider' in providers


def test_opencv_installation():
    """Verify OpenCV is installed"""
    import cv2

    print(f"✓ OpenCV version: {cv2.__version__}")

    # Test basic image operation
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 100)
    print(f"✓ OpenCV operations work")


def test_database_dependencies():
    """Verify SQLAlchemy and Alembic are installed"""
    import sqlalchemy
    import alembic

    print(f"✓ SQLAlchemy version: {sqlalchemy.__version__}")
    print(f"✓ Alembic version: {alembic.__version__}")


def test_cli_dependencies():
    """Verify CLI dependencies are installed"""
    import click
    import rich
    import tqdm

    print(f"✓ Click version: {click.__version__}")
    print(f"✓ Rich version: {rich.__version__}")
    print(f"✓ tqdm version: {tqdm.__version__}")


def test_web_dependencies():
    """Verify FastAPI and Uvicorn are installed"""
    import fastapi
    import uvicorn

    print(f"✓ FastAPI version: {fastapi.__version__}")
    print(f"✓ Uvicorn version: {uvicorn.__version__}")


@pytest.mark.slow
def test_insightface_model_download():
    """
    Test InsightFace model initialization (slow, downloads models)

    Run with: pytest -m slow tests/test_gpu_setup.py::test_insightface_model_download
    """
    from insightface.app import FaceAnalysis

    # Initialize model (will download if not present)
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1)  # Use CPU

    print(f"✓ InsightFace model initialized")
    print(f"✓ Models loaded: {len(app.models)} models")


def test_device_selection():
    """Test automatic device selection logic"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Selected device: MPS (Metal GPU)")
    else:
        device = torch.device("cpu")
        print(f"✓ Selected device: CPU")

    # Test tensor creation
    x = torch.randn(10, 10, device=device)
    assert x.device.type == device.type
    print(f"✓ Device selection works: {device}")


if __name__ == "__main__":
    """Run tests manually for quick verification"""
    print("=" * 60)
    print("GPU/MPS Setup Verification")
    print("=" * 60)
    print()

    tests = [
        test_python_version,
        test_numpy_version,
        test_pytorch_installation,
        test_mps_available,
        test_mps_tensor_operations,
        test_faiss_installation,
        test_insightface_installation,
        test_onnxruntime_installation,
        test_opencv_installation,
        test_database_dependencies,
        test_cli_dependencies,
        test_web_dependencies,
        test_device_selection,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_func in tests:
        test_name = test_func.__name__.replace("test_", "").replace("_", " ").title()
        try:
            print(f"\n{test_name}:")
            print("-" * 60)
            test_func()
            passed += 1
            print(f"✅ PASSED")
        except pytest.skip.Exception as e:
            skipped += 1
            print(f"⏭️  SKIPPED: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed > 0:
        exit(1)
