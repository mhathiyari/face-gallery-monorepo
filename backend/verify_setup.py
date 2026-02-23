#!/usr/bin/env python3
"""
Simple setup verification script
Run this to verify all dependencies are correctly installed
"""

import sys


def check_python():
    """Check Python version"""
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    assert sys.version_info >= (3, 11), "Python 3.11+ required"


def check_numpy():
    """Check NumPy"""
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
    # Quick test
    assert np.array([1, 2, 3]).sum() == 6


def check_torch():
    """Check PyTorch and MPS"""
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.backends.mps.is_available():
        print(f"✓ MPS (Metal GPU) available")
        # Test MPS operation
        x = torch.randn(10, 10, device="mps")
        assert x.device.type == "mps"
    else:
        print(f"⚠ MPS not available (using CPU)")


def check_faiss():
    """Check FAISS"""
    import faiss
    import numpy as np
    print(f"✓ FAISS {faiss.__version__}")
    # Quick test
    index = faiss.IndexFlatL2(128)
    vectors = np.random.random((10, 128)).astype('float32')
    index.add(vectors)
    assert index.ntotal == 10


def check_opencv():
    """Check OpenCV"""
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")


def check_insightface():
    """Check InsightFace"""
    from insightface.app import FaceAnalysis
    print(f"✓ InsightFace available")


def check_databases():
    """Check database libraries"""
    import sqlalchemy
    import alembic
    print(f"✓ SQLAlchemy {sqlalchemy.__version__}")
    print(f"✓ Alembic {alembic.__version__}")


def check_cli():
    """Check CLI libraries"""
    import click
    import rich
    print(f"✓ Click {click.__version__}")
    print(f"✓ Rich (installed)")


def check_web():
    """Check web frameworks"""
    import fastapi
    import uvicorn
    print(f"✓ FastAPI {fastapi.__version__}")
    print(f"✓ Uvicorn {uvicorn.__version__}")


def main():
    """Run all checks"""
    print("=" * 60)
    print("Face Search System - Setup Verification")
    print("=" * 60)
    print()

    checks = [
        ("Python", check_python),
        ("NumPy", check_numpy),
        ("PyTorch & MPS", check_torch),
        ("FAISS", check_faiss),
        ("OpenCV", check_opencv),
        ("InsightFace", check_insightface),
        ("Database (SQLAlchemy, Alembic)", check_databases),
        ("CLI (Click, Rich)", check_cli),
        ("Web (FastAPI, Uvicorn)", check_web),
    ]

    passed = 0
    failed = []

    for name, check_func in checks:
        try:
            print(f"\n{name}:")
            print("-" * 60)
            check_func()
            passed += 1
        except Exception as e:
            failed.append((name, str(e)))
            print(f"❌ FAILED: {e}")

    print()
    print("=" * 60)
    if failed:
        print(f"⚠️  {len(failed)} checks failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print(f"✅ {passed} checks passed")
        return 1
    else:
        print(f"✅ All {passed} checks passed!")
        print()
        print("🚀 System is ready for development!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
