"""Smoke tests for the frontend Flask application."""

import sys
from pathlib import Path

# Ensure frontend package is importable (dev mode without pip install)
_frontend_dir = str(Path(__file__).resolve().parent.parent / "frontend")
if _frontend_dir not in sys.path:
    sys.path.insert(0, _frontend_dir)

import pytest


@pytest.fixture
def client():
    # Try installed package first, fall back to dev-mode import
    try:
        from face_gallery_frontend.app import app
    except ImportError:
        from app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"html" in response.data.lower()


def test_collections_returns_json(client):
    response = client.get("/api/collections")
    assert response.status_code == 200
    assert response.content_type.startswith("application/json")
    data = response.get_json()
    assert isinstance(data, list)
