"""
setup.py for face-gallery monorepo.

Needed because the source tree does not follow the standard layout:
  - face_search lives under backend/src/face_search/
  - face_gallery_frontend lives under frontend/

pyproject.toml handles metadata; this file maps package dirs.
"""

from setuptools import setup, find_packages

setup(
    package_dir={
        "face_search": "backend/src/face_search",
        "face_gallery_frontend": "frontend",
    },
    packages=[
        "face_search",
        "face_search.cli",
        "face_search.models",
        "face_search.search",
        "face_search.storage",
        "face_gallery_frontend",
    ],
    package_data={
        "face_gallery_frontend": [
            "static/*",
            "static/**/*",
        ],
    },
)
