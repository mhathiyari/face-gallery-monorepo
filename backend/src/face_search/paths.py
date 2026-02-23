"""
Canonical path resolution for Face Gallery.

Single source of truth for where data, config, and state files live.
Supports both dev mode (running from repo root) and installed mode (pip install).

All user-writable state goes under ~/.face-gallery/ (overridable via
$FACE_GALLERY_DATA_HOME). In dev mode, config falls back to <repo>/config/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _is_dev_mode() -> bool:
    """Detect whether we're running from a repo checkout vs pip install.

    Checks if the repo layout markers exist relative to this file's location.
    In a pip install, face_search lives in site-packages and the frontend/ dir
    won't be a sibling of the package root.
    """
    # backend/src/face_search/paths.py -> backend/src/face_search -> backend/src -> backend -> repo root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return (repo_root / "frontend" / "app.py").is_file()


def get_repo_root() -> Optional[Path]:
    """Return the repo root path in dev mode, None in installed mode."""
    if not _is_dev_mode():
        return None
    return Path(__file__).resolve().parent.parent.parent.parent


def get_data_home() -> Path:
    """Return the base directory for all Face Gallery user data.

    Default: ~/.face-gallery/
    Override: $FACE_GALLERY_DATA_HOME
    """
    env = os.environ.get("FACE_GALLERY_DATA_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".face-gallery"


def get_config_path() -> Path:
    """Return the path to config.json.

    Search order:
    1. $FACE_GALLERY_DATA_HOME/config.json (if exists)
    2. ~/.face-gallery/config.json (if exists)
    3. <repo>/config/config.json (dev mode, if exists)
    4. Falls back to ~/.face-gallery/config.json (will be created by init)
    """
    # Check user data home first
    data_home_config = get_data_home() / "config.json"
    if data_home_config.is_file():
        return data_home_config

    # In dev mode, check repo config
    repo_root = get_repo_root()
    if repo_root is not None:
        repo_config = repo_root / "config" / "config.json"
        if repo_config.is_file():
            return repo_config
        # Also try the example config
        example_config = repo_root / "config" / "config.example.json"
        if example_config.is_file():
            return example_config

    # Default location (may not exist yet)
    return data_home_config


def get_collections_json_path() -> Path:
    """Return the path to collections.json (UI state)."""
    # In dev mode, keep in frontend/ dir for backward compat
    repo_root = get_repo_root()
    if repo_root is not None:
        return repo_root / "frontend" / "collections.json"
    return get_data_home() / "collections.json"


def get_settings_json_path() -> Path:
    """Return the path to settings.json (UI state)."""
    repo_root = get_repo_root()
    if repo_root is not None:
        return repo_root / "frontend" / "settings.json"
    return get_data_home() / "settings.json"


def get_uploads_dir() -> Path:
    """Return the path to the uploads directory."""
    repo_root = get_repo_root()
    if repo_root is not None:
        return repo_root / "data" / "uploads"
    return get_data_home() / "data" / "uploads"


def get_default_photos_dir() -> Path:
    """Return the default photos directory."""
    repo_root = get_repo_root()
    if repo_root is not None:
        return repo_root / "data" / "photos"
    return get_data_home() / "data" / "photos"


def get_default_collections_dir() -> Path:
    """Return the default collections directory."""
    repo_root = get_repo_root()
    if repo_root is not None:
        return repo_root / "data" / "collections"
    return get_data_home() / "data" / "collections"


def ensure_data_home() -> Path:
    """Create the data home directory structure if it doesn't exist.

    Returns the data home path.
    """
    data_home = get_data_home()
    data_home.mkdir(parents=True, exist_ok=True)
    (data_home / "data" / "photos").mkdir(parents=True, exist_ok=True)
    (data_home / "data" / "collections").mkdir(parents=True, exist_ok=True)
    (data_home / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    return data_home
