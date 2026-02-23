"""
Configuration loader for Face Gallery frontend.
Loads from unified config in ../config/ directory.
"""

import json
import os
from pathlib import Path


def load_config():
    """
    Load configuration from config/config.json.
    Falls back to config.example.json if config.json doesn't exist.
    """
    # Try to find config directory
    config_paths = [
        Path("/app/config/config.json"),  # Docker path
        Path(__file__).parent.parent / "config" / "config.json",  # Manual install
        Path.home() / "face-gallery" / "config" / "config.json",  # Alternative
    ]

    example_paths = [
        Path("/app/config/config.example.json"),
        Path(__file__).parent.parent / "config" / "config.example.json",
    ]

    # Try to load config.json
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"[CONFIG] Loaded config from: {config_path}")
            return _process_config(config)

    # Fall back to example config
    for example_path in example_paths:
        if example_path.exists():
            with open(example_path, 'r') as f:
                config = json.load(f)
            print(f"[CONFIG] Using example config from: {example_path}")
            print("[CONFIG] WARNING: Copy config.example.json to config.json and customize it!")
            return _process_config(config)

    # No config found, return defaults
    print("[CONFIG] WARNING: No config file found, using defaults")
    return get_default_config()


def _process_config(config):
    """
    Process config, expanding paths and applying defaults.
    """
    # Apply defaults for missing values
    defaults = get_default_config()

    # Merge with defaults
    for section, values in defaults.items():
        if section not in config:
            config[section] = values
        elif isinstance(values, dict):
            for key, default_value in values.items():
                if key not in config[section]:
                    config[section][key] = default_value

    # Expand paths
    if "paths" in config:
        for key, path in config["paths"].items():
            if path and isinstance(path, str):
                # Expand ~ and environment variables
                expanded = os.path.expanduser(os.path.expandvars(path))
                config["paths"][key] = expanded

    # Expand security root_dir
    if "security" in config and "root_dir" in config["security"]:
        root_dir = config["security"]["root_dir"]
        if root_dir and isinstance(root_dir, str):
            config["security"]["root_dir"] = os.path.expanduser(os.path.expandvars(root_dir))

    # Expand drive paths
    if "drive" in config:
        for key in ["credentials_path", "token_path"]:
            if key in config["drive"]:
                path = config["drive"][key]
                if path and isinstance(path, str):
                    config["drive"][key] = os.path.expanduser(os.path.expandvars(path))

    return config


def get_default_config():
    """
    Return default configuration.
    """
    return {
        "paths": {
            "photos_dir": "./data/photos",
            "collections_dir": "./data/collections",
            "uploads_dir": "./data/uploads",
            "backend_root": "./backend"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 5050,
            "debug": False
        },
        "model": {
            "name": "buffalo_l",
            "device": "auto",
            "min_confidence": 0.5
        },
        "clustering": {
            "eps": 0.6,
            "min_samples": 2
        },
        "indexing": {
            "batch_size": 32,
            "enable_deduplication": True,
            "checkpoint_interval": 100
        },
        "search": {
            "max_results": 10,
            "min_similarity": 0.8
        },
        "drive": {
            "credentials_path": "./credentials.json",
            "token_path": "./token.json",
            "enabled": False
        },
        "security": {
            "root_dir": "~"
        }
    }


def get_env_overrides():
    """
    Get configuration overrides from environment variables.
    """
    overrides = {}

    # Map environment variables to config paths
    env_map = {
        'PHOTOS_DIR': ('paths', 'photos_dir'),
        'COLLECTIONS_DIR': ('paths', 'collections_dir'),
        'UPLOADS_DIR': ('paths', 'uploads_dir'),
        'FACE_VIEWER_HOST': ('server', 'host'),
        'FACE_VIEWER_PORT': ('server', 'port'),
        'FACE_VIEWER_DEBUG': ('server', 'debug'),
        'MODEL_DEVICE': ('model', 'device'),
        'MODEL_NAME': ('model', 'name'),
        'BATCH_SIZE': ('indexing', 'batch_size'),
        'CLUSTERING_EPS': ('clustering', 'eps'),
        'CLUSTERING_MIN_SAMPLES': ('clustering', 'min_samples'),
    }

    for env_var, (section, key) in env_map.items():
        value = os.getenv(env_var)
        if value is not None:
            if section not in overrides:
                overrides[section] = {}

            # Type conversion
            if key == 'port':
                value = int(value)
            elif key == 'debug':
                value = value.lower() in ('1', 'true', 'yes')
            elif key in ('batch_size', 'min_samples'):
                value = int(value)
            elif key == 'eps':
                value = float(value)

            overrides[section][key] = value
            print(f"[CONFIG] Environment override: {env_var} -> {section}.{key} = {value}")

    return overrides


def get_config():
    """
    Load config and apply environment overrides.
    """
    config = load_config()
    overrides = get_env_overrides()

    # Apply overrides
    for section, values in overrides.items():
        if section in config:
            config[section].update(values)
        else:
            config[section] = values

    return config


if __name__ == "__main__":
    # Test config loading
    config = get_config()
    print("\n=== Loaded Configuration ===")
    print(json.dumps(config, indent=2))
