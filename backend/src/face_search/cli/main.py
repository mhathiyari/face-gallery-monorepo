"""
face-gallery CLI — manage and serve Face Gallery from the command line.

Usage:
    face-gallery serve [--host HOST] [--port PORT] [--debug]
    face-gallery sort <input> <output> [--eps EPS] [--min-samples N]
    face-gallery init
    face-gallery version
"""

from __future__ import annotations

import json
import sys

import click

from face_search import __version__


@click.group()
def cli() -> None:
    """Face Gallery — sort your photos by the people in them."""


@cli.command()
def version() -> None:
    """Print the installed version."""
    click.echo(f"face-gallery {__version__}")


@cli.command()
def init() -> None:
    """Create ~/.face-gallery/ with a default configuration."""
    from face_search.paths import ensure_data_home, get_config_path, get_data_home

    data_home = ensure_data_home()
    click.echo(f"Data directory: {data_home}")

    config_path = get_data_home() / "config.json"
    if config_path.exists():
        click.echo(f"Config already exists: {config_path}")
    else:
        # Write default config with absolute paths
        default_config = {
            "paths": {
                "photos_dir": str(data_home / "data" / "photos"),
                "collections_dir": str(data_home / "data" / "collections"),
                "uploads_dir": str(data_home / "data" / "uploads"),
                "backend_root": "",
            },
            "server": {
                "host": "0.0.0.0",
                "port": 5050,
                "debug": False,
            },
            "model": {
                "name": "buffalo_l",
                "device": "auto",
                "min_confidence": 0.5,
            },
            "clustering": {
                "eps": 0.6,
                "min_samples": 2,
            },
            "indexing": {
                "batch_size": 32,
                "enable_deduplication": True,
                "checkpoint_interval": 100,
            },
            "search": {
                "max_results": 10,
                "min_similarity": 0.8,
            },
            "drive": {
                "credentials_path": "./credentials.json",
                "token_path": "./token.json",
                "enabled": False,
            },
            "security": {
                "root_dir": "~",
            },
        }
        config_path.write_text(json.dumps(default_config, indent=2), encoding="utf-8")
        click.echo(f"Config created: {config_path}")

    click.echo("Initialization complete.")


@cli.command()
@click.option("--host", default=None, help="Bind address (default: from config or 0.0.0.0)")
@click.option("--port", default=None, type=int, help="Port (default: from config or 5050)")
@click.option("--debug", is_flag=True, default=False, help="Enable Flask debug mode")
def serve(host: str | None, port: int | None, debug: bool) -> None:
    """Start the Face Gallery web UI."""
    # Try importing from installed package first, then fall back to dev mode
    try:
        from face_gallery_frontend.app import app
        from face_gallery_frontend.config_loader import get_config
    except ImportError:
        # Dev mode fallback — frontend/ not installed as a package
        from face_search.paths import get_repo_root

        repo_root = get_repo_root()
        if repo_root is not None:
            frontend_dir = str(repo_root / "frontend")
            if frontend_dir not in sys.path:
                sys.path.insert(0, frontend_dir)
        try:
            from app import app  # type: ignore[no-redef]
            from config_loader import get_config  # type: ignore[no-redef]
        except ImportError:
            click.echo(
                "Error: Cannot find the Face Gallery frontend.\n"
                "Either install with `pip install face-gallery` or "
                "run from the repo root.",
                err=True,
            )
            raise SystemExit(1)

    config = get_config()
    server_config = config.get("server", {})

    final_host = host or server_config.get("host", "0.0.0.0")
    final_port = port or server_config.get("port", 5050)
    final_debug = debug or server_config.get("debug", False)

    click.echo(f"Starting Face Gallery on {final_host}:{final_port}")
    app.run(host=final_host, port=final_port, debug=final_debug)


@cli.command()
@click.argument("input_folder")
@click.argument("output_folder")
@click.option("--eps", default=0.5, type=float, help="DBSCAN epsilon (0.3 strict – 0.6 lenient)")
@click.option("--min-samples", default=2, type=int, help="Minimum faces per cluster")
@click.option("--min-confidence", default=0.5, type=float, help="Minimum face detection confidence")
@click.option("--model", "model_name", default="buffalo_l", help="InsightFace model name")
@click.option("--device", default="auto", help="Compute device (auto/cuda/mps/cpu)")
@click.option("--batch-size", default=16, type=int, help="Batch size for indexing")
def sort(
    input_folder: str,
    output_folder: str,
    eps: float,
    min_samples: int,
    min_confidence: float,
    model_name: str,
    device: str,
    batch_size: int,
) -> None:
    """Sort images by person identity.

    Scans INPUT_FOLDER for photos, detects and clusters faces,
    then organizes images into person folders under OUTPUT_FOLDER.
    """
    from face_search import sort_images_by_person

    result = sort_images_by_person(
        input_folder,
        output_folder,
        model_name=model_name,
        device=device,
        min_confidence=min_confidence,
        eps=eps,
        min_samples=min_samples,
        batch_size=batch_size,
    )

    if result["persons"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
