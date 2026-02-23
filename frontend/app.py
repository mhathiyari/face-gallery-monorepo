from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3

from flask import Flask, jsonify, request, send_file
from PIL import Image
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Import unified config system
from config_loader import get_config


APP_ROOT = Path(__file__).parent.resolve()

# Legacy paths for backward compatibility
CONFIG_PATH = APP_ROOT / "config.json"
COLLECTIONS_PATH = APP_ROOT / "collections.json"
SETTINGS_PATH = APP_ROOT / "settings.json"

# UPLOADS_DIR will be set from config after loading
UPLOADS_DIR = APP_ROOT / "uploads"  # Default, will be updated

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@dataclass
class Job:
    job_id: str
    command: List[str]
    status: str = "queued"
    logs: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, command: List[str]) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, command=command)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def run_job(self, job: Job) -> None:
        def _run() -> None:
            job.status = "running"
            job.started_at = time.time()
            try:
                process = subprocess.Popen(
                    job.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    self._append_log(job, line.rstrip())
                return_code = process.wait()
                job.status = "completed" if return_code == 0 else "failed"
            except Exception as exc:  # pragma: no cover - defensive
                self._append_log(job, f"Job error: {exc}")
                job.status = "failed"
            finally:
                job.finished_at = time.time()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _append_log(self, job: Job, line: str) -> None:
        with self._lock:
            job.logs.append(line)
            if len(job.logs) > 2000:
                job.logs = job.logs[-2000:]


job_manager = JobManager()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_config() -> Dict[str, Any]:
    """
    Load unified config and transform to format compatible with existing code.
    Returns a flattened config dict for backward compatibility.
    """
    # Load from unified config system
    unified_config = get_config()

    # root_dir is a security boundary - defaults to home directory to allow access to user files
    # This can be configured in config.json under "security" -> "root_dir"
    default_root_dir = unified_config.get("security", {}).get("root_dir", str(Path.home()))

    # Transform to flat structure for backward compatibility
    config = {
        # Map new paths to old keys
        "root_dir": default_root_dir,
        "face_search_root": unified_config.get("paths", {}).get("backend_root", "./backend"),
        "drive_credentials_path": unified_config.get("drive", {}).get("credentials_path", str(APP_ROOT / "credentials.json")),
        "drive_token_path": unified_config.get("drive", {}).get("token_path", str(APP_ROOT / "token.json")),

        # Keep full config for new features
        "_unified": unified_config
    }

    return config


def _get_uploads_dir() -> Path:
    """Get uploads directory from config."""
    config = _load_config()
    unified = config.get("_unified", {})
    uploads_path = unified.get("paths", {}).get("uploads_dir", str(APP_ROOT / "uploads"))
    uploads_dir = Path(uploads_path).expanduser()
    uploads_dir.mkdir(parents=True, exist_ok=True)
    return uploads_dir


def _ensure_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Path must be within configured root_dir") from exc


def _collection_store() -> Dict[str, Any]:
    return _read_json(COLLECTIONS_PATH, {"collections": []})


def _save_collection_store(payload: Dict[str, Any]) -> None:
    _write_json(COLLECTIONS_PATH, payload)


def _settings() -> Dict[str, Any]:
    return _read_json(
        SETTINGS_PATH,
        {
            "active_collection_id": None,
            "last_input_folder": "",
            "last_output_folder": "",
        },
    )


def _save_settings(payload: Dict[str, Any]) -> None:
    _write_json(SETTINGS_PATH, payload)


def _get_active_collection() -> Optional[Dict[str, Any]]:
    settings = _settings()
    collection_id = settings.get("active_collection_id")
    if not collection_id:
        return None
    for collection in _collection_store()["collections"]:
        if collection["id"] == collection_id:
            return collection
    return None


def _labels_path(collection_path: Path) -> Path:
    return collection_path / "labels.json"


def _load_labels(collection_path: Path) -> Dict[str, str]:
    return _read_json(_labels_path(collection_path), {})


def _save_labels(collection_path: Path, labels: Dict[str, str]) -> None:
    _write_json(_labels_path(collection_path), labels)


def _person_rep_path(person_folder: Path) -> Path:
    return person_folder / "representative.json"


def _hidden_person_rep_path(person_folder: Path) -> Path:
    return person_folder / ".representative.json"


def _load_person_rep_data(person_folder: Path) -> Optional[Dict[str, Any]]:
    rep_path = _person_rep_path(person_folder)
    hidden_path = _hidden_person_rep_path(person_folder)
    if rep_path.exists():
        data = _read_json(rep_path, None)
    elif hidden_path.exists():
        data = _read_json(hidden_path, None)
    else:
        return None
    return data if isinstance(data, dict) else None


def _load_person_rep(person_folder: Path) -> Optional[str]:
    data = _load_person_rep_data(person_folder)
    if isinstance(data, dict) and data:
        for key in ("representative", "filename", "file", "name", "image_name"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        if len(data) == 1:
            only_value = next(iter(data.values()))
            if isinstance(only_value, str) and only_value:
                return only_value
    return None


def _save_person_rep(person_folder: Path, filename: str) -> None:
    _write_json(_person_rep_path(person_folder), {"representative": filename})


def _image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def _shares_path(collection_path: Path) -> Path:
    return collection_path / "shares.json"


def _load_shares(collection_path: Path) -> Dict[str, Any]:
    return _read_json(_shares_path(collection_path), {"collection_link": None, "people": {}})


def _save_shares(collection_path: Path, shares: Dict[str, Any]) -> None:
    _write_json(_shares_path(collection_path), shares)


def _slugify(text: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "collection"


def _unique_collection_id(base: str, existing: List[str]) -> str:
    if base not in existing:
        return base
    counter = 2
    while f"{base}_{counter}" in existing:
        counter += 1
    return f"{base}_{counter}"


def _collection_stats(collection_path: Path) -> Dict[str, int]:
    person_folders = [p for p in collection_path.iterdir() if p.is_dir() and p.name.startswith("person_")]
    image_count = 0
    for folder in person_folders:
        image_count += len(_image_files(folder))
    return {"person_count": len(person_folders), "image_count": image_count}


def _cluster_id_from_person(person_id: str) -> Optional[int]:
    if not person_id.startswith("person_"):
        return None
    suffix = person_id.split("_", 1)[1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _cluster_faces(collection_path: Path, cluster_id: int) -> List[Dict[str, Any]]:
    db_path = collection_path / "collections" / "image_sorter" / "metadata.db"
    if not db_path.exists():
        return []
    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                SELECT image_path, bbox, confidence
                FROM faces
                WHERE collection_id = ? AND cluster_id = ? AND deleted = 0
                """,
                ("image_sorter", cluster_id),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            return []
    finally:
        connection.close()

    faces = []
    for image_path, bbox_json, confidence in rows:
        try:
            bbox = json.loads(bbox_json)
        except json.JSONDecodeError:
            continue
        faces.append(
            {
                "image_path": image_path,
                "bbox": bbox,
                "confidence": confidence,
            }
        )
    return faces


def _match_bbox_for_file(collection_path: Path, cluster_id: int, filename: str) -> Optional[Dict[str, Any]]:
    faces = _cluster_faces(collection_path, cluster_id)
    if not faces:
        return None
    file_path = Path(filename)
    file_stem = file_path.stem
    best_match = None
    best_conf = -1.0

    for face in faces:
        image_name = Path(face["image_path"]).name
        image_stem = Path(image_name).stem
        if image_name == filename or file_stem.startswith(image_stem) or image_stem.startswith(file_stem):
            conf = float(face["confidence"])
            if conf > best_conf:
                best_conf = conf
                best_match = face["bbox"]

    return best_match


def _drive_service(config: Dict[str, Any]):
    creds = None
    token_path = Path(config["drive_token_path"]).expanduser()
    credentials_path = Path(config["drive_credentials_path"]).expanduser()
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), DRIVE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Drive credentials not found at {credentials_path}. "
                    "Download OAuth client credentials and save as credentials.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return build("drive", "v3", credentials=creds)


def _drive_create_folder(service, name: str, parent_id: Optional[str]) -> str:
    metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        metadata["parents"] = [parent_id]
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def _drive_upload_file(service, file_path: Path, parent_id: str) -> str:
    media = MediaFileUpload(str(file_path), resumable=True)
    metadata = {"name": file_path.name, "parents": [parent_id]}
    uploaded = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return uploaded["id"]


def _drive_set_anyone_read(service, file_id: str) -> None:
    permission = {"type": "anyone", "role": "reader"}
    service.permissions().create(fileId=file_id, body=permission).execute()


def _drive_folder_link(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def _find_person_folder_function(script_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("find_person_folder", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load find_person_folder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_person_folder


@app.route("/")
def index() -> Any:
    return app.send_static_file("index.html")


@app.route("/api/config", methods=["GET", "POST"])
def config() -> Any:
    if request.method == "GET":
        return jsonify(_load_config())
    payload = request.get_json(force=True)
    config_data = _load_config()
    if "root_dir" in payload:
        config_data["root_dir"] = payload["root_dir"]
    if "face_search_root" in payload:
        config_data["face_search_root"] = payload["face_search_root"]
    _write_json(CONFIG_PATH, config_data)
    return jsonify(config_data)


@app.route("/api/collections", methods=["GET", "POST"])
def collections() -> Any:
    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()

    if request.method == "GET":
        store = _collection_store()
        response = []
        for entry in store["collections"]:
            path = Path(entry["path"]).expanduser()
            stats = {"person_count": 0, "image_count": 0}
            if path.exists():
                stats = _collection_stats(path)
            response.append({**entry, **stats})
        return jsonify(response)

    payload = request.get_json(force=True)
    path = Path(payload.get("path", "")).expanduser()
    if not path.exists():
        return jsonify({"error": "Collection path not found"}), 400
    _ensure_within_root(path, root_dir)

    store = _collection_store()
    existing_ids = [c["id"] for c in store["collections"]]
    base_id = _slugify(path.name)
    collection_id = _unique_collection_id(base_id, existing_ids)
    entry = {
        "id": collection_id,
        "name": path.name,
        "path": str(path),
    }
    store["collections"].append(entry)
    _save_collection_store(store)

    settings = _settings()
    if not settings.get("active_collection_id"):
        settings["active_collection_id"] = collection_id
        _save_settings(settings)

    return jsonify(entry), 201


@app.route("/api/collections/<collection_id>", methods=["DELETE"])
def delete_collection(collection_id: str) -> Any:
    store = _collection_store()
    collections_list = store["collections"]
    existing = next((c for c in collections_list if c["id"] == collection_id), None)
    if not existing:
        return jsonify({"error": "Unknown collection id"}), 404

    store["collections"] = [c for c in collections_list if c["id"] != collection_id]
    _save_collection_store(store)

    settings = _settings()
    if settings.get("active_collection_id") == collection_id:
        next_active = store["collections"][0]["id"] if store["collections"] else None
        settings["active_collection_id"] = next_active
        _save_settings(settings)

    return jsonify({"status": "deleted", "active_collection_id": settings.get("active_collection_id")})


@app.route("/api/collections/active", methods=["GET", "POST"])
def active_collection() -> Any:
    if request.method == "GET":
        active = _get_active_collection()
        return jsonify(active)

    payload = request.get_json(force=True)
    collection_id = payload.get("id")
    if not collection_id:
        return jsonify({"error": "Missing collection id"}), 400

    store = _collection_store()
    if not any(c["id"] == collection_id for c in store["collections"]):
        return jsonify({"error": "Unknown collection id"}), 404

    settings = _settings()
    settings["active_collection_id"] = collection_id
    _save_settings(settings)
    return jsonify({"active_collection_id": collection_id})


@app.route("/api/settings", methods=["GET"])
def settings() -> Any:
    return jsonify(_settings())


@app.route("/api/people", methods=["GET"])
def people() -> Any:
    active = _get_active_collection()
    if not active:
        return jsonify([])

    collection_path = Path(active["path"]).expanduser()
    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    _ensure_within_root(collection_path, root_dir)
    if not collection_path.exists():
        return jsonify([])

    labels = _load_labels(collection_path)
    people = []
    for folder in sorted(collection_path.iterdir()):
        if not folder.is_dir():
            continue
        if not (folder.name.startswith("person_") or folder.name == "unmatched"):
            continue
        images = _image_files(folder)
        rep_name = _load_person_rep(folder)
        rep_source = "representative"
        if rep_name and not (folder / rep_name).exists():
            rep_name = None
        if not rep_name:
            cluster_id = _cluster_id_from_person(folder.name)
            if cluster_id is not None:
                rep_source = "cluster"
            else:
                rep_source = "fallback"
        if not rep_name and images:
            rep_name = images[0].name
        label = labels.get(folder.name, folder.name)
        people.append(
            {
                "id": folder.name,
                "label": label,
                "count": len(images),
                "representative": rep_name,
                "representative_source": rep_source,
            }
        )

    sort_mode = request.args.get("sort", "name")
    if sort_mode == "size":
        people.sort(key=lambda item: item["count"], reverse=True)
    else:
        people.sort(key=lambda item: item["label"].lower())

    return jsonify(people)


@app.route("/api/shares", methods=["GET"])
def shares() -> Any:
    active = _get_active_collection()
    if not active:
        return jsonify({"collection_link": None, "people": {}})

    collection_path = Path(active["path"]).expanduser()
    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    _ensure_within_root(collection_path, root_dir)
    return jsonify(_load_shares(collection_path))


@app.route("/api/person/<person_id>", methods=["GET"])
def person_detail(person_id: str) -> Any:
    active = _get_active_collection()
    if not active:
        return jsonify({"error": "No active collection"}), 400

    collection_path = Path(active["path"]).expanduser()
    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    _ensure_within_root(collection_path, root_dir)
    person_folder = collection_path / person_id
    if not person_folder.exists():
        return jsonify({"error": "Person folder not found"}), 404

    page = max(int(request.args.get("page", 1)), 1)
    page_size = max(int(request.args.get("page_size", 60)), 1)

    images = _image_files(person_folder)
    total = len(images)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = images[start:end]

    return jsonify(
        {
            "id": person_id,
            "total": total,
            "page": page,
            "page_size": page_size,
            "images": [p.name for p in page_items],
        }
    )


@app.route("/api/person/<person_id>/label", methods=["POST"])
def update_label(person_id: str) -> Any:
    active = _get_active_collection()
    if not active:
        return jsonify({"error": "No active collection"}), 400

    payload = request.get_json(force=True)
    label = payload.get("label", "").strip()
    if not label:
        return jsonify({"error": "Label cannot be empty"}), 400

    collection_path = Path(active["path"]).expanduser()
    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    _ensure_within_root(collection_path, root_dir)
    labels = _load_labels(collection_path)
    labels[person_id] = label
    _save_labels(collection_path, labels)

    return jsonify({"id": person_id, "label": label})


@app.route("/api/person/<person_id>/representative", methods=["POST"])
def update_representative(person_id: str) -> Any:
    active = _get_active_collection()
    if not active:
        return jsonify({"error": "No active collection"}), 400

    payload = request.get_json(force=True)
    filename = payload.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "Filename cannot be empty"}), 400

    collection_path = Path(active["path"]).expanduser()
    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    _ensure_within_root(collection_path, root_dir)
    person_folder = collection_path / person_id
    if not (person_folder / filename).exists():
        return jsonify({"error": "File not found"}), 404

    _save_person_rep(person_folder, filename)

    return jsonify({"id": person_id, "representative": filename})


@app.route("/api/jobs", methods=["GET"])
def list_jobs() -> Any:
    jobs = []
    for job in job_manager.list_jobs():
        jobs.append(
            {
                "id": job.job_id,
                "status": job.status,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            }
        )
    return jsonify(jobs)


@app.route("/api/jobs/<job_id>", methods=["GET"])
def job_status(job_id: str) -> Any:
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(
        {
            "id": job.job_id,
            "status": job.status,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "logs": job.logs,
        }
    )


@app.route("/api/jobs/sort", methods=["POST"])
def sort_job() -> Any:
    payload = request.get_json(force=True)
    input_folder = payload.get("input_folder")
    output_folder = payload.get("output_folder")
    if not input_folder or not output_folder:
        return jsonify({"error": "Missing input or output folder"}), 400

    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    input_path = Path(input_folder).expanduser()
    output_path = Path(output_folder).expanduser()
    _ensure_within_root(input_path, root_dir)
    _ensure_within_root(output_path, root_dir)

    face_search_root = Path(config_data["face_search_root"]).expanduser()
    script_path = face_search_root / "examples" / "sort_images_by_person.py"
    if not script_path.exists():
        return jsonify({"error": "sort_images_by_person.py not found"}), 400

    output_path.mkdir(parents=True, exist_ok=True)

    command = ["python", str(script_path), str(input_path), str(output_path)]
    job = job_manager.create_job(command)
    job_manager.run_job(job)

    settings = _settings()
    settings["last_input_folder"] = str(input_path)
    settings["last_output_folder"] = str(output_path)
    _save_settings(settings)

    return jsonify({"job_id": job.job_id}), 202


@app.route("/api/drive/upload", methods=["POST"])
def drive_upload() -> Any:
    active = _get_active_collection()
    if not active:
        return jsonify({"error": "No active collection"}), 400

    payload = request.get_json(force=True)
    mode = payload.get("mode", "gallery")
    if mode not in {"gallery", "people"}:
        return jsonify({"error": "Invalid mode"}), 400

    collection_path = Path(active["path"]).expanduser()
    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    _ensure_within_root(collection_path, root_dir)

    service = _drive_service(config_data)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_name = f"{active['name']}_{timestamp}"
    root_id = _drive_create_folder(service, root_name, None)

    shares_payload = {"collection_link": None, "people": {}}

    person_folders = [p for p in collection_path.iterdir() if p.is_dir() and (p.name.startswith("person_") or p.name == "unmatched")]
    person_folders.sort(key=lambda p: p.name)

    for folder in person_folders:
        folder_id = _drive_create_folder(service, folder.name, root_id)
        for image_path in _image_files(folder):
            _drive_upload_file(service, image_path, folder_id)

        if mode == "people":
            _drive_set_anyone_read(service, folder_id)
            shares_payload["people"][folder.name] = _drive_folder_link(folder_id)

    _drive_set_anyone_read(service, root_id)
    shares_payload["collection_link"] = _drive_folder_link(root_id)
    _save_shares(collection_path, shares_payload)

    return jsonify(
        {
            "collection_link": shares_payload["collection_link"],
            "people_count": len(shares_payload["people"]),
        }
    )


@app.route("/api/search", methods=["POST"])
def search() -> Any:
    active = _get_active_collection()
    if not active:
        return jsonify({"error": "No active collection"}), 400

    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Missing filename"}), 400

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = UPLOADS_DIR / filename
    file.save(upload_path)

    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    _ensure_within_root(Path(active["path"]).expanduser(), root_dir)
    face_search_root = Path(config_data["face_search_root"]).expanduser()
    script_path = face_search_root / "examples" / "find_person_folder_fixed.py"
    if not script_path.exists():
        return jsonify({"error": "find_person_folder.py not found"}), 400

    find_person_folder = _find_person_folder_function(script_path)

    try:
        result = find_person_folder(
            str(upload_path),
            active["path"],
            top_k=20,
            min_cosine_similarity=0.6,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if not result:
        return jsonify({"match": None})

    scores = result.get("all_scores", {})
    top_matches = sorted(
        (
            {
                "folder": folder,
                "count": int(stats["count"]),
                "avg_similarity": float(stats["avg_similarity"]),
            }
            for folder, stats in scores.items()
        ),
        key=lambda item: item["count"] * item["avg_similarity"],
        reverse=True,
    )[:3]

    return jsonify(
        {
            "match": {
                "person_folder": result.get("person_folder"),
                "confidence": float(result.get("confidence")) if result.get("confidence") is not None else None,
                "match_count": int(result.get("match_count")) if result.get("match_count") is not None else 0,
                "max_similarity": float(result.get("max_similarity")) if result.get("max_similarity") is not None else None,
            },
            "top_matches": top_matches,
        }
    )


@app.route("/media/<collection_id>/<person_id>/<filename>")
def media(collection_id: str, person_id: str, filename: str) -> Any:
    store = _collection_store()
    collection = next((c for c in store["collections"] if c["id"] == collection_id), None)
    if not collection:
        return jsonify({"error": "Collection not found"}), 404

    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    collection_path = Path(collection["path"]).expanduser()
    _ensure_within_root(collection_path, root_dir)
    file_path = collection_path / person_id / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path)


@app.route("/face/<collection_id>/<person_id>/<filename>")
def face_crop(collection_id: str, person_id: str, filename: str) -> Any:
    store = _collection_store()
    collection = next((c for c in store["collections"] if c["id"] == collection_id), None)
    if not collection:
        return jsonify({"error": "Collection not found"}), 404

    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    collection_path = Path(collection["path"]).expanduser()
    _ensure_within_root(collection_path, root_dir)

    file_path = collection_path / person_id / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    cluster_id = _cluster_id_from_person(person_id)
    if cluster_id is None:
        return send_file(file_path)

    bbox = _match_bbox_for_file(collection_path, cluster_id, filename)
    if not bbox:
        return send_file(file_path)

    try:
        with Image.open(file_path) as image:
            x1 = max(int(bbox.get("x1", 0)), 0)
            y1 = max(int(bbox.get("y1", 0)), 0)
            x2 = max(int(bbox.get("x2", image.width)), 0)
            y2 = max(int(bbox.get("y2", image.height)), 0)

            width = max(x2 - x1, 0)
            height = max(y2 - y1, 0)
            pad_x = int(width * 0.3)
            pad_y = int(height * 0.3)

            x1 = max(x1 - pad_x, 0)
            y1 = max(y1 - pad_y, 0)
            x2 = min(x2 + pad_x, image.width)
            y2 = min(y2 + pad_y, image.height)

            if x2 <= x1 or y2 <= y1:
                return send_file(file_path)

            face = image.crop((x1, y1, x2, y2))
            if face.mode != "RGB":
                face = face.convert("RGB")
            buffer = BytesIO()
            face.save(buffer, format="JPEG", quality=88)
            buffer.seek(0)
            return send_file(buffer, mimetype="image/jpeg")
    except Exception:
        return send_file(file_path)


@app.route("/face-rep/<collection_id>/<person_id>")
def face_representative(collection_id: str, person_id: str) -> Any:
    store = _collection_store()
    collection = next((c for c in store["collections"] if c["id"] == collection_id), None)
    if not collection:
        return jsonify({"error": "Collection not found"}), 404

    config_data = _load_config()
    root_dir = Path(config_data["root_dir"]).expanduser()
    collection_path = Path(collection["path"]).expanduser()
    _ensure_within_root(collection_path, root_dir)

    person_folder = collection_path / person_id
    if not person_folder.exists():
        return jsonify({"error": "Person folder not found"}), 404

    rep_data = _load_person_rep_data(person_folder)
    rep_name = _load_person_rep(person_folder)
    if rep_name:
        file_path = person_folder / rep_name
    else:
        images = _image_files(person_folder)
        if not images:
            return jsonify({"error": "No images"}), 404
        file_path = images[0]

    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    bbox = None
    if isinstance(rep_data, dict):
        bbox = rep_data.get("bbox")

    if not isinstance(bbox, dict):
        cluster_id = _cluster_id_from_person(person_id)
        if cluster_id is not None:
            bbox = _match_bbox_for_file(collection_path, cluster_id, file_path.name)

    if not isinstance(bbox, dict):
        response = send_file(file_path)
        response.headers["Cache-Control"] = "no-store"
        return response

    try:
        with Image.open(file_path) as image:
            x1 = max(int(bbox.get("x1", 0)), 0)
            y1 = max(int(bbox.get("y1", 0)), 0)
            x2 = max(int(bbox.get("x2", image.width)), 0)
            y2 = max(int(bbox.get("y2", image.height)), 0)
            x2 = min(x2, image.width)
            y2 = min(y2, image.height)
            if x2 <= x1 or y2 <= y1:
                response = send_file(file_path)
                response.headers["Cache-Control"] = "no-store"
                return response

            face = image.crop((x1, y1, x2, y2))
            if face.mode != "RGB":
                face = face.convert("RGB")
            buffer = BytesIO()
            face.save(buffer, format="JPEG", quality=88)
            buffer.seek(0)
            response = send_file(buffer, mimetype="image/jpeg")
            response.headers["Cache-Control"] = "no-store"
            return response
    except Exception:
        response = send_file(file_path)
        response.headers["Cache-Control"] = "no-store"
        return response


@app.route("/api/reset-uploads", methods=["POST"])
def reset_uploads() -> Any:
    if UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    config = _load_config()
    unified = config.get("_unified", {})

    # Get server config with environment variable overrides
    server_config = unified.get("server", {})
    host = os.environ.get("FACE_VIEWER_HOST", server_config.get("host", "0.0.0.0"))
    port = int(os.environ.get("FACE_VIEWER_PORT", server_config.get("port", 5050)))
    debug_env = os.environ.get("FACE_VIEWER_DEBUG", "").lower() in {"1", "true", "yes"}
    debug = debug_env or server_config.get("debug", False)

    print(f"Starting Face Gallery on {host}:{port}")
    print(f"Debug mode: {debug}")
    app.run(host=host, port=port, debug=debug)
