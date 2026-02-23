# Face Search Viewer

Local web UI for sorting face collections, browsing people, searching by image, and sharing results.

## Features
- Run the face-sorting job against a folder of images.
- Switch between multiple sorted collections.
- People grid with representative face crop (from `.representative.json`).
- Person detail view with full images or face-crop toggle.
- Search-by-image using `find_person_folder_fixed.py`.
- Drive sharing: upload a gallery or per-person folders and copy share links.

## Requirements
- Python 3.10+ recommended.
- Face search engine repo at `~/gitworkspace/face-search-system` (or set in `config.json`).
- Google Drive OAuth credentials (optional, for sharing).

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U uv
uv pip install -r requirements.txt
```

If you will run sorting or search, install the face-search-system deps:

```bash
uv pip install -r requirements-face-search.txt
```

## Configure
Edit `config.json`:

```json
{
  "root_dir": "~",
  "face_search_root": "~/gitworkspace/face-search-system",
  "drive_credentials_path": "./credentials.json",
  "drive_token_path": "./token.json"
}
```

Notes:
- `root_dir` limits read/write operations for safety.
- `face_search_root` must point to the face-search-system repo.
- For Drive sharing, place `credentials.json` in this repo (downloaded from Google Cloud OAuth).

## Run

```bash
. .venv/bin/activate
FACE_VIEWER_PORT=5050 python app.py
```

Open `http://127.0.0.1:5050` in a browser.

Environment variables:
- `FACE_VIEWER_HOST` (default `0.0.0.0`)
- `FACE_VIEWER_PORT` (default `5000`)
- `FACE_VIEWER_DEBUG` (`1`/`true` to enable debug)

## Usage

1) **Add collection**
- Provide the path to a sorted folder (contains `person_###` folders + `collections/`).
- Click the row to make it active.

2) **Sort job (owner mode)**
- Toggle Owner mode.
- Set input/output folders.
- Start sorting (runs `sort_images_by_person.py`).

3) **People grid**
- Displays representative faces from `.representative.json` in each person folder.
- Click a person to open details.
- Use the crop toggle to switch between face crops and full images.

4) **Search by image**
- Upload a query photo.
- Uses `find_person_folder_fixed.py` against the active collection.

5) **Sharing (Google Drive)**
- Click "Upload gallery" for a single share link.
- Click "Upload people" for per-person share links.
- First run opens a browser for OAuth consent and creates `token.json`.

## Data Files
- `collections.json`: list of collections the UI knows about.
- `settings.json`: active collection + last used folders.
- `labels.json`: per-person labels (optional).
- `.representative.json`: per-person representative metadata (image + bbox).
- `shares.json`: stored Drive share links.

## Notes
- Keep `credentials.json` and `token.json` out of version control.
- The viewer never renames folders; it only writes labels and representative metadata.
- The Drive upload flow is "upload once" (no sync).

