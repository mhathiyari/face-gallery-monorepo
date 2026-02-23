# Face Search Viewer - Design Doc

## Summary
Face Search Viewer is a local web UI that sits on top of the face-search-system scripts. It lets a host sort images by person, browse people, search by image, and optionally share results via Google Drive.

## Goals
- Provide a simple local UI for non-technical hosts.
- Keep data local by default and allow LAN viewing.
- Support multiple collections with quick switching.
- Allow light metadata edits (labels, representative choice).
- Provide a basic sharing path using free storage (Google Drive).
- Keep the system easy to rerun for occasional updates.

## Non-goals
- No authentication or per-user accounts.
- No merge/split cluster workflows.
- No multi-tenant deployment.
- No incremental background sync to cloud storage (upload once).

## Architecture

### Components
- **Frontend**: Static HTML/CSS/JS served by Flask.
- **Backend**: Flask API and job runner.
- **Face Search Engine**: External repo `face-search-system`.
- **Storage**: Local folders + JSON metadata + SQLite from face-search-system.

### Data flow
1) Host runs a sort job on an input folder.
2) The face-search-system creates `person_###` folders and a `collections/` DB.
3) The viewer reads metadata and renders people grid.
4) Host can rename labels or pick a new representative.
5) Optional: upload a gallery to Google Drive and generate share links.

## Folder Structure
A collection (sorted output) looks like:

```
collection_root/
  person_000/
    .representative.json
    IMG_001.jpg
  person_001/
    .representative.json
    IMG_002.jpg
  unmatched/
  collections/
    image_sorter/
      metadata.db
      index.faiss
  labels.json (optional)
  shares.json (optional)
```

## Data Model

### labels.json
```json
{
  "person_000": "Ayesha",
  "person_001": "Arjun"
}
```

### .representative.json
Stored in each `person_###` folder, produced by the sorter. Example:
```json
{
  "image_name": "IMG_001.jpg",
  "image_path": "/path/to/IMG_001.jpg",
  "bbox": {"x1": 10, "y1": 20, "x2": 200, "y2": 250},
  "confidence": 0.92,
  "selection_reason": "frontal_high_confidence"
}
```

### shares.json
```json
{
  "collection_link": "https://drive.google.com/drive/folders/...",
  "people": {
    "person_000": "https://drive.google.com/drive/folders/..."
  }
}
```

## API Surface (MVP)

- `GET /api/config`
- `POST /api/config`
- `GET /api/settings`
- `GET /api/collections`
- `POST /api/collections`
- `DELETE /api/collections/<id>`
- `GET /api/collections/active`
- `POST /api/collections/active`
- `GET /api/people?sort=name|size`
- `GET /api/person/<id>`
- `POST /api/person/<id>/label`
- `POST /api/person/<id>/representative`
- `POST /api/jobs/sort`
- `GET /api/jobs/<id>`
- `POST /api/search`
- `GET /api/shares`
- `POST /api/drive/upload`

Media endpoints:
- `GET /media/<collection>/<person>/<filename>` (original)
- `GET /face/<collection>/<person>/<filename>` (cluster crop)
- `GET /face-rep/<collection>/<person>` (representative crop from .representative.json)

## UI Structure

### Collections panel
- Add collection by path.
- Click row to activate.
- Remove only the active collection.

### Sort job panel
- Owner mode gated.
- Input/output folders + progress logs.

### People grid
- Representative crop using `.representative.json`.
- Sort by name/size.
- Copy share link per person (if available).

### Person detail
- Full grid of images.
- Toggle between face crops and full images.
- Set representative (owner mode only).

### Search
- Upload image to find the best match folder.

### Sharing
- Upload gallery (single public link).
- Upload people (per-person links).

## Permissions model
- **Owner mode**: required for write actions (sort job, labels, representative).
- **Guest mode**: browse and search only.

## Sharing model (Google Drive)
- OAuth using user account (Drive API).
- Upload once, no sync.
- Two modes:
  - Gallery: share only the root folder.
  - People: share each person folder.

## Performance notes
- For thousands of photos, paging is used in the person detail view.
- Cropping is done on-demand and served as JPEG.
- Caching is avoided for representative crops to reflect updates quickly.

## Error handling
- Input validation for paths.
- Guardrails: all paths must stay within `root_dir`.
- User-visible errors via UI.

## Testing / Verification
- Run a small collection end-to-end.
- Verify search-by-image returns a correct folder.
- Verify representative crop matches `.representative.json`.
- Verify Drive upload produces links in `shares.json`.

## Future extensions
- Thumbnail caching on disk.
- Auth for LAN guest mode.
- Sync or delta uploads to Drive.
- Pluggable storage backends (S3/R2/B2).
- Export static gallery HTML.

## Current State / TODOs
- MVP UI and backend are implemented and running locally.
- Representative crops use `.representative.json` with 30% padding.
- Drive upload is one-shot (no sync) and stores links in `shares.json`.
- Remaining work: finalize tests, add troubleshooting notes, and polish error handling for Drive upload retries.
