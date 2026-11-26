# NEMESIS-X backend (Stage 2)

FastAPI + YOLO/ByteTrack + SQLite. Runs fully local; **no API key needed**.

## Setup
1. Python 3.10+.
2. Install PyTorch first (CUDA build for RTX 3050/4050): https://pytorch.org/get-started/locally/
   (CPU-only also works, just slower.)
3. Then:
   ```
   python -m venv .venv
   # Windows: .venv\Scripts\activate      Ubuntu: source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --port 8000
   ```
4. Open http://localhost:8000 (first analysis downloads the YOLO weights, so it needs internet once).

## Deployment

- **Render backend:** create a Blueprint from this repository and use `render.yaml`. Set
  `NEMESIS_CORS_ORIGINS` to the exact Vercel site origin (for example
  `https://your-site.vercel.app`, no path). Render's free filesystem is ephemeral; uploaded
  videos and SQLite data can be lost on redeploy/restart. For persistent data, attach a paid
  Render disk and set `NEMESIS_DATA_DIR` to its mount path.
- **Vercel frontend:** set the project root to this directory and the environment variable
  `NEMESIS_API_BASE_URL` to the Render service origin (for example
  `https://nemesis-x.onrender.com`). Vercel builds the static site with `npm run build`.
  The frontend uses this URL for API calls, video playback, and evidence links.
- The backend runs CPU inference on Render. Choose an instance with enough memory for PyTorch
  and video processing; the free tier may not have sufficient resources for analysis.
- The API currently has no authentication. CORS only restricts browser-origin access; it does not
  protect uploads or analysis endpoints from direct clients. Do not treat this configuration as
  production-safe for an untrusted public audience without adding authentication and abuse limits.

## Config (env vars)
NEMESIS_MODEL (default yolov8n.pt) · NEMESIS_CONF (0.25) · NEMESIS_STRIDE (minimum 1) · NEMESIS_MAX_ANALYSIS_FRAMES (700) · NEMESIS_IMGSZ (416) · NEMESIS_DATA_DIR (persistent data directory) · NEMESIS_CORS_ORIGINS (comma-separated browser origins)

## API
GET /api/health · /api/system · /api/system/configuration · POST /api/videos · GET /api/videos · GET /api/videos/{id}
POST /api/videos/{id}/analyze · GET .../tracks · .../tracks/{tid} · .../detections?t=SEC
GET .../events · .../memory?offset=0&limit=50 · .../spatial · .../frame?t=SEC · .../file
GET .../anomalies · POST .../anomalies/{candidate_id}/review · GET .../evidence?offset=0&limit=30
GET /api/videos/{id}/report.pdf · POST /api/chat {question, video_id?}

The PDF report is available for a selected video after analysis completes. It includes an overall
incident overview and segmented activity timeline based on stored detections and tracks, reviewed
motion flags, video metadata, and evidence limitations. It does not automatically classify or
confirm incidents; review the original footage before drawing conclusions.

The anomalies endpoint also reports per-class eligibility and pending/confirmed/dismissed review
counts. Empty candidate lists mean the configured sample thresholds were not met or no outlier was
detected; they do not indicate that the request or analysis failed.

## Notes
- Ultralytics is AGPL-3.0; check its licence for your use.
- `/api/system` reports CPU/RAM use, CUDA/GPU telemetry, and processing state/FPS; VRAM is not applicable during CPU-only inference.
- Video inference samples at most 700 frames by default and uses 416px model input; the effective stride is saved per video. This speeds up long clips at the cost of temporal detail between samples.
- Coordinates are image-plane pixels, not metric 3D.
- Track event records mark first/last observations only; they are not incident classifications.
- Statistical anomaly candidates are exploratory within-class image-plane speed deviations (robust z-score >= 2.0) with at least 8 observations per track and 5 tracks per class. Small cohorts can produce unstable flags; camera motion or tracker switches can cause false positives.
- Model-generated candidates come from a deterministic Isolation Forest fitted per video and object class using track speed, path efficiency, detector confidence, and observation count. Scores are not probabilities; each class needs at least 5 tracks with 8 observations each. Treat small-cohort results as review cues only.
- Candidate review is persisted per video; neither statistical nor model-generated candidates are automatically treated as verified events.
- Evidence Explorer items are paginated summaries of stored detection boxes with first/last timestamps, source-frame links, reasoning basis, and explicit data limitations.
- Evidence Q&A summarizes tracked classes and timeline activity, and links vehicle motion outliers to reviewable timestamps. These image-plane cues are not collision detections and cannot establish impact, injury, or cause.
- Semantic event/incident classification and cross-video re-identification remain unimplemented.
