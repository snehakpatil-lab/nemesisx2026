import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("NEMESIS_DATA_DIR") or ROOT / "data").expanduser()
UPLOADS = DATA / "uploads"
DB = DATA / "nemesis.db"
CORS_ORIGINS = tuple(
    origin.strip().rstrip("/")
    for origin in os.getenv("NEMESIS_CORS_ORIGINS", "").split(",")
    if origin.strip()
)
MODEL = os.getenv("NEMESIS_MODEL", "yolov8n.pt")   # downloaded on first run
CONF = float(os.getenv("NEMESIS_CONF", "0.25"))
STRIDE = int(os.getenv("NEMESIS_STRIDE", "1"))     # minimum frame stride
MAX_ANALYSIS_FRAMES = int(os.getenv("NEMESIS_MAX_ANALYSIS_FRAMES", "700"))
IMGSZ = int(os.getenv("NEMESIS_IMGSZ", "416"))
ALLOWED = {".mp4", ".avi", ".mov"}
if STRIDE < 1 or MAX_ANALYSIS_FRAMES < 1 or IMGSZ < 160:
    raise ValueError("NEMESIS_STRIDE and NEMESIS_MAX_ANALYSIS_FRAMES must be positive; NEMESIS_IMGSZ must be at least 160.")
UPLOADS.mkdir(parents=True, exist_ok=True)
