from __future__ import annotations

import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "INvideos.csv"
CATEGORY_METADATA_PATH = PROJECT_ROOT / "category_metadata.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "youtube_models.joblib"
DATABASE_PATH = ARTIFACT_DIR / "youtube_analytics.sqlite"
LOG_PATH = ARTIFACT_DIR / "youtube_analytics.log"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("youtube_analytics")
