from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "weather_history.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "weather_features.csv"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
FEATURES_PATH = MODEL_DIR / "features.json"
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.joblib"
EXPERIMENTS_PATH = MODEL_DIR / "experiments.csv"

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "cloud_cover",
    "sunshine_hours",
    "month",
    "season_encoded",
    "temp_humidity_index",
    "pressure_drop_signal",
]

BASE_INPUT_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "cloud_cover",
    "sunshine_hours",
]

TARGET_COLUMN = "rain"

