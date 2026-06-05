from __future__ import annotations

import joblib
import pandas as pd

from src.config import BASE_INPUT_COLUMNS, FEATURE_COLUMNS, LATEST_MODEL_PATH
from src.preprocessing.preprocessor import engineer_features


def _default_month() -> int:
    return pd.Timestamp.today().month


def build_prediction_frame(payload: dict) -> pd.DataFrame:
    row = {column: float(payload[column]) for column in BASE_INPUT_COLUMNS}
    row["date"] = pd.Timestamp(year=pd.Timestamp.today().year, month=int(payload.get("month", _default_month())), day=15)
    row["rainfall_mm"] = 0.0
    features = engineer_features(pd.DataFrame([row]))
    return features[FEATURE_COLUMNS]


def load_model():
    if not LATEST_MODEL_PATH.exists():
        from src.training.train_models import train_all_models

        train_all_models()
    return joblib.load(LATEST_MODEL_PATH)


def predict_rainfall(payload: dict) -> dict:
    model = load_model()
    frame = build_prediction_frame(payload)
    probability = float(model.predict_proba(frame)[0, 1])
    label = int(probability >= 0.5)
    estimated_rainfall = max(0.0, (probability - 0.32) * 42 + (float(payload["humidity"]) - 55) * 0.08)
    confidence = max(probability, 1 - probability)
    return {
        "rain_probability": round(probability * 100, 2),
        "rainfall_prediction": "Rain expected" if label else "No significant rain expected",
        "estimated_rainfall_mm": round(estimated_rainfall if label else estimated_rainfall * 0.25, 2),
        "confidence_score": round(confidence * 100, 2),
    }

