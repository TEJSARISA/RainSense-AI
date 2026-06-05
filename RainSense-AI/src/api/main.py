from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import METRICS_PATH, MODEL_INFO_PATH
from src.mlops.tracking import log_prediction, read_json
from src.prediction.predictor import predict_rainfall


class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=-30, le=60)
    humidity: float = Field(..., ge=0, le=100)
    pressure: float = Field(..., ge=850, le=1100)
    wind_speed: float = Field(..., ge=0, le=120)
    cloud_cover: float = Field(..., ge=0, le=100)
    sunshine_hours: float = Field(..., ge=0, le=24)
    month: int = Field(7, ge=1, le=12)


app = FastAPI(
    title="RainSense AI API",
    version="1.0.0",
    description="Rainfall prediction and weather intelligence API.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "RainSense AI"}


@app.post("/predict")
def predict(payload: WeatherInput) -> dict:
    request_payload = payload.model_dump()
    result = predict_rainfall(request_payload)
    log_prediction(request_payload, result)
    return result


@app.get("/model-info")
def model_info() -> dict:
    return read_json(MODEL_INFO_PATH, {"message": "Model has not been trained yet."})


@app.get("/metrics")
def metrics() -> dict:
    return read_json(METRICS_PATH, {"message": "Metrics are unavailable until training has completed."})

