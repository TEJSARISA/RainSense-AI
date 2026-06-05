from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def rainfall_trends(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.set_index("date")["rainfall_mm"].resample("ME").sum().reset_index()


def seasonal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("season", as_index=False).agg(
        avg_rainfall_mm=("rainfall_mm", "mean"),
        rainy_days=("rain", "sum"),
        avg_humidity=("humidity", "mean"),
        avg_cloud_cover=("cloud_cover", "mean"),
    )


def monthly_forecast(df: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    trends = rainfall_trends(df)
    series = trends["rainfall_mm"].rolling(3, min_periods=1).mean()
    last_date = trends["date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    baseline = float(series.tail(6).mean())
    seasonal = [baseline * (1 + 0.25 * np.sin(i / 6 * 2 * np.pi)) for i in range(months)]
    return pd.DataFrame({"date": future_dates, "forecast_rainfall_mm": np.round(seasonal, 2)})


def detect_weather_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    features = df[["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "sunshine_hours"]].fillna(df.median(numeric_only=True))
    detector = IsolationForest(contamination=0.035, random_state=42)
    flagged = df.copy()
    flagged["anomaly"] = detector.fit_predict(features)
    return flagged[flagged["anomaly"] == -1]
