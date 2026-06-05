from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH


def generate_sample_weather_data(path: Path = RAW_DATA_PATH, rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic weather dataset for local demos and tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    month = dates.month.to_numpy()
    seasonal_wave = np.sin((month - 1) / 12 * 2 * np.pi)
    monsoon_wave = np.exp(-((month - 7.5) ** 2) / 5.5)

    temperature = 24 + 8 * seasonal_wave + rng.normal(0, 3.2, rows)
    humidity = np.clip(54 + 34 * monsoon_wave - 10 * seasonal_wave + rng.normal(0, 9, rows), 20, 100)
    pressure = np.clip(1013 - 8 * monsoon_wave + rng.normal(0, 5, rows), 985, 1035)
    wind_speed = np.clip(8 + 5 * monsoon_wave + rng.gamma(2.0, 1.2, rows), 0, 40)
    cloud_cover = np.clip(28 + 55 * monsoon_wave + 0.42 * humidity + rng.normal(0, 13, rows) - 25, 0, 100)
    sunshine_hours = np.clip(10.5 - cloud_cover / 12 + rng.normal(0, 1.2, rows), 0, 13)

    rain_score = (
        0.055 * humidity
        + 0.045 * cloud_cover
        + 0.09 * wind_speed
        - 0.05 * sunshine_hours
        - 0.025 * (pressure - 1000)
        + 1.4 * monsoon_wave
        + rng.normal(0, 1.2, rows)
        - 6.0
    )
    rain_probability = 1 / (1 + np.exp(-rain_score))
    rain = rng.binomial(1, np.clip(rain_probability, 0.02, 0.98))
    rainfall_mm = np.where(rain == 1, rng.gamma(2.1, 7.0, rows) * (0.35 + monsoon_wave), 0)

    df = pd.DataFrame(
        {
            "date": dates,
            "temperature": temperature.round(2),
            "humidity": humidity.round(2),
            "pressure": pressure.round(2),
            "wind_speed": wind_speed.round(2),
            "cloud_cover": cloud_cover.round(2),
            "sunshine_hours": sunshine_hours.round(2),
            "rainfall_mm": rainfall_mm.round(2),
            "rain": rain,
        }
    )

    missing_idx = rng.choice(rows, size=max(5, rows // 80), replace=False)
    df.loc[missing_idx, "humidity"] = np.nan
    outlier_idx = rng.choice(rows, size=max(3, rows // 250), replace=False)
    df.loc[outlier_idx, "wind_speed"] *= 3.5

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_weather_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return generate_sample_weather_data(path)
    return pd.read_csv(path, parse_dates=["date"])

