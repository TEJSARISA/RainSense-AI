from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, PROCESSED_DATA_PATH


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned = cleaned[cleaned[column].between(lower, upper) | cleaned[column].isna()]
    return cleaned.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["date"] = pd.to_datetime(featured["date"])
    featured["month"] = featured["date"].dt.month
    featured["season"] = featured["month"].map(season_from_month)
    season_map = {"winter": 0, "summer": 1, "monsoon": 2, "post_monsoon": 3}
    featured["season_encoded"] = featured["season"].map(season_map)
    featured["temp_humidity_index"] = featured["temperature"] * featured["humidity"] / 100
    featured["pressure_drop_signal"] = 1013.25 - featured["pressure"]
    return featured


def build_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def prepare_dataset(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    numeric = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "sunshine_hours", "rainfall_mm"]
    clean = remove_outliers_iqr(df, numeric)
    featured = engineer_features(clean)
    for column in FEATURE_COLUMNS:
        if featured[column].isna().any():
            featured[column] = featured[column].fillna(featured[column].median())
    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        featured.to_csv(PROCESSED_DATA_PATH, index=False)
    return featured

