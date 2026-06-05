from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "cloud_cover",
    "sunshine_hours",
    "rainfall_mm",
    "rain",
}

RANGES = {
    "temperature": (-30, 60),
    "humidity": (0, 100),
    "pressure": (850, 1100),
    "wind_speed": (0, 120),
    "cloud_cover": (0, 100),
    "sunshine_hours": (0, 24),
    "rainfall_mm": (0, 500),
    "rain": (0, 1),
}


def validate_weather_data(df: pd.DataFrame) -> dict:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    report = {
        "rows": int(len(df)),
        "missing_columns": sorted(missing),
        "missing_values": df.isna().sum().astype(int).to_dict(),
        "range_violations": {},
        "is_valid": len(missing) == 0,
    }

    for column, (lower, upper) in RANGES.items():
        if column in df:
            violations = df[column].dropna().between(lower, upper).eq(False).sum()
            report["range_violations"][column] = int(violations)
            if violations:
                report["is_valid"] = False
    return report

