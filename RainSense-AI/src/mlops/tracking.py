from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import EXPERIMENTS_PATH, LOG_DIR


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text())


def log_experiment(model_name: str, metrics: dict, artifact_path: str) -> None:
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "artifact_path": artifact_path,
        **{k: round(float(v), 5) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    history = pd.read_csv(EXPERIMENTS_PATH) if EXPERIMENTS_PATH.exists() else pd.DataFrame()
    pd.concat([history, pd.DataFrame([row])], ignore_index=True).to_csv(EXPERIMENTS_PATH, index=False)


def log_prediction(payload: dict, result: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
            "result": result,
        },
        default=str,
    )
    with (LOG_DIR / "predictions.jsonl").open("a") as handle:
        handle.write(line + "\n")

