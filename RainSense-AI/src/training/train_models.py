from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import FEATURE_COLUMNS, FEATURES_PATH, LATEST_MODEL_PATH, METRICS_PATH, MODEL_DIR, MODEL_INFO_PATH, TARGET_COLUMN
from src.mlops.tracking import log_experiment, write_json
from src.preprocessing.data_loader import load_weather_data
from src.preprocessing.preprocessor import build_numeric_pipeline, prepare_dataset
from src.training.keras_model import KerasBinaryClassifier, tensorflow_available


def optional_models() -> dict:
    models = {}
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=130,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        models["XGBoost"] = GradientBoostingClassifier(random_state=43)

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(n_estimators=160, learning_rate=0.05, random_state=42, verbose=-1)
    except Exception:
        models["LightGBM"] = GradientBoostingClassifier(random_state=44)

    return models


def model_zoo() -> dict:
    neural_network = (
        KerasBinaryClassifier(epochs=25, batch_size=32, verbose=0, random_state=42)
        if tensorflow_available()
        else MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=450, random_state=42, early_stopping=True)
    )
    return {
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=9, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "TensorFlow/Keras Neural Network": neural_network,
        **optional_models(),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1")
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train_all_models() -> dict:
    raw = load_weather_data()
    dataset = prepare_dataset(raw)
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, stratify=y, random_state=42)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, estimator in model_zoo().items():
        pipeline = Pipeline([("preprocess", build_numeric_pipeline()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test, x_train, y_train)
        safe_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        artifact = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(pipeline, artifact)
        log_experiment(name, metrics, str(artifact))
        results[name] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        if metrics["f1"] > best_score:
            best_name = name
            best_score = metrics["f1"]
            best_pipeline = pipeline

    joblib.dump(best_pipeline, LATEST_MODEL_PATH)
    write_json(METRICS_PATH, results)
    write_json(FEATURES_PATH, {"features": FEATURE_COLUMNS})
    write_json(
        MODEL_INFO_PATH,
        {
            "model_name": best_name,
            "version": datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M%S"),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target": TARGET_COLUMN,
            "rows": int(len(dataset)),
            "features": FEATURE_COLUMNS,
        },
    )
    return {"best_model": best_name, "metrics": results}


if __name__ == "__main__":
    print(json.dumps(train_all_models(), indent=2))
