from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import FEATURE_COLUMNS


def feature_importance(model, x: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    elif y is not None:
        values = permutation_importance(model, x, y, scoring="f1", n_repeats=6, random_state=42).importances_mean
    else:
        values = np.zeros(len(FEATURE_COLUMNS))
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values}).sort_values("importance", ascending=False)


def shap_summary(model, x: pd.DataFrame, sample_size: int = 150):
    sample = x.sample(min(sample_size, len(x)), random_state=42)
    try:
        import shap

        transformed = model.named_steps["preprocess"].transform(sample)
        estimator = model.named_steps["model"]
        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed)
        return values, sample
    except Exception:
        return None, sample

