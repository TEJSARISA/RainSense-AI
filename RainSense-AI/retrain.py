from __future__ import annotations

import logging

from src.training.train_models import train_all_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def automated_retraining_pipeline() -> dict:
    logging.info("Starting RainSense automated retraining pipeline")
    result = train_all_models()
    logging.info("Retraining finished. Champion model: %s", result["best_model"])
    return result


if __name__ == "__main__":
    automated_retraining_pipeline()

