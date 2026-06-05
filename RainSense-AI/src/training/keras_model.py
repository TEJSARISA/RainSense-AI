from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible TensorFlow/Keras binary classifier."""

    def __init__(self, epochs: int = 25, batch_size: int = 32, verbose: int = 0, random_state: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def _build_model(self, input_dim: int):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(48, activation="relu"),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.Dense(24, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, x, y):
        self.model_ = self._build_model(x.shape[1])
        self.model_.fit(np.asarray(x), np.asarray(y), epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict_proba(self, x):
        probabilities = self.model_.predict(np.asarray(x), verbose=0).ravel()
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False

