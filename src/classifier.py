"""Serving wrapper around the trained classifier artifact."""

from dataclasses import dataclass

import joblib

from src import config


@dataclass
class Prediction:
    label: str
    confidence: float
    model_version: str
    needs_human: bool


class TfidfClassifier:
    """Loads the artifact written by scripts/train_baseline.py.

    The artifact holds the fitted pipeline, the confidence threshold chosen on
    train data, and a version string that goes into every audit record.
    """

    def __init__(self, path=config.CLASSIFIER_PATH):
        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.threshold = artifact["threshold"]
        self.model_version = artifact["model_version"]

    def predict(self, text):
        proba = self.pipeline.predict_proba([text])[0]
        best = proba.argmax()
        confidence = float(proba[best])
        return Prediction(
            label=str(self.pipeline.classes_[best]),
            confidence=confidence,
            model_version=self.model_version,
            needs_human=confidence < self.threshold,
        )
