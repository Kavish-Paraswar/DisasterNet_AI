"""
Classification Service — Repo A Pipeline
Loads the Keras disaster.h5 model ONCE at import time.
Provides safe_classify() which never raises — returns partial result on failure.
"""
import os
import numpy as np

# NOTE: KERAS_BACKEND="torch" is set in app.py before this module is imported.

from keras.models import load_model
from config.settings import CLASSIFICATION_MODEL_PATH, CLASSIFICATION_LABELS
from utils.image_processing import preprocess_for_classification

# ── Load model ONCE at module init ──────────────────────────────────────────────
print("[ClassificationService] Loading disaster.h5 ...")
_model = load_model(CLASSIFICATION_MODEL_PATH)
print("[ClassificationService] Model loaded successfully.")


def safe_classify(image_bgr: np.ndarray) -> dict:
    """
    Run the classification pipeline.
    Always returns a dict with 'prediction' and 'confidence'.
    Never raises — catches errors internally.
    """
    try:
        x = preprocess_for_classification(image_bgr)
        predictions = _model.predict(x)
        result_idx = int(np.argmax(predictions, axis=-1)[0])
        confidence = float(predictions[0][result_idx])

        return {
            "prediction": CLASSIFICATION_LABELS[result_idx],
            "confidence": round(confidence * 100, 2),
        }

    except Exception as e:
        print(f"[ClassificationService] ERROR: {e}")
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
        }
