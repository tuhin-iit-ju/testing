"""
CT Scan anomaly detection service.
Uses the sklearn pipeline (scaler → PCA → IsolationForest) for robust,
architecture-independent inference.
"""
import io
import os
import joblib
import numpy as np
from PIL import Image

from config import settings

IMAGE_SIZE = 16   # scaler trained on 16×16 = 256 features

_artifacts: dict | None = None


def _load_artifacts():
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    _artifacts = {}
    for key, path in [
        ("scaler", settings.CT_SCALER_PATH),
        ("pca", settings.CT_PCA_PATH),
        ("iso_forest", settings.CT_IF_PATH),
    ]:
        if not os.path.exists(path):
            print(f"[CT] Skipping {key} — not found: {path}")
            continue
        _artifacts[key] = joblib.load(path)
        print(f"[CT] Loaded {key}")

    return _artifacts


def predict(image_bytes: bytes) -> dict:
    arts = _load_artifacts()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")   # grayscale
    except Exception:
        raise ValueError("Could not open image. Please upload a valid JPEG or PNG file.")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32).flatten() / 255.0   # shape (784,)
    X = arr.reshape(1, -1)

    anomaly_score = 0.5   # fallback
    prediction = "Unable to analyze"
    details: dict = {}

    if "scaler" in arts and "pca" in arts and "iso_forest" in arts:
        X_scaled = arts["scaler"].transform(X)
        X_pca = arts["pca"].transform(X_scaled)
        decision = arts["iso_forest"].decision_function(X_pca)[0]
        label = arts["iso_forest"].predict(X_pca)[0]   # 1 = normal, -1 = anomaly

        # Convert decision function to 0-1 (higher = more anomalous)
        anomaly_score = float(1 / (1 + np.exp(decision * 5)))

        if label == -1:
            prediction = "Anomaly Detected"
            confidence = anomaly_score
        else:
            prediction = "Normal"
            confidence = 1.0 - anomaly_score

        details = {
            "isolation_forest_label": int(label),
            "anomaly_score": round(anomaly_score, 4),
            "decision_function_raw": round(float(decision), 4),
        }
    else:
        confidence = 0.5
        details = {"error": "Some CT models not loaded — using fallback"}

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "anomaly_score": round(anomaly_score, 4),
        "details": details,
    }
