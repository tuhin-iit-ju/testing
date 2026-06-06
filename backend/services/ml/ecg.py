"""
ECG inference service.
Matches the exact logic from ECG/Test/test_ecg.py.
"""
import io
import numpy as np
import pandas as pd

from config import settings

WINDOW_SIZE = 3600
SAMPLING_RATE = 360
CLASS_NAMES = [
    "Normal", "Supraventricular", "Ventricular", "Conduction Disorder",
    "Myocardial Infarction", "Hypertrophy", "Ischemia/ST-T", "Atrial Fibrillation",
]
WEIGHTS = {"resnet": 0.45, "inception": 0.35, "transformer": 0.20}

_models: dict | None = None


def _get_models():
    global _models
    if _models is not None:
        return _models

    import os
    import tensorflow as tf

    # Register custom layer used in transformer model
    @tf.keras.utils.register_keras_serializable()
    class PositionalEncoding(tf.keras.layers.Layer):
        def call(self, x):
            seq_len = tf.shape(x)[1]
            d_model = tf.shape(x)[2]
            pos = tf.range(seq_len, dtype=tf.float32)[:, None]
            dim = tf.range(d_model, dtype=tf.float32)[None, :]
            angle = pos / tf.pow(10000.0, (2 * (dim // 2)) / tf.cast(d_model, tf.float32))
            even = tf.cast(tf.math.floormod(dim, 2) == 0, tf.float32)
            odd = 1 - even
            encoding = tf.sin(angle) * even + tf.cos(angle) * odd
            return x + encoding[None, :, :]

    custom_objects = {"PositionalEncoding": PositionalEncoding}
    files = {
        "resnet": settings.ECG_RESNET_PATH,
        "inception": settings.ECG_INCEPTION_PATH,
        "transformer": settings.ECG_TRANSFORMER_PATH,
    }

    _models = {}
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"[ECG] Skipping {name} — not found: {path}")
            continue
        _models[name] = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print(f"[ECG] Loaded {name}")

    return _models


def _preprocess(signal: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, filtfilt, iirnotch
    nyq = 0.5 * SAMPLING_RATE
    low = 0.5 / nyq
    high = min(45.0 / nyq, 0.99)
    b, a = butter(3, [low, high], btype="bandpass")
    signal = filtfilt(b, a, signal)
    w0 = 50.0 / nyq
    if w0 < 1.0:
        bn, an = iirnotch(w0, 30)
        signal = filtfilt(bn, an, signal)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal.astype(np.float32)


def predict(csv_bytes: bytes) -> dict:
    try:
        loaded = _get_models()
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ECG analysis requires TensorFlow. "
            "Free up disk space then run: pip install tensorflow --no-cache-dir"
        ) from e
    if not loaded:
        raise RuntimeError("No ECG models loaded")

    df = pd.read_csv(io.BytesIO(csv_bytes), comment="#", header=None)
    col = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
    signal = col.values.astype(np.float32)

    n_segments = len(signal) // WINDOW_SIZE
    if n_segments == 0:
        raise ValueError(f"Signal too short ({len(signal)} samples). Need at least {WINDOW_SIZE}.")

    # Build one (n_segments, WINDOW_SIZE, 1) batch and call each model exactly once
    windows = np.stack([
        _preprocess(signal[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE])
        for i in range(n_segments)
    ]).reshape(n_segments, WINDOW_SIZE, 1)

    # Eager call beats .predict() for small batches; weighted-average across models
    per_model_probs = {k: loaded[k](windows, training=False).numpy() for k in loaded}
    ensemble_per_segment = sum(WEIGHTS[k] * per_model_probs[k] for k in per_model_probs)

    segment_top_idx = ensemble_per_segment.argmax(axis=1)
    segment_conf = ensemble_per_segment.max(axis=1)
    segments = [
        {
            "segment": i + 1,
            "prediction": CLASS_NAMES[int(segment_top_idx[i])],
            "confidence": float(segment_conf[i]),
        }
        for i in range(n_segments)
    ]

    avg = ensemble_per_segment.mean(axis=0)
    top_idx = int(np.argmax(avg))
    segment_agreement = float((segment_top_idx == top_idx).mean())
    confidence_variance = float(segment_conf.var())

    return {
        "prediction": CLASS_NAMES[top_idx],
        "confidence": round(float(avg[top_idx]), 4),
        "class_probabilities": {CLASS_NAMES[i]: round(float(avg[i]), 4) for i in range(len(CLASS_NAMES))},
        "segments": segments[:10],
        "total_segments": n_segments,
        "segment_agreement": round(segment_agreement, 4),
        "confidence_variance": round(confidence_variance, 6),
    }
