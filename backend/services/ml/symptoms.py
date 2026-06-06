"""
Symptom-based disease prediction service.
Matches the feature mapping logic from All Disease Checkup/test.py.
"""
import os
import pickle

import numpy as np

from config import settings

_artifacts: dict | None = None


def _load():
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    _artifacts = {}
    for name, path in [
        ("symptom_model",  settings.SYMPTOM_MODEL_PATH),
        ("label_encoder",  settings.LABEL_ENCODER_PATH),
        ("feature_columns", settings.FEATURE_COLS_PATH),
    ]:
        if not os.path.exists(path):
            print(f"[SYMPTOMS] Skipping {name} — not found: {path}")
            continue
        with open(path, "rb") as f:
            _artifacts[name] = pickle.load(f)
        print(f"[SYMPTOMS] Loaded {name}")

    return _artifacts


def _map_answers(answers: dict, feature_columns: list) -> dict:
    features = {col: 0 for col in feature_columns}

    if "age" in answers:
        features["age"] = int(answers["age"])
    if "sex" in answers:
        features["sex"] = 1 if str(answers["sex"]).upper() == "M" else 0

    features["smoking_history"] = int(answers.get("smoking_history", 0))
    features["chest_pain"]      = int(answers.get("chest_pain", 0))

    radiates = int(answers.get("chest_pain_radiates", 0))
    features["chest_pain_radiates"] = radiates
    if radiates:
        features["sweating_with_pain"] = 1

    pal = answers.get("palpitation_type", "none")
    if pal != "none":      features["palpitation"]        = 1
    if pal == "irregular": features["irregular_heartbeat"] = 1
    if pal == "slow":      features["slow_heartbeat"]     = 1

    features["syncope"] = int(answers.get("syncope", 0))

    breath = answers.get("breath_pattern", "none")
    if breath != "none":       features["shortness_of_breath"] = 1
    if breath == "lying_down": features["worse_lying_down"]    = 1

    features["leg_swelling"] = int(answers.get("leg_swelling", 0))

    cough = answers.get("cough_type", "none")
    if cough != "none":  features["cough"]            = 1
    if cough == "blood": features["cough_with_blood"] = 1

    features["fever"]                = int(answers.get("fever", 0))
    features["skin_lesion"]          = int(answers.get("skin_lesion", 0))
    features["lesion_changing"]      = int(answers.get("lesion_changing", 0))
    features["sun_exposure_history"] = int(answers.get("sun_exposure_history", 0))

    lf = answers.get("lesion_features", "none")
    if lf == "irregular_border":   features["lesion_irregular_border"] = 1
    elif lf == "multiple_colors":  features["lesion_color_multiple"]   = 1
    elif lf == "bleeding_itching": features["lesion_bleeding_itching"] = 1

    sys_ans = answers.get("systemic", "none")
    if sys_ans in ("fatigue", "both"):     features["fatigue"]            = 1
    if sys_ans in ("weight_loss", "both"): features["sudden_weight_loss"] = 1

    if int(answers.get("other_systemic", 0)):
        features["nausea"] = features["dizziness"] = 1

    risk = answers.get("risk_factors", "none")
    if risk in ("hypertension", "both"): features["hypertension_history"] = 1
    if risk in ("diabetes", "both"):     features["diabetes_history"]     = 1

    return features


def predict(answers: dict) -> dict:
    arts = _load()
    if "symptom_model" not in arts:
        raise RuntimeError("Symptom model not loaded")

    f_cols = arts["feature_columns"]
    fvec   = _map_answers(answers, f_cols)
    X      = np.array([[fvec[col] for col in f_cols]], dtype=np.float32)
    proba  = arts["symptom_model"].predict_proba(X)[0]
    idx    = int(np.argmax(proba))
    enc    = arts["label_encoder"]

    top3 = [
        {"disease": enc.inverse_transform([i])[0], "probability": round(float(proba[i]) * 100, 1)}
        for i in np.argsort(proba)[::-1][:3]
    ]

    return {
        "prediction":     enc.inverse_transform([idx])[0],
        "confidence":     round(float(proba[idx]), 4),
        "inconclusive":   float(proba[idx]) < 0.50,
        "top3":           top3,
        "filled_columns": sum(1 for v in fvec.values() if v != 0),
        "total_columns":  len(f_cols),
    }
