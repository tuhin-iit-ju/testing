"""
Maps raw confidence/prediction data to a 0-100 risk score and a risk level string.
"""

_HIGH_RISK_CONDITIONS = {
    # X-Ray
    "Pneumonia", "Cardiomegaly", "Edema", "Consolidation", "Atelectasis",
    # ECG
    "Myocardial Infarction", "Atrial Fibrillation", "Ventricular",
    "Ischemia/ST-T", "Conduction Disorder",
    # Skin
    "mel", "bcc", "akiec",
    # Symptom / general
    "Myocardial Infarction", "Heart Failure", "Pneumonia",
    "Atrial Fibrillation", "Lung Cancer", "Melanoma",
}

_MODERATE_RISK_CONDITIONS = {
    "Hypertrophy", "Supraventricular", "bkl", "df",
    "COPD", "Asthma", "Hypertension",
}


def compute_risk_score(test_type: str, prediction: str, confidence: float, anomaly_score: float = 0.0) -> tuple[int, str]:
    """
    Returns (risk_score: int 0-100, risk_level: str).

    For CT scan, anomaly_score (0-1, higher = more anomalous) overrides prediction logic.
    """
    if test_type == "ct":
        raw = anomaly_score * 100
        score = int(min(100, max(0, raw)))
    elif prediction in _HIGH_RISK_CONDITIONS or prediction.lower() in {c.lower() for c in _HIGH_RISK_CONDITIONS}:
        score = int(60 + confidence * 40)
    elif prediction in _MODERATE_RISK_CONDITIONS or prediction.lower() in {c.lower() for c in _MODERATE_RISK_CONDITIONS}:
        score = int(30 + confidence * 30)
    elif prediction in ("No Finding", "Normal", "nv", "vasc"):
        score = int(confidence * 25)
    else:
        score = int(40 + confidence * 20)

    score = min(100, max(0, score))

    if score <= 30:
        level = "low"
    elif score <= 60:
        level = "moderate"
    else:
        level = "high"

    return score, level
