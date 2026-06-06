"""
seed_demo.py — create demo patient + doctor accounts and ~12 synthetic reports.

Run once from healthbridge/backend/:
    python seed_demo.py

The script is idempotent for users (skips if they already exist) and replaces
any existing reports for the demo patient so re-runs produce a clean dataset.
"""
import asyncio
import random
from datetime import datetime, timedelta

from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext

from config import settings

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ─────────────────────────────────────────────────────────────────────────────
# Credentials
# ─────────────────────────────────────────────────────────────────────────────

PATIENT = {
    "email":             "patient@uyecare.com",
    "password":          "Patient@2025",
    "name":              "Ayesha Rahman",
    "role":              "patient",
    "date_of_birth":     "1992-04-15",
    "sex":               "Female",
    "blood_group":       "B+",
    "phone":             "+8801711000001",
    "emergency_contact": "Karim Rahman: +8801711000002",
    "address":           "Dhanmondi, Dhaka, Dhaka",
}

DOCTOR = {
    "email":            "doctor@uyecare.com",
    "password":         "Doctor@2025",
    "name":             "Sarah Khan",
    "role":             "doctor",
    "phone":            "+8801711000010",
    "specialty":        "Cardiologist",
    "license_no":       "BMDC-78421",
    "hospital":         "Square Hospital, Dhaka",
    "experience_years": 12,
    "bio":              "Cardiologist with 12 years of clinical experience, focused on "
                        "preventive cardiology and arrhythmia management.",
    "status":           "active",      # auto-active so demo doctor can log in immediately
}

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic reports — varied test types, predictions and risk levels
# so the dashboard charts are visually interesting.
# ─────────────────────────────────────────────────────────────────────────────
# (test_type, prediction, confidence, risk_score, risk_level, details)

REPORTS = [
    ("xray", "Pneumonia",            0.87, 75, "high", {
        "detected_conditions": [{"name": "Pneumonia", "probability": 0.87}],
        "all_probabilities": {"No Finding": 0.04, "Pneumonia": 0.87, "Cardiomegaly": 0.18,
                              "Edema": 0.12, "Consolidation": 0.41, "Atelectasis": 0.08},
        "per_model": {
            "DenseNet121": {"No Finding": 0.05, "Pneumonia": 0.89, "Cardiomegaly": 0.17, "Edema": 0.10, "Consolidation": 0.39, "Atelectasis": 0.07},
            "ResNet50":    {"No Finding": 0.04, "Pneumonia": 0.86, "Cardiomegaly": 0.21, "Edema": 0.13, "Consolidation": 0.44, "Atelectasis": 0.09},
            "ViT-Base":    {"No Finding": 0.03, "Pneumonia": 0.86, "Cardiomegaly": 0.16, "Edema": 0.13, "Consolidation": 0.40, "Atelectasis": 0.08},
        },
        "model_agreement": 0.92,
    }),
    ("xray", "No Finding",            0.92, 12, "low",  {
        "detected_conditions": [],
        "all_probabilities": {"No Finding": 0.92, "Pneumonia": 0.04, "Cardiomegaly": 0.03,
                              "Edema": 0.05, "Consolidation": 0.02, "Atelectasis": 0.06},
        "model_agreement": 0.95,
    }),
    ("xray", "Cardiomegaly",          0.71, 68, "high", {
        "detected_conditions": [{"name": "Cardiomegaly", "probability": 0.71}],
        "all_probabilities": {"No Finding": 0.08, "Pneumonia": 0.12, "Cardiomegaly": 0.71,
                              "Edema": 0.34, "Consolidation": 0.04, "Atelectasis": 0.09},
        "model_agreement": 0.88,
    }),
    ("ecg", "Atrial Fibrillation",    0.84, 82, "high", {
        "class_probabilities": {"Normal": 0.05, "Atrial Fibrillation": 0.84,
                                "Supraventricular": 0.04, "Ventricular": 0.02,
                                "Conduction Disorder": 0.01, "Myocardial Infarction": 0.02,
                                "Hypertrophy": 0.01, "Ischemia/ST-T": 0.01},
        "total_segments": 6,
        "segment_agreement": 0.83,
        "confidence_variance": 0.0021,
    }),
    ("ecg", "Normal",                 0.91,  8, "low",  {
        "class_probabilities": {"Normal": 0.91, "Atrial Fibrillation": 0.02,
                                "Supraventricular": 0.01, "Ventricular": 0.01,
                                "Conduction Disorder": 0.01, "Myocardial Infarction": 0.01,
                                "Hypertrophy": 0.02, "Ischemia/ST-T": 0.01},
        "total_segments": 8,
        "segment_agreement": 1.0,
        "confidence_variance": 0.0008,
    }),
    ("ct", "Normal",                  0.82, 18, "low",  {
        "isolation_forest_label": 1,
        "anomaly_score": 0.18,
        "decision_function_raw": 0.412,
    }),
    ("ct", "Anomaly Detected",        0.74, 74, "high", {
        "isolation_forest_label": -1,
        "anomaly_score": 0.74,
        "decision_function_raw": -0.205,
    }),
    ("skin", "nv",                    0.79, 22, "low",  {
        "description": "Melanocytic Nevi",
        "all_probabilities": {"akiec": 0.02, "bcc": 0.03, "bkl": 0.05, "df": 0.02,
                              "mel": 0.07, "nv": 0.79, "vasc": 0.02},
        "margin": 0.72, "inconclusive": False,
    }),
    ("skin", "mel",                   0.68, 71, "high", {
        "description": "Melanoma",
        "all_probabilities": {"akiec": 0.04, "bcc": 0.05, "bkl": 0.08, "df": 0.02,
                              "mel": 0.68, "nv": 0.11, "vasc": 0.02},
        "margin": 0.57, "inconclusive": False,
    }),
    ("symptoms", "Myocardial Infarction", 0.62, 85, "high", {
        "top3": [
            {"disease": "Myocardial Infarction", "probability": 62.0},
            {"disease": "Atrial Fibrillation",   "probability": 18.0},
            {"disease": "Ischemia",              "probability": 12.0},
        ],
        "inconclusive": False, "filled_columns": 12, "total_columns": 28,
    }),
    ("symptoms", "Pneumonia",          0.71, 58, "moderate", {
        "top3": [
            {"disease": "Pneumonia", "probability": 71.0},
            {"disease": "Asthma",    "probability": 15.0},
            {"disease": "COPD",      "probability": 10.0},
        ],
        "inconclusive": False, "filled_columns": 9, "total_columns": 28,
    }),
    ("symptoms", "Hypertension",       0.55, 45, "moderate", {
        "top3": [
            {"disease": "Hypertension", "probability": 55.0},
            {"disease": "Diabetes",     "probability": 20.0},
            {"disease": "Normal",       "probability": 18.0},
        ],
        "inconclusive": False, "filled_columns": 7, "total_columns": 28,
    }),
]


async def _next_display_id(db, role: str) -> str:
    counter_id = "patient_seq" if role == "patient" else "doctor_seq"
    prefix     = "PAT"         if role == "patient" else "DOC"
    counter = await db.counters.find_one_and_update(
        {"_id": counter_id},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,
    )
    return f"{prefix}-{counter['seq']:03d}"


async def get_or_create_user(db, profile: dict):
    existing = await db.users.find_one({"email": profile["email"]})
    if existing:
        return existing["_id"], False

    doc = profile.copy()
    doc["password"]   = pwd.hash(doc["password"])
    doc["display_id"] = await _next_display_id(db, profile["role"])
    doc.setdefault("status", "active")

    result = await db.users.insert_one(doc)
    return result.inserted_id, True


async def main():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DB_NAME]

    # ── Users ─────────────────────────────────────────────────────────────
    pat_id, pat_new = await get_or_create_user(db, PATIENT)
    print(f"[{'NEW' if pat_new else 'KEEP'}] Patient : {PATIENT['email']}  (_id={pat_id})")

    doc_id, doc_new = await get_or_create_user(db, DOCTOR)
    print(f"[{'NEW' if doc_new else 'KEEP'}] Doctor  : {DOCTOR['email']}  (_id={doc_id})")

    # ── Reports — wipe + reseed so re-runs give a clean dataset ───────────
    deleted = await db.reports.delete_many({"patient_id": str(pat_id)})
    if deleted.deleted_count:
        print(f"[CLEAN] Removed {deleted.deleted_count} existing reports for the demo patient")

    base = datetime.utcnow() - timedelta(days=45)
    docs = []
    for i, (test_type, prediction, confidence, risk_score, risk_level, details) in enumerate(REPORTS):
        # Spread reports across the last ~45 days, in chronological order
        days_offset = (45.0 / len(REPORTS)) * i + random.uniform(-1, 1)
        created = base + timedelta(days=days_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        docs.append({
            "patient_id": str(pat_id),
            "test_type":  test_type,
            "prediction": prediction,
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "details":    details,
            "created_at": created,
        })

    result = await db.reports.insert_many(docs)
    print(f"[OK]    Inserted {len(result.inserted_ids)} reports for {PATIENT['email']}")

    print()
    print("─" * 60)
    print("  DEMO CREDENTIALS")
    print("─" * 60)
    print(f"  Patient    : {PATIENT['email']}")
    print(f"  Password   : {PATIENT['password']}")
    print(f"")
    print(f"  Doctor     : {DOCTOR['email']}")
    print(f"  Password   : {DOCTOR['password']}")
    print("─" * 60)

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
