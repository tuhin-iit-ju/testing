import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from bson import ObjectId

from auth.router import get_current_user
from models.schemas import UserOut, AnalysisResult, SymptomAnswers
from database import get_db
from services.risk_score import compute_risk_score
import services.groq_service as groq_svc
import services.cloudinary_service as cdn

router = APIRouter(prefix="/api/analyze", tags=["analyze"])

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/tiff", "image/bmp", "application/octet-stream",
}
ALLOWED_CSV_TYPES   = {"text/csv", "application/csv", "text/plain", "application/octet-stream"}


async def _save_report(db, user_id: str, report: dict) -> str:
    report["patient_id"] = user_id
    report["created_at"] = datetime.utcnow()
    result = await db.reports.insert_one(report)
    return str(result.inserted_id)


async def _build_response(report_id: str, test_type: str, prediction: str,
                    confidence: float, details: dict, gradcam: str | None = None,
                    anomaly_score: float = 0.0) -> AnalysisResult:
    risk_score, risk_level = compute_risk_score(test_type, prediction, confidence, anomaly_score)

    db = get_db()
    await db.reports.update_one(
        {"_id": ObjectId(report_id)},
        {"$set": {"risk_score": risk_score, "risk_level": risk_level}},
    )

    recommendation = None
    try:
        recommendation = groq_svc.generate_recommendation(test_type, prediction, confidence, risk_level)
    except Exception as e:
        print(f"[ROUTE] Groq recommendation failed: {e}")

    return AnalysisResult(
        report_id=report_id,
        test_type=test_type,
        prediction=prediction,
        confidence=confidence,
        risk_level=risk_level,
        risk_score=risk_score,
        details=details,
        gradcam_image=gradcam,
        recommendation=recommendation,
    )


# ── X-Ray ────────────────────────────────────────────────────────────────────

@router.post("/xray", response_model=AnalysisResult)
async def analyze_xray(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, "Upload a JPEG or PNG image")

    from services.ml import xray as xray_svc
    image_bytes = await file.read()
    try:
        result = xray_svc.predict(image_bytes)
    except Exception as e:
        raise HTTPException(500, f"X-Ray inference failed: {e}")

    image_url  = cdn.upload_bytes(image_bytes, "xray")
    gradcam_url = cdn.upload_base64(result.get("gradcam_image", ""), "gradcam")

    db = get_db()
    doc = {
        "test_type": "xray",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "details": {
            "detected_conditions": result.get("detected_conditions", []),
            "all_probabilities": result.get("all_probabilities", {}),
            "per_model": result.get("per_model", {}),
            "model_agreement": result.get("model_agreement"),
        },
        "image_url": image_url,
        "gradcam_image": gradcam_url,
    }
    report_id = await _save_report(db, current_user.id, doc)
    return await _build_response(report_id, "xray", result["prediction"], result["confidence"],
                                 doc["details"], gradcam_url)


# ── ECG ──────────────────────────────────────────────────────────────────────

@router.post("/ecg", response_model=AnalysisResult)
async def analyze_ecg(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    csv_bytes = await file.read()
    from services.ml import ecg as ecg_svc
    try:
        result = ecg_svc.predict(csv_bytes)
    except Exception as e:
        raise HTTPException(500, f"ECG inference failed: {e}")

    db = get_db()
    doc = {
        "test_type": "ecg",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "details": {
            "class_probabilities": result.get("class_probabilities", {}),
            "total_segments": result.get("total_segments", 1),
            "segments": result.get("segments", []),
            "segment_agreement": result.get("segment_agreement"),
            "confidence_variance": result.get("confidence_variance"),
        },
        # ECG is CSV — no image to upload
    }
    report_id = await _save_report(db, current_user.id, doc)
    return await _build_response(report_id, "ecg", result["prediction"], result["confidence"], doc["details"])


# ── CT Scan ───────────────────────────────────────────────────────────────────

@router.post("/ct", response_model=AnalysisResult)
async def analyze_ct(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, "Upload a JPEG or PNG image")

    image_bytes = await file.read()
    from services.ml import ct_scan as ct_svc
    try:
        result = ct_svc.predict(image_bytes)
    except Exception as e:
        raise HTTPException(500, f"CT inference failed: {e}")

    image_url = cdn.upload_bytes(image_bytes, "ct")

    db = get_db()
    doc = {
        "test_type": "ct",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "details": result.get("details", {}),
        "image_url": image_url,
    }
    report_id = await _save_report(db, current_user.id, doc)
    return await _build_response(report_id, "ct", result["prediction"], result["confidence"],
                                 doc["details"], anomaly_score=result.get("anomaly_score", 0.0))


# ── Skin Disease ─────────────────────────────────────────────────────────────

@router.post("/skin", response_model=AnalysisResult)
async def analyze_skin(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, "Upload a JPEG or PNG image")

    image_bytes = await file.read()
    from services.ml import skin as skin_svc
    try:
        result = skin_svc.predict(image_bytes)
    except Exception as e:
        raise HTTPException(500, f"Skin inference failed: {e}")

    image_url   = cdn.upload_bytes(image_bytes, "skin")
    gradcam_url = cdn.upload_base64(result.get("gradcam_image", ""), "gradcam")

    db = get_db()
    doc = {
        "test_type": "skin",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "details": {
            "description": result.get("description", ""),
            "all_probabilities": result.get("all_probabilities", {}),
            "model1_result": result.get("model1_result"),
            "model2_result": result.get("model2_result"),
            "model3_result": result.get("model3_result"),
            "margin": result.get("margin"),
            "inconclusive": result.get("inconclusive", False),
        },
        "image_url": image_url,
        "gradcam_image": gradcam_url,
    }
    report_id = await _save_report(db, current_user.id, doc)
    return await _build_response(report_id, "skin", result["prediction"], result["confidence"],
                                 doc["details"], gradcam_url)


# ── Symptom Checker ───────────────────────────────────────────────────────────

@router.post("/symptoms", response_model=AnalysisResult)
async def analyze_symptoms(
    body: SymptomAnswers,
    current_user: UserOut = Depends(get_current_user),
):
    from services.ml import symptoms as sym_svc
    try:
        result = sym_svc.predict(body.model_dump())
    except Exception as e:
        raise HTTPException(500, f"Symptom prediction failed: {e}")

    # Generate prescription before saving so it is persisted with the report
    prescription = None
    try:
        prescription = groq_svc.generate_symptom_prescription(
            prediction  = result["prediction"],
            top3        = result.get("top3", []),
            confidence  = result["confidence"],
            filled      = result.get("filled_columns", 0),
            total       = result.get("total_columns", 28),
            risk_level  = "high" if result["confidence"] > 0.70 else "moderate",
        )
    except Exception as e:
        print(f"[ROUTE] Symptom prescription failed: {e}")

    db = get_db()
    doc = {
        "test_type": "symptoms",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "details": {
            "top3":           result.get("top3", []),
            "inconclusive":   result.get("inconclusive", False),
            "filled_columns": result.get("filled_columns", 0),
            "total_columns":  result.get("total_columns", 28),
            "prescription":   prescription,
        },
    }
    report_id = await _save_report(db, current_user.id, doc)

    risk_score, risk_level = compute_risk_score(
        "symptoms", result["prediction"], result["confidence"], 0
    )
    await db.reports.update_one(
        {"_id": ObjectId(report_id)},
        {"$set": {"risk_score": risk_score, "risk_level": risk_level}},
    )
    from models.schemas import AnalysisResult
    return AnalysisResult(
        report_id      = report_id,
        test_type      = "symptoms",
        prediction     = result["prediction"],
        confidence     = result["confidence"],
        risk_level     = risk_level,
        risk_score     = risk_score,
        details        = doc["details"],
        recommendation = prescription,
    )
