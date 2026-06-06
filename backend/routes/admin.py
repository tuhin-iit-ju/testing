from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from datetime import datetime

from auth.router import require_admin
from models.schemas import UserOut
from database import get_db
import services.email_service as email_svc

router = APIRouter(prefix="/api/admin", tags=["admin"])

_PROFILE_FIELDS = [
    "date_of_birth", "sex", "blood_group", "phone",
    "emergency_contact", "address",
    "specialty", "license_no", "hospital", "experience_years", "bio",
]


def _fmt_dt(v):
    return v.isoformat() if isinstance(v, datetime) else v


def _ser_user(u: dict, report_count: int = 0) -> dict:
    base = {
        "id":          str(u["_id"]),
        "display_id":  u.get("display_id", ""),
        "name":        u["name"],
        "email":       u["email"],
        "role":        u["role"],
        "status":      u.get("status", "active"),
        "report_count": report_count,
    }
    for f in _PROFILE_FIELDS:
        base[f] = u.get(f)
    return base


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def admin_stats(_: UserOut = Depends(require_admin)):
    db = get_db()

    total_patients         = await db.users.count_documents({"role": "patient"})
    total_doctors          = await db.users.count_documents({"role": "doctor", "status": "active"})
    pending_doctors        = await db.users.count_documents({"role": "doctor", "status": "pending_approval"})
    total_reports          = await db.reports.count_documents({})

    risk_dist = {"low": 0, "moderate": 0, "high": 0}
    async for r in db.reports.find({}, {"risk_level": 1}):
        lvl = r.get("risk_level", "low")
        risk_dist[lvl] = risk_dist.get(lvl, 0) + 1

    type_dist: dict[str, int] = {}
    async for r in db.reports.find({}, {"test_type": 1}):
        t = r.get("test_type", "unknown")
        type_dist[t] = type_dist.get(t, 0) + 1

    condition_counts: dict[str, int] = {}
    async for r in db.reports.find({}, {"prediction": 1}):
        key = r.get("prediction", "Unknown")
        condition_counts[key] = condition_counts.get(key, 0) + 1
    top_conditions = sorted(condition_counts.items(), key=lambda x: -x[1])[:10]

    recent_reports = []
    async for r in db.reports.find(
        {}, {"patient_id": 1, "test_type": 1, "prediction": 1, "risk_level": 1, "risk_score": 1, "created_at": 1}
    ).sort("created_at", -1).limit(10):
        try:
            patient = await db.users.find_one({"_id": ObjectId(r["patient_id"])}, {"name": 1, "display_id": 1})
            p_name  = patient["name"]       if patient else "Unknown"
            p_did   = patient.get("display_id", "") if patient else ""
        except Exception:
            p_name, p_did = "Unknown", ""
        recent_reports.append({
            "report_id":    str(r["_id"]),
            "patient_name": p_name,
            "patient_id_label": p_did,
            "test_type":    r.get("test_type", ""),
            "prediction":   r.get("prediction", ""),
            "risk_level":   r.get("risk_level", "low"),
            "risk_score":   r.get("risk_score", 0),
            "created_at":   _fmt_dt(r.get("created_at")),
        })

    return {
        "total_patients":    total_patients,
        "total_doctors":     total_doctors,
        "pending_doctors":   pending_doctors,
        "total_reports":     total_reports,
        "risk_distribution": risk_dist,
        "test_type_distribution": type_dist,
        "top_conditions":    [{"name": n, "count": c} for n, c in top_conditions],
        "recent_reports":    recent_reports,
    }


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(_: UserOut = Depends(require_admin)):
    db = get_db()
    users = []
    async for u in db.users.find({}, {"password": 0}).sort("_id", -1):
        uid = str(u["_id"])
        report_count = await db.reports.count_documents({"patient_id": uid})
        users.append(_ser_user(u, report_count))
    return users


@router.patch("/users/{user_id}/role")
async def update_user_role(user_id: str, body: dict, _: UserOut = Depends(require_admin)):
    new_role = body.get("role")
    if new_role not in ("patient", "doctor", "admin"):
        raise HTTPException(400, "Invalid role")
    db = get_db()
    try:
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": new_role}},
        )
    except Exception:
        raise HTTPException(400, "Invalid user ID")
    if result.matched_count == 0:
        raise HTTPException(404, "User not found")
    return {"ok": True}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, _: UserOut = Depends(require_admin)):
    db = get_db()
    try:
        result = await db.users.delete_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(400, "Invalid user ID")
    if result.deleted_count == 0:
        raise HTTPException(404, "User not found")
    await db.reports.delete_many({"patient_id": user_id})
    return {"ok": True}


# ── Doctor approval ───────────────────────────────────────────────────────────

@router.get("/doctors/pending")
async def pending_doctors(_: UserOut = Depends(require_admin)):
    db = get_db()
    doctors = []
    async for u in db.users.find(
        {"role": "doctor", "status": "pending_approval"}, {"password": 0}
    ).sort("_id", -1):
        doctors.append(_ser_user(u))
    return doctors


@router.patch("/doctors/{user_id}/approve")
async def approve_doctor(user_id: str, _: UserOut = Depends(require_admin)):
    db = get_db()
    try:
        doc = await db.users.find_one_and_update(
            {"_id": ObjectId(user_id), "role": "doctor"},
            {"$set": {"status": "active"}},
        )
    except Exception:
        raise HTTPException(400, "Invalid ID")
    if not doc:
        raise HTTPException(404, "Doctor not found")
    email_svc.send_doctor_approved(doc["email"], doc["name"])
    return {"ok": True}


@router.patch("/doctors/{user_id}/reject")
async def reject_doctor(user_id: str, _: UserOut = Depends(require_admin)):
    db = get_db()
    try:
        doc = await db.users.find_one_and_update(
            {"_id": ObjectId(user_id), "role": "doctor"},
            {"$set": {"status": "rejected"}},
        )
    except Exception:
        raise HTTPException(400, "Invalid ID")
    if not doc:
        raise HTTPException(404, "Doctor not found")
    email_svc.send_doctor_rejected(doc["email"], doc["name"])
    return {"ok": True}


# ── Reports ───────────────────────────────────────────────────────────────────

@router.get("/reports")
async def list_all_reports(
    skip: int = 0, limit: int = 50, _: UserOut = Depends(require_admin)
):
    db = get_db()
    reports = []
    async for r in db.reports.find(
        {}, {"gradcam_image": 0}
    ).sort("created_at", -1).skip(skip).limit(limit):
        try:
            patient = await db.users.find_one(
                {"_id": ObjectId(r["patient_id"])}, {"name": 1, "email": 1, "display_id": 1}
            )
            p_name  = patient["name"]             if patient else "Unknown"
            p_email = patient["email"]            if patient else ""
            p_did   = patient.get("display_id","") if patient else ""
        except Exception:
            p_name, p_email, p_did = "Unknown", "", ""
        reports.append({
            "report_id":     str(r["_id"]),
            "patient_id":    r["patient_id"],
            "patient_label": p_did or p_name,
            "patient_name":  p_name,
            "patient_email": p_email,
            "test_type":     r.get("test_type", ""),
            "prediction":    r.get("prediction", ""),
            "confidence":    r.get("confidence", 0),
            "risk_level":    r.get("risk_level", "low"),
            "risk_score":    r.get("risk_score", 0),
            "created_at":    _fmt_dt(r.get("created_at")),
        })
    total = await db.reports.count_documents({})
    return {"reports": reports, "total": total}
