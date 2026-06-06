from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from datetime import datetime

from auth.router import get_current_user, require_doctor
from models.schemas import UserOut, MessageCreate
from database import get_db

router = APIRouter(prefix="/api/doctor", tags=["doctor"])


def _ser(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    for k, v in doc.items():
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


@router.get("/patients")
async def list_patients(_: UserOut = Depends(require_doctor)):
    db = get_db()

    # Fetch all patients
    patients = []
    async for u in db.users.find({"role": "patient"}):
        uid = str(u["_id"])

        # Latest report + max risk score
        latest = await db.reports.find_one(
            {"patient_id": uid},
            sort=[("created_at", -1)]
        )
        count = await db.reports.count_documents({"patient_id": uid})

        # Compute max risk score across all reports
        max_risk = 0
        risk_level = "low"
        async for r in db.reports.find({"patient_id": uid}, {"risk_score": 1, "risk_level": 1}):
            if r.get("risk_score", 0) > max_risk:
                max_risk = r["risk_score"]
                risk_level = r.get("risk_level", "low")

        patients.append({
            "patient_id": uid,
            "name": u["name"],
            "email": u["email"],
            "risk_score": max_risk,
            "risk_level": risk_level,
            "last_test": latest["created_at"].isoformat() if latest else None,
            "total_reports": count,
        })

    patients.sort(key=lambda x: x["risk_score"], reverse=True)
    return patients


@router.get("/patients/search")
async def search_patients(q: str = "", _: UserOut = Depends(require_doctor)):
    db = get_db()
    if not q.strip():
        return []

    pattern = {"$regex": q.strip(), "$options": "i"}
    query = {
        "role": "patient",
        "$or": [
            {"name": pattern},
            {"display_id": pattern},
            {"email": pattern},
        ],
    }

    patients = []
    async for u in db.users.find(query, {"password": 0}).limit(20):
        uid = str(u["_id"])
        count = await db.reports.count_documents({"patient_id": uid})
        latest = await db.reports.find_one({"patient_id": uid}, sort=[("created_at", -1)])

        max_risk, risk_level = 0, "low"
        async for r in db.reports.find({"patient_id": uid}, {"risk_score": 1, "risk_level": 1}):
            if r.get("risk_score", 0) > max_risk:
                max_risk = r["risk_score"]
                risk_level = r.get("risk_level", "low")

        patients.append({
            "patient_id":   uid,
            "display_id":   u.get("display_id", ""),
            "name":         u["name"],
            "email":        u["email"],
            "blood_group":  u.get("blood_group"),
            "phone":        u.get("phone"),
            "sex":          u.get("sex"),
            "date_of_birth": u.get("date_of_birth"),
            "risk_score":   max_risk,
            "risk_level":   risk_level,
            "last_test":    latest["created_at"].isoformat() if latest else None,
            "total_reports": count,
        })

    return patients


@router.get("/patients/{patient_id}")
async def patient_detail(patient_id: str, _: UserOut = Depends(require_doctor)):
    db = get_db()
    try:
        u = await db.users.find_one({"_id": ObjectId(patient_id), "role": "patient"})
    except Exception:
        raise HTTPException(400, "Invalid patient ID")
    if not u:
        raise HTTPException(404, "Patient not found")

    reports = []
    async for r in db.reports.find({"patient_id": patient_id}, {"gradcam_image": 0}).sort("created_at", -1):
        r["report_id"] = str(r.pop("_id"))
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
        r.pop("patient_id", None)
        reports.append(r)

    return {
        "patient_id":        patient_id,
        "display_id":        u.get("display_id", ""),
        "name":              u["name"],
        "email":             u["email"],
        "phone":             u.get("phone"),
        "sex":               u.get("sex"),
        "blood_group":       u.get("blood_group"),
        "date_of_birth":     u.get("date_of_birth"),
        "emergency_contact": u.get("emergency_contact"),
        "address":           u.get("address"),
        "reports":           reports,
    }


@router.get("/stats")
async def stats(_: UserOut = Depends(require_doctor)):
    db = get_db()

    total_patients = await db.users.count_documents({"role": "patient"})
    total_reports  = await db.reports.count_documents({})

    # Condition frequency
    condition_counts: dict[str, int] = {}
    async for r in db.reports.find({}, {"prediction": 1, "test_type": 1}):
        key = r.get("prediction", "Unknown")
        condition_counts[key] = condition_counts.get(key, 0) + 1

    # Risk distribution
    risk_dist = {"low": 0, "moderate": 0, "high": 0}
    async for r in db.reports.find({}, {"risk_level": 1}):
        lvl = r.get("risk_level", "low")
        risk_dist[lvl] = risk_dist.get(lvl, 0) + 1

    # Test type distribution
    type_dist: dict[str, int] = {}
    async for r in db.reports.find({}, {"test_type": 1}):
        t = r.get("test_type", "unknown")
        type_dist[t] = type_dist.get(t, 0) + 1

    top_conditions = sorted(condition_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "total_patients": total_patients,
        "total_reports": total_reports,
        "top_conditions": [{"name": n, "count": c} for n, c in top_conditions],
        "risk_distribution": risk_dist,
        "test_type_distribution": type_dist,
    }


# ── Messaging ─────────────────────────────────────────────────────────────────

@router.post("/message")
async def send_message(body: MessageCreate, doctor: UserOut = Depends(require_doctor)):
    db = get_db()
    try:
        _ = ObjectId(body.patient_id)
    except Exception:
        raise HTTPException(400, "Invalid patient ID")

    msg = {
        "from_user_id":    doctor.id,
        "from_role":       "doctor",
        "from_name":       doctor.name,
        "to_user_id":      body.patient_id,
        "content":         body.content,
        "report_id":       body.report_id,
        "report_type":     body.report_type,
        "report_prediction": body.report_prediction,
        "created_at":      datetime.utcnow(),
        "read":            False,
    }
    result = await db.messages.insert_one(msg)
    return {"message_id": str(result.inserted_id)}


@router.get("/messages/{patient_id}")
async def doctor_messages(patient_id: str, _: UserOut = Depends(require_doctor)):
    db = get_db()
    msgs = []
    async for m in db.messages.find(
        {"$or": [{"from_user_id": patient_id}, {"to_user_id": patient_id}]}
    ).sort("created_at", 1):
        msgs.append({
            "id":               str(m["_id"]),
            "from_role":        m["from_role"],
            "from_name":        m.get("from_name", ""),
            "content":          m["content"],
            "report_id":        m.get("report_id"),
            "report_type":      m.get("report_type"),
            "report_prediction": m.get("report_prediction"),
            "created_at":       m["created_at"].isoformat(),
        })
    return msgs


@router.get("/report/{report_id}")
async def get_full_report(report_id: str, _: UserOut = Depends(require_doctor)):
    db = get_db()
    try:
        doc = await db.reports.find_one({"_id": ObjectId(report_id)})
    except Exception:
        raise HTTPException(400, "Invalid report ID")
    if not doc:
        raise HTTPException(404, "Report not found")

    doc["report_id"] = str(doc.pop("_id"))
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc
