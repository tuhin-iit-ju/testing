from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from datetime import datetime

from auth.router import get_current_user
from models.schemas import UserOut, ReportSummary
from database import get_db

router = APIRouter(prefix="/api/history", tags=["history"])


def _serialize_report(doc: dict) -> dict:
    doc["report_id"] = str(doc.pop("_id"))
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    doc.pop("patient_id", None)
    return doc


@router.get("")
async def get_history(current_user: UserOut = Depends(get_current_user)):
    db = get_db()
    cursor = db.reports.find(
        {"patient_id": current_user.id},
        {"gradcam_image": 0}   # exclude large base64 from list
    ).sort("created_at", -1)

    reports = []
    async for doc in cursor:
        reports.append(_serialize_report(doc))
    return reports


@router.get("/{report_id}")
async def get_report(report_id: str, current_user: UserOut = Depends(get_current_user)):
    db = get_db()
    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(400, "Invalid report ID")

    doc = await db.reports.find_one({"_id": oid, "patient_id": current_user.id})
    if not doc:
        raise HTTPException(404, "Report not found")
    return _serialize_report(doc)


@router.delete("/{report_id}")
async def delete_report(report_id: str, current_user: UserOut = Depends(get_current_user)):
    db = get_db()
    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(400, "Invalid report ID")

    result = await db.reports.delete_one({"_id": oid, "patient_id": current_user.id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Report not found")
    return {"deleted": True}
