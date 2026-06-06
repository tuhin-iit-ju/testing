from fastapi import APIRouter, Depends
from datetime import datetime

from auth.router import get_current_user
from models.schemas import UserOut
from database import get_db

router = APIRouter(prefix="/api/health-profile", tags=["health-profile"])


def _clean(doc: dict) -> dict:
    doc.pop("_id", None)
    doc.pop("patient_id", None)
    if isinstance(doc.get("updated_at"), datetime):
        doc["updated_at"] = doc["updated_at"].isoformat()
    return doc


@router.get("")
async def get_profile(current_user: UserOut = Depends(get_current_user)):
    db = get_db()
    doc = await db.health_profiles.find_one({"patient_id": current_user.id})
    if not doc:
        return {}
    return _clean(doc)


@router.put("")
async def save_profile(body: dict, current_user: UserOut = Depends(get_current_user)):
    db = get_db()
    body["patient_id"] = current_user.id
    body["updated_at"] = datetime.utcnow()
    await db.health_profiles.update_one(
        {"patient_id": current_user.id},
        {"$set": body},
        upsert=True,
    )
    return {"ok": True}
