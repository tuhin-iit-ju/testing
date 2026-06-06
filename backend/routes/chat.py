from fastapi import APIRouter, Depends, HTTPException

from auth.router import get_current_user
from models.schemas import UserOut, ChatRequest, ChatResponse
import services.groq_service as groq_svc

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, current_user: UserOut = Depends(get_current_user)):
    history = [{"role": m.role, "content": m.content} for m in body.history]
    history.append({"role": "user", "content": body.message})

    try:
        reply = groq_svc.chat(history, context=body.context)
    except ValueError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Chat service error: {e}")

    return ChatResponse(reply=reply)


@router.get("/patient/unread-count")
async def unread_count(current_user: UserOut = Depends(get_current_user)):
    from database import get_db
    db = get_db()
    count = await db.messages.count_documents({"to_user_id": current_user.id, "read": False})
    return {"count": count}


@router.get("/patient/messages")
async def patient_messages(current_user: UserOut = Depends(get_current_user)):
    from database import get_db
    db = get_db()
    msgs = []
    async for m in db.messages.find({"to_user_id": current_user.id}).sort("created_at", 1):
        msgs.append({
            "id":               str(m["_id"]),
            "from_role":        m["from_role"],
            "from_name":        m.get("from_name", "Doctor"),
            "content":          m["content"],
            "report_id":        m.get("report_id"),
            "report_type":      m.get("report_type"),
            "report_prediction": m.get("report_prediction"),
            "created_at":       m["created_at"].isoformat(),
            "read":             m.get("read", False),
        })
    # Mark all as read
    await db.messages.update_many({"to_user_id": current_user.id}, {"$set": {"read": True}})
    return msgs
