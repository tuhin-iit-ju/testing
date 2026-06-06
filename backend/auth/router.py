from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId

from database import get_db
from models.schemas import UserCreate, UserLogin, UserOut, Token
from auth.service import hash_password, verify_password, create_access_token, decode_token

router = APIRouter(prefix="/api/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

_PROFILE_FIELDS = [
    "date_of_birth", "sex", "blood_group", "phone",
    "emergency_contact", "address",
    "specialty", "license_no", "hospital", "experience_years", "bio",
]


def _serialize_user(doc: dict) -> UserOut:
    kwargs = {f: doc.get(f) for f in _PROFILE_FIELDS}
    return UserOut(
        id=str(doc["_id"]),
        display_id=doc.get("display_id", ""),
        name=doc["name"],
        email=doc["email"],
        role=doc["role"],
        status=doc.get("status", "active"),
        **kwargs,
    )


async def _next_display_id(db, role: str) -> str:
    counter_id = "patient_seq" if role == "patient" else "doctor_seq"
    prefix     = "PAT"         if role == "patient" else "DOC"
    doc = await db.counters.find_one_and_update(
        {"_id": counter_id},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,
    )
    return f"{prefix}-{doc['seq']:03d}"


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = get_db()
    doc = await db.users.find_one({"_id": ObjectId(payload["sub"])})
    if not doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _serialize_user(doc)


async def require_doctor(user: UserOut = Depends(get_current_user)) -> UserOut:
    if user.role != "doctor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor access required")
    return user


async def require_admin(user: UserOut = Depends(get_current_user)) -> UserOut:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


@router.post("/register", response_model=Token)
async def register(body: UserCreate):
    db = get_db()
    if await db.users.find_one({"email": body.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    display_id = await _next_display_id(db, body.role) if body.role in ("patient", "doctor") else ""
    # Doctors start as pending; patients are immediately active
    user_status = "pending_approval" if body.role == "doctor" else "active"

    doc = {
        "name":       body.name,
        "email":      body.email,
        "password":   hash_password(body.password),
        "role":       body.role,
        "display_id": display_id,
        "status":     user_status,
    }
    for f in _PROFILE_FIELDS:
        v = getattr(body, f, None)
        if v is not None:
            doc[f] = v

    result = await db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    user  = _serialize_user(doc)
    token = create_access_token({"sub": str(result.inserted_id), "role": body.role})
    return Token(access_token=token, user=user)


@router.post("/login", response_model=Token)
async def login(body: UserLogin):
    db = get_db()
    doc = await db.users.find_one({"email": body.email})
    if not doc or not verify_password(body.password, doc["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_status = doc.get("status", "active")
    if user_status == "pending_approval":
        raise HTTPException(
            status_code=403,
            detail="Your account is pending admin approval. You will be notified once approved."
        )
    if user_status == "rejected":
        raise HTTPException(
            status_code=403,
            detail="Your registration was rejected. Please contact support."
        )

    user  = _serialize_user(doc)
    token = create_access_token({"sub": str(doc["_id"]), "role": doc["role"]})
    return Token(access_token=token, user=user)


@router.get("/me", response_model=UserOut)
async def me(current_user: UserOut = Depends(get_current_user)):
    return current_user


_EDITABLE_FIELDS = {
    "name", "phone", "date_of_birth", "sex", "blood_group",
    "emergency_contact", "address",
    "specialty", "license_no", "hospital", "experience_years", "bio",
}


@router.patch("/me", response_model=UserOut)
async def update_me(body: dict, current_user: UserOut = Depends(get_current_user)):
    """Patch the current user's profile. Only whitelisted fields are accepted."""
    updates = {k: v for k, v in body.items() if k in _EDITABLE_FIELDS}
    if not updates:
        return current_user

    db = get_db()
    await db.users.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": updates},
    )
    doc = await db.users.find_one({"_id": ObjectId(current_user.id)})
    return _serialize_user(doc)
