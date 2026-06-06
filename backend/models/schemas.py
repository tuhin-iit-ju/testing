from pydantic import BaseModel, EmailStr, Field
from typing import Any, Optional
from datetime import datetime


# ── Auth ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "patient"                # "patient" | "doctor"

    # ── Patient profile fields ───────────────────────────────────────────────
    date_of_birth:     Optional[str] = None   # YYYY-MM-DD
    sex:               Optional[str] = None   # Male | Female | Other
    blood_group:       Optional[str] = None   # A+, A-, B+, …
    phone:             Optional[str] = None
    emergency_contact: Optional[str] = None   # "Name: phone"
    address:           Optional[str] = None

    # ── Doctor profile fields ────────────────────────────────────────────────
    specialty:        Optional[str] = None
    license_no:       Optional[str] = None
    hospital:         Optional[str] = None
    experience_years: Optional[int] = None
    bio:              Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id:         str
    display_id: str = ""
    name:       str
    email:      str
    role:       str
    status:     str = "active"           # active | pending_approval | rejected

    # ── Patient profile ──────────────────────────────────────────────────────
    date_of_birth:     Optional[str] = None
    sex:               Optional[str] = None
    blood_group:       Optional[str] = None
    phone:             Optional[str] = None
    emergency_contact: Optional[str] = None
    address:           Optional[str] = None

    # ── Doctor profile ───────────────────────────────────────────────────────
    specialty:        Optional[str] = None
    license_no:       Optional[str] = None
    hospital:         Optional[str] = None
    experience_years: Optional[int] = None
    bio:              Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserOut


# ── Analysis responses ───────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    report_id:    str
    test_type:    str
    prediction:   str
    confidence:   float
    risk_level:   str           # "low" | "moderate" | "high"
    risk_score:   int           # 0-100
    details:      dict[str, Any]
    gradcam_image: Optional[str] = None   # base64 PNG
    recommendation: Optional[str] = None
    created_at:   datetime = Field(default_factory=datetime.utcnow)


# ── Symptom checker ───────────────────────────────────────────────────────────

class SymptomAnswers(BaseModel):
    age: int
    sex: str
    smoking_history:    int = 0
    chief_complaint:    str
    chest_pain:         int = 0
    chest_pain_radiates: int = 0
    palpitation_type:   str = "none"
    syncope:            int = 0
    breath_pattern:     str = "none"
    leg_swelling:       int = 0
    cough_type:         str = "none"
    fever:              int = 0
    skin_lesion:        int = 0
    lesion_changing:    int = 0
    sun_exposure_history: int = 0
    lesion_features:    str = "none"
    systemic:           str = "none"
    other_systemic:     int = 0
    risk_factors:       str = "none"


# ── History ──────────────────────────────────────────────────────────────────

class ReportSummary(BaseModel):
    report_id:  str
    test_type:  str
    prediction: str
    confidence: float
    risk_level: str
    risk_score: int
    created_at: datetime


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    str    # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


# ── Doctor ───────────────────────────────────────────────────────────────────

class PatientSummary(BaseModel):
    patient_id:   str
    name:         str
    email:        str
    risk_score:   int
    risk_level:   str
    last_test:    Optional[datetime]
    total_reports: int


class MessageCreate(BaseModel):
    patient_id:        str
    content:           str
    report_id:         Optional[str] = None
    report_type:       Optional[str] = None
    report_prediction: Optional[str] = None


class MessageOut(BaseModel):
    id:         str
    from_role:  str
    content:    str
    created_at: datetime
