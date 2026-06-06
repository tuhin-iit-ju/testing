from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import close_db
from auth.router import router as auth_router
from routes.analyze import router as analyze_router
from routes.history import router as history_router
from routes.doctor import router as doctor_router
from routes.chat import router as chat_router
from routes.admin import router as admin_router
from routes.health_profile import router as health_profile_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_db()


app = FastAPI(
    title="UyeCare API",
    description="Clinical AI Diagnostic Platform — UyeCare",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(analyze_router)
app.include_router(history_router)
app.include_router(doctor_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(health_profile_router)


@app.get("/")
async def root():
    return {"message": "UyeCare API is running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}
