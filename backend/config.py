from pathlib import Path
from pydantic_settings import BaseSettings

# d:\Projects\UyeCare  (three levels up from healthbridge/backend)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    DB_NAME: str = "healthbridge"
    SECRET_KEY: str = "healthbridge-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    GROQ_API_KEY: str = ""

    # ── Gmail ────────────────────────────────────────────────────────────────
    GMAIL_USER: str = ""
    GMAIL_APP_PASSWORD: str = ""

    # ── Cloudinary ───────────────────────────────────────────────────────────
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""

    # ── X-Ray model weights ──────────────────────────────────────────────────
    XRAY_DENSENET_PATH: str = str(_PROJECT_ROOT / "X-Ray" / "Trained Model" / "densenet121_chest_xray.pth")
    XRAY_RESNET_PATH:   str = str(_PROJECT_ROOT / "X-Ray" / "Trained Model" / "resnet50_chest_xray.pth")
    XRAY_VIT_PATH:      str = str(_PROJECT_ROOT / "X-Ray" / "Trained Model" / "vit_base_chest_xray.pth")

    # ── ECG model weights ────────────────────────────────────────────────────
    ECG_RESNET_PATH:       str = str(_PROJECT_ROOT / "ECG" / "Trained Model" / "resnet_final.keras")
    ECG_INCEPTION_PATH:    str = str(_PROJECT_ROOT / "ECG" / "Trained Model" / "inception_final.keras")
    ECG_TRANSFORMER_PATH:  str = str(_PROJECT_ROOT / "ECG" / "Trained Model" / "transformer_final.keras")

    # ── CT scan models ───────────────────────────────────────────────────────
    CT_AUTOENCODER_PATH: str = str(_PROJECT_ROOT / "ct_scan" / "Models" / "autoencoder_best.pth")
    CT_ANOMALY_NET_PATH: str = str(_PROJECT_ROOT / "ct_scan" / "Models" / "anomaly_net_best.pth")
    CT_IF_PATH:          str = str(_PROJECT_ROOT / "ct_scan" / "Models" / "isolation_forest.pkl")
    CT_PCA_PATH:         str = str(_PROJECT_ROOT / "ct_scan" / "Models" / "pca.pkl")
    CT_SCALER_PATH:      str = str(_PROJECT_ROOT / "ct_scan" / "Models" / "scaler.pkl")

    # ── Skin disease models ──────────────────────────────────────────────────
    SKIN_MODEL1_PATH: str = str(_PROJECT_ROOT / "skin disease" / "skin_model_package" / "models" / "skin_model_1.pth")
    SKIN_MODEL2_PATH: str = str(_PROJECT_ROOT / "skin disease" / "skin_model_package" / "models" / "skin_model_2.pth")
    SKIN_MODEL3_PATH: str = str(_PROJECT_ROOT / "skin disease" / "skin_model_package" / "models" / "skin_model_3.pth")

    # ── Symptom / All-disease checkup ────────────────────────────────────────
    SYMPTOM_MODEL_PATH:  str = str(_PROJECT_ROOT / "All Disease Checkup" / "trained_models" / "symptom_model.pkl")
    LABEL_ENCODER_PATH:  str = str(_PROJECT_ROOT / "All Disease Checkup" / "trained_models" / "label_encoder.pkl")
    FEATURE_COLS_PATH:   str = str(_PROJECT_ROOT / "All Disease Checkup" / "trained_models" / "feature_columns.pkl")
    QUESTION_ENGINE_PATH: str = str(_PROJECT_ROOT / "All Disease Checkup" / "trained_models" / "question_engine.pkl")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
