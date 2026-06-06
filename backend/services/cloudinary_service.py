import base64
import io

import cloudinary
import cloudinary.uploader

from config import settings

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)


def upload_bytes(image_bytes: bytes, folder: str, public_id: str | None = None) -> str | None:
    try:
        result = cloudinary.uploader.upload(
            image_bytes,
            folder=f"healthbridge/{folder}",
            public_id=public_id,
            resource_type="image",
        )
        return result["secure_url"]
    except Exception as e:
        print(f"[CLOUDINARY] upload failed: {e}")
        return None


def upload_base64(b64_string: str, folder: str, public_id: str | None = None) -> str | None:
    if not b64_string:
        return None
    # Strip data-URI prefix if present (e.g. "data:image/png;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(b64_string)
        return upload_bytes(image_bytes, folder, public_id)
    except Exception as e:
        print(f"[CLOUDINARY] base64 decode/upload failed: {e}")
        return None
