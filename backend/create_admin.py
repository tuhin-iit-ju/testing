"""
Run once to seed the admin user:
  python create_admin.py
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from config import settings

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

ADMIN_EMAIL    = "admin@uyecare.com"
ADMIN_PASSWORD = "UyeCare@Admin2025"
ADMIN_NAME     = "UyeCare Admin"

async def main():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DB_NAME]

    existing = await db.users.find_one({"email": ADMIN_EMAIL})
    if existing:
        print(f"[INFO] Admin already exists: {ADMIN_EMAIL}")
        client.close()
        return

    result = await db.users.insert_one({
        "name":     ADMIN_NAME,
        "email":    ADMIN_EMAIL,
        "password": pwd.hash(ADMIN_PASSWORD),
        "role":     "admin",
    })
    print(f"[OK] Admin created — id: {result.inserted_id}")
    print(f"     Email   : {ADMIN_EMAIL}")
    print(f"     Password: {ADMIN_PASSWORD}")
    client.close()

asyncio.run(main())
