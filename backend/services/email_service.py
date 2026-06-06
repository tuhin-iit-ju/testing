import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import settings


def _send(to: str, subject: str, html: str) -> None:
    if not settings.GMAIL_USER or not settings.GMAIL_APP_PASSWORD:
        print("[Email] Gmail not configured — skipping send")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"UyeCare <{settings.GMAIL_USER}>"
        msg["To"]      = to
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo()
            s.starttls()
            s.login(settings.GMAIL_USER, settings.GMAIL_APP_PASSWORD)
            s.sendmail(settings.GMAIL_USER, to, msg.as_string())
        print(f"[Email] Sent '{subject}' → {to}")
    except Exception as e:
        print(f"[Email] Failed to send to {to}: {e}")


def send_async(to: str, subject: str, html: str) -> None:
    threading.Thread(target=_send, args=(to, subject, html), daemon=True).start()


# ── Templates ─────────────────────────────────────────────────────────────────

def send_doctor_approved(to: str, name: str) -> None:
    subject = "Your UyeCare Doctor Account Has Been Approved"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:560px;margin:auto;padding:32px;background:#f8fafc;border-radius:12px;">
      <div style="text-align:center;margin-bottom:24px;">
        <span style="display:inline-block;background:#2563eb;color:white;padding:10px 20px;border-radius:8px;font-size:20px;font-weight:bold;">UyeCare</span>
      </div>
      <h2 style="color:#1e293b;margin-bottom:8px;">Welcome aboard, Dr. {name}!</h2>
      <p style="color:#475569;line-height:1.6;">
        Great news — your UyeCare doctor account has been <strong style="color:#16a34a;">approved</strong> by our admin team.
        You can now log in to the Doctor Portal and start viewing patient reports, sending messages, and using the platform.
      </p>
      <div style="text-align:center;margin:32px 0;">
        <a href="http://localhost:5173/login"
           style="background:#2563eb;color:white;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:bold;font-size:15px;">
          Log In to UyeCare
        </a>
      </div>
      <p style="color:#94a3b8;font-size:12px;text-align:center;margin-top:24px;">
        If you did not register for UyeCare, please ignore this email.
      </p>
    </div>
    """
    send_async(to, subject, html)


def send_doctor_rejected(to: str, name: str) -> None:
    subject = "Update on Your UyeCare Doctor Account Application"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:560px;margin:auto;padding:32px;background:#f8fafc;border-radius:12px;">
      <div style="text-align:center;margin-bottom:24px;">
        <span style="display:inline-block;background:#2563eb;color:white;padding:10px 20px;border-radius:8px;font-size:20px;font-weight:bold;">UyeCare</span>
      </div>
      <h2 style="color:#1e293b;margin-bottom:8px;">Hello, Dr. {name}</h2>
      <p style="color:#475569;line-height:1.6;">
        After review, your UyeCare doctor account application has <strong style="color:#dc2626;">not been approved</strong> at this time.
        If you believe this is a mistake or would like to provide additional information, please contact our support team.
      </p>
      <p style="color:#94a3b8;font-size:12px;text-align:center;margin-top:32px;">
        UyeCare · AI-powered healthcare platform
      </p>
    </div>
    """
    send_async(to, subject, html)
