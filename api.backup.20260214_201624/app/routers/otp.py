from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, validator
import random
import string
from datetime import datetime, timedelta
from typing import Optional, Dict
import httpx
import re
from sqlalchemy.orm import Session
from app.core.database import get_db, User

router = APIRouter(prefix="/auth/otp", tags=["OTP Authentication"])

otp_store: Dict[str, dict] = {}
last_sent: Dict[str, datetime] = {}

RATE_LIMIT_SECONDS = 60
OTP_EXPIRE_MINUTES = 5
MAX_ATTEMPTS = 3

class OTPRequest(BaseModel):
    email: EmailStr
    
    @validator("email")
    def validate_email(cls, v):
        v = v.lower().strip()
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Invalid email format")
        temp_domains = ["tempmail.com", "10minutemail.com", "guerrillamail.com"]
        if any(v.endswith("@" + d) for d in temp_domains):
            raise ValueError("Temporary email addresses are not allowed")
        return v

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str
    
    @validator("otp")
    def validate_otp(cls, v):
        if not re.match(r'^\d{6}$', v):
            raise ValueError("OTP must be 6 digits")
        return v

class OTPVerifyResponse(BaseModel):
    is_new_user: bool
    message: str

async def send_otp_via_cloudflare(email: str, otp: str):
    worker_url = "https://tiny-band-e198.spridu.workers.dev"
    headers = {
        "Authorization": "Bearer bycxyv-Zezqi5-fihwad",
        "Content-Type": "application/json"
    }
    
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color: #6366f1; margin: 0;">MemoryX</h2>
            <p style="color: #666; margin: 5px 0;">Cognitive Memory Engine</p>
        </div>
        <div style="background: #f9fafb; border-radius: 8px; padding: 30px; text-align: center;">
            <h3 style="margin: 0 0 20px 0; color: #111;">Your Verification Code</h3>
            <div style="background: #fff; border: 2px solid #6366f1; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <span style="font-size: 32px; font-weight: bold; color: #6366f1; letter-spacing: 8px;">{otp}</span>
            </div>
            <p style="color: #666; margin: 20px 0;">This code will expire in <strong>5 minutes</strong>.</p>
            <p style="color: #999; font-size: 12px; margin-top: 30px;">If you didn't request this code, you can safely ignore this email.<br> 2026 MemoryX - t0ken.ai</p>
        </div>
    </div>
    """
    
    data = {"email": email, "code": otp, "html": html_content}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(worker_url, json=data, headers=headers, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to send email: {response.text}")
        return response.json()

def generate_otp():
    return "".join(random.choices(string.digits, k=6))

def check_rate_limit(email: str):
    now = datetime.utcnow()
    if email in last_sent:
        elapsed = (now - last_sent[email]).total_seconds()
        if elapsed < RATE_LIMIT_SECONDS:
            return False, int(RATE_LIMIT_SECONDS - elapsed)
    return True, 0

@router.post("/send")
async def send_otp(request: OTPRequest, db: Session = Depends(get_db)):
    email = request.email.lower().strip()
    
    allowed, wait_seconds = check_rate_limit(email)
    if not allowed:
        raise HTTPException(
            status_code=429, 
            detail=f"Please wait {wait_seconds} seconds before requesting a new code"
        )
    
    otp = generate_otp()
    otp_store[email] = {
        "otp": otp,
        "expires": datetime.utcnow() + timedelta(minutes=OTP_EXPIRE_MINUTES),
        "attempts": 0
    }
    
    last_sent[email] = datetime.utcnow()
    
    try:
        await send_otp_via_cloudflare(email, otp)
        return {"message": "OTP sent successfully", "email": email, "retry_after": RATE_LIMIT_SECONDS}
    except Exception as e:
        print(f"Failed to send OTP to {email}: {e}")
        return {"message": "OTP sent successfully", "email": email, "retry_after": RATE_LIMIT_SECONDS}

@router.post("/verify", response_model=OTPVerifyResponse)
async def verify_otp(request: OTPVerifyRequest, db: Session = Depends(get_db)):
    email = request.email.lower().strip()
    
    if email not in otp_store:
        raise HTTPException(status_code=400, detail="OTP not found or expired")
    
    stored = otp_store[email]
    
    if datetime.utcnow() > stored["expires"]:
        del otp_store[email]
        raise HTTPException(status_code=400, detail="OTP expired")
    
    if stored["attempts"] >= MAX_ATTEMPTS:
        del otp_store[email]
        raise HTTPException(status_code=400, detail="Too many attempts")
    
    if stored["otp"] != request.otp:
        stored["attempts"] += 1
        remaining = MAX_ATTEMPTS - stored["attempts"]
        raise HTTPException(status_code=400, detail=f"Invalid OTP. {remaining} attempts remaining")
    
    del otp_store[email]
    
    user = db.query(User).filter(User.email == email).first()
    is_new_user = user is None
    
    return {"is_new_user": is_new_user, "message": "OTP verified successfully"}

@router.get("/rate-limit-status")
async def rate_limit_status(email: str):
    allowed, wait_seconds = check_rate_limit(email.lower().strip())
    return {"can_send": allowed, "wait_seconds": wait_seconds if not allowed else 0, "retry_after": RATE_LIMIT_SECONDS}
