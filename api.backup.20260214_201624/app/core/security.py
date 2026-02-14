import hashlib
import hmac
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional
from app.core.config import get_settings

settings = get_settings()

SECRET_KEY = settings.secret_key.encode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using SHA256 HMAC"""
    expected_hash = hashlib.sha256((plain_password + settings.secret_key).encode()).hexdigest()
    return hmac.compare_digest(expected_hash, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256((password + settings.secret_key).encode()).hexdigest()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError:
        return None
