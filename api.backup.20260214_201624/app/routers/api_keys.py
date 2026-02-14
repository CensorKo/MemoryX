from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import hashlib
import secrets
from app.core.database import get_db
from app.routers.auth import get_current_user
from app.core.database import User, APIKey

router = APIRouter(prefix="/api-keys", tags=["api-keys"])

class APIKeyCreate(BaseModel):
    name: str = "New Key"

class APIKeyResponse(BaseModel):
    id: int
    name: str
    is_active: bool
    created_at: str
    key: str = None  # Only shown on creation
    
    class Config:
        from_attributes = True

@router.get("", response_model=List[APIKeyResponse])
def list_keys(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()
    return [
        {
            "id": k.id,
            "name": k.name,
            "is_active": k.is_active,
            "created_at": k.created_at.isoformat() if k.created_at else None
        }
        for k in keys
    ]

@router.post("", response_model=APIKeyResponse)
def create_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    api_key = "omx_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    db_key = APIKey(
        user_id=current_user.id,
        key_hash=key_hash,
        name=key_data.name
    )
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    
    return {
        "id": db_key.id,
        "name": db_key.name,
        "is_active": db_key.is_active,
        "created_at": db_key.created_at.isoformat() if db_key.created_at else None,
        "key": api_key  # Only shown once
    }

@router.delete("/{key_id}")
def delete_key(key_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    key = db.query(APIKey).filter(APIKey.id == key_id, APIKey.user_id == current_user.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API Key not found")
    
    db.delete(key)
    db.commit()
    return {"message": "API Key deleted"}

@router.get("/{key_id}/cursor-config")
def get_cursor_config(key_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    key = db.query(APIKey).filter(APIKey.id == key_id, APIKey.user_id == current_user.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API Key not found")
    
    return {
        "mcpServers": {
            "openmemoryx": {
                "command": "python3",
                "args": ["-m", "openmemoryx_mcp"],
                "env": {
                    "OPENMEMORYX_API_KEY": "YOUR_API_KEY",
                    "OPENMEMORYX_URL": "http://192.168.31.65:8000"
                }
            }
        }
    }
