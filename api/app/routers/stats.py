from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import hashlib

from app.core.database import get_db, APIKey
from app.core.database import Project

router = APIRouter(prefix="/stats", tags=["stats"])

def get_current_user_api(x_api_key: str = Header(None), db: Session = Depends(get_db)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash, APIKey.is_active == True).first()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return api_key.user_id, api_key.user

@router.get("", response_model=dict)
async def get_stats(
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    user_id, user = user_data
    
    # Get project count
    project_count = db.query(Project).filter(Project.user_id == user_id).count()
    
    # Get API key count
    api_key_count = db.query(APIKey).filter(APIKey.user_id == user_id).count()
    
    # Calculate account age
    account_created = user.created_at.strftime("%Y-%m-%d") if user.created_at else "2026-02-13"
    
    # Import memory stats from service
    from app.services.memory_tasks import get_user_memories
    memories = get_user_memories(str(user_id), limit=10000)
    
    # Handle different response formats
    if isinstance(memories, dict):
        memory_list = memories.get("results", memories.get("data", []))
    else:
        memory_list = memories if isinstance(memories, list) else []
    
    total_memories = len(memory_list)
    
    return {
        "success": True,
        "data": {
            "total_memories": total_memories,
            "total_projects": project_count,
            "total_api_keys": api_key_count,
            "api_calls_today": 0,
            "api_calls_this_month": 0,
            "storage_used": f"{total_memories * 10} KB",
            "account_created": account_created
        }
    }

@router.get("/usage", response_model=dict)
async def get_usage_stats(
    days: int = 30,
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    user_id, user = user_data
    
    # Generate daily stats for the last N days
    daily = []
    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        daily.append({
            "date": date,
            "calls": 0,
            "memories_created": 0
        })
    
    daily.reverse()
    
    # Get memories and analyze cognitive sectors
    from app.services.memory_tasks import get_user_memories
    memories = get_user_memories(str(user_id), limit=10000)
    
    # Handle different response formats
    if isinstance(memories, dict):
        memory_list = memories.get("results", memories.get("data", []))
    else:
        memory_list = memories if isinstance(memories, list) else []
    
    sectors = {"episodic": 0, "semantic": 0, "procedural": 0, "emotional": 0, "reflective": 0, "other": 0}
    for memory in memory_list:
        if isinstance(memory, dict):
            sector = memory.get("cognitive_sector") or memory.get("sector_primary", "other")
            if sector in sectors:
                sectors[sector] += 1
            else:
                sectors["other"] += 1
    
    # Get top projects
    projects = db.query(Project).filter(Project.user_id == user_id).all()
    top_projects = [
        {"id": p.id, "name": p.name, "memory_count": 0}
        for p in projects[:5]
    ]
    
    return {
        "success": True,
        "data": {
            "daily": daily,
            "cognitive_sectors": sectors,
            "top_projects": top_projects
        }
    }