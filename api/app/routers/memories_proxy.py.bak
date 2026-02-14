from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import hashlib

from app.core.database import get_db, User, APIKey
from app.core.security import verify_token
from app.services.memory_tasks import (
    process_memory, search_memory, get_user_memories, 
    get_memory_by_id, delete_memory as delete_memory_service,
    update_memory_task
)
from app.core.celery_config import celery_app

router = APIRouter(prefix="/v1", tags=["memories"])

# Schemas
class MemoryCreate(BaseModel):
    content: str
    project_id: Optional[str] = "default"
    metadata: Optional[dict] = {}

class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    metadata: Optional[dict] = None

class SearchQuery(BaseModel):
    query: str
    project_id: Optional[str] = None
    limit: Optional[int] = 10

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None

# Auth helper
def get_current_user_api(x_api_key: str = Header(None), db: Session = Depends(get_db)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash, APIKey.is_active == True).first()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return api_key.user_id, api_key.key_hash

@router.post("/memories", response_model=dict)
async def create_memory(
    memory: MemoryCreate,
    background_tasks: BackgroundTasks,
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    user_id, api_key = user_data
    
    memory_data = {
        "user_id": str(user_id),
        "content": memory.content,
        "project_id": memory.project_id or "default",
        "metadata": memory.metadata or {},
        "created_at": datetime.utcnow().isoformat()
    }
    
    task = process_memory.delay(memory_data, api_key)
    
    return {
        "success": True,
        "message": "Memory queued for processing",
        "task_id": task.id,
        "status": "pending",
        "note": "Processing may take a few seconds due to AI classification"
    }

@router.get("/memories", response_model=dict)
async def list_memories(
    project_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    user_data: tuple = Depends(get_current_user_api)
):
    user_id, api_key = user_data
    
    result = get_user_memories(user_id, project_id, limit, offset)
    
    # Handle different response formats from MemoryX
    if isinstance(result, dict):
        memories = result.get("results", result.get("data", []))
        total = result.get("count", result.get("total", len(memories)))
    else:
        memories = result
        total = len(result)
    
    return {
        "success": True,
        "data": memories,
        "total": total
    }

@router.get("/memories/{memory_id}", response_model=dict)
async def get_memory(
    memory_id: str,
    user_data: tuple = Depends(get_current_user_api)
):
    user_id, api_key = user_data
    
    memory = get_memory_by_id(memory_id, user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {
        "success": True,
        "data": memory
    }

@router.put("/memories/{memory_id}", response_model=dict)
async def update_memory(
    memory_id: str,
    update: MemoryUpdate,
    user_data: tuple = Depends(get_current_user_api)
):
    user_id, api_key = user_data
    
    update_data = {}
    if update.content is not None:
        update_data["content"] = update.content
    if update.metadata is not None:
        update_data["metadata"] = update.metadata
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    update_data["updated_at"] = datetime.utcnow().isoformat()
    
    task = update_memory_task.delay(memory_id, update_data, api_key)
    
    return {
        "success": True,
        "message": "Memory update queued for processing",
        "task_id": task.id,
        "status": "pending"
    }

@router.delete("/memories/{memory_id}", response_model=dict)
async def delete_memory(
    memory_id: str,
    user_data: tuple = Depends(get_current_user_api)
):
    user_id, api_key = user_data
    
    success = delete_memory_service(memory_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {
        "success": True,
        "message": "Memory deleted successfully"
    }

@router.post("/memories/search", response_model=dict)
async def search_memories(
    query: SearchQuery,
    user_data: tuple = Depends(get_current_user_api)
):
    user_id, api_key = user_data
    
    search_data = {
        "user_id": str(user_id),
        "query": query.query,
        "project_id": query.project_id,
        "limit": query.limit or 10
    }
    
    result = search_memory.delay(search_data, api_key)
    response = result.get(timeout=30)
    
    return {
        "success": True,
        "data": response.get("results", []),
        "query": query.query
    }

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }