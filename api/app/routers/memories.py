"""
Memory Router - Queue-based processing for all write operations
所有写操作走队列处理
"""
from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import hashlib
import os

from app.core.database import get_db, User, APIKey, Project
from app.core.security import verify_token
from app.core.config import get_settings
from app.services.memory_core.memory_service import MemoryService

# Import Celery tasks for queue processing
from app.services.memory_tasks import (
    process_memory,
    update_memory_task,
    search_memory as search_memory_task
)

router = APIRouter(prefix="/v1", tags=["memories"])

# Global memory service instance for read operations
_memory_service: Optional[MemoryService] = None

def get_memory_service() -> MemoryService:
    """Get or create MemoryService instance."""
    global _memory_service
    if _memory_service is None:
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": os.getenv("QDRANT_COLLECTION", "mem0"),
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                    "embedding_model_dims": 1024
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": os.getenv("LLM_MODEL", "gemma3-27b-q8"),
                    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": os.getenv("EMBED_MODEL", "bge-m3"),
                    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    "embedding_dims": 1024
                }
            }
        }
        _memory_service = MemoryService(config)
    return _memory_service

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
    """Create a new memory with AI classification (queue-based processing).
    
    All memory creation is processed asynchronously via Celery queue.
    Use GET /api/v1/tasks/{task_id} to check processing status.
    """
    user_id, api_key = user_data
    
    memory_data = {
        "user_id": str(user_id),
        "content": memory.content,
        "project_id": memory.project_id or "default",
        "metadata": memory.metadata or {},
        "created_at": datetime.utcnow().isoformat()
    }
    
    # 队列处理（所有创建操作）
    task = process_memory.delay(memory_data, api_key)
    
    return {
        "success": True,
        "message": "Memory queued for processing",
        "task_id": task.id,
        "status": "pending",
        "note": "Use GET /api/v1/tasks/{task_id} to check status"
    }

@router.get("/memories", response_model=dict)
async def list_memories(
    project_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    """List memories for a user."""
    user_id, api_key = user_data
    
    service = get_memory_service()
    
    filters = {"user_id": str(user_id)}
    if project_id:
        filters["project_id"] = project_id
    
    memories = service.get_all(filters=filters, limit=limit)
    
    return {
        "success": True,
        "data": memories,
        "total": len(memories)
    }

@router.get("/memories/{memory_id}", response_model=dict)
async def get_memory(
    memory_id: str,
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    """Get a specific memory by ID."""
    user_id, api_key = user_data
    
    service = get_memory_service()
    memory = service.get(memory_id)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Check ownership
    if memory.get("user_id") != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "success": True,
        "data": memory
    }

@router.put("/memories/{memory_id}", response_model=dict)
async def update_memory(
    memory_id: str,
    update: MemoryUpdate,
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    """Update a memory (queue-based processing).
    
    All updates are processed asynchronously via Celery queue.
    """
    user_id, api_key = user_data
    
    # Check existing memory
    service = get_memory_service()
    existing = service.get(memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    if existing.get("user_id") != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    update_data = {}
    if update.content is not None:
        update_data["content"] = update.content
    if update.metadata is not None:
        update_data["metadata"] = update.metadata
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    update_data["updated_at"] = datetime.utcnow().isoformat()
    
    # 队列处理（所有更新操作）
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
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    """Delete a memory (sync processing - usually fast)."""
    user_id, api_key = user_data
    
    service = get_memory_service()
    
    # Check existing memory
    existing = service.get(memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    if existing.get("user_id") != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = service.delete(memory_id)
    
    return {
        "success": success,
        "message": "Memory deleted successfully" if success else "Failed to delete memory"
    }

@router.post("/memories/search", response_model=dict)
async def search_memories(
    query: SearchQuery,
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    """Search memories using vector similarity + filters (sync processing).
    
    Search is synchronous to provide immediate results to users.
    """
    user_id, api_key = user_data
    
    service = get_memory_service()
    
    filters = {"user_id": str(user_id)}
    if query.project_id:
        filters["project_id"] = query.project_id
    
    results = service.search(
        query=query.query,
        filters=filters,
        limit=query.limit or 10
    )
    
    return {
        "success": True,
        "data": results,
        "query": query.query
    }

@router.get("/tasks/{task_id}", response_model=dict)
async def get_task_status(task_id: str):
    """Get Celery task status (for async operations)."""
    from app.core.celery_config import celery_app
    
    task = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task.status,
        "ready": task.ready()
    }
    
    if task.ready():
        if task.successful():
            response["result"] = task.result
        else:
            response["error"] = str(task.result)
    
    return response
