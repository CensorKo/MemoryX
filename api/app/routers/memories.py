"""
Memory Router - Direct implementation with MemoryService
整合旧架构的核心记忆功能
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
from typing import Optional

router = APIRouter(prefix="/v1", tags=["memories"])

# Global memory service instance
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
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    """Create a new memory with AI classification."""
    user_id, api_key = user_data
    
    service = get_memory_service()
    
    # Add user_id and project_id to metadata
    metadata = memory.metadata or {}
    metadata["user_id"] = str(user_id)
    metadata["project_id"] = memory.project_id or "default"
    
    result = service.add(
        memory.content,
        user_id=str(user_id),
        project_id=memory.project_id or "default",
        metadata=metadata
    )
    
    return {
        "success": True,
        "message": "Memory created successfully",
        "data": result
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
    """Update a memory."""
    user_id, api_key = user_data
    
    service = get_memory_service()
    
    # Check existing memory
    existing = service.get(memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    if existing.get("user_id") != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    updates = {}
    if update.content is not None:
        updates["content"] = update.content
    if update.metadata is not None:
        updates["metadata"] = update.metadata
    
    result = service.update(memory_id, updates)
    
    return {
        "success": True,
        "message": "Memory updated successfully",
        "data": result
    }

@router.delete("/memories/{memory_id}", response_model=dict)
async def delete_memory(
    memory_id: str,
    user_data: tuple = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    """Delete a memory."""
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
    """Search memories using vector similarity + filters."""
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
