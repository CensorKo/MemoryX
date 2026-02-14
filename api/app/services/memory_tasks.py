"""
Memory Tasks - Direct implementation using MemoryService
不再使用 HTTP 代理模式
"""
from celery import shared_task
from typing import Optional, Dict, Any, List

# 配置 (直接初始化 MemoryService)
def _get_service_config():
    import os
    return {
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

@shared_task(bind=True, max_retries=3)
def process_memory(self, memory_data: dict, api_key: str):
    """Process memory creation (async via Celery)."""
    try:
        from app.services.memory_core.memory_service import MemoryService
        
        service = MemoryService(_get_service_config())
        
        result = service.add(
            memory_data["content"],
            user_id=memory_data.get("user_id"),
            project_id=memory_data.get("project_id", "default"),
            metadata=memory_data.get("metadata", {})
        )
        
        return result
        
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)

@shared_task(bind=True, max_retries=3)
def update_memory_task(self, memory_id: str, update_data: dict, api_key: str):
    """Process memory update (async via Celery)."""
    try:
        from app.services.memory_core.memory_service import MemoryService
        
        service = MemoryService(_get_service_config())
        
        result = service.update(memory_id, update_data)
        return result
        
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)

@shared_task
def search_memory(query_data: dict, api_key: str):
    """Search memories (sync - returns result directly)."""
    try:
        from app.services.memory_core.memory_service import MemoryService
        
        service = MemoryService(_get_service_config())
        
        results = service.search(
            query=query_data.get("query", ""),
            filters=query_data.get("filters", {}),
            limit=query_data.get("limit", 10)
        )
        
        return {"results": results}
        
    except Exception as exc:
        return {"error": str(exc), "results": []}

def get_user_memories(user_id: str, project_id: str = None, limit: int = 100, offset: int = 0):
    """Get user memories (direct function, not async)."""
    try:
        from app.services.memory_core.memory_service import MemoryService
        
        service = MemoryService(_get_service_config())
        
        filters = {"user_id": user_id}
        if project_id:
            filters["project_id"] = project_id
        
        memories = service.get_all(filters=filters, limit=limit)
        return memories
        
    except Exception as exc:
        return []

def get_memory_by_id(memory_id: str, user_id: str):
    """Get memory by ID (direct function, not async)."""
    try:
        from app.services.memory_core.memory_service import MemoryService
        
        service = MemoryService(_get_service_config())
        
        memory = service.get(memory_id)
        if memory and memory.get("user_id") == user_id:
            return memory
        return None
        
    except Exception as exc:
        return None

def delete_memory(memory_id: str, user_id: str):
    """Delete memory (direct function, not async)."""
    try:
        from app.services.memory_core.memory_service import MemoryService
        
        service = MemoryService(_get_service_config())
        
        # Check ownership
        memory = service.get(memory_id)
        if not memory or memory.get("user_id") != user_id:
            return False
        
        return service.delete(memory_id)
        
    except Exception as exc:
        return False
