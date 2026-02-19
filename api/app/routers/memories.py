from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import hashlib
import logging
import asyncio

from app.core.database import (
    get_db, User, APIKey, UserQuota, 
    get_or_create_quota, SubscriptionTier, QUOTA_LIMITS, PRICING
)
from app.services.memory_core.graph_memory_service import graph_memory_service
from app.services.memory_queue import (
    add_memory_task,
    batch_add_memory_task
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["memories"])


class MemoryCreate(BaseModel):
    content: str
    project_id: Optional[str] = "default"
    metadata: Optional[dict] = {}


class MemoryBatchItem(BaseModel):
    content: str
    metadata: Optional[dict] = {}


class MemoryBatchCreate(BaseModel):
    memories: List[MemoryBatchItem]
    project_id: Optional[str] = "default"


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


class QuotaInfo(BaseModel):
    tier: str
    cloud_search_used: int
    cloud_search_limit: int
    memories_created: int
    memories_limit: int
    remaining_cloud_search: int
    remaining_memories: int


def get_current_user_with_quota(
    x_api_key: str = Header(None), 
    db: Session = Depends(get_db)
) -> tuple:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    
    api_key = db.query(APIKey).filter(
        APIKey.key_hash == x_api_key, 
        APIKey.is_active == True
    ).first()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    user = db.query(User).filter(User.id == api_key.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    quota = get_or_create_quota(db, user.id)
    
    return user.id, user.subscription_tier, quota, x_api_key


@router.post("/memories", response_model=dict)
async def create_memory(
    memory: MemoryCreate,
    user_data: tuple = Depends(get_current_user_with_quota),
    db: Session = Depends(get_db)
):
    """
    添加记忆 - 异步队列处理
    
    记忆添加操作通过 Celery 队列异步处理，防止 LLM 被打爆。
    返回 task_id 可用于查询处理状态。
    """
    user_id, tier, quota, api_key = user_data
    
    can_create, remaining = quota.can_create_memory(tier)
    if not can_create:
        raise HTTPException(
            status_code=402,
            detail=f"Monthly memory limit reached ({QUOTA_LIMITS[tier]['memories_per_month']}). Please upgrade to Pro for unlimited memories."
        )
    
    metadata = memory.metadata or {}
    metadata["project_id"] = memory.project_id
    
    task = add_memory_task.delay(
        user_id=str(user_id),
        content=memory.content,
        metadata=metadata
    )
    
    quota.increment_memories_created()
    db.commit()
    
    return {
        "success": True,
        "message": "Memory task queued for processing",
        "task_id": task.id,
        "status": "pending",
        "remaining_quota": remaining - 1
    }


@router.post("/memories/batch", response_model=dict)
async def batch_create_memories(
    batch: MemoryBatchCreate,
    user_data: tuple = Depends(get_current_user_with_quota),
    db: Session = Depends(get_db)
):
    """
    批量添加记忆 - 异步队列处理
    
    批量记忆添加操作通过 Celery 队列异步处理。
    返回 task_id 可用于查询处理状态。
    """
    user_id, tier, quota, api_key = user_data
    
    if len(batch.memories) == 0:
        raise HTTPException(status_code=400, detail="No memories provided")
    
    if len(batch.memories) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 memories per batch")
    
    can_batch, remaining = quota.can_batch_upload(tier)
    if not can_batch:
        raise HTTPException(
            status_code=402,
            detail=f"Daily batch upload limit reached ({QUOTA_LIMITS[tier]['batch_upload_per_day']}). Please upgrade to Pro."
        )
    
    can_create, mem_remaining = quota.can_create_memory(tier)
    if not can_create and tier == SubscriptionTier.FREE:
        allowed = min(len(batch.memories), mem_remaining)
        if allowed == 0:
            raise HTTPException(
                status_code=402,
                detail=f"Monthly memory limit reached. Please upgrade to Pro."
            )
        batch.memories = batch.memories[:allowed]
    
    contents = [item.content for item in batch.memories]
    metadatas = [item.metadata or {} for item in batch.memories]
    
    for i, m in enumerate(metadatas):
        m["project_id"] = batch.project_id
    
    task = batch_add_memory_task.delay(
        user_id=str(user_id),
        contents=contents,
        metadatas=metadatas
    )
    
    quota.increment_memories_created(len(batch.memories))
    quota.increment_batch_uploads()
    db.commit()
    
    return {
        "success": True,
        "message": f"Batch of {len(batch.memories)} memories queued for processing",
        "task_id": task.id,
        "status": "pending",
        "queued_count": len(batch.memories),
        "remaining_quota": remaining - len(batch.memories)
    }


@router.get("/memories/task/{task_id}", response_model=dict)
async def get_task_status(
    task_id: str,
    user_data: tuple = Depends(get_current_user_with_quota)
):
    """
    查询任务状态
    
    用于查询异步任务（添加/更新/删除记忆）的处理状态。
    """
    from celery.result import AsyncResult
    
    task_result = AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.result
        else:
            response["error"] = str(task_result.result)
    
    return response


@router.post("/memories/search", response_model=dict)
async def search_memories(
    query: SearchQuery,
    user_data: tuple = Depends(get_current_user_with_quota),
    db: Session = Depends(get_db)
):
    user_id, tier, quota, api_key = user_data
    
    can_search, remaining = quota.can_cloud_search(tier)
    if not can_search:
        raise HTTPException(
            status_code=402,
            detail=f"Cloud search quota exhausted ({QUOTA_LIMITS[tier]['cloud_search_per_month']}/month). Upgrade to Pro for unlimited searches."
        )
    
    try:
        context = await graph_memory_service.get_context_for_query(
            user_id=str(user_id),
            query=query.query,
            limit=query.limit or 10
        )
        
        quota.increment_cloud_search()
        db.commit()
        
        new_remaining = remaining - 1 if remaining > 0 else -1
        
        return {
            "success": True,
            "data": context.get("vector_memories", []),
            "graph_context": context.get("graph_entities", []),
            "extracted_entities": context.get("extracted_entities", []),
            "query": query.query,
            "remaining_quota": new_remaining,
            "tier": tier.value
        }
    except Exception as e:
        logger.error(f"Search memories failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search memories: {str(e)}"
        )


@router.post("/memories/graph/search", response_model=dict)
async def search_graph(
    query: SearchQuery,
    user_data: tuple = Depends(get_current_user_with_quota),
    db: Session = Depends(get_db)
):
    user_id, tier, quota, api_key = user_data
    
    can_search, remaining = quota.can_cloud_search(tier)
    if not can_search:
        raise HTTPException(
            status_code=402,
            detail="Cloud search quota exhausted"
        )
    
    try:
        context = await graph_memory_service.get_context_for_query(
            user_id=str(user_id),
            query=query.query,
            limit=query.limit or 10
        )
        
        quota.increment_cloud_search()
        db.commit()
        
        return {
            "success": True,
            "vector_results": context.get("vector_memories", []),
            "graph_results": context.get("graph_entities", []),
            "entities": context.get("extracted_entities", []),
            "query": query.query
        }
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search graph: {str(e)}"
        )


@router.get("/quota", response_model=dict)
async def get_quota_info(
    user_data: tuple = Depends(get_current_user_with_quota)
):
    user_id, tier, quota, api_key = user_data
    
    limits = QUOTA_LIMITS[tier]
    
    _, search_remaining = quota.can_cloud_search(tier)
    _, mem_remaining = quota.can_create_memory(tier)
    _, batch_remaining = quota.can_batch_upload(tier)
    
    return {
        "success": True,
        "quota": {
            "tier": tier.value,
            "price": PRICING[tier],
            "cloud_search": {
                "used": quota.cloud_search_used,
                "limit": limits["cloud_search_per_month"],
                "remaining": search_remaining
            },
            "memories": {
                "used": quota.memories_created,
                "limit": limits["memories_per_month"],
                "remaining": mem_remaining
            },
            "batch_upload": {
                "used": quota.batch_uploads_today,
                "limit": limits["batch_upload_per_day"],
                "remaining": batch_remaining
            }
        }
    }
