from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
import httpx
import json

from app.core.database import (
    get_db, User, APIKey, UserQuota, 
    get_or_create_quota, SubscriptionTier, QUOTA_LIMITS
)
from app.services.memory_core.graph_memory_service import graph_memory_service
from app.services.memory_queue import add_memory_task
from app.core.celery_config import celery_app
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/v1", tags=["conversations"])


class MessageItem(BaseModel):
    role: str
    content: str
    tokens: Optional[int] = 0
    timestamp: Optional[int] = None


class ConversationFlushRequest(BaseModel):
    conversation_id: Optional[str] = None
    messages: List[MessageItem]
    total_tokens: Optional[int] = 0
    client_model_version: Optional[int] = 1


class SensitiveFilterResult(BaseModel):
    has_sensitive: bool
    filtered_content: str
    sensitive_count: int


SENSITIVE_FILTER_PROMPT = """分析以下对话，识别并标记敏感信息：
- 个人身份信息（姓名、身份证、电话、地址）
- 财务信息（银行卡、密码、金额）
- 健康信息（病历、诊断）
- 位置信息（精确地址、GPS）

对话内容：
{conversation}

请严格按以下 JSON 格式返回，不要包含其他内容：
{{"has_sensitive": true或false, "sensitive_spans": [{{"start": 起始位置, "end": 结束位置, "type": "类型", "replacement": "[已过滤]"}}], "safe_content": "过滤后的安全内容"}}"""


async def filter_sensitive_with_llm(content: str) -> SensitiveFilterResult:
    """使用 LLM 过滤敏感信息 - OpenAI 兼容 API"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": settings.llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个敏感信息识别助手。请分析对话内容，识别并替换所有敏感信息。只返回JSON格式结果。"
                        },
                        {
                            "role": "user",
                            "content": SENSITIVE_FILTER_PROMPT.format(conversation=content)
                        }
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM filter failed: {response.status_code} - {response.text}")
                return SensitiveFilterResult(
                    has_sensitive=False,
                    filtered_content=content,
                    sensitive_count=0
                )
            
            data = response.json()
            llm_response = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            result = json.loads(llm_response)
            
            return SensitiveFilterResult(
                has_sensitive=result.get("has_sensitive", False),
                filtered_content=result.get("safe_content", content),
                sensitive_count=len(result.get("sensitive_spans", []))
            )
            
    except json.JSONDecodeError as e:
        logger.error(f"LLM response parse failed: {e}")
        return SensitiveFilterResult(
            has_sensitive=False,
            filtered_content=content,
            sensitive_count=0
        )
    except Exception as e:
        logger.error(f"LLM filter error: {e}")
        return SensitiveFilterResult(
            has_sensitive=False,
            filtered_content=content,
            sensitive_count=0
        )


def process_conversation_task(conversation_data: dict):
    """后台任务：处理对话提取记忆 - 通过队列异步处理"""
    try:
        user_id = conversation_data["user_id"]
        messages = conversation_data["messages"]
        
        combined_content = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in messages
        ])
        
        import asyncio
        filter_result = asyncio.run(filter_sensitive_with_llm(combined_content))
        
        if filter_result.has_sensitive:
            logger.info(f"LLM filtered {filter_result.sensitive_count} sensitive items for user {user_id}")
        
        task = add_memory_task.delay(
            user_id=user_id,
            content=filter_result.filtered_content,
            metadata={
                "conversation_id": conversation_data.get("conversation_id"),
                "message_count": len(messages),
                "total_tokens": conversation_data.get("total_tokens", 0),
                "has_sensitive": filter_result.has_sensitive,
                "source": "conversation_flush",
                "project_id": conversation_data.get("project_id", "default")
            }
        )
        
        logger.info(f"Queued memory task for user {user_id}: {task.id}")
        return {"task_id": task.id, "status": "queued"}
        
    except Exception as e:
        logger.error(f"Failed to process conversation: {e}")
        raise


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


@router.post("/conversations/flush")
async def flush_conversation(
    request: ConversationFlushRequest,
    background_tasks: BackgroundTasks,
    user_data: tuple = Depends(get_current_user_with_quota),
    db: Session = Depends(get_db)
):
    """
    批量提交对话 - 触发记忆提取
    
    流程：
    1. 接收缓冲的对话消息
    2. 敏感信息过滤
    3. 后台异步处理：LLM 提取记忆 + 向量化 + 图构建
    """
    user_id, tier, quota, api_key = user_data
    
    if not request.messages or len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    can_create, remaining = quota.can_create_memory(tier)
    if not can_create:
        raise HTTPException(
            status_code=402,
            detail=f"Monthly memory limit reached. Please upgrade to Pro."
        )
    
    conversation_data = {
        "user_id": str(user_id),
        "conversation_id": request.conversation_id,
        "messages": [m.model_dump() for m in request.messages],
        "total_tokens": request.total_tokens or 0,
        "project_id": "default"
    }
    
    background_tasks.add_task(process_conversation_task, conversation_data)
    
    quota.increment_memories_created()
    db.commit()
    
    return {
        "status": "ok",
        "message": "Conversation queued for processing",
        "message_count": len(request.messages),
        "extracted_count": 1,
        "server_model_version": 1,
        "sync_required": False,
        "remaining_quota": remaining - 1
    }


@router.post("/conversations/realtime")
async def realtime_message(
    message: MessageItem,
    user_data: tuple = Depends(get_current_user_with_quota),
    db: Session = Depends(get_db)
):
    """
    实时消息接收 - 立即处理
    
    用于高优先级消息，绕过缓冲直接处理
    """
    user_id, tier, quota, api_key = user_data
    
    if not message.content or len(message.content) < 2:
        return {"status": "skipped", "reason": "content_too_short"}
    
    can_create, remaining = quota.can_create_memory(tier)
    if not can_create:
        raise HTTPException(
            status_code=402,
            detail="Monthly memory limit reached."
        )
    
    filter_result = await filter_sensitive_with_llm(message.content)
    
    task = add_memory_task.delay(
        user_id=str(user_id),
        content=filter_result.filtered_content,
        metadata={
            "role": message.role,
            "tokens": message.tokens,
            "source": "realtime",
            "has_sensitive": filter_result.has_sensitive
        }
    )
    
    quota.increment_memories_created()
    db.commit()
    
    return {
        "status": "queued",
        "task_id": task.id,
        "has_sensitive": filter_result.has_sensitive,
        "remaining_quota": remaining - 1
    }
