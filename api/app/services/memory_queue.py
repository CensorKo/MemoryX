"""
Memory Queue Tasks - Celery 异步任务

所有记忆操作（添加/删除/修改）通过队列异步处理，防止 LLM 被打爆。
搜索操作保持同步，保证响应速度。
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from celery import shared_task

from app.core.celery_config import celery_app
from app.services.memory_core.graph_memory_service import graph_memory_service

logger = logging.getLogger(__name__)


def run_async(coro):
    """在同步任务中运行异步函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name="memory.add",
    bind=True,
    max_retries=3,
    default_retry_delay=10
)
def add_memory_task(
    self,
    user_id: str,
    content: str,
    metadata: Dict = None,
    skip_judge: bool = False
) -> Dict[str, Any]:
    """
    异步添加记忆任务
    
    Args:
        user_id: 用户ID
        content: 记忆内容
        metadata: 元数据
        skip_judge: 是否跳过LLM判断
        
    Returns:
        处理结果
    """
    try:
        logger.info(f"[Queue] Processing add_memory for user {user_id}")
        
        result = run_async(
            graph_memory_service.add_memory(
                user_id=user_id,
                content=content,
                metadata=metadata,
                skip_judge=skip_judge
            )
        )
        
        logger.info(f"[Queue] Add memory completed: {result.get('stats', {})}")
        return result
        
    except Exception as e:
        logger.error(f"[Queue] Add memory failed: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    name="memory.batch_add",
    bind=True,
    max_retries=3,
    default_retry_delay=10
)
def batch_add_memory_task(
    self,
    user_id: str,
    contents: List[str],
    metadatas: List[Dict] = None
) -> List[Dict[str, Any]]:
    """
    异步批量添加记忆任务
    
    Args:
        user_id: 用户ID
        contents: 记忆内容列表
        metadatas: 元数据列表
        
    Returns:
        处理结果列表
    """
    try:
        logger.info(f"[Queue] Processing batch_add for user {user_id}, count: {len(contents)}")
        
        results = []
        for i, content in enumerate(contents):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else None
            
            result = run_async(
                graph_memory_service.add_memory(
                    user_id=user_id,
                    content=content,
                    metadata=metadata
                )
            )
            results.append(result)
        
        logger.info(f"[Queue] Batch add completed: {len(results)} items")
        return results
        
    except Exception as e:
        logger.error(f"[Queue] Batch add failed: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    name="memory.update",
    bind=True,
    max_retries=3,
    default_retry_delay=10
)
def update_memory_task(
    self,
    user_id: str,
    content: str,
    metadata: Dict = None
) -> Dict[str, Any]:
    """
    异步更新记忆任务（通过添加新内容触发LLM判断更新）
    
    Args:
        user_id: 用户ID
        content: 新内容
        metadata: 元数据
        
    Returns:
        处理结果
    """
    try:
        logger.info(f"[Queue] Processing update_memory for user {user_id}")
        
        result = run_async(
            graph_memory_service.add_memory(
                user_id=user_id,
                content=content,
                metadata=metadata
            )
        )
        
        logger.info(f"[Queue] Update memory completed: {result.get('stats', {})}")
        return result
        
    except Exception as e:
        logger.error(f"[Queue] Update memory failed: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    name="memory.delete",
    bind=True,
    max_retries=3,
    default_retry_delay=10
)
def delete_memory_task(
    self,
    user_id: str,
    content: str,
    metadata: Dict = None
) -> Dict[str, Any]:
    """
    异步删除记忆任务（通过添加矛盾内容触发LLM判断删除）
    
    Args:
        user_id: 用户ID
        content: 矛盾内容
        metadata: 元数据
        
    Returns:
        处理结果
    """
    try:
        logger.info(f"[Queue] Processing delete_memory for user {user_id}")
        
        result = run_async(
            graph_memory_service.add_memory(
                user_id=user_id,
                content=content,
                metadata=metadata
            )
        )
        
        logger.info(f"[Queue] Delete memory completed: {result.get('stats', {})}")
        return result
        
    except Exception as e:
        logger.error(f"[Queue] Delete memory failed: {e}")
        raise self.retry(exc=e)
