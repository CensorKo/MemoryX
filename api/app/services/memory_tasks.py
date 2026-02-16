"""
Memory Tasks Module - Celery tasks for memory processing
记忆任务模块 - 用于记忆处理的Celery任务
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from celery import shared_task

from app.services.memory_service import (
    MemoryService, 
    Memory,
    init_memory_service,
    get_memory_service
)
from app.services.classification import quick_classify, ClassificationResult
from app.services.scoring import ScoringFactors, calculate_memory_score

logger = logging.getLogger(__name__)

# 全局记忆服务实例（将在任务中延迟初始化）
_memory_service: Optional[MemoryService] = None


def _get_memory_service() -> MemoryService:
    """获取或初始化记忆服务"""
    global _memory_service
    if _memory_service is None:
        _memory_service = init_memory_service()
    return _memory_service


@shared_task(bind=True, max_retries=3)
def process_memory(self, memory_data: dict, api_key: str = None):
    """
    处理新记忆 - 分类、嵌入、存储
    
    Args:
        memory_data: 记忆数据字典
            - user_id: 用户ID
            - content: 记忆内容
            - project_id: 项目ID（可选）
            - metadata: 元数据（可选）
        api_key: API密钥（用于权限验证，已在外层验证）
    
    Returns:
        dict: 处理结果
    """
    try:
        logger.info(f"Processing memory for user {memory_data.get('user_id')}")
        
        service = _get_memory_service()
        
        # 提取数据
        user_id = memory_data.get("user_id")
        content = memory_data.get("content")
        project_id = memory_data.get("project_id", "default")
        metadata = memory_data.get("metadata", {})
        
        if not user_id or not content:
            return {
                "success": False,
                "error": "Missing required fields: user_id and content"
            }
        
        # 使用同步方法创建记忆（Celery任务中使用同步）
        memory = service.create_memory_sync(
            user_id=user_id,
            content=content,
            project_id=project_id,
            metadata=metadata
        )
        
        logger.info(f"Memory created: {memory.id}")
        
        return {
            "success": True,
            "memory_id": memory.id,
            "category": memory.category,
            "importance": memory.importance,
            "tags": memory.tags,
            "summary": memory.summary,
            "score": memory.score,
            "created_at": memory.created_at
        }
        
    except Exception as exc:
        logger.error(f"Failed to process memory: {exc}", exc_info=True)
        # 重试
        raise self.retry(exc=exc, countdown=10)


@shared_task(bind=True, max_retries=3)
def update_memory_task(self, memory_id: str, update_data: dict, api_key: str = None):
    """
    更新记忆任务
    
    Args:
        memory_id: 记忆ID
        update_data: 更新数据
        api_key: API密钥
    
    Returns:
        dict: 更新结果
    """
    try:
        logger.info(f"Updating memory {memory_id}")
        
        service = _get_memory_service()
        
        # 注意：需要从某处获取user_id，这里使用update_data中的或从memory获取
        # 实际上应该在外层验证后传入
        user_id = update_data.get("user_id")
        
        if not user_id:
            # 尝试先获取记忆
            memory = service.get_memory(memory_id, "")
            if not memory:
                return {
                    "success": False,
                    "error": "Memory not found"
                }
            user_id = memory.user_id
        
        # 过滤掉不需要的字段
        allowed_fields = {"content", "metadata", "project_id"}
        updates = {k: v for k, v in update_data.items() if k in allowed_fields}
        
        updated_memory = service.update_memory(memory_id, user_id, updates)
        
        if not updated_memory:
            return {
                "success": False,
                "error": "Memory not found or update failed"
            }
        
        logger.info(f"Memory updated: {memory_id}")
        
        return {
            "success": True,
            "memory_id": updated_memory.id,
            "updated_at": updated_memory.updated_at
        }
        
    except Exception as exc:
        logger.error(f"Failed to update memory: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=10)


@shared_task(bind=True, max_retries=3)
def search_memory(self, query_data: dict, api_key: str = None):
    """
    搜索记忆任务
    
    Args:
        query_data: 查询数据
            - user_id: 用户ID
            - query: 查询字符串
            - project_id: 项目ID（可选）
            - limit: 结果数量限制（可选）
        api_key: API密钥
    
    Returns:
        dict: 搜索结果
    """
    try:
        import asyncio
        
        logger.info(f"Searching memories for user {query_data.get('user_id')}")
        
        service = _get_memory_service()
        
        user_id = query_data.get("user_id")
        query = query_data.get("query")
        project_id = query_data.get("project_id")
        limit = query_data.get("limit", 10)
        
        if not user_id or not query:
            return {
                "success": False,
                "error": "Missing required fields: user_id and query",
                "results": []
            }
        # 直接调用同步方法
        results = service.search_memories_sync(
            user_id=user_id,
            query=query,
            project_id=project_id,
            limit=limit
            loop.close()
        
        # 格式化结果
        formatted_results = []
        for result in results:
            memory = result.memory
            formatted_results.append({
                "id": memory.id,
                "content": memory.content,
                "category": memory.category,
                "importance": memory.importance,
                "tags": memory.tags,
                "summary": memory.summary,
                "metadata": memory.metadata,
                "created_at": memory.created_at,
                "score": result.score,
                "vector_score": result.vector_score,
                "semantic_score": result.semantic_score
            })
        
        logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
        
        return {
            "success": True,
            "results": formatted_results,
            "total": len(formatted_results),
            "query": query
        }
        
    except Exception as exc:
        logger.error(f"Failed to search memories: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=10)


def get_user_memories(
    user_id: str, 
    project_id: Optional[str] = None, 
    limit: int = 100, 
    offset: int = 0
) -> Dict[str, Any]:
    """
    获取用户记忆列表（同步函数）
    
    Args:
        user_id: 用户ID
        project_id: 项目ID（可选）
        limit: 数量限制
        offset: 偏移量
    
    Returns:
        dict: 包含记忆列表和总数
    """
    try:
        service = _get_memory_service()
        
        memories, total = service.get_memories(
            user_id=user_id,
            project_id=project_id,
            limit=limit,
            offset=offset
        )
        
        formatted_memories = []
        for memory in memories:
            formatted_memories.append({
                "id": memory.id,
                "content": memory.content[:200] + "..." if len(memory.content) > 200 else memory.content,
                "category": memory.category,
                "importance": memory.importance,
                "tags": memory.tags,
                "summary": memory.summary,
                "project_id": memory.project_id,
                "created_at": memory.created_at,
                "score": memory.score
            })
        
        return {
            "success": True,
            "results": formatted_memories,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as exc:
        logger.error(f"Failed to get user memories: {exc}", exc_info=True)
        return {
            "success": False,
            "error": str(exc),
            "results": [],
            "total": 0
        }


def get_memory_by_id(memory_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    通过ID获取记忆详情（同步函数）
    
    Args:
        memory_id: 记忆ID
        user_id: 用户ID
    
    Returns:
        dict or None: 记忆详情
    """
    try:
        service = _get_memory_service()
        
        memory = service.get_memory(memory_id, user_id)
        
        if not memory:
            return None
        
        return {
            "id": memory.id,
            "content": memory.content,
            "category": memory.category,
            "subcategory": memory.subcategory,
            "importance": memory.importance,
            "tags": memory.tags,
            "summary": memory.summary,
            "entities": memory.entities,
            "metadata": memory.metadata,
            "project_id": memory.project_id,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "score": memory.score
        }
        
    except Exception as exc:
        logger.error(f"Failed to get memory {memory_id}: {exc}", exc_info=True)
        return None


def delete_memory(memory_id: str, user_id: str) -> bool:
    """
    删除记忆（同步函数）
    
    Args:
        memory_id: 记忆ID
        user_id: 用户ID
    
    Returns:
        bool: 是否删除成功
    """
    try:
        service = _get_memory_service()
        
        success = service.delete_memory(memory_id, user_id)
        
        if success:
            logger.info(f"Memory deleted: {memory_id}")
        
        return success
        
    except Exception as exc:
        logger.error(f"Failed to delete memory {memory_id}: {exc}", exc_info=True)
        return False


@shared_task
def cleanup_old_memories(days: int = 365, dry_run: bool = True):
    """
    清理旧记忆任务
    
    Args:
        days: 清理多少天前的记忆
        dry_run: 是否为试运行（不实际删除）
    
    Returns:
        dict: 清理结果
    """
    logger.info(f"Memory cleanup task started (days={days}, dry_run={dry_run})")
    
    # TODO: 实现旧记忆清理逻辑
    # 可以基于分数、访问时间等因素决定删除哪些记忆
    
    return {
        "success": True,
        "dry_run": dry_run,
        "message": "Cleanup task placeholder - not implemented yet"
    }


@shared_task
def recalculate_memory_scores():
    """
    重新计算所有记忆分数
    用于定时任务，更新记忆的时效性分数
    """
    logger.info("Recalculating memory scores task started")
    
    # TODO: 实现分数重新计算逻辑
    # 可以基于时间衰减重新计算所有记忆的分数
    
    return {
        "success": True,
        "message": "Score recalculation placeholder - not implemented yet"
    }