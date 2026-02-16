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
from app.services.scoring import ScoringFactors, calculate_memory_score, calculate_recency
from app.services.memory_service import COLLECTION_NAME

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
        
        # 使用asyncio运行异步搜索
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                service.search_memories(
                    user_id=user_id,
                    query=query,
                    project_id=project_id,
                    limit=limit
                )
            )
        finally:
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
    
    基于以下条件清理记忆：
    - 创建时间超过指定天数
    - 分数低于阈值（低价值记忆）
    - 未被标记为重要或置顶
    
    Args:
        days: 清理多少天前的记忆（默认365天）
        dry_run: 是否为试运行（不实际删除）
    
    Returns:
        dict: 清理结果
    """
    logger.info(f"Memory cleanup task started (days={days}, dry_run={dry_run})")
    
    try:
        service = _get_memory_service()
        from datetime import datetime, timedelta
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        score_threshold = 0.3  # 分数低于此值的记忆将被清理
        
        # 构建查询：查找旧记忆（低分数且未标记重要）
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="created_at",
                    range=Range(lt=cutoff_date)
                ),
                FieldCondition(
                    key="score",
                    range=Range(lt=score_threshold)
                )
            ]
        )
        
        # 分页获取需要清理的记忆
        memories_to_clean = []
        next_offset = None
        
        while True:
            results, next_offset = service.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=100,
                offset=next_offset
            )
            
            for point in results:
                payload = point.payload or {}
                # 排除被标记为重要或置顶的记忆
                metadata = payload.get("metadata", {})
                if not metadata.get("is_pinned") and not metadata.get("is_important"):
                    memories_to_clean.append({
                        "id": point.id,
                        "user_id": payload.get("user_id"),
                        "score": payload.get("score", 0),
                        "created_at": payload.get("created_at"),
                        "content_preview": payload.get("content", "")[:50]
                    })
            
            if next_offset is None:
                break
        
        total_found = len(memories_to_clean)
        deleted_count = 0
        
        # 执行删除（如果不是dry_run）
        if not dry_run and memories_to_clean:
            # 批量删除，每次100个
            batch_size = 100
            for i in range(0, len(memories_to_clean), batch_size):
                batch = memories_to_clean[i:i + batch_size]
                point_ids = [m["id"] for m in batch]
                
                service.client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=point_ids
                )
                deleted_count += len(batch)
                logger.info(f"Deleted batch of {len(batch)} memories")
        
        result = {
            "success": True,
            "dry_run": dry_run,
            "days_threshold": days,
            "score_threshold": score_threshold,
            "total_found": total_found,
            "deleted_count": deleted_count if not dry_run else 0,
            "message": f"Found {total_found} old memories to clean up" + 
                      (f", deleted {deleted_count}" if not dry_run else " (dry run, no deletions)"),
            "sample": memories_to_clean[:5]  # 返回前5个作为样本
        }
        
        logger.info(f"Memory cleanup task completed: {result['message']}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to cleanup old memories: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "dry_run": dry_run
        }


@shared_task
def recalculate_memory_scores(batch_size: int = 100):
    """
    重新计算所有记忆分数
    
    基于时间衰减重新计算所有记忆的分数，用于定时任务更新记忆的时效性分数。
    
    Args:
        batch_size: 每批处理的记忆数量
    
    Returns:
        dict: 重新计算结果
    """
    logger.info(f"Recalculating memory scores task started (batch_size={batch_size})")
    
    try:
        service = _get_memory_service()
        from datetime import datetime
        from qdrant_client.models import ScoredPoint
        
        updated_count = 0
        total_processed = 0
        errors = []
        
        # 分页获取所有记忆
        next_offset = None
        
        while True:
            results, next_offset = service.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                break
            
            # 准备更新的点
            points_to_update = []
            
            for point in results:
                try:
                    payload = point.payload or {}
                    
                    # 获取记忆信息
                    created_at_str = payload.get("created_at")
                    importance = payload.get("importance", 3)
                    current_score = payload.get("score", 0.5)
                    
                    if not created_at_str:
                        continue
                    
                    # 解析创建时间
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    except ValueError:
                        created_at = datetime.utcnow()
                    
                    # 根据重要性确定衰减级别
                    if importance >= 5:
                        importance_level = "critical"
                    elif importance >= 4:
                        importance_level = "high"
                    elif importance >= 3:
                        importance_level = "medium"
                    elif importance >= 2:
                        importance_level = "low"
                    else:
                        importance_level = "trivial"
                    
                    # 计算新的时效性分数
                    new_recency = calculate_recency(created_at, None, importance_level)
                    
                    # 计算新分数（保持重要性权重，更新时效性）
                    scoring_factors = ScoringFactors(
                        importance=importance,
                        recency=new_recency,
                        frequency=0.1,  # 默认低频率
                        relevance=0.5,  # 中性相关性
                        category_boost=1.0,
                        user_interaction=0.5
                    )
                    new_score = calculate_memory_score(scoring_factors)
                    
                    # 只有分数变化超过阈值才更新
                    if abs(new_score - current_score) > 0.01:
                        # 更新payload中的分数
                        updated_payload = payload.copy()
                        updated_payload["score"] = new_score
                        updated_payload["score_updated_at"] = datetime.utcnow().isoformat()
                        
                        points_to_update.append({
                            "id": point.id,
                            "payload": updated_payload
                        })
                        updated_count += 1
                    
                    total_processed += 1
                    
                except Exception as item_error:
                    errors.append(f"Error processing memory {point.id}: {str(item_error)}")
                    logger.warning(f"Failed to recalculate score for memory {point.id}: {item_error}")
            
            # 批量更新
            if points_to_update:
                for point_data in points_to_update:
                    service.client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload=point_data["payload"],
                        points=[point_data["id"]]
                    )
                logger.info(f"Updated {len(points_to_update)} memory scores in batch")
            
            if next_offset is None:
                break
        
        result = {
            "success": True,
            "total_processed": total_processed,
            "updated_count": updated_count,
            "unchanged_count": total_processed - updated_count,
            "error_count": len(errors),
            "errors": errors[:10],  # 只返回前10个错误
            "message": f"Processed {total_processed} memories, updated {updated_count} scores"
        }
        
        logger.info(f"Memory score recalculation completed: {result['message']}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to recalculate memory scores: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }