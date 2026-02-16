"""
Memory Service Module V2 - Hybrid Storage
记忆服务模块 V2 - 混合存储（Qdrant + PostgreSQL）

架构说明：
- Qdrant: 只存储向量 + 过滤/排序字段
- PostgreSQL: 存储完整记忆数据
"""
import os
import uuid
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict

import httpx
import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

from app.services.classification import (
    ClassificationResult, quick_classify,
    MemoryCategory, MemoryImportance
)
from app.services.scoring import calculate_memory_score, ScoringFactors

logger = logging.getLogger(__name__)

# 集合配置
COLLECTION_NAME = "memories"
VECTOR_SIZE = 1024  # bge-m3 维度


@dataclass
class Memory:
    """记忆数据模型"""
    id: str
    user_id: str
    content: str
    project_id: str = "default"
    category: str = "other"
    subcategory: Optional[str] = None
    importance: int = 3
    tags: List[str] = None
    summary: Optional[str] = None
    entities: List[Dict] = None
    metadata: Dict = None
    created_at: str = None
    updated_at: str = None
    score: float = 0.0
    vector_id: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.entities is None:
            self.entities = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.vector_id is None:
            self.vector_id = self.id


@dataclass
class SearchResult:
    """搜索结果模型"""
    memory: Memory
    score: float
    vector_score: float
    semantic_score: float
    temporal_score: float
    category_score: float


class EmbeddingService:
    """向量嵌入服务"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "ollama")
        self.model = model or os.getenv("EMBED_MODEL", "bge-m3")
        self.base_url = os.getenv("LLM_BASE_URL", "http://192.168.31.65:11434")
        self.timeout = 30.0
    
    def get_embedding_sync(self, text: str) -> List[float]:
        """同步获取向量嵌入 - Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": text[:8000]
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
                
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return [0.0] * VECTOR_SIZE


class DatabaseService:
    """PostgreSQL 数据库服务"""
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql://memoryx:memoryx123@localhost:5432/memoryx")
    
    def _get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(self.db_url)
    
    def create_memory(self, memory: Memory) -> Memory:
        """创建记忆记录"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memory_vectors 
                    (id, user_id, content, project_id, category, subcategory, 
                     importance, tags, summary, entities, metadata, score, 
                     created_at, updated_at, vector_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING created_at
                """, (
                    memory.id, memory.user_id, memory.content, memory.project_id,
                    memory.category, memory.subcategory, memory.importance,
                    json.dumps(memory.tags), memory.summary,
                    json.dumps(memory.entities), json.dumps(memory.metadata),
                    memory.score, memory.created_at, memory.updated_at, memory.vector_id
                ))
                result = cur.fetchone()
                memory.created_at = result[0].isoformat() if result else memory.created_at
            conn.commit()
        return memory
    
    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """获取单个记忆"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM memory_vectors 
                    WHERE id = %s AND user_id = %s
                """, (memory_id, user_id))
                row = cur.fetchone()
                if row:
                    return self._row_to_memory(row)
        return None
    
    def get_memories_by_ids(self, ids: List[str], user_id: str) -> Dict[str, Memory]:
        """批量获取记忆"""
        if not ids:
            return {}
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM memory_vectors 
                    WHERE id = ANY(%s::uuid[]) AND user_id = %s
                """, (ids, user_id))
                rows = cur.fetchall()
                return {row['id']: self._row_to_memory(row) for row in rows}
    
    def update_memory(self, memory_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Memory]:
        """更新记忆"""
        memory = self.get_memory(memory_id, user_id)
        if not memory:
            return None
        
        # 构建更新字段
        set_clauses = []
        values = []
        
        if "content" in updates:
            set_clauses.append("content = %s")
            values.append(updates["content"])
        if "metadata" in updates:
            set_clauses.append("metadata = %s")
            values.append(json.dumps({**memory.metadata, **updates["metadata"]}))
        
        set_clauses.append("updated_at = %s")
        values.append(datetime.utcnow().isoformat())
        values.extend([memory_id, user_id])
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE memory_vectors 
                    SET {', '.join(set_clauses)}
                    WHERE id = %s AND user_id = %s
                """, values)
            conn.commit()
        
        return self.get_memory(memory_id, user_id)
    
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """删除记忆"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM memory_vectors 
                    WHERE id = %s AND user_id = %s
                """, (memory_id, user_id))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
    
    def _row_to_memory(self, row: Dict) -> Memory:
        """数据库行转 Memory 对象"""
        return Memory(
            id=str(row['id']),
            user_id=row['user_id'],
            content=row['content'],
            project_id=row['project_id'] or 'default',
            category=row['category'] or 'other',
            subcategory=row['subcategory'],
            importance=row['importance'] or 3,
            tags=row['tags'] if isinstance(row['tags'], list) else json.loads(row['tags'] or '[]'),
            summary=row['summary'],
            entities=row['entities'] if isinstance(row['entities'], list) else json.loads(row['entities'] or '[]'),
            metadata=row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'] or '{}'),
            score=float(row['score'] or 0),
            created_at=row['created_at'].isoformat() if row['created_at'] else None,
            updated_at=row['updated_at'].isoformat() if row['updated_at'] else None,
            vector_id=row['vector_id']
        )


class MemoryService:
    """记忆服务主类 - 混合存储"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, qdrant_url: Optional[str] = None, api_key: Optional[str] = None):
        if self._initialized:
            return
            
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # Qdrant 客户端
        if self.qdrant_api_key:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        else:
            self.client = QdrantClient(url=self.qdrant_url)
        
        # 嵌入服务
        self.embedding_service = EmbeddingService()
        
        # 数据库服务
        self.db_service = DatabaseService()
        
        # 确保集合存在
        self._ensure_collection()
        
        self._initialized = True
        logger.info(f"MemoryService V2 initialized (Hybrid: Qdrant + PostgreSQL)")
    
    def _ensure_collection(self):
        """确保集合存在"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {COLLECTION_NAME}")
                
                # 创建索引
                self._create_indexes()
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _create_indexes(self):
        """创建 Qdrant payload 索引"""
        try:
            for field in ["user_id", "project_id", "category"]:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_type="keyword"
                )
            logger.info("Created Qdrant payload indexes")
        except Exception as e:
            logger.warning(f"Index creation may have failed: {e}")
    
    def create_memory_sync(
        self,
        user_id: str,
        content: str,
        project_id: str = "default",
        metadata: Optional[Dict] = None
    ) -> Memory:
        """同步创建记忆"""
        memory_id = str(uuid.uuid4())
        
        # 分类
        classification = quick_classify(content)
        
        # 计算分数
        scoring_factors = ScoringFactors(
            importance=classification.importance.value,
            recency=1.0,
            category_boost=self._get_category_boost(classification.category),
            user_interaction=1.0
        )
        memory_score = calculate_memory_score(scoring_factors)
        
        # 创建 Memory 对象
        memory = Memory(
            id=memory_id,
            user_id=user_id,
            content=content,
            project_id=project_id,
            category=classification.category.value,
            subcategory=classification.subcategory,
            importance=classification.importance.value,
            tags=classification.tags,
            summary=classification.summary,
            entities=classification.entities,
            metadata=metadata or {},
            score=memory_score
        )
        
        # 获取向量
        embedding = self.embedding_service.get_embedding_sync(content)
        
        # Qdrant: 只存必要字段
        qdrant_payload = {
            "user_id": user_id,
            "project_id": project_id,
            "category": memory.category,
            "importance": memory.importance,
            "created_at": memory.created_at
        }
        
        point = PointStruct(
            id=memory.vector_id,
            vector=embedding,
            payload=qdrant_payload
        )
        
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        # PostgreSQL: 存完整数据
        self.db_service.create_memory(memory)
        
        logger.info(f"Created memory {memory_id} (Qdrant + PostgreSQL)")
        return memory
    
    def search_memories_sync(
        self,
        user_id: str,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[SearchResult]:
        """同步搜索记忆"""
        # 1. 获取查询向量
        query_vector = self.embedding_service.get_embedding_sync(query)
        
        # 2. 构建过滤条件
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        
        if project_id:
            must_conditions.append(
                FieldCondition(key="project_id", match=MatchValue(value=project_id))
            )
        
        if category:
            must_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        
        # 3. Qdrant 向量搜索
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(must=must_conditions),
            limit=limit * 2,
            score_threshold=0.3
        )
        
        if not results.points:
            return []
        
        # 4. 批量获取 PostgreSQL 详情
        ids = [p.id for p in results.points]
        id_to_score = {p.id: p.score for p in results.points}
        memories = self.db_service.get_memories_by_ids(ids, user_id)
        
        # 5. 构建结果
        search_results = []
        query_lower = query.lower()
        
        for point in results.points:
            memory = memories.get(point.id)
            if not memory:
                continue
            
            vector_score = point.score
            semantic_score = self._calculate_semantic_score(memory, query_lower)
            temporal_score = self._calculate_temporal_score(memory)
            category_score = self._calculate_category_score(memory, query_lower)
            
            final_score = (
                vector_score * 0.5 +
                semantic_score * 0.2 +
                temporal_score * 0.15 +
                category_score * 0.15
            )
            
            search_results.append(SearchResult(
                memory=memory,
                score=final_score,
                vector_score=vector_score,
                semantic_score=semantic_score,
                temporal_score=temporal_score,
                category_score=category_score
            ))
        
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:limit]
    
    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """获取单个记忆"""
        return self.db_service.get_memory(memory_id, user_id)
    
    def update_memory(self, memory_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Memory]:
        """更新记忆"""
        memory = self.db_service.update_memory(memory_id, user_id, updates)
        
        # 如果内容更新，重新生成向量
        if memory and "content" in updates:
            embedding = self.embedding_service.get_embedding_sync(memory.content)
            
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(
                    id=memory.vector_id,
                    vector=embedding,
                    payload={
                        "user_id": memory.user_id,
                        "project_id": memory.project_id,
                        "category": memory.category,
                        "importance": memory.importance,
                        "created_at": memory.created_at
                    }
                )]
            )
        
        return memory
    
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """删除记忆"""
        memory = self.db_service.get_memory(memory_id, user_id)
        if not memory:
            return False
        
        # 删除 Qdrant 向量
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=[memory_id]
            )
        except Exception as e:
            logger.warning(f"Qdrant delete failed: {e}")
        
        # 删除 PostgreSQL 记录
        return self.db_service.delete_memory(memory_id, user_id)
    
    def _get_category_boost(self, category: MemoryCategory) -> float:
        """获取类别加成"""
        boosts = {
            MemoryCategory.PREFERENCE: 1.2,
            MemoryCategory.PERSON: 1.3,
            MemoryCategory.GOAL: 1.1,
            MemoryCategory.EVENT: 0.9,
        }
        return boosts.get(category, 1.0)
    
    def _calculate_semantic_score(self, memory: Memory, query_lower: str) -> float:
        """计算语义分数"""
        score = 0.0
        content_lower = memory.content.lower()
        
        # 关键词匹配
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 1 and word in content_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_temporal_score(self, memory: Memory) -> float:
        """计算时间衰减分数"""
        try:
            created = datetime.fromisoformat(memory.created_at.replace('Z', '+00:00'))
            days_old = (datetime.utcnow() - created.replace(tzinfo=None)).days
            return max(0.5, 1.0 - days_old * 0.01)
        except:
            return 0.7
    
    def _calculate_category_score(self, memory: Memory, query_lower: str) -> float:
        """计算类别匹配分数"""
        category_keywords = {
            "preference": ["喜欢", "讨厌", "prefer", "like"],
            "task": ["任务", "待办", "task", "todo"],
            "goal": ["目标", "计划", "goal", "plan"],
        }
        
        keywords = category_keywords.get(memory.category, [])
        for kw in keywords:
            if kw in query_lower:
                return 1.0
        return 0.5


# 单例访问
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """获取记忆服务单例"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


def init_memory_service(qdrant_url=None, api_key=None):
    """初始化记忆服务（兼容旧接口）"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService(qdrant_url, api_key)
    return _memory_service


    # 兼容别名
    search_memories = search_memories_sync
    create_memory = create_memory_sync

