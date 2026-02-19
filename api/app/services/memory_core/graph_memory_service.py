"""
自定义图记忆服务 - 不依赖 function calling
直接使用 LLM + prompt 提取实体和关系，然后写入 Neo4j
"""

from typing import List, Dict, Any, Optional
import logging
import httpx
import json
import asyncio
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import get_settings
from app.core.database import SessionLocal, Fact, Memory

logger = logging.getLogger(__name__)
settings = get_settings()

MEMORY_UPDATE_PROMPT = """你是一个智能记忆管理器，负责管理用户的记忆系统。
你可以执行四种操作：(1) ADD 添加新记忆，(2) UPDATE 更新已有记忆，(3) DELETE 删除记忆，(4) NONE 无需操作。

比较新提取的事实与已有记忆，对每个事实决定操作类型：

## 操作规则：

### 1. ADD（添加）
如果新事实包含记忆中不存在的新信息，则添加。
示例：
- 已有记忆: [{{"id": "0", "text": "用户是软件工程师"}}]
- 新事实: ["名字叫张三"]
- 操作结果: {{
  "memory": [
    {{"id": "0", "text": "用户是软件工程师", "event": "NONE", "reason": "与当前事实无关，保持不变"}},
    {{"id": "1", "text": "名字叫张三", "event": "ADD", "reason": "新信息：用户姓名，记忆中不存在"}}
  ]
}}

### 2. UPDATE（更新）
如果新事实与已有记忆相关但信息不同或更完整，则更新。保持相同ID。
示例：
- 已有记忆: [{{"id": "0", "text": "用户喜欢披萨"}}, {{"id": "1", "text": "用户喜欢打板球"}}]
- 新事实: ["喜欢吃鸡肉披萨", "喜欢和朋友一起打板球"]
- 操作结果: {{
  "memory": [
    {{"id": "0", "text": "用户喜欢鸡肉披萨", "event": "UPDATE", "old_memory": "用户喜欢披萨", "reason": "信息更具体：明确了披萨口味"}},
    {{"id": "1", "text": "用户喜欢和朋友一起打板球", "event": "UPDATE", "old_memory": "用户喜欢打板球", "reason": "信息更完整：补充了社交场景"}}
  ]
}}

### 3. DELETE（删除）
如果新事实与已有记忆矛盾，则删除。保持相同ID。
示例：
- 已有记忆: [{{"id": "0", "text": "名字叫张三"}}, {{"id": "1", "text": "喜欢吃奶酪披萨"}}]
- 新事实: ["不喜欢奶酪披萨"]
- 操作结果: {{
  "memory": [
    {{"id": "0", "text": "名字叫张三", "event": "NONE", "reason": "与当前事实无关，保持不变"}},
    {{"id": "1", "text": "喜欢吃奶酪披萨", "event": "DELETE", "reason": "矛盾：新信息明确表示不喜欢奶酪披萨"}}
  ]
}}

### 4. NONE（无操作）
如果新事实与已有记忆相同或已被包含，则不操作。
示例：
- 已有记忆: [{{"id": "0", "text": "名字叫张三"}}]
- 新事实: ["名字叫张三"]
- 操作结果: {{
  "memory": [
    {{"id": "0", "text": "名字叫张三", "event": "NONE", "reason": "重复：与已有记忆完全相同"}}
  ]
}}

## 重要提示：
- 检测用户输入的语言，用相同语言记录记忆
- ADD 操作需要生成新的 ID（递增数字）
- UPDATE 和 DELETE 操作必须使用已有记忆的 ID
- 每个操作必须提供 reason 字段，说明判断理由
- 只返回 JSON 格式，不要其他内容
"""

def get_memory_update_messages(existing_memories: List[Dict], new_facts: List[str]) -> str:
    """构建记忆更新 prompt"""
    if existing_memories:
        memory_str = json.dumps(existing_memories, ensure_ascii=False, indent=2)
        current_memory_part = f"""
当前记忆内容：
```
{memory_str}
```
"""
    else:
        current_memory_part = """
当前记忆为空。

"""
    
    facts_str = json.dumps(new_facts, ensure_ascii=False, indent=2)
    
    return f"""{MEMORY_UPDATE_PROMPT}

{current_memory_part}

新提取的事实：
```
{facts_str}
```

请分析新事实，决定对记忆的操作，返回 JSON 格式：
{{
  "memory": [
    {{
      "id": "<记忆ID>",
      "text": "<记忆内容>",
      "event": "<ADD/UPDATE/DELETE/NONE>",
      "old_memory": "<仅UPDATE时需要，原记忆内容>",
      "reason": "<判断理由，说明为什么选择这个操作>"
    }}
  ]
}}
"""

EXTRACT_FACTS_PROMPT = """从以下对话中提取所有独立的事实/记忆。

对话内容：
{text}

## 提取规则：
1. 将复杂句子拆分为简单、独立的原子事实
2. 每个事实应该是一个完整的陈述句
3. 过滤掉问候语、废话、无意义的内容
4. 保留重要信息：偏好、经历、关系、计划、观点等
5. 对事实进行分类：preference(偏好), fact(事实), plan(计划), experience(经历), opinion(观点)
6. 检测输入语言，用相同语言记录事实

## 示例：

示例1（中文）：
输入: "张三在北京阿里云工作，他喜欢喝咖啡，最近在学习Python编程"
输出: {{
  "facts": [
    {{"content": "张三在北京工作", "category": "fact", "importance": "medium"}},
    {{"content": "张三在阿里云工作", "category": "fact", "importance": "medium"}},
    {{"content": "张三喜欢喝咖啡", "category": "preference", "importance": "medium"}},
    {{"content": "张三最近在学习Python编程", "category": "fact", "importance": "medium"}}
  ]
}}

示例2（英文）：
输入: "John works at Google in Mountain View. He loves playing tennis on weekends."
输出: {{
  "facts": [
    {{"content": "John works at Google", "category": "fact", "importance": "medium"}},
    {{"content": "John works in Mountain View", "category": "fact", "importance": "medium"}},
    {{"content": "John loves playing tennis on weekends", "category": "preference", "importance": "medium"}}
  ]
}}

示例3（日文）：
输入: "田中さんは東京に住んでいて、寿司が大好きです。"
输出: {{
  "facts": [
    {{"content": "田中さんは東京に住んでいる", "category": "fact", "importance": "medium"}},
    {{"content": "田中さんは寿司が大好き", "category": "preference", "importance": "medium"}}
  ]
}}

示例4（无有效信息）：
输入: "你好，今天天气不错。"
输出: {{
  "facts": []
}}

严格按以上 JSON 格式返回：
{{
  "facts": [
    {{"content": "事实内容", "category": "分类", "importance": "high/medium/low"}}
  ]
}}
"""

EXTRACT_ENTITIES_PROMPT = """分析以下文本，提取所有实体和它们之间的关系。

文本：
{text}

## 提取规则：
1. 实体类型：人物(person)、地点(location)、组织(organization)、技能(skill)、爱好(hobby)、物品(item)、事件(event)、时间(time)
2. 关系类型用动词或短语表示（如：喜欢、住在、学习、工作于、loves、lives_in、works_at）
3. 检测输入语言，用相同语言记录实体和关系
4. 如果文本中提到"我/I/私"等第一人称，使用 "USER_ID" 作为实体名

## 示例：

示例1（中文）：
输入: "张三在北京阿里云工作，他喜欢喝咖啡"
输出: {{
  "entities": [
    {{"name": "张三", "type": "person"}},
    {{"name": "北京", "type": "location"}},
    {{"name": "阿里云", "type": "organization"}},
    {{"name": "咖啡", "type": "item"}}
  ],
  "relations": [
    {{"source": "张三", "target": "北京", "relation": "住在"}},
    {{"source": "张三", "target": "阿里云", "relation": "工作于"}},
    {{"source": "张三", "target": "咖啡", "relation": "喜欢"}}
  ]
}}

示例2（英文）：
输入: "John lives in New York and works at Microsoft. He enjoys playing basketball."
输出: {{
  "entities": [
    {{"name": "John", "type": "person"}},
    {{"name": "New York", "type": "location"}},
    {{"name": "Microsoft", "type": "organization"}},
    {{"name": "basketball", "type": "hobby"}}
  ],
  "relations": [
    {{"source": "John", "target": "New York", "relation": "lives_in"}},
    {{"source": "John", "target": "Microsoft", "relation": "works_at"}},
    {{"source": "John", "target": "basketball", "relation": "enjoys"}}
  ]
}}

示例3（第一人称）：
输入: "我在上海工作，喜欢打篮球"
输出: {{
  "entities": [
    {{"name": "USER_ID", "type": "person"}},
    {{"name": "上海", "type": "location"}},
    {{"name": "篮球", "type": "hobby"}}
  ],
  "relations": [
    {{"source": "USER_ID", "target": "上海", "relation": "工作于"}},
    {{"source": "USER_ID", "target": "篮球", "relation": "喜欢"}}
  ]
}}

严格按以下 JSON 格式返回，不要包含其他内容：
{{
  "entities": [
    {{"name": "实体名", "type": "类型", "properties": {{"可选属性": "值"}}}}
  ],
  "relations": [
    {{"source": "源实体", "target": "目标实体", "relation": "关系类型"}}
  ]
}}
"""


class GraphMemoryService:
    def __init__(self):
        self.neo4j_driver = None
        self.qdrant_clients: Dict[str, QdrantClient] = {}
        self._init_neo4j()
    
    def _init_neo4j(self):
        try:
            self.neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            logger.info(f"Neo4j connected: {settings.neo4j_uri}")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
    
    def _get_qdrant_client(self, user_id: str) -> QdrantClient:
        collection_name = f"{settings.qdrant_collection}_{user_id[:8]}"
        
        if collection_name not in self.qdrant_clients:
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )
            
            try:
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1024,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            
            self.qdrant_clients[collection_name] = client
        
        return self.qdrant_clients[collection_name]
    
    async def _call_llm(self, messages: List[Dict], temperature: float = 0.1) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "temperature": temperature
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"LLM call failed: {response.status_code} - {response.text}")
            
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    async def _call_qwen(self, messages: List[Dict], temperature: float = 0.1) -> str:
        qwen_url = getattr(settings, 'qwen_base_url', settings.ollama_base_url)
        qwen_model = getattr(settings, 'qwen_model', 'qwen3-14b-sft')
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{qwen_url}/qwen/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": qwen_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Qwen call failed: {response.status_code} - {response.text}")
            
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    async def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": "你是一个专业的记忆提取助手。请从对话中提取所有独立的原子事实，只返回JSON格式。"},
            {"role": "user", "content": EXTRACT_FACTS_PROMPT.format(text=text)}
        ]
        
        try:
            response = await self._call_llm(messages)
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result.get("facts", [])
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in extract_facts: {e}")
        except Exception as e:
            logger.error(f"Fact extraction error: {e}")
        
        return [{"content": text, "category": "fact", "importance": "medium"}]
    
    async def _get_embedding(self, text: str) -> List[float]:
        embed_url = getattr(settings, 'embed_base_url', settings.ollama_base_url)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{embed_url}/v1/embeddings",
                headers={"Content-Type": "application/json"},
                json={
                    "model": settings.embed_model,
                    "input": text
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Embedding failed: {response.status_code}")
            
            data = response.json()
            return data.get("data", [{}])[0].get("embedding", [])
    
    async def extract_entities_and_relations(self, text: str, user_id: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "你是一个专业的实体关系提取助手。请准确提取文本中的实体和关系，只返回JSON格式。"},
            {"role": "user", "content": EXTRACT_ENTITIES_PROMPT.format(text=text)}
        ]
        
        try:
            response = await self._call_llm(messages)
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                for entity in result.get("entities", []):
                    if entity.get("name") in ["我", "本人", "自己"]:
                        entity["name"] = user_id
                
                for relation in result.get("relations", []):
                    if relation.get("source") in ["我", "本人", "自己"]:
                        relation["source"] = user_id
                    if relation.get("target") in ["我", "本人", "自己"]:
                        relation["target"] = user_id
                
                return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
        
        return {"entities": [], "relations": []}
    
    def save_to_neo4j(self, user_id: str, entities: List[Dict], relations: List[Dict]):
        if not self.neo4j_driver:
            logger.warning("Neo4j not connected, skipping graph save")
            return
        
        with self.neo4j_driver.session() as session:
            for entity in entities:
                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "Entity")
                properties = entity.get("properties", {})
                
                if not entity_name:
                    continue
                
                query = f"""
                MERGE (e:{entity_type} {{name: $name, user_id: $user_id}})
                SET e += $properties
                """
                
                try:
                    session.run(query, name=entity_name, user_id=user_id, properties=properties)
                    logger.debug(f"Saved entity: {entity_name} ({entity_type})")
                except Exception as e:
                    logger.error(f"Failed to save entity {entity_name}: {e}")
            
            for relation in relations:
                source = relation.get("source", "")
                target = relation.get("target", "")
                relation_type = relation.get("relation", "RELATED_TO")
                
                if not source or not target:
                    continue
                
                relation_type = relation_type.upper().replace(" ", "_")
                relation_type = "".join(c for c in relation_type if c.isalnum() or c == "_")
                
                if not relation_type:
                    relation_type = "RELATED_TO"
                
                query = f"""
                MATCH (s {{name: $source, user_id: $user_id}})
                MATCH (t {{name: $target, user_id: $user_id}})
                MERGE (s)-[r:{relation_type}]->(t)
                """
                
                try:
                    session.run(query, source=source, target=target, user_id=user_id)
                    logger.debug(f"Saved relation: {source} --{relation_type}--> {target}")
                except Exception as e:
                    logger.error(f"Failed to save relation {source}->{target}: {e}")
    
    async def save_to_qdrant(self, user_id: str, memory_id: str, content: str, metadata: Dict = None, entities: List[Dict] = None, relations: List[Dict] = None, category: str = "fact", importance: str = "medium", fact_id: int = None):
        try:
            client = self._get_qdrant_client(user_id)
            collection_name = f"{settings.qdrant_collection}_{user_id[:8]}"
            
            embedding = await self._get_embedding(content)
            
            entity_names = [e.get("name", "") for e in (entities or []) if e.get("name")]
            relation_list = [f"{r.get('source','')}-{r.get('relation','')}-{r.get('target','')}" for r in (relations or [])]
            
            payload = {
                "content": content,
                "user_id": user_id,
                "metadata": metadata or {},
                "entity_names": entity_names,
                "relations": relation_list,
                "category": category,
                "importance": importance
            }
            
            if fact_id is not None:
                payload["fact_id"] = fact_id
            
            point = PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload
            )
            
            client.upsert(collection_name=collection_name, points=[point])
            logger.debug(f"Saved to Qdrant: {memory_id}")
        except Exception as e:
            logger.error(f"Failed to save to Qdrant: {e}")
    
    def _parse_entities_from_names(self, entity_names: List[str]) -> List[Dict]:
        entities = []
        for name in entity_names:
            if name:
                entity_type = "person"
                if any(loc in name for loc in ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安", "重庆", "天津", "苏州", "长沙", "郑州", "青岛", "大连", "宁波", "厦门", "无锡", "济南", "福州", "哈尔滨", "沈阳", "昆明", "南宁", "贵阳", "太原", "石家庄", "合肥", "南昌", "长春", "兰州", "海口", "乌鲁木齐", "拉萨", "银川", "呼和浩特", "西宁"]):
                    entity_type = "location"
                elif any(org in name for org in ["公司", "集团", "科技", "阿里", "腾讯", "百度", "字节", "华为", "小米", "京东", "美团", "滴滴", "拼多多", "快手", "网易", "微博", "知乎", "B站", "OPPO", "vivo", "联想", "海尔", "格力", "比亚迪", "蔚来", "小鹏", "理想"]):
                    entity_type = "organization"
                elif any(skill in name for skill in ["Python", "Java", "JavaScript", "Go", "Rust", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript", "SQL", "HTML", "CSS", "React", "Vue", "Angular", "Node", "Django", "Flask", "Spring", "TensorFlow", "PyTorch", "Keras", "OpenCV", "Linux", "Docker", "Kubernetes"]):
                    entity_type = "skill"
                elif any(item in name for item in ["咖啡", "茶", "酒", "菜", "饭", "面", "肉", "鱼", "虾", "蟹", "水果", "蔬菜", "饮料", "零食"]):
                    entity_type = "item"
                entities.append({"name": name, "type": entity_type})
        return entities
    
    def _parse_relations_from_list(self, relation_list: List[str]) -> List[Dict]:
        relations = []
        for rel_str in relation_list:
            parts = rel_str.split("-", 2)
            if len(parts) == 3:
                relations.append({
                    "source": parts[0],
                    "relation": parts[1],
                    "target": parts[2]
                })
        return relations
    
    async def search_related_memories(self, user_id: str, new_facts: List[str], limit: int = 5, score_threshold: float = 0.7) -> List[Dict]:
        """
        基于向量语义搜索获取相关记忆
        
        Args:
            user_id: 用户ID
            new_facts: 新提取的事实列表
            limit: 每个事实搜索返回的最大记忆数
            score_threshold: 相似度阈值，低于此值的结果会被过滤
        
        Returns:
            相关记忆列表（去重后）
        """
        if not new_facts:
            return []
        
        try:
            client = self._get_qdrant_client(user_id)
            collection_name = f"{settings.qdrant_collection}_{user_id[:8]}"
            
            embeddings = await self._get_embeddings_batch(new_facts)
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            all_memories = {}
            
            for embedding in embeddings:
                try:
                    results = client.query_points(
                        collection_name=collection_name,
                        query=embedding,
                        limit=limit,
                        score_threshold=score_threshold,
                        query_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="user_id",
                                    match=MatchValue(value=user_id)
                                )
                            ]
                        )
                    )
                    
                    for point in results.points:
                        if point.id not in all_memories:
                            all_memories[point.id] = {
                                "id": point.id,
                                "text": point.payload.get("content", ""),
                                "score": point.score,
                                "category": point.payload.get("category", "fact"),
                                "importance": point.payload.get("importance", "medium"),
                                "entity_names": point.payload.get("entity_names", []),
                                "relation_list": point.payload.get("relation_list", []),
                                "fact_id": point.payload.get("fact_id"),
                                "vector_id": point.id,
                                "entities": self._parse_entities_from_names(point.payload.get("entity_names", [])),
                                "relations": self._parse_relations_from_list(point.payload.get("relation_list", []))
                            }
                except Exception as e:
                    logger.error(f"Failed to search in Qdrant: {e}")
            
            memories = list(all_memories.values())
            memories.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info(f"Found {len(memories)} related memories from vector search (threshold={score_threshold})")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search related memories: {e}")
            return []
    
    async def get_existing_memories(self, user_id: str) -> List[Dict]:
        """
        获取用户所有记忆（仅用于兼容旧接口，不推荐使用）
        推荐使用 search_related_memories 基于向量语义搜索
        """
        db = SessionLocal()
        try:
            facts = db.query(Fact).filter(Fact.user_id == int(user_id) if user_id.isdigit() else 1).all()
            memories = []
            for i, fact in enumerate(facts):
                memories.append({
                    "id": str(i),
                    "text": fact.content,
                    "fact_id": fact.id,
                    "vector_id": fact.vector_id,
                    "category": fact.category,
                    "importance": fact.importance,
                    "entities": fact.entities or [],
                    "relations": fact.relations or []
                })
            return memories
        except Exception as e:
            logger.error(f"Failed to get existing memories: {e}")
            return []
        finally:
            db.close()
    
    async def update_memory_with_judgment(self, user_id: str, new_facts: List[str], existing_memories: List[Dict], input_content: str = "") -> Dict[str, Any]:
        import uuid
        import time
        
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        if not new_facts:
            return {"memory": [], "trace_id": trace_id}
        
        prompt = get_memory_update_messages(existing_memories, new_facts)
        
        messages = [
            {"role": "system", "content": "你是一个智能记忆管理器。请分析新事实并决定对记忆的操作，只返回JSON格式结果。"},
            {"role": "user", "content": prompt}
        ]
        
        llm_response = ""
        parsed_operations = []
        reasoning_list = []
        execution_success = True
        error_message = None
        
        try:
            llm_response = await self._call_qwen(messages)
            
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                result = json.loads(json_str)
                parsed_operations = result.get("memory", [])
                
                reasoning_list = [op.get("reason", "") for op in parsed_operations if op.get("reason")]
            else:
                raise ValueError("No valid JSON found in LLM response")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in update_memory_with_judgment: {e}")
            execution_success = False
            error_message = f"JSON parse error: {str(e)}"
            
            default_memory = []
            for i, fact in enumerate(new_facts):
                default_memory.append({
                    "id": str(len(existing_memories) + i),
                    "text": fact,
                    "event": "ADD",
                    "reason": "默认添加（LLM响应解析失败）"
                })
            parsed_operations = default_memory
            result = {"memory": default_memory}
            
        except Exception as e:
            logger.error(f"Memory judgment error: {e}")
            execution_success = False
            error_message = str(e)
            result = {"memory": []}
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        db = SessionLocal()
        try:
            from app.core.database import MemoryJudgment
            
            judgment_record = MemoryJudgment(
                trace_id=trace_id,
                user_id=int(user_id) if user_id.isdigit() else 1,
                operation_type="MEMORY_UPDATE",
                input_content=input_content,
                extracted_facts=new_facts,
                existing_memories=existing_memories,
                llm_response=llm_response,
                parsed_operations=parsed_operations,
                reasoning="\n".join(reasoning_list) if reasoning_list else None,
                execution_success=execution_success,
                error_message=error_message,
                model_name=getattr(settings, 'qwen_model', 'qwen3-14b-sft'),
                latency_ms=latency_ms
            )
            db.add(judgment_record)
            db.commit()
            logger.info(f"Saved judgment record: {trace_id}")
        except Exception as e:
            logger.error(f"Failed to save judgment record: {e}")
            db.rollback()
        finally:
            db.close()
        
        result["trace_id"] = trace_id
        return result
    
    def update_neo4j_entities(self, user_id: str, old_entities: List[Dict], new_entities: List[Dict], old_relations: List[Dict], new_relations: List[Dict]):
        if not self.neo4j_driver:
            logger.warning("Neo4j not connected, skipping graph update")
            return
        
        old_entity_names = {e.get("name") for e in old_entities if e.get("name")}
        new_entity_names = {e.get("name") for e in new_entities if e.get("name")}
        
        entities_to_remove = old_entity_names - new_entity_names
        entities_to_add = new_entity_names - old_entity_names
        entities_to_update = old_entity_names & new_entity_names
        
        old_relation_set = set()
        for r in old_relations:
            key = (r.get("source"), r.get("relation"), r.get("target"))
            if all(key):
                old_relation_set.add(key)
        
        new_relation_set = set()
        for r in new_relations:
            key = (r.get("source"), r.get("relation"), r.get("target"))
            if all(key):
                new_relation_set.add(key)
        
        relations_to_remove = old_relation_set - new_relation_set
        relations_to_add = new_relation_set - old_relation_set
        
        with self.neo4j_driver.session() as session:
            for source, relation, target in relations_to_remove:
                relation_type_safe = relation.upper().replace(" ", "_")
                relation_type_safe = "".join(c for c in relation_type_safe if c.isalnum() or c == "_")
                
                if not relation_type_safe:
                    relation_type_safe = "RELATED_TO"
                
                try:
                    query = f"""
                    MATCH (s {{name: $source, user_id: $user_id}})-[r:{relation_type_safe}]->(t {{name: $target, user_id: $user_id}})
                    DELETE r
                    """
                    session.run(query, source=source, target=target, user_id=user_id)
                    logger.debug(f"Removed relation: {source} --{relation_type_safe}--> {target}")
                except Exception as e:
                    logger.error(f"Failed to remove relation {source}->{target}: {e}")
            
            for entity_name in entities_to_remove:
                try:
                    check_query = """
                    MATCH (e {name: $name, user_id: $user_id})
                    OPTIONAL MATCH (e)-[r]-()
                    WITH e, count(r) as rel_count
                    RETURN rel_count
                    """
                    result = session.run(check_query, name=entity_name, user_id=user_id)
                    record = result.single()
                    
                    if record and record["rel_count"] == 0:
                        delete_query = """
                        MATCH (e {name: $name, user_id: $user_id})
                        DELETE e
                        """
                        session.run(delete_query, name=entity_name, user_id=user_id)
                        logger.debug(f"Removed orphan entity: {entity_name}")
                    else:
                        logger.debug(f"Kept entity: {entity_name} (still has relations)")
                except Exception as e:
                    logger.error(f"Failed to check/remove entity {entity_name}: {e}")
            
            for entity in new_entities:
                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "Entity")
                properties = entity.get("properties", {})
                
                if not entity_name:
                    continue
                
                try:
                    query = f"""
                    MERGE (e:{entity_type} {{name: $name, user_id: $user_id}})
                    SET e += $properties, e.updated_at = datetime()
                    """
                    session.run(query, name=entity_name, user_id=user_id, properties=properties)
                    logger.debug(f"Updated entity: {entity_name} ({entity_type})")
                except Exception as e:
                    logger.error(f"Failed to update entity {entity_name}: {e}")
            
            for source, relation, target in relations_to_add:
                relation_type_safe = relation.upper().replace(" ", "_")
                relation_type_safe = "".join(c for c in relation_type_safe if c.isalnum() or c == "_")
                
                if not relation_type_safe:
                    relation_type_safe = "RELATED_TO"
                
                try:
                    query = f"""
                    MATCH (s {{name: $source, user_id: $user_id}})
                    MATCH (t {{name: $target, user_id: $user_id}})
                    MERGE (s)-[r:{relation_type_safe}]->(t)
                    SET r.updated_at = datetime()
                    """
                    session.run(query, source=source, target=target, user_id=user_id)
                    logger.debug(f"Added relation: {source} --{relation_type_safe}--> {target}")
                except Exception as e:
                    logger.error(f"Failed to add relation {source}->{target}: {e}")
        
        return {
            "entities_removed": list(entities_to_remove),
            "entities_added": list(entities_to_add),
            "entities_updated": list(entities_to_update),
            "relations_removed": [f"{s}-{r}-{t}" for s, r, t in relations_to_remove],
            "relations_added": [f"{s}-{r}-{t}" for s, r, t in relations_to_add]
        }
    
    def delete_from_neo4j(self, user_id: str, entities: List[Dict], relations: List[Dict]):
        if not self.neo4j_driver:
            logger.warning("Neo4j not connected, skipping graph deletion")
            return
        
        with self.neo4j_driver.session() as session:
            for relation in relations:
                source = relation.get("source", "")
                target = relation.get("target", "")
                relation_type = relation.get("relation", "RELATED_TO")
                
                if not source or not target:
                    continue
                
                relation_type_safe = relation_type.upper().replace(" ", "_")
                relation_type_safe = "".join(c for c in relation_type_safe if c.isalnum() or c == "_")
                
                if not relation_type_safe:
                    relation_type_safe = "RELATED_TO"
                
                try:
                    query = f"""
                    MATCH (s {{name: $source, user_id: $user_id}})-[r:{relation_type_safe}]->(t {{name: $target, user_id: $user_id}})
                    DELETE r
                    """
                    session.run(query, source=source, target=target, user_id=user_id)
                    logger.debug(f"Deleted relation: {source} --{relation_type_safe}--> {target}")
                except Exception as e:
                    logger.error(f"Failed to delete relation {source}->{target}: {e}")
            
            for entity in entities:
                entity_name = entity.get("name", "")
                if not entity_name:
                    continue
                
                try:
                    check_query = """
                    MATCH (e {name: $name, user_id: $user_id})
                    OPTIONAL MATCH (e)-[r]-()
                    WITH e, count(r) as rel_count
                    RETURN rel_count
                    """
                    result = session.run(check_query, name=entity_name, user_id=user_id)
                    record = result.single()
                    
                    if record and record["rel_count"] == 0:
                        delete_query = """
                        MATCH (e {name: $name, user_id: $user_id})
                        DELETE e
                        """
                        session.run(delete_query, name=entity_name, user_id=user_id)
                        logger.debug(f"Deleted entity: {entity_name} (no remaining relations)")
                    else:
                        logger.debug(f"Skipped entity deletion: {entity_name} (has {record['rel_count'] if record else '?'} relations)")
                except Exception as e:
                    logger.error(f"Failed to check/delete entity {entity_name}: {e}")
    
    def delete_from_qdrant(self, user_id: str, vector_id: str) -> bool:
        try:
            client = self._get_qdrant_client(user_id)
            collection_name = f"{settings.qdrant_collection}_{user_id[:8]}"
            
            from qdrant_client.models import PointIdsList
            client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=[vector_id])
            )
            logger.debug(f"Deleted from Qdrant: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            return False
    
    async def execute_memory_operations(self, user_id: str, memory_operations: List[Dict], existing_memories: List[Dict], metadata: Dict = None) -> Dict[str, Any]:
        import uuid
        
        db = SessionLocal()
        added = []
        updated = []
        deleted = []
        
        try:
            for op in memory_operations:
                event = op.get("event", "NONE")
                memory_id = op.get("id", "")
                text = op.get("text", "")
                old_memory = op.get("old_memory", "")
                
                if event == "ADD":
                    extraction = await self.extract_entities_and_relations(text, user_id)
                    entities = extraction.get("entities", [])
                    relations = extraction.get("relations", [])
                    
                    vector_id = str(uuid.uuid4())
                    
                    try:
                        fact_record = Fact(
                            user_id=int(user_id) if user_id.isdigit() else 1,
                            content=text,
                            category="fact",
                            importance="medium",
                            vector_id=vector_id,
                            entities=entities,
                            relations=relations
                        )
                        db.add(fact_record)
                        db.flush()
                        fact_id = fact_record.id
                        
                        await self.save_to_qdrant(user_id, vector_id, text, metadata, entities, relations, fact_id=fact_id)
                        
                        self.save_to_neo4j(user_id, entities, relations)
                        
                        added.append({
                            "id": vector_id,
                            "fact_id": fact_id,
                            "content": text,
                            "entities": entities,
                            "relations": relations
                        })
                    except Exception as e:
                        logger.error(f"Failed to add fact to DB: {e}")
                
                elif event == "UPDATE":
                    existing = next((m for m in existing_memories if m.get("id") == memory_id), None)
                    if existing:
                        vector_id = existing.get("vector_id")
                        fact_id = existing.get("fact_id")
                        old_entities = existing.get("entities", [])
                        old_relations = existing.get("relations", [])
                        old_content = existing.get("text", "")
                        
                        extraction = await self.extract_entities_and_relations(text, user_id)
                        new_entities = extraction.get("entities", [])
                        new_relations = extraction.get("relations", [])
                        
                        if vector_id:
                            await self.save_to_qdrant(user_id, vector_id, text, metadata, new_entities, new_relations)
                        
                        graph_changes = self.update_neo4j_entities(
                            user_id, old_entities, new_entities, old_relations, new_relations
                        )
                        
                        try:
                            fact_record = db.query(Fact).filter(Fact.id == fact_id).first()
                            if fact_record:
                                fact_record.content = text
                                fact_record.entities = new_entities
                                fact_record.relations = new_relations
                                db.flush()
                            
                            updated.append({
                                "id": vector_id,
                                "fact_id": fact_id,
                                "content": text,
                                "old_content": old_content,
                                "entities": new_entities,
                                "relations": new_relations,
                                "graph_changes": graph_changes
                            })
                        except Exception as e:
                            logger.error(f"Failed to update fact in DB: {e}")
                
                elif event == "DELETE":
                    existing = next((m for m in existing_memories if m.get("id") == memory_id), None)
                    if existing:
                        vector_id = existing.get("vector_id") or existing.get("id")
                        fact_id = existing.get("fact_id")
                        
                        if not fact_id and vector_id:
                            fact_by_vector = db.query(Fact).filter(Fact.vector_id == str(vector_id)).first()
                            if fact_by_vector:
                                fact_id = fact_by_vector.id
                        
                        fact_record = db.query(Fact).filter(Fact.id == fact_id).first() if fact_id else None
                        entities_to_delete = []
                        relations_to_delete = []
                        
                        if fact_record:
                            entities_to_delete = fact_record.entities or []
                            relations_to_delete = fact_record.relations or []
                        
                        if vector_id:
                            self.delete_from_qdrant(user_id, vector_id)
                        
                        if relations_to_delete:
                            self.delete_from_neo4j(user_id, entities_to_delete, relations_to_delete)
                        
                        try:
                            if fact_record:
                                db.delete(fact_record)
                            
                            deleted.append({
                                "id": vector_id,
                                "fact_id": fact_id,
                                "content": existing.get("text", ""),
                                "entities": entities_to_delete,
                                "relations": relations_to_delete
                            })
                        except Exception as e:
                            logger.error(f"Failed to delete fact from DB: {e}")
            
            db.commit()
        except Exception as e:
            logger.error(f"Failed to execute memory operations: {e}")
            db.rollback()
        finally:
            db.close()
        
        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "stats": {
                "added_count": len(added),
                "updated_count": len(updated),
                "deleted_count": len(deleted)
            }
        }
    
    async def add_memory(self, user_id: str, content: str, metadata: Dict = None, skip_judge: bool = False) -> Dict[str, Any]:
        import uuid
        
        db = SessionLocal()
        try:
            memory_record = Memory(
                content=content,
                user_id=int(user_id) if user_id.isdigit() else 1,
                meta=metadata or {}
            )
            db.add(memory_record)
            db.flush()
            memory_db_id = memory_record.id
        except Exception as e:
            logger.error(f"Failed to create memory record: {e}")
            db.rollback()
            memory_db_id = None
        finally:
            if db:
                db.close()
        
        facts = await self.extract_facts(content)
        logger.info(f"Extracted {len(facts)} facts from content")
        
        if not facts:
            return {
                "id": None,
                "content": content,
                "facts": [],
                "event": "NONE",
                "message": "No facts extracted"
            }
        
        fact_contents = [f.get("content", "") for f in facts if f.get("content")]
        
        if skip_judge:
            all_entities = []
            all_relations = []
            stored_facts = []
            
            for fact in facts:
                fact_content = fact.get("content", "")
                category = fact.get("category", "fact")
                importance = fact.get("importance", "medium")
                
                extraction = await self.extract_entities_and_relations(fact_content, user_id)
                entities = extraction.get("entities", [])
                relations = extraction.get("relations", [])
                
                all_entities.extend(entities)
                all_relations.extend(relations)
                
                vector_id = str(uuid.uuid4())
                await self.save_to_qdrant(user_id, vector_id, fact_content, metadata, entities, relations, category, importance)
                
                db = SessionLocal()
                try:
                    fact_record = Fact(
                        memory_id=memory_db_id,
                        user_id=int(user_id) if user_id.isdigit() else 1,
                        content=fact_content,
                        category=category,
                        importance=importance,
                        vector_id=vector_id,
                        entities=entities,
                        relations=relations
                    )
                    db.add(fact_record)
                    db.commit()
                except Exception as e:
                    logger.error(f"Failed to save fact to DB: {e}")
                    db.rollback()
                finally:
                    db.close()
                
                stored_facts.append({
                    "id": vector_id,
                    "content": fact_content,
                    "category": category,
                    "importance": importance,
                    "entities": entities,
                    "relations": relations
                })
            
            self.save_to_neo4j(user_id, all_entities, all_relations)
            
            return {
                "id": stored_facts[0].get("id") if stored_facts else None,
                "content": content,
                "facts": stored_facts,
                "entities": all_entities,
                "relations": all_relations,
                "event": "ADD",
                "facts_count": len(stored_facts)
            }
        
        existing_memories = await self.search_related_memories(user_id, fact_contents, limit=5, score_threshold=0.7)
        judgment = await self.update_memory_with_judgment(user_id, fact_contents, existing_memories, content)
        
        memory_operations = judgment.get("memory", [])
        trace_id = judgment.get("trace_id", "")
        result = await self.execute_memory_operations(user_id, memory_operations, existing_memories, metadata)
        
        if trace_id:
            db = SessionLocal()
            try:
                from app.core.database import MemoryJudgment
                judgment_record = db.query(MemoryJudgment).filter(MemoryJudgment.trace_id == trace_id).first()
                if judgment_record:
                    judgment_record.executed_operations = {
                        "added": result["added"],
                        "updated": result["updated"],
                        "deleted": result["deleted"],
                        "stats": result["stats"]
                    }
                    db.commit()
            except Exception as e:
                logger.error(f"Failed to update judgment record with execution results: {e}")
                db.rollback()
            finally:
                db.close()
        
        return {
            "id": result["added"][0].get("id") if result["added"] else None,
            "content": content,
            "event": "PROCESSED",
            "trace_id": trace_id,
            "operations": {
                "added": result["added"],
                "updated": result["updated"],
                "deleted": result["deleted"]
            },
            "stats": result["stats"],
            "extracted_facts": facts
        }
    
    async def search_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            client = self._get_qdrant_client(user_id)
            collection_name = f"{settings.qdrant_collection}_{user_id[:8]}"
            
            query_embedding = await self._get_embedding(query)
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            results = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )
            
            memories = []
            vector_ids = []
            
            for result in results.points:
                vector_ids.append(str(result.id))
                memories.append({
                    "id": str(result.id),
                    "memory": result.payload.get("content", ""),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {}),
                    "entity_names": result.payload.get("entity_names", []),
                    "relations": result.payload.get("relations", []),
                    "category": result.payload.get("category", "fact"),
                    "importance": result.payload.get("importance", "medium")
                })
            
            if vector_ids:
                db = SessionLocal()
                try:
                    fact_records = db.query(Fact).filter(Fact.vector_id.in_(vector_ids)).all()
                    fact_map = {f.vector_id: f for f in fact_records}
                    
                    for memory in memories:
                        fact = fact_map.get(memory["id"])
                        if fact:
                            memory["fact_id"] = fact.id
                            memory["entities"] = fact.entities or []
                            memory["relations"] = fact.relations or []
                except Exception as e:
                    logger.error(f"Failed to query facts: {e}")
                finally:
                    db.close()
            
            return memories
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_graph(self, user_id: str, entity_name: str = None, relation_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.neo4j_driver:
            return []
        
        results = []
        
        with self.neo4j_driver.session() as session:
            if entity_name:
                query = """
                MATCH (e {name: $name, user_id: $user_id})
                OPTIONAL MATCH (e)-[r]->(t)
                OPTIONAL MATCH (s)-[r2]->(e)
                RETURN e.name as entity, labels(e) as types,
                       collect(DISTINCT {target: t.name, relation: type(r)}) as outgoing,
                       collect(DISTINCT {source: s.name, relation: type(r2)}) as incoming
                LIMIT $limit
                """
                result = session.run(query, name=entity_name, user_id=user_id, limit=limit)
            else:
                query = """
                MATCH (e {user_id: $user_id})
                OPTIONAL MATCH (e)-[r]->(t)
                RETURN e.name as entity, labels(e) as types,
                       collect(DISTINCT {target: t.name, relation: type(r)}) as relations
                LIMIT $limit
                """
                result = session.run(query, user_id=user_id, limit=limit)
            
            for record in result:
                results.append(dict(record))
        
        return results
    
    async def get_context_for_query(self, user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
        vector_results = await self.search_memories(user_id, query, limit)
        
        entity_names_set = set()
        for mem in vector_results:
            for name in mem.get("entity_names", []):
                entity_names_set.add(name)
        
        graph_results = []
        for entity_name in list(entity_names_set)[:5]:
            graph_data = self.search_graph(user_id, entity_name=entity_name, limit=5)
            graph_results.extend(graph_data)
        
        return {
            "vector_memories": vector_results,
            "graph_entities": graph_results,
            "extracted_entities": [{"name": name} for name in list(entity_names_set)[:10]]
        }


    
    async def extract_entities_concurrent(self, texts: List[str], user_id: str, concurrency: int = 3) -> List[Dict[str, Any]]:
        if not texts:
            return []
        
        if len(texts) == 1:
            result = await self.extract_entities_and_relations(texts[0], user_id)
            return [result]
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def extract_one(text: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.extract_entities_and_relations(text, user_id)
        
        results = await asyncio.gather(*[extract_one(text) for text in texts])
        return list(results)
    
    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if len(texts) == 1:
            return [await self._get_embedding(texts[0])]
        
        embed_url = getattr(settings, 'embed_base_url', settings.ollama_base_url)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{embed_url}/v1/embeddings",
                headers={"Content-Type": "application/json"},
                json={
                    "model": settings.embed_model,
                    "input": texts
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Batch embedding failed: {response.status_code}")
            
            data = response.json()
            return [item.get("embedding", []) for item in data.get("data", [])]
    
    async def add_memories_batch(self, user_id: str, contents: List[str], metadatas: List[Dict] = None, concurrency: int = 3) -> List[Dict[str, Any]]:
        import uuid
        
        if not contents:
            return []
        
        if len(contents) == 1:
            return [await self.add_memory(user_id, contents[0], metadatas[0] if metadatas else None)]
        
        start_time = asyncio.get_event_loop().time()
        
        extractions = await self.extract_entities_concurrent(contents, user_id, concurrency=concurrency)
        extraction_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Concurrent extraction ({concurrency}): {len(contents)} texts in {extraction_time:.2f}s")
        
        all_entities = []
        all_relations = []
        for extraction in extractions:
            all_entities.extend(extraction.get("entities", []))
            all_relations.extend(extraction.get("relations", []))
        
        self.save_to_neo4j(user_id, all_entities, all_relations)
        neo4j_time = asyncio.get_event_loop().time() - start_time - extraction_time
        logger.info(f"Neo4j batch write: {neo4j_time:.2f}s")
        
        start_embed = asyncio.get_event_loop().time()
        embeddings = await self._get_embeddings_batch(contents)
        embed_time = asyncio.get_event_loop().time() - start_embed
        logger.info(f"Batch embedding: {len(contents)} texts in {embed_time:.2f}s")
        
        memory_ids = [str(uuid.uuid4()) for _ in contents]
        points = []
        for i, (content, embedding, memory_id) in enumerate(zip(contents, embeddings, memory_ids)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            entities_i = extractions[i].get("entities", [])
            relations_i = extractions[i].get("relations", [])
            entity_names = [e.get("name", "") for e in entities_i if e.get("name")]
            relation_list = [f"{r.get('source','')}-{r.get('relation','')}-{r.get('target','')}" for r in relations_i]
            points.append(PointStruct(
                id=memory_id,
                vector=embedding,
                payload={
                    "content": content,
                    "user_id": user_id,
                    "metadata": metadata,
                    "entity_names": entity_names,
                    "relations": relation_list
                }
            ))
        
        client = self._get_qdrant_client(user_id)
        collection_name = f"{settings.qdrant_collection}_{user_id[:8]}"
        client.upsert(collection_name=collection_name, points=points)
        qdrant_time = asyncio.get_event_loop().time() - start_time - extraction_time - neo4j_time - embed_time
        logger.info(f"Qdrant batch write: {qdrant_time:.2f}s")
        
        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Batch add {len(contents)} memories: {total_time:.2f}s ({len(contents)/total_time:.1f} mem/s)")
        
        return [
            {
                "id": memory_id,
                "content": content,
                "entities": extractions[i].get("entities", []),
                "relations": extractions[i].get("relations", []),
                "event": "ADD"
            }
            for i, (content, memory_id) in enumerate(zip(contents, memory_ids))
        ]


graph_memory_service = GraphMemoryService()
