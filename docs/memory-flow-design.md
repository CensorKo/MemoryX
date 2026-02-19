# MemoryX 记忆流程设计文档

## 一、服务架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           记忆服务架构                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      API 路由层                                      │   │
│  │                                                                      │   │
│  │  memories.py ──────────→ graph_memory_service.py (自定义实现)        │   │
│  │  conversations.py ─────→ mem0_service.py (mem0库封装)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      存储层                                          │   │
│  │                                                                      │   │
│  │  PostgreSQL (memories, facts) ←→ Qdrant (向量) ←→ Neo4j (图)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 两套服务对比

| 特性 | graph_memory_service | mem0_service |
|------|---------------------|--------------|
| 文件位置 | `services/memory_core/graph_memory_service.py` | `services/memory_core/mem0_service.py` |
| 实现方式 | 手动 LLM + Prompt | mem0 库封装 |
| 依赖 | 无 function calling | 依赖 function calling |
| 稳定性 | ⭐⭐⭐ 稳定 | ⭐⭐ 依赖库 |
| 使用路由 | `/memories/*` | `/conversations/*` |
| 推荐使用 | ✅ 推荐 | 兼容保留 |

---

## 二、mem0 源码深度分析

### 2.1 mem0 核心架构

mem0 的核心代码位于 `/Users/censor/Downloads/mem0/mem0/memory/` 目录：

```
mem0/memory/
├── main.py           # Memory 主类 (核心入口)
├── graph_memory.py   # MemoryGraph 类 (图存储)
├── base.py           # MemoryBase 基类
├── storage/          # 存储层
├── utils.py          # 工具函数
└── telemetry.py      # 遥测
```

### 2.2 mem0 添加记忆流程 (main.py)

```python
# mem0/memory/main.py - add() 方法核心逻辑

def add(self, messages, user_id=None, agent_id=None, run_id=None, metadata=None, infer=True):
    """
    mem0 的 add 方法流程:
    
    1. 参数验证和过滤器构建
       - 必须提供 user_id, agent_id 或 run_id 之一
       - 构建 base_metadata_template 和 effective_query_filters
    
    2. 并行执行 (ThreadPoolExecutor)
       ├── _add_to_vector_store()  # 向量存储
       └── _add_to_graph()         # 图存储
    
    3. 返回结果
       - 如果启用图: {"results": [...], "relations": [...]}
       - 否则: {"results": [...]}
    """
```

**mem0 的 _add_to_vector_store 详细流程：**

```python
# mem0/memory/main.py

def _add_to_vector_store(self, messages, metadata, filters, infer):
    if not infer:
        # 直接存储原始消息
        for message_dict in messages:
            msg_embeddings = self.embedding_model.embed(msg_content, "add")
            mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)
        return returned_memories
    
    # infer=True 时 (默认)
    # Step 1: 解析消息
    parsed_messages = parse_messages(messages)
    
    # Step 2: LLM 提取事实
    system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)
    response = self.llm.generate_response(
        messages=[{"role": "system", "content": system_prompt}, ...],
        response_format={"type": "json_object"}
    )
    new_retrieved_facts = json.loads(response)["facts"]
    
    # Step 3: 搜索已有记忆
    for fact in new_retrieved_facts:
        embeddings = self.embedding_model.embed(fact, "add")
        existing_memories = self.vector_store.search(
            query_vector=embeddings, 
            filters=filters, 
            limit=5
        )
        retrieved_old_memory.extend(existing_memories)
    
    # Step 4: LLM 判断 ADD/UPDATE/DELETE/NONE
    function_calling_prompt = get_update_memory_messages(...)
    response = self.llm.generate_response(
        messages=[{"role": "user", "content": function_calling_prompt}],
        response_format={"type": "json_object"}
    )
    new_memories_with_actions = json.loads(response)
    
    # Step 5: 执行操作
    for resp in new_memories_with_actions.get("memory", []):
        if event_type == "ADD":
            memory_id = self._create_memory(...)
        elif event_type == "UPDATE":
            self._update_memory(...)
        elif event_type == "DELETE":
            self._delete_memory(...)
    
    return returned_memories
```

### 2.3 mem0 图存储流程 (graph_memory.py)

```python
# mem0/memory/graph_memory.py - MemoryGraph 类

class MemoryGraph:
    def __init__(self, config):
        self.graph = Neo4jGraph(...)  # langchain_neo4j
        self.embedding_model = EmbedderFactory.create(...)
        self.llm = LlmFactory.create(...)
        self.threshold = 0.7  # 相似度阈值
    
    def add(self, data, filters):
        """
        mem0 图添加流程:
        
        1. _retrieve_nodes_from_data(data, filters)
           - 使用 LLM + function calling 提取实体
           - Tool: EXTRACT_ENTITIES_TOOL
           - 返回: {"实体名": "实体类型", ...}
        
        2. _establish_nodes_relations_from_data(data, filters, entity_type_map)
           - 使用 LLM + function calling 提取关系
           - Tool: RELATIONS_TOOL
           - 返回: [{source, relationship, destination}, ...]
        
        3. _search_graph_db(node_list, filters)
           - 向量搜索相似节点
           - Cypher: vector.similarity.cosine(n.embedding, $n_embedding)
           - 返回已有关系
        
        4. _get_delete_entities_from_search_output(search_output, data, filters)
           - LLM 判断需要删除的关系
           - Tool: DELETE_MEMORY_TOOL_GRAPH
        
        5. _delete_entities() + _add_entities()
           - 执行 Cypher 删除和添加
        """
        
    def _retrieve_nodes_from_data(self, data, filters):
        """使用 function calling 提取实体"""
        search_results = self.llm.generate_response(
            messages=[
                {"role": "system", "content": "You are a smart assistant..."},
                {"role": "user", "content": data}
            ],
            tools=[EXTRACT_ENTITIES_TOOL]  # function calling
        )
        # 解析 tool_calls 获取实体
        entity_type_map = {}
        for tool_call in search_results["tool_calls"]:
            for item in tool_call["arguments"]["entities"]:
                entity_type_map[item["entity"]] = item["entity_type"]
        return entity_type_map
```

### 2.4 mem0 的 Function Calling Tools

mem0 使用 OpenAI function calling 格式定义工具：

```python
# mem0/graphs/tools.py

EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "entity_type": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}

RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "Establish relationships among the entities.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "properties": {
                            "source": {"type": "string"},
                            "relationship": {"type": "string"},
                            "destination": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}
```

### 2.5 mem0 的 Prompt 设计

```python
# mem0/configs/prompts.py

FACT_RETRIEVAL_PROMPT = """You are a Personal Information Organizer...

Types of Information to Remember:
1. Store Personal Preferences
2. Maintain Important Personal Details
3. Track Plans and Intentions
4. Remember Activity and Service Preferences
5. Monitor Health and Wellness Preferences
6. Store Professional Details
7. Miscellaneous Information Management

Few shot examples:
Input: Hi, my name is John. I am a software engineer.
Output: {"facts": ["Name is John", "Is a Software engineer"]}
"""

DEFAULT_UPDATE_MEMORY_PROMPT = """You are a smart memory manager...

Compare newly retrieved facts with the existing memory. For each new fact:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change
"""
```

---

## 三、自定义实现 vs mem0 源码对比

### 3.1 架构对比

| 维度 | graph_memory_service (自定义) | mem0 (源码) |
|------|------------------------------|-------------|
| **LLM 调用方式** | 直接 HTTP POST 到 OpenAI 兼容 API | LlmFactory 抽象层 |
| **实体提取** | Prompt + JSON 解析 | Function Calling + tool_calls |
| **关系提取** | Prompt + JSON 解析 | Function Calling + tool_calls |
| **记忆判断** | 独立 Qwen3-14B-SFT 模型 | 同一 LLM 模型 |
| **向量存储** | 直接 Qdrant Client | VectorStoreFactory 抽象 |
| **图存储** | 直接 Neo4j Driver | langchain_neo4j 封装 |
| **并发处理** | asyncio + Semaphore | ThreadPoolExecutor |

### 3.2 核心差异分析

#### 差异 1: LLM 调用方式

**mem0 方式 (依赖 function calling):**
```python
# mem0 使用 OpenAI function calling
response = self.llm.generate_response(
    messages=[...],
    tools=[EXTRACT_ENTITIES_TOOL]  # 需要 LLM 支持 function calling
)
entities = response["tool_calls"][0]["arguments"]["entities"]
```

**自定义方式 (纯 Prompt):**
```python
# 直接使用 Prompt + JSON 解析
messages = [
    {"role": "system", "content": "你是一个实体提取助手..."},
    {"role": "user", "content": EXTRACT_ENTITIES_PROMPT.format(text=text)}
]
response = await self._call_llm(messages)
result = json.loads(extract_json(response))  # 手动解析 JSON
```

**优劣对比:**
- mem0 方式: 结构化输出更可靠，但依赖 LLM 支持 function calling
- 自定义方式: 兼容性更好，任何 OpenAI 兼容 API 都可用，但 JSON 解析可能失败

#### 差异 2: 记忆判断逻辑

**mem0 方式:**
```python
# mem0 在 _add_to_vector_store 中统一处理
# 使用同一个 LLM 判断 ADD/UPDATE/DELETE/NONE
function_calling_prompt = get_update_memory_messages(
    existing_memories, new_facts
)
response = self.llm.generate_response(
    messages=[{"role": "user", "content": function_calling_prompt}],
    response_format={"type": "json_object"}
)
```

**自定义方式:**
```python
# 分离的记忆判断服务
# 使用专门的 Qwen3-14B-SFT 模型
async def judge_memory_operation(self, new_memory, existing_memories):
    messages = [
        {"role": "system", "content": "你是一个记忆管理助手..."},
        {"role": "user", "content": JUDGE_MEMORY_PROMPT.format(...)}
    ]
    response = await self._call_qwen(messages)  # 专用模型
    return json.loads(response)
```

**优劣对比:**
- mem0 方式: 统一模型，配置简单，但可能不够精准
- 自定义方式: 专用模型更精准，但需要维护多个模型

#### 差异 3: 图存储实现

**mem0 方式:**
```python
# mem0 使用 langchain_neo4j
from langchain_neo4j import Neo4jGraph
self.graph = Neo4jGraph(url, username, password)
self.graph.query(cypher_query, params=params)
```

**自定义方式:**
```python
# 直接使用 neo4j driver
from neo4j import GraphDatabase
self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
with self.neo4j_driver.session() as session:
    session.run(query, **params)
```

**优劣对比:**
- mem0 方式: langchain 封装更高级，但引入额外依赖
- 自定义方式: 更轻量，直接控制，但需要手写 Cypher

### 3.3 数据流对比

**mem0 数据流:**
```
messages → parse_messages() → LLM (facts) → 向量搜索已有记忆
         → LLM (判断 ADD/UPDATE/DELETE) → 执行操作
         → 并行: 向量存储 + 图存储
```

**自定义数据流 (已优化为向量语义搜索):**
```
content → LLM (facts) → 对每个 fact:
         → BGE-M3 向量化 → Qdrant 语义搜索相关记忆 (score_threshold=0.7)
         → Qwen (判断 ADD/UPDATE/DELETE/NONE) → LLM (实体/关系)
         → Qdrant (向量) + PostgreSQL (facts) + Neo4j (图)
```

**关键优化点:**
1. **向量语义搜索**: 用新事实的向量在 Qdrant 中搜索相似记忆，而非数据库全量查询
2. **相似度阈值**: score_threshold=0.7，过滤掉不相关的记忆
3. **减少 LLM 输入**: 只将相关记忆传给 LLM 判断，降低成本
4. **提高准确性**: 语义相似的记忆更可能是需要 UPDATE/DELETE 的目标

### 3.4 关键代码对比

#### 实体提取对比

**mem0 (graph_memory.py):**
```python
def _retrieve_nodes_from_data(self, data, filters):
    search_results = self.llm.generate_response(
        messages=[
            {"role": "system", "content": f"You are a smart assistant..."},
            {"role": "user", "content": data}
        ],
        tools=[EXTRACT_ENTITIES_TOOL]
    )
    entity_type_map = {}
    for tool_call in search_results["tool_calls"]:
        if tool_call["name"] != "extract_entities":
            continue
        for item in tool_call["arguments"]["entities"]:
            entity_type_map[item["entity"]] = item["entity_type"]
    return entity_type_map
```

**自定义 (graph_memory_service.py):**
```python
async def extract_entities_and_relations(self, text: str, user_id: str):
    messages = [
        {"role": "system", "content": "你是一个专业的实体关系提取助手..."},
        {"role": "user", "content": EXTRACT_ENTITIES_PROMPT.format(text=text)}
    ]
    response = await self._call_llm(messages)
    
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    json_str = response[json_start:json_end]
    result = json.loads(json_str)
    
    # 替换"我"为 user_id
    for entity in result.get("entities", []):
        if entity.get("name") in ["我", "本人", "自己"]:
            entity["name"] = user_id
    return result
```

#### 记忆判断对比

**mem0 (main.py):**
```python
# 在 _add_to_vector_store 中
function_calling_prompt = get_update_memory_messages(
    retrieved_old_memory, new_retrieved_facts, user_id
)
response = self.llm.generate_response(
    messages=[{"role": "user", "content": function_calling_prompt}],
    response_format={"type": "json_object"}
)
new_memories_with_actions = json.loads(response)

for resp in new_memories_with_actions.get("memory", []):
    if resp.get("event") == "ADD":
        self._create_memory(...)
    elif resp.get("event") == "UPDATE":
        self._update_memory(...)
```

**自定义 (graph_memory_service.py) - 已优化为向量语义搜索:**
```python
async def search_related_memories(self, user_id: str, new_facts: List[str], limit: int = 5, score_threshold: float = 0.7) -> List[Dict]:
    """
    基于向量语义搜索获取相关记忆
    
    1. 用 BGE-M3 对新事实进行向量化
    2. 在 Qdrant 中搜索相似记忆 (score_threshold=0.7)
    3. 去重并按相似度排序返回
    """
    embeddings = await self._get_embeddings_batch(new_facts)
    
    all_memories = {}
    for embedding in embeddings:
        results = client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=limit,
            score_threshold=score_threshold,  # 过滤低相似度结果
            query_filter=Filter(must=[...])
        )
        for point in results.points:
            all_memories[point.id] = {
                "id": point.id,
                "text": point.payload.get("content", ""),
                "score": point.score,
                ...
            }
    
    return list(all_memories.values())

async def update_memory_with_judgment(self, user_id: str, new_facts: List[str], existing_memories: List[Dict], input_content: str = "") -> Dict[str, Any]:
    """
    使用 Qwen3-14B-SFT 判断记忆操作
    
    existing_memories 来自向量语义搜索，而非数据库全量查询
    """
    prompt = get_memory_update_messages(existing_memories, new_facts)
    messages = [
        {"role": "system", "content": "你是一个智能记忆管理器..."},
        {"role": "user", "content": prompt}
    ]
    response = await self._call_qwen(messages)  # 专用 Qwen 模型
    result = json.loads(extract_json(response))
    return result
```

---

## 四、结论与建议

### 4.1 为什么选择自定义实现

1. **兼容性更好**: 不依赖 function calling，任何 OpenAI 兼容 API 都可用
2. **更灵活**: 可以针对不同任务使用不同模型 (Llama 提取, Qwen 判断)
3. **更轻量**: 不需要 langchain_neo4j 等额外依赖
4. **更可控**: 直接控制 Cypher 查询，便于优化

### 4.2 mem0 的优势

1. **开箱即用**: 配置简单，快速集成
2. **结构化输出**: function calling 输出更可靠
3. **社区支持**: 持续更新，bug 修复

### 4.3 最终建议

**推荐使用 graph_memory_service (自定义实现)**，原因：
- 已针对私有化部署优化 (Llama3.1-8B + Qwen3-14B-SFT)
- 不依赖 function calling，兼容性更好
- 已实现完整的向量 + 图 + 关系数据库存储
- 保留 mem0_service 作为兼容层

---

## 二、核心数据流

### 2.1 添加记忆流程 (graph_memory_service) - 已优化为向量语义搜索

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         add_memory(content, user_id)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 创建原始记忆记录 (PostgreSQL)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ INSERT INTO memories (content, user_id, meta)                       │   │
│  │ → 获得 memory_db_id                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 2: LLM 提取事实 (Llama3.1-8B)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ extract_facts(content)                                               │   │
│  │                                                                      │   │
│  │ 输入: "张三在北京阿里云工作，他喜欢喝咖啡"                            │   │
│  │ 输出: [                                                              │   │
│  │   {"content": "张三在北京工作", "category": "fact"},                 │   │
│  │   {"content": "张三在阿里云工作", "category": "fact"},               │   │
│  │   {"content": "张三喜欢喝咖啡", "category": "preference"}            │   │
│  │ ]                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 3: 向量语义搜索相关记忆 (BGE-M3 + Qdrant) ★ 关键优化                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ search_related_memories(user_id, facts, limit=5, score_threshold=0.7)│   │
│  │                                                                      │   │
│  │ 1. _get_embeddings_batch(facts) → 批量向量化                         │   │
│  │ 2. qdrant.query_points(embedding, limit=5, score_threshold=0.7)     │   │
│  │ 3. 去重并按相似度排序返回相关记忆                                     │   │
│  │                                                                      │   │
│  │ 输出: [                                                              │   │
│  │   {"id": "abc123", "text": "张三在北京工作", "score": 0.95},         │   │
│  │   {"id": "def456", "text": "张三喜欢喝咖啡", "score": 0.88}          │   │
│  │ ]                                                                    │   │
│  │                                                                      │   │
│  │ ★ 只返回语义相关的记忆，过滤掉不相关的，减少 LLM 输入长度            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 4: LLM 判断记忆操作 (Qwen3-14B-SFT)                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ update_memory_with_judgment(user_id, facts, related_memories)       │   │
│  │                                                                      │   │
│  │ 输入: 新事实 + 相关记忆（来自向量搜索，而非数据库全量查询）           │   │
│  │ 输出: [                                                              │   │
│  │   {"id": "abc123", "text": "张三在上海工作", "event": "UPDATE",      │   │
│  │    "old_memory": "张三在北京工作", "reason": "工作地点变更"},         │   │
│  │   {"id": "new1", "text": "张三在腾讯工作", "event": "ADD",           │   │
│  │    "reason": "新的工作单位信息"}                                     │   │
│  │ ]                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 5: 执行记忆操作                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ execute_memory_operations(user_id, operations)                       │   │
│  │                                                                      │   │
│  │ for op in operations:                                                │   │
│  │   if op.event == "ADD":                                              │   │
│  │     - extract_entities_and_relations() → Llama3.1-8B                │   │
│  │     - save_to_qdrant() → BGE-M3 向量化 + Qdrant 存储                │   │
│  │     - save_to_postgres() → PostgreSQL 存储                           │   │
│  │     - save_to_neo4j() → Neo4j 图存储                                 │   │
│  │   elif op.event == "UPDATE":                                         │   │
│  │     - 差量更新实体和关系                                             │   │
│  │     - 更新向量、数据库、图                                           │   │
│  │   elif op.event == "DELETE":                                         │   │
│  │     - 删除向量、图关系、孤立实体、数据库记录                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键优化说明:**

| 优化点 | 旧方式 | 新方式 |
|--------|--------|--------|
| 获取已有记忆 | PostgreSQL 全量查询 | Qdrant 向量语义搜索 |
| 数据量 | 用户所有记忆 | 只返回相关记忆 (limit=5) |
| 过滤机制 | 无 | score_threshold=0.7 |
| LLM 输入长度 | 随记忆量增长 | 固定在合理范围 |
| 准确性 | 不相关记忆干扰判断 | 只处理语义相关的记忆 |

### 2.2 搜索记忆流程 (graph_memory_service)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     get_context_for_query(query, user_id)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 向量搜索 (Qdrant)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ search_memories(query, user_id)                                      │   │
│  │                                                                      │   │
│  │ 1. _get_embedding(query) → 查询向量                                  │   │
│  │ 2. qdrant.query_points() → 相似向量                                  │   │
│  │ 3. 查询 PostgreSQL facts 表获取完整信息                              │   │
│  │                                                                      │   │
│  │ 返回: [{id, memory, score, entities, relations, fact_id}, ...]      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 2: 图搜索 (Neo4j)                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ for entity_name in entity_names:                                     │   │
│  │   search_graph(user_id, entity_name)                                 │   │
│  │                                                                      │   │
│  │ Cypher:                                                              │   │
│  │ MATCH (e {name: $name, user_id: $user_id})                          │   │
│  │ OPTIONAL MATCH (e)-[r]->(t)   // 出边                                │   │
│  │ OPTIONAL MATCH (s)-[r2]->(e)  // 入边                                │   │
│  │ RETURN e.name, labels(e), outgoing, incoming                        │   │
│  │                                                                      │   │
│  │ 返回: [{entity, types, outgoing, incoming}, ...]                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 3: 组装返回                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ return {                                                             │   │
│  │   vector_memories: [...],    // 向量搜索结果                         │   │
│  │   graph_entities: [...],     // 图上下文                             │   │
│  │   extracted_entities: [...]  // 提取的实体名                         │   │
│  │ }                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、数据库设计

### 3.1 PostgreSQL

#### memories 表 (原始记忆)
```sql
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    project_id INTEGER REFERENCES projects(id),
    cognitive_sector VARCHAR,
    confidence FLOAT,
    meta JSONB DEFAULT '{}'::jsonb,
    embedding_id VARCHAR,  -- 已废弃
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### facts 表 (原子事实) ⭐核心关联表
```sql
CREATE TABLE facts (
    id SERIAL PRIMARY KEY,
    memory_id INTEGER REFERENCES memories(id),
    user_id INTEGER REFERENCES users(id),
    
    content TEXT NOT NULL,              -- 事实内容
    category VARCHAR(50) DEFAULT 'fact', -- fact/preference/plan/experience/opinion
    importance VARCHAR(20) DEFAULT 'medium',
    
    vector_id VARCHAR,                  -- ⭐ 关联 Qdrant 向量ID
    entities JSONB DEFAULT '[]'::jsonb, -- 提取的实体
    relations JSONB DEFAULT '[]'::jsonb, -- 提取的关系
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2 Qdrant (向量库)

```python
collection_name = f"memoryx_{user_id[:8]}"

# 向量维度: 1024 (BGE-M3)
# 距离: COSINE

# Payload 结构
payload = {
    "content": "事实内容",
    "user_id": "用户ID",
    "metadata": {},
    "entity_names": ["实体1", "实体2"],
    "relations": ["实体1-关系-实体2"],
    "category": "fact",
    "importance": "medium"
}
```

### 3.3 Neo4j (图数据库)

```cypher
// 实体节点 (按类型分标签)
(:人物 {name: "张三", user_id: "user_123"})
(:地点 {name: "北京", user_id: "user_123"})
(:组织 {name: "阿里云", user_id: "user_123"})
(:技能 {name: "Python", user_id: "user_123"})
(:物品 {name: "咖啡", user_id: "user_123"})

// 关系 (动词命名)
(:人物)-[:工作于]->(:组织)
(:人物)-[:喜欢]->(:物品)
(:人物)-[:学习]->(:技能)
(:人物)-[:住在]->(:地点)
```

---

## 四、关键代码位置

### 4.1 文件结构

```
api/
├── app/
│   ├── core/
│   │   ├── config.py              # 配置 (服务地址、模型名)
│   │   └── database.py            # 数据库模型 (Memory, Fact)
│   │
│   ├── routers/
│   │   ├── memories.py            # /memories/* 路由
│   │   └── conversations.py       # /conversations/* 路由
│   │
│   └── services/
│       └── memory_core/
│           ├── graph_memory_service.py  # ⭐ 核心服务 (推荐)
│           └── mem0_service.py          # mem0 库封装 (兼容)
```

### 4.2 graph_memory_service.py 关键方法

```python
class GraphMemoryService:
    
    # ========== 添加记忆 ==========
    async def add_memory(self, user_id: str, content: str, metadata: Dict = None, skip_judge: bool = False) -> Dict[str, Any]:
        """
        添加记忆主入口
        流程: 创建记录 → 提取事实 → 对每个事实处理 → 存储图数据
        """
    
    # ========== 搜索记忆 ==========
    async def search_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        向量搜索
        流程: 生成查询向量 → Qdrant搜索 → 查询facts表获取完整信息
        """
    
    async def get_context_for_query(self, user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        搜索入口 (路由调用)
        返回: {vector_memories, graph_entities, extracted_entities}
        """
    
    # ========== LLM 调用 ==========
    async def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """提取事实 - Llama3.1-8B"""
    
    async def extract_entities_and_relations(self, text: str, user_id: str) -> Dict[str, Any]:
        """提取实体和关系 - Llama3.1-8B"""
    
    async def judge_memory_operation(self, new_memory: str, existing_memories: List[str]) -> Dict[str, Any]:
        """记忆去重判断 - Qwen3-14B-SFT"""
    
    # ========== 向量操作 ==========
    async def _get_embedding(self, text: str) -> List[float]:
        """生成向量 - BGE-M3"""
    
    async def save_to_qdrant(self, user_id: str, memory_id: str, content: str, ...):
        """存储向量到 Qdrant"""
    
    # ========== 图操作 ==========
    def save_to_neo4j(self, user_id: str, entities: List[Dict], relations: List[Dict]):
        """存储实体和关系到 Neo4j"""
    
    def search_graph(self, user_id: str, entity_name: str = None, ...) -> List[Dict[str, Any]]:
        """搜索图 - 获取实体的关联关系"""
    
    # ========== 批量操作 ==========
    async def add_memories_batch(self, user_id: str, contents: List[str], ...) -> List[Dict[str, Any]]:
        """批量添加记忆 - 并发处理"""
    
    async def extract_entities_concurrent(self, texts: List[str], user_id: str, ...) -> List[Dict[str, Any]]:
        """并发提取实体"""
```

---

## 五、服务调用配置

### 5.1 服务地址

| 服务 | 地址 | 路径 | 用途 |
|------|------|------|------|
| 统一入口 | 192.168.31.65:11436 | - | NGINX 负载均衡 |
| Llama3.1-8B | 统一入口 | `/v1/chat/completions` | 事实提取、实体提取 |
| Qwen3-14B-SFT | 统一入口 | `/qwen/v1/chat/completions` | 记忆判断 |
| BGE-M3 | 统一入口 | `/v1/embeddings` | 向量生成 |

### 5.2 环境变量 (config.py)

```python
class Settings(BaseSettings):
    # 统一入口
    base_url: str = "http://192.168.31.65:11436"
    
    # LLM - Llama3.1-8B (实体/事实提取)
    ollama_base_url: str = "http://192.168.31.65:11436"
    llm_model: str = "llama3.1-8b"
    
    # Qwen3-14B-SFT (记忆判断)
    qwen_base_url: str = "http://192.168.31.65:11436"
    qwen_model: str = "qwen3-14b-sft"
    
    # Embedding (BGE-M3)
    embed_base_url: str = "http://192.168.31.65:11436"
    embed_model: str = "bge-m3"
    
    # 数据库
    database_url: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    qdrant_host: str
    qdrant_port: int
```

---

## 六、API 接口

### 6.1 添加记忆

```
POST /api/v1/memories
Headers: X-API-Key: {api_key}

Request:
{
  "content": "张三在北京阿里云工作，他喜欢喝咖啡"
}

Response:
{
  "success": true,
  "message": "Memory created successfully",
  "data": {
    "id": "uuid-xxx",
    "content": "张三在北京阿里云工作，他喜欢喝咖啡",
    "facts": [
      {
        "id": "uuid-1",
        "content": "张三在北京工作",
        "category": "fact",
        "entities": [...],
        "relations": [...]
      }
    ],
    "entities": [...],
    "relations": [...],
    "event": "ADD",
    "facts_count": 3
  }
}
```

### 6.2 搜索记忆

```
POST /api/v1/memories/search
Headers: X-API-Key: {api_key}

Request:
{
  "query": "张三喜欢什么",
  "limit": 5
}

Response:
{
  "success": true,
  "data": [
    {
      "id": "uuid-xxx",
      "memory": "张三喜欢喝咖啡",
      "score": 0.90,
      "fact_id": 1,
      "entities": [...],
      "relations": [...]
    }
  ],
  "graph_context": [
    {
      "entity": "张三",
      "types": ["人物"],
      "outgoing": [
        {"target": "咖啡", "relation": "喜欢"},
        {"target": "阿里云", "relation": "工作于"}
      ],
      "incoming": []
    }
  ],
  "extracted_entities": [{"name": "张三"}, {"name": "咖啡"}]
}
```

---

## 七、Prompt 模板

### 7.1 事实提取 (EXTRACT_FACTS_PROMPT)

```
从以下对话中提取所有独立的事实/记忆。

对话内容：
{text}

提取规则：
1. 将复杂句子拆分为简单、独立的原子事实
2. 每个事实应该是一个完整的陈述句
3. 过滤掉问候语、废话、无意义的内容
4. 保留重要信息：偏好、经历、关系、计划、观点等
5. 对事实进行分类：preference(偏好), fact(事实), plan(计划), experience(经历), opinion(观点)

返回 JSON:
{
  "facts": [
    {"content": "事实内容", "category": "分类", "importance": "high/medium/low"}
  ]
}
```

### 7.2 实体关系提取 (EXTRACT_ENTITIES_PROMPT)

```
分析以下文本，提取所有实体和它们之间的关系。

文本：
{text}

返回 JSON:
{
  "entities": [
    {"name": "实体名", "type": "类型", "properties": {}}
  ],
  "relations": [
    {"source": "源实体", "target": "目标实体", "relation": "关系类型"}
  ]
}
```

### 7.3 记忆判断 (JUDGE_MEMORY_PROMPT)

```
已有记忆：
{existing_memories}

新信息：
{new_memory}

判断规则：
1. 新内容与已有记忆无关 -> {"type": "ADD"}
2. 对已有记忆的更新/补充 -> {"type": "UPDATE", "target": "要更新的记忆"}
3. 与已有记忆重复 -> {"type": "IGNORE", "reason": "重复原因"}
```

---

## 八、注意事项

### 8.1 不要修改的核心逻辑

1. **事实提取先于实体提取**
   - 原文 → 事实 → 实体
   - 不要直接在原文上提取实体

2. **向量存储的是事实，不是原文**
   - 每个事实独立存储一个向量
   - 一个原文可能拆分为多个事实

3. **facts 表是核心关联表**
   - vector_id 关联 Qdrant
   - entities/relations 关联 Neo4j

4. **记忆去重使用 Qwen3-14B-SFT**
   - 不要使用规则判断
   - 通过 LLM 判断 ADD/UPDATE/IGNORE

### 8.2 两套服务的选择

| 场景 | 推荐服务 | 原因 |
|------|----------|------|
| 新功能开发 | graph_memory_service | 稳定、可控 |
| 对话处理 | mem0_service | 已有集成 |
| 批量操作 | graph_memory_service | 支持并发 |

---

## 九、更新日志

- 2025-02-20: 创建文档，明确两套服务架构和核心流程
- 2025-02-20: 完善 Prompt 设计，添加 ID 追踪和 DELETE 操作支持，优化多语言支持

---

## 五、Prompt 设计对比

### 5.1 记忆更新 Prompt 对比

#### mem0 (DEFAULT_UPDATE_MEMORY_PROMPT)

```python
DEFAULT_UPDATE_MEMORY_PROMPT = """You are a smart memory manager...
You can perform four operations: (1) add, (2) update, (3) delete, (4) no change.

1. Add: If new information not present...
2. Update: If information is totally different...
3. Delete: If information contradicts...
4. No Change: If information already present...

Example with ID tracking:
Old Memory: [{"id": "0", "text": "User is a software engineer"}]
Retrieved facts: ["Name is John"]
New Memory: {
  "memory": [
    {"id": "0", "text": "User is a software engineer", "event": "NONE"},
    {"id": "1", "text": "Name is John", "event": "ADD"}
  ]
}
"""
```

#### 自定义实现 (MEMORY_UPDATE_PROMPT)

```python
MEMORY_UPDATE_PROMPT = """你是一个智能记忆管理器，负责管理用户的记忆系统。
你可以执行四种操作：(1) ADD 添加新记忆，(2) UPDATE 更新已有记忆，(3) DELETE 删除记忆，(4) NONE 无需操作。

## 操作规则：

### 1. ADD（添加）
如果新事实包含记忆中不存在的新信息，则添加。
示例：
- 已有记忆: [{"id": "0", "text": "用户是软件工程师"}]
- 新事实: ["名字叫张三"]
- 操作结果: {
  "memory": [
    {"id": "0", "text": "用户是软件工程师", "event": "NONE"},
    {"id": "1", "text": "名字叫张三", "event": "ADD"}
  ]
}

### 2. UPDATE（更新）
如果新事实与已有记忆相关但信息不同或更完整，则更新。保持相同ID。

### 3. DELETE（删除）
如果新事实与已有记忆矛盾，则删除。保持相同ID。

### 4. NONE（无操作）
如果新事实与已有记忆相同或已被包含，则不操作。

## 重要提示：
- 检测用户输入的语言，用相同语言记录记忆
- ADD 操作需要生成新的 ID（递增数字）
- UPDATE 和 DELETE 操作必须使用已有记忆的 ID
- 只返回 JSON 格式，不要其他内容
"""
```

#### 对比总结

| 维度 | mem0 | 自定义 |
|------|------|--------|
| 语言 | 英文 | 中文 |
| 操作类型 | ADD/UPDATE/DELETE/NONE | ADD/UPDATE/DELETE/NONE |
| ID追踪 | ✅ 支持 | ✅ 支持 |
| Few-shot | 4个详细示例 | 4个详细示例 |
| 多语言 | 英文为主 | ✅ 自动检测语言 |
| 长度 | ~2000字符 | ~1500字符 |

---

### 5.2 事实提取 Prompt 对比

#### mem0 (FACT_RETRIEVAL_PROMPT)

```python
FACT_RETRIEVAL_PROMPT = """You are a Personal Information Organizer...

Types of Information to Remember:
1. Store Personal Preferences
2. Maintain Important Personal Details
3. Track Plans and Intentions
4. Remember Activity and Service Preferences
5. Monitor Health and Wellness Preferences
6. Store Professional Details
7. Miscellaneous Information Management

Few shot examples:
Input: Hi, my name is John. I am a software engineer.
Output: {"facts": ["Name is John", "Is a Software engineer"]}

- Today's date is 2025-02-20
- Detect the language and record facts in the same language
"""
```

#### 自定义实现 (EXTRACT_FACTS_PROMPT)

```python
EXTRACT_FACTS_PROMPT = """从以下对话中提取所有独立的事实/记忆。

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
输出: {
  "facts": [
    {"content": "张三在北京工作", "category": "fact", "importance": "medium"},
    {"content": "张三在阿里云工作", "category": "fact", "importance": "medium"},
    {"content": "张三喜欢喝咖啡", "category": "preference", "importance": "medium"},
    {"content": "张三最近在学习Python编程", "category": "fact", "importance": "medium"}
  ]
}

示例2（英文）：
输入: "John works at Google in Mountain View. He loves playing tennis on weekends."
输出: {
  "facts": [
    {"content": "John works at Google", "category": "fact", "importance": "medium"},
    {"content": "John works in Mountain View", "category": "fact", "importance": "medium"},
    {"content": "John loves playing tennis on weekends", "category": "preference", "importance": "medium"}
  ]
}

示例3（日文）：
输入: "田中さんは東京に住んでいて、寿司が大好きです。"
输出: {
  "facts": [
    {"content": "田中さんは東京に住んでいる", "category": "fact", "importance": "medium"},
    {"content": "田中さんは寿司が大好き", "category": "preference", "importance": "medium"}
  ]
}
"""
```

#### 对比总结

| 维度 | mem0 | 自定义 |
|------|------|--------|
| 语言 | 英文 | 中文 |
| 输出格式 | 简单字符串列表 | 结构化对象 (content, category, importance) |
| 分类 | 无 | ✅ 5种分类 |
| 重要性 | 无 | ✅ high/medium/low |
| Few-shot | 6个英文示例 | 4个多语言示例 (中/英/日) |
| 多语言 | 检测语言 | ✅ 多语言示例 |

---

### 5.3 实体关系提取 Prompt 对比

#### mem0 (EXTRACT_RELATIONS_PROMPT + Function Calling)

```python
EXTRACT_RELATIONS_PROMPT = """
You are an advanced algorithm designed to extract structured information...

1. Extract only explicitly stated information
2. Establish relationships among the entities provided
3. Use "USER_ID" as source for self-references

Relationships:
- Use consistent, general, and timeless relationship types
- Example: Prefer "professor" over "became_professor"

# 使用 Function Calling 返回结构化数据
tools=[EXTRACT_ENTITIES_TOOL]
"""
```

#### 自定义实现 (EXTRACT_ENTITIES_PROMPT)

```python
EXTRACT_ENTITIES_PROMPT = """分析以下文本，提取所有实体和它们之间的关系。

## 提取规则：
1. 实体类型：人物(person)、地点(location)、组织(organization)、技能(skill)、爱好(hobby)、物品(item)、事件(event)、时间(time)
2. 关系类型用动词或短语表示（如：喜欢、住在、学习、工作于、loves、lives_in、works_at）
3. 检测输入语言，用相同语言记录实体和关系
4. 如果文本中提到"我/I/私"等第一人称，使用 "USER_ID" 作为实体名

## 示例：

示例1（中文）：
输入: "张三在北京阿里云工作，他喜欢喝咖啡"
输出: {
  "entities": [
    {"name": "张三", "type": "person"},
    {"name": "北京", "type": "location"},
    {"name": "阿里云", "type": "organization"},
    {"name": "咖啡", "type": "item"}
  ],
  "relations": [
    {"source": "张三", "target": "北京", "relation": "住在"},
    {"source": "张三", "target": "阿里云", "relation": "工作于"},
    {"source": "张三", "target": "咖啡", "relation": "喜欢"}
  ]
}

示例2（英文）：
输入: "John lives in New York and works at Microsoft."
输出: {
  "entities": [
    {"name": "John", "type": "person"},
    {"name": "New York", "type": "location"},
    {"name": "Microsoft", "type": "organization"}
  ],
  "relations": [
    {"source": "John", "target": "New York", "relation": "lives_in"},
    {"source": "John", "target": "Microsoft", "relation": "works_at"}
  ]
}

示例3（第一人称）：
输入: "我在上海工作，喜欢打篮球"
输出: {
  "entities": [
    {"name": "USER_ID", "type": "person"},
    {"name": "上海", "type": "location"},
    {"name": "篮球", "type": "hobby"}
  ],
  "relations": [
    {"source": "USER_ID", "target": "上海", "relation": "工作于"},
    {"source": "USER_ID", "target": "篮球", "relation": "喜欢"}
  ]
}
"""
```

#### 对比总结

| 维度 | mem0 | 自定义 |
|------|------|--------|
| 语言 | 英文 | 中文 |
| 返回方式 | Function Calling | JSON字符串解析 |
| 实体属性 | 只有 name, type | ✅ 支持自定义 properties |
| 分离处理 | 实体和关系分开提取 | ✅ 一次提取两者 |
| 多语言 | 英文为主 | ✅ 多语言示例 |
| 实体类型 | 未明确分类 | ✅ 8种类型 |

---

### 5.4 Prompt 设计原则

#### 为什么使用中文 Prompt

1. **模型选择**: 后端 LLM (Qwen3-14B-SFT) 是中文擅长的模型
2. **理解更准确**: 中文 Prompt 让中文模型理解更准确
3. **输出更自然**: 生成的记忆内容更符合中文表达习惯

#### 多语言支持策略

1. **自动检测**: Prompt 中明确要求检测用户输入语言
2. **同语言记录**: 用检测到的语言记录事实和实体
3. **Few-shot 示例**: 提供中/英/日多语言示例引导模型

#### ID 追踪机制

```python
# 记忆结构
{
  "id": "0",           # 记忆ID
  "text": "张三在北京工作",  # 记忆内容
  "event": "UPDATE",   # 操作类型
  "old_memory": "张三在北京"  # 仅UPDATE时需要
}

# ID 生成规则
- ADD: 生成新ID (递增数字)
- UPDATE: 保持原有ID
- DELETE: 保持原有ID
- NONE: 保持原有ID
```

#### DELETE 操作实现

```python
async def execute_memory_operations(self, user_id, memory_operations, existing_memories, metadata):
    for op in memory_operations:
        if op["event"] == "DELETE":
            existing = next((m for m in existing_memories if m["id"] == op["id"]), None)
            if existing:
                # 1. 获取要删除的实体和关系
                fact_record = db.query(Fact).filter(Fact.id == existing["fact_id"]).first()
                entities_to_delete = fact_record.entities or []
                relations_to_delete = fact_record.relations or []
                
                # 2. 删除向量 (Qdrant)
                self.delete_from_qdrant(user_id, existing["vector_id"])
                
                # 3. 删除图数据 (Neo4j)
                if relations_to_delete:
                    self.delete_from_neo4j(user_id, entities_to_delete, relations_to_delete)
                
                # 4. 删除数据库记录 (PostgreSQL)
                db.delete(fact_record)
```

#### DELETE 完整清理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DELETE 操作清理流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 获取要删除的数据                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ fact_record = db.query(Fact).filter(Fact.id == fact_id).first()     │   │
│  │ entities_to_delete = fact_record.entities                           │   │
│  │ relations_to_delete = fact_record.relations                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 2: 删除向量数据 (Qdrant)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ client.delete(                                                       │   │
│  │     collection_name=collection_name,                                 │   │
│  │     points_selector=PointIdsList(points=[vector_id])                │   │
│  │ )                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 3: 删除图关系 (Neo4j)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ for relation in relations_to_delete:                                 │   │
│  │     MATCH (s)-[r:RELATION_TYPE]->(t)                                 │   │
│  │     DELETE r                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 4: 检查并删除孤立实体 (Neo4j)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ for entity in entities_to_delete:                                    │   │
│  │     # 检查实体是否还有其他关系                                        │   │
│  │     MATCH (e)-[r]-()                                                 │   │
│  │     WITH count(r) as rel_count                                       │   │
│  │     IF rel_count == 0:                                               │   │
│  │         DELETE e  # 只有孤立实体才删除                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 5: 删除数据库记录 (PostgreSQL)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ db.delete(fact_record)                                               │   │
│  │ db.commit()                                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 实体删除的安全检查

```python
def delete_from_neo4j(self, user_id, entities, relations):
    # 1. 先删除关系
    for relation in relations:
        MATCH (s)-[r:RELATION_TYPE]->(t)
        DELETE r
    
    # 2. 检查实体是否孤立
    for entity in entities:
        # 查询实体剩余关系数
        MATCH (e {name: $name, user_id: $user_id})
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as rel_count
        RETURN rel_count
        
        # 只有孤立实体才删除
        if rel_count == 0:
            MATCH (e {name: $name})
            DELETE e
        else:
            # 实体被其他记忆引用，保留
            logger.debug(f"Skipped entity: {name} (has {rel_count} relations)")
```

#### 为什么需要检查孤立实体

```
场景：多个记忆引用同一实体

记忆1: "张三在北京工作" → 实体: 张三, 北京
记忆2: "张三喜欢喝咖啡" → 实体: 张三, 咖啡

删除记忆1时：
- 删除关系: 张三 --工作于--> 北京
- 检查张三: 还有 1 个关系 (喜欢咖啡)，不删除
- 检查北京: 有 0 个关系，删除

结果：
- 张三实体保留（被记忆2引用）
- 北京实体删除（孤立）
```

---

### 5.5 UPDATE 操作完整流程

#### UPDATE 清理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UPDATE 操作完整流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 获取旧数据                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ old_entities = existing["entities"]                                 │   │
│  │ old_relations = existing["relations"]                               │   │
│  │ old_content = existing["text"]                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 2: 提取新数据                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ extraction = extract_entities_and_relations(new_text)               │   │
│  │ new_entities = extraction["entities"]                               │   │
│  │ new_relations = extraction["relations"]                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 3: 更新向量数据 (Qdrant)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # 覆盖原有向量                                                       │   │
│  │ save_to_qdrant(user_id, vector_id, new_text, new_entities, ...)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 4: 计算图变更 (差量更新)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # 实体变更                                                           │   │
│  │ entities_to_remove = old_entity_names - new_entity_names            │   │
│  │ entities_to_add = new_entity_names - old_entity_names               │   │
│  │ entities_to_update = old_entity_names & new_entity_names            │   │
│  │                                                                      │   │
│  │ # 关系变更                                                           │   │
│  │ relations_to_remove = old_relation_set - new_relation_set           │   │
│  │ relations_to_add = new_relation_set - old_relation_set              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 5: 执行图更新 (Neo4j)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # 1. 删除不再需要的关系                                               │   │
│  │ for relation in relations_to_remove:                                 │   │
│  │     MATCH (s)-[r:RELATION]->(t) DELETE r                             │   │
│  │                                                                      │   │
│  │ # 2. 清理孤立实体                                                     │   │
│  │ for entity in entities_to_remove:                                    │   │
│  │     if count((e)-[r]-()) == 0: DELETE e                             │   │
│  │                                                                      │   │
│  │ # 3. 更新/添加实体                                                    │   │
│  │ for entity in new_entities:                                          │   │
│  │     MERGE (e) SET e += properties, e.updated_at = datetime()        │   │
│  │                                                                      │   │
│  │ # 4. 添加新关系                                                       │   │
│  │ for relation in relations_to_add:                                    │   │
│  │     MATCH (s), (t) MERGE (s)-[r:RELATION]->(t)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 6: 更新数据库记录 (PostgreSQL)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ fact_record.content = new_text                                       │   │
│  │ fact_record.entities = new_entities                                  │   │
│  │ fact_record.relations = new_relations                                │   │
│  │ db.commit()                                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 差量更新示例

```
场景：记忆内容从 "张三在北京阿里云工作" 更新为 "张三在上海腾讯工作"

旧数据：
- entities: [张三, 北京, 阿里云]
- relations: [张三-工作于-北京, 张三-工作于-阿里云]

新数据：
- entities: [张三, 上海, 腾讯]
- relations: [张三-工作于-上海, 张三-工作于-腾讯]

差量计算：
- entities_to_remove: [北京, 阿里云]
- entities_to_add: [上海, 腾讯]
- entities_to_update: [张三]
- relations_to_remove: [张三-工作于-北京, 张三-工作于-阿里云]
- relations_to_add: [张三-工作于-上海, 张三-工作于-腾讯]

执行结果：
1. 删除关系: 张三 --工作于--> 北京
2. 删除关系: 张三 --工作于--> 阿里云
3. 检查北京: 孤立，删除实体
4. 检查阿里云: 孤立，删除实体
5. 更新实体: 张三 (updated_at = now)
6. 创建实体: 上海
7. 创建实体: 腾讯
8. 创建关系: 张三 --工作于--> 上海
9. 创建关系: 张三 --工作于--> 腾讯
```

#### update_neo4j_entities 方法

```python
def update_neo4j_entities(self, user_id, old_entities, new_entities, old_relations, new_relations):
    # 1. 计算实体差量
    old_entity_names = {e.get("name") for e in old_entities}
    new_entity_names = {e.get("name") for e in new_entities}
    
    entities_to_remove = old_entity_names - new_entity_names  # 需要删除
    entities_to_add = new_entity_names - old_entity_names     # 需要添加
    entities_to_update = old_entity_names & new_entity_names  # 需要更新
    
    # 2. 计算关系差量
    old_relation_set = {(r["source"], r["relation"], r["target"]) for r in old_relations}
    new_relation_set = {(r["source"], r["relation"], r["target"]) for r in new_relations}
    
    relations_to_remove = old_relation_set - new_relation_set
    relations_to_add = new_relation_set - old_relation_set
    
    # 3. 执行变更
    with session:
        # 删除旧关系
        for source, relation, target in relations_to_remove:
            MATCH (s)-[r:RELATION]->(t) DELETE r
        
        # 清理孤立实体
        for entity_name in entities_to_remove:
            if count((e)-[r]-()) == 0:
                DELETE e
        
        # 更新/创建实体
        for entity in new_entities:
            MERGE (e) SET e += properties, e.updated_at = datetime()
        
        # 创建新关系
        for source, relation, target in relations_to_add:
            MATCH (s), (t) MERGE (s)-[r:RELATION]->(t)
    
    # 4. 返回变更记录
    return {
        "entities_removed": list(entities_to_remove),
        "entities_added": list(entities_to_add),
        "entities_updated": list(entities_to_update),
        "relations_removed": [...],
        "relations_added": [...]
    }
```

#### UPDATE vs DELETE 对比

| 操作 | 向量 (Qdrant) | 图 (Neo4j) | 数据库 (PostgreSQL) |
|------|--------------|------------|---------------------|
| ADD | 创建新向量 | 创建实体+关系 | 创建 Fact 记录 |
| UPDATE | 覆盖原向量 | 差量更新实体+关系 | 更新 Fact 记录 |
| DELETE | 删除向量 | 删除关系+孤立实体 | 删除 Fact 记录 |

---

### 5.6 LLM 判断追踪机制

#### 为什么需要追踪 LLM 判断

```
问题场景：
1. LLM 判断错误导致记忆数据不一致
2. 无法追溯为什么某个记忆被 ADD/UPDATE/DELETE
3. 缺乏 LLM 准确率统计，难以优化 Prompt
4. 出现问题时无法定位原因

解决方案：
- 每次记忆判断生成唯一 trace_id
- 存储完整的输入、输出、推理过程
- 记录执行结果和错误信息
- 支持后续人工审核和统计分析
```

#### MemoryJudgment 数据库表结构

```sql
CREATE TABLE memory_judgments (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(36) UNIQUE NOT NULL,      -- UUID 追踪ID
    user_id INTEGER NOT NULL REFERENCES users(id),
    
    -- 输入信息
    operation_type VARCHAR(20) NOT NULL,        -- MEMORY_UPDATE
    input_content TEXT NOT NULL,                -- 用户原始输入
    extracted_facts JSONB,                      -- LLM 提取的事实
    existing_memories JSONB,                    -- 已有记忆列表
    
    -- LLM 输出
    llm_response TEXT NOT NULL,                 -- LLM 原始响应
    parsed_operations JSONB,                    -- 解析后的操作列表
    reasoning TEXT,                             -- 推理理由汇总
    
    -- 执行结果
    executed_operations JSONB,                  -- 实际执行的操作
    execution_success BOOLEAN DEFAULT TRUE,     -- 是否执行成功
    error_message TEXT,                         -- 错误信息
    
    -- 元信息
    model_name VARCHAR(100),                    -- 使用的模型
    latency_ms INTEGER,                         -- 响应延迟(ms)
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- 审核字段
    is_verified BOOLEAN DEFAULT FALSE,          -- 是否已审核
    verified_at TIMESTAMP,                      -- 审核时间
    verification_result VARCHAR(20),            -- CORRECT/INCORRECT/PARTIAL
    verification_notes TEXT                     -- 审核备注
);

CREATE INDEX idx_memory_judgments_trace_id ON memory_judgments(trace_id);
CREATE INDEX idx_memory_judgments_user_id ON memory_judgments(user_id);
CREATE INDEX idx_memory_judgments_created_at ON memory_judgments(created_at);
```

#### 追踪流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM 判断追踪流程                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 生成 trace_id                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ trace_id = uuid.uuid4()  # 例如: "a1b2c3d4-e5f6-7890-abcd-ef123456" │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 2: 调用 LLM 判断                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ prompt = get_memory_update_messages(existing_memories, new_facts)   │   │
│  │ llm_response = await self._call_qwen(messages)                      │   │
│  │ parsed_operations = json.loads(extract_json(llm_response))          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 3: 存储判断记录                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ judgment_record = MemoryJudgment(                                   │   │
│  │     trace_id=trace_id,                                              │   │
│  │     user_id=user_id,                                                │   │
│  │     input_content=input_content,                                    │   │
│  │     extracted_facts=new_facts,                                      │   │
│  │     existing_memories=existing_memories,                            │   │
│  │     llm_response=llm_response,                                      │   │
│  │     parsed_operations=parsed_operations,                            │   │
│  │     reasoning=reasoning,                                            │   │
│  │     model_name="qwen3-14b-sft",                                     │   │
│  │     latency_ms=latency_ms                                           │   │
│  │ )                                                                   │   │
│  │ db.add(judgment_record)                                             │   │
│  │ db.commit()                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Step 4: 执行操作并更新记录                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ execution_result = execute_memory_operations(...)                   │   │
│  │ judgment_record.executed_operations = execution_result              │   │
│  │ judgment_record.execution_success = True/False                      │   │
│  │ db.commit()                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Prompt 要求包含推理理由

```python
# 记忆更新 Prompt 要求
MEMORY_UPDATE_PROMPT = """
分析新事实并决定对记忆的操作。

对于每个操作，必须提供 reason 字段说明理由：
- ADD: 为什么这是新记忆？
- UPDATE: 为什么需要更新？旧记忆有什么问题？
- DELETE: 为什么需要删除？是否已过时或错误？
- NONE: 为什么不需要操作？

返回格式：
{
  "memory": [
    {
      "id": "0",
      "text": "张三在上海工作",
      "event": "UPDATE",
      "old_memory": "张三在北京工作",
      "reason": "用户明确表示工作地点从北京变更到上海，需要更新工作地点信息"
    },
    {
      "id": "1",
      "text": "张三喜欢喝咖啡",
      "event": "ADD",
      "reason": "这是新的个人偏好信息，之前没有记录"
    }
  ]
}
"""
```

#### 追踪记录示例

```json
{
  "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef123456",
  "user_id": 1,
  "operation_type": "MEMORY_UPDATE",
  "input_content": "我现在在上海腾讯工作，之前在北京阿里云",
  "extracted_facts": [
    "张三在上海腾讯工作",
    "张三之前在北京阿里云工作"
  ],
  "existing_memories": [
    {"id": "0", "text": "张三在北京工作"},
    {"id": "1", "text": "张三在阿里云工作"}
  ],
  "llm_response": "{\"memory\": [...]}",
  "parsed_operations": [
    {
      "id": "0",
      "text": "张三在上海工作",
      "event": "UPDATE",
      "old_memory": "张三在北京工作",
      "reason": "工作地点从北京变更为上海"
    },
    {
      "id": "1",
      "text": "张三在腾讯工作",
      "event": "UPDATE",
      "old_memory": "张三在阿里云工作",
      "reason": "工作单位从阿里云变更为腾讯"
    }
  ],
  "reasoning": "工作地点从北京变更为上海\n工作单位从阿里云变更为腾讯",
  "executed_operations": {
    "entities_updated": ["张三"],
    "relations_removed": ["张三-工作于-北京", "张三-工作于-阿里云"],
    "relations_added": ["张三-工作于-上海", "张三-工作于-腾讯"]
  },
  "execution_success": true,
  "model_name": "qwen3-14b-sft",
  "latency_ms": 350,
  "created_at": "2025-02-20T10:30:00Z"
}
```

#### 使用场景

1. **问题排查**: 通过 trace_id 查询完整的判断过程
```python
# 查询某个追踪记录
record = db.query(MemoryJudgment).filter(
    MemoryJudgment.trace_id == "a1b2c3d4-e5f6-7890-abcd-ef123456"
).first()

print(f"输入: {record.input_content}")
print(f"LLM推理: {record.reasoning}")
print(f"执行结果: {record.executed_operations}")
```

2. **准确率统计**: 定期分析 LLM 判断准确性
```python
# 统计已审核记录的准确率
verified_records = db.query(MemoryJudgment).filter(
    MemoryJudgment.is_verified == True
).all()

correct_count = sum(1 for r in verified_records if r.verification_result == "CORRECT")
accuracy = correct_count / len(verified_records) * 100
```

3. **性能监控**: 监控 LLM 响应延迟
```python
# 查询最近一小时的平均延迟
recent_records = db.query(MemoryJudgment).filter(
    MemoryJudgment.created_at > datetime.now() - timedelta(hours=1)
).all()

avg_latency = sum(r.latency_ms for r in recent_records) / len(recent_records)
```

---

## 六、TODO 项目

### 6.1 LLM 准确率统计程序 (后期实现)

```python
# TODO: 定期统计校对 LLM 的准确率
# 
# 功能需求:
# 1. 定期扫描未审核的判断记录
# 2. 自动对比 LLM 判断与实际执行结果
# 3. 识别异常判断 (如 ADD 后立即 DELETE)
# 4. 生成准确率报告
# 5. 支持人工审核接口
#
# 实现思路:
# - 使用 APScheduler 定时任务
# - 分析 executed_operations 与 parsed_operations 的一致性
# - 检测记忆生命周期异常 (短时间内多次变更)
# - 输出统计报告到文件或数据库
#
# 接口设计:
# GET /api/judgments/stats         - 获取准确率统计
# GET /api/judgments/{trace_id}    - 查询追踪记录
# POST /api/judgments/{trace_id}/verify - 人工审核
```

### 6.2 其他待实现功能

| 功能 | 优先级 | 说明 |
|------|--------|------|
| 记忆回滚 | 高 | 根据 trace_id 回滚记忆操作 |
| 延迟告警 | 中 | LLM 响应超过阈值时告警 |
| 自动清理 | 低 | 定期清理过期的判断记录 |
| 批量审核 | 中 | 支持批量审核判断记录 |
| 准确率看板 | 低 | 可视化展示 LLM 准确率趋势 |

#### 关键差异

| 维度 | UPDATE | DELETE |
|------|--------|--------|
| 向量处理 | 覆盖 (upsert) | 删除 |
| 实体处理 | 差量更新 | 检查孤立后删除 |
| 关系处理 | 差量更新 | 全部删除 |
| 变更记录 | 详细记录增删改 | 记录删除内容 |
