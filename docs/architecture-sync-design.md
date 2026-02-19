# MemoryX 同步架构设计方案

## 一、核心设计原则

1. **服务端统一控制模型**：客户端不配置语义模型，完全由服务端决定
2. **精确 Token 计数**：使用 tokenizer 库精确计算，不估算（解决多语言问题）
3. **增量同步为主**：基于版本号的增量同步，减少数据传输
4. **批量处理对话**：缓冲对话流，在触发点批量提取记忆

---

## 二、对话流缓冲处理

### 2.1 设计目标

- 不逐条处理消息，缓冲后批量处理
- 减少云端 API 调用次数
- 保证记忆提取的上下文完整性

### 2.2 插件端 - ConversationBuffer

```
┌─────────────────────────────────────────────────────────┐
│                   OpenClaw 插件端                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌──────────────────────────────┐   │
│  │ 消息接收    │───▶│ ConversationBuffer           │   │
│  │             │    │ - messages: List[Message]    │   │
│  │ on_message  │    │ - token_count: int (精确)    │   │
│  └─────────────┘    │ - started_at: datetime       │   │
│                     │ - conversation_id: string     │   │
│                     └──────────────┬───────────────┘   │
│                                    │                    │
│         ┌──────────────────────────┼────────────────┐   │
│         ▼                          ▼                ▼   │
│  ┌────────────┐           ┌────────────┐    ┌────────┐ │
│  │ Token阈值  │           │ 对话结束   │    │ 超时   │ │
│  │ >2000      │           │ on_close   │    │ >30min │ │
│  └─────┬──────┘           └─────┬──────┘    └───┬────┘ │
│        │                        │               │      │
│        └────────────────────────┴───────────────┘      │
│                                 │                      │
│                                 ▼                      │
│                     ┌─────────────────────┐            │
│                     │ 批量提交到云端      │            │
│                     │ POST /conversations │            │
│                     │        /flush       │            │
│                     └─────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 2.3 触发条件

| 条件 | 阈值 | 说明 |
|------|------|------|
| Token 阈值 | 2000 tokens | 对话积累到一定量，触发批量处理 |
| 对话结束 | on_conversation_end | 用户关闭窗口/明确结束会话 |
| 时间超时 | 30 分钟无活动 | 防止缓冲无限增长 |

### 2.4 Token 精确计数方案

```typescript
// 插件端使用 js-tiktoken 或调用服务端接口
import { encoding_for_model } from "tiktoken";

class ConversationBuffer {
  private encoder: Tiktoken;
  
  constructor() {
    // 使用 cl100k_base 编码（GPT-4/3.5 通用）
    this.encoder = encoding_for_model("gpt-4");
  }
  
  countTokens(text: string): number {
    return this.encoder.encode(text).length;
  }
  
  addMessage(role: string, content: string): void {
    const tokens = this.countTokens(content);
    this.messages.push({ role, content, tokens, timestamp: Date.now() });
    this.tokenCount += tokens;
    
    if (this.tokenCount >= this.TOKEN_THRESHOLD) {
      this.flush();
    }
  }
}
```

**多语言支持**：
- tiktoken 支持 Unicode，可正确处理中文、日文、韩文等
- 服务端使用同样的 tiktoken 库保持一致性

---

## 三、用户模型版本管理

### 3.1 设计目标

- 客户端不配置模型，完全由服务端控制
- 服务端根据区域/用户群体动态选择最优模型
- 换模型时触发全量重同步

### 3.2 数据库模型扩展

```python
class UserModelConfig(Base):
    """用户模型配置 - 服务端控制"""
    __tablename__ = "user_model_configs"
    
    user_id = Column(String(64), primary_key=True)
    
    # 嵌入模型配置（服务端决定）
    embedding_model = Column(String(100), nullable=False, default="text-embedding-3-small")
    embedding_dimension = Column(Integer, default=1536)
    embedding_provider = Column(String(50), default="openai")  # openai/azure/custom
    
    # 模型版本（换模型时递增）
    model_version = Column(Integer, default=1)
    
    # 区域标记（服务端根据此选择模型）
    region = Column(String(20), nullable=True)  # cn/us/eu/...
    
    # 同步状态
    last_sync_version = Column(Integer, default=0)  # 数据版本
    sync_status = Column(String(20), default="synced")  # synced/resync_required
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_full_sync_at = Column(DateTime, nullable=True)


class MemoryVector(Base):
    """现有模型扩展 - 添加模型绑定"""
    # ... 现有字段 ...
    
    # 新增：模型绑定（此向量由哪个模型生成）
    embedding_model = Column(String(100), nullable=True)
    model_version = Column(Integer, default=1)
    
    # 数据版本（用于增量同步）
    data_version = Column(Integer, default=0)
    is_deleted = Column(Boolean, default=False)  # 软删除
```

### 3.3 服务端模型管理流程

```
┌─────────────────────────────────────────────────────────┐
│                 服务端模型管理                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  场景1: 正常同步                                        │
│  ┌──────────┐    GET /sync?since=v100    ┌──────────┐  │
│  │ 本地插件  │ ─────────────────────────▶ │  云端    │  │
│  │ v=100    │                            │ v=150    │  │
│  │ model=A  │ ◀───────────────────────── │ model=A  │  │
│  └──────────┘    返回增量数据             └──────────┘  │
│                                                         │
│  场景2: 服务端换模型（后台决策）                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. 服务端检测某区域模型表现不佳                    │  │
│  │ 2. 更新 UserModelConfig:                          │  │
│  │    - embedding_model = "new-model"                │  │
│  │    - model_version += 1                           │  │
│  │    - sync_status = "resync_required"              │  │
│  │ 3. 后台任务：用新模型重新向量化所有记忆            │  │
│  │ 4. 完成后：sync_status = "synced"                 │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  场景3: 插件请求同步时发现模型变更                      │
│  ┌──────────┐    GET /sync?model_ver=1   ┌──────────┐  │
│  │ 本地插件  │ ─────────────────────────▶ │  云端    │  │
│  │ model=A  │                            │ model=B  │  │
│  │ ver=1    │ ◀───────────────────────── │ ver=2    │  │
│  └──────────┘    status: full_resync     └──────────┘  │
│                 new_model: B                           │
│                 new_dimension: 3072                    │
│                                                         │
│  场景4: 插件执行全量同步                               │
│  ┌──────────┐    GET /sync/full        ┌──────────┐   │
│  │ 本地插件  │ ───────────────────────▶ │  云端    │   │
│  │ 清空本地  │                          │          │   │
│  │ 拉取全量  │ ◀─────────────────────── │ 返回全部 │   │
│  └──────────┘                          └──────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.4 API 设计

#### 3.4.1 批量提交对话

```
POST /conversations/flush

Request:
{
  "user_id": "user-xxx",
  "conversation_id": "conv-123",
  "messages": [
    {"role": "user", "content": "我喜欢吃寿司", "tokens": 8, "timestamp": 1234567890},
    {"role": "assistant", "content": "好的，我记住了", "tokens": 6, "timestamp": 1234567891}
  ],
  "total_tokens": 14,
  "client_model_version": 1  // 插件当前模型版本
}

Response:
{
  "status": "ok",
  "extracted_count": 2,       // 提取的记忆数
  "server_model_version": 1,  // 服务端模型版本
  "sync_required": false      // 是否需要同步
}

Response (模型变更):
{
  "status": "ok",
  "extracted_count": 2,
  "server_model_version": 2,
  "sync_required": true,
  "sync_type": "full"  // full = 全量同步
}
```

#### 3.4.2 增量同步

```
GET /sync/incremental?user_id=xxx&last_version=100&model_version=1

Response (正常):
{
  "status": "ok",
  "server_model_version": 1,
  "current_version": 150,
  "changes": {
    "vectors": [
      {
        "id": "uuid-1",
        "content": "用户喜欢吃寿司",
        "vector_id": "qdrant-uuid-1",
        "category": "preference",
        "data_version": 101
      }
    ],
    "deleted_ids": ["uuid-old"]
  },
  "graph": {
    "entities": [...],
    "relations": [...]
  }
}

Response (模型变更):
{
  "status": "full_resync_required",
  "reason": "model_changed",
  "server_model_version": 2,
  "new_embedding_model": "text-embedding-3-large",
  "new_dimension": 3072
}
```

#### 3.4.3 全量同步

```
GET /sync/full?user_id=xxx&model_version=2

Response:
{
  "status": "ok",
  "server_model_version": 2,
  "embedding_model": "text-embedding-3-large",
  "dimension": 3072,
  "total_count": 500,
  "vectors": [...],  // 分批返回
  "graph": {
    "entities": [...],
    "relations": [...],
    "communities": [...]
  },
  "has_more": true,
  "next_offset": 100
}
```

---

## 四、同步机制详解

### 4.1 同步触发时机

| 触发点 | 动作 | 说明 |
|--------|------|------|
| 应用启动 | 检查模型版本 | 启动时检查是否需要全量同步 |
| 窗口获得焦点 | 增量同步 | 用户切回窗口时拉取最新 |
| 对话结束后 | 等待 5 秒后同步 | 确保云端处理完成 |
| 定时轮询 | 每 5 分钟增量同步 | 保持数据新鲜度 |

### 4.2 离线处理

```
┌─────────────────────────────────────────────────────────┐
│                   离线队列机制                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  离线时：                                               │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. 对话正常缓冲到 ConversationBuffer           │    │
│  │ 2. 触发 flush 时，写入 IndexedDB 离线队列      │    │
│  │ 3. 本地查询继续使用 Kùzu + LanceDB             │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  恢复在线时：                                           │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. 检测网络恢复                                 │    │
│  │ 2. 按顺序上传离线队列中的对话                   │    │
│  │ 3. 执行增量同步                                 │    │
│  │ 4. 清空离线队列                                 │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.3 冲突处理

**原则**：本地优先，服务端作为备份和权威源

| 场景 | 处理方式 |
|------|----------|
| 本地有，云端无 | 上传本地数据 |
| 云端有，本地无 | 拉取云端数据 |
| 两边都有，内容不同 | 以版本号高的为准（后更新的覆盖） |
| 删除冲突 | 删除操作优先（删除即删除） |

---

## 五、服务端处理流程

### 5.1 对话处理流水线

```
┌─────────────────────────────────────────────────────────┐
│              POST /conversations/flush 处理流程         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐                                       │
│  │ 1. 接收对话  │                                       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ 2. 检查模型  │ ──── 模型变更 ──▶ 返回 full_resync   │
│  │    版本      │                                       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ 3. 敏感信息  │                                       │
│  │    过滤(LLM) │                                       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ 4. LLM 批量  │                                       │
│  │    提取记忆  │                                       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ 5. 向量化    │ ◀─── 使用用户绑定的模型              │
│  │    (Embedding)│                                      │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ 6. 图构建    │                                       │
│  │    (Neo4j)   │                                       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ 7. 更新版本  │                                       │
│  │    data_ver++│                                       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ 8. 返回响应  │                                       │
│  └──────────────┘                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 敏感信息过滤（LLM）

```python
SENSITIVE_FILTER_PROMPT = """
分析以下对话，识别并标记敏感信息：
- 个人身份信息（姓名、身份证、电话、地址）
- 财务信息（银行卡、密码、金额）
- 健康信息（病历、诊断）
- 位置信息（精确地址、GPS）

对话内容：
{conversation}

请返回 JSON 格式：
{
  "has_sensitive": true/false,
  "sensitive_spans": [
    {"start": 10, "end": 20, "type": "phone", "replacement": "[电话号码]"}
  ],
  "safe_content": "过滤后的安全内容"
}
"""
```

### 5.3 换模型后台任务

```python
async def reindex_user_embeddings(user_id: str, new_model: str):
    """用户换模型时，重新向量化所有记忆"""
    
    # 1. 更新配置，标记状态
    config = await get_user_model_config(user_id)
    config.sync_status = "resync_required"
    config.embedding_model = new_model
    config.model_version += 1
    
    # 2. 批量重新向量化
    memories = await get_all_memories(user_id)
    for batch in chunk(memories, 100):
        embeddings = await get_embeddings_batch(
            [m.content for m in batch],
            model=new_model
        )
        await update_vectors(batch, embeddings)
    
    # 3. 更新 Qdrant 中的向量
    await qdrant_reindex(user_id, new_model)
    
    # 4. 标记完成
    config.sync_status = "synced"
    config.last_full_sync_at = datetime.utcnow()
```

---

## 六、插件端架构

### 6.1 模块划分

```typescript
// 插件端核心模块
modules/
├── ConversationBuffer.ts    // 对话缓冲器
├── SyncManager.ts           // 同步管理器
├── OfflineQueue.ts          // 离线队列
├── LocalSearch.ts           // 本地搜索（Kùzu + LanceDB）
└── TokenCounter.ts          // Token 精确计数
```

### 6.2 ConversationBuffer 实现

```typescript
interface Message {
  role: "user" | "assistant";
  content: string;
  tokens: number;
  timestamp: number;
}

interface ConversationBufferConfig {
  tokenThreshold: number;      // 2000
  timeoutMinutes: number;      // 30
  onFlush: (messages: Message[]) => Promise<void>;
}

class ConversationBuffer {
  private messages: Message[] = [];
  private tokenCount: number = 0;
  private startedAt: number = Date.now();
  private timeoutId: NodeJS.Timeout | null = null;
  
  constructor(private config: ConversationBufferConfig) {}
  
  addMessage(role: "user" | "assistant", content: string): void {
    const tokens = countTokens(content);  // 精确计数
    this.messages.push({ role, content, tokens, timestamp: Date.now() });
    this.tokenCount += tokens;
    this.resetTimeout();
    
    if (this.tokenCount >= this.config.tokenThreshold) {
      this.flush();
    }
  }
  
  async flush(): Promise<void> {
    if (this.messages.length === 0) return;
    
    const messages = [...this.messages];
    this.messages = [];
    this.tokenCount = 0;
    
    try {
      await this.config.onFlush(messages);
    } catch (error) {
      // 网络失败，加入离线队列
      await offlineQueue.add(messages);
    }
  }
  
  endConversation(): void {
    this.flush();
    this.clearTimeout();
  }
  
  private resetTimeout(): void {
    if (this.timeoutId) clearTimeout(this.timeoutId);
    this.timeoutId = setTimeout(
      () => this.flush(),
      this.config.timeoutMinutes * 60 * 1000
    );
  }
}
```

### 6.3 SyncManager 实现

```typescript
interface SyncState {
  modelVersion: number;
  lastDataVersion: number;
  embeddingModel: string;
  embeddingDimension: number;
}

class SyncManager {
  private state: SyncState;
  
  async checkSyncRequired(): Promise<{ required: boolean; type: "incremental" | "full" }> {
    const serverInfo = await fetch("/sync/info").then(r => r.json());
    
    if (serverInfo.model_version > this.state.modelVersion) {
      return { required: true, type: "full" };
    }
    
    if (serverInfo.current_version > this.state.lastDataVersion) {
      return { required: true, type: "incremental" };
    }
    
    return { required: false, type: "incremental" };
  }
  
  async incrementalSync(): Promise<void> {
    const response = await fetch(
      `/sync/incremental?since=${this.state.lastDataVersion}`
    );
    const data = await response.json();
    
    if (data.status === "full_resync_required") {
      await this.fullSync(data.server_model_version);
      return;
    }
    
    // 应用增量更新
    await this.applyChanges(data.changes);
    this.state.lastDataVersion = data.current_version;
  }
  
  async fullSync(newModelVersion: number): Promise<void> {
    // 清空本地数据
    await localSearch.clearAll();
    
    // 拉取全量数据
    let offset = 0;
    while (true) {
      const response = await fetch(`/sync/full?offset=${offset}`);
      const data = await response.json();
      
      await localSearch.bulkInsert(data.vectors, data.graph);
      
      if (!data.has_more) break;
      offset = data.next_offset;
    }
    
    // 更新状态
    this.state.modelVersion = newModelVersion;
    this.state.lastDataVersion = data.current_version;
  }
}
```

---

## 七、整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MemoryX 系统架构                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        OpenClaw 插件端                             │ │
│  │                                                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │ │
│  │  │Conversation  │  │ SyncManager  │  │ LocalSearch          │    │ │
│  │  │Buffer        │  │              │  │                      │    │ │
│  │  │              │  │ - 检查同步   │  │ ┌────────┐┌────────┐│    │ │
│  │  │ - Token精确  │  │ - 增量同步   │  │ │ Kùzu   ││LanceDB ││    │ │
│  │  │   计数       │  │ - 全量同步   │  │ │(图存储)││(向量)  ││    │ │
│  │  │ - 触发flush  │  │ - 状态管理   │  │ └────────┘└────────┘│    │ │
│  │  │              │  │              │  │                      │    │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │ │
│  │         │                 │                     │                │ │
│  │  ┌──────┴───────┐         │                     │                │ │
│  │  │OfflineQueue  │         │                     │                │ │
│  │  │(IndexedDB)   │         │                     │                │ │
│  │  └──────────────┘         │                     │                │ │
│  └───────────────────────────┼─────────────────────┼────────────────┘ │
│                              │                     │                  │
│         批量提交              │ 增量/全量同步        │ 本地查询        │
│         ▼                    ▼                     ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        MemoryX 云端                               │ │
│  │                                                                   │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                    API Gateway                              │ │ │
│  │  │  - POST /conversations/flush  (批量处理)                    │ │ │
│  │  │  - GET  /sync/incremental     (增量同步)                    │ │ │
│  │  │  - GET  /sync/full            (全量同步)                    │ │ │
│  │  │  - GET  /sync/info            (同步状态)                    │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                              │                                   │ │
│  │  ┌───────────────────────────┼───────────────────────────────┐   │ │
│  │  │                           ▼                               │   │ │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │   │ │
│  │  │  │ Sensitive    │  │ Memory       │  │ Graph          │  │   │ │
│  │  │  │ Filter (LLM) │─▶│ Extractor    │─▶│ Builder        │  │   │ │
│  │  │  └──────────────┘  └──────────────┘  └────────────────┘  │   │ │
│  │  │                                                │         │   │ │
│  │  │         ┌──────────────────────────────────────┘         │   │ │
│  │  │         ▼                                                │   │ │
│  │  │  ┌────────────────────────────────────────────────────┐ │   │ │
│  │  │  │         UserModelConfig (模型版本管理)             │ │   │ │
│  │  │  │  - 服务端控制模型选择                              │ │   │ │
│  │  │  │  - 区域化模型部署                                  │ │   │ │
│  │  │  │  - 换模型时触发全量重索引                          │ │   │ │
│  │  │  └────────────────────────────────────────────────────┘ │   │ │
│  │  └─────────────────────────────────────────────────────────┘   │ │
│  │                              │                                   │ │
│  │  ┌───────────────────────────┼───────────────────────────────┐  │ │
│  │  │                    存储层                                 │  │ │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │  │ │
│  │  │  │ Neo4j        │  │ Qdrant       │  │ PostgreSQL     │  │  │ │
│  │  │  │ (知识图谱)   │  │ (向量存储)   │  │ (元数据+配置)  │  │  │ │
│  │  │  │              │  │              │  │                │  │  │ │
│  │  │  │ - 实体关系   │  │ - 带 model   │  │ - UserModel    │  │  │ │
│  │  │  │ - 社区结构   │  │   字段       │  │   Config       │  │  │ │
│  │  │  │              │  │ - data_ver   │  │ - MemoryVector │  │  │ │
│  │  │  └──────────────┘  └──────────────┘  └────────────────┘  │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 八、关键决策总结

| 决策点 | 选择 | 原因 |
|--------|------|------|
| Token 计数 | tiktoken 精确计算 | 解决多语言问题 |
| 模型配置 | 完全服务端控制 | 统一管理，避免不一致 |
| 同步策略 | 增量为主，换模型时全量 | 减少数据传输 |
| 触发机制 | Token 阈值 + 对话结束 + 超时 | 兼顾实时性和效率 |
| 离线处理 | IndexedDB 队列 | 保证数据不丢失 |
| 冲突解决 | 版本号优先 | 简单可靠 |

---

## 九、设计决策确认

| 问题 | 决策 | 说明 |
|------|------|------|
| Token 阈值 | 2000 tokens | 合理，不需要动态调整 |
| 超时时间 | 30 分钟 | 合理 |
| 全量同步速率 | ≤ 1MB/s | 考虑用户带宽限制，启用压缩 |
| 模型切换检测 | 预留机制，暂不实现 | 服务端统一控制，后续按需添加 |
| 多设备同步 | 不支持 | 基于机器码隔离，用户需主动绑定 |

### 9.1 多设备处理策略

现有机制已满足需求：
- 每台机器基于硬件指纹（IOPlatformUUID/machine-id/主板UUID）生成唯一 user_id
- 天然隔离，无需额外处理
- 用户可通过 `/agents/claim` 流程主动绑定多设备

```
┌─────────────────────────────────────────────────────────┐
│                   多设备隔离机制                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  设备 A (machine_fingerprint: abc123)                   │
│  └── user_id: hash(abc123)                             │
│      └── 独立的记忆空间                                 │
│                                                         │
│  设备 B (machine_fingerprint: def456)                   │
│  └── user_id: hash(def456)                             │
│      └── 独立的记忆空间                                 │
│                                                         │
│  绑定流程（用户主动触发）                               │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. 设备 A 生成认领码                           │    │
│  │ 2. 用户在后台验证认领码                        │    │
│  │ 3. 设备 A 完成绑定，数据迁移到人类账户         │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 9.2 数据传输优化

```
┌─────────────────────────────────────────────────────────┐
│                   数据传输规范                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  批次大小：固定 200 条/批                               │
│  - 简化客户端逻辑，无需动态调整                         │
│  - 服务端统一处理分页                                   │
│                                                         │
│  压缩策略：                                             │
│  - 上传：客户端 Gzip 压缩后上传                         │
│  - 下载：服务端 Gzip 压缩后返回                         │
│  - 文本数据压缩率通常可达 70-80%                        │
│                                                         │
│  客户端原则：                                           │
│  - 不控制拉取速率，循环请求直到完成                     │
│  - 逻辑简单：请求 → 处理 → 下一批                       │
│                                                         │
│  API 响应：                                             │
│  {                                                      │
│    "batch_size": 200,                                   │
│    "has_more": true,                                    │
│    "next_offset": 200                                   │
│  }                                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 9.3 离线恢复批量上传

```
┌─────────────────────────────────────────────────────────┐
│               离线恢复上传流程                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  恢复在线时：                                           │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. 检测网络恢复                                 │    │
│  │ 2. 从 IndexedDB 读取离线队列                   │    │
│  │ 3. 合并多段对话（按时间排序）                   │    │
│  │ 4. Gzip 压缩                                   │    │
│  │ 5. 批量上传到 /conversations/flush-batch       │    │
│  │ 6. 清空离线队列                                 │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  POST /conversations/flush-batch                        │
│  Content-Encoding: gzip                                 │
│  {                                                      │
│    "user_id": "xxx",                                    │
│    "conversations": [                                   │
│      {"conversation_id": "c1", "messages": [...]},      │
│      {"conversation_id": "c2", "messages": [...]}       │
│    ]                                                    │
│  }                                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 9.4 服务端消息队列

```
┌─────────────────────────────────────────────────────────┐
│              服务端异步处理架构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  问题：大量对话同时提交可能打爆 LLM API                 │
│                                                         │
│  解决方案：消息队列 + 异步处理                          │
│                                                         │
│  ┌─────────┐     ┌─────────────┐     ┌──────────────┐  │
│  │ API     │────▶│ Redis/Celery│────▶│ Worker Pool  │  │
│  │ Gateway │     │ 消息队列    │     │ LLM 处理器   │  │
│  └─────────┘     └─────────────┘     └──────────────┘  │
│       │                                     │          │
│       │  立即返回 OK                        │          │
│       ▼                                     ▼          │
│  ┌─────────────┐                    ┌──────────────┐  │
│  │ 客户端继续  │                    │ 处理完成     │  │
│  │ 无需轮询    │                    │ 更新版本号   │  │
│  └─────────────┘                    └──────────────┘  │
│                                                         │
│  关键设计：                                             │
│  - 客户端不关心处理状态，只关心向量数据库版本           │
│  - 处理完成后自动更新 data_version                      │
│  - 客户端下次同步时自然拿到新数据                       │
│                                                         │
│  处理流程：                                             │
│  1. 接收请求 → 入队 → 立即返回 OK                      │
│  2. Worker 异步处理 → 敏感信息过滤                     │
│  3. LLM 提取记忆 → 生成向量 → 构建图                   │
│  4. 更新 data_version → 客户端下次同步获取             │
│                                                         │
│  并发控制：                                             │
│  - Worker 数量根据 LLM API 限额动态调整               │
│  - 优先级队列：实时请求 > 批量上传                     │
│  - 失败重试：指数退避，最多 3 次                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 9.5 客户端搜索回退机制

```
┌─────────────────────────────────────────────────────────┐
│              本地搜索 → 服务端回退                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  搜索流程：                                             │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. 本地向量搜索 (LanceDB)                      │    │
│  │ 2. 计算置信度分数                              │    │
│  │ 3. 置信度 >= 阈值 → 返回结果                   │    │
│  │ 4. 置信度 < 阈值 → 调用服务端搜索              │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  置信度判断：                                           │
│  - top_k 结果的最高相似度 < 0.7 → 低置信               │
│  - top_k 结果为空 → 低置信                             │
│  - 本地向量库为空 → 直接走服务端                       │
│                                                         │
│  服务端搜索 API：                                       │
│  POST /search                                           │
│  {                                                      │
│    "user_id": "xxx",                                    │
│    "query": "用户偏好",                                 │
│    "limit": 5                                           │
│  }                                                      │
│                                                         │
│  Response:                                              │
│  {                                                      │
│    "results": [...],                                    │
│    "confidence": 0.92,                                  │
│    "source": "cloud"                                    │
│  }                                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 十、后续开发计划

1. **Phase 1**：服务端消息队列
   - Redis/Celery 配置
   - 任务队列定义
   - Worker Pool 实现（LLM 处理）

2. **Phase 2**：服务端 API
   - POST /conversations/flush（入队，立即返回 OK）
   - POST /conversations/flush-batch（批量入队）
   - GET /sync/incremental
   - GET /sync/full
   - POST /search（服务端搜索，低置信回退）

3. **Phase 3**：插件端
   - ConversationBuffer（Token 精确计数）
   - SyncManager（增量/全量同步，无轮询）
   - LocalSearch（LanceDB + 置信度判断）
   - OfflineQueue + 批量上传

4. **Phase 4**：模型切换
   - 后台重索引任务
   - 全量同步触发
