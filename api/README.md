# MemoryX API Server

HTTP API 服务器，提供记忆存储和查询功能。

## 快速开始

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API 端点

### 记忆管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/memories` | 添加记忆 |
| GET | `/api/memories` | 列出记忆 |
| POST | `/api/memories/search` | 搜索记忆 |
| DELETE | `/api/memories/{id}` | 删除记忆 |

### Agent 自动注册

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/agents/auto-register` | 自动注册机器 |
| POST | `/api/agents/claim/init` | 初始化认领 |

## 环境变量

```env
DATABASE_URL=postgresql://user:pass@host/db
REDIS_URL=redis://host:6379/0
SECRET_KEY=your-secret-key
OLLAMA_HOST=http://host:11434
```

## Docker 构建

```bash
docker build -f Dockerfile.api -t memoryx-api .
docker run -p 8000:8000 memoryx-api
```

Last updated: Sun Feb 15 00:15:42 CST 2026

Dependencies fixed: Sun Feb 15 00:18:39 CST 2026

Trigger: Sun Feb 15 00:21:16 CST 2026

Debug: Sun Feb 15 00:22:54 CST 2026

Fix static files: Sun Feb 15 00:24:14 CST 2026
Sun Feb 15 00:32:47 CST 2026
Test Sun Feb 15 00:36:22 CST 2026

