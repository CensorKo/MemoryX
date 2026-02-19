# MemoryX LLM 部署方案

## 硬件环境

### LLM 服务器 (192.168.31.10)
- **GPU**: 3x NVIDIA V100 (32GB)
- **用途**: MemoryX 记忆服务的 LLM 推理

### Embedding 服务器 (192.168.31.65)
- **GPU**: 2x NVIDIA Tesla P40 (24GB)
- **用途**: BGE-M3 向量生成

## Embedding 服务 (BGE-M3)

### 双 GPU 负载均衡架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                 BGE-M3 双 P40 负载均衡                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌─────────────────┐                              │
│                    │     NGINX       │                              │
│                    │   Port: 11436   │                              │
│                    │   负载均衡       │                              │
│                    └────────┬────────┘                              │
│                             │                                       │
│           ┌─────────────────┼─────────────────┐                    │
│           │                 │                 │                    │
│           ▼                 ▼                 ▸                    │
│    ┌────────────┐    ┌────────────┐                               │
│    │  GPU 0     │    │  GPU 1     │                               │
│    │ P40 24GB   │    │ P40 24GB   │                               │
│    │ Port:11434 │    │ Port:11435 │                               │
│    │ bge-m3 Q8  │    │ bge-m3 Q8  │                               │
│    └────────────┘    └────────────┘                               │
│                                                                     │
│    显存: 1079MB      显存: 1079MB                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 性能测试

| 模式 | 10次请求耗时 | QPS | 提升 |
|------|-------------|-----|------|
| 单实例 GPU0 串行 | 1.768s | 5.7/s | - |
| 单实例 GPU1 串行 | 1.813s | 5.5/s | - |
| **负载均衡 并发** | **0.650s** | **15.4/s** | **2.75x** |

### 配置文件

**bge-ha.yaml** (`/data/ollama_models/bge-ha.yaml`)

```yaml
services:
  ollama-bge-0:
    image: ollama/ollama:latest
    container_name: ollama_bge_0
    ports:
      - "11434:11434"
    volumes:
      - /data/ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    environment:
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_KEEP_ALIVE=-1
      - OLLAMA_NUM_PARALLEL=4

  ollama-bge-1:
    image: ollama/ollama:latest
    container_name: ollama_bge_1
    ports:
      - "11435:11434"
    volumes:
      - /data/ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    environment:
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_KEEP_ALIVE=-1
      - OLLAMA_NUM_PARALLEL=4

  nginx:
    image: nginx:alpine
    container_name: bge_nginx
    ports:
      - "11436:80"
    volumes:
      - ./nginx-bge.conf:/etc/nginx/nginx.conf:ro
```

### 使用方式

```python
# 直连单个实例
embed_url = "http://192.168.31.65:11434"  # GPU0
embed_url = "http://192.168.31.65:11435"  # GPU1

# 负载均衡 (推荐)
embed_url = "http://192.168.31.65:11436"  # NGINX
```

## LLM 模型对比测试

### 测试模型

| 模型 | 量化 | 显存占用 | 端口 |
|------|------|----------|------|
| Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 | INT4 | 5.4 GB | 11401 |
| Qwen2.5-14B-Instruct-GPTQ-Int8 | INT8 | 12 GB | 11402 |

### 速度对比

| 任务 | Llama3.1-8B | Qwen2.5-14B | 差距 |
|------|-------------|-------------|------|
| 中文实体提取 (5次平均) | 0.57s | 1.13s | Llama 快 2x |
| 多语言处理 | 0.3-0.4s | 0.9-2.2s | Llama 快 2-5x |
| 复杂实体关系提取 | 2.94s | 9.51s | Llama 快 3.2x |

### 准确度对比

| 任务 | Llama3.1-8B | Qwen2.5-14B |
|------|-------------|-------------|
| 多语言支持 | 100% (8语言) | 100% (8语言) |
| 实体提取 | 100% | 100% |
| 关系提取 | 100% | 100% |
| 事实提取 | 100% | 100% |
| 记忆更新判断 | 40% | 90% |
| JSON稳定性 (默认) | 0% | 80% |
| JSON稳定性 (强制) | 100% | 100% |

### 综合评估

| 模型 | 综合得分 | 评级 |
|------|----------|------|
| Llama3.1-8B-Int4 | 83% | ✓ 完全胜任 |
| Qwen2.5-14B-Int8 | 85% | ✓ 完全胜任 |

## 推荐方案: 3 GPU 高并发部署

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    3 GPU 高并发部署 (5实例)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌─────────────────┐                              │
│                    │     NGINX       │                              │
│                    │   Port: 11434   │                              │
│                    │   负载均衡       │                              │
│                    └────────┬────────┘                              │
│                             │                                       │
│       ┌─────────────────────┼─────────────────────┐                │
│       │                     │                     │                │
│       ▼                     ▼                     ▼                │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │   GPU 0     │    │     GPU 1       │    │     GPU 2       │    │
│  │ 2x Llama 8B │    │   2x Llama 8B   │    │   1x Qwen 14B   │    │
│  │ Port:11401  │    │  Port:11403     │    │   Port:11405    │    │
│  │ Port:11402  │    │  Port:11404     │    │                 │    │
│  │ 实体提取    │    │   实体提取      │    │   逻辑推理      │    │
│  └─────────────┘    └─────────────────┘    └─────────────────┘    │
│                                                                     │
│  显存: 11GB/GPU      显存: 11GB/GPU        显存: 16.5GB            │
│  实例: 2x            实例: 2x              实例: 1x                 │
│  总计: 4x Llama (实体提取) + 1x Qwen (逻辑推理) = 5 实例           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 任务分配

| GPU | 模型 | 实例数 | 用途 | 端口 |
|-----|------|--------|------|------|
| GPU 0 | Llama3.1-8B-INT4 | 2 | 实体提取 | 11401, 11402 |
| GPU 1 | Llama3.1-8B-INT4 | 2 | 实体提取 | 11403, 11404 |
| GPU 2 | Qwen2.5-14B-INT8 | 1 | 逻辑推理 | 11405 |

### 性能指标

| 指标 | 旧方案 (3实例) | 新方案 (5实例) | 提升 |
|------|---------------|---------------|------|
| Llama 实例数 | 2 | 4 | 2x |
| 实体提取并发 | 2 | 4 | 2x |
| Qwen 上下文 | 8K | 16K | 2x |
| Qwen 并发请求 | 16 | 32 | 2x |
| 显存利用率 | ~60% | ~90% | 1.5x |
| 队列消除速度 | 3x | 5-6x | 2x |

### 显存分析

| GPU | 模型 | 单实例显存 | 实例数 | 总显存 | 可用显存 | 状态 |
|-----|------|-----------|--------|--------|----------|------|
| GPU 0 | Llama3.1-8B-INT4 | ~5.5GB | 2 | ~11GB | 32GB | ✓ 充足 |
| GPU 1 | Llama3.1-8B-INT4 | ~5.5GB | 2 | ~11GB | 32GB | ✓ 充足 |
| GPU 2 | Qwen2.5-14B-INT8 | ~16GB | 1 | ~16GB | 32GB | ✓ 够用 |
| GPU 2 | Qwen2.5-14B-INT8 | ~16GB | 2 | ~32GB | 32GB | ✗ OOM |

> **注意**: Qwen2.5-14B-INT8 单实例需要约 16GB 显存，GPU2 (32GB) 只能跑 1 个实例。如果要跑 2 个实例需要 32GB+ 显存。

### vLLM 调优参数

**Llama3.1-8B (每GPU 2实例)**
```yaml
--gpu-memory-utilization 0.45  # 每实例 45%，共 90%
--max-model-len 8192           # 8K 上下文
--block-size 64                # KV cache 块大小
--max-num-seqs 16              # 最大并发序列
```

**Qwen2.5-14B (单实例)**
```yaml
--gpu-memory-utilization 0.85  # 85% 显存
--max-model-len 16384          # 16K 上下文 (大文档)
--block-size 128               # 更大的 KV cache 块
--max-num-seqs 32              # 32 并发请求
--enable-prefix-caching        # 前缀缓存 (相同 prompt 加速)
--enable-chunked-prefill       # 分块预填充 (大 batch 效率)
```

## Docker Compose 配置

### 文件: memoryx-hybrid.yaml

```yaml
services:
  vllm-llama-gpu0-0:
    image: vllm/vllm-openai:latest
    container_name: vllm_llama_gpu0_0
    restart: always
    environment:
      - NCCL_P2P_DISABLE=1
    ports:
      - "11401:8000"
    volumes:
      - /data/projects/models:/model_data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    command: >
      --model /model_data/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
      --dtype half
      --gpu-memory-utilization 0.45
      --max-model-len 8192
      --block-size 64
      --max-num-seqs 16
      --served-model-name llama3.1-8b
      --trust-remote-code

  vllm-llama-gpu0-1:
    image: vllm/vllm-openai:latest
    container_name: vllm_llama_gpu0_1
    restart: always
    environment:
      - NCCL_P2P_DISABLE=1
    ports:
      - "11402:8000"
    volumes:
      - /data/projects/models:/model_data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    command: >
      --model /model_data/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
      --dtype half
      --gpu-memory-utilization 0.45
      --max-model-len 8192
      --block-size 64
      --max-num-seqs 16
      --served-model-name llama3.1-8b
      --trust-remote-code

  vllm-llama-gpu1-0:
    image: vllm/vllm-openai:latest
    container_name: vllm_llama_gpu1_0
    restart: always
    environment:
      - NCCL_P2P_DISABLE=1
    ports:
      - "11403:8000"
    volumes:
      - /data/projects/models:/model_data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    command: >
      --model /model_data/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
      --dtype half
      --gpu-memory-utilization 0.45
      --max-model-len 8192
      --block-size 64
      --max-num-seqs 16
      --served-model-name llama3.1-8b
      --trust-remote-code

  vllm-llama-gpu1-1:
    image: vllm/vllm-openai:latest
    container_name: vllm_llama_gpu1_1
    restart: always
    environment:
      - NCCL_P2P_DISABLE=1
    ports:
      - "11404:8000"
    volumes:
      - /data/projects/models:/model_data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    command: >
      --model /model_data/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
      --dtype half
      --gpu-memory-utilization 0.45
      --max-model-len 8192
      --block-size 64
      --max-num-seqs 16
      --served-model-name llama3.1-8b
      --trust-remote-code

  vllm-qwen-gpu2:
    image: vllm/vllm-openai:latest
    container_name: vllm_qwen_gpu2
    restart: always
    environment:
      - NCCL_P2P_DISABLE=1
    ports:
      - "11405:8000"
    volumes:
      - /data/projects/models:/model_data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
    command: >
      --model /model_data/Qwen2.5-14B-Instruct-GPTQ-Int8
      --dtype half
      --gpu-memory-utilization 0.85
      --max-model-len 16384
      --block-size 128
      --max-num-seqs 32
      --served-model-name qwen2.5-14b
      --trust-remote-code
      --enable-prefix-caching
      --enable-chunked-prefill

  nginx:
    image: nginx:alpine
    container_name: vllm_nginx
    restart: always
    ports:
      - "11434:80"
    volumes:
      - ./nginx-hybrid.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - vllm-llama-gpu0-0
      - vllm-llama-gpu0-1
      - vllm-llama-gpu1-0
      - vllm-llama-gpu1-1
      - vllm-qwen-gpu2
```

### 文件: nginx-hybrid.conf

```nginx
worker_processes auto;
events {
    worker_connections 2048;
}

http {
    upstream llama_backend {
        least_conn;
        server vllm-llama-gpu0-0:8000 weight=1;
        server vllm-llama-gpu0-1:8000 weight=1;
        server vllm-llama-gpu1-0:8000 weight=1;
        server vllm-llama-gpu1-1:8000 weight=1;
    }

    upstream qwen_backend {
        server vllm-qwen-gpu2:8000;
    }

    server {
        listen 80;
        
        client_max_body_size 100M;
        proxy_read_timeout 300s;
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;

        location / {
            if ($request_body ~* "qwen") {
                proxy_pass http://qwen_backend;
            }
            proxy_pass http://llama_backend;
            
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_buffering off;
        }

        location /qwen/ {
            rewrite ^/qwen/(.*) /$1 break;
            proxy_pass http://qwen_backend;
            proxy_set_header Host $host;
            proxy_buffering off;
        }

        location /health {
            access_log off;
            return 200 "OK";
            add_header Content-Type text/plain;
        }
    }
}
```

## 代码配置

### config.py

```python
# LLM 服务配置
# GPU 0-1: Llama3.1-8B (实体提取)
LLAMA_BASE_URL = "http://192.168.31.10:11401"  # 或通过 NGINX 负载均衡
LLAMA_MODEL = "llama3.1-8b"

# GPU 2: Qwen2.5-14B (记忆判断)
QWEN_BASE_URL = "http://192.168.31.10:11403"
QWEN_MODEL = "qwen2.5-14b"
```

### 任务路由

```python
async def extract_entities(text: str) -> dict:
    """实体提取 → Llama3.1-8B (速度快)"""
    return await call_llm(LLAMA_BASE_URL, LLAMA_MODEL, text)

async def judge_memory_action(existing: str, new_info: str) -> dict:
    """记忆判断 → Qwen2.5-14B (准确度高)"""
    return await call_llm(QWEN_BASE_URL, QWEN_MODEL, prompt)
```

## 搜索流程优化

### 优化前

```
查询 → 向量搜索(~100ms) → LLM实体提取(~500ms) → 图搜索(~10ms) = ~610ms
```

### 优化后

```
存储时: 实体信息存入 Qdrant payload (entity_names)
查询 → 向量搜索(~100ms) → 直接获取实体 → 图搜索(~10ms) = ~110ms
提升: 5.5x
```

## 启动命令

```bash
# 启动混合部署
cd /data/projects/models
docker-compose -f memoryx-hybrid.yaml up -d

# 检查状态
docker ps
curl http://localhost:11401/v1/models  # Llama GPU0
curl http://localhost:11402/v1/models  # Llama GPU1
curl http://localhost:11403/v1/models  # Qwen GPU2

# 查看日志
docker logs vllm_llama_gpu0 --tail 20
docker logs vllm_qwen_gpu2 --tail 20
```

## 总结

| 项目 | 方案 |
|------|------|
| 实体提取 | 4x Llama3.1-8B (GPU 0-1 各2实例) - 4x 并发 |
| 逻辑推理 | 1x Qwen2.5-14B (GPU 2) - 16K上下文 32并发 |
| 搜索优化 | 存储时记录实体，搜索时跳过 LLM - 5.5x |
| 并发处理 | 5 实例并行 - 5-6x |
| 总提升 | 综合性能提升 10-15x |

## 统一入口架构

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    统一入口 (192.168.31.65:11436)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌─────────────────┐                              │
│                    │   NGINX 统一    │                              │
│                    │   Port: 11436   │                              │
│                    │   路由转发       │                              │
│                    └────────┬────────┘                              │
│                             │                                       │
│       ┌─────────────────────┼─────────────────────┐                │
│       │                     │                     │                │
│       ▼                     ▼                     ▼                │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │ /embed/*    │    │ /v1/chat/*      │    │ /qwen/*         │    │
│  │ /v1/embed   │    │ /v1/complete    │    │                 │    │
│  │ /api/embed  │    │                 │    │                 │    │
│  └──────┬──────┘    └────────┬────────┘    └────────┬────────┘    │
│         │                    │                      │              │
│         ▼                    ▼                      ▼              │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │ 31.65 本地  │    │ 31.10 Llama     │    │ 31.10 Qwen      │    │
│  │ BGE-M3 双卡 │    │ 4实例负载均衡    │    │ GPU2            │    │
│  │ QPS: 15.4/s │    │ Port:11401-11404│    │ Port:11405      │    │
│  └─────────────┘    └─────────────────┘    └─────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 路由规则

| 路径 | 目标 | 用途 |
|------|------|------|
| `/v1/embeddings` | BGE-M3 (31.65) | 向量生成 |
| `/api/embeddings` | BGE-M3 (31.65) | 向量生成 |
| `/embed/*` | BGE-M3 (31.65) | 向量生成 |
| `/v1/chat/completions` | Llama (31.10:11401-11404) | 实体提取 |
| `/v1/completions` | Llama (31.10:11401-11404) | 实体提取 |
| `/llama/*` | Llama (31.10:11401-11404) | 实体提取 |
| `/qwen/*` | Qwen (31.10:11405) | 逻辑推理 |

### nginx-unified.conf

```nginx
worker_processes auto;
events {
    worker_connections 2048;
}

http {
    upstream bge_backend {
        least_conn;
        server ollama-bge-0:11434 weight=1;
        server ollama-bge-1:11434 weight=1;
    }
    
    upstream llama_backend {
        least_conn;
        server 192.168.31.10:11401 weight=1;
        server 192.168.31.10:11402 weight=1;
        server 192.168.31.10:11403 weight=1;
        server 192.168.31.10:11404 weight=1;
    }

    upstream qwen_backend {
        server 192.168.31.10:11405;
    }

    server {
        listen 80;
        
        client_max_body_size 100M;
        proxy_read_timeout 300s;
        proxy_connect_timeout 60s;

        location /embed/ {
            rewrite ^/embed/(.*) /$1 break;
            proxy_pass http://bge_backend;
            proxy_buffering off;
        }

        location /v1/embeddings {
            proxy_pass http://bge_backend;
            proxy_buffering off;
        }

        location /api/embeddings {
            proxy_pass http://bge_backend;
            proxy_buffering off;
        }

        location /llama/ {
            rewrite ^/llama/(.*) /$1 break;
            proxy_pass http://llama_backend;
            proxy_buffering off;
        }

        location /qwen/ {
            rewrite ^/qwen/(.*) /$1 break;
            proxy_pass http://qwen_backend;
            proxy_buffering off;
        }

        location /v1/chat/completions {
            proxy_pass http://llama_backend;
            proxy_buffering off;
        }

        location /v1/completions {
            proxy_pass http://llama_backend;
            proxy_buffering off;
        }

        location /health {
            access_log off;
            return 200 "OK";
            add_header Content-Type text/plain;
        }
    }
}
```

### 使用方式

```python
# 统一入口 (推荐)
UNIFIED_URL = "http://192.168.31.65:11436"

# Embedding
response = requests.post(f"{UNIFIED_URL}/v1/embeddings", json={
    "model": "bge-m3",
    "input": "test text"
})

# LLM (自动路由到 Llama)
response = requests.post(f"{UNIFIED_URL}/v1/chat/completions", json={
    "model": "llama3.1-8b",
    "messages": [{"role": "user", "content": "..."}]
})

# Qwen (逻辑推理)
response = requests.post(f"{UNIFIED_URL}/qwen/v1/chat/completions", json={
    "model": "qwen2.5-14b",
    "messages": [{"role": "user", "content": "..."}]
})
```

### 性能测试结果

| 测试项 | 结果 |
|--------|------|
| Embedding (bge-m3) | ✓ 正常 |
| LLM (Llama 负载均衡) | ✓ 正常 |
| Qwen (逻辑推理) | ✓ 正常 |
| 实体提取并发 (10次) | 2.98s (vs 单实例 7.19s) |
| 性能提升 | **4.8x** |

### 完整启动命令

```bash
# === 31.65 (Embedding + 统一入口) ===
ssh root@192.168.31.65
cd /data/ollama_models
docker-compose -f bge-ha.yaml up -d

# 预热 bge-m3
curl -s http://localhost:11434/api/embeddings -d '{"model":"bge-m3","prompt":"test"}'
curl -s http://localhost:11435/api/embeddings -d '{"model":"bge-m3","prompt":"test"}'

# === 31.10 (LLM - 5实例) ===
ssh root@192.168.31.10
cd /data/projects/models
docker-compose -f memoryx-hybrid.yaml up -d

# 等待模型加载 (~3分钟)
sleep 180

# 检查状态
curl http://localhost:11401/v1/models  # Llama GPU0-0
curl http://localhost:11402/v1/models  # Llama GPU0-1
curl http://localhost:11403/v1/models  # Llama GPU1-0
curl http://localhost:11404/v1/models  # Llama GPU1-1
curl http://localhost:11405/v1/models  # Qwen GPU2

# 查看显存
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# === 测试统一入口 ===
curl http://192.168.31.65:11436/health
curl http://192.168.31.65:11436/v1/embeddings -d '{"model":"bge-m3","input":"test"}'
curl http://192.168.31.65:11436/v1/chat/completions -d '{"model":"llama3.1-8b","messages":[{"role":"user","content":"hello"}]}'
curl http://192.168.31.65:11436/qwen/v1/chat/completions -d '{"model":"qwen2.5-14b","messages":[{"role":"user","content":"1+1=?"}]}'
```
