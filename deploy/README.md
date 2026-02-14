# MemoryX 部署文档

⚠️ **注意：敏感配置文件已移至私有仓库**

包含以下敏感信息的文件已移至 `t0ken-ai/AgentsOnly` 私有仓库：
- 公网 Nginx 配置（含内网 IP）
- 环境变量（含密码）
- 服务器清单

## 公开仓库中的通用配置

本目录包含可以公开的通用部署配置：

### Nginx
- `nginx/memoryx-internal.conf` - 内网服务器 Nginx 配置（无敏感信息）

### 脚本
- `scripts/deploy-safe.sh` - 安全部署脚本
- `scripts/deploy-docker.sh` - Docker 部署脚本
- `scripts/webhook_server.py` - Webhook 接收器

### Systemd
- `systemd/memoryx-api-docker.service` - API 容器服务
- `systemd/memoryx-webhook.service` - Webhook 服务

## 私有仓库中的敏感配置

在 `t0ken-ai/AgentsOnly` 私有仓库中：

```
AgentsOnly/
├── nginx/
│   └── t0ken-public.conf      # 公网 Nginx（含内网 IP）
├── environments/
│   ├── production.env         # 31.65 环境变量
│   └── test.env               # 31.66 环境变量
├── inventory/
│   └── hosts.ini              # 服务器清单
└── scripts/
    ├── deploy.sh              # 部署脚本
    └── init-server.sh         # 初始化脚本
```

## 快速开始

1. 访问私有仓库获取完整部署指南：
   ```
   https://github.com/t0ken-ai/AgentsOnly
   ```

2. 按照 `AgentsOnly/README.md` 进行服务器初始化

3. 使用 `deploy-safe.sh` 进行应用更新

## 部署架构

```
公网用户 ──▶ 公网 Nginx (AgentsOnly)
              ├── SSL 终止
              ├── GitHub IP 白名单 (/deploy)
              └── 路由到内网
                    ├──▶ 31.65 (生产)
                    └──▶ 31.66 (测试)
```

---
*更多详情见私有仓库文档*
