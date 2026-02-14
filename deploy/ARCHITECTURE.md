# MemoryX 网络架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              公网服务器                                   │
│                         Nginx (SSL 终止)                                 │
│                         t0ken.ai:443                                     │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  /api/*    ──────▶ 192.168.31.65:8000 (Docker API)             │   │
│  │  /api/docs ──────▶ 192.168.31.65:8000 (Swagger UI)              │   │
│  │  /portal/* ──────▶ 192.168.31.65:80  (内网 Nginx Portal)       │   │
│  │  /*        ──────▶ 192.168.31.65:80  (内网 Nginx 官网)         │   │
│  │  /deploy   ──────▶ 192.168.31.65:9000 (Webhook)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WireGuard/OpenVPN
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         内网服务器 (192.168.31.65)                        │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Nginx :80                                                        │ │
│  │   ├── /portal/*  → /data/memoryx/static/portal/index.html        │ │
│  │   └── /*         → /data/memoryx/static/index.html               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Docker Container (memoryx-api)                                   │ │
│  │   - Port 8000                                                     │ │
│  │   - /api/* 路由                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Webhook Server :9000                                             │ │
│  │   - 接收 GitHub Actions 部署通知                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## 访问地址

### 公网访问 (https://t0ken.ai)

| 路径 | 说明 | 后端服务 |
|------|------|----------|
| `/` | 官网首页 | 内网 Nginx (192.168.31.65:80) |
| `/portal` | 用户 Portal | 内网 Nginx (192.168.31.65:80) |
| `/privacy.html` | 隐私政策 | 内网 Nginx (192.168.31.65:80) |
| `/api/*` | API 接口 | Docker (192.168.31.65:8000) |
| `/api/docs` | Swagger 文档 | Docker (192.168.31.65:8000) |
| `/deploy` | Webhook 接口 | Webhook Server (192.168.31.65:9000) |

## 配置文件位置

### 公网服务器
```
/etc/nginx/sites-available/t0ken-public.conf
```

### 内网服务器 (31.65)
```
/etc/nginx/sites-available/memoryx-internal.conf
```

## 部署步骤

### 1. 公网服务器配置

```bash
# 复制配置
sudo cp deploy/nginx/t0ken-public.conf /etc/nginx/sites-available/t0ken
sudo ln -sf /etc/nginx/sites-available/t0ken /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# 检查并重载
sudo nginx -t
sudo systemctl reload nginx
```

### 2. 内网服务器 (31.65) 配置

```bash
# 复制配置
sudo cp deploy/nginx/memoryx-internal.conf /etc/nginx/sites-available/memoryx
sudo ln -sf /etc/nginx/sites-available/memoryx /etc/nginx/sites-enabled/

# 检查并重载
sudo nginx -t
sudo systemctl reload nginx
```

### 3. 测试访问

```bash
# 从内网测试
curl http://192.168.31.65:8000/health
curl http://192.168.31.65/
curl http://192.168.31.65/portal

# 从公网测试
curl https://t0ken.ai/health
curl https://t0ken.ai/api/health
curl https://t0ken.ai/api/docs
```

## 故障排查

### 公网无法访问内网

1. 检查 VPN/隧道连接
```bash
# 公网服务器
ping 192.168.31.65
curl http://192.168.31.65:8000/health
```

2. 检查防火墙
```bash
# 内网服务器 (31.65)
sudo iptables -L -n | grep 8000
sudo iptables -L -n | grep 80
```

### 路径问题

```bash
# 检查内网 nginx 日志
sudo tail -f /var/log/nginx/memoryx-error.log

# 检查公网 nginx 日志
sudo tail -f /var/log/nginx/t0ken-error.log
```
