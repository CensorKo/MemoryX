# Pre-import email_validator to fix pydantic issue
import email_validator

# MemoryX API - Cognitive Memory System
# Trigger build: 2026-02-15

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session
import httpx
import hashlib
import os
import logging

from app.core.config import get_settings
from app.core.database import engine, Base, get_db
from app.core.database import APIKey
from app.routers import auth, api_keys, memories, projects, stats
from app.routers import conversations
from app.routers.otp import router as otp_router
from app.routers.firebase_auth import router as firebase_router
from app.routers.agent_autoregister import router as agent_router
from app.routers.agent_claim import router as claim_router
from app.core.celery_config import celery_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

# 创建数据库表
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    logger.info("Starting up MemoryX API...")
    yield
    logger.info("Shutting down MemoryX API...")


app = FastAPI(
    title=settings.app_name,
    description="MemoryX - Free Cognitive Memory API with Queue",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(auth.router, prefix="/api")
app.include_router(api_keys.router, prefix="/api")
app.include_router(memories.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(otp_router, prefix="/api")
app.include_router(firebase_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(claim_router, prefix="/api")

# Mount static files (optional - only if directories exist)
static_dirs = {
    "/portal": "/app/static/portal",
    "/docs": "/app/static",
    "/admin": "/app/static/admin",
    "/doc": "/app/static/doc",
}

for route, directory in static_dirs.items():
    if os.path.exists(directory):
        app.mount(route, StaticFiles(directory=directory, html=True), name=route.split("/")[1] if route != "/docs" else "docs")

# Static pages
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    landing_html_path = "/app/static/index.html"
    if os.path.exists(landing_html_path):
        with open(landing_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>MemoryX</h1><p>Coming soon...</p>")

@app.get("/privacy.html", response_class=HTMLResponse)
async def privacy_page():
    privacy_html_path = "/app/static/privacy.html"
    if os.path.exists(privacy_html_path):
        with open(privacy_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Privacy Policy</h1><p>Coming soon...</p>")

@app.get("/terms.html", response_class=HTMLResponse)
async def terms_page():
    terms_html_path = "/app/static/terms.html"
    if os.path.exists(terms_html_path):
        with open(terms_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Terms of Service</h1><p>Coming soon...</p>")

@app.get("/agent-register", response_class=HTMLResponse)
async def agent_register_guide():
    guide_html_path = "/app/static/agent-register-guide.html"
    if os.path.exists(guide_html_path):
        with open(guide_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Agent Registration Guide</h1><p>Coming soon...</p>")

@app.get("/admin/my-machines", response_class=HTMLResponse)
async def my_machines_page():
    machines_html_path = "/app/static/my-machines.html"
    if os.path.exists(machines_html_path):
        with open(machines_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>My Machines</h1><p>Coming soon...</p>")

@app.get("/api/health")
def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "ok",
            "database": "ok",
            "qdrant": "configured"  # 将在运行时表示实际状态
        }
    }


@app.get("/api/health/detailed")
def health_check_detailed():
    """详细健康检查端点"""
    services_status = {
        "api": "ok",
        "database": "unknown",
        "qdrant": "unknown",
        "celery": "unknown"
    }
    
    # 检查数据库
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        services_status["database"] = "ok"
    except Exception as e:
        services_status["database"] = f"error: {str(e)}"
    
    # 检查 Qdrant
    try:
        from qdrant_client import QdrantClient
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        collections = client.get_collections()
        services_status["qdrant"] = "ok"
    except Exception as e:
        services_status["qdrant"] = f"error: {str(e)}"
    
    # 检查 Celery
    try:
        # 简单的检查 - 能否访问结果后端
        celery_app.connection().ensure_connection(max_retries=1)
        services_status["celery"] = "ok"
    except Exception as e:
        services_status["celery"] = f"error: {str(e)}"
    
    all_ok = all(status == "ok" for status in services_status.values())
    
    return {
        "status": "healthy" if all_ok else "degraded",
        "version": "1.0.0",
        "services": services_status
    }