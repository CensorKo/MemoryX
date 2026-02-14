# Pre-import email_validator to fix pydantic issue
import email_validator

from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session
import httpx
import hashlib
import os
from app.core.config import get_settings
from app.core.database import engine, Base, get_db
from app.core.database import APIKey
from app.routers import auth, api_keys, memories, projects, stats
from app.routers.otp import router as otp_router
from app.routers.firebase_auth import router as firebase_router
from app.routers.agent_autoregister import router as agent_router
from app.routers.agent_claim import router as claim_router
from app.core.celery_config import celery_app

settings = get_settings()

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    description="MemoryX - Free Cognitive Memory API with Queue",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
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
app.include_router(projects.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(otp_router, prefix="/api")
app.include_router(firebase_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(claim_router, prefix="/api")

# Mount static files
app.mount("/portal", StaticFiles(directory="/app/static/portal", html=True), name="portal")
app.mount("/docs", StaticFiles(directory="/app/static", html=True), name="docs")
app.mount("/admin", StaticFiles(directory="/app/static/admin", html=True), name="admin")
app.mount("/doc", StaticFiles(directory="/app/static/doc", html=True), name="doc")

# Static pages
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    landing_html_path = "/app/static/openmemoryx-landing.html"
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
    return {"status": "healthy", "version": "1.0.0"}
