from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/openmemoryx"
    
    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 10080
    
    # Vector Store (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "mem0"
    
    # Ollama (for AI classification)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemma3-27b-q8"
    embed_model: str = "bge-m3"
    
    # App settings
    app_name: str = "MemoryX API"
    debug: bool = False
    
    # Optional: Encryption
    memoryx_master_key: Optional[str] = None
    
    # Firebase Auth Configuration (optional, for client-side auth)
    firebase_api_key: Optional[str] = None
    firebase_auth_domain: Optional[str] = None
    firebase_project_id: Optional[str] = None
    firebase_storage_bucket: Optional[str] = None
    firebase_messaging_sender_id: Optional[str] = None
    firebase_app_id: Optional[str] = None
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
