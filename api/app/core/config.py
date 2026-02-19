"""
MemoryX 配置模块

支持三种配置方式（优先级从高到低）：
1. 环境变量（Docker/K8s 部署推荐）
2. .env 文件
3. 默认值

环境变量命名规则：
- 直接使用字段名大写，如：DATABASE_URL, QDRANT_HOST
- 或使用前缀 MEMORYX_，如：MEMORYX_DATABASE_URL

Docker 部署示例：
  docker run -e DATABASE_URL=postgresql://... -e QDRANT_HOST=qdrant ...
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    database_url: str = "postgresql://memoryx:memoryx123@192.168.31.66:5432/memoryx"
    
    valkey_host: str = "192.168.31.66"
    valkey_port: int = 6379
    valkey_password: Optional[str] = None
    redis_url: str = "redis://192.168.31.66:6379/0"
    
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 10080
    
    qdrant_host: str = "192.168.31.66"
    qdrant_port: int = 6333
    qdrant_collection: str = "memoryx"
    qdrant_api_key: Optional[str] = None
    
    neo4j_host: str = "192.168.31.66"
    neo4j_http_port: int = 7474
    neo4j_bolt_port: int = 7687
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memoryx123"
    neo4j_uri: str = "bolt://192.168.31.66:7687"
    
    base_url: str = "http://192.168.31.65:11436"
    
    ollama_base_url: str = "http://192.168.31.65:11436"
    llm_model: str = "llama3.1-8b"
    
    qwen_base_url: str = "http://192.168.31.65:11436"
    qwen_model: str = "qwen3-14b-sft"
    
    embed_base_url: str = "http://192.168.31.65:11436"
    embed_model: str = "bge-m3"
    
    app_name: str = "MemoryX API"
    debug: bool = False
    
    memoryx_master_key: Optional[str] = None
    
    firebase_api_key: Optional[str] = None
    firebase_auth_domain: Optional[str] = None
    firebase_project_id: Optional[str] = None
    firebase_storage_bucket: Optional[str] = None
    firebase_messaging_sender_id: Optional[str] = None
    firebase_app_id: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    return Settings()
