from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/openmemoryx"
    redis_url: str = "redis://localhost:6379/0"
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 10080
    openmemoryx_url: str = "http://localhost:8000"
    app_name: str = "MemoryX API"
    debug: bool = False
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
