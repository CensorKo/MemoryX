from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text, JSON, Float, LargeBinary, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
from enum import Enum

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_timeout=30,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


QUOTA_LIMITS = {
    SubscriptionTier.FREE: {
        "cloud_search_per_month": 50,
        "memories_per_month": 100,
        "batch_upload_per_day": 50,
    },
    SubscriptionTier.PRO: {
        "cloud_search_per_month": -1,
        "memories_per_month": -1,
        "batch_upload_per_day": -1,
    },
    SubscriptionTier.ENTERPRISE: {
        "cloud_search_per_month": -1,
        "memories_per_month": -1,
        "batch_upload_per_day": -1,
    },
}

PRICING = {
    SubscriptionTier.FREE: 0,
    SubscriptionTier.PRO: 9.9,
    SubscriptionTier.ENTERPRISE: 99.0,
}


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    machine_fingerprint = Column(String(64), nullable=True, index=True)
    machine_hash = Column(String(32), nullable=True, index=True)
    is_machine_account = Column(Boolean, default=False)
    last_login = Column(DateTime, nullable=True)
    
    firebase_uid = Column(String, nullable=True)
    
    merged_to_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    subscription_tier = Column(
        SQLEnum(SubscriptionTier),
        default=SubscriptionTier.FREE,
        nullable=False
    )
    subscription_start = Column(DateTime, nullable=True)
    subscription_end = Column(DateTime, nullable=True)
    
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    quota = relationship("UserQuota", back_populates="user", uselist=False, cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    is_machine_default = Column(Boolean, default=False)
    machine_fingerprint = Column(String(64), nullable=True, index=True)
    
    user = relationship("User", back_populates="projects")


class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), default="Default")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    is_auto_generated = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="api_keys")
    project = relationship("Project")


class Memory(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    
    cognitive_sector = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    embedding_id = Column(String, nullable=True, index=True)
    
    facts = relationship("Fact", back_populates="memory", cascade="all, delete-orphan")


class Fact(Base):
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True, index=True)
    memory_id = Column(Integer, ForeignKey("memories.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    content = Column(Text, nullable=False)
    category = Column(String(50), default="fact")
    importance = Column(String(20), default="medium")
    
    vector_id = Column(String, nullable=True, index=True)
    
    entities = Column(JSON, default=list)
    relations = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    memory = relationship("Memory", back_populates="facts")


class MemoryJudgment(Base):
    __tablename__ = "memory_judgments"
    
    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(36), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    operation_type = Column(String(20), nullable=False)
    
    input_content = Column(Text, nullable=False)
    extracted_facts = Column(JSON, default=list)
    existing_memories = Column(JSON, default=list)
    
    llm_response = Column(Text, nullable=False)
    parsed_operations = Column(JSON, default=list)
    
    reasoning = Column(Text, nullable=True)
    
    executed_operations = Column(JSON, default=dict)
    execution_success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    model_name = Column(String(100), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    is_verified = Column(Boolean, default=False)
    verified_at = Column(DateTime, nullable=True)
    verification_result = Column(String(20), nullable=True)
    verification_notes = Column(Text, nullable=True)


class UserEncryptionKey(Base):
    __tablename__ = "user_encryption_keys"
    
    user_id = Column(String(64), primary_key=True, index=True)
    encrypted_dek = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    key_version = Column(Integer, default=1)


class UserQuota(Base):
    __tablename__ = "user_quotas"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    
    cloud_search_used = Column(Integer, default=0)
    memories_created = Column(Integer, default=0)
    batch_uploads_today = Column(Integer, default=0)
    
    period_start = Column(DateTime, default=datetime.utcnow)
    last_reset_date = Column(DateTime, default=datetime.utcnow)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="quota")
    
    def check_and_reset_monthly(self) -> bool:
        now = datetime.utcnow()
        if self.period_start.month != now.month or self.period_start.year != now.year:
            self.cloud_search_used = 0
            self.memories_created = 0
            self.period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return True
        return False
    
    def check_and_reset_daily(self) -> bool:
        now = datetime.utcnow()
        if self.last_reset_date.date() < now.date():
            self.batch_uploads_today = 0
            self.last_reset_date = now
            return True
        return False
    
    def can_cloud_search(self, tier: SubscriptionTier) -> tuple:
        self.check_and_reset_monthly()
        limit = QUOTA_LIMITS[tier]["cloud_search_per_month"]
        if limit == -1:
            return True, -1
        remaining = max(0, limit - self.cloud_search_used)
        return self.cloud_search_used < limit, remaining
    
    def can_create_memory(self, tier: SubscriptionTier) -> tuple:
        self.check_and_reset_monthly()
        limit = QUOTA_LIMITS[tier]["memories_per_month"]
        if limit == -1:
            return True, -1
        remaining = max(0, limit - self.memories_created)
        return self.memories_created < limit, remaining
    
    def can_batch_upload(self, tier: SubscriptionTier) -> tuple:
        self.check_and_reset_daily()
        limit = QUOTA_LIMITS[tier]["batch_upload_per_day"]
        if limit == -1:
            return True, -1
        remaining = max(0, limit - self.batch_uploads_today)
        return self.batch_uploads_today < limit, remaining
    
    def increment_cloud_search(self):
        self.check_and_reset_monthly()
        self.cloud_search_used += 1
    
    def increment_memories_created(self, count: int = 1):
        self.check_and_reset_monthly()
        self.memories_created += count
    
    def increment_batch_uploads(self, count: int = 1):
        self.check_and_reset_daily()
        self.batch_uploads_today += count


def get_or_create_quota(db, user_id: int) -> UserQuota:
    quota = db.query(UserQuota).filter(UserQuota.user_id == user_id).first()
    if not quota:
        quota = UserQuota(user_id=user_id)
        db.add(quota)
        db.commit()
        db.refresh(quota)
    return quota


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
