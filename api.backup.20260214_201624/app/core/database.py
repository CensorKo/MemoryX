from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text, JSON, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from app.core.config import get_settings

settings = get_settings()

# Create engine with connection pooling for high concurrency
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

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Machine auto-registration fields
    machine_fingerprint = Column(String(64), nullable=True, index=True)
    machine_hash = Column(String(32), nullable=True, index=True)
    is_machine_account = Column(Boolean, default=False)
    last_login = Column(DateTime, nullable=True)
    
    # For Firebase Auth
    firebase_uid = Column(String, nullable=True)
    
    # For account merging
    merged_to_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Machine auto-registration fields
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
    
    # Machine auto-registration fields
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
    
    # AI Classification
    cognitive_sector = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Metadata (using 'meta' to avoid SQLAlchemy reserved word)
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # For vector search
    embedding_id = Column(String, nullable=True, index=True)


class UserEncryptionKey(Base):
    """Stores encrypted DEK (Data Encryption Key) for each user."""
    __tablename__ = "user_encryption_keys"
    
    # Using user_id (string) as primary key for easier lookup with vector store
    user_id = Column(String(64), primary_key=True, index=True)
    
    # The DEK encrypted with system master key
    encrypted_dek = Column(LargeBinary, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Key version for future key rotation
    key_version = Column(Integer, default=1)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
