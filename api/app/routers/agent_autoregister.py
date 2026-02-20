from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets
import hashlib

from app.core.database import get_db, User, Project, APIKey

router = APIRouter(prefix="/agents", tags=["Agent Auto Registration"])

class AgentRegisterRequest(BaseModel):
    machine_fingerprint: str = Field(..., min_length=16, max_length=64)
    agent_type: str = Field(...)
    agent_name: str = Field(default="unnamed-agent")
    platform: str = Field(default="unknown")
    platform_version: str = Field(default=None)
    python_version: str = Field(default=None)

class AgentRegisterResponse(BaseModel):
    agent_id: str
    api_key: str
    project_id: int
    is_new_machine: bool
    message: str

@router.post("/auto-register", response_model=AgentRegisterResponse)
async def auto_register_agent(
    request: AgentRegisterRequest,
    db: Session = Depends(get_db)
):
    """Agent 自动注册接口 - 基于机器指纹创建账户"""
    
    machine_hash = hashlib.sha256(
        request.machine_fingerprint.encode()
    ).hexdigest()[:16]
    
    machine_email = f"machine_{machine_hash}@t0ken.ai"
    
    # 检查是否已存在
    user = db.query(User).filter(User.email == machine_email).first()
    is_new_machine = False
    
    if not user:
        # 创建新机器用户
        is_new_machine = True
        from app.core.security import get_password_hash
        
        random_password = secrets.token_urlsafe(32)
        
        user = User(
            email=machine_email,
            hashed_password=get_password_hash(random_password),
            is_active=True,
            is_machine_account=True,
            machine_fingerprint=request.machine_fingerprint,
            machine_hash=machine_hash
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # 创建默认项目
        project = Project(
            name=f"Machine-{machine_hash[:8]}",
            description=f"Auto-created for {request.agent_name} on {request.platform}",
            owner_id=user.id,
            is_machine_default=True,
            machine_fingerprint=request.machine_fingerprint
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        
        # 创建 API Key - 使用 key_hash 字段
        api_key_value = f"mx_m_{machine_hash}_{secrets.token_hex(16)}"
        api_key = APIKey(
            key_hash=api_key_value,
            name=f"Auto-generated for {request.agent_name}",
            user_id=user.id,
            project_id=project.id,
            is_auto_generated=True
        )
        db.add(api_key)
        db.commit()
    else:
        # 获取现有项目和 API Key
        project = db.query(Project).filter(
            Project.owner_id == user.id,
            Project.is_machine_default == True
        ).first()
        
        if not project:
            project = Project(
                name=f"Machine-{machine_hash[:8]}",
                description="Default project",
                owner_id=user.id,
                is_machine_default=True,
                machine_fingerprint=request.machine_fingerprint
            )
            db.add(project)
            db.commit()
            db.refresh(project)
        
        api_key = db.query(APIKey).filter(
            APIKey.user_id == user.id,
            APIKey.project_id == project.id
        ).first()
        
        if not api_key:
            api_key_value = f"mx_m_{machine_hash}_{secrets.token_hex(16)}"
            api_key = APIKey(
                key_hash=api_key_value,
                name=f"Auto-generated for {request.agent_name}",
                user_id=user.id,
                project_id=project.id,
                is_auto_generated=True
            )
            db.add(api_key)
            db.commit()
        
        user.last_login = datetime.utcnow()
        db.commit()
    
    return AgentRegisterResponse(
        agent_id=f"agent_{machine_hash}_{secrets.token_hex(8)}",
        api_key=api_key.key_hash,
        project_id=project.id,
        is_new_machine=is_new_machine,
        message="Welcome! Your agent is connected to your machine's memory pool." if is_new_machine else "Welcome back! Connected to existing memory pool."
    )

@router.get("/machine-stats")
async def get_machine_stats(
    api_key: str,
    db: Session = Depends(get_db)
):
    """获取当前机器上的 Agent 统计信息"""
    
    import hashlib
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_record = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    if not key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user = db.query(User).filter(User.id == key_record.user_id).first()
    if not user or not user.machine_fingerprint:
        raise HTTPException(status_code=400, detail="Not a machine account")
    
    # 统计
    agents = db.query(APIKey).filter(APIKey.user_id == user.id).all()
    
    total_memories = 0  # 简化版本，不查询 Memory 表
    
    return {
        "machine_fingerprint": user.machine_fingerprint[:16] + "...",
        "machine_hash": user.machine_hash,
        "total_agents": len(agents),
        "agents": [{"name": a.name, "created_at": str(a.created_at)} for a in agents],
        "total_memories": total_memories,
        "project_id": key_record.project_id
    }
