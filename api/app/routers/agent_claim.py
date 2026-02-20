from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets
import hashlib

from app.core.database import get_db, User, Project, APIKey

router = APIRouter(prefix="/agents/claim", tags=["Agent Account Claiming"])

# In-memory storage for claim requests (use Redis in production)
claim_requests = {}

class ClaimCodeRequest(BaseModel):
    machine_fingerprint: str
    api_key: str

class ClaimCodeResponse(BaseModel):
    claim_code: str
    expires_in: int
    claim_url: str

class VerifyClaimRequest(BaseModel):
    claim_code: str
    user_email: str

class BindAgentRequest(BaseModel):
    claim_code: str
    machine_fingerprint: str
    api_key: str

@router.post("/initiate", response_model=ClaimCodeResponse)
async def initiate_claim(
    request: ClaimCodeRequest,
    db: Session = Depends(get_db)
):
    """Agent 发起认领请求 - 生成验证码"""
    
    import hashlib
    key_hash = hashlib.sha256(request.api_key.encode()).hexdigest()
    api_key_record = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    if not api_key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    if not user or user.machine_fingerprint != request.machine_fingerprint:
        raise HTTPException(status_code=403, detail="Fingerprint mismatch")
    
    # 生成6位验证码
    claim_code = secrets.token_hex(3).upper()
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    
    claim_requests[claim_code] = {
        "machine_fingerprint": request.machine_fingerprint,
        "api_key": request.api_key,
        "user_id": user.id,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "status": "pending",
        "claimed_by_email": None
    }
    
    claim_url = f"https://t0ken.ai/admin/claim?code={claim_code}"
    
    return ClaimCodeResponse(
        claim_code=claim_code,
        expires_in=600,
        claim_url=claim_url
    )

@router.get("/status/{claim_code}")
async def check_claim_status(claim_code: str):
    """检查认领状态"""
    if claim_code not in claim_requests:
        raise HTTPException(status_code=404, detail="Claim code not found")
    
    claim = claim_requests[claim_code]
    
    if datetime.utcnow() > claim["expires_at"]:
        del claim_requests[claim_code]
        raise HTTPException(status_code=410, detail="Claim code expired")
    
    return {
        "status": claim["status"],
        "expires_at": claim["expires_at"].isoformat()
    }

@router.post("/verify")
async def verify_claim(
    request: VerifyClaimRequest,
    db: Session = Depends(get_db)
):
    """用户在后台验证认领码"""
    if request.claim_code not in claim_requests:
        raise HTTPException(status_code=404, detail="Invalid claim code")
    
    claim = claim_requests[request.claim_code]
    
    if datetime.utcnow() > claim["expires_at"]:
        del claim_requests[request.claim_code]
        raise HTTPException(status_code=410, detail="Claim code expired")
    
    # 查找人类账户
    human_user = db.query(User).filter(
        User.email == request.user_email,
        User.is_machine_account == False
    ).first()
    
    if not human_user:
        raise HTTPException(status_code=404, detail="Human account not found")
    
    claim["status"] = "verified"
    claim["claimed_by_email"] = request.user_email
    claim["human_user_id"] = human_user.id
    
    return {
        "status": "verified",
        "message": "Verification successful. Agent will complete the binding."
    }

@router.post("/complete")
async def complete_binding(
    request: BindAgentRequest,
    db: Session = Depends(get_db)
):
    """Agent 完成最终绑定 - 迁移数据到人类账户"""
    if request.claim_code not in claim_requests:
        raise HTTPException(status_code=404, detail="Claim code not found")
    
    claim = claim_requests[request.claim_code]
    
    if claim["status"] != "verified":
        raise HTTPException(status_code=400, detail="Claim not verified")
    
    if claim["machine_fingerprint"] != request.machine_fingerprint:
        raise HTTPException(status_code=403, detail="Fingerprint mismatch")
    
    machine_user_id = claim["user_id"]
    human_user_id = claim["human_user_id"]
    
    try:
        # 迁移项目
        projects = db.query(Project).filter(Project.owner_id == machine_user_id).all()
        for project in projects:
            project.owner_id = human_user_id
        
        # 迁移 API Keys
        api_keys = db.query(APIKey).filter(APIKey.user_id == machine_user_id).all()
        for key in api_keys:
            key.user_id = human_user_id
        
        # 停用机器账户
        machine_user = db.query(User).filter(User.id == machine_user_id).first()
        if machine_user:
            machine_user.is_active = False
            machine_user.merged_to_user_id = human_user_id
        
        db.commit()
        
        claim["status"] = "completed"
        
        return {
            "status": "success",
            "message": "Machine account bound to human account",
            "migrated_projects": len(projects),
            "migrated_memories": 0
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Binding failed: {str(e)}")
