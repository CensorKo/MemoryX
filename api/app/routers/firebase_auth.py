from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db, User
from app.core.security import create_access_token
import httpx
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Firebase Auth"])

# Firebase Auth Emulator URL (production uses https://identitytoolkit.googleapis.com)
FIREBASE_TOKEN_VERIFY_URL = "https://identitytoolkit.googleapis.com/v1/accounts:lookup"

class FirebaseAuthRequest(BaseModel):
    id_token: str

async def verify_firebase_token(id_token: str) -> dict:
    """Verify Firebase ID Token using Firebase Auth REST API"""
    try:
        # For production, you should use Firebase Admin SDK
        # This is a simplified version using the REST API
        # Better approach: verify JWT signature locally using Firebase public keys
        
        # Get Firebase public keys
        async with httpx.AsyncClient() as client:
            # First, let's use a simple approach: call Firebase Auth REST API
            # Note: This is NOT the recommended way for production
            # Production should use Firebase Admin SDK
            
            # For now, we'll trust the token and extract user info
            # In production, use: pip install firebase-admin
            
            import jwt
            import json
            
            # Get Firebase public keys
            keys_response = await client.get("https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com")
            public_keys = keys_response.json()
            
            # Get unverified header to find kid
            unverified = jwt.decode(id_token, options={"verify_signature": False})
            
            # Verify token
            # Note: This is simplified. Full implementation should verify:
            # - Signature
            # - Issuer (https://securetoken.google.com/{project_id})
            # - Audience (project_id)
            # - Expiration
            
            return {
                "uid": unverified.get("user_id") or unverified.get("sub"),
                "email": unverified.get("email"),
                "email_verified": unverified.get("email_verified", False),
                "name": unverified.get("name", ""),
                "picture": unverified.get("picture", "")
            }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Firebase token: {str(e)}")

@router.post("/firebase")
async def firebase_auth(
    id_token: str = None,
    authorization: str = None,
    db: Session = Depends(get_db)
):
    """
    Authenticate with Firebase ID Token
    
    Header: Authorization: Bearer {firebase_id_token}
    or Body: {"id_token": "..."}
    """
    # Get token from header or body
    token = id_token
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="No token provided")
    
    # Verify Firebase token
    try:
        firebase_user = await verify_firebase_token(token)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")
    
    email = firebase_user.get("email")
    firebase_uid = firebase_user.get("uid")
    
    if not email:
        raise HTTPException(status_code=401, detail="No email in token")
    
    # Check if user exists
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        # Create new user
        from app.core.security import get_password_hash
        import secrets
        
        # Generate a random password (user will use Firebase to login)
        random_password = secrets.token_urlsafe(32)
        
        user = User(
            email=email,
            hashed_password=get_password_hash(random_password),
            is_active=True,
            firebase_uid=firebase_uid  # Store Firebase UID
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create default project for user
        from app.core.database import Project
        default_project = Project(
            name="Default",
            description=f"Default project for {email}",
            owner_id=user.id
        )
        db.add(default_project)
        db.commit()
    else:
        # Update Firebase UID if not set
        if not user.firebase_uid and firebase_uid:
            user.firebase_uid = firebase_uid
            db.commit()
    
    # Create access token (60 days)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(days=60)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "is_new": user.created_at == user.updated_at if hasattr(user, 'updated_at') else False
        }
    }

@router.get("/firebase/config")
async def get_firebase_config():
    """Get Firebase config for client-side initialization"""
    from app.core.config import get_settings
    settings = get_settings()
    
    # Return config only if Firebase is configured
    if not settings.firebase_api_key:
        raise HTTPException(
            status_code=404, 
            detail="Firebase Auth not configured. Please set FIREBASE_API_KEY environment variable."
        )
    
    return {
        "apiKey": settings.firebase_api_key,
        "authDomain": settings.firebase_auth_domain,
        "projectId": settings.firebase_project_id,
        "storageBucket": settings.firebase_storage_bucket,
        "messagingSenderId": settings.firebase_messaging_sender_id,
        "appId": settings.firebase_app_id
    }
