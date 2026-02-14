from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import hashlib

from app.core.database import get_db
from app.core.database import APIKey
from app.core.database import Project

router = APIRouter(prefix="/projects", tags=["projects"])

# Schemas
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str

# Auth helper
def get_current_user_api(x_api_key: str = Header(None), db: Session = Depends(get_db)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash, APIKey.is_active == True).first()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return api_key.user_id

@router.get("", response_model=dict)
async def list_projects(
    user_id: int = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    projects = db.query(Project).filter(Project.user_id == user_id).all()
    
    return {
        "success": True,
        "data": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None
            }
            for p in projects
        ],
        "total": len(projects)
    }

@router.post("", response_model=dict)
async def create_project(
    project: ProjectCreate,
    user_id: int = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    db_project = Project(
        user_id=user_id,
        name=project.name,
        description=project.description
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    return {
        "success": True,
        "data": {
            "id": db_project.id,
            "name": db_project.name,
            "description": db_project.description,
            "created_at": db_project.created_at.isoformat() if db_project.created_at else None,
            "updated_at": db_project.updated_at.isoformat() if db_project.updated_at else None
        },
        "message": "Project created successfully"
    }

@router.get("/{project_id}", response_model=dict)
async def get_project(
    project_id: int,
    user_id: int = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {
        "success": True,
        "data": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at.isoformat() if project.created_at else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None
        }
    }

@router.put("/{project_id}", response_model=dict)
async def update_project(
    project_id: int,
    update: ProjectUpdate,
    user_id: int = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if update.name is not None:
        project.name = update.name
    if update.description is not None:
        project.description = update.description
    
    project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(project)
    
    return {
        "success": True,
        "data": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at.isoformat() if project.created_at else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None
        },
        "message": "Project updated successfully"
    }

@router.delete("/{project_id}", response_model=dict)
async def delete_project(
    project_id: int,
    user_id: int = Depends(get_current_user_api),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    
    return {
        "success": True,
        "message": "Project deleted successfully"
    }