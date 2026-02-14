import httpx
from celery import shared_task
from app.core.config import get_settings

settings = get_settings()

@shared_task(bind=True, max_retries=3)
def process_memory(self, memory_data: dict, api_key: str):
    try:
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{settings.openmemoryx_url}/v1/memories",
                json=memory_data,
                headers=headers,
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException as exc:
        raise self.retry(exc=exc, countdown=10)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code >= 500:
            raise self.retry(exc=exc, countdown=10)
        raise
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)

@shared_task(bind=True, max_retries=3)
def update_memory_task(self, memory_id: str, update_data: dict, api_key: str):
    try:
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.put(
                f"{settings.openmemoryx_url}/v1/memories/{memory_id}",
                json=update_data,
                headers=headers,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException as exc:
        raise self.retry(exc=exc, countdown=10)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code >= 500:
            raise self.retry(exc=exc, countdown=10)
        raise
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)

@shared_task
def search_memory(query_data: dict, api_key: str):
    try:
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.openmemoryx_url}/v1/memories/search",
                json=query_data,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
    except Exception as exc:
        return {"error": str(exc), "results": []}

def get_user_memories(user_id: str, project_id: str = None, limit: int = 100, offset: int = 0):
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            "user_id": user_id,
            "limit": limit,
            "offset": offset
        }
        if project_id:
            params["project_id"] = project_id
            
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{settings.openmemoryx_url}/v1/memories",
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            return result.get("data", [])
            
    except Exception as exc:
        return []

def get_memory_by_id(memory_id: str, user_id: str):
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{settings.openmemoryx_url}/v1/memories/{memory_id}",
                headers=headers,
                params={"user_id": user_id},
                timeout=30.0
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
            
    except Exception as exc:
        return None

def delete_memory(memory_id: str, user_id: str):
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.delete(
                f"{settings.openmemoryx_url}/v1/memories/{memory_id}",
                headers=headers,
                params={"user_id": user_id},
                timeout=30.0
            )
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True
            
    except Exception as exc:
        return False