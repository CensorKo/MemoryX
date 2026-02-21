"""
MemoryX Client - Python SDK for MemoryX API
"""

import json
import hashlib
import platform
import socket
import uuid
from typing import Optional, List, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class MemoryXClient:
    """MemoryX Memory Client"""
    
    DEFAULT_BASE_URL = "https://t0ken.ai/api"
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize MemoryX client
        
        Args:
            api_key: API Key (if already registered)
            base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.machine_fingerprint = self._generate_fingerprint()
        
    def _generate_fingerprint(self) -> str:
        """Generate machine fingerprint based on hardware info"""
        machine_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "mac": hex(uuid.getnode()),
        }
        fingerprint_str = json.dumps(machine_info, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]
    
    def _request(self, method: str, endpoint: str, data: Optional[dict] = None) -> dict:
        """Send HTTP request"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"memoryx-python/1.0.6"
        }
        
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            
        req = Request(
            url,
            data=json.dumps(data).encode() if data else None,
            headers=headers,
            method=method
        )
        
        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            error_body = e.read().decode()
            try:
                error_data = json.loads(error_body)
                raise MemoryXError(error_data.get("detail", error_data.get("message", f"HTTP {e.code}")))
            except json.JSONDecodeError:
                raise MemoryXError(f"HTTP {e.code}: {error_body}")
        except Exception as e:
            raise MemoryXError(f"Request failed: {str(e)}")
    
    def auto_register(self, agent_type: str = "python_sdk", agent_name: str = "memoryx-client") -> dict:
        """
        Auto-register machine account
        
        Args:
            agent_type: Agent type (default: "python_sdk")
            agent_name: Agent name (default: "memoryx-client")
        
        Returns:
            Registration result with api_key, user_id, etc.
        """
        data = {
            "machine_fingerprint": self.machine_fingerprint,
            "agent_type": agent_type,
            "agent_name": agent_name,
            "platform": platform.system().lower(),
            "platform_version": platform.version(),
            "python_version": platform.python_version()
        }
        
        result = self._request("POST", "/agents/auto-register", data)
        
        if result.get("api_key"):
            self.api_key = result["api_key"]
            self._save_config()
            
        return result

    def _save_config(self):
        """Save config to local file"""
        import os
        config_dir = os.path.expanduser("~/.memoryx")
        os.makedirs(config_dir, exist_ok=True)
        
        config = {
            "api_key": self.api_key,
            "machine_fingerprint": self.machine_fingerprint,
            "base_url": self.base_url
        }
        
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(config, f)
    
    def _load_config(self) -> bool:
        """Load config from local file"""
        import os
        config_path = os.path.expanduser("~/.memoryx/config.json")
        
        if not os.path.exists(config_path):
            return False
            
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            if config.get("machine_fingerprint") == self.machine_fingerprint:
                self.api_key = config.get("api_key")
                self.base_url = config.get("base_url", self.DEFAULT_BASE_URL)
                return True
        except Exception:
            pass
            
        return False
    
    def add(self, content: str, project_id: str = "default", 
            metadata: Optional[dict] = None) -> dict:
        """
        Store a memory
        
        Args:
            content: Memory content to store
            project_id: Project ID (default: "default")
            metadata: Additional metadata
            
        Returns:
            {"success": True, "task_id": "...", "status": "pending"}
            
        Example:
            >>> memory.add("User prefers dark mode")
            >>> memory.add("User works at Google", project_id="work")
        """
        if not self.api_key:
            raise MemoryXError("Not registered. Call auto_register() first.")
            
        data = {
            "content": content,
            "project_id": project_id,
            "metadata": metadata or {}
        }
        
        return self._request("POST", "/v1/memories", data)
    
    def search(self, query: str, project_id: Optional[str] = None,
               limit: int = 10) -> dict:
        """
        Search memories by semantic similarity
        
        Args:
            query: Search query
            project_id: Project ID filter (optional)
            limit: Maximum results (default: 10)
            
        Returns:
            {"success": True, "data": [...], "query": "..."}
            
        Example:
            >>> results = memory.search("user preferences")
            >>> for item in results["data"]:
            ...     print(item["content"])
        """
        if not self.api_key:
            raise MemoryXError("Not registered. Call auto_register() first.")
            
        data = {
            "query": query,
            "limit": limit
        }
        
        if project_id:
            data["project_id"] = project_id
            
        return self._request("POST", "/v1/memories/search", data)
    
    def list(self, project_id: Optional[str] = None, 
             limit: int = 50, offset: int = 0) -> dict:
        """
        List all memories
        
        Args:
            project_id: Project ID filter (optional)
            limit: Maximum results (default: 50)
            offset: Pagination offset (default: 0)
            
        Returns:
            {"success": True, "data": [...], "total": 100, "limit": 50, "offset": 0}
            
        Example:
            >>> memories = memory.list(limit=10)
            >>> for m in memories["data"]:
            ...     print(f"{m['id']}: {m['content']}")
        """
        if not self.api_key:
            raise MemoryXError("Not registered. Call auto_register() first.")
            
        params = f"?limit={limit}&offset={offset}"
        if project_id:
            params += f"&project_id={project_id}"
            
        return self._request("GET", f"/v1/memories/list{params}")
    
    def delete(self, memory_id: str) -> dict:
        """
        Delete a memory
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            {"success": True, "message": "Memory deleted successfully"}
            
        Example:
            >>> memory.delete("abc123")
        """
        if not self.api_key:
            raise MemoryXError("Not registered. Call auto_register() first.")
            
        return self._request("DELETE", f"/v1/memories/{memory_id}")
    
    def get_task_status(self, task_id: str) -> dict:
        """
        Get async task status
        
        Args:
            task_id: Task ID returned from add()
            
        Returns:
            {"task_id": "...", "status": "SUCCESS|PENDING|FAILURE", "result": {...}}
        """
        if not self.api_key:
            raise MemoryXError("Not registered. Call auto_register() first.")
            
        return self._request("GET", f"/v1/memories/task/{task_id}")
    
    def get_quota(self) -> dict:
        """
        Get quota information
        
        Returns:
            {"success": True, "quota": {"tier": "free", "memories": {...}, ...}}
        """
        if not self.api_key:
            raise MemoryXError("Not registered. Call auto_register() first.")
            
        return self._request("GET", "/v1/quota")


class MemoryXError(Exception):
    """MemoryX Error"""
    pass


def connect_memory(base_url: Optional[str] = None, verbose: bool = True) -> MemoryXClient:
    """
    Quick connect to MemoryX
    
    Auto-detects local config, or auto-registers if not found.
    
    Args:
        base_url: API base URL (optional, defaults to https://t0ken.ai/api)
        verbose: Print usage guide (default: True)
        
    Returns:
        MemoryXClient instance
        
    Example:
        >>> from memoryx import connect_memory
        >>> memory = connect_memory()
        >>> memory.add("User prefers dark mode")
        >>> results = memory.search("user preferences")
        >>> memories = memory.list()
    """
    client = MemoryXClient(base_url=base_url)
    
    if client._load_config():
        if verbose:
            print("Connected to MemoryX")
    else:
        if verbose:
            print("First time setup, registering...")
        result = client.auto_register()
        
        if result.get("api_key"):
            if verbose:
                print(f"MemoryX activated")
                print(f"  Agent ID: {result.get('agent_id', 'N/A')}")
        else:
            raise MemoryXError(f"Registration failed: {result}")
    
    if verbose:
        _print_usage_guide()
    
    return client


def _print_usage_guide():
    """Print usage guide"""
    guide = """
MemoryX Usage:
  memory.add(content)           - Store a memory
  memory.search(query)          - Search memories
  memory.list()                 - List all memories
  memory.delete(memory_id)      - Delete a memory
  memory.get_quota()            - Get quota info

Example:
  memory.add("User prefers dark mode")
  results = memory.search("preferences")
  for m in results["data"]:
      print(m["content"])
"""
    print(guide)
