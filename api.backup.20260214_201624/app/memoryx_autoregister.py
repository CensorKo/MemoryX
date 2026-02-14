"""
MemoryX Agent Auto-Registration SDK
单机隔离版 - 同一机器上的所有 Agent 自动共享记忆
"""
import os
import sys
import platform
import hashlib
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict
import requests

class MemoryXAutoRegister:
    """
    MemoryX 自动注册客户端
    
    特性：
    - 基于机器指纹自动注册
    - 同一机器上的所有 Agent 共享记忆
    - 跨机器完全隔离
    """
    
    API_BASE = "https://t0ken.ai/api"
    CONFIG_FILE = Path.home() / ".memoryx" / "agent_config.json"
    
    def __init__(self, agent_type: str = "generic", agent_name: Optional[str] = None):
        """
        初始化并自动注册（如需要）
        
        Args:
            agent_type: Agent 类型，如 "claude", "gpt", "custom"
            agent_name: Agent 名称，可选
        """
        self.agent_type = agent_type
        self.agent_name = agent_name or f"{agent_type}-agent"
        self.config = self._load_config()
        
        # 如果没有 API Key，自动注册
        if not self.config.get("api_key"):
            self._auto_register()
    
    def _get_machine_fingerprint(self) -> str:
        """生成机器唯一指纹"""
        components = []
        
        try:
            if platform.system() == "Darwin":  # macOS
                # 读取硬件 UUID
                result = subprocess.run(
                    ["ioreg", "-d2", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, text=True
                )
                for line in result.stdout.split("\n"):
                    if "IOPlatformUUID" in line:
                        uuid = line.split('"')[-2]
                        components.append(uuid)
                        break
                        
            elif platform.system() == "Linux":
                # 读取 machine-id
                machine_id = Path("/etc/machine-id").read_text().strip()
                components.append(machine_id)
                
                # 添加硬件信息增强唯一性
                result = subprocess.run(
                    ["cat", "/proc/cpuinfo"],
                    capture_output=True, text=True
                )
                for line in result.stdout.split("\n"):
                    if "serial" in line.lower() or "uuid" in line.lower():
                        components.append(line.split(":")[-1].strip())
                        break
                        
            elif platform.system() == "Windows":
                # 读取主板 UUID
                result = subprocess.run(
                    ["wmic", "csproduct", "get", "uuid"],
                    capture_output=True, text=True
                )
                components.append(result.stdout.split("\n")[1].strip())
        
        except Exception as e:
            # 回退方案：基于用户目录生成
            print(f"Warning: Could not get hardware fingerprint: {e}")
            components.append(str(Path.home()))
            components.append(os.getenv("USER", "unknown"))
        
        # 组合并哈希
        fingerprint_raw = "|".join(components)
        return hashlib.sha256(fingerprint_raw.encode()).hexdigest()[:32]
    
    def _auto_register(self):
        """自动注册新 Agent"""
        fingerprint = self._get_machine_fingerprint()
        
        payload = {
            "machine_fingerprint": fingerprint,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": sys.version
        }
        
        try:
            response = requests.post(
                f"{self.API_BASE}/agents/auto-register",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            self.config = {
                "api_key": data["api_key"],
                "project_id": data["project_id"],
                "machine_fingerprint": fingerprint,
                "agent_id": data["agent_id"]
            }
            self._save_config()
            
            print(f"✅ MemoryX: Agent registered successfully!")
            print(f"   Machine ID: {fingerprint[:8]}...")
            print(f"   Project: {data['project_id']}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ MemoryX: Auto-registration failed: {e}")
            raise
    
    def _load_config(self) -> Dict:
        """加载本地配置"""
        if self.CONFIG_FILE.exists():
            return json.loads(self.CONFIG_FILE.read_text())
        return {}
    
    def _save_config(self):
        """保存配置到本地"""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.CONFIG_FILE.write_text(json.dumps(self.config, indent=2))
        # 设置权限，仅当前用户可读
        os.chmod(self.CONFIG_FILE, 0o600)
    
    @property
    def api_key(self) -> str:
        """获取 API Key"""
        return self.config.get("api_key", "")
    
    @property
    def project_id(self) -> str:
        """获取项目 ID"""
        return self.config.get("project_id", "default")
    
    def get_stats(self) -> Dict:
        """获取当前机器上的 Agent 统计信息"""
        try:
            response = requests.get(
                f"{self.API_BASE}/agents/machine-stats",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except:
            return {}


# 便捷函数：一行代码接入 MemoryX
def connect_memory(agent_type: str = "generic", agent_name: Optional[str] = None):
    """
    一行代码连接 MemoryX
    
    示例：
        from memoryx import connect_memory
        memory = connect_memory(agent_type="claude")
        
        # 存储记忆
        memory.add("用户喜欢Python")
        
        # 检索记忆
        results = memory.search("用户偏好")
    """
    from memoryx import MemoryXClient
    
    # 先自动注册（如需要）
    auto_reg = MemoryXAutoRegister(agent_type=agent_type, agent_name=agent_name)
    
    # 返回配置好的客户端
    return MemoryXClient(
        api_key=auto_reg.api_key,
        project_id=auto_reg.project_id
    )


# 向后兼容
class AutoRegisterClient(MemoryXAutoRegister):
    """别名，向后兼容"""
    pass
