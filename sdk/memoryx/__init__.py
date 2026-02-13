"""
MemoryX Python SDK
让 AI Agents 轻松拥有持久记忆
"""

__version__ = "1.0.5"
__author__ = "MemoryX Team"

from .client import MemoryXClient, connect_memory

__all__ = ["MemoryXClient", "connect_memory"]
