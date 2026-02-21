"""
MemoryX Python SDK
Give your AI agents long-term memory
"""

__version__ = "1.1.0"
__author__ = "MemoryX Team"

from .client import MemoryXClient, MemoryXError, connect_memory

__all__ = ["MemoryXClient", "MemoryXError", "connect_memory"]
