"""
Encryption utilities for MemoryX - AES-256-GCM implementation.

Key hierarchy:
- Master Key (from env MEMORYX_MASTER_KEY) -> encrypts User DEK
- User DEK (random 256-bit) -> encrypts actual content
"""

import os
import secrets
import hashlib
import base64
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2


class EncryptionManager:
    """Manages encryption/decryption with AES-256-GCM."""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize with master key from environment or parameter.
        
        Args:
            master_key: Optional master key (falls back to MEMORYX_MASTER_KEY env var)
        """
        key_source = master_key or os.getenv("MEMORYX_MASTER_KEY")
        if not key_source:
            raise ValueError("MEMORYX_MASTER_KEY not set in environment")
        
        # Derive 256-bit master key from string using PBKDF2
        # Using a fixed salt for deterministic key derivation
        salt = b"memoryx_master_salt_v1"
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        self.master_key = kdf.derive(key_source.encode())
    
    def generate_dek(self) -> bytes:
        """Generate a new 256-bit Data Encryption Key."""
        return secrets.token_bytes(32)
    
    def encrypt_dek(self, dek: bytes) -> bytes:
        """
        Encrypt a DEK with the master key.
        
        Args:
            dek: The 256-bit DEK to encrypt
            
        Returns:
            Encrypted DEK with nonce prepended (nonce + ciphertext)
        """
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self.master_key)
        encrypted = aesgcm.encrypt(nonce, dek, None)
        # Prepend nonce for decryption
        return nonce + encrypted
    
    def decrypt_dek(self, encrypted_dek: bytes) -> bytes:
        """
        Decrypt a DEK with the master key.
        
        Args:
            encrypted_dek: The encrypted DEK (nonce + ciphertext)
            
        Returns:
            The decrypted 256-bit DEK
        """
        nonce = encrypted_dek[:12]
        ciphertext = encrypted_dek[12:]
        aesgcm = AESGCM(self.master_key)
        return aesgcm.decrypt(nonce, ciphertext, None)
    
    def encrypt_content(self, content: str, dek: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt content with a DEK using AES-256-GCM.
        
        Args:
            content: The plaintext content to encrypt
            dek: The 256-bit Data Encryption Key
            
        Returns:
            Tuple of (encrypted_content, nonce)
        """
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(dek)
        encrypted = aesgcm.encrypt(nonce, content.encode('utf-8'), None)
        return encrypted, nonce
    
    def decrypt_content(self, encrypted_content: bytes, nonce: bytes, dek: bytes) -> str:
        """
        Decrypt content with a DEK using AES-256-GCM.
        
        Args:
            encrypted_content: The encrypted content
            nonce: The 12-byte nonce used for encryption
            dek: The 256-bit Data Encryption Key
            
        Returns:
            The decrypted plaintext content
        """
        aesgcm = AESGCM(dek)
        decrypted = aesgcm.decrypt(nonce, encrypted_content, None)
        return decrypted.decode('utf-8')
    
    @staticmethod
    def encode_base64(data: bytes) -> str:
        """Encode bytes to base64 string."""
        return base64.b64encode(data).decode('ascii')
    
    @staticmethod
    def decode_base64(data: str) -> bytes:
        """Decode base64 string to bytes."""
        return base64.b64decode(data)


# Global encryption manager instance
_encryption_manager: Optional[EncryptionManager] = None


def get_encryption_manager() -> EncryptionManager:
    """Get or create the global encryption manager instance."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def reset_encryption_manager():
    """Reset the encryption manager (useful for testing)."""
    global _encryption_manager
    _encryption_manager = None
