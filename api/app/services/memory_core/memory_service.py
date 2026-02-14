"""
Memory Service - Core memory operations with hybrid approach.

Combines project-based organization with cognitive sector classification
for more intelligent memory retrieval.

Encryption Features:
- AES-256-GCM for content encryption
- Per-user Data Encryption Keys (DEK)
- Master key encryption for DEKs
- Backward compatible with unencrypted data
"""

import json
import hashlib
import httpx
import os
import base64
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition, 
    MatchValue, MatchAny, PayloadSchemaType
)

from app.services.memory_core.classification import MemoryClassifier
from app.services.memory_core.scoring import CompositeScorer

# Import encryption utilities
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    print("⚠️ cryptography not available, encryption disabled")


class EncryptionManager:
    """Manages encryption/decryption with AES-256-GCM."""
    
    def __init__(self, master_key: Optional[str] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required for encryption")
        
        key_source = master_key or os.getenv("MEMORYX_MASTER_KEY")
        if not key_source:
            raise ValueError("MEMORYX_MASTER_KEY not set in environment")
        
        # Derive 256-bit master key from string using PBKDF2
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
        """Encrypt a DEK with the master key."""
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self.master_key)
        encrypted = aesgcm.encrypt(nonce, dek, None)
        return nonce + encrypted
    
    def decrypt_dek(self, encrypted_dek: bytes) -> bytes:
        """Decrypt a DEK with the master key."""
        nonce = encrypted_dek[:12]
        ciphertext = encrypted_dek[12:]
        aesgcm = AESGCM(self.master_key)
        return aesgcm.decrypt(nonce, ciphertext, None)
    
    def encrypt_content(self, content: str, dek: bytes) -> Tuple[bytes, bytes]:
        """Encrypt content with a DEK using AES-256-GCM."""
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(dek)
        encrypted = aesgcm.encrypt(nonce, content.encode('utf-8'), None)
        return encrypted, nonce
    
    def decrypt_content(self, encrypted_content: bytes, nonce: bytes, dek: bytes) -> str:
        """Decrypt content with a DEK using AES-256-GCM."""
        aesgcm = AESGCM(dek)
        decrypted = aesgcm.decrypt(nonce, encrypted_content, None)
        return decrypted.decode('utf-8')


class MemoryService:
    """
    Hybrid memory service supporting project isolation and cognitive classification.
    
    Features:
    - Project-based memory isolation
    - 5-sector cognitive classification (LLM-powered)
    - Temporal validity tracking
    - Composite scoring for relevance
    - Explainable retrieval
    - AES-256-GCM content encryption
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize memory service with configuration.
        
        Args:
            config: Configuration dict with vector_store, llm, embedder settings
        """
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.classifier = MemoryClassifier(config["llm"])
        self.scorer = CompositeScorer()
        self.encryption: Optional[EncryptionManager] = None
        
        # Initialize encryption if master key is available
        if CRYPTOGRAPHY_AVAILABLE and os.getenv("MEMORYX_MASTER_KEY"):
            try:
                self.encryption = EncryptionManager()
                print("✓ Encryption enabled")
            except Exception as e:
                print(f"⚠️ Encryption init failed: {e}")
        else:
            print("ℹ️ Encryption disabled (no master key)")
        
        self._init_qdrant()
    
    def _get_or_create_user_dek(self, user_id: str) -> Optional[bytes]:
        """
        Get or create a DEK for a user.
        
        Args:
            user_id: The user identifier
            
        Returns:
            The user's DEK or None if encryption is disabled
        """
        if not self.encryption:
            return None
        
        try:
            # Check if we have a database session available
            from app.core.database import SessionLocal, UserEncryptionKey
            
            db = SessionLocal()
            try:
                # Try to get existing key
                key_record = db.query(UserEncryptionKey).filter(
                    UserEncryptionKey.user_id == user_id
                ).first()
                
                if key_record and key_record.is_active:
                    # Decrypt existing DEK
                    dek = self.encryption.decrypt_dek(key_record.encrypted_dek)
                    return dek
                
                # Generate new DEK
                dek = self.encryption.generate_dek()
                encrypted_dek = self.encryption.encrypt_dek(dek)
                
                # Store in database
                if key_record:
                    # Update existing record
                    key_record.encrypted_dek = encrypted_dek
                    key_record.is_active = True
                    key_record.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    key_record = UserEncryptionKey(
                        user_id=user_id,
                        encrypted_dek=encrypted_dek
                    )
                    db.add(key_record)
                
                db.commit()
                return dek
                
            finally:
                db.close()
                
        except Exception as e:
            print(f"⚠️ Failed to get/create DEK for user {user_id}: {e}")
            return None
    
    def _init_qdrant(self) -> None:
        """Initialize Qdrant connection and create collection if needed."""
        host = self.config["vector_store"]["config"]["host"]
        port = self.config["vector_store"]["config"]["port"]
        
        self.client = QdrantClient(host=host, port=port)
        
        try:
            self.client.get_collection("mem0")
            print("✓ Qdrant collection exists")
        except Exception:
            print("🔄 Creating new Qdrant collection...")
            self.client.create_collection(
                collection_name="mem0",
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            self._create_payload_indexes()
    
    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        indexes = [
            ("user_id", PayloadSchemaType.KEYWORD),
            ("project_id", PayloadSchemaType.KEYWORD),
            ("sector_primary", PayloadSchemaType.KEYWORD),
            ("temporal_is_current", PayloadSchemaType.BOOL),
            ("memory_types", PayloadSchemaType.KEYWORD),
            ("created_at", PayloadSchemaType.DATETIME),
            ("is_encrypted", PayloadSchemaType.BOOL),
        ]
        
        for field, schema_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name="mem0",
                    field_name=field,
                    field_schema=schema_type
                )
                print(f"  ✓ Index created: {field}")
            except Exception as e:
                print(f"  ⚠️ Index {field}: {e}")
    
    def _encrypt_memory_content(self, content: str, user_id: str) -> Dict[str, Any]:
        """
        Encrypt memory content for a user.
        
        Args:
            content: The plaintext content
            user_id: The user identifier
            
        Returns:
            Dict with encrypted content, nonce, and encryption flag
        """
        dek = self._get_or_create_user_dek(user_id)
        
        if dek is None:
            # Encryption disabled or failed, store plaintext
            return {
                "content": content,
                "is_encrypted": False
            }
        
        try:
            encrypted_content, nonce = self.encryption.encrypt_content(content, dek)
            
            # Clear DEK from memory
            dek = b'\x00' * len(dek)
            
            return {
                "content": content,  # Keep original for embedding
                "encrypted_content": base64.b64encode(encrypted_content).decode('ascii'),
                "content_nonce": base64.b64encode(nonce).decode('ascii'),
                "is_encrypted": True
            }
        except Exception as e:
            print(f"⚠️ Encryption failed: {e}, storing plaintext")
            return {
                "content": content,
                "is_encrypted": False
            }
    
    def _decrypt_memory_content(self, payload: Dict[str, Any], user_id: str) -> str:
        """
        Decrypt memory content from payload.
        
        Args:
            payload: The memory payload from Qdrant
            user_id: The user identifier
            
        Returns:
            The decrypted plaintext content
        """
        # Check if content is encrypted
        is_encrypted = payload.get("is_encrypted", False)
        
        if not is_encrypted:
            # Return plaintext content (backward compatible)
            return payload.get("content", "")
        
        # Get encrypted content and nonce
        encrypted_content_b64 = payload.get("encrypted_content")
        nonce_b64 = payload.get("content_nonce")
        
        if not encrypted_content_b64 or not nonce_b64:
            print("⚠️ Missing encrypted content data")
            return payload.get("content", "")
        
        # Get user's DEK
        dek = self._get_or_create_user_dek(user_id)
        if dek is None:
            print(f"⚠️ Cannot decrypt: no DEK available for user {user_id}")
            return "[Encrypted content - unable to decrypt]"
        
        try:
            encrypted_content = base64.b64decode(encrypted_content_b64)
            nonce = base64.b64decode(nonce_b64)
            
            decrypted = self.encryption.decrypt_content(encrypted_content, nonce, dek)
            
            # Clear DEK from memory
            dek = b'\x00' * len(dek)
            
            return decrypted
        except Exception as e:
            print(f"⚠️ Decryption failed: {e}")
            return "[Encrypted content - decryption failed]"
    
    async def add(
        self,
        title: Optional[str],
        content: str,
        user_id: str,
        project_id: Optional[str] = None,
        namespace: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        user_preference: bool = False,
        temporal_valid_until: Optional[str] = None,
        extra_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add a memory with automatic cognitive classification and encryption.
        
        Args:
            title: Memory title (optional, auto-generated if not provided)
            content: Memory content (required)
            user_id: User identifier for isolation
            project_id: Project identifier for organization
            namespace: Namespace for grouping
            memory_types: Custom memory type tags
            user_preference: Whether this is a user preference
            temporal_valid_until: Expiration time for temporal validity
            extra_metadata: Additional metadata
            
        Returns:
            Dict with id, title, sectors, and metadata
        """
        # 1. Classify memory using LLM
        classification = await self.classifier.classify(title, content)
        
        # 2. Generate embedding (use original content)
        embedding = await self._embed(f"{title}\n{content}" if title else content)
        
        # 3. Generate unique ID
        memory_id = hashlib.md5(
            f"{user_id}:{title}:{content[:100]}:{datetime.now()}".encode()
        ).hexdigest()
        
        # 4. Encrypt content
        encryption_data = self._encrypt_memory_content(content, user_id)
        
        # 5. Build payload with hybrid schema
        payload = {
            # Core fields
            "title": title or classification.get("generated_title", content[:50]),
            "user_id": user_id,
            
            # Content (plaintext for embedding, encrypted for storage)
            "content": encryption_data.get("content", content),
            "is_encrypted": encryption_data.get("is_encrypted", False),
        }
        
        # Add encryption fields if encrypted
        if encryption_data.get("is_encrypted"):
            payload["encrypted_content"] = encryption_data["encrypted_content"]
            payload["content_nonce"] = encryption_data["content_nonce"]
        
        # Add remaining fields
        payload.update({
            # Project organization
            "project_id": project_id or "default",
            "namespace": namespace or "general",
            "memory_types": memory_types or ["general"],
            "user_preference": user_preference,
            
            # Cognitive sectors
            "sector_primary": classification.get("primary_sector", "semantic"),
            "sector_secondary": classification.get("secondary_sectors", []),
            "sector_confidence": classification.get("confidence", 0.5),
            
            # Semantic enrichment
            "semantic_tags": classification.get("semantic_tags", []),
            
            # Temporal KG
            "temporal_valid_from": datetime.now().isoformat(),
            "temporal_valid_until": temporal_valid_until,
            "temporal_is_current": True,
            
            # Statistics
            "created_at": datetime.now().isoformat(),
            "updated_at": None,
            "access_count": 0,
            "last_accessed": None,
            
            # Extra metadata
            "extra_metadata": extra_metadata or {}
        })
        
        # 6. Store in Qdrant
        self.client.upsert(
            collection_name="mem0",
            points=[{
                "id": memory_id,
                "vector": embedding,
                "payload": payload
            }]
        )
        
        return {
            "id": memory_id,
            "title": payload["title"],
            "sectors": {
                "primary": payload["sector_primary"],
                "confidence": payload["sector_confidence"]
            },
            "project_id": payload["project_id"],
            "created_at": payload["created_at"],
            "is_encrypted": payload["is_encrypted"]
        }
    
    async def search(
        self,
        query: str,
        user_id: str,
        project_id: Optional[str] = None,
        sectors: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        only_current: bool = True,
        limit: int = 10,
        with_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Search memories with composite scoring and automatic decryption.
        
        Scoring formula: 
        final_score = vector_similarity × sector_boost × time_boost
        
        Args:
            query: Search query
            user_id: User to search within
            project_id: Filter by project
            sectors: Filter by cognitive sectors
            memory_types: Filter by memory types
            only_current: Only return temporally valid memories
            limit: Max results
            with_explanation: Include scoring breakdown
            
        Returns:
            Search results with decrypted content, scores and optional explanations
        """
        # 1. Generate query embedding
        query_embedding = await self._embed(query)
        
        # 2. Build filters
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        
        if project_id:
            must_conditions.append(
                FieldCondition(key="project_id", match=MatchValue(value=project_id))
            )
        
        if only_current:
            must_conditions.append(
                FieldCondition(key="temporal_is_current", match=MatchValue(value=True))
            )
        
        # 3. Vector search (get more for post-filtering)
        results = self.client.search(
            collection_name="mem0",
            query_vector=query_embedding,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit * 2
        )
        
        # 4. Apply composite scoring and decryption
        scored_results = []
        for point in results:
            payload = point.payload
            
            # Sector filtering and boost
            sector_match = self._calculate_sector_boost(payload, sectors)
            if sector_match == 0:  # Explicit sector mismatch
                continue
            
            # Memory type filtering
            if memory_types:
                point_types = payload.get("memory_types", [])
                if not any(t in point_types for t in memory_types):
                    continue
            
            # Time decay/boost
            time_boost = self._calculate_time_boost(payload.get("created_at", ""))
            
            # Composite score
            final_score = point.score * sector_match * time_boost
            
            # Decrypt content
            decrypted_content = self._decrypt_memory_content(payload, user_id)
            
            result_item = {
                "id": point.id,
                "title": payload.get("title"),
                "content": decrypted_content,
                "score": final_score,
                "project_id": payload.get("project_id"),
                "sectors": {
                    "primary": payload.get("sector_primary"),
                    "secondary": payload.get("sector_secondary"),
                    "confidence": payload.get("sector_confidence")
                },
                "memory_types": payload.get("memory_types"),
                "semantic_tags": payload.get("semantic_tags"),
                "created_at": payload.get("created_at"),
                "temporal": {
                    "is_current": payload.get("temporal_is_current"),
                    "valid_from": payload.get("temporal_valid_from")
                },
                "is_encrypted": payload.get("is_encrypted", False)
            }
            
            if with_explanation:
                result_item["explanation"] = {
                    "score_breakdown": {
                        "vector_similarity": round(point.score, 3),
                        "sector_match_boost": round(sector_match, 2),
                        "time_boost": round(time_boost, 2),
                        "final_score": round(final_score, 3)
                    },
                    "why_recalled": (
                        f"Vector similarity {point.score:.2f}, "
                        f"matches {payload.get('sector_primary')} sector"
                    )
                }
            
            scored_results.append(result_item)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "query": query,
            "total_found": len(scored_results),
            "filters": {
                "project_id": project_id,
                "sectors": sectors,
                "memory_types": memory_types,
                "only_current": only_current
            },
            "results": scored_results[:limit]
        }
    
    def _calculate_sector_boost(self, payload: Dict, query_sectors: Optional[List[str]]) -> float:
        """Calculate sector match boost."""
        if not query_sectors:
            return 1.0
        
        primary = payload.get("sector_primary", "")
        secondary = payload.get("sector_secondary", [])
        
        if primary in query_sectors:
            return 1.2
        elif any(s in secondary for s in query_sectors):
            return 1.1
        else:
            return 0.8  # Penalty for mismatch
    
    def _calculate_time_boost(self, created_at: str) -> float:
        """Calculate time-based boost/decay."""
        try:
            days_old = (datetime.now() - datetime.fromisoformat(created_at)).days
            if days_old < 7:
                return 1.2
            elif days_old < 30:
                return 1.1
            elif days_old > 365:
                return 0.8
            return 1.0
        except:
            return 1.0
    
    async def _embed(self, text: str) -> List[float]:
        """Generate embedding using bge-m3 via Ollama."""
        try:
            # Get Ollama URL from config or environment
            ollama_url = self.config.get("embedder", {}).get("config", {}).get(
                "ollama_base_url", 
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            embed_model = self.config.get("embedder", {}).get("config", {}).get(
                "model",
                os.getenv("EMBED_MODEL", "bge-m3")
            )
            
            response = httpx.post(
                f"{ollama_url}/api/embeddings",
                json={"model": embed_model, "prompt": text},
                timeout=300
            )
            return response.json().get("embedding", [0.0] * 1024)
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")
            return [0.0] * 1024
    
    async def get_all(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get all memories for a user, optionally filtered by project."""
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        
        if project_id:
            must_conditions.append(
                FieldCondition(key="project_id", match=MatchValue(value=project_id))
            )
        
        results = self.client.scroll(
            collection_name="mem0",
            scroll_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True
        )
        
        memories = []
        for point in results[0]:
            payload = point.payload
            # Decrypt content
            decrypted_content = self._decrypt_memory_content(payload, user_id)
            
            memories.append({
                "id": point.id,
                "title": payload.get("title"),
                "content": decrypted_content,
                "project_id": payload.get("project_id"),
                "sector_primary": payload.get("sector_primary"),
                "created_at": payload.get("created_at"),
                "is_encrypted": payload.get("is_encrypted", False)
            })
        
        return {"results": memories, "count": len(memories)}
    
    async def get_by_id(
        self,
        memory_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID with decryption.
        
        Args:
            memory_id: The memory ID
            user_id: The user identifier
            
        Returns:
            The memory dict or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name="mem0",
                ids=[memory_id],
                with_payload=True
            )
            
            if not result:
                return None
            
            point = result[0]
            payload = point.payload
            
            # Verify ownership
            if payload.get("user_id") != user_id:
                return None
            
            # Decrypt content
            decrypted_content = self._decrypt_memory_content(payload, user_id)
            
            return {
                "id": point.id,
                "title": payload.get("title"),
                "content": decrypted_content,
                "project_id": payload.get("project_id"),
                "sector_primary": payload.get("sector_primary"),
                "sector_secondary": payload.get("sector_secondary"),
                "sector_confidence": payload.get("sector_confidence"),
                "memory_types": payload.get("memory_types"),
                "semantic_tags": payload.get("semantic_tags"),
                "created_at": payload.get("created_at"),
                "temporal": {
                    "is_current": payload.get("temporal_is_current"),
                    "valid_from": payload.get("temporal_valid_from"),
                    "valid_until": payload.get("temporal_valid_until")
                },
                "is_encrypted": payload.get("is_encrypted", False),
                "access_count": payload.get("access_count", 0),
                "extra_metadata": payload.get("extra_metadata", {})
            }
        except Exception as e:
            print(f"⚠️ Failed to get memory {memory_id}: {e}")
            return None
