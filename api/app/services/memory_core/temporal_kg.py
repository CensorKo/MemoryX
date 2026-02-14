"""
Temporal Knowledge Graph - Time-aware memory validity.

Tracks when memories were true and their evolution over time.
Supports:
- Validity periods (valid_from, valid_until)
- Memory supersession (newer facts replace older ones)
- Point-in-time queries (what was true at X?)
- Timeline reconstruction
"""

from datetime import datetime
from typing import Optional, List, Dict, Any


class TemporalKnowledgeGraph:
    """
    Temporal layer for memory validity tracking.
    
    Unlike simple TTL expiration, tracks the evolution of truth:
    - "We used Vue 2" (valid 2023-01 to 2024-06)
    - "We now use Vue 3" (valid 2024-06 to present, supersedes above)
    """
    
    def __init__(self, memory_service):
        """
        Initialize TKG with reference to memory service.
        
        Args:
            memory_service: MemoryService instance for storage operations
        """
        self.memory_service = memory_service
    
    async def add_with_temporal(
        self,
        title: str,
        content: str,
        user_id: str,
        entity: str,  # Entity this memory is about (e.g., "tech_stack", "architecture")
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        supersedes: Optional[str] = None,  # ID of memory this replaces
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add a memory with temporal validity.
        
        Args:
            title: Memory title
            content: Memory content
            user_id: User identifier
            entity: Entity this fact describes (for timeline grouping)
            valid_from: When this becomes valid (default: now)
            valid_until: When this expires (None = permanent)
            supersedes: ID of previous memory this replaces
            
        Returns:
            Created memory with temporal metadata
        """
        valid_from = valid_from or datetime.now()
        
        # If superseding another memory, mark it as outdated
        if supersedes:
            await self._mark_superseded(supersedes)
        
        # Add the new memory
        result = await self.memory_service.add(
            title=title,
            content=content,
            user_id=user_id,
            temporal_valid_until=valid_until.isoformat() if valid_until else None,
            extra_metadata={
                "temporal_entity": entity,
                "temporal_valid_from": valid_from.isoformat(),
                "supersedes": supersedes,
                "superseded_by": None
            },
            **kwargs
        )
        
        return result
    
    async def get_timeline(
        self,
        entity: str,
        user_id: str,
        project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of an entity's evolution.
        
        Args:
            entity: Entity to track (e.g., "tech_stack")
            user_id: User identifier
            project_id: Optional project filter
            
        Returns:
            Chronological list of states with validity periods
        """
        # Search for all memories about this entity
        results = await self.memory_service.search(
            query=entity,
            user_id=user_id,
            project_id=project_id,
            only_current=False,  # Include historical
            limit=50
        )
        
        # Filter to memories with temporal metadata
        temporal_memories = [
            r for r in results["results"]
            if r.get("temporal", {}).get("valid_from")
        ]
        
        # Sort by valid_from date
        timeline = sorted(
            temporal_memories,
            key=lambda x: x["temporal"]["valid_from"]
        )
        
        # Build timeline with periods
        timeline_view = []
        for i, mem in enumerate(timeline):
            valid_from = mem["temporal"]["valid_from"]
            valid_until = mem["temporal"].get("valid_until")
            
            # Determine period end
            if valid_until:
                period_end = valid_until
            elif i < len(timeline) - 1:
                period_end = timeline[i + 1]["temporal"]["valid_from"]
            else:
                period_end = "present"
            
            timeline_view.append({
                "memory_id": mem["id"],
                "title": mem["title"],
                "content": mem["content"][:200] + "...",
                "period": {
                    "from": valid_from,
                    "to": period_end
                },
                "is_current": mem["temporal"].get("is_current", True),
                "sector": mem["sectors"]["primary"]
            })
        
        return timeline_view
    
    async def query_at_time(
        self,
        entity: str,
        timestamp: datetime,
        user_id: str,
        project_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query what was true about an entity at a specific time.
        
        Args:
            entity: Entity to query
            timestamp: Point in time
            user_id: User identifier
            project_id: Optional project filter
            
        Returns:
            Memory that was valid at that time, or None
        """
        # Get timeline
        timeline = await self.get_timeline(entity, user_id, project_id)
        
        # Find memory valid at timestamp
        for mem in reversed(timeline):  # Check from newest
            valid_from = datetime.fromisoformat(mem["period"]["from"])
            valid_to = mem["period"]["to"]
            
            if valid_from <= timestamp:
                if valid_to == "present" or datetime.fromisoformat(valid_to) >= timestamp:
                    return mem
        
        return None
    
    async def _mark_superseded(self, memory_id: str) -> None:
        """Mark a memory as superseded by a newer one."""
        # Implementation would update the memory in Qdrant
        # to set is_current=False and add superseded_by
        pass
    
    def is_valid_at(self, memory: Dict[str, Any], timestamp: datetime) -> bool:
        """
        Check if a memory was valid at a given timestamp.
        
        Args:
            memory: Memory dict with temporal fields
            timestamp: Time to check
            
        Returns:
            True if memory was valid at timestamp
        """
        temporal = memory.get("temporal", {})
        
        valid_from = temporal.get("valid_from")
        valid_until = temporal.get("valid_until")
        
        if not valid_from:
            return True  # No temporal constraints
        
        try:
            valid_from_dt = datetime.fromisoformat(valid_from)
            
            if valid_from_dt > timestamp:
                return False  # Not yet valid
            
            if valid_until:
                valid_until_dt = datetime.fromisoformat(valid_until)
                if valid_until_dt < timestamp:
                    return False  # No longer valid
            
            return True
            
        except (ValueError, TypeError):
            return True
    
    def get_current_value(self, timeline: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Get current/most recent value from timeline.
        
        Args:
            timeline: Timeline from get_timeline()
            
        Returns:
            Current memory or None
        """
        if not timeline:
            return None
        
        # Return last entry (most recent)
        current = timeline[-1]
        
        if current.get("is_current", True):
            return current
        
        return None
