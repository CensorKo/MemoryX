"""
Composite Scoring - Multi-factor relevance scoring.

Combines multiple signals for more intelligent memory retrieval:
- Vector similarity (semantic match)
- Sector alignment (cognitive relevance)
- Temporal decay/boost (recency)
- Access frequency (popularity)
"""

from datetime import datetime
from typing import Dict, Any


class CompositeScorer:
    """
    Composite scoring engine for memory relevance.
    
    Final Score = VectorSimilarity × SectorBoost × TimeBoost × AccessBoost
    """
    
    # Time decay/boost parameters
    RECENT_DAYS = 7
    RECENT_BOOST = 1.2
    
    MONTH_DAYS = 30
    MONTH_BOOST = 1.1
    
    OLD_DAYS = 365
    OLD_PENALTY = 0.8
    
    # Sector match parameters
    PRIMARY_MATCH_BOOST = 1.2
    SECONDARY_MATCH_BOOST = 1.1
    MISMATCH_PENALTY = 0.8
    
    # Access frequency parameters
    MAX_ACCESS_BOOST = 1.2
    ACCESS_DECAY = 0.02  # Per access
    
    def calculate_score(
        self,
        vector_similarity: float,
        payload: Dict[str, Any],
        query_sectors: list = None,
        access_count: int = 0
    ) -> Dict[str, float]:
        """
        Calculate composite score for a memory.
        
        Args:
            vector_similarity: Base vector similarity score (0-1)
            payload: Memory payload with metadata
            query_sectors: Sectors requested in query
            access_count: Number of times memory was accessed
            
        Returns:
            Dict with final_score and breakdown
        """
        # 1. Sector match boost
        sector_boost = self._calculate_sector_boost(
            payload, 
            query_sectors
        )
        
        # 2. Temporal boost/decay
        time_boost = self._calculate_time_boost(
            payload.get("created_at", "")
        )
        
        # 3. Access frequency boost
        access_boost = self._calculate_access_boost(access_count)
        
        # 4. Calculate final score
        final_score = (
            vector_similarity * 
            sector_boost * 
            time_boost * 
            access_boost
        )
        
        return {
            "final_score": final_score,
            "breakdown": {
                "vector_similarity": round(vector_similarity, 4),
                "sector_boost": round(sector_boost, 2),
                "time_boost": round(time_boost, 2),
                "access_boost": round(access_boost, 2)
            },
            "formula": "vector × sector × time × access"
        }
    
    def _calculate_sector_boost(
        self, 
        payload: Dict[str, Any], 
        query_sectors: list
    ) -> float:
        """
        Calculate sector match boost.
        
        - Primary match: +20%
        - Secondary match: +10%
        - Mismatch: -20%
        """
        if not query_sectors:
            return 1.0
        
        primary = payload.get("sector_primary", "")
        secondary = payload.get("sector_secondary", [])
        
        if primary in query_sectors:
            return self.PRIMARY_MATCH_BOOST
        elif any(s in secondary for s in query_sectors):
            return self.SECONDARY_MATCH_BOOST
        else:
            return self.MISMATCH_PENALTY
    
    def _calculate_time_boost(self, created_at: str) -> float:
        """
        Calculate time-based boost/decay.
        
        - Within 7 days: +20%
        - Within 30 days: +10%
        - Older than 1 year: -20%
        """
        try:
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            days_old = (datetime.now() - created).days
            
            if days_old < self.RECENT_DAYS:
                return self.RECENT_BOOST
            elif days_old < self.MONTH_DAYS:
                return self.MONTH_BOOST
            elif days_old > self.OLD_DAYS:
                return self.OLD_PENALTY
            else:
                return 1.0
                
        except (ValueError, TypeError):
            return 1.0
    
    def _calculate_access_boost(self, access_count: int) -> float:
        """
        Calculate access frequency boost.
        
        More frequently accessed memories get slight boost
        (indicates usefulness).
        """
        boost = 1.0 + (access_count * self.ACCESS_DECAY)
        return min(boost, self.MAX_ACCESS_BOOST)
    
    def explain_score(self, score_result: Dict[str, float]) -> str:
        """
        Generate human-readable explanation of scoring.
        
        Args:
            score_result: Result from calculate_score()
            
        Returns:
            Human-readable explanation
        """
        bd = score_result["breakdown"]
        
        explanations = []
        
        # Vector similarity
        if bd["vector_similarity"] > 0.8:
            explanations.append("high semantic similarity")
        elif bd["vector_similarity"] > 0.5:
            explanations.append("moderate semantic match")
        else:
            explanations.append("weak semantic match")
        
        # Sector boost
        if bd["sector_boost"] > 1.1:
            explanations.append("matches requested cognitive sector")
        elif bd["sector_boost"] < 1.0:
            explanations.append("sector mismatch")
        
        # Time boost
        if bd["time_boost"] > 1.1:
            explanations.append("recently created")
        elif bd["time_boost"] < 1.0:
            explanations.append("older memory")
        
        # Access boost
        if bd["access_boost"] > 1.1:
            explanations.append("frequently accessed")
        
        return "; ".join(explanations) if explanations else "standard relevance"
