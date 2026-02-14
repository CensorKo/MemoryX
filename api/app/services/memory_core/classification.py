"""
Memory Classifier - LLM-powered cognitive sector classification.

Inspired by human memory systems, classifies memories into 5 sectors:
- Episodic: Events, conversations, experiences
- Semantic: Facts, knowledge, user preferences
- Procedural: Steps, processes, how-to guides
- Emotional: Feelings, satisfaction, complaints
- Reflective: Insights, patterns, recommendations
"""

import json
import httpx
from typing import Dict, List, Any, Optional


class MemoryClassifier:
    """
    LLM-based memory classifier using Ollama.
    
    Automatically categorizes memories into cognitive sectors
    for more intelligent retrieval.
    """
    
    # Sector definitions for LLM prompt
    SECTOR_DEFINITIONS = {
        "episodic": "Specific events, conversations, meetings, what happened",
        "semantic": "Facts, knowledge, tech stack, user preferences, configurations",
        "procedural": "Steps, workflows, how-to guides, operations, deployment",
        "emotional": "Feelings, satisfaction, complaints, excitement, frustration",
        "reflective": "Insights, patterns, lessons learned, recommendations"
    }
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize classifier with LLM configuration.
        
        Args:
            llm_config: Configuration for Ollama LLM
        """
        self.config = llm_config
        self.base_url = llm_config["config"]["ollama_base_url"]
        self.model = llm_config["config"]["model"]
    
    async def classify(self, title: Optional[str], content: str) -> Dict[str, Any]:
        """
        Classify memory content into cognitive sectors.
        
        Args:
            title: Memory title (optional)
            content: Memory content
            
        Returns:
            Classification result with primary/secondary sectors,
            confidence score, and semantic tags
        """
        prompt = self._build_prompt(title, content)
        
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=300
            )
            
            result = response.json()
            classification = json.loads(result.get("response", "{}"))
            
            # Validate and normalize
            return self._normalize_classification(classification, title, content)
            
        except Exception as e:
            print(f"⚠️ Classification failed: {e}")
            return self._fallback_classification(title, content)
    
    def _build_prompt(self, title: Optional[str], content: str) -> str:
        """Build classification prompt for LLM."""
        sector_desc = "\n".join([
            f"  - {k}: {v}" 
            for k, v in self.SECTOR_DEFINITIONS.items()
        ])
        
        return f"""Analyze the following memory and classify it into cognitive sectors.

Memory Title: {title or "N/A"}
Memory Content:
{content[:800]}...

Sector Definitions:
{sector_desc}

Tasks:
1. Determine PRIMARY sector (most relevant one)
2. Determine SECONDARY sectors (0-2 additional relevant sectors)
3. Extract 5-10 semantic keywords/tags
4. If title is empty/missing, generate a concise title (<50 chars)
5. Assign confidence score (0.0-1.0)

Output JSON:
{{
  "primary_sector": "semantic",
  "secondary_sectors": ["procedural"],
  "confidence": 0.92,
  "semantic_tags": ["docker", "deployment", "git", "workflow"],
  "generated_title": "Docker deployment workflow"
}}

Response (JSON only):"""
    
    def _normalize_classification(
        self, 
        classification: Dict, 
        title: Optional[str], 
        content: str
    ) -> Dict[str, Any]:
        """Normalize and validate classification result."""
        # Ensure valid primary sector
        primary = classification.get("primary_sector", "semantic")
        if primary not in self.SECTOR_DEFINITIONS:
            primary = "semantic"
        
        # Ensure valid secondary sectors
        secondary = classification.get("secondary_sectors", [])
        secondary = [s for s in secondary if s in self.SECTOR_DEFINITIONS and s != primary]
        secondary = secondary[:2]  # Max 2 secondary
        
        # Ensure confidence in range
        confidence = max(0.0, min(1.0, classification.get("confidence", 0.5)))
        
        # Ensure semantic tags
        tags = classification.get("semantic_tags", [])
        if not tags:
            # Extract simple keywords from content
            tags = self._extract_basic_keywords(content)
        
        # Generate title if needed
        generated_title = classification.get("generated_title")
        if not title and not generated_title:
            generated_title = content[:50]
        
        return {
            "primary_sector": primary,
            "secondary_sectors": secondary,
            "confidence": confidence,
            "semantic_tags": tags[:10],  # Max 10 tags
            "generated_title": generated_title or title
        }
    
    def _fallback_classification(
        self, 
        title: Optional[str], 
        content: str
    ) -> Dict[str, Any]:
        """Fallback classification when LLM fails."""
        # Simple keyword-based fallback
        content_lower = content.lower()
        
        # Check for procedural indicators
        if any(w in content_lower for w in ["step", "how to", "guide", "deploy", "install"]):
            primary = "procedural"
        # Check for emotional indicators
        elif any(w in content_lower for w in ["like", "love", "hate", "frustrated", "happy"]):
            primary = "emotional"
        # Check for episodic indicators
        elif any(w in content_lower for w in ["yesterday", "meeting", "discussed", "we talked"]):
            primary = "episodic"
        # Check for reflective indicators
        elif any(w in content_lower for w in ["should", "recommend", "lesson", "insight"]):
            primary = "reflective"
        else:
            primary = "semantic"
        
        return {
            "primary_sector": primary,
            "secondary_sectors": [],
            "confidence": 0.5,
            "semantic_tags": self._extract_basic_keywords(content),
            "generated_title": title or content[:50]
        }
    
    def _extract_basic_keywords(self, content: str) -> List[str]:
        """Extract basic keywords as fallback."""
        # Simple keyword extraction (in production, use NLP library)
        words = content.lower().split()
        # Filter common stop words and short words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}
        keywords = [w for w in words if len(w) > 4 and w not in stop_words]
        return list(set(keywords))[:10]  # Deduplicate and limit
