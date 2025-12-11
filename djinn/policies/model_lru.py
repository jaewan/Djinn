"""
Model LRU Eviction Policy: Least-Recently-Used model eviction.

Evicts the model that was accessed longest ago, with safety constraints:
- Never evict active models (currently executing)
- Never evict pinned models (priority > 0)
"""

import logging
from typing import List, Optional
from djinn.interfaces.model_eviction_policy import (
    ModelEvictionPolicy,
    ModelEvictionCandidate,
)

logger = logging.getLogger(__name__)


class ModelLRUPolicy(ModelEvictionPolicy):
    """
    Least-Recently-Used eviction policy for model weights.
    
    Selects the model that was accessed longest ago, excluding:
    - Active models (currently executing)
    - Pinned models (priority > 0)
    """
    
    def select_victim(
        self,
        candidates: List[ModelEvictionCandidate],
        bytes_needed: int
    ) -> Optional[str]:
        """
        Select least-recently-used model for eviction.
        
        Args:
            candidates: List of models that could be evicted
            bytes_needed: Target bytes to free
        
        Returns:
            model_id to evict, or None if no eligible candidates
        """
        # Filter out ineligible models
        eligible = [
            c for c in candidates
            if not c.is_active and c.priority <= 0
        ]
        
        if not eligible:
            logger.warning(
                f"No eligible models for eviction "
                f"(need {bytes_needed / 1024**2:.1f}MB, "
                f"{len(candidates)} candidates, all active or pinned)"
            )
            return None
        
        # Select least-recently-used
        victim = min(eligible, key=lambda c: c.last_access)
        
        logger.debug(
            f"Selected for LRU eviction: {victim.model_id[:16]}... "
            f"(last_access={victim.last_access:.1f}, size={victim.size_bytes / 1024**2:.1f}MB)"
        )
        
        return victim.model_id

