"""
ModelEvictionPolicy: Pluggable policy for choosing which models to evict.

Similar to EvictionPolicy but for model weights rather than KV caches.
Different workloads need different strategies:
- Multi-model serving: LRU (evict least-recently-used model)
- Priority-based: Never evict critical models
- Size-aware: Prefer evicting large models to free more space
- Predictive: ML-learned (predict next model access)

This interface enables workload-specific optimization without core changes.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ModelEvictionCandidate:
    """Information about a model that could be evicted."""
    model_id: str
    last_access: float          # Unix timestamp of last execution
    size_bytes: int             # Model weight size
    is_active: bool             # Whether currently executing
    priority: int = 0           # Priority (higher = keep longer)


class ModelEvictionPolicy(ABC):
    """
    Pluggable policy for selecting which model to evict under memory pressure.
    
    The coordinator calls this when ring buffer is full and needs to load a new model.
    The policy decides which model's weights to swap to host RAM.
    """
    
    @abstractmethod
    def select_victim(
        self,
        candidates: List[ModelEvictionCandidate],
        bytes_needed: int
    ) -> Optional[str]:
        """
        Select one model to evict to free space.
        
        Args:
            candidates: List of models that can be evicted
            bytes_needed: Target bytes to free
        
        Returns:
            model_id to evict, or None if no suitable victim
        
        Constraints:
            - MUST NOT evict active models (is_active=True)
            - Should prefer models that free enough space
            - Can return None if eviction would be harmful
        
        Example (LRU):
            # Filter out active models
            eligible = [c for c in candidates if not c.is_active]
            if not eligible:
                return None
            # Select least-recently-used
            return min(eligible, key=lambda c: c.last_access).model_id
        
        Example (Size-aware):
            # Prefer evicting large models to free more space
            eligible = [c for c in candidates if not c.is_active]
            # Sort by size descending
            eligible_sorted = sorted(eligible, key=lambda c: c.size_bytes, reverse=True)
            # Return largest that frees enough
            for c in eligible_sorted:
                if c.size_bytes >= bytes_needed:
                    return c.model_id
            # If none large enough, return largest
            return eligible_sorted[0].model_id if eligible_sorted else None
        """
        pass

