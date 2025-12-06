"""
LRU Eviction Policy: Least-Recently-Used for non-signal workloads.

Useful for workloads that don't have explicit signals:
- Batch processing
- Training with checkpointing
- Inference without agent signals

Strategy: Evict oldest accessed sessions first.
"""

import logging
from typing import List

from djinn.interfaces import EvictionPolicy, EvictionCandidate

logger = logging.getLogger(__name__)


class LRUPolicy(EvictionPolicy):
    """
    Least-Recently-Used eviction policy.
    
    Evicts sessions in order of last access time, freeing the oldest
    idle sessions first. This is a standard memory management strategy.
    
    Suitable for:
    - Batch workloads without explicit phase signals
    - Training checkpointing
    - Multi-workload scenarios
    
    NOTE: Reference implementation for future batch workload support.
    Can be activated post-OSDI by setting eviction_policy in djinn.yaml.
    """
    
    def __init__(self, config: dict = None):
        """Initialize policy."""
        self.config = config or {}
        logger.info("LRUPolicy initialized")
    
    def select_victims(
        self,
        candidates: List[EvictionCandidate],
        bytes_needed: int
    ) -> List[str]:
        """
        Select least-recently-used sessions for eviction.
        
        Args:
            candidates: Potential sessions to evict
            bytes_needed: Target bytes to free
        
        Returns:
            Session IDs in LRU order
        
        Logic:
            1. Sort by last_access (oldest first)
            2. Select until we free bytes_needed
            3. Never evict REALTIME QoS
        """
        # Filter out REALTIME (too important to evict)
        evictable = [c for c in candidates if c.qos_class != 'realtime']
        
        if not evictable:
            logger.warning(f"No evictable sessions (all REALTIME or similar)")
            return []
        
        # Sort by last access (LRU)
        sorted_victims = sorted(evictable, key=lambda c: c.last_access)
        
        # Select until we have enough bytes
        victims = []
        freed_bytes = 0
        for candidate in sorted_victims:
            if freed_bytes >= bytes_needed:
                break
            victims.append(candidate.session_id)
            freed_bytes += candidate.size_bytes
            logger.debug(f"Selected for LRU eviction: {candidate.session_id[:12]} "
                        f"({candidate.size_bytes / (1024**2):.1f}MB, "
                        f"qos={candidate.qos_class})")
        
        if freed_bytes < bytes_needed:
            logger.warning(f"LRU: Selected {len(victims)} sessions but only freed "
                          f"{freed_bytes / (1024**2):.1f}MB (needed {bytes_needed / (1024**2):.1f}MB)")
        
        return victims

