"""
Signal-Driven Eviction Policy: Only evict sessions that explicitly signaled readiness.

This is the current Djinn policy, optimized for agent workloads where clients
explicitly signal when KV cache can be safely evicted (IO_WAIT phase).

Benefits:
- No data loss: only evict when client approves
- Predictable: matches application semantics
- Safe: ideal for interactive workloads

Downside:
- Requires client cooperation
- Not applicable to workloads without explicit signals
"""

import logging
from typing import List

from djinn.interfaces import EvictionPolicy, EvictionCandidate

logger = logging.getLogger(__name__)


class SignalDrivenPolicy(EvictionPolicy):
    """
    Eviction policy for signal-aware workloads (e.g., reasoning agents).
    
    Only evicts sessions that have explicitly signaled readiness via
    the semantic phase API (e.g., IO_WAIT signal means "safe to evict").
    
    This prevents data loss and maintains application semantics.
    
    NOTE: Currently active policy for Djinn's agent-focused workloads.
    Post-OSDI, can be swapped for LRUPolicy or custom policies via configuration.
    """
    
    def __init__(self, config: dict = None):
        """Initialize policy."""
        self.config = config or {}
        logger.info("SignalDrivenPolicy initialized")
    
    def select_victims(
        self,
        candidates: List[EvictionCandidate],
        bytes_needed: int
    ) -> List[str]:
        """
        Select only signal-managed sessions for eviction.
        
        Args:
            candidates: Potential sessions to evict
            bytes_needed: Target bytes to free
        
        Returns:
            Session IDs that are signal-managed (safe to evict)
        
        Logic:
            1. Filter to only signal-managed sessions
            2. Sort by last access (LRU within signal-managed)
            3. Return enough to free bytes_needed
        """
        # Filter to signal-managed sessions
        signal_managed = [c for c in candidates if c.is_signal_managed]
        
        if not signal_managed:
            logger.warning(f"No signal-managed sessions available for eviction "
                          f"(needed {bytes_needed} bytes, {len(candidates)} total candidates)")
            return []
        
        # Sort by last access (LRU)
        sorted_victims = sorted(signal_managed, key=lambda c: c.last_access)
        
        # Select until we have enough bytes
        victims = []
        freed_bytes = 0
        for candidate in sorted_victims:
            if freed_bytes >= bytes_needed:
                break
            victims.append(candidate.session_id)
            freed_bytes += candidate.size_bytes
            logger.debug(f"Selected for eviction (signal-driven): {candidate.session_id[:12]} "
                        f"({candidate.size_bytes / (1024**2):.1f}MB, "
                        f"idle {candidate.last_access:.1f}s)")
        
        if freed_bytes < bytes_needed:
            logger.warning(f"Selected {len(victims)} sessions but only freed {freed_bytes / (1024**2):.1f}MB "
                          f"(needed {bytes_needed / (1024**2):.1f}MB)")
        
        return [v for v in victims]

