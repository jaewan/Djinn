"""
EvictionPolicy: Pluggable policy for choosing which sessions to evict.

Different workloads need different eviction strategies:
- Interactive agents: Signal-driven (only evict when agent signals)
- Batch training: LRU (evict least-recently-used)
- QoS-aware: Priority (never evict REALTIME, prefer BATCH)
- Predictive: ML-learned (predict next access time)

This interface enables workload-specific optimization without core changes.
"""

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class EvictionCandidate:
    """Information about a session that could be evicted."""
    session_id: str
    last_access: float          # Unix timestamp of last GPU operation
    size_bytes: int             # KV cache size
    qos_class: str              # QoS class (e.g., 'batch', 'interactive')
    is_signal_managed: bool     # Whether client is using signal_phase()


class EvictionPolicy(ABC):
    """
    Pluggable policy for selecting which sessions to evict under memory pressure.
    
    The scheduler calls this when GPU memory is exhausted and needs to free space.
    The policy decides which sessions' KV caches to swap to CPU.
    """
    
    @abstractmethod
    def select_victims(
        self, 
        candidates: List[EvictionCandidate], 
        bytes_needed: int
    ) -> List[str]:
        """
        Select sessions to evict to free at least bytes_needed.
        
        Args:
            candidates: List of sessions that can be evicted
            bytes_needed: Target bytes to free
        
        Returns:
            List of session_ids to evict (in order, if priority matters)
        
        Constraints:
            - Must return enough sessions to free bytes_needed (if possible)
            - Can return empty list if eviction would be harmful
            - Should not evict same session twice
        
        Example (LRU):
            candidates_sorted = sorted(candidates, key=lambda c: c.last_access)
            victims = []
            freed = 0
            for c in candidates_sorted:
                if freed >= bytes_needed:
                    break
                victims.append(c.session_id)
                freed += c.size_bytes
            return victims
        
        Example (Signal-driven):
            # Only evict if client explicitly signaled readiness
            return [c.session_id for c in candidates if c.is_signal_managed]
        """
        pass

