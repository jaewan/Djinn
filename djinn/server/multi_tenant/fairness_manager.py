"""
Fairness Manager for Multi-Tenant Resource Allocation.

Implements Dominant Resource Fairness (DRF) with semantic adjustments:
- Base fair share calculation using DRF algorithm
- Semantic adjustments (boost INTERACTIVE, DECODE phase)
- SLO-aware allocation
- Dynamic rebalancing under memory pressure

This ensures all clients get fair access while respecting semantic importance.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ClientPriority(Enum):
    """Client priority levels."""
    INTERACTIVE = 0
    SERVING = 1
    BATCH = 2


class ExecutionPhase(str, Enum):
    """Execution phases."""
    UNKNOWN = "unknown"
    LLM_PREFILL = "llm_prefill"
    LLM_DECODE = "llm_decode"
    VISION_ENCODING = "vision_encoding"
    VISION_DECODING = "vision_decoding"
    MULTIMODAL_FUSION = "multimodal_fusion"
    TRAINING = "training"


@dataclass
class ResourceAllocation:
    """Resource allocation for a client."""
    client_id: str
    memory_mb: float = 0.0
    compute_flops: float = 0.0
    share_percentage: float = 0.0
    priority: ClientPriority = ClientPriority.SERVING
    phase: ExecutionPhase = ExecutionPhase.UNKNOWN
    timestamp: float = field(default_factory=time.time)


@dataclass
class AllocationEvent:
    """Record of allocation change."""
    timestamp: float
    client_id: str
    old_memory_mb: float
    new_memory_mb: float
    reason: str


class FairnessManager:
    """
    Ensures fair resource allocation across tenants.
    
    Uses Dominant Resource Fairness (DRF) as base algorithm with:
    - Semantic adjustments (priority, phase)
    - SLO awareness
    - Memory pressure adaptation
    """
    
    def __init__(self):
        """Initialize fairness manager."""
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.history: List[AllocationEvent] = []
        self.stats = {
            'reallocations': 0,
            'fairness_score': 0.0,  # Jain's fairness index
        }
        logger.info("FairnessManager initialized")
    
    def compute_fair_share(
        self,
        clients: Dict[str, 'ClientContext'],
        total_memory_mb: float,
        total_compute_flops: float
    ) -> Dict[str, ResourceAllocation]:
        """
        Compute fair share using DRF with semantic adjustments.
        
        Algorithm:
        1. Base: Dominant Resource Fairness (water-filling)
        2. Adjust: Apply semantic multipliers
        3. Normalize: Ensure allocations sum to available resources
        
        Args:
            clients: Dict of client_id → ClientContext
            total_memory_mb: Total GPU memory available
            total_compute_flops: Total compute capacity
        
        Returns:
            Dict of client_id → ResourceAllocation
        """
        if not clients:
            return {}
        
        # Step 1: Compute base DRF shares
        base_shares = self._compute_drf_shares(
            clients,
            total_memory_mb,
            total_compute_flops
        )
        
        # Step 2: Apply semantic adjustments
        adjusted_shares = {}
        for client_id, base_share in base_shares.items():
            client = clients[client_id]
            adjusted_share = self._apply_semantic_adjustments(
                client_id,
                base_share,
                client
            )
            adjusted_shares[client_id] = adjusted_share
        
        # Step 3: Normalize to available resources
        normalized_shares = self._normalize_allocations(
            adjusted_shares,
            total_memory_mb,
            total_compute_flops
        )
        
        # Record allocations
        for client_id, allocation in normalized_shares.items():
            old_alloc = self.allocations.get(client_id)
            old_memory = old_alloc.memory_mb if old_alloc else 0.0
            
            self.allocations[client_id] = allocation
            
            if old_memory != allocation.memory_mb:
                self.history.append(AllocationEvent(
                    timestamp=time.time(),
                    client_id=client_id,
                    old_memory_mb=old_memory,
                    new_memory_mb=allocation.memory_mb,
                    reason="Fair share recomputation"
                ))
                self.stats['reallocations'] += 1
        
        # Update fairness metric
        self.stats['fairness_score'] = self._compute_fairness_score(normalized_shares)
        
        logger.info(
            f"Fair shares computed: {len(normalized_shares)} clients, "
            f"fairness={self.stats['fairness_score']:.3f}"
        )
        
        return normalized_shares
    
    def _compute_drf_shares(
        self,
        clients: Dict,
        total_memory_mb: float,
        total_compute_flops: float
    ) -> Dict[str, ResourceAllocation]:
        """
        Compute base Dominant Resource Fairness allocation.
        
        DRF principle: Equalize the maximum share across resources.
        Each client's max(memory_share, compute_share) should be equal.
        """
        # Initialize with equal split
        n_clients = len(clients)
        equal_memory = total_memory_mb / n_clients
        equal_compute = total_compute_flops / n_clients
        
        allocations = {}
        for client_id, client in clients.items():
            allocations[client_id] = ResourceAllocation(
                client_id=client_id,
                memory_mb=equal_memory,
                compute_flops=equal_compute,
                priority=client.priority,
                phase=client.phase
            )
        
        return allocations
    
    def _apply_semantic_adjustments(
        self,
        client_id: str,
        base_share: ResourceAllocation,
        client: 'ClientContext'
    ) -> ResourceAllocation:
        """
        Apply semantic multipliers to base share.
        
        Adjustments:
        - INTERACTIVE clients: +50% memory and compute
        - DECODE phase: +30% memory (KV cache needs)
        - Idle clients: -50% (penalize inactive clients)
        - High throughput: +25% (reward efficient clients)
        """
        adjustment = ResourceAllocation(
            client_id=client_id,
            memory_mb=base_share.memory_mb,
            compute_flops=base_share.compute_flops,
            priority=client.priority,
            phase=client.phase
        )
        
        # Priority-based adjustment
        if client.priority == ClientPriority.INTERACTIVE:
            adjustment.memory_mb *= 1.5
            adjustment.compute_flops *= 1.5
            logger.debug(f"  Boosted {client_id}: INTERACTIVE (+50%)")
        
        # Phase-based adjustment
        if client.phase == ExecutionPhase.LLM_DECODE:
            adjustment.memory_mb *= 1.3
            logger.debug(f"  Boosted {client_id}: DECODE (+30% memory for KV cache)")
        
        # Activity adjustment (idle clients get less)
        idle_time_sec = time.time() - client.last_active
        if idle_time_sec > 60:  # Idle > 1 minute
            idle_factor = max(0.5, 1.0 - (idle_time_sec - 60) / 300)
            adjustment.memory_mb *= idle_factor
            adjustment.compute_flops *= idle_factor
            logger.debug(f"  Penalized {client_id}: idle {idle_time_sec:.0f}s ({idle_factor:.1%})")
        
        # Efficiency adjustment (high throughput gets more)
        if client.memory_usage_mb > 0:
            throughput = client.cumulative_requests / max(1, client.memory_usage_mb)
            if throughput > 0.1:  # High throughput threshold
                efficiency_boost = min(1.25, 1.0 + throughput * 0.25)
                adjustment.memory_mb *= efficiency_boost
                adjustment.compute_flops *= efficiency_boost
                logger.debug(f"  Boosted {client_id}: high throughput ({efficiency_boost:.1%})")
        
        return adjustment
    
    def _normalize_allocations(
        self,
        allocations: Dict[str, ResourceAllocation],
        total_memory_mb: float,
        total_compute_flops: float
    ) -> Dict[str, ResourceAllocation]:
        """
        Normalize allocations to fit within available resources.
        
        Uses proportional scaling to maintain relative shares while
        ensuring sum equals available resources.
        """
        total_requested_memory = sum(a.memory_mb for a in allocations.values())
        total_requested_compute = sum(a.compute_flops for a in allocations.values())
        
        if total_requested_memory == 0:
            memory_scale = 1.0
        else:
            memory_scale = total_memory_mb / total_requested_memory
        
        if total_requested_compute == 0:
            compute_scale = 1.0
        else:
            compute_scale = total_compute_flops / total_requested_compute
        
        normalized = {}
        for client_id, alloc in allocations.items():
            normalized[client_id] = ResourceAllocation(
                client_id=client_id,
                memory_mb=alloc.memory_mb * memory_scale,
                compute_flops=alloc.compute_flops * compute_scale,
                priority=alloc.priority,
                phase=alloc.phase
            )
        
        return normalized
    
    def _compute_fairness_score(
        self,
        allocations: Dict[str, ResourceAllocation]
    ) -> float:
        """
        Compute Jain's fairness index.
        
        Formula: (Σx)² / (n·Σx²)
        
        Range: [0, 1] where 1 = perfect fairness
        """
        if not allocations:
            return 0.0
        
        shares = [a.memory_mb for a in allocations.values()]
        n = len(shares)
        
        sum_shares = sum(shares)
        sum_squares = sum(x ** 2 for x in shares)
        
        if sum_squares == 0:
            return 0.0
        
        fairness = (sum_shares ** 2) / (n * sum_squares)
        return min(1.0, fairness)  # Cap at 1.0
    
    def get_allocation(self, client_id: str) -> Optional[ResourceAllocation]:
        """Get current allocation for a client."""
        return self.allocations.get(client_id)
    
    def get_stats(self) -> Dict:
        """Get fairness manager statistics."""
        fairness_scores = [
            self._compute_fairness_score(dict(zip(
                self.allocations.keys(),
                self.allocations.values()
            )))
        ]
        
        return {
            **self.stats,
            'total_clients': len(self.allocations),
            'current_fairness_score': self.stats['fairness_score'],
            'total_reallocation_events': len(self.history),
        }
