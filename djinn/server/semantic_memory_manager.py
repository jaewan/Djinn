"""
Semantic Memory Manager for Phase 2 Enhancement.

Leverages SRG structure for intelligent memory decisions:
1. Lifetime-based eviction (evict when last consumer finishes)
2. Semantic eviction priorities (understand data importance)
3. Phase-aware memory budgets (different strategies for prefill vs decode)
4. Cost-based recomputation (should we recompute or cache?)

This module implements optimizations ONLY POSSIBLE with semantic information.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from djinn.core.types import ExecutionPhase

logger = logging.getLogger(__name__)


class DataResidency(str, Enum):
    """Data lifetime and properties."""
    EPHEMERAL_ACTIVATION = "ephemeral_activation"  # Temporary, can discard
    PERSISTENT_WEIGHT = "persistent_weight"        # Model params, keep
    STATEFUL_KV_CACHE = "stateful_kv_cache"       # Accumulating state
    GRADIENT = "gradient"


class EvictionPriority(int, Enum):
    """Eviction priority (lower = evict first)."""
    CRITICAL = 10      # Must keep (e.g., KV cache during decode)
    HIGH = 7           # Important (e.g., weights for active phase)
    MEDIUM = 5         # Normal (e.g., most activations)
    LOW = 3            # Can recompute cheaply
    EPHEMERAL = 1      # Temporary, discard immediately


@dataclass
class TensorLifetime:
    """Lifetime information for a tensor in the computation graph."""
    tensor_id: str
    producer_node_id: str         # Node that creates this tensor
    consumer_node_ids: Set[str]   # Nodes that use this tensor
    last_consumer_node_id: str    # Last node in execution order
    data_residency: DataResidency
    execution_phase: ExecutionPhase
    size_bytes: int
    flop_cost: int                # FLOPs to recompute
    can_evict_immediately: bool   # Can evict after last consumer
    eviction_priority: EvictionPriority


class LifetimeBasedEvictor:
    """
    Evict tensors based on computation graph lifetime analysis.
    
    Novel capability: Only possible with SRG!
    - Traditional: LRU eviction (based on wall-clock time)
    - Djinn: Evict immediately when last consumer finishes
    
    This enables fine-grained memory management without guessing.
    """
    
    def __init__(self):
        """Initialize lifetime-based evictor."""
        self.tensor_lifetimes: Dict[str, TensorLifetime] = {}
        self.node_execution_order: Dict[str, int] = {}
        self.stats = {
            "lifetime_analyses": 0,
            "early_evictions": 0,
            "memory_saved_bytes": 0,
        }
        logger.info("LifetimeBasedEvictor initialized")
    
    def analyze_graph_lifetimes(self, srg_nodes: List[Dict], srg_edges: List[Dict]) -> None:
        """
        Analyze SRG to determine tensor lifetimes.
        
        Args:
            srg_nodes: List of nodes with {id, operation, output_id, metadata}
            srg_edges: List of edges with {source_id, target_id, tensor_id, metadata}
        """
        # Build execution order
        self.node_execution_order = {node['id']: idx for idx, node in enumerate(srg_nodes)}
        
        # Build producer/consumer map
        producers = {}
        consumers = {}
        
        for edge in srg_edges:
            tensor_id = edge.get('tensor_id', f"{edge['source_id']}_{edge['target_id']}")
            producers[tensor_id] = edge['source_id']
            if tensor_id not in consumers:
                consumers[tensor_id] = set()
            consumers[tensor_id].add(edge['target_id'])
        
        # Analyze each tensor
        for tensor_id, producer_id in producers.items():
            consumer_ids = consumers.get(tensor_id, set())
            
            if not consumer_ids:
                # Dead tensor (produced but not consumed)
                last_consumer = producer_id
                can_evict = True
            else:
                # Find last consumer
                last_consumer = max(
                    consumer_ids,
                    key=lambda nid: self.node_execution_order.get(nid, 0)
                )
                can_evict = True
            
            # Find source node metadata
            producer_metadata = self._find_node_metadata(srg_nodes, producer_id)
            
            lifetime = TensorLifetime(
                tensor_id=tensor_id,
                producer_node_id=producer_id,
                consumer_node_ids=consumer_ids,
                last_consumer_node_id=last_consumer,
                data_residency=DataResidency(
                    producer_metadata.get('residency', 'ephemeral_activation')
                ),
                execution_phase=ExecutionPhase(
                    producer_metadata.get('phase', 'unknown')
                ),
                size_bytes=producer_metadata.get('memory_bytes', 0),
                flop_cost=producer_metadata.get('flop_cost', 0),
                can_evict_immediately=can_evict,
                eviction_priority=self._compute_priority(producer_metadata)
            )
            
            self.tensor_lifetimes[tensor_id] = lifetime
        
        self.stats["lifetime_analyses"] += 1
        logger.info(
            "Analyzed %d tensor lifetimes (producers=%d, consumers=%d)",
            len(self.tensor_lifetimes),
            len(producers),
            len(consumers)
        )
    
    def get_tensors_to_evict_after_node(self, node_id: str) -> List[str]:
        """
        Get tensors that should be evicted after a node completes.
        
        Returns: List of tensor IDs whose lifetime ends at this node
        """
        to_evict = []
        for tensor_id, lifetime in self.tensor_lifetimes.items():
            if (lifetime.last_consumer_node_id == node_id and 
                lifetime.can_evict_immediately):
                to_evict.append(tensor_id)
        
        return to_evict
    
    def _find_node_metadata(self, nodes: List[Dict], node_id: str) -> Dict:
        """Find node metadata by ID."""
        for node in nodes:
            if node['id'] == node_id:
                return node.get('metadata', {})
        return {}
    
    def _compute_priority(self, metadata: Dict) -> EvictionPriority:
        """Compute eviction priority from node metadata."""
        residency = metadata.get('residency', 'ephemeral_activation')
        phase = metadata.get('phase', 'unknown')
        
        # KV cache during decode: CRITICAL (don't evict)
        if residency == 'stateful_kv_cache' and phase == 'llm_decode':
            return EvictionPriority.CRITICAL
        
        # Weights: HIGH (keep as long as needed)
        if residency == 'persistent_weight':
            return EvictionPriority.HIGH
        
        # KV cache during prefill: MEDIUM (can recompute)
        if residency == 'stateful_kv_cache':
            return EvictionPriority.MEDIUM
        
        # Activations: LOW to EPHEMERAL (recompute if needed)
        flops = metadata.get('flop_cost', 1000)
        if flops < 1000:
            return EvictionPriority.EPHEMERAL
        return EvictionPriority.LOW
    
    def get_stats(self) -> Dict:
        """Get lifetime analysis statistics."""
        return {
            **self.stats,
            "total_tensors_analyzed": len(self.tensor_lifetimes),
        }


class PhaseAwareMemoryManager:
    """
    Adjust memory allocation strategy based on execution phase.
    
    Novel capability: Only possible with semantic phase annotations!
    - Traditional: Fixed memory allocation
    - Djinn: Different strategies for prefill vs decode
    """
    
    def __init__(self, total_gpu_memory_mb: float):
        """
        Initialize phase-aware memory manager.
        
        Args:
            total_gpu_memory_mb: Total GPU memory in MB
        """
        self.total_memory_mb = total_gpu_memory_mb
        self.current_phase = ExecutionPhase.UNKNOWN
        self.budgets: Dict[str, float] = {}
        self.stats = {
            "phase_switches": 0,
            "budget_violations": 0,
        }
        logger.info("PhaseAwareMemoryManager initialized (%.0f MB)", total_gpu_memory_mb)
    
    def adjust_for_phase(self, phase: ExecutionPhase) -> None:
        """
        Adjust memory budgets based on execution phase.
        
        Uses semantic annotations to predict memory needs.
        """
        if phase == self.current_phase:
            return  # No change
        
        if phase == ExecutionPhase.LLM_PREFILL:
            # Prefill: Parallel attention, needs activation memory
            self.budgets = {
                'weights': 0.3 * self.total_memory_mb,      # 30%
                'activations': 0.6 * self.total_memory_mb,  # 60%
                'kv_cache': 0.1 * self.total_memory_mb      # 10%
            }
            logger.info("Memory budget adjusted for LLM_PREFILL")
        
        elif phase == ExecutionPhase.LLM_DECODE:
            # Decode: Sequential, memory-bound, growing KV cache
            self.budgets = {
                'weights': 0.3 * self.total_memory_mb,      # 30%
                'activations': 0.1 * self.total_memory_mb,  # 10%
                'kv_cache': 0.6 * self.total_memory_mb      # 60%
            }
            logger.info("Memory budget adjusted for LLM_DECODE")
        
        elif phase == ExecutionPhase.VISION_ENCODING:
            # Vision: Conv-heavy, needs intermediate feature maps
            self.budgets = {
                'weights': 0.4 * self.total_memory_mb,      # 40%
                'activations': 0.6 * self.total_memory_mb,  # 60%
                'kv_cache': 0.0                             # 0%
            }
            logger.info("Memory budget adjusted for VISION_ENCODING")
        
        else:
            # Unknown: Conservative allocation
            self.budgets = {
                'weights': 0.5 * self.total_memory_mb,
                'activations': 0.4 * self.total_memory_mb,
                'kv_cache': 0.1 * self.total_memory_mb
            }
        
        self.current_phase = phase
        self.stats["phase_switches"] += 1
    
    def check_allocation(self, category: str, size_mb: float) -> bool:
        """
        Check if allocation would exceed phase budget.
        
        Args:
            category: One of 'weights', 'activations', 'kv_cache'
            size_mb: Size in MB
        
        Returns:
            True if allocation fits within budget
        """
        if not self.budgets:
            # No budget set yet
            return size_mb <= self.total_memory_mb
        
        return size_mb <= self.budgets.get(category, 0)
    
    def get_eviction_priority_order(self) -> List[str]:
        """
        Get eviction priority based on current phase.
        
        Returns: List of categories in eviction order (lowest priority first)
        """
        if self.current_phase == ExecutionPhase.LLM_DECODE:
            # Decode: Never evict KV cache, evict activations first
            return ['activations', 'weights', 'kv_cache']
        
        elif self.current_phase == ExecutionPhase.LLM_PREFILL:
            # Prefill: Activations important, can evict old KV cache
            return ['kv_cache', 'weights', 'activations']
        
        elif self.current_phase == ExecutionPhase.VISION_ENCODING:
            # Vision: Activations important, no KV cache
            return ['weights', 'activations']
        
        else:
            # Default: evict by recency
            return ['activations', 'kv_cache', 'weights']
    
    def get_stats(self) -> Dict:
        """Get memory management statistics."""
        return {
            **self.stats,
            "current_phase": self.current_phase.value,
            "budgets_mb": self.budgets,
        }


class RecomputationVsStorageDecider:
    """
    Decide whether to cache or recompute based on cost model.
    
    Heuristic: If recomputation cost < storage + retrieval cost, recompute.
    """
    
    def __init__(self, network_bandwidth_gbps: float = 10.0):
        """
        Initialize recomputation decision helper.
        
        Args:
            network_bandwidth_gbps: Network bandwidth in Gbps
        """
        self.network_bandwidth_gbps = network_bandwidth_gbps
        self.stats = {
            "recompute_decisions": 0,
            "store_decisions": 0,
        }
    
    def should_cache_tensor(
        self,
        tensor_size_mb: float,
        flop_cost: int,
        num_consumers: int,
        gpu_tflops: float = 10.0,
    ) -> bool:
        """
        Decide whether to cache or recompute.
        
        Args:
            tensor_size_mb: Size of tensor in MB
            flop_cost: FLOPs to recompute
            num_consumers: Number of consumers (reuse factor)
            gpu_tflops: GPU compute capacity in TFLOPS
        
        Returns:
            True if should cache, False if should recompute
        """
        # Multi-use: always cache
        if num_consumers > 1:
            self.stats["store_decisions"] += 1
            return True
        
        # Recomputation cost (milliseconds)
        # FLOPs / (TFLOPS * 1e12 ops/second) = seconds, convert to ms
        recompute_ms = (flop_cost / (gpu_tflops * 1e12)) * 1000
        
        # Network transfer cost (milliseconds)
        # Size in MB, convert to Gb: MB * 8 = Gb
        # Then divide by bandwidth: Gb / (Gbps) = seconds, convert to ms
        transfer_ms = (tensor_size_mb * 8 / self.network_bandwidth_gbps) * 1000
        
        # Cache cost: storage + retrieval (single consumer)
        # For single consumer, transfer happens once to store and once to retrieve (if needed)
        # But for our purposes, we assume retrieval from cache is nearly free
        cache_cost_ms = transfer_ms
        
        # Cache if recomputation is more expensive
        if recompute_ms > cache_cost_ms:
            self.stats["store_decisions"] += 1
            return True
        else:
            self.stats["recompute_decisions"] += 1
            return False
    
    def get_stats(self) -> Dict:
        """Get recomputation statistics."""
        return self.stats
