"""
Memory management for Djinn framework.

Features:
- Automatic graph compaction (prevents OOM in long-running workloads)
- Memory monitoring and budgeting
- Garbage collection hooks
- Cache eviction policies
"""

import gc
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# MEMORY MONITORING
# ============================================================================

class MemoryPressure(str, Enum):
    """System memory pressure levels."""
    NORMAL = "normal"  # <50% of limit
    MODERATE = "moderate"  # 50-75%
    HIGH = "high"  # 75-90%
    CRITICAL = "critical"  # >90%


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    used_bytes: int
    limit_bytes: int
    pressure: MemoryPressure
    
    @property
    def utilization(self) -> float:
        """Percentage of memory limit used."""
        return (self.used_bytes / self.limit_bytes * 100) if self.limit_bytes > 0 else 0


class MemoryMonitor:
    """Monitor and manage memory usage."""
    
    def __init__(self, limit_mb: int = 1024):
        self.limit_bytes = limit_mb * 1024 * 1024
        self._tracked_objects: Dict[str, int] = {}  # name → size in bytes
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        used = sum(self._tracked_objects.values())
        
        # Determine pressure level
        utilization = used / self.limit_bytes if self.limit_bytes > 0 else 0
        if utilization < 0.5:
            pressure = MemoryPressure.NORMAL
        elif utilization < 0.75:
            pressure = MemoryPressure.MODERATE
        elif utilization < 0.90:
            pressure = MemoryPressure.HIGH
        else:
            pressure = MemoryPressure.CRITICAL
        
        return MemoryStats(used, self.limit_bytes, pressure)
    
    def track(self, name: str, size_bytes: int) -> None:
        """Track memory allocation."""
        self._tracked_objects[name] = size_bytes
        
        stats = self.get_stats()
        if stats.pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL):
            logger.warning(f"High memory pressure: {stats.utilization:.1f}% used")
    
    def untrack(self, name: str) -> None:
        """Stop tracking allocation."""
        self._tracked_objects.pop(name, None)
    
    def trigger_gc(self) -> int:
        """Trigger garbage collection, return freed bytes."""
        before = sum(self._tracked_objects.values())
        gc.collect()
        after = sum(self._tracked_objects.values())
        freed = before - after
        
        if freed > 0:
            logger.info(f"GC freed {freed / 1024 / 1024:.1f}MB")
        
        return freed


# ============================================================================
# GRAPH COMPACTION (Automatic Memory Reclaim)
# ============================================================================

class GraphCompactor:
    """
    Automatically compacts computation graphs to reclaim memory.
    
    Strategy:
    1. Track operation count
    2. Monitor memory usage
    3. When thresholds exceeded, compact graph
    4. Keep only unmaterialized nodes
    """
    
    # Configuration (can be adjusted per workload)
    COMPACTION_THRESHOLD_OPS = 500  # Compact every N operations
    MEMORY_LIMIT_MB = 100  # Max 100MB for graph metadata
    MEMORY_AGGRESSIVE_THRESHOLD_MB = 80  # Aggressive compaction at 80%
    
    def __init__(self, graph_builder):
        """Initialize compactor with reference to graph builder."""
        self.graph_builder = graph_builder
        self._operation_count = 0
        self._last_compaction_count = 0
        self._compaction_count = 0
        self._monitor = MemoryMonitor(self.MEMORY_LIMIT_MB)
    
    def on_operation_added(self, lazy_tensor) -> None:
        """Called when operation is added to graph."""
        self._operation_count += 1
        
        # Estimate size
        size_bytes = self._estimate_tensor_size(lazy_tensor)
        self._monitor.track(f"tensor_{lazy_tensor.id}", size_bytes)
        
        # Check if compaction needed
        if self._should_compact():
            self.compact()
    
    def _should_compact(self) -> bool:
        """Check if compaction threshold exceeded."""
        # Check operation count
        ops_since_compact = self._operation_count - self._last_compaction_count
        if ops_since_compact >= self.COMPACTION_THRESHOLD_OPS:
            return True
        
        # Check memory usage
        stats = self._monitor.get_stats()
        if stats.utilization >= self.MEMORY_AGGRESSIVE_THRESHOLD_MB:
            return True
        
        return False
    
    def compact(self) -> int:
        """
        Compact graph by removing materialized nodes.
        
        Returns: Number of nodes removed
        """
        logger.info("Starting graph compaction...")
        
        if not hasattr(self.graph_builder, 'nodes'):
            logger.warning("Graph builder doesn't support compaction")
            return 0
        
        # Count nodes before
        nodes_before = len(self.graph_builder.nodes)
        
        # Remove materialized nodes
        nodes_to_keep = []
        for node in self.graph_builder.nodes.values():
            # Keep if not materialized OR if it's a root node
            if (not self._is_materialized(node) or 
                node == getattr(self.graph_builder, 'root_tensor', None)):
                nodes_to_keep.append(node)
            else:
                # Untrack from monitor
                self._monitor.untrack(f"tensor_{node.id}")
        
        # Update graph
        self.graph_builder.nodes = {n.id: n for n in nodes_to_keep}
        
        # Trigger GC
        freed_bytes = self._monitor.trigger_gc()
        
        # Update stats
        removed = nodes_before - len(nodes_to_keep)
        self._last_compaction_count = self._operation_count
        self._compaction_count += 1
        
        logger.info(
            f"Compaction #{self._compaction_count}: "
            f"removed {removed} nodes, freed {freed_bytes / 1024:.1f}KB"
        )
        
        return removed
    
    def _is_materialized(self, node) -> bool:
        """Check if node has been materialized (executed)."""
        return (hasattr(node, '_materialized_value') and 
                node._materialized_value is not None)
    
    def _estimate_tensor_size(self, tensor) -> int:
        """Estimate size of lazy tensor in bytes."""
        # Conservative estimate: 250 bytes per node + shape info
        size = 250
        
        if hasattr(tensor, 'shape') and tensor.shape:
            # Add size for shape tuple
            size += len(tensor.shape) * 8
        
        if hasattr(tensor, 'metadata') and tensor.metadata:
            # Add size for metadata
            import sys
            size += sys.getsizeof(tensor.metadata, default=100)
        
        return size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compaction statistics."""
        memory_stats = self._monitor.get_stats()
        return {
            'compaction_count': self._compaction_count,
            'total_operations': self._operation_count,
            'memory_utilization': memory_stats.utilization,
            'memory_pressure': memory_stats.pressure.value,
        }


# ============================================================================
# GLOBAL MEMORY MANAGER
# ============================================================================

_global_memory_monitor: Optional[MemoryMonitor] = None
_global_graph_compactor: Optional[GraphCompactor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor (lazy initialization)."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor(limit_mb=1024)
    return _global_memory_monitor


def get_graph_compactor() -> Optional[GraphCompactor]:
    """Get global graph compactor."""
    return _global_graph_compactor


def set_graph_compactor(compactor: GraphCompactor) -> None:
    """Set global graph compactor."""
    global _global_graph_compactor
    _global_graph_compactor = compactor


# ============================================================================
# BACKWARD COMPATIBILITY: Legacy GPU Memory Manager
# ============================================================================

class GPUMemoryManager:
    """
    Legacy GPU memory manager (backward compatibility).
    
    Kept for compatibility with existing coordinator code.
    New code should use MemoryMonitor and GraphCompactor instead.
    """
    
    def __init__(self):
        self.monitor = get_memory_monitor()
    
    def register(self, tensor, transfer_id):
        """Register tensor transfer (no-op for Phase 1)."""
        pass
    
    def allocate(self, size_bytes: int, device: str = "gpu:0") -> bool:
        """Check if allocation is possible given memory constraints."""
        stats = self.monitor.get_stats()
        return stats.utilization + size_bytes <= stats.limit_bytes
    
    def track_transfer(self, tensor_id: str, size_bytes: int):
        """Track memory transfer."""
        self.monitor.track(f"transfer_{tensor_id}", size_bytes)
    
    def get_utilization(self) -> float:
        """Get current memory utilization percentage."""
        stats = self.monitor.get_stats()
        return stats.utilization


__all__ = [
    'MemoryMonitor',
    'GraphCompactor',
    'GPUMemoryManager',  # Backward compat
    'MemoryStats',
    'MemoryPressure',
    'get_memory_monitor',
    'get_graph_compactor',
    'set_graph_compactor',
]