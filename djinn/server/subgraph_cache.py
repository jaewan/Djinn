"""
Cache built subgraphs to avoid rebuilding.

Key insight: Same tensor DAG → Same subgraph
Cache subgraphs by DAG structure, not tensor values.
"""

import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class CachedSubgraph:
    """Cached subgraph with metadata."""
    subgraph: Any  # RemoteSubgraph
    dag_hash: str
    operation_count: int
    build_time_ms: float
    access_count: int = 0

    @property
    def hit_rate_score(self) -> float:
        """Score for LRU eviction (higher = keep longer)."""
        # Favor frequently accessed, expensive-to-build subgraphs
        return self.access_count * (self.build_time_ms / 1000.0) * self.operation_count


class SubgraphCache:
    """
    Thread-safe cache for built subgraphs.

    Avoids rebuilding subgraphs for repeated patterns.
    Cache key: Hash of DAG structure (operations + connections)
    """

    def __init__(self, max_entries: int = 100):
        self.cache: Dict[str, CachedSubgraph] = {}
        self.max_entries = max_entries
        self.lock = threading.RLock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get_or_build(self, target_tensor, builder) -> Any:
        """
        Get cached subgraph or build new one.

        Args:
            target_tensor: LazyTensor to build subgraph for
            builder: SubgraphBuilder instance

        Returns:
            RemoteSubgraph (cached or newly built)
        """
        # Compute DAG hash
        dag_hash = self._compute_dag_hash(target_tensor)

        # ✅ PROFILING: Measure graph cache lookup time
        import time
        try:
            from .profiling_context import get_profiler, record_phase
            profiler = get_profiler()
            if profiler and profiler.enabled:
                with record_phase('graph_cache_lookup', metadata={'dag_hash': dag_hash[:8]}):
        # Check cache
                    with self.lock:
                        if dag_hash in self.cache:
                            cached = self.cache[dag_hash]
                            cached.access_count += 1
                            self.stats['hits'] += 1
                            logger.debug(f"Subgraph cache HIT: {cached.operation_count} ops")
                            return cached.subgraph
            else:
                # No profiling - check cache normally
                with self.lock:
                    if dag_hash in self.cache:
                        cached = self.cache[dag_hash]
                        cached.access_count += 1
                        self.stats['hits'] += 1
                        logger.debug(f"Subgraph cache HIT: {cached.operation_count} ops")
                        return cached.subgraph
        except ImportError:
            # Fallback if profiling not available
            with self.lock:
                if dag_hash in self.cache:
                    cached = self.cache[dag_hash]
                    cached.access_count += 1
                    self.stats['hits'] += 1
                    logger.debug(f"Subgraph cache HIT: {cached.operation_count} ops")
                    return cached.subgraph

        # Cache miss - build subgraph
        self.stats['misses'] += 1
        start = time.perf_counter()

        subgraph = builder.build_remote_subgraph(target_tensor, defer_metadata=True)

        build_time_ms = (time.perf_counter() - start) * 1000

        # Cache it
        with self.lock:
            # Evict if needed (LRU by hit rate score)
            if len(self.cache) >= self.max_entries:
                self._evict_lru()

            self.cache[dag_hash] = CachedSubgraph(
                subgraph=subgraph,
                dag_hash=dag_hash,
                operation_count=len(subgraph.operations),
                build_time_ms=build_time_ms
            )

        logger.info(f"Built subgraph: {len(subgraph.operations)} ops in {build_time_ms:.1f}ms")
        return subgraph

    def _evict_lru(self):
        """Evict least recently/frequently used subgraph."""
        if not self.cache:
            return

        # Find entry with lowest hit rate score
        worst_key = min(self.cache.items(),
                       key=lambda x: x[1].hit_rate_score)[0]

        del self.cache[worst_key]
        self.stats['evictions'] += 1
        logger.debug(f"Evicted cached subgraph: {worst_key[:8]}...")

    def _compute_dag_hash(self, tensor) -> str:
        """
        Compute hash of DAG structure.

        Based on operations and connections, NOT tensor values.
        Same DAG = same hash, regardless of tensor values.
        """
        visited = set()
        structure = []

        def traverse(t):
            if id(t) in visited:
                return f"ref_{id(t)}"
            visited.add(id(t))

            # Access raw attributes to avoid triggering properties
            operation = object.__getattribute__(t, '_operation')
            inputs = object.__getattribute__(t, '_inputs')

            # Build structure representation
            input_refs = []
            for inp in inputs:
                if hasattr(inp, '_operation'):  # LazyTensor
                    input_refs.append(traverse(inp))
                else:
                    # Concrete tensor - use type info
                    input_refs.append(f"const_{type(inp).__name__}")

            node_str = f"{operation}({','.join(input_refs)})"
            structure.append(node_str)
            return f"node_{id(t)}"

        traverse(tensor)

        # Hash the structure
        structure_str = '|'.join(structure)
        return hashlib.sha256(structure_str.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'entries': len(self.cache),
                'max_entries': self.max_entries,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1),
                'total_requests': self.stats['hits'] + self.stats['misses']
            }

    def clear(self):
        """Clear all cached subgraphs."""
        with self.lock:
            cleared = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared {cleared} cached subgraphs")


# Global cache instance
_subgraph_cache = SubgraphCache()


def get_subgraph_cache() -> SubgraphCache:
    """Get global subgraph cache."""
    return _subgraph_cache
