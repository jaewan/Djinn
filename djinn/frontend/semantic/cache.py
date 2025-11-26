"""
Advanced two-level caching for semantic analysis.

Features:
- Deterministic SHA-256 hashing
- Thread-safe operations (RLock)
- Bounded memory (LRU + size limits)
- Cache persistence (save/load)
- Adaptive sizing
- Performance monitoring

Cache Levels:
- Level 1: Topology cache (pattern matching results) - expensive, reused often
- Level 2: Shape cache (cost estimates) - cheap, shape-dependent
"""

import threading
import hashlib
import time
import pickle
import os
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata for eviction policy."""
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    last_access: float = field(default_factory=time.time)


class SemanticAnnotatorCache:
    """
    Thread-safe two-level cache with memory management.
    
    Level 1: Topology cache (pattern matching - expensive)
    Level 2: Shape cache (cost estimation - cheap)
    """
    
    def __init__(
        self,
        max_topology_entries: int = 1000,
        max_shape_entries: int = 10000,
        max_memory_mb: int = 500,
        enable_persistence: bool = True,
        cache_dir: str = ".genie_cache"
    ):
        # Thread safety
        self._lock = threading.RLock()
        
        # Level 1: Topology cache (pattern matching - expensive)
        self._topology_cache: Dict[str, CacheEntry] = {}
        self._topology_lru = OrderedDict()
        
        # Level 2: Shape cache (cost estimation - cheap)
        self._shape_cache: Dict[Tuple[str, str], CacheEntry] = {}
        self._shape_lru = OrderedDict()
        
        # Configuration
        self.max_topology_entries = max_topology_entries
        self.max_shape_entries = max_shape_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_persistence = enable_persistence
        self.cache_dir = cache_dir
        
        # Statistics
        self._stats = {
            'topology_hits': 0,
            'topology_misses': 0,
            'shape_hits': 0,
            'shape_misses': 0,
            'evictions': 0,
            'memory_bytes': 0,
            'total_requests': 0,
        }
        
        # Adaptive sizing state
        self._recent_hit_rates = []
        self._last_resize_time = time.time()
        
        # Create cache directory
        if enable_persistence:
            os.makedirs(cache_dir, exist_ok=True)
    
    # ===================================================================
    # PUBLIC API
    # ===================================================================
    
    def get_or_compute_topology(
        self,
        graph,
        compute_fn: Callable,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Get topology analysis from cache or compute.
        
        Args:
            graph: Computation graph
            compute_fn: Function to compute if cache miss
            **kwargs: Additional args for compute_fn
        
        Returns:
            (result, was_cached)
        """
        with self._lock:
            self._stats['total_requests'] += 1
            
            # Compute topology hash
            topo_hash = self._compute_topology_hash(graph)
            
            # Check cache
            if topo_hash in self._topology_cache:
                entry = self._topology_cache[topo_hash]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Update LRU
                self._topology_lru.move_to_end(topo_hash)
                
                self._stats['topology_hits'] += 1
                logger.debug(f"Topology cache HIT: {topo_hash[:8]}")
                
                return entry.data, True
            
            # Cache miss - compute
            self._stats['topology_misses'] += 1
            logger.info(f"Topology cache MISS: running pattern matching")
        
        # Release lock during expensive computation
        result = compute_fn(graph, **kwargs)
        
        # Re-acquire lock to store result
        with self._lock:
            # Estimate size
            size_bytes = self._estimate_size(result)
            
            # Check memory limit
            if not self._can_cache(size_bytes):
                logger.warning(f"Cannot cache result (size: {size_bytes} bytes)")
                return result, False
            
            # Store in cache
            entry = CacheEntry(
                data=result,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self._topology_cache[topo_hash] = entry
            self._topology_lru[topo_hash] = True
            self._stats['memory_bytes'] += size_bytes
            
            # Evict if needed
            self._evict_if_needed('topology')
            
            return result, False
    
    def get_or_compute_shape(
        self,
        graph,
        topology_hash: str,
        compute_fn: Callable,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Get shape-specific analysis from cache or compute.
        
        Args:
            graph: Computation graph
            topology_hash: Pre-computed topology hash
            compute_fn: Function to compute if cache miss
            **kwargs: Additional args for compute_fn
        
        Returns:
            (result, was_cached)
        """
        with self._lock:
            # Compute shape hash
            shape_hash = self._compute_shape_hash(graph)
            cache_key = (topology_hash, shape_hash)
            
            # Check cache
            if cache_key in self._shape_cache:
                entry = self._shape_cache[cache_key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Update LRU
                self._shape_lru.move_to_end(cache_key)
                
                self._stats['shape_hits'] += 1
                logger.debug(f"Shape cache HIT: {shape_hash[:8]}")
                
                return entry.data, True
            
            # Cache miss
            self._stats['shape_misses'] += 1
            logger.debug(f"Shape cache MISS: recomputing costs")
        
        # Release lock during computation
        result = compute_fn(graph, **kwargs)
        
        # Re-acquire lock to store
        with self._lock:
            size_bytes = self._estimate_size(result)
            
            if not self._can_cache(size_bytes):
                return result, False
            
            entry = CacheEntry(
                data=result,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self._shape_cache[cache_key] = entry
            self._shape_lru[cache_key] = True
            self._stats['memory_bytes'] += size_bytes
            
            self._evict_if_needed('shape')
            
            return result, False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_topo = self._stats['topology_hits'] + self._stats['topology_misses']
            total_shape = self._stats['shape_hits'] + self._stats['shape_misses']
            
            return {
                'topology_cache': {
                    'size': len(self._topology_cache),
                    'max_size': self.max_topology_entries,
                    'hits': self._stats['topology_hits'],
                    'misses': self._stats['topology_misses'],
                    'hit_rate': (self._stats['topology_hits'] / total_topo 
                                if total_topo > 0 else 0.0),
                },
                'shape_cache': {
                    'size': len(self._shape_cache),
                    'max_size': self.max_shape_entries,
                    'hits': self._stats['shape_hits'],
                    'misses': self._stats['shape_misses'],
                    'hit_rate': (self._stats['shape_hits'] / total_shape 
                                if total_shape > 0 else 0.0),
                },
                'memory': {
                    'current_bytes': self._stats['memory_bytes'],
                    'max_bytes': self.max_memory_bytes,
                    'utilization': (self._stats['memory_bytes'] / 
                                   self.max_memory_bytes if self.max_memory_bytes > 0 else 0.0),
                },
                'overall': {
                    'total_requests': self._stats['total_requests'],
                    'evictions': self._stats['evictions'],
                },
            }
    
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._topology_cache.clear()
            self._topology_lru.clear()
            self._shape_cache.clear()
            self._shape_lru.clear()
            self._stats['memory_bytes'] = 0
            logger.info("Cache cleared")
    
    # ===================================================================
    # HASH COMPUTATION (Deterministic with SHA-256)
    # ===================================================================
    
    def _compute_topology_hash(self, graph) -> str:
        """
        Compute deterministic topology hash (structure only, not shapes).
        
        Strategy:
        - Small graphs (<50 nodes): Simple operation sequence hash
        - Large graphs: Weisfeiler-Lehman hash
        """
        num_nodes = len(list(graph.nodes()))
        
        # Fast path for small graphs
        if num_nodes < 50:
            return self._simple_topology_hash(graph)
        
        # Slow path for complex graphs
        return self._weisfeiler_lehman_hash(graph)
    
    def _simple_topology_hash(self, graph) -> str:
        """Simple hash for small graphs (O(N log N))."""
        operations = []
        
        for node in graph.nodes():
            op = getattr(node, 'operation', 'unknown')
            operations.append(op)
        
        # Sort for determinism (canonical ordering)
        canonical = "|".join(sorted(operations))
        hash_bytes = hashlib.sha256(canonical.encode('utf-8')).digest()
        return hash_bytes.hex()[:16]
    
    def _weisfeiler_lehman_hash(self, graph) -> str:
        """
        Weisfeiler-Lehman hash for complex graphs.
        More accurate than simple hash but slower (O(N²·I)).
        """
        # Initialize labels
        labels = {}
        for node in graph.nodes():
            node_id = getattr(node, 'id', id(node))
            op = getattr(node, 'operation', 'unknown')
            labels[node_id] = op
        
        # WL refinement iterations (3 is usually sufficient)
        for iteration in range(3):
            new_labels = {}
            
            for node in graph.nodes():
                node_id = getattr(node, 'id', id(node))
                
                # Collect neighbor labels (sorted for determinism)
                neighbor_labels = []
                inputs = getattr(node, 'inputs', [])
                for inp in inputs:
                    inp_id = getattr(inp, 'id', id(inp))
                    if inp_id in labels:
                        neighbor_labels.append(labels[inp_id])
                
                # New label = hash(current + sorted neighbors)
                neighbor_labels.sort()
                combined = f"{labels[node_id]}:{','.join(neighbor_labels)}"
                label_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()[:8]
                new_labels[node_id] = label_hash
            
            labels = new_labels
        
        # Canonical graph signature
        canonical = "|".join(sorted(labels.values()))
        hash_bytes = hashlib.sha256(canonical.encode('utf-8')).digest()
        return hash_bytes.hex()[:16]
    
    def _compute_shape_hash(self, graph) -> str:
        """Hash of concrete tensor shapes (deterministic and canonical)."""
        shape_components = []
        
        # Canonical node ordering
        nodes_list = list(graph.nodes())
        node_ids = sorted([getattr(n, 'id', id(n)) for n in nodes_list])
        
        for node in nodes_list:
            shape = getattr(node, 'shape', None)
            shape_str = str(tuple(shape)) if shape else "unknown"
            shape_components.append(shape_str)
        
        # Sort for determinism across runs
        canonical = "|".join(sorted(shape_components))
        hash_bytes = hashlib.sha256(canonical.encode('utf-8')).digest()
        return hash_bytes.hex()[:16]
    
    # ===================================================================
    # MEMORY MANAGEMENT
    # ===================================================================
    
    def _can_cache(self, size_bytes: int) -> bool:
        """Check if we can cache an entry of given size."""
        if self._stats['memory_bytes'] + size_bytes <= self.max_memory_bytes:
            return True
        
        # Try to make room
        self._force_evict(size_bytes)
        
        # Check again
        return self._stats['memory_bytes'] + size_bytes <= self.max_memory_bytes
    
    def _force_evict(self, needed_bytes: int):
        """Aggressively evict to make room."""
        target_free = needed_bytes + (self.max_memory_bytes * 0.1)  # 10% buffer
        
        while self._stats['memory_bytes'] + target_free > self.max_memory_bytes:
            # Try topology cache first (larger entries)
            if self._topology_lru:
                oldest = next(iter(self._topology_lru))
                self._evict_entry('topology', oldest)
                continue
            
            # Then shape cache
            if self._shape_lru:
                oldest = next(iter(self._shape_lru))
                self._evict_entry('shape', oldest)
                continue
            
            # Can't evict anymore
            break
    
    def _evict_if_needed(self, cache_type: str):
        """Smart eviction with age and LRU consideration."""
        if cache_type == 'topology':
            cache = self._topology_cache
            lru = self._topology_lru
            max_size = self.max_topology_entries
        else:
            cache = self._shape_cache
            lru = self._shape_lru
            max_size = self.max_shape_entries
        
        if len(cache) <= max_size:
            return
        
        # Age-aware eviction (prefer old entries)
        current_time = time.time()
        eviction_candidates = []
        
        for key in lru:
            entry = cache[key]
            age = current_time - entry.timestamp
            
            # Evict if older than 1 hour or low access count
            if age > 3600 or entry.access_count < 2:
                eviction_candidates.append(key)
        
        # Evict candidates
        for key in eviction_candidates[:len(cache) - max_size]:
            self._evict_entry(cache_type, key)
        
        # If still over limit, fall back to pure LRU
        while len(cache) > max_size:
            oldest = next(iter(lru))
            self._evict_entry(cache_type, oldest)
    
    def _evict_entry(self, cache_type: str, key):
        """Evict a single entry."""
        if cache_type == 'topology':
            entry = self._topology_cache.pop(key, None)
            self._topology_lru.pop(key, None)
        else:
            entry = self._shape_cache.pop(key, None)
            self._shape_lru.pop(key, None)
        
        if entry:
            self._stats['memory_bytes'] -= entry.size_bytes
            self._stats['evictions'] += 1
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        import sys
        try:
            return sys.getsizeof(obj, default=1000)
        except:
            return 1000  # Conservative estimate
    
    # ===================================================================
    # CACHE PERSISTENCE
    # ===================================================================
    
    def save(self, filename: Optional[str] = None):
        """Save cache to disk."""
        if not self.enable_persistence:
            return
        
        filename = filename or os.path.join(self.cache_dir, "semantic_cache.pkl")
        
        with self._lock:
            # Prepare serializable data
            cache_data = {
                'version': '1.0',
                'timestamp': time.time(),
                'topology': {
                    k: {
                        'data': v.data,
                        'timestamp': v.timestamp,
                        'access_count': v.access_count,
                    }
                    for k, v in self._topology_cache.items()
                },
                'stats': self._stats.copy(),
            }
            
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.info(f"Cache saved to {filename} "
                          f"({len(cache_data['topology'])} entries)")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
    
    def load(self, filename: Optional[str] = None):
        """Load cache from disk."""
        if not self.enable_persistence:
            return
        
        filename = filename or os.path.join(self.cache_dir, "semantic_cache.pkl")
        
        if not os.path.exists(filename):
            logger.debug(f"No cache file found: {filename}")
            return
        
        with self._lock:
            try:
                with open(filename, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Validate version
                if cache_data.get('version') != '1.0':
                    logger.warning("Cache version mismatch, ignoring")
                    return
                
                # Restore topology cache
                for key, entry_data in cache_data['topology'].items():
                    entry = CacheEntry(
                        data=entry_data['data'],
                        timestamp=entry_data['timestamp'],
                        access_count=entry_data['access_count'],
                        size_bytes=self._estimate_size(entry_data['data'])
                    )
                    self._topology_cache[key] = entry
                    self._topology_lru[key] = True
                    self._stats['memory_bytes'] += entry.size_bytes
                
                logger.info(f"Cache loaded from {filename} "
                          f"({len(self._topology_cache)} entries)")
            
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    # ===================================================================
    # ADAPTIVE SIZING
    # ===================================================================
    
    def adapt_sizes(self):
        """Dynamically adjust cache sizes based on workload."""
        with self._lock:
            # Only adapt if we have enough data
            if self._stats['total_requests'] < 100:
                return
            
            # Don't adapt too frequently
            if time.time() - self._last_resize_time < 300:  # 5 minutes
                return
            
            # Calculate hit rates
            total_topo = (self._stats['topology_hits'] + 
                         self._stats['topology_misses'])
            topo_hit_rate = (self._stats['topology_hits'] / total_topo 
                            if total_topo > 0 else 0.0)
            
            # Track recent hit rates
            self._recent_hit_rates.append(topo_hit_rate)
            if len(self._recent_hit_rates) > 10:
                self._recent_hit_rates.pop(0)
            
            # Adaptive logic
            if topo_hit_rate > 0.90:
                # High hit rate - increase capacity
                new_size = min(5000, int(self.max_topology_entries * 1.2))
                if new_size > self.max_topology_entries:
                    logger.info(f"Increasing topology cache: "
                              f"{self.max_topology_entries} → {new_size}")
                    self.max_topology_entries = new_size
            
            elif topo_hit_rate < 0.50:
                # Low hit rate but cache not full - workload is diverse
                if len(self._topology_cache) < self.max_topology_entries:
                    new_size = min(5000, int(self.max_topology_entries * 1.5))
                    logger.info(f"Diverse workload, increasing topology cache: "
                              f"{self.max_topology_entries} → {new_size}")
                    self.max_topology_entries = new_size
            
            self._last_resize_time = time.time()
