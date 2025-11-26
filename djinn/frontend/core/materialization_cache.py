"""
Phase 6C: Materialization Cache

Avoid redundant graph execution by caching materialization results.

Key idea: Hash lazy tensors by (operation, input_ids), not object identity.
This allows us to reuse previously computed results for identical operations.

Impact: Transformer attention with 1M control checks: 1M executions → ~100 unique
"""

from typing import Optional, Any, Dict, Tuple
import hashlib
import threading


class MaterializationCache:
    """
    Cache materialization results to avoid redundant execution.
    
    Caches based on operation semantics + input signatures, not object identity.
    This is critical for transformers where the same operation is evaluated
    many times with the same inputs (e.g., in attention layers).
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize materialization cache.
        
        Args:
            max_size: Maximum cache entries (LRU eviction beyond this)
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _compute_hash(self, lazy_tensor) -> str:
        """
        Compute hash for a lazy tensor based on operation + input signatures.
        
        Returns same hash for semantically identical operations, even if
        the tensor objects are different.
        """
        try:
            # Get operation
            operation = object.__getattribute__(lazy_tensor, '_operation')
            
            # Get inputs and compute their signatures
            inputs = object.__getattribute__(lazy_tensor, '_inputs')
            kwargs = object.__getattribute__(lazy_tensor, '_kwargs') or {}
            
            # Build signature from operation + input info
            input_sigs = []
            for inp in inputs:
                if hasattr(inp, '_operation'):  # Another LazyTensor
                    input_sigs.append(self._compute_hash(inp))
                elif hasattr(inp, 'shape'):  # Regular tensor
                    input_sigs.append(f"tensor:{str(inp.shape)}")
                elif isinstance(inp, (int, float, str, bool)):
                    input_sigs.append(f"scalar:{repr(inp)}")
                else:
                    input_sigs.append(f"obj:{type(inp).__name__}")
            
            # Include kwargs in signature
            kwargs_sig = ";".join(
                f"{k}={repr(v)}" for k, v in sorted(kwargs.items())
            )
            
            # Combine into hash
            signature = f"{operation}|{','.join(input_sigs)}|{kwargs_sig}"
            hash_obj = hashlib.sha256(signature.encode())
            return hash_obj.hexdigest()[:16]  # Short hash
        
        except Exception as e:
            # If hashing fails, return None (no caching for this tensor)
            return None
    
    def get(self, lazy_tensor) -> Optional[Any]:
        """
        Get cached materialization result if available.
        
        Args:
            lazy_tensor: LazyTensor to check
        
        Returns:
            Cached result if available, None otherwise
        """
        tensor_hash = self._compute_hash(lazy_tensor)
        if tensor_hash is None:
            return None
        
        with self.lock:
            if tensor_hash in self.cache:
                self.hits += 1
                return self.cache[tensor_hash]
            else:
                self.misses += 1
                return None
    
    def put(self, lazy_tensor, result: Any):
        """
        Cache a materialization result.
        
        Args:
            lazy_tensor: LazyTensor that was materialized
            result: The materialized (concrete) result
        """
        tensor_hash = self._compute_hash(lazy_tensor)
        if tensor_hash is None:
            return  # Can't cache
        
        with self.lock:
            # Simple LRU: if full, clear 25% of entries
            if len(self.cache) >= self.max_size:
                # Remove first 25% (oldest entries in insertion order)
                entries_to_remove = int(self.max_size * 0.25)
                for _ in range(entries_to_remove):
                    if self.cache:
                        self.cache.pop(next(iter(self.cache)))
                self.evictions += 1
            
            self.cache[tensor_hash] = result
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total * 100 if total > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'current_size': len(self.cache),
                'max_size': self.max_size,
            }


# Global materialization cache
_global_materialization_cache = MaterializationCache(max_size=1000)


def get_materialization_cache() -> MaterializationCache:
    """Get global materialization cache."""
    return _global_materialization_cache


def cache_materialization(lazy_tensor, result: Any):
    """Cache a materialization result."""
    _global_materialization_cache.put(lazy_tensor, result)


def get_cached_materialization(lazy_tensor) -> Optional[Any]:
    """Get cached materialization result if available."""
    return _global_materialization_cache.get(lazy_tensor)

