"""
Lazy metadata capture for deferred semantic annotation.

This module implements lazy metadata evaluation - storing operation information
during graph capture without expensive semantic analysis, then computing rich
metadata later during scheduling when full context is available.

This implements the fundamental fix for the torch.randn() overhead issue:
instead of capturing metadata at capture time (0.88ms per call), we defer it
to the scheduling phase where full context (module hierarchy, pattern info) 
is available.

Key design principle: Separation of Concerns
- Stage 1 (Capture): Build skeleton graph FAST (0.05ms per op)
- Stage 2 (Structure): Extract module hierarchy (FX tracing)
- Stage 3 (Semantic): Enrich graph with semantics (full context available)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
import threading


@dataclass
class MetadataPlaceholder:
    """
    Lazy metadata that's computed on-demand during scheduling.
    
    Stores minimal information during graph capture, defers expensive
    semantic analysis until the scheduler needs it and has full context.
    
    Attributes:
        operation: Name of the operation (e.g., 'aten::randn')
        inputs: Tuple of input shapes/values for later analysis
        kwargs: Operation arguments (dtype, device, etc.)
        _computed_metadata: Cached result of expensive computation
        _lock: Thread-safe access for lazy initialization
    """
    operation: str
    inputs: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    _computed_metadata: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    
    def get_metadata(self, capture_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Lazily compute metadata on first access.
        
        Thread-safe: Multiple threads can call this concurrently, but metadata
        is only computed once and cached.
        
        Args:
            capture_fn: Optional function to compute full metadata if None is stored.
                       Signature: capture_fn(operation, inputs, kwargs) -> Dict
        
        Returns:
            Dictionary with metadata. If capture_fn not provided, returns minimal
            metadata with just operation name.
        """
        # Fast path: already computed
        if self._computed_metadata is not None:
            return self._computed_metadata
        
        # Slow path: first access - compute lazily
        with self._lock:
            # Double-check after acquiring lock
            if self._computed_metadata is not None:
                return self._computed_metadata
            
            # Compute metadata
            if capture_fn:
                self._computed_metadata = capture_fn(
                    self.operation,
                    list(self.inputs),
                    self.kwargs
                )
            else:
                # Minimal metadata - just operation name
                # This is used during graph capture when full semantics aren't available
                self._computed_metadata = {
                    'operation': self.operation,
                    'lazy': True  # Flag indicating this was deferred
                }
            
            return self._computed_metadata
    
    def is_computed(self) -> bool:
        """Check if metadata has been computed yet."""
        return self._computed_metadata is not None
    
    def force_compute(self, capture_fn: Callable) -> Dict[str, Any]:
        """Force immediate computation of metadata."""
        if not self.is_computed():
            self.get_metadata(capture_fn)
        return self._computed_metadata

    # Dict-like interface for compatibility with scheduler
    def get(self, key: str, default=None):
        """Dict-like get method."""
        metadata = self.get_metadata()
        return metadata.get(key, default)

    def __getitem__(self, key: str):
        """Dict-like getitem method."""
        metadata = self.get_metadata()
        return metadata[key]

    def __contains__(self, key: str) -> bool:
        """Dict-like contains method."""
        metadata = self.get_metadata()
        return key in metadata

    def keys(self):
        """Dict-like keys method."""
        metadata = self.get_metadata()
        return metadata.keys()

    def values(self):
        """Dict-like values method."""
        metadata = self.get_metadata()
        return metadata.values()

    def items(self):
        """Dict-like items method."""
        metadata = self.get_metadata()
        return metadata.items()


def create_lazy_metadata(operation: str, inputs: tuple, kwargs: Dict[str, Any]) -> MetadataPlaceholder:
    """
    Convenience factory for creating lazy metadata placeholders.
    
    Args:
        operation: Operation name (e.g., 'aten::randn')
        inputs: Tuple of input arguments
        kwargs: Keyword arguments
    
    Returns:
        MetadataPlaceholder instance ready for lazy evaluation
    """
    return MetadataPlaceholder(
        operation=operation,
        inputs=inputs,
        kwargs=kwargs
    )
