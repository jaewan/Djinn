"""
Model Registry: Stable tensor identity for GPU disaggregation.

Maps PyTorch model parameters to stable identifiers using parameter names.
This enables server-side caching to work correctly across requests.

Key Design Decisions:
1. Use parameter names as identifiers (stable across requests)
2. No content hashing needed (parameter names are sufficient)
3. Input tensors use UUID (never cached, always unique)
4. Thread-safe singleton for process-wide tracking

This is Phase 1, Component 1 of the enhancement plan.
"""

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorIdentity:
    """
    Stable identifier for model tensors.
    
    Immutable to allow use as dict key.
    
    For model parameters: Uses parameter name (stable across requests)
    For input tensors: Uses UUID (unique per tensor, never cached)
    """
    identifier: str  # "model.layer.weight" or "input:uuid"
    shape: Tuple[int, ...]
    dtype: torch.dtype
    is_parameter: bool
    
    def __str__(self) -> str:
        return self.identifier
    
    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts."""
        return hash((self.identifier, self.shape, str(self.dtype), self.is_parameter))
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, TensorIdentity):
            return False
        return (self.identifier == other.identifier and
                self.shape == other.shape and
                self.dtype == other.dtype and
                self.is_parameter == other.is_parameter)


class ModelRegistry:
    """
    Maps PyTorch tensors to stable identifiers.
    
    Key insight: For GPU disaggregation, we don't need content hashing
    for model parameters because:
    1. Parameter names are stable across requests
    2. Same model = same names = same tensors
    3. Server can cache by name, not by content
    
    Thread-safe singleton for process-wide tensor tracking.
    """
    
    _instance: Optional['ModelRegistry'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_registry()
        return cls._instance
    
    def _init_registry(self):
        """Initialize registry state."""
        # Map Python id() → TensorIdentity (for fast lookup)
        self._id_cache: Dict[int, TensorIdentity] = {}
        
        # Map model_id → parameter names → TensorIdentity (for model tracking)
        self._model_params: Dict[str, Dict[str, TensorIdentity]] = {}
        
        # Thread safety
        self._registry_lock = threading.Lock()
        
        # Stats
        self.stats = {
            'parameters_registered': 0,
            'inputs_registered': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def register_model(self, model: nn.Module, model_id: Optional[str] = None) -> Dict[str, TensorIdentity]:
        """
        Register all model parameters and return stable identifiers.
        
        Args:
            model: PyTorch model
            model_id: Optional model identifier (default: model class name)
        
        Returns:
            Dict mapping parameter names to TensorIdentity objects
        """
        if model_id is None:
            model_id = model.__class__.__name__
        
        with self._registry_lock:
            param_identities = {}
            
            # Register parameters
            for name, param in model.named_parameters():
                identity = TensorIdentity(
                    identifier=f"{model_id}.{name}",
                    shape=tuple(param.shape),
                    dtype=param.dtype,
                    is_parameter=True
                )
                
                # Cache by both Parameter object id and data tensor id for fast lookup
                # This handles both param and param.data lookups
                self._id_cache[id(param)] = identity
                self._id_cache[id(param.data)] = identity
                param_identities[name] = identity
                self.stats['parameters_registered'] += 1
            
            # Register buffers (batch norm stats, etc.)
            for name, buffer in model.named_buffers():
                identity = TensorIdentity(
                    identifier=f"{model_id}.buffer.{name}",
                    shape=tuple(buffer.shape),
                    dtype=buffer.dtype,
                    is_parameter=True
                )
                # Buffers are already tensors, so just cache by id
                self._id_cache[id(buffer)] = identity
                param_identities[f"buffer.{name}"] = identity
                self.stats['parameters_registered'] += 1
            
            # Store model parameters
            self._model_params[model_id] = param_identities
            
            logger.info(
                f"Registered model '{model_id}': {len(param_identities)} parameters/buffers"
            )
            
            return param_identities
    
    def get_identity(self, tensor: torch.Tensor) -> TensorIdentity:
        """
        Get stable identity for a tensor.
        
        For model parameters: Returns registered identity
        For input tensors: Creates identity based on UUID (unique per tensor)
        
        Args:
            tensor: PyTorch tensor to identify
        
        Returns:
            TensorIdentity that remains stable across program runs (for parameters)
            or unique per tensor (for inputs)
        """
        # Fast path: Check if already registered (model parameter)
        python_id = id(tensor)
        
        with self._registry_lock:
            if python_id in self._id_cache:
                self.stats['cache_hits'] += 1
                return self._id_cache[python_id]
            
            # Slow path: Input tensor (not a model parameter)
            # FIX: Use UUID to prevent collisions (inputs are never cached)
            # Inputs are always new, so we don't need stable IDs
            self.stats['cache_misses'] += 1
            self.stats['inputs_registered'] += 1
            
            identity = TensorIdentity(
                identifier=f"input:{uuid.uuid4().hex}",  # Unique per tensor
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                is_parameter=False
            )
            
            # Don't cache input identities (they're unique anyway)
            # This prevents memory leaks from accumulating UUIDs
            return identity
    
    def get_model_tensors(self, model_id: str) -> Dict[str, TensorIdentity]:
        """
        Get all registered tensors for a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dict mapping parameter names to TensorIdentity objects
        """
        with self._registry_lock:
            return self._model_params.get(model_id, {}).copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Get registry statistics."""
        with self._registry_lock:
            return self.stats.copy()
    
    def clear(self):
        """Clear all registered models (for testing)."""
        with self._registry_lock:
            self._id_cache.clear()
            self._model_params.clear()
            self.stats = {
                'parameters_registered': 0,
                'inputs_registered': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            logger.info("Model registry cleared")


# Global singleton accessor
_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry


def parse_identifier(identifier: str) -> Tuple[str, str]:
    """
    Parse identifier into (model_id, tensor_name) for compatibility with tensor_registry.
    
    This enables integration between client-side model_registry and server-side tensor_registry.
    
    Args:
        identifier: Full identifier from TensorIdentity (e.g., "GPT2.transformer.h.0.attn.weight")
    
    Returns:
        Tuple of (model_id, tensor_name)
        Example: ("GPT2", "transformer.h.0.attn.weight")
    
    Raises:
        ValueError: If identifier is for an input tensor (doesn't have model_id)
    
    Note:
        Server-side tensor_registry uses (model_id, tensor_name, version) as keys.
        This function converts model_registry identifiers to that format.
    """
    if identifier.startswith("input:"):
        raise ValueError(
            f"Input tensor identifiers don't have model_id/tensor_name: {identifier}"
        )
    
    # Split on first dot: "GPT2.transformer.h.0.attn.weight" → ("GPT2", "transformer.h.0.attn.weight")
    parts = identifier.split(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        # Fallback: assume entire string is tensor_name (no model_id prefix)
        logger.warning(
            f"Identifier '{identifier}' doesn't have model_id prefix, using 'unknown' as model_id"
        )
        return "unknown", identifier

