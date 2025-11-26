"""
Smart Tensor Registry for caching tensors across requests.

This component avoids redundant transfers of model weights by caching them on the remote GPU
and reusing them across requests. It uses semantic metadata (model_id, tensor_name) instead of
expensive content hashing to identify tensors.

Design Principles:
1. Version-aware keys: Incorporates model revision to invalidate on model updates
2. UUID-based handles: Stable references to GPU memory (not process-local id())
3. Async-safe: Uses asyncio.Lock for concurrent request handling
4. Scope: Targets persistent tensors (weights, KV cache), not ephemeral activations
5. Memory guardrails: Tracks bytes per model to prevent cache growth from starving execution
6. Integration: Hooks into existing GPUCache eviction for consistency
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Set
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


@dataclass
class RemoteHandle:
    """Reference to tensor on remote GPU."""
    
    device_id: str                    # Which GPU (e.g., "cuda:0")
    tensor_id: str                    # Stable UUID (not process-local id())
    shape: torch.Size
    dtype: torch.dtype
    timestamp: float                  # For LRU eviction
    version: int = 0                  # Model version (for invalidation)
    tensor_bytes: int = 0             # Size in bytes (for memory tracking)
    
    def is_valid(self, tensor: Optional[torch.Tensor] = None) -> bool:
        """Validate handle against tensor metadata."""
        if tensor is None:
            return True  # Assume valid if no tensor to check
        
        return (self.shape == tensor.shape and 
                self.dtype == tensor.dtype)


@dataclass
class TensorRegistryStats:
    """Statistics tracked by the registry."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_saved: int = 0
    overhead_ms: float = 0.0
    invalidations: int = 0
    version_conflicts: int = 0


class SmartTensorRegistry:
    """
    Cache tensors on remote GPU using semantic information.
    
    Key Design Decision: Use (model_id, tensor_name, version) not content hash
    - Avoids 50ms hashing overhead
    - Leverages SRG semantic metadata
    - Simple invalidation by model_id
    - Version-aware for model updates
    
    Thread Safety:
    - All mutations guarded by asyncio.Lock
    - Safe for concurrent async requests
    
    Memory Safety:
    - Tracks bytes per model
    - Refuses registration if exceeds budget
    - Integrates with GPUCache eviction
    """
    
    def __init__(
        self, 
        max_cached_models: int = 5,
        max_bytes_per_model: Optional[int] = None,
        max_total_bytes: Optional[int] = None
    ):
        """
        Initialize tensor registry.
        
        Args:
            max_cached_models: Maximum number of models to cache (LRU eviction)
            max_bytes_per_model: Max memory per model (e.g., 1GB = 1073741824 bytes)
            max_total_bytes: Max total memory across all models (e.g., 5GB)
        """
        self.max_cached_models = max_cached_models
        self.max_bytes_per_model = max_bytes_per_model
        self.max_total_bytes = max_total_bytes
        
        # Map: (model_id, tensor_name, version) → RemoteHandle
        self.registry: Dict[Tuple[str, str, int], RemoteHandle] = {}
        
        # Track models for eviction
        self.model_timestamps: Dict[str, float] = {}
        self.model_versions: Dict[str, int] = {}
        self.model_memory_bytes: Dict[str, int] = {}
        
        # Statistics
        self.stats = TensorRegistryStats()
        
        # Concurrency protection
        self._lock = asyncio.Lock()
        
        # Set for ephemeral tensor names (never cache)
        self._ephemeral_patterns = {
            'activation', 'intermediate', 'hidden_state',
            'gradient', 'attention_output', 'mlp_output'
        }
        
        logger.info(
            "SmartTensorRegistry initialized: "
            "max_models=%d, max_bytes_per_model=%s, max_total_bytes=%s",
            max_cached_models,
            max_bytes_per_model,
            max_total_bytes
        )
    
    async def check_and_register(
        self,
        model_id: str,
        tensor_name: str,
        tensor: Optional[torch.Tensor] = None,
        model_version: int = 0,
        remote_handle: Optional[RemoteHandle] = None
    ) -> Tuple[bool, Optional[RemoteHandle]]:
        """
        Check if tensor is cached, register if not.
        
        This is the main API. Call with tensor details to check cache status.
        
        Args:
            model_id: Unique model identifier (e.g., "gpt2-xl-v1.0")
            tensor_name: Semantic tensor name (e.g., "transformer.layer.0.self_attn.weight")
            tensor: The tensor object (used for validation, can be None on cache miss)
            model_version: Model version for invalidation on updates
            remote_handle: If provided, registers this tensor in the cache
        
        Returns:
            Tuple of (needs_transfer, remote_handle)
            - needs_transfer=False, handle=RemoteHandle: Tensor is cached, use handle
            - needs_transfer=True, handle=None: Tensor not cached, must transfer
        """
        start_time = time.time()
        
        # Check if this is an ephemeral tensor (never cache)
        if self._is_ephemeral_tensor(tensor_name):
            self.stats.overhead_ms += (time.time() - start_time) * 1000
            return True, None
        
        async with self._lock:
            cache_key = (model_id, tensor_name, model_version)
            
            # Check for version conflict
            if model_id in self.model_versions:
                if self.model_versions[model_id] != model_version:
                    # Model was updated, invalidate old entries
                    self._invalidate_model_locked(model_id)
                    self.stats.version_conflicts += 1
            
            # Check cache
            if cache_key in self.registry:
                handle = self.registry[cache_key]
                
                # Validate handle (shape/dtype must match)
                if handle.is_valid(tensor):
                    self.stats.hits += 1
                    self.stats.bytes_saved += handle.tensor_bytes
                    self.model_timestamps[model_id] = time.time()
                    
                    logger.debug(
                        "Tensor registry HIT: model_id=%s, tensor=%s, "
                        "hit_rate=%.1f%%",
                        model_id,
                        tensor_name,
                        100 * self.stats.hits / (self.stats.hits + self.stats.misses or 1)
                    )
                    
                    self.stats.overhead_ms += (time.time() - start_time) * 1000
                    return False, handle
            
            # Cache miss
            self.stats.misses += 1
            
            # Check if we need to evict
            num_unique_models = len(set(k[0] for k in self.registry.keys()))
            if num_unique_models >= self.max_cached_models:
                await self._evict_oldest_model_locked()
            
            # Register new tensor
            if remote_handle is not None:
                # Check memory budgets
                if self._check_memory_budget(model_id, remote_handle.tensor_bytes):
                    self.registry[cache_key] = remote_handle
                    self.model_timestamps[model_id] = time.time()
                    self.model_versions[model_id] = model_version
                    self.model_memory_bytes[model_id] = (
                        self.model_memory_bytes.get(model_id, 0) + remote_handle.tensor_bytes
                    )
                    
                    logger.debug(
                        "Tensor registered: model_id=%s, tensor=%s, bytes=%d",
                        model_id,
                        tensor_name,
                        remote_handle.tensor_bytes
                    )
                else:
                    logger.warning(
                        "Tensor registration rejected: model_id=%s, tensor=%s, "
                        "bytes=%d exceeds budget",
                        model_id,
                        tensor_name,
                        remote_handle.tensor_bytes
                    )
            
            self.stats.overhead_ms += (time.time() - start_time) * 1000
            return True, remote_handle
    
    def _check_memory_budget(self, model_id: str, tensor_bytes: int) -> bool:
        """Check if registering this tensor would exceed memory limits."""
        model_memory = self.model_memory_bytes.get(model_id, 0)
        
        # Check per-model budget
        if self.max_bytes_per_model is not None:
            if model_memory + tensor_bytes > self.max_bytes_per_model:
                return False
        
        # Check total budget
        if self.max_total_bytes is not None:
            total_memory = sum(self.model_memory_bytes.values())
            if total_memory + tensor_bytes > self.max_total_bytes:
                return False
        
        return True
    
    def _is_ephemeral_tensor(self, tensor_name: str) -> bool:
        """Check if tensor should never be cached."""
        for pattern in self._ephemeral_patterns:
            if pattern in tensor_name.lower():
                return True
        return False
    
    async def _evict_oldest_model_locked(self):
        """Evict least recently used model (must be called with lock held)."""
        if not self.model_timestamps:
            return
        
        # Find oldest model
        oldest_model = min(
            self.model_timestamps.items(), 
            key=lambda x: x[1]
        )[0]
        
        # Remove all tensors for this model
        keys_to_remove = [k for k in self.registry.keys() if k[0] == oldest_model]
        for key in keys_to_remove:
            del self.registry[key]
        
        del self.model_timestamps[oldest_model]
        del self.model_memory_bytes[oldest_model]
        if oldest_model in self.model_versions:
            del self.model_versions[oldest_model]
        
        self.stats.evictions += 1
        logger.info("Evicted model (LRU): %s", oldest_model)
    
    def _invalidate_model_locked(self, model_id: str):
        """Invalidate all tensors for a model (for version updates)."""
        keys_to_remove = [k for k in self.registry.keys() if k[0] == model_id]
        for key in keys_to_remove:
            del self.registry[key]
        
        if model_id in self.model_timestamps:
            del self.model_timestamps[model_id]
        if model_id in self.model_memory_bytes:
            del self.model_memory_bytes[model_id]
        if model_id in self.model_versions:
            del self.model_versions[model_id]
        
        self.stats.invalidations += 1
        logger.info("Invalidated model cache: %s", model_id)
    
    async def invalidate_model(self, model_id: str):
        """Public API to invalidate all tensors for a model."""
        async with self._lock:
            self._invalidate_model_locked(model_id)
    
    async def invalidate_tensor(
        self, 
        model_id: str, 
        tensor_name: str, 
        model_version: int
    ):
        """Invalidate a specific tensor."""
        async with self._lock:
            cache_key = (model_id, tensor_name, model_version)
            if cache_key in self.registry:
                handle = self.registry[cache_key]
                del self.registry[cache_key]
                
                # Update memory tracking
                if model_id in self.model_memory_bytes:
                    self.model_memory_bytes[model_id] -= handle.tensor_bytes
                    if self.model_memory_bytes[model_id] <= 0:
                        del self.model_memory_bytes[model_id]
    
    def get_stats(self) -> Dict:
        """Get registry statistics."""
        total_requests = self.stats.hits + self.stats.misses
        hit_rate = (100 * self.stats.hits / total_requests) if total_requests > 0 else 0
        
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate_percent': hit_rate,
            'evictions': self.stats.evictions,
            'bytes_saved': self.stats.bytes_saved,
            'bytes_saved_mb': self.stats.bytes_saved / (1024 * 1024),
            'overhead_ms': self.stats.overhead_ms,
            'invalidations': self.stats.invalidations,
            'version_conflicts': self.stats.version_conflicts,
            'models_cached': len(set(k[0] for k in self.registry.keys())),
            'total_tensors_cached': len(self.registry),
            'total_bytes_cached': sum(self.model_memory_bytes.values()),
        }
    
    def create_handle(
        self,
        device_id: str,
        tensor: torch.Tensor,
        model_version: int = 0
    ) -> RemoteHandle:
        """Create a remote handle for a tensor."""
        return RemoteHandle(
            device_id=device_id,
            tensor_id=str(uuid.uuid4()),  # Stable UUID, not process-local id()
            shape=tensor.shape,
            dtype=tensor.dtype,
            timestamp=time.time(),
            version=model_version,
            tensor_bytes=tensor.numel() * tensor.element_size()
        )
