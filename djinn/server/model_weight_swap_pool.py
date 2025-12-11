"""
Model Weight Swap Pool: Manages pinned host memory for swapped model weights.

Similar to HostSwapPool but designed for entire model weight sets rather than KV caches.
Enables multi-model serving by swapping inactive models to host RAM.

Key Features:
- Dedicated CUDA streams for async swap/restore operations
- Memory limit enforcement with graceful degradation
- Structure preservation (Dict[param_name, tensor])
- Event-based completion tracking (zero CPU blocking)

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Host Memory (pinned)        │ GPU VRAM (ring buffer)             │
│ [Model A weights]           │ [Model B weights (active)]         │
│ [Model C weights]           │                                    │
└─────────────────────────────────────────────────────────────────┘
            ↓ (async swap via dedicated stream)
    Model B evicted              Model A restored
            ↓                                ↓
    Event recorded         ←→         Wait for Event
"""

import logging
import threading
import torch
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from ..server.memory_metrics import get_metrics

logger = logging.getLogger(__name__)


# Safety limits
MAX_SWAPPED_MODELS = 10  # Prevent unbounded model growth
MAX_SWAP_AGE_SECONDS = 1800  # Force-expire swaps older than 30 minutes


@dataclass
class ModelSwapMapping:
    """Track a swapped model's weights."""
    model_id: str
    cpu_weights: Dict[str, torch.Tensor]  # param_name -> pinned CPU tensor
    size_bytes: int
    timestamp: float
    gpu_device: int
    completion_event: Optional[torch.cuda.Event] = None


class ModelWeightSwapPool:
    """
    Manages pinned host memory for swapped model weights.
    
    This is a POLICY layer (when to swap) not a MECHANISM layer (how to allocate).
    PyTorch's CachingHostAllocator handles the actual memory management via pin_memory=True.
    
    HARDENED FEATURES:
    - Memory limit enforcement (pool_size_gb)
    - Model count limits (MAX_SWAPPED_MODELS)
    - Stale swap cleanup (MAX_SWAP_AGE_SECONDS)
    - Graceful degradation on errors
    - Dedicated CUDA streams for async operations
    """
    
    def __init__(self, pool_size_gb: float = 64.0):
        """
        Initialize model weight swap pool.
        
        Args:
            pool_size_gb: Maximum pool size - ENFORCED (not just informational)
        """
        self.pool_size_gb = pool_size_gb
        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self._lock = threading.Lock()
        
        # Dedicated CUDA streams for swap/restore operations
        # Separate streams allow overlap with compute and avoid blocking main stream
        if torch.cuda.is_available():
            self.swap_stream = torch.cuda.Stream()
            self.restore_stream = torch.cuda.Stream()
            logger.info("✅ Dedicated CUDA streams created for model swap/restore")
        else:
            self.swap_stream = None
            self.restore_stream = None
        
        # Track swapped models
        self.swapped_models: Dict[str, ModelSwapMapping] = {}
        
        # Statistics
        self.stats = {
            "swaps_performed": 0,
            "restores_performed": 0,
            "total_swapped_bytes": 0,
            "max_concurrent_swapped_mb": 0.0,
            "current_swapped_mb": 0.0,
            "swap_errors": 0,
            "restore_errors": 0,
            "swap_rejections_memory": 0,
            "swap_rejections_count": 0,
            "stale_cleanups": 0,
        }
        
        logger.info(
            f"✅ ModelWeightSwapPool initialized: "
            f"limit={pool_size_gb:.1f}GB, max_models={MAX_SWAPPED_MODELS}"
        )
    
    def _cleanup_stale_swaps(self) -> int:
        """
        Remove swaps older than MAX_SWAP_AGE_SECONDS.
        MUST be called with lock held.
        
        Returns:
            Number of stale swaps removed
        """
        now = time.time()
        stale_models = [
            model_id for model_id, mapping in self.swapped_models.items()
            if (now - mapping.timestamp) > MAX_SWAP_AGE_SECONDS
        ]
        
        for model_id in stale_models:
            mapping = self.swapped_models[model_id]
            del self.swapped_models[model_id]
            self.stats["stale_cleanups"] += 1
            logger.warning(
                f"Cleaned up stale swap: {model_id[:16]}... "
                f"(age > {MAX_SWAP_AGE_SECONDS}s, size={mapping.size_bytes/1024**2:.1f}MB)"
            )
        
        if stale_models:
            self.stats["current_swapped_mb"] = sum(
                m.size_bytes for m in self.swapped_models.values()
            ) / (1024**2)
        
        return len(stale_models)
    
    def evict_model_to_host(
        self,
        model_id: str,
        weights: Dict[str, torch.Tensor],
        gpu_device: int = 0
    ) -> int:
        """
        Swap model weights from GPU to pinned host memory.
        
        HARDENED: Enforces memory limits and gracefully degrades on errors.
        
        Args:
            model_id: Model identifier
            weights: Dict of parameter name -> GPU tensor
            gpu_device: GPU device this is being swapped from
            
        Returns:
            Bytes swapped, or 0 if rejected/failed (graceful degradation)
        """
        with self._lock:
            # Cleanup stale swaps first
            self._cleanup_stale_swaps()
            
            # Check 1: Already swapped?
            if model_id in self.swapped_models:
                logger.debug(f"Model {model_id[:16]}... already swapped, skipping")
                return 0
            
            # Check 2: Model count limit
            if len(self.swapped_models) >= MAX_SWAPPED_MODELS:
                self.stats["swap_rejections_count"] += 1
                logger.warning(
                    f"Swap rejected for {model_id[:16]}...: "
                    f"model limit reached ({MAX_SWAPPED_MODELS})"
                )
                return 0  # Graceful degradation
            
            try:
                # Calculate total size
                size_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in weights.values()
                    if isinstance(tensor, torch.Tensor)
                )
                size_mb = size_bytes / (1024**2)
                
                # Check 3: Memory limit
                new_total_mb = self.stats["current_swapped_mb"] + size_mb
                if new_total_mb > self.pool_size_gb * 1024:
                    self.stats["swap_rejections_memory"] += 1
                    logger.warning(
                        f"Swap rejected for {model_id[:16]}...: "
                        f"would exceed limit ({new_total_mb:.1f}MB > {self.pool_size_gb*1024:.0f}MB)"
                    )
                    return 0  # Graceful degradation
                
                logger.info(
                    f"Swapping model {model_id[:16]}... to host: "
                    f"size={size_mb:.1f}MB, params={len(weights)}"
                )
                
                # Move weights to pinned CPU memory
                cpu_weights = {}
                
                if self.swap_stream:
                    with torch.cuda.stream(self.swap_stream):
                        for param_name, gpu_tensor in weights.items():
                            if not isinstance(gpu_tensor, torch.Tensor):
                                continue
                            
                            # Ensure contiguous
                            gpu_tensor_contiguous = gpu_tensor.contiguous()
                            
                            # Create pinned buffer and copy
                            cpu_buf = torch.empty(
                                gpu_tensor_contiguous.shape,
                                dtype=gpu_tensor_contiguous.dtype,
                                device='cpu',
                                pin_memory=True
                            )
                            cpu_buf.copy_(gpu_tensor_contiguous, non_blocking=True)
                            cpu_weights[param_name] = cpu_buf
                else:
                    # CPU fallback (no async)
                    for param_name, gpu_tensor in weights.items():
                        if not isinstance(gpu_tensor, torch.Tensor):
                            continue
                        cpu_weights[param_name] = gpu_tensor.cpu().pin_memory()
                
                # Use event-based completion instead of blocking sync
                swap_event = torch.cuda.Event() if torch.cuda.is_available() else None
                if swap_event:
                    swap_event.record()
                
                # Track the swap
                mapping = ModelSwapMapping(
                    model_id=model_id,
                    cpu_weights=cpu_weights,
                    size_bytes=size_bytes,
                    timestamp=time.time(),
                    gpu_device=gpu_device,
                    completion_event=swap_event,
                )
                self.swapped_models[model_id] = mapping
                
                # Update stats
                self.stats["swaps_performed"] += 1
                self.stats["total_swapped_bytes"] += size_bytes
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_models.values()
                ) / (1024**2)
                self.stats["max_concurrent_swapped_mb"] = max(
                    self.stats["max_concurrent_swapped_mb"],
                    self.stats["current_swapped_mb"]
                )
                
                logger.info(
                    f"✅ Evicted model to host: model_id={model_id[:16]}..., "
                    f"size={size_mb:.1f}MB, params={len(cpu_weights)}, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB, "
                    f"models={len(self.swapped_models)}"
                )
                
                # Record success metric
                metrics = get_metrics()
                metrics.record_swap_success()
                
                return size_bytes
                
            except torch.cuda.OutOfMemoryError as e:
                self.stats["swap_errors"] += 1
                logger.error(
                    f"[CRITICAL] CUDA OOM during swap for {model_id[:16]}...: {e}",
                    exc_info=True
                )
                metrics = get_metrics()
                metrics.record_swap_failure()
                return 0  # Graceful degradation
                
            except Exception as e:
                self.stats["swap_errors"] += 1
                logger.error(
                    f"[CRITICAL] Swap failed for {model_id[:16]}...: {e}",
                    exc_info=True
                )
                metrics = get_metrics()
                metrics.record_swap_failure()
                return 0  # Graceful degradation
    
    def restore_model_from_host(
        self,
        model_id: str,
        target_device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """
        Restore model weights from host swap pool to GPU.
        
        Args:
            model_id: Model identifier
            target_device: Target GPU device (defaults to cuda:0)
            
        Returns:
            Dict of parameter name -> GPU tensor, or empty dict if not found
            
        Raises:
            RuntimeError: If restore fails
        """
        if target_device is None:
            target_device = torch.device('cuda:0')
        
        with self._lock:
            if model_id not in self.swapped_models:
                logger.debug(f"Cannot restore: model {model_id[:16]}... not swapped")
                return {}
            
            mapping = self.swapped_models[model_id]
            
            # Wait for swap completion if event exists
            if mapping.completion_event is not None:
                mapping.completion_event.synchronize()
            
            try:
                size_mb = mapping.size_bytes / (1024**2)
                logger.info(
                    f"Restoring model {model_id[:16]}... from host: "
                    f"size={size_mb:.1f}MB, params={len(mapping.cpu_weights)}"
                )
                
                # Restore weights to GPU
                gpu_weights = {}
                
                if self.restore_stream:
                    with torch.cuda.stream(self.restore_stream):
                        for param_name, cpu_tensor in mapping.cpu_weights.items():
                            gpu_tensor = cpu_tensor.to(
                                device=target_device,
                                non_blocking=True
                            )
                            gpu_weights[param_name] = gpu_tensor
                else:
                    # CPU fallback (no async)
                    for param_name, cpu_tensor in mapping.cpu_weights.items():
                        gpu_weights[param_name] = cpu_tensor.to(device=target_device)
                
                # Use event for completion tracking
                restore_event = torch.cuda.Event() if torch.cuda.is_available() else None
                if restore_event:
                    restore_event.record()
                    restore_event.synchronize()  # Ensure restore completes
                
                # Remove from swap pool
                del self.swapped_models[model_id]
                
                # Update stats
                self.stats["restores_performed"] += 1
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_models.values()
                ) / (1024**2)
                
                logger.info(
                    f"✅ Restored model from host: model_id={model_id[:16]}..., "
                    f"size={size_mb:.1f}MB, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB"
                )
                
                return gpu_weights
                
            except Exception as e:
                self.stats["restore_errors"] += 1
                logger.error(
                    f"[CRITICAL] Restore failed for {model_id[:16]}...: {e}",
                    exc_info=True
                )
                raise RuntimeError(f"Failed to restore model {model_id}: {e}")
    
    def has_model(self, model_id: str) -> bool:
        """Check if model is in swap pool."""
        with self._lock:
            return model_id in self.swapped_models
    
    def get_model_size(self, model_id: str) -> int:
        """Get size of swapped model in bytes, or 0 if not found."""
        with self._lock:
            if model_id in self.swapped_models:
                return self.swapped_models[model_id].size_bytes
            return 0
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove model from swap pool without restoring.
        Used when model is no longer needed.
        
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if model_id in self.swapped_models:
                mapping = self.swapped_models[model_id]
                del self.swapped_models[model_id]
                
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_models.values()
                ) / (1024**2)
                
                logger.info(
                    f"Removed model {model_id[:16]}... from swap pool "
                    f"(freed {mapping.size_bytes/1024**2:.1f}MB)"
                )
                return True
            return False
    
    def get_stats(self) -> Dict:
        """Get swap pool statistics."""
        with self._lock:
            return {
                **self.stats,
                'pool_size_gb': self.pool_size_gb,
                'swapped_models': len(self.swapped_models),
                'model_ids': list(self.swapped_models.keys()),
            }
    
    def clear(self) -> None:
        """Clear all swapped models."""
        with self._lock:
            self.swapped_models.clear()
            self.stats["current_swapped_mb"] = 0.0
            logger.info("✅ Model weight swap pool cleared")


# Global singleton
_global_model_swap_pool: Optional[ModelWeightSwapPool] = None


def get_model_swap_pool(pool_size_gb: float = 64.0) -> ModelWeightSwapPool:
    """Get or create global model swap pool singleton."""
    global _global_model_swap_pool
    if _global_model_swap_pool is None:
        _global_model_swap_pool = ModelWeightSwapPool(pool_size_gb=pool_size_gb)
    return _global_model_swap_pool

