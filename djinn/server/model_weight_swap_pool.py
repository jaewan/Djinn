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
        
        # Pre-allocated pinned buffers for zero-copy streaming
        # model_id -> pinned CPU buffer (entire model as contiguous bytes)
        self._model_pinned_buffers: Dict[str, torch.Tensor] = {}
        
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
    
    def preallocate_pinned_buffer(self, model_id: str, size_bytes: int) -> bool:
        """
        Pre-allocate pinned buffer for a model during registration.
        
        This avoids allocation overhead during eviction and ensures the buffer
        is ready for direct streaming.
        
        Args:
            model_id: Model identifier
            size_bytes: Size of buffer to allocate
            
        Returns:
            True if successful, False if allocation failed
        """
        with self._lock:
            # Check if already allocated
            if model_id in self._model_pinned_buffers:
                logger.debug(f"Pinned buffer already exists for {model_id[:16]}...")
                return True
            
            try:
                size_mb = size_bytes / (1024**2)
                logger.info(f"Pre-allocating {size_mb:.1f}MB pinned buffer for {model_id[:16]}...")
                
                self._model_pinned_buffers[model_id] = torch.empty(
                    size_bytes,
                    dtype=torch.uint8,
                    device='cpu',
                    pin_memory=True
                )
                
                logger.info(f"✅ Pre-allocated pinned buffer for {model_id[:16]}...")
                return True
                
            except Exception as e:
                logger.error(f"Failed to pre-allocate pinned buffer for {model_id[:16]}...: {e}")
                return False
    
    def evict_model_direct(
        self,
        model_id: str,
        ring_buffer,  # WeightRingBuffer - avoid circular import
        gpu_device: int = 0
    ) -> int:
        """
        OPTIMIZED: Stream model weights directly from ring buffer to pre-allocated pinned buffer.
        
        This eliminates:
        - GPU->GPU clone (get_model_weights_from_buffer)
        - Per-tensor Python loops
        - Per-tensor pinned memory allocation
        
        Instead: Single bulk GPU->CPU copy of contiguous ring buffer region.
        
        Args:
            model_id: Model identifier
            ring_buffer: WeightRingBuffer instance
            gpu_device: GPU device this is being swapped from
            
        Returns:
            Bytes swapped, or 0 if rejected/failed
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
                return 0
            
            try:
                # Get model registration from ring buffer
                if model_id not in ring_buffer.registrations:
                    logger.error(f"Model {model_id[:16]}... not registered in ring buffer")
                    return 0
                
                reg = ring_buffer.registrations[model_id]
                
                # Check model state
                from djinn.backend.runtime.ring_buffer import ModelState
                if reg.state != ModelState.RESIDENT:
                    logger.error(
                        f"Cannot evict {model_id[:16]}...: not resident (state={reg.state})"
                    )
                    return 0
                
                size_bytes = reg.total_bytes
                size_mb = size_bytes / (1024**2)
                
                # Check 3: Memory limit
                new_total_mb = self.stats["current_swapped_mb"] + size_mb
                if new_total_mb > self.pool_size_gb * 1024:
                    self.stats["swap_rejections_memory"] += 1
                    logger.warning(
                        f"Swap rejected for {model_id[:16]}...: "
                        f"would exceed limit ({new_total_mb:.1f}MB > {self.pool_size_gb*1024:.0f}MB)"
                    )
                    return 0
                
                logger.info(
                    f"Swapping model {model_id[:16]}... to host (DIRECT): "
                    f"size={size_mb:.1f}MB, offset={reg.buffer_start_offset / 1024**2:.1f}MB"
                )
                
                # Get pre-allocated pinned buffer (should exist from registration)
                if model_id not in self._model_pinned_buffers:
                    logger.warning(
                        f"Pinned buffer not pre-allocated for {model_id[:16]}..., "
                        f"allocating now (this adds overhead)"
                    )
                    self._model_pinned_buffers[model_id] = torch.empty(
                        size_bytes,
                        dtype=torch.uint8,
                        device='cpu',
                        pin_memory=True
                    )
                
                pinned_buf = self._model_pinned_buffers[model_id]
                
                # Handle wrapped models (when buffer wraps around due to skip-end)
                is_wrapped = reg.buffer_end_offset < reg.buffer_start_offset
                
                if is_wrapped:
                    # Wrapped model: copy in two parts [start:capacity] + [0:end]
                    # Use reg.total_bytes (actual model size) to determine copy sizes
                    # Buffer offsets may include padding, so we trust the model size
                    
                    # Calculate part sizes based on buffer layout
                    buffer_part1_size = ring_buffer.capacity_bytes - reg.buffer_start_offset
                    buffer_part2_size = reg.buffer_end_offset
                    
                    # Clamp to actual model size (pinned buffer size)
                    model_size = pinned_buf.numel()  # = reg.total_bytes
                    
                    # Determine how much to copy from each part
                    if buffer_part1_size >= model_size:
                        # Entire model fits in part 1 (shouldn't normally happen for wrapped)
                        part1_size = model_size
                        part2_size = 0
                    else:
                        part1_size = buffer_part1_size
                        part2_size = min(buffer_part2_size, model_size - part1_size)
                    
                    logger.info(
                        f"Model {model_id[:16]}... is wrapped in ring buffer - "
                        f"using two-part bulk copy: part1={part1_size / 1024**2:.1f}MB, "
                        f"part2={part2_size / 1024**2:.1f}MB"
                    )
                    
                    # Two-part bulk copy: ring_buffer -> pinned (GPU->CPU)
                    if self.swap_stream:
                        with torch.cuda.stream(self.swap_stream):
                            # Part 1: [start:start+part1_size] -> pinned[0:part1_size]
                            if part1_size > 0:
                                pinned_buf[:part1_size].copy_(
                                    ring_buffer.buffer[reg.buffer_start_offset:reg.buffer_start_offset + part1_size],
                                    non_blocking=True
                                )
                            # Part 2: [0:part2_size] -> pinned[part1_size:part1_size+part2_size]
                            if part2_size > 0:
                                pinned_buf[part1_size:part1_size + part2_size].copy_(
                                    ring_buffer.buffer[:part2_size],
                                    non_blocking=True
                                )
                            swap_event = torch.cuda.Event()
                            swap_event.record()
                    else:
                        # CPU fallback
                        if part1_size > 0:
                            pinned_buf[:part1_size].copy_(
                                ring_buffer.buffer[reg.buffer_start_offset:reg.buffer_start_offset + part1_size]
                            )
                        if part2_size > 0:
                            pinned_buf[part1_size:part1_size + part2_size].copy_(
                                ring_buffer.buffer[:part2_size]
                            )
                        swap_event = None
                else:
                    # Normal contiguous model
                    buffer_slice = ring_buffer.buffer[reg.buffer_start_offset:reg.buffer_end_offset]
                    
                    if buffer_slice.numel() != pinned_buf.numel():
                        raise RuntimeError(
                            f"Size mismatch for {model_id[:16]}...: "
                            f"buffer_slice={buffer_slice.numel()} bytes, "
                            f"pinned_buf={pinned_buf.numel()} bytes"
                        )
                    
                    # Single bulk copy: ring_buffer -> pinned (GPU->CPU)
                    if self.swap_stream:
                        with torch.cuda.stream(self.swap_stream):
                            pinned_buf.copy_(buffer_slice, non_blocking=True)
                            swap_event = torch.cuda.Event()
                            swap_event.record()
                    else:
                        # CPU fallback
                        pinned_buf.copy_(buffer_slice)
                        swap_event = None
                
                # Track the swap (store metadata, not the actual weights dict)
                mapping = ModelSwapMapping(
                    model_id=model_id,
                    cpu_weights={},  # Empty - we use pinned_buf instead
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
                    f"✅ Evicted model to host (DIRECT): model_id={model_id[:16]}..., "
                    f"size={size_mb:.1f}MB, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB, "
                    f"models={len(self.swapped_models)}"
                )
                
                # Record success metric
                metrics = get_metrics()
                metrics.record_swap_success()
                
                return size_bytes
                
            except Exception as e:
                self.stats["swap_errors"] += 1
                logger.error(
                    f"[CRITICAL] Direct swap failed for {model_id[:16]}...: {e}",
                    exc_info=True
                )
                metrics = get_metrics()
                metrics.record_swap_failure()
                return 0
    
    def synchronize_eviction(self, model_id: str) -> float:
        """
        Wait for a specific model's eviction to complete.
        
        This should be called BEFORE restoration to prevent PCIe bandwidth
        contention between eviction (GPU->CPU) and restoration (CPU->GPU).
        
        Args:
            model_id: Model identifier
            
        Returns:
            Time spent waiting in milliseconds
        """
        import time as time_module
        
        with self._lock:
            if model_id not in self.swapped_models:
                return 0.0
            
            mapping = self.swapped_models[model_id]
            if mapping.completion_event is None:
                return 0.0
            
            start = time_module.perf_counter()
            mapping.completion_event.synchronize()
            wait_ms = (time_module.perf_counter() - start) * 1000
            
            if wait_ms > 1.0:
                bandwidth = (mapping.size_bytes / 1024**3) / (wait_ms / 1000) if wait_ms > 0 else 0
                logger.info(
                    f"  Waited {wait_ms:.1f}ms for eviction of {model_id[:16]}... "
                    f"({bandwidth:.1f} GB/s)"
                )
            
            return wait_ms
    
    def synchronize_all_evictions(self) -> float:
        """
        Wait for all pending evictions to complete.
        
        This should be called before starting restorations to ensure
        PCIe bandwidth is fully available.
        
        Returns:
            Total time spent waiting in milliseconds
        """
        import time as time_module
        
        with self._lock:
            total_wait_ms = 0.0
            
            for model_id, mapping in self.swapped_models.items():
                if mapping.completion_event is not None:
                    start = time_module.perf_counter()
                    mapping.completion_event.synchronize()
                    wait_ms = (time_module.perf_counter() - start) * 1000
                    total_wait_ms += wait_ms
            
            if total_wait_ms > 1.0:
                logger.info(f"  Total eviction sync time: {total_wait_ms:.1f}ms")
            
            return total_wait_ms
    
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
    
    def restore_model_direct(
        self,
        model_id: str,
        ring_buffer,  # WeightRingBuffer - avoid circular import
        target_device: torch.device = None
    ) -> int:
        """
        OPTIMIZED: Stream model weights directly from pinned buffer to ring buffer.
        
        This eliminates:
        - Per-tensor Python loops
        - Per-tensor GPU allocation
        - GPU->GPU copy after restoration
        
        Instead: Single bulk CPU->GPU copy to contiguous ring buffer region.
        
        Args:
            model_id: Model identifier
            ring_buffer: WeightRingBuffer instance
            target_device: Target GPU device (defaults to cuda:0)
            
        Returns:
            Bytes restored, or 0 if failed
            
        Raises:
            RuntimeError: If restore fails
        """
        if target_device is None:
            target_device = torch.device('cuda:0')
        
        with self._lock:
            if model_id not in self.swapped_models:
                logger.debug(f"Cannot restore: model {model_id[:16]}... not swapped")
                return 0
            
            mapping = self.swapped_models[model_id]
            
            # Wait for swap completion if event exists
            import time as time_module
            evict_wait_start = time_module.perf_counter()
            if mapping.completion_event is not None:
                mapping.completion_event.synchronize()
            evict_wait_ms = (time_module.perf_counter() - evict_wait_start) * 1000
            if evict_wait_ms > 1.0:
                logger.info(
                    f"  Waited {evict_wait_ms:.1f}ms for prior eviction of {model_id[:16]}... to complete"
                )
            
            try:
                # Get pinned buffer
                if model_id not in self._model_pinned_buffers:
                    raise RuntimeError(
                        f"Pinned buffer not found for {model_id[:16]}... "
                        f"(was evict_model_direct used?)"
                    )
                
                pinned_buf = self._model_pinned_buffers[model_id]
                size_mb = mapping.size_bytes / (1024**2)
                
                logger.info(
                    f"Restoring model {model_id[:16]}... from host (DIRECT): "
                    f"size={size_mb:.1f}MB"
                )
                
                # Get model registration
                if model_id not in ring_buffer.registrations:
                    raise RuntimeError(f"Model {model_id[:16]}... not registered in ring buffer")
                
                reg = ring_buffer.registrations[model_id]
                
                # PROFILING: Track timing breakdown
                import time as time_module
                profile_start = time_module.perf_counter()
                
                # Allocate space in ring buffer (may trigger defragmentation)
                # This must be called with ring_buffer lock held
                with ring_buffer.lock:
                    alloc_start = time_module.perf_counter()
                    start_offset = ring_buffer._allocate_contiguous_space(reg.total_bytes)
                    alloc_ms = (time_module.perf_counter() - alloc_start) * 1000
                    
                    # Single bulk copy: pinned -> ring_buffer (CPU->GPU)
                    copy_start = time_module.perf_counter()
                    if self.restore_stream:
                        with torch.cuda.stream(self.restore_stream):
                            ring_buffer.buffer[start_offset:start_offset + reg.total_bytes].copy_(
                                pinned_buf,
                                non_blocking=True
                            )
                            # Record event for completion tracking
                            restore_event = torch.cuda.Event()
                            restore_event.record()
                    else:
                        # CPU fallback
                        ring_buffer.buffer[start_offset:start_offset + reg.total_bytes].copy_(pinned_buf)
                        restore_event = None
                    copy_queue_ms = (time_module.perf_counter() - copy_start) * 1000
                    
                    # Wait for copy completion (outside stream context for accurate timing)
                    sync_start = time_module.perf_counter()
                    if restore_event:
                        restore_event.synchronize()
                    sync_ms = (time_module.perf_counter() - sync_start) * 1000
                    
                    # Update model's buffer tracking first
                    update_start = time_module.perf_counter()
                    old_start = reg.buffer_start_offset
                    reg.buffer_start_offset = start_offset
                    reg.buffer_end_offset = start_offset + reg.total_bytes
                    
                    # Update layer offsets in bulk (compute offset delta)
                    offset_delta = start_offset - old_start
                    for layer_name in reg.layer_names:
                        alloc = reg.layer_allocations[layer_name]
                        if alloc.is_resident and alloc.offset != -1:
                            alloc.offset += offset_delta
                    
                    # Update state
                    from djinn.backend.runtime.ring_buffer import ModelState
                    reg.state = ModelState.RESIDENT
                    reg.swap_timestamp = None
                    update_ms = (time_module.perf_counter() - update_start) * 1000
                
                # Log timing breakdown
                total_profile_ms = (time_module.perf_counter() - profile_start) * 1000
                bandwidth_gbps = (mapping.size_bytes / 1024**3) / (sync_ms / 1000) if sync_ms > 0 else 0
                logger.info(
                    f"  Restore profile {model_id[:16]}...: "
                    f"alloc={alloc_ms:.1f}ms, copy_queue={copy_queue_ms:.1f}ms, "
                    f"sync={sync_ms:.1f}ms ({bandwidth_gbps:.1f} GB/s actual), "
                    f"update={update_ms:.1f}ms, total={total_profile_ms:.1f}ms"
                )
                
                # Remove from swap pool
                del self.swapped_models[model_id]
                
                # Update stats
                self.stats["restores_performed"] += 1
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_models.values()
                ) / (1024**2)
                
                logger.info(
                    f"✅ Restored model from host (DIRECT): model_id={model_id[:16]}..., "
                    f"size={size_mb:.1f}MB, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB"
                )
                
                return mapping.size_bytes
                
            except Exception as e:
                self.stats["restore_errors"] += 1
                logger.error(
                    f"[CRITICAL] Direct restore failed for {model_id[:16]}...: {e}",
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

