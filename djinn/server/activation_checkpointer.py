"""
Activation Checkpointer: Save/restore intermediate activations to host memory.

Purpose:
- Enable breakpoint debugging by checkpointing activations at layer boundaries
- Reuse HostSwapPool pattern for pinned memory allocation
- Support zero-cost context switching (pause GPU, run another job, resume)

Architecture:
- Checkpoint: Serialize multiple activation tensors into pinned host buffer
- Restore: Transfer activations from host back to GPU via async DMA
- Metadata tracking: Shapes, dtypes, offsets for correct reconstruction

Integration:
- BreakpointManager calls checkpoint() at layer boundaries
- BreakpointExecutor calls restore() to resume execution
- SemanticIdleDetector triggers automatic checkpointing on idle detection
"""

import logging
import threading
import time
import torch
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TensorMetadata:
    """Metadata for a single tensor in checkpoint."""
    name: str
    dtype: torch.dtype
    shape: torch.Size
    nbytes: int
    host_offset: int  # Offset in pinned buffer


@dataclass
class CheckpointMetadata:
    """Metadata for a complete checkpoint."""
    checkpoint_id: str
    session_id: str
    layer_index: int
    timestamp: float
    total_bytes: int
    tensors: Dict[str, TensorMetadata] = field(default_factory=dict)
    device: torch.device = None


class ActivationCheckpointer:
    """
    Save and restore intermediate activations to/from pinned host memory.
    
    Enables breakpoint debugging by allowing model execution to be paused,
    intermediate activations saved, GPU yielded to another session, then
    resumed with activations restored.
    """
    
    def __init__(
        self,
        host_pool_size_gb: float = 64.0,
        alignment: int = 256
    ):
        """
        Initialize activation checkpointer.
        
        Args:
            host_pool_size_gb: Size of pinned memory pool for checkpoints
            alignment: Byte alignment for allocations
        """
        self.alignment = alignment
        self._lock = threading.Lock()
        
        # Allocate pinned memory pool
        self.pool_size_bytes = int(host_pool_size_gb * 1024**3)
        try:
            pool_tensor = torch.empty(self.pool_size_bytes, dtype=torch.uint8)
            self.pool = pool_tensor.pin_memory()
            logger.info(f"✅ Allocated activation checkpoint pool: {host_pool_size_gb:.1f}GB")
        except RuntimeError as e:
            logger.error(f"❌ Failed to allocate pinned memory: {e}")
            raise RuntimeError(
                f"Cannot allocate {host_pool_size_gb}GB pinned memory for activation checkpoints. "
                f"Check ulimit -l and available system RAM."
            ) from e
        
        # Checkpoint tracking
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.current_offset = 0
        self._freed_regions: List[Tuple[int, int]] = []  # (offset, size) for fragmentation tracking
        
        # Statistics
        self.stats = {
            "checkpoints_created": 0,
            "checkpoints_restored": 0,
            "total_checkpoint_bytes": 0,
            "max_concurrent_checkpoints_mb": 0.0,
            "checkpoint_errors": 0,
            "restore_errors": 0,
        }
        
        logger.info(
            f"ActivationCheckpointer initialized: "
            f"pool_size={host_pool_size_gb:.1f}GB, "
            f"alignment={alignment}B"
        )
    
    def checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        layer_index: int,
        activations: Dict[str, torch.Tensor],
        device: torch.device = None
    ) -> Tuple[CheckpointMetadata, float]:
        """
        Save activations to pinned host memory.
        
        Args:
            session_id: Session identifier
            checkpoint_id: Unique checkpoint identifier
            layer_index: Layer number for reference
            activations: Dict of tensor name -> tensor to checkpoint
            device: CUDA device (if None, infer from first activation)
        
        Returns:
            (CheckpointMetadata, elapsed_time_seconds)
        
        Raises:
            RuntimeError: If insufficient space or invalid activations
        """
        if not activations:
            logger.warning(f"Checkpoint {checkpoint_id}: no activations provided")
            return CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                session_id=session_id,
                layer_index=layer_index,
                timestamp=time.time(),
                total_bytes=0,
                device=device
            ), 0.0
        
        start_time = time.perf_counter()
        
        try:
            # Infer device from first activation if not specified
            if device is None:
                first_tensor = next(iter(activations.values()))
                device = first_tensor.device
            
            # Calculate total size needed
            total_bytes = sum(t.element_size() * t.numel() for t in activations.values())
            
            with self._lock:
                # Align offset
                aligned_offset = (self.current_offset + self.alignment - 1) & ~(self.alignment - 1)
                
                # Check capacity
                if aligned_offset + total_bytes > self.pool_size_bytes:
                    available = self.pool_size_bytes - aligned_offset
                    raise RuntimeError(
                        f"Checkpoint pool exhausted: requested {total_bytes / 1024**2:.1f}MB, "
                        f"available {available / 1024**2:.1f}MB in {self.pool_size_bytes / 1024**3:.1f}GB pool"
                    )
                
                # Copy tensors to host
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    session_id=session_id,
                    layer_index=layer_index,
                    timestamp=time.time(),
                    total_bytes=total_bytes,
                    device=device,
                    tensors={}
                )
                
                current_host_offset = aligned_offset
                
                for name, tensor in activations.items():
                    # Get tensor info
                    tensor_bytes = tensor.element_size() * tensor.numel()
                    
                    # Create view in host pool
                    host_view = self.pool[current_host_offset:current_host_offset + tensor_bytes]
                    host_view_reshaped = host_view.view(tensor.dtype)[:tensor.numel()]
                    
                    # Move to CPU and copy (handles GPU tensors efficiently)
                    if tensor.is_cuda:
                        # Move to CPU first (necessary for copy)
                        tensor_cpu = tensor.cpu()
                        host_view_reshaped.copy_(tensor_cpu.flatten())
                    else:
                        # Already on CPU
                        host_view_reshaped.copy_(tensor.flatten())
                    
                    # Record metadata
                    metadata.tensors[name] = TensorMetadata(
                        name=name,
                        dtype=tensor.dtype,
                        shape=tensor.shape,
                        nbytes=tensor_bytes,
                        host_offset=current_host_offset
                    )
                    
                    current_host_offset += tensor_bytes
                
                # Track checkpoint
                self.checkpoints[checkpoint_id] = metadata
                self.current_offset = current_host_offset
                
                # Update statistics
                self.stats["checkpoints_created"] += 1
                self.stats["total_checkpoint_bytes"] += total_bytes
                current_checkpoints_mb = sum(
                    m.total_bytes for m in self.checkpoints.values()
                ) / 1024**2
                self.stats["max_concurrent_checkpoints_mb"] = max(
                    self.stats["max_concurrent_checkpoints_mb"],
                    current_checkpoints_mb
                )
                
                elapsed = time.perf_counter() - start_time
                
                logger.info(
                    f"✅ Checkpointed activations: checkpoint_id={checkpoint_id}, "
                    f"session_id={session_id}, layer={layer_index}, "
                    f"size={total_bytes / 1024**2:.1f}MB, "
                    f"tensors={len(activations)}, "
                    f"time={elapsed*1000:.1f}ms"
                )
                
                return metadata, elapsed
        
        except Exception as e:
            self.stats["checkpoint_errors"] += 1
            logger.error(f"❌ Checkpoint failed: {e}")
            raise
    
    def restore(
        self,
        checkpoint_id: str,
        device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Restore activations from host memory to GPU.
        
        Args:
            checkpoint_id: Checkpoint to restore
            device: Target GPU device
        
        Returns:
            (Dict[name -> tensor], elapsed_time_seconds)
        
        Raises:
            KeyError: If checkpoint not found
        """
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                if checkpoint_id not in self.checkpoints:
                    raise KeyError(f"Checkpoint {checkpoint_id} not found")
                
                metadata = self.checkpoints[checkpoint_id]
                activations = {}
                
                # Restore each tensor
                for name, tensor_meta in metadata.tensors.items():
                    # Get view from host pool
                    host_view = self.pool[tensor_meta.host_offset:tensor_meta.host_offset + tensor_meta.nbytes]
                    
                    # Reshape to original dtype (efficient - just changes metadata, no copy)
                    element_size = torch.tensor([], dtype=tensor_meta.dtype).element_size()
                    host_view_typed = host_view.view(tensor_meta.dtype)[:tensor_meta.nbytes // element_size]
                    
                    # Create GPU tensor and copy
                    gpu_tensor = torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device=device
                    )
                    
                    # Copy from host to GPU
                    gpu_tensor.copy_(host_view_typed.reshape(tensor_meta.shape))
                    
                    activations[name] = gpu_tensor
                
                self.stats["checkpoints_restored"] += 1
                elapsed = time.perf_counter() - start_time
                
                logger.info(
                    f"✅ Restored activations: checkpoint_id={checkpoint_id}, "
                    f"session_id={metadata.session_id}, layer={metadata.layer_index}, "
                    f"size={metadata.total_bytes / 1024**2:.1f}MB, "
                    f"tensors={len(activations)}, "
                    f"time={elapsed*1000:.1f}ms, "
                    f"async={self.enable_async}"
                )
                
                return activations, elapsed
        
        except Exception as e:
            self.stats["restore_errors"] += 1
            logger.error(f"❌ Restore failed: {e}")
            raise
    
    def release_checkpoint(self, checkpoint_id: str) -> int:
        """
        Release checkpoint to free host memory.
        
        Args:
            checkpoint_id: Checkpoint to release
        
        Returns:
            Bytes freed
        """
        with self._lock:
            metadata = self.checkpoints.pop(checkpoint_id, None)
            if metadata is None:
                return 0
            
            bytes_freed = metadata.total_bytes
            
            # Track freed region for potential compaction
            if metadata.tensors:
                first_offset = min(m.host_offset for m in metadata.tensors.values())
                self._freed_regions.append((first_offset, bytes_freed))
                
                # Check if we can compact (freed region at end)
                if first_offset + bytes_freed == self.current_offset:
                    self.current_offset = first_offset
                    self._freed_regions.remove((first_offset, bytes_freed))
                    logger.debug(f"Immediate compaction: freed end region {bytes_freed / 1024**2:.1f}MB")
            
            logger.debug(
                f"Released checkpoint {checkpoint_id}: "
                f"freed {bytes_freed / 1024**2:.1f}MB, "
                f"fragmented_regions={len(self._freed_regions)}"
            )
            
            return bytes_freed
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a checkpoint without restoring."""
        with self._lock:
            return self.checkpoints.get(checkpoint_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpointer statistics."""
        with self._lock:
            current_checkpoints_mb = sum(
                m.total_bytes for m in self.checkpoints.values()
            ) / 1024**2
            
            return {
                **self.stats,
                "pool_size_bytes": self.pool_size_bytes,
                "pool_size_gb": self.pool_size_bytes / 1024**3,
                "current_allocated_bytes": self.current_offset,
                "current_allocated_mb": self.current_offset / 1024**2,
                "active_checkpoints": len(self.checkpoints),
                "current_checkpoints_mb": current_checkpoints_mb,
                "utilization_percent": (self.current_offset / self.pool_size_bytes * 100)
                    if self.pool_size_bytes > 0 else 0,
            }
    
    def clear(self) -> None:
        """Clear all checkpoints and reset allocator."""
        with self._lock:
            cleared_count = len(self.checkpoints)
            self.checkpoints.clear()
            self.current_offset = 0
            self._freed_regions.clear()
            logger.info(f"Cleared activation checkpoint pool ({cleared_count} checkpoints)")


# Global singleton instance
_global_checkpointer: Optional[ActivationCheckpointer] = None
_checkpointer_lock = threading.Lock()


def get_activation_checkpointer(pool_size_gb: float = 64.0) -> ActivationCheckpointer:
    """Get or create global activation checkpointer singleton."""
    global _global_checkpointer
    
    if _global_checkpointer is None:
        with _checkpointer_lock:
            if _global_checkpointer is None:
                _global_checkpointer = ActivationCheckpointer(host_pool_size_gb=pool_size_gb)
    
    return _global_checkpointer

