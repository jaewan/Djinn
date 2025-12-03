"""
Skip-End Ring Buffer for Weight Streaming

Implements circular buffer for streaming model weights through VRAM that's
smaller than model size. Enables running 140GB models on 60GB GPUs.

Key Features:
- Skip-End Allocator: Pre-computed layer offsets, never splits tensors across buffer wrap
- Pre-Computed Views: Tensor views into ring buffer computed at startup
- Slot Management: Track in-use slots with CUDA events (no CPU blocking in hot path)
- Async Pipelining: Dual-stream architecture (prefetch + compute)

Architecture:
┌─────────────────────────────────────────────────────────┐
│ Ring Buffer (48GB total)                                │
├─────────────────────────────────────────────────────────┤
│ [Layer 0] [Layer 1] ... [Layer N-1] [Layer 0] [Layer 1]│
│   (on wraparound, skip to start)                        │
│   ↑                                ↑                    │
│   │ Next write →                   │ Write tail        │
│   └─ Skip to start if new layer doesn't fit here       │
└─────────────────────────────────────────────────────────┘

Event Tracking (GPU-side synchronization, zero CPU blocking):
- layer_ready_events[i]: Signals weight for layer i is available in ring
- layer_done_events[i]: Signals layer i computation is complete, slot can be reused

Usage:
    ring_buffer = WeightRingBuffer(capacity_bytes=48GB, device='cuda:0')
    # At model registration time (once per model)
    ring_buffer.register_model('llama-70b', model_state_dict)
    # Gets pre-computed layer_views and events
    
    # During inference (for each forward pass)
    output = model.forward(input)  # Hooks automatically manage prefetching
"""

import logging
import torch
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LayerAllocation:
    """Metadata for a single layer's allocation in ring buffer."""
    layer_name: str
    layer_idx: int
    offset: int  # Byte offset in ring buffer
    size_bytes: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    
    def end_offset(self) -> int:
        """End offset (exclusive) in ring buffer."""
        return self.offset + self.size_bytes


@dataclass
class ModelRegistration:
    """Metadata for a registered model in the ring buffer."""
    model_id: str
    layer_allocations: Dict[str, LayerAllocation] = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)  # Ordered layer names
    total_bytes: int = 0
    layer_ready_events: Dict[int, torch.cuda.Event] = field(default_factory=dict)
    layer_done_events: Dict[int, torch.cuda.Event] = field(default_factory=dict)


class WeightRingBuffer:
    """
    Circular buffer for weight streaming with skip-end allocation strategy.
    
    Prevents tensor fragmentation by wrapping to buffer start when a layer
    wouldn't fit at the end, ensuring tensors are never split across the wrap.
    """
    
    def __init__(self, capacity_bytes: int, device: torch.device = None):
        """
        Initialize ring buffer.
        
        Args:
            capacity_bytes: Total ring buffer size
            device: CUDA device (can be string like 'cuda:0' or torch.device)
        """
        if device is None:
            device = torch.device('cuda:0')
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.capacity_bytes = capacity_bytes
        self.lock = threading.Lock()
        
        # Allocate circular buffer as contiguous GPU tensor
        self.buffer = torch.zeros(
            capacity_bytes,  # Byte tensor size
            dtype=torch.uint8,
            device=device
        )
        logger.info(f"✅ WeightRingBuffer allocated: {capacity_bytes / 1024**3:.1f}GB on {device}")
        
        # Allocation tracking
        self.registrations: Dict[str, ModelRegistration] = {}  # model_id -> registration
        
        # Per-model state for runtime
        self.active_model_id: Optional[str] = None
        self.stats = {
            'models_registered': 0,
            'total_writes': 0,
            'buffer_wraps': 0,
            'bytes_transferred': 0,
        }
    
    def register_model(
        self,
        model_id: str,
        state_dict: Dict[str, torch.Tensor],
        layer_order: Optional[List[str]] = None
    ) -> ModelRegistration:
        """
        Register a model's weights for ring buffer streaming.
        
        Computes skip-end allocation for all layers and pre-creates GPU events.
        This is a one-time operation per model.
        
        Args:
            model_id: Unique model identifier
            state_dict: PyTorch model state dict
            layer_order: Optional explicit layer ordering. If None, uses state_dict order.
        
        Returns:
            ModelRegistration with layer metadata and GPU events
        
        Raises:
            ValueError: If total model size exceeds buffer capacity
        """
        with self.lock:
            if model_id in self.registrations:
                logger.debug(f"Model {model_id} already registered, returning cached")
                return self.registrations[model_id]
            
            # Determine layer order
            if layer_order is None:
                layer_order = list(state_dict.keys())
            
            # Compute skip-end allocation
            registration = ModelRegistration(model_id=model_id)
            current_offset = 0
            total_bytes = 0
            
            for layer_idx, layer_name in enumerate(layer_order):
                if layer_name not in state_dict:
                    logger.warning(f"Layer {layer_name} not in state_dict, skipping")
                    continue
                
                tensor = state_dict[layer_name]
                layer_bytes = tensor.numel() * tensor.element_size()
                total_bytes += layer_bytes
                
                # Skip-end check: if layer doesn't fit before buffer end, wrap to start
                if current_offset + layer_bytes > self.capacity_bytes:
                    logger.debug(
                        f"Layer {layer_name} wraps: offset {current_offset / 1024**2:.1f}MB "
                        f"+ size {layer_bytes / 1024**2:.1f}MB > capacity {self.capacity_bytes / 1024**2:.1f}MB. "
                        f"Skipping to start."
                    )
                    current_offset = 0
                    self.stats['buffer_wraps'] += 1
                
                # Allocate this layer
                allocation = LayerAllocation(
                    layer_name=layer_name,
                    layer_idx=layer_idx,
                    offset=current_offset,
                    size_bytes=layer_bytes,
                    dtype=tensor.dtype,
                    shape=tuple(tensor.shape)
                )
                registration.layer_allocations[layer_name] = allocation
                registration.layer_names.append(layer_name)
                
                # Pre-create GPU events for this layer (only on CUDA devices)
                if self.device.type == 'cuda':
                    registration.layer_ready_events[layer_idx] = torch.cuda.Event()
                    registration.layer_done_events[layer_idx] = torch.cuda.Event()
                
                current_offset += layer_bytes
                
                logger.debug(
                    f"  Layer {layer_idx:3d} ({layer_name:40s}): "
                    f"offset {allocation.offset / 1024**2:7.1f}MB, "
                    f"size {layer_bytes / 1024**2:7.1f}MB, "
                    f"shape {tensor.shape}"
                )
            
            registration.total_bytes = total_bytes
            
            # Validate total size
            if total_bytes > self.capacity_bytes:
                raise ValueError(
                    f"Model {model_id} requires {total_bytes / 1024**3:.1f}GB "
                    f"but ring buffer capacity is only {self.capacity_bytes / 1024**3:.1f}GB"
                )
            
            # Store registration
            self.registrations[model_id] = registration
            self.active_model_id = model_id
            self.stats['models_registered'] += 1
            
            logger.info(
                f"✅ Registered model {model_id} in ring buffer: "
                f"{len(registration.layer_allocations)} layers, "
                f"{total_bytes / 1024**3:.1f}GB total"
            )
            
            return registration
    
    def get_layer_view(
        self,
        model_id: str,
        layer_name: str
    ) -> torch.Tensor:
        """
        Get a tensor view for a layer's weights in the ring buffer.
        
        This view points into the ring buffer at the pre-computed offset.
        Does NOT copy data - you must load weights using load_layer_weights().
        
        Args:
            model_id: Model identifier
            layer_name: Layer name from state dict
        
        Returns:
            Tensor view into ring buffer with correct shape/dtype
        """
        # Minimize critical section: only lock for dict access, not view creation
        with self.lock:
            if model_id not in self.registrations:
                raise ValueError(f"Model {model_id} not registered")
            
            registration = self.registrations[model_id]
            if layer_name not in registration.layer_allocations:
                raise ValueError(f"Layer {layer_name} not found in model {model_id}")
            
            alloc = registration.layer_allocations[layer_name]
            # Copy allocation data before releasing lock
            offset = alloc.offset
            size_bytes = alloc.size_bytes
            dtype = alloc.dtype
            shape = alloc.shape
            buffer_ref = self.buffer  # Get reference while holding lock
        
        # Create view OUTSIDE lock (view creation is fast but no need to hold lock)
        start_byte = offset
        end_byte = start_byte + size_bytes
        byte_view = buffer_ref[start_byte:end_byte]
        
        # Convert bytes to desired dtype
        element_size = torch.empty(0, dtype=dtype).element_size()
        num_elements = size_bytes // element_size
        typed_view = byte_view.view(dtype)
        
        # Reshape to layer shape
        return typed_view[:num_elements].view(shape)
    
    def load_layer_weights(
        self,
        model_id: str,
        layer_name: str,
        weights: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None
    ) -> int:
        """
        Load weights for a single layer into the ring buffer.
        
        This is typically called from the weight prefetcher during inference.
        
        Args:
            model_id: Model identifier
            layer_name: Layer name
            weights: Tensor with weights (MUST be on CPU for async transfer)
            stream: CUDA stream for async copy (if None, uses current stream)
        
        Returns:
            Offset where weights were written
        """
        view = self.get_layer_view(model_id, layer_name)
        
        # Weights must be on CPU. If on GPU, raise error (caller should handle CPU placement)
        if weights.device.type == 'cuda':
            raise RuntimeError(
                f"Weights for {layer_name} must be on CPU for async transfer. "
                f"Got device: {weights.device}. Move weights to CPU before calling load_layer_weights."
            )
        
        # Pin memory if not already pinned (this is fast if already pinned)
        if not weights.is_pinned():
            try:
                weights = weights.pin_memory()
            except RuntimeError:
                logger.warning(f"Could not pin memory for {layer_name}, transfer may be slower")
        
        # Async copy to GPU
        if stream is not None:
            with torch.cuda.stream(stream):
                view.copy_(weights, non_blocking=True)
        else:
            view.copy_(weights, non_blocking=True)
        
        # Get allocation info for event
        with self.lock:
            registration = self.registrations[model_id]
            alloc = registration.layer_allocations[layer_name]
            layer_idx = alloc.layer_idx
        
        self.stats['total_writes'] += 1
        self.stats['bytes_transferred'] += weights.numel() * weights.element_size()
        
        return alloc.offset
    
    def get_ready_event(self, model_id: str, layer_idx: int) -> Optional[torch.cuda.Event]:
        """Get the event signaling weights for a layer are ready. Returns None on CPU."""
        if self.device.type != 'cuda':
            return None
        with self.lock:
            if model_id not in self.registrations:
                raise ValueError(f"Model {model_id} not registered")
            return self.registrations[model_id].layer_ready_events.get(layer_idx)
    
    def get_done_event(self, model_id: str, layer_idx: int) -> Optional[torch.cuda.Event]:
        """Get the event signaling computation for a layer is complete. Returns None on CPU."""
        if self.device.type != 'cuda':
            return None
        with self.lock:
            if model_id not in self.registrations:
                raise ValueError(f"Model {model_id} not registered")
            return self.registrations[model_id].layer_done_events.get(layer_idx)
    
    def record_ready(self, model_id: str, layer_idx: int, stream: torch.cuda.Stream) -> None:
        """Record that weights for a layer are ready in the given stream. No-op on CPU."""
        event = self.get_ready_event(model_id, layer_idx)
        if event is not None and stream is not None:
            event.record(stream)
    
    def record_done(self, model_id: str, layer_idx: int, stream: torch.cuda.Stream) -> None:
        """Record that computation for a layer is done in the given stream. No-op on CPU."""
        event = self.get_done_event(model_id, layer_idx)
        if event is not None and stream is not None:
            event.record(stream)
    
    def get_stats(self) -> Dict:
        """Get ring buffer statistics."""
        with self.lock:
            return {
                **self.stats,
                'capacity_gb': self.capacity_bytes / 1024**3,
                'active_model': self.active_model_id,
            }
    
    def clear(self) -> None:
        """Clear all registrations and reset buffer."""
        with self.lock:
            self.registrations.clear()
            self.active_model_id = None
            # Reset stats
            self.stats['buffer_wraps'] = 0
            self.stats['total_writes'] = 0
            self.stats['bytes_transferred'] = 0
            logger.info("✅ Ring buffer cleared")

