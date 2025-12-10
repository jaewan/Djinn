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
from enum import Enum

logger = logging.getLogger(__name__)


class RingBufferConfig(Enum):
    """Configuration for ablation study of ring buffer mechanisms."""
    
    # Full optimization (default)
    FULL = {
        'enable_pipelining': True,      # Async dual-stream transfers
        'enable_skip_end': True,        # Skip-end allocation strategy
        'enable_fractional': True,      # Fractional residency optimization
    }
    
    # Ablation: No pipelining (serial transfers)
    NO_PIPELINING = {
        'enable_pipelining': False,     # Force synchronous transfers
        'enable_skip_end': True,
        'enable_fractional': True,
    }
    
    # Ablation: No skip-end (allow straddling)
    NO_SKIP_END = {
        'enable_pipelining': True,
        'enable_skip_end': False,       # Allow tensors to straddle wrap
        'enable_fractional': True,
    }
    
    # Ablation: No fractional residency (stream entire model)
    NO_FRACTIONAL = {
        'enable_pipelining': True,
        'enable_skip_end': True,
        'enable_fractional': False,     # Stream 100% of weights
    }
    
    # Ablation: No optimizations (baseline)
    BASELINE = {
        'enable_pipelining': False,
        'enable_skip_end': False,
        'enable_fractional': False,
    }
    
    def get_config(self) -> Dict[str, bool]:
        """Get configuration dictionary."""
        return self.value


@dataclass
class LayerAllocation:
    """Metadata for a single layer's allocation in ring buffer."""
    layer_name: str
    layer_idx: int
    offset: int  # Byte offset in ring buffer (or -1 for streamed layers)
    size_bytes: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    is_resident: bool = True  # True if permanently resident, False if streamed
    slot_id: Optional[int] = None  # For streamed layers: which slot to use at runtime
    
    def end_offset(self) -> int:
        """End offset (exclusive) in ring buffer."""
        if self.offset == -1:
            raise ValueError(f"Streamed layer {self.layer_name} has no fixed offset")
        return self.offset + self.size_bytes


@dataclass
class TransformerBlock:
    """
    A transformer layer as the atomic unit of streaming.
    
    Represents one transformer block (e.g., model.layers.0) containing
    all its parameters (attention, MLP, norms). This is the granularity
    at which we stream weights.
    """
    block_idx: int                    # e.g., 0 for model.layers.0
    block_name: str                   # e.g., "model.layers.0"
    param_names: List[str]            # All params in this block
    total_bytes: int                  # Combined size of all params
    is_resident: bool = True          # True if permanently in buffer
    
    # For streamed blocks:
    host_buffer: Optional[torch.Tensor] = None  # Concatenated pinned weights
    param_offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # param → (start, end) in host_buffer


@dataclass
class StreamingSlot:
    """Metadata for a reusable streaming slot in the ring buffer."""
    slot_id: int
    offset: int  # Fixed offset in ring buffer
    capacity_bytes: int
    current_block_idx: Optional[int] = None  # Which block currently occupies this slot
    ready_event: Optional[torch.cuda.Event] = None
    done_event: Optional[torch.cuda.Event] = None


@dataclass
class ModelRegistration:
    """Metadata for a registered model in the ring buffer."""
    model_id: str
    layer_allocations: Dict[str, LayerAllocation] = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)  # Ordered layer names (parameter names)
    total_bytes: int = 0
    
    # Block-level tracking (NEW: for layer-granularity streaming)
    blocks: List[TransformerBlock] = field(default_factory=list)  # Ordered list of transformer blocks
    block_to_slot: Dict[int, int] = field(default_factory=dict)  # block_idx -> slot_id for streamed blocks
    
    # Legacy param-level tracking (kept for backward compatibility during transition)
    resident_layers: set = field(default_factory=set)  # Layer names that are permanently resident
    streamed_layers: List[str] = field(default_factory=list)  # Layers that must be streamed (in exec order)
    resident_bytes: int = 0  # Total bytes of resident layers
    streamed_bytes: int = 0  # Total bytes of streamed layers
    
    # Module to parameters mapping (for hooks that operate on modules)
    # e.g., "model.layers.32.mlp" -> ["model.layers.32.mlp.gate_proj.weight", ...]
    module_to_params: Dict[str, List[str]] = field(default_factory=dict)
    
    # Streaming infrastructure
    streaming_slots: List[StreamingSlot] = field(default_factory=list)  # Pool of reusable slots
    layer_to_slot: Dict[str, int] = field(default_factory=dict)  # For streamed layers: layer_name -> slot_id
    host_weights: Dict[str, torch.Tensor] = field(default_factory=dict)  # Pinned CPU tensors for streamed layers
    
    # Event tracking
    layer_ready_events: Dict[int, torch.cuda.Event] = field(default_factory=dict)
    layer_done_events: Dict[int, torch.cuda.Event] = field(default_factory=dict)
    tied_weights: Dict[str, str] = field(default_factory=dict)  # alias -> canonical name


class WeightRingBuffer:
    """
    Circular buffer for weight streaming with skip-end allocation strategy.
    
    Prevents tensor fragmentation by wrapping to buffer start when a layer
    wouldn't fit at the end, ensuring tensors are never split across the wrap.
    """
    
    def __init__(
        self,
        capacity_bytes: int,
        device: torch.device = None,
        config: RingBufferConfig = None
    ):
        """
        Initialize ring buffer.
        
        Args:
            capacity_bytes: Total ring buffer size
            device: CUDA device (can be string like 'cuda:0' or torch.device)
            config: RingBufferConfig for ablation studies (default: FULL)
        """
        if device is None:
            device = torch.device('cuda:0')
        elif isinstance(device, str):
            device = torch.device(device)
        
        if config is None:
            config = RingBufferConfig.FULL
        
        self.device = device
        self.capacity_bytes = capacity_bytes
        self.lock = threading.Lock()
        
        # Configuration for this instance (for ablation studies)
        if isinstance(config, RingBufferConfig):
            self.config = config.get_config()
            self.config_name = config.name
        else:
            self.config = config
            self.config_name = "CUSTOM"
        
        logger.info(f"🔧 Ring buffer config: {self.config_name}")
        for key, value in self.config.items():
            logger.info(f"   {key}: {value}")
        
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
        
        Computes allocation for layers and handles fractional residency when
        model size exceeds buffer capacity.
        
        Args:
            model_id: Unique model identifier
            state_dict: PyTorch model state dict
            layer_order: Optional explicit layer ordering. If None, uses state_dict order.
        
        Returns:
            ModelRegistration with layer metadata and GPU events
        
        Raises:
            ValueError: If model too large and fractional residency disabled
        """
        with self.lock:
            if model_id in self.registrations:
                logger.debug(f"Model {model_id} already registered, returning cached")
                return self.registrations[model_id]
            
            # Determine layer order
            if layer_order is None:
                layer_order = list(state_dict.keys())
            
            # Compute total model size
            total_bytes = sum(
                state_dict[name].numel() * state_dict[name].element_size()
                for name in layer_order if name in state_dict
            )
            
            # Check if model fits entirely in buffer
            is_oversubscribed = total_bytes > self.capacity_bytes
            
            if not is_oversubscribed:
                # Simple case: everything fits
                return self._register_fully_resident(model_id, state_dict, layer_order, total_bytes)
            else:
                # Oversubscribed case: fractional residency
                config_dict = self.config.get_config() if isinstance(self.config, RingBufferConfig) else self.config
                enable_fractional = config_dict.get('enable_fractional', True)
                
                if not enable_fractional:
                    raise ValueError(
                        f"Model {model_id} requires {total_bytes / 1024**3:.1f}GB "
                        f"but ring buffer capacity is only {self.capacity_bytes / 1024**3:.1f}GB. "
                        f"Enable fractional residency to allow partial model loading."
                    )
                
                return self._register_with_fractional_residency(
                    model_id, state_dict, layer_order, total_bytes
                )
    
    def _partition_into_blocks(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer_order: List[str]
    ) -> List[TransformerBlock]:
        """
        Partition model parameters into transformer blocks.
        
        A transformer block is the atomic unit of streaming. For LLaMA/Mistral:
        - model.layers.N.* → one block
        - model.embed_tokens.* → one block
        - model.norm.* → one block
        - lm_head.* → one block
        
        Args:
            state_dict: Model state dict
            layer_order: Ordered list of parameter names
            
        Returns:
            List of TransformerBlock objects in execution order
        """
        from collections import defaultdict
        
        # Group parameters by block prefix
        block_params = defaultdict(list)
        
        for param_name in layer_order:
            if param_name not in state_dict:
                continue
            
            # Detect block boundary
            # Pattern: model.layers.N.* → block "model.layers.N"
            parts = param_name.split('.')
            
            if 'layers' in parts:
                # Find the layer number
                try:
                    layer_idx = parts.index('layers')
                    if layer_idx + 1 < len(parts):
                        # Block name is up to and including the layer number
                        block_name = '.'.join(parts[:layer_idx + 2])
                    else:
                        # Fallback: treat as single-param block
                        block_name = param_name
                except (ValueError, IndexError):
                    # Fallback: treat as single-param block
                    block_name = param_name
            else:
                # Non-layer params (embeddings, norm, lm_head) are single-param blocks
                block_name = '.'.join(parts[:-1]) if len(parts) > 1 else param_name
            
            block_params[block_name].append(param_name)
        
        # Convert to TransformerBlock objects
        blocks = []
        for block_idx, (block_name, param_names) in enumerate(sorted(block_params.items())):
            # Calculate total size
            total_bytes = sum(
                state_dict[name].numel() * state_dict[name].element_size()
                for name in param_names if name in state_dict
            )
            
            block = TransformerBlock(
                block_idx=block_idx,
                block_name=block_name,
                param_names=param_names,
                total_bytes=total_bytes,
                is_resident=True,  # Will be updated during registration
                host_buffer=None,
                param_offsets={}
            )
            blocks.append(block)
        
        logger.info(f"Partitioned model into {len(blocks)} transformer blocks")
        for i, block in enumerate(blocks[:5]):  # Show first 5
            logger.debug(
                f"  Block {i}: {block.block_name} "
                f"({len(block.param_names)} params, {block.total_bytes / 1024**2:.1f}MB)"
            )
        if len(blocks) > 5:
            logger.debug(f"  ... and {len(blocks) - 5} more blocks")
        
        return blocks
    
    def _register_fully_resident(
        self,
        model_id: str,
        state_dict: Dict[str, torch.Tensor],
        layer_order: List[str],
        total_bytes: int
    ) -> ModelRegistration:
        """
        Register a model that fits entirely in the ring buffer.
        
        All layers are allocated as resident with fixed offsets.
        """
        registration = ModelRegistration(model_id=model_id)
        current_offset = 0
        
        for layer_idx, layer_name in enumerate(layer_order):
            if layer_name not in state_dict:
                logger.warning(f"Layer {layer_name} not in state_dict, skipping")
                continue
            
            tensor = state_dict[layer_name]
            layer_bytes = tensor.numel() * tensor.element_size()
            
            # Skip-end check: if layer doesn't fit before buffer end, wrap to start
            if self.config.get('enable_skip_end', True) and current_offset + layer_bytes > self.capacity_bytes:
                logger.debug(
                    f"Layer {layer_name} wraps: offset {current_offset / 1024**2:.1f}MB "
                    f"+ size {layer_bytes / 1024**2:.1f}MB > capacity. Skipping to start."
                )
                current_offset = 0
                self.stats['buffer_wraps'] += 1
            
            # Allocate this layer as resident
            allocation = LayerAllocation(
                layer_name=layer_name,
                layer_idx=layer_idx,
                offset=current_offset,
                size_bytes=layer_bytes,
                dtype=tensor.dtype,
                shape=tuple(tensor.shape),
                is_resident=True,
                slot_id=None
            )
            registration.layer_allocations[layer_name] = allocation
            registration.layer_names.append(layer_name)
            registration.resident_layers.add(layer_name)
            
            # Pre-create GPU events for this layer (only on CUDA devices)
            if self.device.type == 'cuda':
                registration.layer_ready_events[layer_idx] = torch.cuda.Event()
                registration.layer_done_events[layer_idx] = torch.cuda.Event()
            
            current_offset += layer_bytes
        
        registration.total_bytes = total_bytes
        registration.resident_bytes = total_bytes
        registration.streamed_bytes = 0
        
        # Copy all weights to ring buffer
        logger.info(f"Copying weights to ring buffer...")
        weights_copied = 0
        for layer_name in registration.resident_layers:
            allocation = registration.layer_allocations[layer_name]
            tensor = state_dict[layer_name]
            
            ring_view = self.buffer[
                allocation.offset : allocation.end_offset()
            ].view(allocation.dtype).view(allocation.shape)
            ring_view.copy_(tensor)
            weights_copied += 1
        
        logger.info(f"✅ Copied {weights_copied} weight tensors to ring buffer")
        
        # Build module→params mapping for hooks
        self._build_module_to_params_mapping(registration)
        
        # Store registration
        self.registrations[model_id] = registration
        self.active_model_id = model_id
        self.stats['models_registered'] += 1
        
        logger.info(
            f"✅ Registered model {model_id} in ring buffer: "
            f"{len(registration.layer_allocations)} layers, "
            f"{total_bytes / 1024**3:.1f}GB total (fully resident)"
        )
        
        return registration
    
    def _register_with_fractional_residency(
        self,
        model_id: str,
        state_dict: Dict[str, torch.Tensor],
        layer_order: List[str],
        total_bytes: int
    ) -> ModelRegistration:
        """
        Register a model with fractional residency (model > buffer capacity).
        
        Uses BLOCK-GRANULARITY streaming: entire transformer blocks are the
        atomic unit, not individual parameters.
        
        Partitions blocks into:
        - Resident blocks: permanently in buffer
        - Streamed blocks: kept on CPU, streamed to reusable slots on demand
        """
        registration = ModelRegistration(model_id=model_id)
        
        logger.info(
            f"⚠️  Model {model_id} ({total_bytes / 1024**3:.1f}GB) exceeds "
            f"ring buffer capacity ({self.capacity_bytes / 1024**3:.1f}GB). "
            f"Using block-granularity fractional residency..."
        )
        
        # Phase 1: Partition model into transformer blocks
        blocks = self._partition_into_blocks(state_dict, layer_order)
        registration.blocks = blocks
        
        # Reserve space for 3 streaming slots (enough for 2-ahead pipelining)
        # Estimate slot size as max block size * 1.1
        max_block_size = max(block.total_bytes for block in blocks)
        slot_capacity = int(max_block_size * 1.1)
        num_slots = 3
        streaming_reserve = slot_capacity * num_slots
        
        resident_budget = self.capacity_bytes - streaming_reserve
        current_offset = 0
        resident_bytes = 0
        
        logger.info(
            f"  Ring buffer allocation:"
        )
        logger.info(
            f"    Resident budget: {resident_budget / 1024**3:.1f}GB"
        )
        logger.info(
            f"    Streaming reserve: {streaming_reserve / 1024**3:.1f}GB ({num_slots} slots × {slot_capacity / 1024**2:.1f}MB)"
        )
        
        # Phase 2: Partition blocks into resident vs streamed
        for block in blocks:
            # Decide if this entire block fits in resident budget
            if resident_bytes + block.total_bytes <= resident_budget:
                # This block is resident
                block.is_resident = True
                
                # Allocate all params in this block
                for param_name in block.param_names:
                    if param_name not in state_dict:
                        continue
                    
                    tensor = state_dict[param_name]
                    param_bytes = tensor.numel() * tensor.element_size()
                    
                    allocation = LayerAllocation(
                        layer_name=param_name,
                        layer_idx=len(registration.layer_names),
                        offset=current_offset,
                        size_bytes=param_bytes,
                        dtype=tensor.dtype,
                        shape=tuple(tensor.shape),
                        is_resident=True,
                        slot_id=None
                    )
                    registration.layer_allocations[param_name] = allocation
                    registration.layer_names.append(param_name)
                    registration.resident_layers.add(param_name)
                    current_offset += param_bytes
                
                resident_bytes += block.total_bytes
            else:
                # This block must be streamed
                block.is_resident = False
                
                # Concatenate all params in this block into one pinned buffer
                block_tensors = []
                param_offset = 0
                block.param_offsets = {}
                
                for param_name in block.param_names:
                    if param_name not in state_dict:
                        continue
                    
                    tensor = state_dict[param_name]
                    param_bytes = tensor.numel() * tensor.element_size()
                    
                    # Track param location within block buffer
                    block.param_offsets[param_name] = (param_offset, param_offset + param_bytes)
                    param_offset += param_bytes
                    
                    # Flatten tensor for concatenation
                    if tensor.is_cuda:
                        tensor = tensor.cpu()
                    block_tensors.append(tensor.flatten().contiguous())
                    
                    # Create allocation (no fixed offset, will use slot)
                    allocation = LayerAllocation(
                        layer_name=param_name,
                        layer_idx=len(registration.layer_names),
                        offset=-1,  # No fixed offset
                        size_bytes=param_bytes,
                        dtype=tensor.dtype,
                        shape=tuple(tensor.shape),
                        is_resident=False,
                        slot_id=None
                    )
                    registration.layer_allocations[param_name] = allocation
                    registration.layer_names.append(param_name)
                    registration.streamed_layers.append(param_name)
                
                # Concatenate all params into one buffer and pin
                if block_tensors:
                    # Convert to bytes for concatenation
                    byte_tensors = [t.view(torch.uint8) for t in block_tensors]
                    concatenated = torch.cat(byte_tensors, dim=0)
                    block.host_buffer = concatenated.pin_memory()
            
            # Pre-create GPU events for this block
            if self.device.type == 'cuda':
                registration.layer_ready_events[block.block_idx] = torch.cuda.Event()
                registration.layer_done_events[block.block_idx] = torch.cuda.Event()
        
        registration.total_bytes = total_bytes
        registration.resident_bytes = resident_bytes
        registration.streamed_bytes = total_bytes - resident_bytes
        
        # Count resident and streamed blocks
        resident_blocks = [b for b in blocks if b.is_resident]
        streamed_blocks = [b for b in blocks if not b.is_resident]
        
        # Phase 3: Allocate streaming slot pool (3 slots for block-granularity)
        if streamed_blocks:
            for slot_id in range(num_slots):
                if current_offset + slot_capacity > self.capacity_bytes:
                    logger.warning(
                        f"Not enough space for all streaming slots. "
                        f"Allocated {slot_id} of {num_slots} slots."
                    )
                    break
                
                slot = StreamingSlot(
                    slot_id=slot_id,
                    offset=current_offset,
                    capacity_bytes=slot_capacity,
                    current_block_idx=None,
                    ready_event=torch.cuda.Event() if self.device.type == 'cuda' else None,
                    done_event=torch.cuda.Event() if self.device.type == 'cuda' else None
                )
                registration.streaming_slots.append(slot)
                current_offset += slot_capacity
            
            logger.info(
                f"  Allocated {len(registration.streaming_slots)} streaming slots "
                f"({slot_capacity / 1024**2:.1f}MB each)"
            )
        
        # Phase 4: Copy resident weights to ring buffer
        logger.info(f"Copying resident weights to ring buffer...")
        weights_copied = 0
        for layer_name in registration.resident_layers:
            allocation = registration.layer_allocations[layer_name]
            tensor = state_dict[layer_name]
            
            ring_view = self.buffer[
                allocation.offset : allocation.end_offset()
            ].view(allocation.dtype).view(allocation.shape)
            ring_view.copy_(tensor)
            weights_copied += 1
        
        logger.info(f"✅ Copied {weights_copied} resident weight tensors to ring buffer")
        
        # Phase 5: Build module→params mapping for hooks (legacy compatibility)
        self._build_module_to_params_mapping(registration)
        
        # Store registration
        self.registrations[model_id] = registration
        self.active_model_id = model_id
        self.stats['models_registered'] += 1
        
        logger.info(
            f"✅ Registered model {model_id} with block-granularity fractional residency:"
        )
        logger.info(
            f"   Total blocks: {len(blocks)}"
        )
        logger.info(
            f"   Resident blocks: {len(resident_blocks)} "
            f"({resident_bytes / 1024**3:.1f}GB, {resident_bytes / total_bytes * 100:.1f}%)"
        )
        logger.info(
            f"   Streamed blocks: {len(streamed_blocks)} "
            f"({registration.streamed_bytes / 1024**3:.1f}GB, {registration.streamed_bytes / total_bytes * 100:.1f}%)"
        )
        logger.info(
            f"   Streaming slots: {len(registration.streaming_slots)} × {slot_capacity / 1024**2:.1f}MB"
        )
        
        return registration
    
    def _build_module_to_params_mapping(self, registration: ModelRegistration) -> None:
        """
        Build mapping from module names to parameter names.
        
        Hooks operate on modules (e.g., "model.layers.32.mlp"),
        but allocations are keyed by parameter names (e.g., "model.layers.32.mlp.gate_proj.weight").
        This mapping allows hooks to find all parameters belonging to a module and its children.
        
        We build mappings at ALL hierarchy levels, so:
        - "model.layers.32.mlp.gate_proj" → [gate_proj.weight, gate_proj.bias]
        - "model.layers.32.mlp" → [gate_proj.weight, up_proj.weight, down_proj.weight, ...]
        - "model.layers.32" → [all params in layer 32]
        """
        from collections import defaultdict
        module_to_params = defaultdict(list)
        
        for param_name in registration.layer_names:
            # Add mapping at all hierarchy levels
            # e.g., "model.layers.32.mlp.gate_proj.weight" maps to:
            # - "model.layers.32.mlp.gate_proj"
            # - "model.layers.32.mlp"
            # - "model.layers.32"
            # - "model.layers"
            # - "model"
            parts = param_name.split('.')
            
            # Build all prefixes (excluding the full param name and empty prefix)
            for i in range(len(parts) - 1, 0, -1):
                module_name = '.'.join(parts[:i])
                module_to_params[module_name].append(param_name)
        
        registration.module_to_params = dict(module_to_params)
    
    def allocate_view(
        self,
        param_name: str,
        shape: Tuple,
        dtype: torch.dtype,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Allocate a contiguous view in the ring buffer for a parameter.
        
        Used for GPU-resident model loading where parameters are ring buffer views
        from initialization, not redirected afterwards.
        
        Args:
            param_name: Parameter name (for tracking)
            shape: Parameter shape
            dtype: Parameter dtype
            device: Target device (defaults to ring buffer's device)
            
        Returns:
            Tensor view that can be used as a model parameter
        """
        if device is None:
            device = self.device
        
        # Calculate required bytes
        import math
        num_elements = math.prod(shape)
        element_size = torch.empty(0, dtype=dtype).element_size()
        bytes_needed = num_elements * element_size
        
        # Allocate in ring buffer (simple linear allocation for now)
        with self.lock:
            if not hasattr(self, '_allocation_offset'):
                self._allocation_offset = 0
            
            offset = self._allocation_offset
            
            # Check if fits
            if offset + bytes_needed > self.capacity_bytes:
                raise RuntimeError(
                    f"Ring buffer full: Cannot allocate {param_name} ({bytes_needed / 1024**3:.2f}GB). "
                    f"Capacity: {self.capacity_bytes / 1024**3:.1f}GB, "
                    f"Available: {(self.capacity_bytes - offset) / 1024**3:.2f}GB"
                )
            
            # Create view
            byte_view = self.buffer[offset:offset + bytes_needed]
            typed_view = byte_view.view(dtype)
            param_view = typed_view[:num_elements].view(shape)
            
            # Track allocation
            self._allocation_offset = offset + bytes_needed
            
            logger.debug(
                f"Allocated {param_name}: {shape} {dtype} at offset {offset / 1024**2:.1f}MB "
                f"(total: {self._allocation_offset / 1024**3:.2f}GB / {self.capacity_bytes / 1024**3:.1f}GB)"
            )
            
            return param_view
    
    def set_tied_weights(self, model_id: str, tied_weights: Dict[str, str]) -> None:
        """
        Register tied weight mappings for a model.
        
        Tied weights occur when multiple parameters share the same underlying tensor
        (e.g., lm_head.weight = transformer.wte.weight in GPT-2).
        
        Args:
            model_id: Model identifier
            tied_weights: Dict mapping alias parameter names to canonical names
        """
        with self.lock:
            if model_id not in self.registrations:
                raise ValueError(f"Model {model_id} not registered")
            
            registration = self.registrations[model_id]
            registration.tied_weights = tied_weights
            
            logger.debug(f"Registered {len(tied_weights)} tied weight(s) for model {model_id}")
    
    def get_layer_view(
        self,
        model_id: str,
        layer_name: str
    ) -> torch.Tensor:
        """
        Get a tensor view for a layer's weights in the ring buffer.
        
        This view points into the ring buffer at the pre-computed offset.
        Does NOT copy data - you must load weights using load_layer_weights().
        
        Handles tied weights by resolving aliases to canonical parameter names.
        
        Args:
            model_id: Model identifier
            layer_name: Layer name from state dict (may be an alias for tied weights)
        
        Returns:
            Tensor view into ring buffer with correct shape/dtype
        """
        # Minimize critical section: only lock for dict access, not view creation
        with self.lock:
            if model_id not in self.registrations:
                raise ValueError(f"Model {model_id} not registered")
            
            registration = self.registrations[model_id]
            
            # Resolve aliases for tied weights
            canonical_layer_name = registration.tied_weights.get(layer_name, layer_name)
            
            if canonical_layer_name not in registration.layer_allocations:
                raise ValueError(f"Layer {layer_name} (canonical: {canonical_layer_name}) not found in model {model_id}")
            
            alloc = registration.layer_allocations[canonical_layer_name]
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

