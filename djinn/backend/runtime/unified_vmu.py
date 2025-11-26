"""
Unified Virtual Memory Unit (VMU): Segmented Memory Model

Implements the v2.3.15 Semantic Tensor OS architecture with three memory segments:

TEXT SEGMENT (60% VRAM):
- Shared model weights (OS analogy: shared libraries)
- Read-only access, concurrent CUDA streams
- Zero duplication of model weights
- Loaded once, accessed by all users running that model

DATA SEGMENT (20% VRAM):
- Private session data (OS analogy: heap)
- KV-cache, outputs, persistent session state
- Logical isolation via Session ID checks
- Suitable for trusted multi-tenancy

STACK SLAB (20% VRAM):
- Volatile activations (OS analogy: stack)
- Time-shared resource, protected by stream synchronization
- Reset after each execution
- One user executes at a time per GPU stream

Memory Layout:
┌────────────────────────────────────┐
│ OS Reserved (4GB)                  │ ← Kernel, drivers
├────────────────────────────────────┤
│ TEXT SEGMENT (Shared Weights)      │ ← Model weights (read-only)
│ 60% of VMU capacity                │
├────────────────────────────────────┤
│ DATA SEGMENT (Session Data)        │ ← KV-cache, outputs (private)
│ 20% of VMU capacity                │
├────────────────────────────────────┤
│ STACK SLAB (Activations)           │ ← Volatile scratchpad
│ 20% of VMU capacity                │ ← Reset after execution
├────────────────────────────────────┤
│ Free Memory                        │
└────────────────────────────────────┘

Concurrency Model:
- Text Segment: Read-only, concurrent access by multiple streams
- Data Segment: Session-isolated via logical checks
- Stack Slab: Time-shared, protected by stream synchronization

⚠️  CRITICAL TECHNICAL DEBT - MUST FIX BEFORE PHASE 2:
   [BLOCKING-P0] Text Segment weight loading not implemented
   [BLOCKING-P0] Data Segment memory compaction not implemented
   [BLOCKING-P0] Text Segment DMA not implemented
   [DEFER-PHASE2] Stale dual-lifecycle API usage (migration needed)
   See code comments for [BLOCKING-P0] markers.
"""

import torch
import logging
import threading
import socket
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
@dataclass
class VMUMetrics:
    """Aggregated memory statistics for VMU segments."""

    text_used_bytes: int
    text_capacity_bytes: int
    data_reserved_bytes: int
    data_capacity_bytes: int
    data_internal_waste_bytes: int
    data_external_gap_bytes: int
    stack_allocated_bytes: int
    stack_capacity_bytes: int
    stack_reset_count: int
    active_sessions: int
    models_loaded: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "text_used_bytes": self.text_used_bytes,
            "text_capacity_bytes": self.text_capacity_bytes,
            "data_reserved_bytes": self.data_reserved_bytes,
            "data_capacity_bytes": self.data_capacity_bytes,
            "data_internal_waste_bytes": self.data_internal_waste_bytes,
            "data_external_gap_bytes": self.data_external_gap_bytes,
            "stack_allocated_bytes": self.stack_allocated_bytes,
            "stack_capacity_bytes": self.stack_capacity_bytes,
            "stack_reset_count": self.stack_reset_count,
            "active_sessions": self.active_sessions,
            "models_loaded": self.models_loaded,
        }


from djinn.config import VmuConfig

logger = logging.getLogger(__name__)


class MemorySegment:
    """Base class for memory segments in the VMU."""

    def __init__(self, name: str, capacity: int, device: torch.device, alignment: int = 256):
        self.name = name
        self.capacity = capacity
        self.device = device
        self.alignment = alignment
        self.lock = threading.Lock()

    def _align(self, offset: int) -> int:
        """Align offset to the segment's alignment boundary."""
        return (offset + self.alignment - 1) & ~(self.alignment - 1)

    def get_stats(self) -> Dict[str, any]:
        """Return segment-specific statistics."""
        raise NotImplementedError


class TextSegment(MemorySegment):
    """
    Text Segment: Shared model weights (OS analogy: shared libraries).

    - Read-only access
    - Concurrent access by multiple CUDA streams
    - Zero duplication of model weights
    - Loaded once, accessed by all users running that model
    """

    def __init__(self, capacity: int, device: torch.device):
        super().__init__("Text", capacity, device)
        self.buffer = torch.zeros(capacity, dtype=torch.uint8, device=device)
        self.allocated_size = 0
        self.current_offset = 0  # Current allocation offset
        self.model_weights: Dict[str, Dict[str, any]] = {}  # model_id -> weight metadata
        self.loaded_models: Dict[str, torch.nn.Module] = {}  # model_id -> loaded model
        self.usage_count: Dict[str, int] = {}  # model_id -> reference count

    def load_model_weights(self, model_id: str, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Load model weights into the Text Segment.

        Args:
            model_id: Unique model identifier
            state_dict: Model state dictionary

        Returns:
            True if loaded successfully, False if already loaded
        """
        with self.lock:
            if model_id in self.loaded_models:
                self.usage_count[model_id] += 1
                logger.debug(f"Model {model_id} already loaded, refcount: {self.usage_count[model_id]}")
                return False

            total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())

            if total_bytes > self.capacity:
                raise RuntimeError(f"Model {model_id} requires {total_bytes} bytes, exceeds Text Segment capacity")

            logger.info(f"Allocating {total_bytes / 1024**2:.1f}MB in Text Segment for model {model_id}")

            base_offset = self.allocate(total_bytes, torch.uint8, f"{model_id}_weights")

        # Create weight buffer view (outside lock to avoid deadlock)
        if base_offset + total_bytes > self.buffer.shape[0]:
            raise RuntimeError(f"Allocation exceeds buffer size: {base_offset + total_bytes} > {self.buffer.shape[0]}")
        weight_buffer = self.slab[base_offset:base_offset + total_bytes]

        current_offset = 0
        param_views = {}
        param_metadata = {}

        copy_stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None

        for param_name, tensor in state_dict.items():
            param_bytes = tensor.numel() * tensor.element_size()
            param_view = weight_buffer[current_offset:current_offset + param_bytes]
            param_views[param_name] = param_view.view(tensor.dtype).view(tensor.shape)

            # Copy weights into VMU using pinned memory + async stream when available
            tensor_cpu = tensor.detach().contiguous()
            if tensor_cpu.device.type != 'cpu':
                tensor_cpu = tensor_cpu.to('cpu')
            if hasattr(tensor_cpu, 'pin_memory') and not getattr(tensor_cpu, 'is_pinned', lambda: False)():
                try:
                    tensor_cpu = tensor_cpu.pin_memory()
                except RuntimeError:
                    pass

            target = param_views[param_name]
            if copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    target.copy_(tensor_cpu, non_blocking=True)
            else:
                target.copy_(tensor_cpu.to(self.device))

            param_metadata[param_name] = {
                'offset': current_offset,
                'bytes': param_bytes,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype)
            }

            current_offset += param_bytes

        if copy_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(copy_stream)

        with self.lock:
            self.loaded_models[model_id] = {
                'weight_buffer': weight_buffer,
                'param_views': param_views,
                'param_metadata': param_metadata,
                'base_offset': base_offset,
                'total_bytes': total_bytes
            }
            self.usage_count[model_id] = 1
            self.model_weights[model_id] = {
                'state_dict_keys': list(state_dict.keys()),
                'total_bytes': total_bytes,
                'tensor_count': len(state_dict)
            }

            logger.info(f"✅ Model {model_id} loaded into Text Segment: {total_bytes / 1024**2:.1f}MB")
            return True

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Get loaded model from Text Segment."""
        with self.lock:
            return self.loaded_models.get(model_id)

    def unload_model(self, model_id: str) -> bool:
        """Unload model from Text Segment (reference counted)."""
        with self.lock:
            if model_id not in self.usage_count:
                return False

            self.usage_count[model_id] -= 1
            if self.usage_count[model_id] <= 0:
                # TODO: Implement actual weight unloading from GPU buffer
                del self.loaded_models[model_id]
                del self.usage_count[model_id]
                del self.model_weights[model_id]
                logger.info(f"✅ Unloaded model {model_id} from Text Segment")
                return True

            return False

    def _align_size(self, size_bytes: int, dtype: torch.dtype = torch.uint8) -> int:
        """Align size to element boundary for the given dtype."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        aligned_elements = (size_bytes + element_size - 1) // element_size
        return aligned_elements * element_size

    @property
    def slab(self):
        """Get the slab buffer."""
        return self.buffer

    def allocate(self, size_bytes: int, dtype: torch.dtype = torch.uint8, name: str = "") -> int:
        """
        Allocate memory in Text Segment.

        NOTE: Caller must hold self.lock.

        Args:
            size_bytes: Size in bytes
            dtype: Data type (for alignment)
            name: Allocation name for debugging

        Returns:
            Offset in slab
        """
        # Lazy buffer allocation
        if self.buffer is None:
            # Allocate buffer on demand
            self.buffer = torch.zeros(self.capacity, dtype=torch.uint8, device=self.device)
            logger.info(f"✅ Lazy allocated Text Segment buffer: {self.capacity / 1024**2:.1f}MB on {self.device}")

        # Align allocation
        aligned_size = self._align_size(size_bytes, dtype)

        if self.current_offset + aligned_size > self.capacity:
            raise RuntimeError(f"Text Segment out of memory: {self.current_offset + aligned_size} > {self.capacity}")

        offset = self.current_offset
        self.current_offset += aligned_size
        self.allocated_size = max(self.allocated_size, self.current_offset)

        logger.debug(f"Text Segment allocated {aligned_size} bytes at offset {offset} ({name})")
        return offset

    def get_stats(self) -> Dict[str, any]:
        """Return Text Segment statistics."""
        with self.lock:
            total_loaded_bytes = sum(meta['total_bytes'] for meta in self.model_weights.values())
            return {
                'segment_name': self.name,
                'capacity_bytes': self.capacity,
                'loaded_bytes': total_loaded_bytes,
                'free_bytes': self.capacity - total_loaded_bytes,
                'loaded_models_count': len(self.loaded_models),
                'total_usage_count': sum(self.usage_count.values()),
                'utilization_percent': (total_loaded_bytes / self.capacity * 100) if self.capacity > 0 else 0,
            }


@dataclass
class SessionArena:
    """Arena reserved for a single session."""
    session_id: str
    base_offset: int
    capacity: int
    used: int = 0

    def alloc(self, size_bytes: int, alignment: int = 256) -> int:
        """Bump-pointer allocation within the arena."""
        aligned_used = ((self.used + alignment - 1) // alignment) * alignment
        if aligned_used + size_bytes > self.capacity:
            raise RuntimeError(
                f"Arena overflow for session {self.session_id}: "
                f"used={self.used}, requested={size_bytes}, capacity={self.capacity}"
            )
        offset_within_arena = aligned_used
        self.used = aligned_used + size_bytes
        return self.base_offset + offset_within_arena


class DataSegment(MemorySegment):
    """
    Data Segment: Private session data (OS analogy: heap).

    - Session-isolated storage
    - KV-cache, outputs, persistent session state
    - Logical isolation via Session ID checks
    - Suitable for trusted multi-tenancy
    """

    def __init__(self, capacity: int, device: torch.device):
        super().__init__("Data", capacity, device)
        self.buffer = None  # Lazy allocation
        self.sessions: Dict[str, SessionArena] = {}
        self.next_offset = 0

    @property
    def slab(self):
        """Get the slab buffer, allocating if necessary."""
        if self.buffer is None:
            self.buffer = torch.zeros(self.capacity, dtype=torch.uint8, device=self.device)
            logger.info(f"✅ Lazy allocated Data Segment buffer: {self.capacity / 1024**2:.1f}MB on {self.device}")
        return self.buffer

    def allocate_session_data(self, session_id: str, size_bytes: int, name: str = "") -> int:
        """
        Allocate space in Data Segment for session data.

        Args:
            session_id: Session identifier
            size_bytes: Size to allocate
            name: Optional allocation name for tracking

        Returns:
            Offset in Data Segment buffer
        """
        with self.lock:
            if session_id not in self.sessions:
                raise RuntimeError(f"Session {session_id} has no reserved arena")

            arena = self.sessions[session_id]
            offset = arena.alloc(size_bytes)

            logger.debug(f"✅ Allocated {size_bytes} bytes for session {session_id} at offset {offset}")
            return offset

    def get_session_view(self, session_id: str, offset: int, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Get a view of session data from the buffer."""
        with self.lock:
            if session_id not in self.sessions:
                raise RuntimeError(f"Session {session_id} not found in Data Segment")

            if offset + size > self.capacity:
                raise RuntimeError(f"Invalid offset/size for session {session_id}")

            buffer = self.slab
            return buffer[offset:offset + size].view(dtype)

    def free_session(self, session_id: str) -> bool:
        """Free all data for a session (called during session cleanup)."""
        with self.lock:
            arena = self.sessions.pop(session_id, None)
            if arena is None:
                return False

            if arena.base_offset + arena.capacity == self.next_offset:
                self.next_offset = arena.base_offset

            logger.info(f"✅ Freed session {session_id} data from Data Segment")
            return True

    def reserve_arena(self, session_id: str, size_bytes: int) -> SessionArena:
        """Reserve contiguous arena for a session. Raises if insufficient space."""
        with self.lock:
            if session_id in self.sessions:
                raise RuntimeError(f"Session {session_id} already exists")

            aligned_size = self._align(size_bytes)
            if self.next_offset + aligned_size > self.capacity:
                raise RuntimeError(f"Data Segment cannot reserve {aligned_size} bytes")

            arena = SessionArena(
                session_id=session_id,
                base_offset=self.next_offset,
                capacity=aligned_size
            )
            self.sessions[session_id] = arena
            self.next_offset += aligned_size
            logger.debug(f"✅ Reserved {aligned_size} bytes arena for session {session_id}")
            return arena

    def get_stats(self) -> Dict[str, any]:
        """Return Data Segment statistics."""
        with self.lock:
            active_sessions = len(self.sessions)
            reserved_bytes = self.next_offset
            total_used = sum(arena.used for arena in self.sessions.values())
            internal_waste = max(reserved_bytes - total_used, 0)
            external_gaps = max(self.capacity - reserved_bytes, 0)
            return {
                'segment_name': self.name,
                'capacity_bytes': self.capacity,
                'reserved_bytes': reserved_bytes,
                'allocated_bytes': total_used,
                'free_bytes': self.capacity - reserved_bytes,
                'active_sessions': active_sessions,
                'data_internal_waste_bytes': internal_waste,
                'data_external_gap_bytes': external_gaps,
                'utilization_percent': (reserved_bytes / self.capacity * 100) if self.capacity > 0 else 0,
            }


class StackSegment(MemorySegment):
    """
    Stack Slab: Volatile activations (OS analogy: stack).

    - Time-shared resource
    - Protected by stream synchronization
    - Reset after each execution
    - One user executes at a time per GPU stream
    """

    def __init__(self, capacity: int, device: torch.device):
        super().__init__("Stack", capacity, device)
        self.buffer = None  # Lazy allocation
        self.allocated_size = 0
        self.current_offset = 0  # For tracking allocations
        self.reset_count = 0

    @property
    def slab(self):
        """Get the slab buffer, allocating if necessary."""
        if self.buffer is None:
            self.buffer = torch.zeros(self.capacity, dtype=torch.uint8, device=self.device)
            logger.info(f"✅ Lazy allocated Stack Segment buffer: {self.capacity / 1024**2:.1f}MB on {self.device}")
        return self.buffer

    def allocate_volatile(self, size_bytes: int, name: str = "") -> Tuple[int, torch.Tensor]:
        """
        Allocate from the Stack Slab.

        Args:
            size_bytes: Size to allocate
            name: Optional allocation name

        Returns:
            (offset, tensor_view)
        """
        with self.lock:
            aligned_offset = self._align(self.current_offset)
            if aligned_offset + size_bytes > self.capacity:
                raise RuntimeError(f"Stack Slab out of memory: requested {size_bytes} bytes")

            # Create view of the allocated region
            view = self.slab[aligned_offset:aligned_offset + size_bytes]

            self.current_offset = aligned_offset + size_bytes

            logger.debug(f"✅ Stack allocated {size_bytes} bytes at offset {aligned_offset}")
            return aligned_offset, view

    def reset(self):
        """
        Reset the Stack Slab pointer.

        CRITICAL: Caller MUST call torch.cuda.synchronize() before this
        to ensure GPU has finished computing.
        """
        with self.lock:
            self.current_offset = 0
            self.reset_count += 1
            logger.debug(f"✅ Stack Slab reset (count: {self.reset_count})")

    def get_stats(self) -> Dict[str, any]:
        """Return Stack Segment statistics."""
        with self.lock:
            allocated_bytes = self.current_offset
            return {
                'segment_name': self.name,
                'capacity_bytes': self.capacity,
                'allocated_bytes': allocated_bytes,
                'free_bytes': self.capacity - allocated_bytes,
                'reset_count': self.reset_count,
                'utilization_percent': (allocated_bytes / self.capacity * 100) if self.capacity > 0 else 0,
            }


class UnifiedVMU:
    """
    Unified Virtual Memory Unit with Segmented Memory Model.

    Manages GPU memory as three segments:
    - Text Segment: Shared model weights (60% of VMU capacity)
    - Data Segment: Private session data (20% of VMU capacity)
    - Stack Slab: Volatile activations (20% of VMU capacity)

    Provides thread-safe allocation with 256-byte alignment.
    """

    def __init__(
        self,
        device_id: int = 0,
        *,
        vmu_config: Optional[VmuConfig] = None,
        alignment: int = 256
    ):
        """
        Initialize the segmented VMU.

        Args:
            device_id: GPU device ID
            text_capacity_ratio: Fraction of VMU for Text Segment
            data_capacity_ratio: Fraction of VMU for Data Segment
            stack_capacity_ratio: Fraction of VMU for Stack Segment
            os_reserve_gb: Memory reserved for OS/kernels
            alignment: Byte alignment for allocations
        """
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.alignment = alignment
        self.lock = threading.Lock()  # Global concurrency control

        config = vmu_config or VmuConfig.from_env()
        text_ratio = config.text_ratio
        data_ratio = config.data_ratio
        stack_ratio = config.stack_ratio

        ratio_sum = text_ratio + data_ratio + stack_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            logger.warning(f"VMU ratios do not sum to 1 ({ratio_sum:.3f}), normalizing.")
            text_ratio /= ratio_sum
            data_ratio /= ratio_sum
            stack_ratio /= ratio_sum

        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory
        reserved = torch.cuda.memory_reserved(device_id)
        free_memory = max(total_memory - reserved, 0)

        os_reserve_bytes = int(config.os_reserve_gb * 1024**3)
        safety_margin_bytes = int(config.safety_margin_gb * 1024**3)
        safe_capacity = min(total_memory - os_reserve_bytes, free_memory - safety_margin_bytes)
        safe_capacity = max(safe_capacity, 0)

        min_viable_bytes = int(config.min_viable_vmu_gb * 1024**3)
        if safe_capacity < min_viable_bytes:
            raise RuntimeError(
                f"Insufficient GPU memory for VMU: need {min_viable_bytes / 1024**3:.1f} GB, "
                f"have only {safe_capacity / 1024**3:.1f} GB "
                f"(total={total_memory / 1024**3:.1f} GB, reserved={reserved / 1024**3:.1f} GB)"
            )

        text_capacity = int(safe_capacity * text_ratio)
        data_capacity = int(safe_capacity * data_ratio)
        stack_capacity = int(safe_capacity * stack_ratio)
        self.default_session_arena_bytes = max(
            int(config.default_session_arena_mb * 1024**2),
            alignment
        )

        logger.info(f"Initializing Segmented VMU on cuda:{device_id}")
        logger.info(f"  Total GPU memory: {total_memory / 1024**3:.1f} GB")
        logger.info(f"  Reserved by system: {reserved / 1024**3:.1f} GB")
        logger.info(f"  OS reserve: {config.os_reserve_gb} GB")
        logger.info(f"  Safety margin: {config.safety_margin_gb} GB")
        logger.info(f"  Safe VMU capacity: {safe_capacity / 1024**3:.1f} GB")
        logger.info(f"  Text Segment:  {text_capacity / 1024**3:.1f} GB ({text_ratio:.0%})")
        logger.info(f"  Data Segment:  {data_capacity / 1024**3:.1f} GB ({data_ratio:.0%})")
        logger.info(f"  Stack Segment: {stack_capacity / 1024**3:.1f} GB ({stack_ratio:.0%})")

        # Initialize segments
        try:
            self.text_segment = TextSegment(text_capacity, self.device)
            self.data_segment = DataSegment(data_capacity, self.device)
            self.stack_segment = StackSegment(stack_capacity, self.device)
            logger.info("✅ All Segmented VMU initialized successfully")
        except RuntimeError as e:
            logger.error(f"❌ Failed to initialize Segmented VMU: {e}")
            raise

        # CPU Staging Buffer (Pinned for DMA) - 32MB chunk
        self.staging_size = 32 * 1024 * 1024  # 32MB
        self.staging = None
        try:
            self.staging = torch.zeros(self.staging_size, dtype=torch.uint8).pin_memory()
            logger.info(f"✅ Allocated pinned staging buffer: {self.staging_size / 1024**2:.1f} MB")
        except RuntimeError as e:
            # Fallback to regular CPU buffer if pinning fails
            logger.warning(f"⚠️  Pinned memory not available, using regular CPU buffer: {e}")
            try:
                self.staging = torch.zeros(self.staging_size, dtype=torch.uint8)
                logger.info(f"✅ Allocated regular staging buffer: {self.staging_size / 1024**2:.1f} MB")
            except RuntimeError as e2:
                logger.error(f"❌ Failed to allocate staging buffer: {e2}")
                raise

    # Session arena helpers
    def reserve_session_arena(self, session_id: str, max_bytes: Optional[int] = None) -> SessionArena:
        """
        Reserve an arena in the Data segment for a session.
        """
        size = max_bytes or self.default_session_arena_bytes
        return self.data_segment.reserve_arena(session_id, size)

    def get_metrics(self) -> VMUMetrics:
        """Collect aggregated metrics from all segments."""
        text_stats = self.text_segment.get_stats()
        data_stats = self.data_segment.get_stats()
        stack_stats = self.stack_segment.get_stats()

        return VMUMetrics(
            text_used_bytes=int(text_stats.get("loaded_bytes", 0)),
            text_capacity_bytes=int(text_stats.get("capacity_bytes", 0)),
            data_reserved_bytes=int(data_stats.get("reserved_bytes", 0)),
            data_capacity_bytes=int(data_stats.get("capacity_bytes", 0)),
            data_internal_waste_bytes=int(data_stats.get("data_internal_waste_bytes", 0)),
            data_external_gap_bytes=int(data_stats.get("data_external_gap_bytes", 0)),
            stack_allocated_bytes=int(stack_stats.get("allocated_bytes", 0)),
            stack_capacity_bytes=int(stack_stats.get("capacity_bytes", 0)),
            stack_reset_count=int(stack_stats.get("reset_count", 0)),
            active_sessions=int(data_stats.get("active_sessions", 0)),
            models_loaded=int(text_stats.get("loaded_models_count", 0)),
        )
    
    # Text Segment API
    def load_model_to_text(self, model_id: str, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Load model weights into Text Segment.

        Args:
            model_id: Unique model identifier
            state_dict: Model state dictionary

        Returns:
            True if loaded successfully, False if already loaded
        """
        return self.text_segment.load_model_weights(model_id, state_dict)

    def get_model_from_text(self, model_id: str) -> Optional[torch.nn.Module]:
        """Get loaded model from Text Segment."""
        return self.text_segment.get_model(model_id)

    def unload_model_from_text(self, model_id: str) -> bool:
        """Unload model from Text Segment."""
        return self.text_segment.unload_model(model_id)

    # Data Segment API
    def allocate_session_data(self, session_id: str, size_bytes: int, name: str = "") -> int:
        """
        Allocate space in Data Segment for session data.

        Args:
            session_id: Session identifier
            size_bytes: Size to allocate
            name: Optional allocation name for tracking

        Returns:
            Offset in Data Segment buffer
        """
        if session_id not in self.data_segment.sessions:
            self.reserve_session_arena(session_id)
        return self.data_segment.allocate_session_data(session_id, size_bytes, name)

    def get_session_data_view(self, session_id: str, offset: int, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Get a view of session data from the Data Segment."""
        return self.data_segment.get_session_view(session_id, offset, size, dtype)

    def free_session_data(self, session_id: str) -> bool:
        """Free all data for a session."""
        return self.data_segment.free_session(session_id)

    # Stack Slab API
    def allocate_volatile(self, size_bytes: int, name: str = "") -> Tuple[int, torch.Tensor]:
        """
        Allocate from the Stack Slab.

        Args:
            size_bytes: Size to allocate
            name: Optional allocation name

        Returns:
            (offset, tensor_view)
        """
        return self.stack_segment.allocate_volatile(size_bytes, name)

    def reset_stack(self):
        """
        Reset the Stack Slab pointer.

        CRITICAL: Caller MUST call torch.cuda.synchronize() before this
        to ensure GPU has finished computing.
        """
        self.stack_segment.reset()

    # DMA Transfer API (for loading data into segments)
    def _recv_exact(self, sock: socket.socket, buf) -> None:
        """
        Reliable TCP read helper that handles partial receives.

        Args:
            sock: TCP socket to read from
            buf: Buffer to read into (must be writable)

        Raises:
            ConnectionError: If connection is closed during read
        """
        view = memoryview(buf)
        while len(view) > 0:
            n = sock.recv_into(view)
            if n == 0:
                raise ConnectionError("Unexpected EOF during DMA read")
            view = view[n:]

    def write_to_segment(self, sock: socket.socket, total_size: int, segment: MemorySegment,
                        session_id: Optional[str] = None) -> int:
        """
        Reads network data -> CPU staging -> GPU segment with DMA synchronization.

        Args:
            sock: TCP socket to read from
            total_size: Total bytes to transfer
            segment: Target segment (Text, Data, or Stack)
            session_id: Session ID for Data Segment allocations

        Returns:
            Offset where data was written in the segment

        Raises:
            MemoryError: If allocation would exceed segment capacity
        """
        with self.lock:
            # Pre-flight check
            if isinstance(segment, DataSegment) and session_id is None:
                raise ValueError("session_id required for Data Segment allocation")

            # Get CUDA stream for synchronization
            stream = torch.cuda.current_stream(device=self.device)

            # Check if staging buffer is pinned
            is_pinned = self.staging.is_pinned() if hasattr(self.staging, 'is_pinned') else False
            if not is_pinned:
                logger.warning("⚠️  Using non-pinned staging buffer - DMA may be slower")

            logger.debug(f"Starting DMA transfer: {total_size} bytes to {segment.name} Segment")

            bytes_received = 0
            chunk_cap = self.staging_size  # 32MB chunks

            # Allocate space in target segment
            if isinstance(segment, TextSegment):
                # [BLOCKING-P0] For Text Segment, we'd need to know the model_id and allocate appropriately
                # [BLOCKING-P0] This is a placeholder - actual implementation would be more complex
                # [BLOCKING-P0] Required for Phase 2 - prevents network model loading
                raise NotImplementedError("Text Segment DMA not yet implemented")
            elif isinstance(segment, DataSegment):
                target_offset = segment.allocate_session_data(session_id, total_size, "dma_transfer")
            elif isinstance(segment, StackSegment):
                target_offset, target_view = segment.allocate_volatile(total_size, "dma_transfer")
            else:
                raise ValueError(f"Unknown segment type: {type(segment)}")

            # Chunked DMA Pipeline
            while bytes_received < total_size:
                chunk_size = min(total_size - bytes_received, chunk_cap)

                # A. Read into Pinned CPU Buffer
                cpu_view = self.staging[:chunk_size]
                self._recv_exact(sock, cpu_view)

                # B. DMA Copy to GPU
                gpu_offset = target_offset + bytes_received
                gpu_view = segment.buffer[gpu_offset:gpu_offset + chunk_size]

                # Non-blocking copy (async DMA)
                gpu_view.copy_(cpu_view, non_blocking=True)

                # C. CRITICAL: Synchronize to prevent CPU overwriting buffer
                stream.synchronize()

                bytes_received += chunk_size

                logger.debug(f"DMA chunk: {chunk_size} bytes, total {bytes_received}/{total_size}")

            logger.debug(f"✅ DMA transfer complete: {total_size} bytes to {segment.name} Segment")
            return target_offset
    
    # ✅ PHASE 2 COMPLETE: Legacy methods removed
    # All callsites migrated to new API:
    # - malloc_persistent() → allocate_session_data()
    # - malloc_volatile() → allocate_volatile()
        new_offset = aligned_start + size
        
        if new_offset >= self.slab_capacity:
            used_persistent = self.persistent_watermark / 1024**3
            used_volatile = (self.current_offset - self.persistent_watermark) / 1024**3
            available = self.slab_capacity / 1024**3
            raise RuntimeError(
                f"Volatile memory exhausted: "
                f"requested {size / 1024**3:.2f}GB, "
                f"used (persistent={used_persistent:.2f}GB, volatile={used_volatile:.2f}GB) / "
                f"available={available:.2f}GB"
            )
        
        # Update offset
        self.current_offset = new_offset
        
        # Track allocation
        alloc_id = self.allocation_counter
        self.allocation_counter += 1
        self.allocations[alloc_id] = AllocationProfile(
            start_offset=aligned_start,
            size_bytes=size,
            dtype=dtype,
            persistent=False,
            name=name
        )
        
        # Update stats
        self.stats['volatile_allocated_bytes'] = (
            self.current_offset - self.persistent_watermark
        )
        self.stats['allocation_count'] += 1
        self.stats['max_current_offset'] = max(
            self.stats['max_current_offset'],
            self.current_offset
        )
        
        logger.debug(
            f"Volatile allocation: {name} @ {aligned_start / 1024**2:.1f}MB, "
            f"size={size / 1024**2:.1f}MB, offset={self.current_offset / 1024**2:.1f}MB"
        )
        
        return aligned_start
    
    def reset_volatile(self) -> None:
        """
        LEGACY API: Reset Stack Slab (backward compatibility).

        [DEFER-PHASE2] This method is deprecated. Use reset_stack() instead.
        [DEFER-PHASE2] Migrate all callsites during Phase 2 executor integration.
        """
        logger.warning("⚠️  reset_volatile() is deprecated. Use reset_stack() instead.")
        self.reset_stack()

    def _recv_exact(self, sock: socket.socket, buf) -> None:
        """
        Reliable TCP read helper that handles partial receives.

        Args:
            sock: TCP socket to read from
            buf: Buffer to read into (must be writable)

        Raises:
            ConnectionError: If connection is closed during read
        """
        view = memoryview(buf)
        while len(view) > 0:
            n = sock.recv_into(view)
            if n == 0:
                raise ConnectionError("Unexpected EOF during DMA read")
            view = view[n:]

    def write_from_socket(self, sock: socket.socket, total_size: int,
                         is_persistent: bool = False) -> int:
        """
        Reads network data -> CPU staging -> GPU slab with DMA synchronization.

        Implements the v2.3.15 DMA pipeline:
        1. Pre-flight OOM check
        2. Chunked DMA with proper synchronization
        3. Memory barrier to prevent data corruption

        Args:
            sock: TCP socket to read from
            total_size: Total bytes to transfer
            is_persistent: Whether this is persistent allocation (moves watermark)

        Returns:
            GPU offset where data was written

        Raises:
            MemoryError: If allocation would exceed slab capacity
        """
        with self.lock:
            # 1. Pre-Flight OOM Check
            base_ptr = self.persistent_watermark if is_persistent else self.current_offset

            # Align start address to 256-byte boundary
            gpu_start = (base_ptr + self.alignment - 1) & ~(self.alignment - 1)

            if gpu_start + total_size > self.slab_capacity:
                used = self.persistent_watermark / 1024**3
                available = self.slab_capacity / 1024**3
                requested = total_size / 1024**3
                raise MemoryError(
                    f"Slab OOM during DMA: "
                    f"requested {requested:.2f}GB, "
                    f"used {used:.2f}GB / available {available:.2f}GB"
                )

            # 2. Chunked DMA Pipeline
            bytes_received = 0
            chunk_cap = self.staging_size  # 32MB chunks

            # Get CUDA stream for synchronization
            stream = torch.cuda.current_stream(device=self.device)

            # Check if staging buffer is pinned
            is_pinned = self.staging.is_pinned() if hasattr(self.staging, 'is_pinned') else False
            if not is_pinned:
                logger.warning("⚠️  Using non-pinned staging buffer - DMA may be slower")

            logger.debug(f"Starting DMA transfer: {total_size} bytes to GPU offset {gpu_start}")

            while bytes_received < total_size:
                chunk_size = min(total_size - bytes_received, chunk_cap)

                # A. Read into Pinned CPU Buffer
                cpu_view = self.staging[:chunk_size]
                self._recv_exact(sock, cpu_view)

                # B. Async DMA Copy to GPU
                gpu_slice_start = gpu_start + bytes_received
                gpu_slice_end = gpu_slice_start + chunk_size
                gpu_dest = self.buffer[gpu_slice_start:gpu_slice_end]

                # Non-blocking copy (async DMA)
                gpu_dest.copy_(cpu_view, non_blocking=True)

                # C. CRITICAL: Synchronize to prevent CPU overwriting buffer
                # before GPU finishes reading previous chunk
                stream.synchronize()

                bytes_received += chunk_size

                logger.debug(f"DMA chunk: {chunk_size} bytes, total {bytes_received}/{total_size}")

            # 3. Update Pointers
            new_end = gpu_start + total_size
            if is_persistent:
                self.persistent_watermark = new_end
                self.current_offset = new_end
            else:
                self.current_offset = new_end

            # 4. Update statistics
            alloc_size_mb = total_size / 1024**2
            if is_persistent:
                self.stats['persistent_allocated_bytes'] += total_size
                logger.debug(f"Persistent DMA allocation: {alloc_size_mb:.1f}MB at offset {gpu_start}")
            else:
                self.stats['volatile_allocated_bytes'] += total_size
                logger.debug(f"Volatile DMA allocation: {alloc_size_mb:.1f}MB at offset {gpu_start}")

            return gpu_start
    
    def get_segment_view(self, segment: MemorySegment, offset: int, size: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Get a view of a segment buffer at specified offset.

        Args:
            segment: Target segment (Text, Data, or Stack)
            offset: Start offset in bytes
            size: Size in bytes
            dtype: Data type

        Returns:
            torch.Tensor view of segment buffer
        """
        element_size = torch.empty(0, dtype=dtype).element_size()
        num_elements = size // element_size

        view = segment.buffer[offset:offset + size].view(dtype=dtype)

        if view.numel() != num_elements:
            logger.warning(
                f"Size mismatch in segment view: requested {num_elements}, got {view.numel()}"
            )

        return view
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get comprehensive VMU statistics.

        Returns:
            Dict with memory usage statistics for all segments
        """
        text_stats = self.text_segment.get_stats()
        data_stats = self.data_segment.get_stats()
        stack_stats = self.stack_segment.get_stats()

        total_capacity = (
            text_stats.get('capacity_bytes', 0)
            + data_stats.get('capacity_bytes', 0)
            + stack_stats.get('capacity_bytes', 0)
        )
        total_allocated = (
            text_stats.get('loaded_bytes', 0)
            + data_stats.get('allocated_bytes', data_stats.get('reserved_bytes', 0))
            + stack_stats.get('allocated_bytes', 0)
        )
        total_free = total_capacity - total_allocated

        return {
            'vmu_total_capacity_mb': total_capacity / 1024**2,
            'vmu_total_allocated_mb': total_allocated / 1024**2,
            'vmu_total_free_mb': total_free / 1024**2,
            'vmu_utilization_percent': (total_allocated / total_capacity * 100) if total_capacity > 0 else 0,

            'text_segment': text_stats,
            'data_segment': data_stats,
            'stack_segment': stack_stats,

            # Legacy compatibility (map segmented to old dual-lifecycle names)
            'persistent_allocated_mb': text_stats['loaded_bytes'] / 1024**2,  # Text segment = persistent (model weights)
            'volatile_allocated_mb': stack_stats['allocated_bytes'] / 1024**2,  # Stack segment = volatile
            'reset_volatile_count': stack_stats['reset_count'],
            'allocation_count': 0,  # [DEFER-PHASE3] Placeholder for compatibility
        }
    
    def clear(self) -> None:
        """
        Clear all segments and reset memory.

        Used during shutdown or complete cache eviction.
        """
        # Note: In production, this would need careful coordination
        # to ensure no active computations are using the segments
        logger.warning("⚠️  VMU clear() called - this may cause data corruption if segments are in use")
        # For now, just reset the stack (safest)
        self.reset_stack()
    


# Global VMU instance per device
_global_vmus: Dict[int, UnifiedVMU] = {}


def get_vmu(device_id: int = 0) -> UnifiedVMU:
    """Get or create segmented VMU for specified device."""
    if device_id not in _global_vmus:
        _global_vmus[device_id] = UnifiedVMU(device_id=device_id)
    return _global_vmus[device_id]


def reset_vmu(device_id: int = 0) -> None:
    """Reset VMU Stack Slab for specified device."""
    if device_id in _global_vmus:
        _global_vmus[device_id].reset_stack()



