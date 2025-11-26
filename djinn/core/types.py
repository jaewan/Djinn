"""
Type definitions and protocols for Djinn framework.

Provides:
- NodeProtocol: Standard interface for graph nodes
- DeviceSpec: Device specification and validation
- OperationInfo: Metadata about tensor operations
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch


# ============================================================================
# EXECUTION PHASES (Semantic Annotations)
# ============================================================================

class ExecutionPhase(str, Enum):
    """Semantic execution phases for operation classification."""
    UNKNOWN = "unknown"
    FORWARD = "forward"  # General forward pass (not specialized)
    LLM_PREFILL = "llm_prefill"  # Parallel attention over input sequence
    LLM_DECODE = "llm_decode"    # Sequential token generation with KV cache
    VISION_ENCODING = "vision_encoding"  # Image feature extraction
    VISION_DECODING = "vision_decoding"  # Feature to output conversion
    MULTIMODAL_FUSION = "multimodal_fusion"  # Cross-modal interaction
    TRAINING = "training"


class DataResidency(str, Enum):
    """Intended lifetime and properties of tensor data."""
    EPHEMERAL_ACTIVATION = "ephemeral_activation"  # Temporary intermediate
    PERSISTENT_WEIGHT = "persistent_weight"  # Model parameters
    STATEFUL_KV_CACHE = "stateful_kv_cache"  # Accumulating KV cache
    GRADIENT = "gradient"  # Gradient tensors


class Modality(str, Enum):
    """Data type being processed."""
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"
    MULTIMODAL_FUSION = "fusion"  # Multiple modalities mixed


# ============================================================================
# NODE PROTOCOL (Standard Interface)
# ============================================================================

@runtime_checkable
class NodeProtocol(Protocol):
    """
    Standard interface for all graph nodes.
    
    Implementations must provide:
    - id: Unique node identifier
    - operation: The ATen operation name
    - inputs: Input nodes (dependency list)
    - metadata: Semantic annotations
    
    This protocol allows different graph representations (FX, LazyTensor DAG, etc.)
    to interoperate through a common interface.
    """
    
    @property
    def id(self) -> str:
        """Unique node identifier (deterministic across runs)."""
        ...
    
    @property
    def operation(self) -> str:
        """Operation name or ATen operator (e.g., 'aten::matmul')."""
        ...
    
    @property
    def inputs(self) -> List[NodeProtocol]:
        """List of input nodes (predecessors in DAG)."""
        ...
    
    @property
    def outputs(self) -> List[NodeProtocol]:
        """List of output nodes (successors in DAG)."""
        ...
    
    @property
    def shape(self) -> Optional[torch.Size]:
        """Output tensor shape (may be inferred lazily)."""
        ...
    
    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Output tensor data type."""
        ...
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Semantic annotations (phase, modality, residency, etc.)."""
        ...


# ============================================================================
# CONCRETE NODE IMPLEMENTATION
# ============================================================================

@dataclass
class ConcreteNode:
    """
    Concrete implementation of NodeProtocol.
    
    Used for all internal graph representations.
    Provides a "standard" node type across the framework.
    """
    
    id: str
    operation: str
    inputs: List[ConcreteNode] = field(default_factory=list)
    outputs: List[ConcreteNode] = field(default_factory=list)
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution information
    execution_time_ms: Optional[float] = None
    memory_bytes: Optional[int] = None
    
    # Cached inference results
    _inferred_shape: Optional[torch.Size] = None
    _inferred_dtype: Optional[torch.dtype] = None
    
    def get_phase(self) -> ExecutionPhase:
        """Get execution phase annotation."""
        phase_str = self.metadata.get('phase', ExecutionPhase.UNKNOWN.value)
        return ExecutionPhase(phase_str)
    
    def set_phase(self, phase: ExecutionPhase) -> None:
        """Set execution phase annotation."""
        self.metadata['phase'] = phase.value
    
    def get_residency(self) -> Optional[DataResidency]:
        """Get data residency annotation."""
        residency_str = self.metadata.get('residency')
        if residency_str:
            return DataResidency(residency_str)
        return None
    
    def set_residency(self, residency: DataResidency) -> None:
        """Set data residency annotation."""
        self.metadata['residency'] = residency.value
    
    def get_modality(self) -> Optional[Modality]:
        """Get modality annotation."""
        modality_str = self.metadata.get('modality')
        if modality_str:
            return Modality(modality_str)
        return None
    
    def set_modality(self, modality: Modality) -> None:
        """Set modality annotation."""
        self.metadata['modality'] = modality.value
    
    def add_input(self, input_node: ConcreteNode) -> None:
        """Add input node and update graph edges."""
        if input_node not in self.inputs:
            self.inputs.append(input_node)
        if self not in input_node.outputs:
            input_node.outputs.append(self)
    
    def remove_input(self, input_node: ConcreteNode) -> None:
        """Remove input node and update graph edges."""
        if input_node in self.inputs:
            self.inputs.remove(input_node)
        if self in input_node.outputs:
            input_node.outputs.remove(self)
    
    def __repr__(self) -> str:
        shape_str = f"shape={self.shape}" if self.shape else "shape=?"
        dtype_str = f"dtype={self.dtype}" if self.dtype else "dtype=?"
        return f"Node(id={self.id}, op={self.operation}, {shape_str}, {dtype_str})"


# ============================================================================
# ADAPTER FOR LEGACY REPRESENTATIONS
# ============================================================================

class DictNodeAdapter:
    """
    Adapter to make legacy dict-based nodes compatible with NodeProtocol.
    
    Enables gradual migration from dict to ConcreteNode.
    """
    
    def __init__(self, node_dict: Dict[str, Any]):
        self._dict = node_dict
        self._inputs_cache: Optional[List[NodeProtocol]] = None
    
    @property
    def id(self) -> str:
        """Get node ID from dict."""
        return self._dict.get('id', f"node_{id(self._dict)}")
    
    @property
    def operation(self) -> str:
        """Get operation name from dict."""
        return self._dict.get('operation', self._dict.get('op', 'unknown'))
    
    @property
    def inputs(self) -> List[NodeProtocol]:
        """Get input nodes, wrapping dicts if needed."""
        if self._inputs_cache is None:
            raw_inputs = self._dict.get('inputs', [])
            self._inputs_cache = []
            for inp in raw_inputs:
                if isinstance(inp, dict):
                    self._inputs_cache.append(DictNodeAdapter(inp))
                elif isinstance(inp, NodeProtocol):
                    self._inputs_cache.append(inp)
                else:
                    # Skip non-node inputs
                    pass
        return self._inputs_cache
    
    @property
    def outputs(self) -> List[NodeProtocol]:
        """Get output nodes (if available)."""
        raw_outputs = self._dict.get('outputs', [])
        result = []
        for out in raw_outputs:
            if isinstance(out, dict):
                result.append(DictNodeAdapter(out))
            elif isinstance(out, NodeProtocol):
                result.append(out)
        return result
    
    @property
    def shape(self) -> Optional[torch.Size]:
        """Get shape from dict."""
        shape = self._dict.get('shape')
        if shape is None:
            return None
        if isinstance(shape, torch.Size):
            return shape
        if isinstance(shape, (list, tuple)):
            return torch.Size(shape)
        return None
    
    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Get dtype from dict."""
        dtype = self._dict.get('dtype')
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            return torch.dtype.__getattribute__(torch, dtype)
        return None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata from dict."""
        return self._dict.get('metadata', {})


# ============================================================================
# DEVICE SPECIFICATION
# ============================================================================

@dataclass
class DeviceSpec:
    """Specification of device (GPU, CPU, etc.)."""
    
    device_type: str  # "cuda", "cpu", "remote_accelerator", etc.
    device_id: int = 0
    # ... more fields as needed
    
    @classmethod
    def from_torch_device(cls, device: torch.device) -> DeviceSpec:
        """Create DeviceSpec from torch.device."""
        return cls(
            device_type=device.type,
            device_id=device.index or 0
        )
    
    def to_torch_device(self) -> torch.device:
        """Convert to torch.device."""
        return torch.device(f"{self.device_type}:{self.device_id}")
    
    def __str__(self) -> str:
        return f"{self.device_type}:{self.device_id}"


# ============================================================================
# OPERATION INFORMATION
# ============================================================================

@dataclass
class OperationInfo:
    """Metadata about a tensor operation."""
    
    name: str  # e.g., "aten::matmul"
    input_count: int
    output_count: int
    is_inplace: bool = False
    supports_autograd: bool = True
    cost_estimate: Optional[float] = None  # FLOPs
    memory_estimate: Optional[int] = None  # Bytes


# ============================================================================
# WORKLOAD CLASSIFICATION
# ============================================================================

class WorkloadType(str, Enum):
    """Classification of compute workload."""
    LLM = "llm"  # Language model
    VISION = "vision"  # Computer vision
    MULTIMODAL = "multimodal"  # Vision + Language
    RECOMMENDATION = "recommendation"  # Recommendation system
    SCIENTIFIC = "scientific"  # Scientific computing
    GENERIC = "generic"  # Unknown or mixed


# ============================================================================
# BACKWARD COMPATIBILITY: Legacy Enum Names
# ============================================================================

# These are kept for backward compatibility with existing code
class MemoryPattern(str, Enum):
    """Memory access patterns for optimization (legacy name)."""
    STREAMING = "streaming"  # One-time use, no reuse
    REUSED = "reused"       # Multiple accesses, cache-friendly
    EPHEMERAL = "ephemeral" # Short-lived intermediate
    PERSISTENT = "persistent"  # Long-lived (e.g., KV cache)
    RANDOM = "random"       # Random access pattern


class MatchingMode(str, Enum):
    """Pattern matching modes (legacy name)."""
    EXHAUSTIVE = "exhaustive"  # Match all patterns (default, safe)
    FAST = "fast"              # Early termination (opt-in)
    REQUIRED_ONLY = "required" # Only match explicitly required patterns


class TransportType(str, Enum):
    """Transport types for data movement."""
    TCP = "tcp"
    DPDK = "dpdk"
    RDMA = "rdma"
