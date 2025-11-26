from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import time
import torch
from enum import Enum

# Import shared types from core.types to avoid circular dependencies
from ...core.types import ExecutionPhase, MemoryPattern


@dataclass
class DataLineage:
	"""Track data flow through the model."""
	source_modules: List[str] = field(default_factory=list)
	source_tensors: List[str] = field(default_factory=list)
	modality: Optional[str] = None  # "vision", "text", "audio", etc.
	transformation_chain: List[str] = field(default_factory=list)


@dataclass
class SemanticMetadata:
	"""Rich semantic metadata for tensors and FX nodes.

	This is the canonical definition used across the project (LazyTensor and FX).
	Enhanced to match HotNets'25 paper requirements.
	"""
	operation_type: str
	tensor_shape: Optional[torch.Size] = None
	dtype: Optional[torch.dtype] = None
	device_hint: str = "remote_accelerator:0"

	# Enhanced semantic enrichment (Paper Section 2.1)
	module_path: Optional[str] = None  # e.g., "VQA.fusion_block.attention"
	semantic_role: Optional[str] = None  # e.g., "cross_attention_projection"
	execution_phase: Optional[ExecutionPhase] = None
	data_lineage: Optional[DataLineage] = None
	memory_pattern: Optional[MemoryPattern] = None
	
	# Model-specific context
	model_module: Optional[str] = None  # Which nn.Module this belongs to
	layer_depth: Optional[int] = None   # Depth in the model
	is_gradient: bool = False           # Whether this is a gradient tensor
	
	# Workload hints for optimization
	workload_hints: Optional[Dict] = None
	kv_cache_related: bool = False  # LLM KV cache operations
	is_activation: bool = False     # Activation vs weight
	requires_sync: bool = False     # Needs synchronization
	
	# Performance hints
	compute_intensity: float = 1.0
	memory_access: str = "sequential"
	recompute_cost: Optional[float] = None
	estimated_flops: Optional[int] = None
	memory_bandwidth_required: Optional[float] = None  # GB/s

	# Scheduling hints (Paper Section 3.2)
	can_parallelize: bool = True
	preferred_device: Optional[str] = None  # Placement hint
	colocation_group: Optional[str] = None  # Co-locate with other ops
	priority: int = 0  # Execution priority
	
	# Metadata versioning
	metadata_version: str = "2.0"  # Updated version
	created_at: float = 0.0
	last_updated: float = 0.0

	def __post_init__(self) -> None:
		if not self.created_at:
			self.created_at = time.time()
		if not self.last_updated:
			self.last_updated = self.created_at
			
	def update(self) -> None:
		"""Update timestamp when metadata changes."""
		self.last_updated = time.time()
		
	def to_dict(self) -> Dict:
		"""Convert to dictionary for serialization."""
		return {
			"operation_type": self.operation_type,
			"tensor_shape": list(self.tensor_shape) if self.tensor_shape else None,
			"dtype": str(self.dtype) if self.dtype else None,
			"device_hint": self.device_hint,
			"module_path": self.module_path,
			"semantic_role": self.semantic_role,
			"execution_phase": self.execution_phase.value if self.execution_phase else None,
			"memory_pattern": self.memory_pattern.value if self.memory_pattern else None,
			"model_module": self.model_module,
			"layer_depth": self.layer_depth,
			"is_gradient": self.is_gradient,
			"kv_cache_related": self.kv_cache_related,
			"priority": self.priority,
			"metadata_version": self.metadata_version
		}