from __future__ import annotations

from typing import Callable, Dict, Any, List, Tuple, Optional
import logging

import torch
import torch.nn as nn

from .phase_detector import get_phase_detector, PhaseAwareHook
from ...core.types import ExecutionPhase

logger = logging.getLogger(__name__)


class HookManager:
	"""Enhanced hook manager with phase detection capabilities.
	
	Collects module context and execution phase information via forward hooks.
	"""

	def __init__(self, enable_phase_detection: bool = True) -> None:
		self.hooks: Dict[str, Callable] = {}
		self.context: Dict[str, Any] = {}
		self._captures: List[Tuple[str, Dict[str, Any]]] = []
		self.enable_phase_detection = enable_phase_detection
		self.phase_detector = get_phase_detector() if enable_phase_detection else None
		self.phase_aware_hooks = {}

	def register_hook(self, module_name: str, hook_fn: Callable) -> None:
		self.hooks[module_name] = hook_fn

	def inject_hooks(self, model: nn.Module) -> None:
		for name, module in model.named_modules():
			# Always capture minimal context
			module.register_forward_hook(lambda m, i, o, n=name: self.capture_context(n, m, i, o))
			
			# Add phase-aware hook if enabled
			if self.enable_phase_detection:
				phase_hook = PhaseAwareHook(module_name=name)
				self.phase_aware_hooks[name] = phase_hook
				module.register_forward_hook(phase_hook)
			
			# Run additional custom hook if provided for this module type
			hook_fn = self.hooks.get(type(module).__name__)
			if hook_fn is not None:
				module.register_forward_hook(lambda m, i, o, n=name, f=hook_fn: f(n, m, i, o))

	def capture_context(self, name: str, module: nn.Module, inputs, output) -> None:
		in_shapes = [getattr(i, "shape", None) for i in inputs] if isinstance(inputs, (list, tuple)) else [getattr(inputs, "shape", None)]
		out_shape = getattr(output, "shape", None)
		info = {
			"module_path": name,
			"module_type": type(module).__name__,
			"input_shapes": in_shapes,
			"output_shape": out_shape,
			"execution_phase": self._detect_phase(module, inputs),
			"semantic_role": self._infer_semantic_role(name, module),
		}
		self.context[name] = info
		self._captures.append((name, info))

	def get_context(self, graph=None):  # noqa: ANN001
		return self.context

	def get_captures(self) -> List[Tuple[str, Dict[str, Any]]]:
		return list(self._captures)

	def _detect_phase(self, module: nn.Module, inputs) -> str:  # noqa: ANN001
		"""Detect execution phase using the phase detector.
		
		Args:
			module: PyTorch module being executed
			inputs: Input tensors
			
		Returns:
			Phase name string
		"""
		if self.enable_phase_detection and self.phase_detector:
			# Get operation name from module
			operation = module.__class__.__name__.lower()
			
			# Convert inputs to list if needed
			input_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]
			
			# Detect phase
			phase = self.phase_detector.detect_phase(operation, input_list)
			
			return phase.value if isinstance(phase, ExecutionPhase) else str(phase)
		
		# Fallback to simple detection
		if hasattr(module, 'training'):
			return 'train' if module.training else 'eval'
		return 'unknown'

	def _infer_semantic_role(self, name: str, module: nn.Module) -> str:  # noqa: ANN001
		mtype = type(module).__name__.lower()
		if 'attn' in mtype or 'attention' in mtype:
			return 'attention'
		if 'conv' in mtype:
			return 'conv'
		if 'norm' in mtype or 'layernorm' in mtype or 'batchnorm' in mtype:
			return 'normalization'
		if 'embedding' in mtype:
			return 'embedding'
		return 'module'
	
	def get_phase_statistics(self) -> Optional[Dict[str, Any]]:
		"""Get phase detection statistics.
		
		Returns:
			Phase statistics dictionary or None if phase detection disabled
		"""
		if self.enable_phase_detection and self.phase_detector:
			return self.phase_detector.get_phase_statistics()
		return None
	
	def get_current_phase(self) -> Optional[ExecutionPhase]:
		"""Get the current execution phase.
		
		Returns:
			Current ExecutionPhase or None
		"""
		if self.enable_phase_detection and self.phase_detector:
			return self.phase_detector.get_current_phase()
		return None
	
	def reset_phase_detector(self):
		"""Reset the phase detector state."""
		if self.enable_phase_detection and self.phase_detector:
			self.phase_detector.reset()
	
	def mark_batch_boundary(self):
		"""Mark boundary between batches for phase detection."""
		if self.enable_phase_detection and self.phase_detector:
			self.phase_detector.mark_batch_boundary()


