from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ...core.graph_interface import Graph
from torch import fx
from typing import Union


class PatternPlugin(ABC):
	"""Base class for pattern recognition plugins."""

	@property
	@abstractmethod
	def name(self) -> str:
		...

	@abstractmethod
	def match(self, graph: Union[Graph, fx.GraphModule]) -> Optional["PatternMatch"]:
		"""Match pattern in graph.

		Args:
			graph: Either a unified Graph interface implementation or FX GraphModule

		Returns:
			PatternMatch if pattern is found, None otherwise
		"""

	# Optional: FX-based matching
	def match_fx(self, gm: fx.GraphModule) -> Optional["PatternMatch"]:  # noqa: D401
		"""Override to support FX-based pattern detection."""
		return None


@dataclass
class PatternMatch:
	"""Result of pattern matching with comprehensive metadata."""
	pattern_name: str
	confidence: float
	matched_nodes: List[str]
	operation_sequence: Optional[List[str]] = None  # Sequence of operations in matched pattern
	optimization_hints: Dict[str, Any] = field(default_factory=dict)  # e.g., fusion opportunities
	metadata: Optional[Dict[str, Any]] = None  # Additional pattern-specific metadata
	subgraph: Optional[Any] = None  # Optional reference to matched subgraph


