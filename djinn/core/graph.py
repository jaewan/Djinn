from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import warnings
from torch import fx

from ..common.async_local import AsyncLocal


@dataclass
class ComputationNode:
	"""Computation node in the legacy graph representation.
	
	.. deprecated:: 2025-09-30
		Use FX GraphModule instead. This class will be removed in a future version.
		See the migration guide in docs/REFACTORING_PLAN.md for details.
	"""
	id: str
	operation: str
	inputs: List[str]
	outputs: List[str]
	metadata: Dict


@dataclass
class ComputationGraph:
	"""Legacy computation graph representation.

	.. deprecated:: 2025-09-30
		Use FX GraphModule with FXGraphAdapter instead. This class will be removed
		in a future version. See the migration guide in docs/REFACTORING_PLAN.md.

	New code should use:
		- `genie.core.graph_builder.GraphBuilder` for building graphs
		- `genie.core.legacy_graph_adapter.LegacyGraphAdapter` for analyzing graphs
	"""
	_nodes: Dict[str, ComputationNode]
	edges: List[Tuple[str, str]]  # (source_id, target_id)
	entry_points: Set[str] = field(default_factory=set)

	def nodes(self):
		"""Return an iterable of nodes (for compatibility with graph interface).

		Returns:
			Iterable of ComputationNode objects
		"""
		return self._nodes.values()

	def get_node(self, node_id: str) -> Optional[ComputationNode]:
		"""Get node by ID."""
		return self._nodes.get(node_id)

	def edges(self):
		"""Return an iterable of edges (for compatibility with graph interface).

		Returns:
			Iterable of (source_id, target_id) tuples
		"""
		return self.edges

	def __post_init__(self):
		"""Emit deprecation warning on first use."""
		warnings.warn(
			"ComputationGraph is deprecated and will be removed in a future version. "
			"Use FX GraphModule with FXGraphAdapter instead. "
			"See docs/REFACTORING_PLAN.md for migration guide.",
			DeprecationWarning,
			stacklevel=3
		)

	@classmethod
	def empty(cls):
		"""Create empty graph for testing.

		Returns:
			ComputationGraph: Empty graph with no nodes, edges, or entry points
		"""
		return cls(_nodes={}, edges=[], entry_points=set())

	def add_node(self, operation: str, node_id: str, inputs: List[str] = None, outputs: List[str] = None, metadata: Dict = None) -> 'ComputationNode':
		"""Add a node to the graph.

		Args:
			operation: Operation type (e.g., 'aten::matmul')
			node_id: Unique identifier for the node
			inputs: List of input node IDs
			outputs: List of output node IDs
			metadata: Additional metadata

		Returns:
			ComputationNode: The created node
		"""
		if inputs is None:
			inputs = []
		if outputs is None:
			outputs = []
		if metadata is None:
			metadata = {}

		node = ComputationNode(
			id=node_id,
			operation=operation,
			inputs=inputs,
			outputs=outputs,
			metadata=metadata
		)

		self._nodes[node_id] = node
		return node

	def add_edge(self, source, target):
		"""Add an edge between two nodes.

		Args:
			source: Source node ID or ComputationNode object
			target: Target node ID or ComputationNode object
		"""
		# Handle both node objects and node IDs
		if hasattr(source, 'id'):
			source_id = source.id
		else:
			source_id = str(source)

		if hasattr(target, 'id'):
			target_id = target.id
		else:
			target_id = str(target)

		# Check that both nodes exist
		if source_id not in self._nodes:
			raise ValueError(f"Source node {source_id} does not exist")
		if target_id not in self._nodes:
			raise ValueError(f"Target node {target_id} does not exist")

		self.edges.append((source_id, target_id))

		# Add target to source's outputs
		if target_id not in self._nodes[source_id].outputs:
			self._nodes[source_id].outputs.append(target_id)

		# Add source to target's inputs
		if source_id not in self._nodes[target_id].inputs:
			self._nodes[target_id].inputs.append(source_id)

	def topological_sort(self) -> List[str]:
		visited: Set[str] = set()
		stack: List[str] = []

		def visit(node_id: str) -> None:
			if node_id in visited:
				return
			visited.add(node_id)
			node = self._nodes[node_id]
			for inp in node.inputs:
				if inp in self._nodes:
					visit(inp)
			stack.append(node_id)

		for node_id in self._nodes:
			visit(node_id)
		return stack

	def validate_invariants(self) -> Tuple[bool, List[str]]:
		"""Validate basic invariants for nodes and edges.

		Checks:
		- All edge endpoints exist in nodes
		- Node ids are unique
		- No self-loops
		- Inputs referenced as strings must exist as node ids
		"""
		errors: List[str] = []
		# Unique ids
		if len(set(self._nodes.keys())) != len(self._nodes):
			errors.append("Duplicate node ids detected")
		# Edge endpoints
		for src, dst in self.edges:
			if src == dst:
				errors.append(f"Self-loop detected: {src}")
			if src not in self._nodes:
				errors.append(f"Edge src missing node: {src}")
			if dst not in self._nodes:
				errors.append(f"Edge dst missing node: {dst}")
		# Input references
		for node in self._nodes.values():
			for inp in node.inputs:
				if isinstance(inp, str) and inp.startswith("lt_") and inp not in self._nodes:
					errors.append(f"Missing input node: {inp} for {node.id}")
		return (len(errors) == 0, errors)


class GraphBuilder:
	"""Thread-local graph builder."""

	_thread_local = AsyncLocal("legacy_graph_builder")

	def __init__(self) -> None:
		self._nodes: Dict[str, ComputationNode] = {}
		self.edges: List[Tuple[str, str]] = []
		self.tensor_to_node: Dict[str, ComputationNode] = {}
		self._concrete_map: Dict[str, object] = {}
		self._lazy_tensors: Dict[str, object] = {}
		self._fx_graph: Optional[fx.Graph] = None

	@classmethod
	def current(cls) -> "GraphBuilder":
		if not hasattr(cls._thread_local, "builder"):
			cls._thread_local.builder = cls()
		return cls._thread_local.builder

	@classmethod
	def reset_current(cls) -> None:
		"""Reset the thread-local builder to a fresh instance."""
		cls._thread_local.builder = cls()

	def add_tensor(self, lazy_tensor) -> None:  # noqa: ANN001
		node = ComputationNode(
			id=lazy_tensor.id,
			operation=lazy_tensor.operation,
			inputs=[self._get_tensor_id(inp) for inp in lazy_tensor.inputs],
			outputs=[lazy_tensor.id],
			metadata={
				"shape": lazy_tensor.shape, 
				"dtype": lazy_tensor.dtype,
				"kwargs": lazy_tensor.kwargs,
				"device": lazy_tensor.device
			},
		)
		self._nodes[node.id] = node
		self.tensor_to_node[lazy_tensor.id] = node
		self._lazy_tensors[lazy_tensor.id] = lazy_tensor
		for inp in lazy_tensor.inputs:
			inp_id = getattr(inp, "id", None)
			if inp_id:
				self.edges.append((inp_id, lazy_tensor.id))

	def reset(self) -> None:
		"""Clear all graph state for a fresh build cycle."""
		self._nodes.clear()
		self.edges.clear()
		self.tensor_to_node.clear()
		self._concrete_map.clear()
		self._lazy_tensors.clear()
		self._fx_graph = None

	def _get_tensor_id(self, tensor) -> str:  # noqa: ANN001
		tid = getattr(tensor, "id", None)
		if tid is not None:
			return tid
		key = f"concrete_{id(tensor)}"
		self._concrete_map[key] = tensor
		return key

	def get_graph(self) -> ComputationGraph:
		"""Get computation graph in legacy format.
		
		.. deprecated:: 2025-09-30
			Use FXGraphBuilder instead for new code. This method returns the legacy
			ComputationGraph format which will be removed in a future version.
		"""
		return ComputationGraph(
			nodes=self._nodes.copy(),
			edges=self.edges.copy(),
			entry_points={nid for nid in self._nodes if not any(e[1] == nid for e in self.edges)},
		)

	def get_concrete(self, tensor_id: str):
		return self._concrete_map.get(tensor_id)

	def get_lazy_tensor(self, tensor_id: str):
		return self._lazy_tensors.get(tensor_id)

	# FX integration hooks (optional for P1)
	def set_fx_graph(self, fx_graph: fx.Graph) -> None:
		"""Record an FX graph to allow cross-referencing with LazyTensor graph."""
		self._fx_graph = fx_graph

	def get_fx_graph(self) -> Optional[fx.Graph]:
		return self._fx_graph


