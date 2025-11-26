from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Optional, Union
import torch.fx as fx

# ✅ PHASE 3: Use unified Graph interface
from djinn.core.graph_interface import Graph as DjinnGraph
from djinn.core.computation_graph_adapter import ComputationGraphAdapter
from djinn.core.graph import ComputationGraph, GraphBuilder  # Legacy, for adapter


@dataclass
class GraphHandoff:
	"""Data passed from LazyTensor to Semantic Analyzer (per interface contract).
	
	Updated for Refactoring #3: Now supports both old ComputationGraph and new
	FX GraphModule for gradual migration.
	"""
	# New format (Refactoring #3)
	fx_graph: Optional[fx.GraphModule] = None
	
	# ✅ PHASE 3: Unified Graph interface (wraps ComputationGraph if needed)
	graph: Optional[DjinnGraph] = None
	# Legacy ComputationGraph (for adapter)
	_legacy_graph: Optional[ComputationGraph] = None
	
	# Common metadata
	lazy_tensors: Dict[str, object] = None
	materialization_frontier: Set[str] = None
	metadata_version: str = "2.0"  # Updated version for FX support
	
	def __post_init__(self):
		"""Initialize default values."""
		if self.lazy_tensors is None:
			self.lazy_tensors = {}
		if self.materialization_frontier is None:
			self.materialization_frontier = set()
	
	@property
	def uses_fx(self) -> bool:
		"""Check if using new FX graph format."""
		return self.fx_graph is not None
	
	def validate(self) -> bool:
		"""Validate handoff data integrity."""
		# Must have either old or new format
		if self.graph is None and self.fx_graph is None:
			return False
		
		# Validate old format if present
		if self.graph is not None and self.fx_graph is None:
			# Only validate tensor mapping if using old format exclusively
			# (Allow empty lazy_tensors for basic validation)
			if self.lazy_tensors:
				# All nodes should have corresponding tensors if map is non-empty
				for node_id in self.graph.nodes:
					if node_id not in self.lazy_tensors:
						return False
			# No dangling references
			for source, target in self.graph.edges:
				if source not in self.graph.nodes or target not in self.graph.nodes:
					return False
		
		# Validate new format if present
		if self.fx_graph is not None:
			# Basic validation - graph should have nodes
			if not hasattr(self.fx_graph, 'graph'):
				return False
		
		return True
	
	def get_graph_for_analysis(self) -> DjinnGraph:
		"""Get graph in unified Graph interface for analysis.
		
		Returns:
			Unified Graph interface (DjinnGraph)
		"""
		if self.graph is not None:
			return self.graph
		elif self._legacy_graph is not None:
			# Wrap legacy ComputationGraph
			return ComputationGraphAdapter(self._legacy_graph)
		else:
			raise ValueError("No graph available in handoff")


def build_graph_handoff() -> GraphHandoff:
	"""Helper to assemble GraphHandoff from current GraphBuilder state.
	
	Updated for Refactoring #3: Attempts to build FX graph first, falls back
	to ComputationGraph for backward compatibility.
	"""
	gb = GraphBuilder.current()
	lazy_map = getattr(gb, "_lazy_tensors", {})
	
	# Try to get FX graph first (new format)
	fx_graph = None
	try:
		from djinn.core.fx_graph_builder import FXGraphBuilder
		fx_builder = FXGraphBuilder.current()
		
		# Check if FX builder has content
		if hasattr(fx_builder, 'value_map') and fx_builder.value_map:
			fx_graph = fx_builder.to_graph_module()
	except (ImportError, AttributeError, Exception) as e:
		# FX not available or empty, fall back to old format
		pass
	
	# ✅ PHASE 3: Get graph in unified format
	# Try new GraphBuilder first (from frontend/core/graph_builder.py)
	unified_graph = None
	legacy_graph = None
	
	try:
		from djinn.frontend.core.graph_builder import get_global_builder as get_new_builder
		new_builder = get_new_builder()
		if new_builder.root_tensor is not None:
			unified_graph = new_builder.get_graph()
	except (ImportError, AttributeError):
		pass
	
	# Fallback to legacy GraphBuilder if needed
	if unified_graph is None:
		legacy_graph = gb.get_graph()
		unified_graph = ComputationGraphAdapter(legacy_graph)
	
	# Compute materialization frontier
	frontier: Set[str] = set()
	if legacy_graph:
		with_outgoing = {src for src, _ in gb.edges}
		for node_id in legacy_graph._nodes.keys():
			if node_id not in with_outgoing:
				frontier.add(node_id)
	else:
		# Use unified graph interface
		for node in unified_graph.nodes():
			if not node.outputs:
				frontier.add(node.id)
	
	return GraphHandoff(
		fx_graph=fx_graph,
		graph=unified_graph,
		_legacy_graph=legacy_graph,
		lazy_tensors=dict(lazy_map),
		materialization_frontier=frontier
	)


