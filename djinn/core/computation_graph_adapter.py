"""
Adapter to make ComputationGraph compatible with unified Graph interface.

This enables gradual migration from ComputationGraph to DjinnGraph.
Once all code is migrated, this adapter can be removed.
"""

from typing import Iterator, Optional, List
from .graph_interface import DjinnGraph, NodeProtocol
from .graph import ComputationGraph, ComputationNode


class ComputationGraphNodeAdapter:
    """Adapter to make ComputationNode compatible with NodeProtocol."""
    
    def __init__(self, node: ComputationNode, graph_adapter: 'ComputationGraphAdapter'):
        self._node = node
        self._graph = graph_adapter
    
    @property
    def id(self) -> str:
        return self._node.id
    
    @property
    def operation(self) -> str:
        return self._node.operation
    
    @property
    def inputs(self) -> List['ComputationGraphNodeAdapter']:
        # Resolve input IDs to node adapters
        result = []
        for inp_id in self._node.inputs:
            inp_node = self._graph.get_node(inp_id)
            if inp_node:
                result.append(inp_node)
        return result
    
    @property
    def outputs(self) -> List['ComputationGraphNodeAdapter']:
        # Resolve output IDs to node adapters
        result = []
        for out_id in self._node.outputs:
            out_node = self._graph.get_node(out_id)
            if out_node:
                result.append(out_node)
        return result
    
    @property
    def shape(self) -> Optional:
        return self._node.metadata.get('shape')
    
    @property
    def dtype(self) -> Optional:
        return self._node.metadata.get('dtype')
    
    @property
    def metadata(self) -> dict:
        return self._node.metadata


class ComputationGraphAdapter(DjinnGraph):
    """
    Adapter to make ComputationGraph compatible with unified Graph interface.
    
    This enables legacy ComputationGraph to work with new Graph-based APIs.
    """
    
    def __init__(self, computation_graph: ComputationGraph):
        self._graph = computation_graph
        self._node_adapters: dict[str, ComputationGraphNodeAdapter] = {}
        self._build_node_cache()
    
    def _build_node_cache(self):
        """Build cache of node adapters."""
        for node in self._graph.nodes():
            self._node_adapters[node.id] = ComputationGraphNodeAdapter(node, self)
    
    def nodes(self) -> Iterator[ComputationGraphNodeAdapter]:
        """Iterate over all nodes."""
        return iter(self._node_adapters.values())
    
    def get_node(self, node_id: str) -> Optional[ComputationGraphNodeAdapter]:
        """Get node by ID."""
        return self._node_adapters.get(node_id)
    
    def topological_order(self) -> List[ComputationGraphNodeAdapter]:
        """Get nodes in topological order."""
        # Use ComputationGraph's topological_sort
        sorted_ids = self._graph.topological_sort()
        return [self._node_adapters[node_id] for node_id in sorted_ids if node_id in self._node_adapters]
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self._node_adapters)
    
    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return len(self._graph.edges)
    
    def stable_id(self) -> str:
        """Compute stable graph identifier."""
        import hashlib
        parts = []
        
        # Add nodes in topological order
        for node_id in self._graph.topological_sort():
            node = self._node_adapters[node_id]
            parts.append(f"{node.id}:{node.operation}")
            if node.shape:
                parts.append(f"shape:{tuple(node.shape)}")
            if node.dtype:
                parts.append(f"dtype:{str(node.dtype)}")
        
        # Add edges
        for src, dst in self._graph.edges:
            parts.append(f"edge:{src}->{dst}")
        
        graph_str = "|".join(parts)
        return hashlib.sha256(graph_str.encode('utf-8')).hexdigest()[:16]

