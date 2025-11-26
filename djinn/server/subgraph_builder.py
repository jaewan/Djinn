"""
SubgraphBuilder: Extracts executable subgraphs from LazyTensor DAG.

This module implements the core optimization described in the network enhancement plan:
instead of executing operations one at a time (causing O(n) network round-trips),
we extract entire computation subgraphs and execute them remotely in a single request.

Key benefits:
- O(1) network round-trips instead of O(n)
- Intermediates stay on GPU (no CPU round-trips)
- Reduced serialization overhead
- Better batching opportunities

Usage:
    builder = SubgraphBuilder()
    subgraph = builder.build_remote_subgraph(target_tensor)
    # Send subgraph to remote server for execution
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Set, Optional
import torch
from enum import Enum

from ..frontend.core.lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


class SemanticNodeType(Enum):
    """Semantic types of computation nodes (mega-ops)."""
    TRANSFORMER_LAYER = "transformer_layer"  # Full transformer block
    ATTENTION_HEAD = "attention_head"  # Single attention head computation
    MLP_BLOCK = "mlp_block"  # Feedforward network block
    LAYER_NORM = "layer_norm"  # Layer normalization
    EMBEDDING = "embedding"  # Embedding lookup and projection
    ACTIVATION = "activation"  # Activation functions
    RESIDUAL_CONNECTION = "residual_connection"  # Skip connection
    FINE_GRAINED = "fine_grained"  # Individual operations (no semantic group)


@dataclass
class MegaNode:
    """
    Semantically meaningful computation node representing multiple fine-grained operations.
    
    Reduces graph size by grouping related operations (e.g., transformer layers).
    """
    node_type: SemanticNodeType
    node_id: int
    operation_group: List[LazyTensor]  # The operations it represents
    inputs: List['MegaNode']  # Dependencies on other mega-nodes
    outputs: List[LazyTensor]  # Output tensors
    metadata: Dict[str, Any]  # Additional info (layer_id, hidden_dim, etc.)
    
    def size_estimate(self) -> int:
        """Estimate the serialization size of this mega-node in bytes."""
        # Each operation is ~100 bytes when serialized
        return len(self.operation_group) * 100 + 200  # +200 for metadata


class SemanticGraphCompactor:
    """
    Compacts fine-grained computation graphs into semantic mega-nodes.
    
    Problem: GPT-2-XL sends 4,800 fine-grained operations
    Solution: Identify semantic patterns (transformer layers) and merge them
    Result: 48 mega-nodes instead of 4,800 operations (100x reduction)
    """
    
    # Pattern signatures to identify transformer layers
    TRANSFORMER_LAYER_PATTERN = {
        "layer_norm_q", "linear_q", "linear_k", "linear_v",  # Attention input projection
        "scale", "matmul_qk", "softmax", "matmul_av",  # Attention core
        "linear_out", "layer_norm",  # Attention output
        "linear_ff1", "activation", "linear_ff2",  # FFN
        "residual_add"  # Skip connection
    }
    
    def __init__(self):
        self.layer_patterns = []
        self.identified_layers = []
        
    def compact(self, operations: List[LazyTensor]) -> List[MegaNode]:
        """
        Compact a list of fine-grained operations into mega-nodes.
        
        Args:
            operations: List of LazyTensor operations in topological order
            
        Returns:
            List of MegaNode representing the compacted graph
        """
        mega_nodes = []
        processed_indices = set()
        
        # Phase 1: Identify semantic patterns (transformer layers)
        pattern_groups = self._identify_transformer_layers(operations)
        
        # Phase 2: Create mega-nodes for identified patterns
        for pattern_group in pattern_groups:
            mega_node = self._create_mega_node(
                operations, 
                pattern_group,
                len(mega_nodes)
            )
            if mega_node:
                mega_nodes.append(mega_node)
                processed_indices.update(pattern_group['indices'])
        
        # Phase 3: Create fine-grained nodes for remaining operations
        for i, op in enumerate(operations):
            if i not in processed_indices:
                mega_node = MegaNode(
                    node_type=SemanticNodeType.FINE_GRAINED,
                    node_id=len(mega_nodes),
                    operation_group=[op],
                    inputs=[],
                    outputs=[op],
                    metadata={'operation': op.operation}
                )
                mega_nodes.append(mega_node)
        
        logger.info(f"✓ Graph compaction: {len(operations)} ops → {len(mega_nodes)} mega-nodes")
        logger.info(f"  Reduction: {len(operations) / max(len(mega_nodes), 1):.1f}x")
        
        return mega_nodes
    
    def _identify_transformer_layers(self, operations: List[LazyTensor]) -> List[Dict[str, Any]]:
        """
        Identify transformer layer patterns in the operation sequence.
        
        Pattern signature: attention + feedforward + layer norm + residual
        """
        patterns = []
        i = 0
        layer_id = 0
        
        while i < len(operations):
            # Look ahead for transformer layer pattern
            window_size = min(25, len(operations) - i)  # Typical transformer layer: 15-25 ops
            window = operations[i:i+window_size]
            
            # Check if this window contains a transformer layer pattern
            op_types = [op.operation for op in window]
            has_attention = any('attention' in op or 'matmul' in op for op in op_types)
            has_ffn = any('linear' in op or 'feedforward' in op for op in op_types)
            has_layernorm = any('layer_norm' in op or 'norm' in op for op in op_types)
            
            if has_attention and has_ffn and has_layernorm and window_size > 10:
                # Found a transformer layer pattern
                patterns.append({
                    'indices': set(range(i, i + window_size)),
                    'layer_id': layer_id,
                    'size': window_size,
                    'type': SemanticNodeType.TRANSFORMER_LAYER,
                    'operations': window
                })
                layer_id += 1
                i += window_size
            else:
                i += 1
        
        return patterns
    
    def _create_mega_node(self, 
                         all_operations: List[LazyTensor],
                         pattern: Dict[str, Any],
                         node_id: int) -> Optional[MegaNode]:
        """Create a mega-node from identified pattern."""
        try:
            indices = pattern['indices']
            operations = [all_operations[i] for i in sorted(indices)]
            
            # Extract metadata
            layer_id = pattern.get('layer_id', 0)
            hidden_dim = self._estimate_hidden_dim(operations)
            num_heads = self._estimate_num_heads(operations)
            
            return MegaNode(
                node_type=pattern['type'],
                node_id=node_id,
                operation_group=operations,
                inputs=[],  # Will be set after all mega-nodes are created
                outputs=[operations[-1]],
                metadata={
                    'layer_id': layer_id,
                    'hidden_dim': hidden_dim,
                    'num_heads': num_heads,
                    'num_operations': len(operations)
                }
            )
        except Exception as e:
            logger.debug(f"Failed to create mega-node: {e}")
            return None
    
    @staticmethod
    def _estimate_hidden_dim(operations: List[LazyTensor]) -> Optional[int]:
        """Estimate hidden dimension from operation shapes."""
        for op in operations:
            if hasattr(op, 'shape') and op.shape:
                # Look for large dimension that's not batch or sequence
                dims = [d for d in op.shape if d and d > 64]
                if dims:
                    return max(dims)
        return None
    
    @staticmethod
    def _estimate_num_heads(operations: List[LazyTensor]) -> Optional[int]:
        """Estimate number of attention heads from operation shapes."""
        for op in operations:
            if 'attention' in str(op.operation).lower():
                if hasattr(op, 'shape') and len(op.shape) >= 2:
                    # Attention outputs typically have num_heads as a dimension
                    possible_heads = [d for d in op.shape if 8 <= d <= 32]
                    if possible_heads:
                        return possible_heads[0]
        return 12  # Default for BERT/GPT-2


@dataclass
class RemoteSubgraph:
    """
    Subgraph to execute remotely.

    Contains:
    - Topologically sorted operations
    - External inputs (tensors from outside this subgraph)
    - Output tensor (final result to return)
    """
    operations: List[LazyTensor]  # Sorted in execution order
    input_tensors: Dict[int, LazyTensor]  # tensor_id → LazyTensor (external inputs)
    output_tensor: LazyTensor  # Final output

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dict for network transfer.

        Returns:
            {
                'operations': [
                    {
                        'op_id': int,
                        'operation': str (e.g., 'aten::matmul'),
                        'inputs': [int, ...],  # IDs of input tensors
                        'kwargs': {...}
                    },
                    ...
                ],
                'input_tensors': {
                    'tensor_id': {
                        'shape': [int, ...],
                        'dtype': str
                    },
                    ...
                },
                'output_id': int
            }
        """
        # Track concrete tensors found during serialization
        # These need to be added to input_tensors and materialized on the client side
        concrete_tensors = {}
        
        def encode_input(inp, op_name='unknown'):
            """Encode an input value for serialization."""
            if isinstance(inp, LazyTensor):
                return id(inp)
            elif isinstance(inp, (int, float, bool)):
                return {'type': 'scalar', 'value': inp}
            elif inp is None:
                return {'type': 'none', 'value': None}
            elif isinstance(inp, tuple):
                return {'type': 'tuple', 'value': [encode_input(v, op_name) for v in inp]}
            elif isinstance(inp, list):
                return {'type': 'list', 'value': [encode_input(v, op_name) for v in inp]}
            elif isinstance(inp, torch.Tensor):
                # Concrete tensor - add to input_tensors and reference by ID
                tensor_id = id(inp)
                if tensor_id not in concrete_tensors:
                    concrete_tensors[tensor_id] = inp
                return tensor_id  # Reference by ID, will be resolved on client side
            elif isinstance(inp, torch.dtype):
                return {'type': 'dtype', 'value': str(inp)}
            elif isinstance(inp, slice):
                return {'type': 'slice', 'value': [inp.start, inp.stop, inp.step]}
            else:
                # Fallback: try to serialize as string
                logger.warning(f"Unknown input type {type(inp)} for operation {op_name}, serializing as string")
                return {'type': 'unknown', 'value': str(inp)}
        
        # Build operations with encoded inputs
        serialized_operations = []
        for op in self.operations:
            serialized_operations.append({
                'op_id': id(op),
                'operation': op.operation,
                'inputs': [encode_input(inp, op.operation) for inp in op.inputs],
                'kwargs': op.kwargs,
                'shape': list(op.shape) if op.shape else None,
                'dtype': str(op.dtype) if op.dtype else None
            })
        
        # Merge concrete tensors found during serialization into input_tensors
        merged_input_tensors = dict(self.input_tensors)
        for tensor_id, tensor in concrete_tensors.items():
            if tensor_id not in merged_input_tensors:
                merged_input_tensors[tensor_id] = tensor
        
        return {
            'operations': serialized_operations,
            'input_tensors': {
                str(tensor_id): {
                    'shape': list(tensor.shape) if hasattr(tensor, 'shape') else [],
                    'dtype': str(tensor.dtype) if hasattr(tensor, 'dtype') else 'float32'
                }
                for tensor_id, tensor in merged_input_tensors.items()
            },
            'output_id': id(self.output_tensor)
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                   input_data: Dict[str, torch.Tensor]) -> 'RemoteSubgraph':
        """
        Reconstruct subgraph from serialized data.

        This is used on the server side to rebuild the computation graph.
        """
        # This would be implemented on the server side
        # For now, just a placeholder
        raise NotImplementedError("Server-side deserialization not yet implemented")


class SubgraphBuilder:
    """
    Extracts executable subgraphs from LazyTensor DAG.

    Strategy (Phase 1 - Simple & Correct):
    - Extract ALL ancestors of target tensor
    - Execute entire chain remotely
    - Only transfer final result back

    Strategy (Phase 2 - Smart):
    - Cost-based fragmentation
    - Mix local/remote execution
    - Optimize transfer vs compute tradeoff
    """

    def build_remote_subgraph(self, target_tensor: LazyTensor,
                             defer_metadata: bool = True) -> RemoteSubgraph:
        """
        Extract subgraph for remote execution.

        Args:
            target_tensor: Final tensor to compute
            defer_metadata: If True, don't access shape/metadata during building
                           to avoid triggering expensive lazy computation

        Algorithm:
        1. Start from target tensor
        2. Traverse backwards (BFS/DFS) to find all ancestors
        3. Identify external inputs (factory ops or leaf tensors)
        4. Sort operations in topological order
        5. Return RemoteSubgraph
        """
        operations = []
        visited = set()
        input_tensors = {}
        concrete_tensors = {}  # Track concrete tensors (model weights) found during traversal

        # ✅ CRITICAL: Don't access shape/metadata during traversal to avoid triggering computation
        # Iterative post-order traversal to avoid recursion limit (fixes GPT-2 XL)
        stack = [(target_tensor, False)]  # (tensor, children_processed)
        pending = set()  # Track nodes waiting for children to be processed
        
        while stack:
            tensor, children_processed = stack.pop()
            
            if not isinstance(tensor, LazyTensor):
                continue
            
            tensor_id = id(tensor)
            
            # Skip if already fully processed
            if tensor_id in visited:
                continue
            
            # ✅ Check if this is an external input WITHOUT accessing shape
            if self._is_external_input_by_operation(tensor):
                input_tensors[tensor_id] = tensor
                visited.add(tensor_id)
                continue
            
            if not children_processed:
                # First visit - schedule revisit after children
                if tensor_id in pending:
                    continue  # Already scheduled
                pending.add(tensor_id)
                stack.append((tensor, True))  # Revisit after children
                
                # Push children (reverse order for correct topological sort)
                # ✅ NEW: Also check for concrete tensors (model weights) in inputs
                for inp in reversed(tensor.inputs):
                    if isinstance(inp, LazyTensor) and id(inp) not in visited:
                        stack.append((inp, False))
                    elif isinstance(inp, torch.Tensor) and not isinstance(inp, LazyTensor):
                        # ✅ FIX: Filter out meta tensors - they have no data and can't be sent to server
                        # Meta tensors are created during shape inference and shouldn't be in input_tensors
                        if inp.device.type == 'meta':
                            logger.debug(
                                f"Skipping meta tensor in operation inputs (shape={inp.shape}, dtype={inp.dtype}). "
                                f"Meta tensors are for shape inference only and cannot be sent to server."
                            )
                            continue
                        
                        # Concrete tensor (model weight) - add as external input
                        inp_id = id(inp)
                        if inp_id not in concrete_tensors:
                            concrete_tensors[inp_id] = inp
                            input_tensors[inp_id] = inp
            else:
                # Second visit - children processed, add to operations
                visited.add(tensor_id)
                operations.append(tensor)

        logger.info(f"Built subgraph: {len(operations)} operations, "
                   f"{len(input_tensors)} external inputs "
                   f"({len(concrete_tensors)} concrete tensors identified)")

        return RemoteSubgraph(
            operations=operations,  # Already in topological order
            input_tensors=input_tensors,
            output_tensor=target_tensor
        )

    def _is_external_input(self, tensor: LazyTensor) -> bool:
        """Check if tensor is an external input (not computed in this subgraph)."""
        if not hasattr(tensor, 'operation'):
            return True
        # Factory operations create tensors from nothing
        factory_ops = {
            'aten::randn', 'aten::zeros', 'aten::ones', 'aten::empty',
            'aten::tensor', 'aten::as_tensor', 'aten::from_numpy',
            'aten::arange', 'aten::linspace', 'aten::logspace',
            'aten::eye', 'aten::full'
        }
        return tensor.operation in factory_ops

    def _is_external_input_by_operation(self, tensor: LazyTensor) -> bool:
        """
        Check if tensor is an external input WITHOUT accessing shape.

        This avoids triggering expensive lazy shape computation during subgraph building.
        """
        # ✅ Access operation directly to avoid property triggers
        operation = object.__getattribute__(tensor, '_operation')

        # Factory operations create tensors from nothing
        factory_ops = {
            'aten::randn', 'aten::zeros', 'aten::ones', 'aten::empty',
            'aten::tensor', 'aten::as_tensor', 'aten::from_numpy',
            'aten::arange', 'aten::linspace', 'aten::logspace',
            'aten::eye', 'aten::full'
        }
        return operation in factory_ops

    def _should_include_in_subgraph(self, tensor: LazyTensor) -> bool:
        """Check if tensor should be included in remote subgraph."""
        # For Phase 1: include all tensors targeting remote devices

        # Check current device property (this should work for remote devices)
        if hasattr(tensor, 'device') and tensor.device:
            device_str = str(tensor.device)
            if ('remote_accelerator' in device_str or
                'privateuseone' in device_str):
                return True

        # Check original device (for remote devices mapped to meta)
        if hasattr(tensor, '_original_device') and tensor._original_device:
            device_str = str(tensor._original_device)
            if ('remote_accelerator' in device_str or
                'privateuseone' in device_str):
                return True

        # Check device string from kwargs
        if hasattr(tensor, '_kwargs') and tensor._kwargs:
            device = tensor._kwargs.get('device')
            if device:
                device_str = str(device)
                return ('remote_accelerator' in device_str or
                       'privateuseone' in device_str)

        return False

    def build_from_device_chain(self, target_tensor: LazyTensor) -> Optional[RemoteSubgraph]:
        """
        Build subgraph only for tensors that target remote devices.

        This is the main entry point for the optimization - only extract
        subgraphs for operations that actually need remote execution.
        """
        # Check if this tensor chain targets remote devices by checking if any
        # ancestor is a remote device
        has_remote_ancestor = self._has_remote_ancestor(target_tensor)

        if not has_remote_ancestor:
            return None

        # Build the subgraph
        return self.build_remote_subgraph(target_tensor)

    def _has_remote_ancestor(self, tensor: LazyTensor) -> bool:
        """Check if tensor has any remote device ancestor."""
        visited = set()

        def check_ancestors(t):
            if id(t) in visited:
                return False
            visited.add(id(t))

            # Check if this tensor is remote
            if self._should_include_in_subgraph(t):
                return True

            # Check inputs
            for inp in t.inputs:
                if isinstance(inp, LazyTensor):
                    if check_ancestors(inp):
                        return True

            return False

        return check_ancestors(tensor)
