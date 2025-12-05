"""
Pattern matchers for LazyDAG graphs.

Key difference from FX patterns:
- Nodes are LazyTensor instances
- Operation is string (e.g., 'aten::matmul')
- Use metadata hints for detection

NOTE: Pattern matchers in this module are kept for reference.
For new implementations, use AdvancedLLMPattern and AdvancedVisionPattern
which support both NetworkX matching and metadata hints.
"""

from typing import List, Optional
from ...patterns.base import PatternPlugin, PatternMatch


class LazyDAGKVCacheMatcher(PatternPlugin):
    """Detect KV cache pattern in LazyDAG."""

    name = "kv_cache"
    expected_operations = frozenset({'cat', 'concat'})

    def match(self, graph) -> Optional[PatternMatch]:
        """Detect recurrent concatenation pattern."""

        for node in graph.nodes():
            if 'cat' in node.operation.lower():
                # Check metadata for hints
                if hasattr(node, 'metadata'):
                    semantic_role = node.metadata.get('semantic_role')
                    if semantic_role == 'kv_cache_update':
                        return PatternMatch(
                            pattern_name='kv_cache',
                            confidence=0.90,
                            matched_nodes=[node.id],
                            optimization_hints={
                                'requires_colocation': True,
                                'colocate_with_decoder': True
                            },
                            metadata={
                                'execution_phase': 'llm_decode',
                                'residency': 'persistent_kv_cache'
                            }
                        )

        return None


class LazyDAGLinearMatcher(PatternPlugin):
    """Detect linear/MLP patterns in LazyDAG."""

    name = "linear"
    expected_operations = frozenset({'linear', 'matmul'})

    def match(self, graph) -> Optional[PatternMatch]:
        """Detect linear transformation patterns."""

        linear_nodes = []
        for node in graph.nodes():
            if 'linear' in node.operation.lower():
                linear_nodes.append(node.id)
            elif 'matmul' in node.operation.lower():
                # Check if this looks like a linear layer (2D input/output)
                if hasattr(node, 'shape') and len(node.shape) == 2:
                    linear_nodes.append(node.id)

        if linear_nodes:
            return PatternMatch(
                pattern_name='linear',
                confidence=0.70,
                matched_nodes=linear_nodes,
                operation_sequence=['matmul'],
                optimization_hints={
                    'can_fuse': True,
                    'compute_bound': True
                },
                metadata={
                    'detection_method': 'operation_analysis'
                }
            )

        return None


# DEPRECATED: Use AdvancedVisionPattern instead (supports metadata hints)
class LazyDAGActivationMatcher(PatternPlugin):
    """Detect activation patterns in LazyDAG."""

    name = "activation"
    expected_operations = frozenset({'relu', 'gelu', 'tanh', 'sigmoid'})

    def match(self, graph) -> Optional[PatternMatch]:
        """Detect activation function patterns."""

        activation_nodes = []
        for node in graph.nodes():
            for op in self.expected_operations:
                if op in node.operation.lower():
                    activation_nodes.append(node.id)

        if activation_nodes:
            return PatternMatch(
                pattern_name='activation',
                confidence=0.80,
                matched_nodes=activation_nodes,
                operation_sequence=list(self.expected_operations),
                optimization_hints={
                    'can_fuse': True,
                    'memory_efficient': True
                },
                metadata={
                    'detection_method': 'operation_analysis'
                }
            )

        return None
