"""
Pattern matchers for LazyDAG graphs.

Key difference from FX patterns:
- Nodes are LazyTensor instances
- Operation is string (e.g., 'aten::matmul')
- Use metadata hints for detection

NOTE: LazyDAGAttentionMatcher and LazyDAGConvolutionMatcher have been
consolidated into AdvancedLLMPattern and AdvancedVisionPattern respectively.
These classes are kept for reference but are no longer registered.
"""

import warnings
from typing import List, Optional
from ...patterns.base import PatternPlugin, PatternMatch


# DEPRECATED: Use AdvancedLLMPattern instead (supports metadata hints)
class LazyDAGAttentionMatcher(PatternPlugin):
    """
    Detect attention pattern in LazyDAG.
    
    DEPRECATED: This class is no longer registered. Use AdvancedLLMPattern
    which now supports both NetworkX matching and metadata hints.
    This class is kept for reference only.
    """
    name = "attention"
    expected_operations = frozenset({'matmul', 'softmax'})
    min_nodes = 3

    def match(self, graph) -> Optional[PatternMatch]:
        warnings.warn(
            "LazyDAGAttentionMatcher is deprecated and no longer registered. "
            "Use AdvancedLLMPattern instead, which supports both NetworkX matching "
            "and metadata hints. See docs/REFACTORING_MIGRATION_GUIDE.md",
            DeprecationWarning,
            stacklevel=2
        )
        """Detect Q@K.T → softmax → @V pattern."""

        # Strategy 1: Use pattern hints from metadata
        attention_nodes = []
        for node in graph.nodes():
            if hasattr(node, 'metadata'):
                hints = node.metadata.get('pattern_hints', {})
                if hints.get('likely_pattern') == 'attention':
                    attention_nodes.append(node.id)

        if attention_nodes:
            # If we have pattern hints, try to find the complete attention pattern
            # Look for the nodes around the hinted node
            for hinted_node_id in attention_nodes:
                complete_pattern = self._expand_pattern_from_hint(graph, hinted_node_id)
                if complete_pattern:
                    return complete_pattern

        # Strategy 2: Structural analysis (fallback)
        return self._detect_by_structure(graph)

    def _expand_pattern_from_hint(self, graph, hinted_node_id) -> Optional[PatternMatch]:
        """Expand pattern from a node with attention hints."""
        # Find the hinted node
        hinted_node = graph.get_node(hinted_node_id)
        if not hinted_node:
            return None

        # Look backwards for softmax and matmul
        pattern_nodes = [hinted_node_id]
        current_node = hinted_node

        # Look back for softmax
        for node in graph.nodes():
            if 'softmax' in node.operation.lower():
                # Check if this softmax feeds into our hinted node
                if self._node_feeds_into(node, hinted_node, graph):
                    pattern_nodes.insert(0, node.id)
                    current_node = node
                    break

        # Look back further for the first matmul (Q @ K.T)
        for node in graph.nodes():
            if 'matmul' in node.operation.lower() and node.id not in pattern_nodes:
                # Check if this matmul feeds into the softmax
                if self._node_feeds_into(node, current_node, graph):
                    pattern_nodes.insert(0, node.id)
                    break

        if len(pattern_nodes) >= 3:  # We should have at least 3 nodes
            return PatternMatch(
                pattern_name='attention',
                confidence=0.85,
                matched_nodes=pattern_nodes,
                operation_sequence=['matmul', 'softmax', 'matmul'],
                optimization_hints={
                    'can_use_flash_attention': True,
                    'supports_kv_cache': True
                },
                metadata={'detection_method': 'metadata_hints_expanded'}
            )

        return None

    def _node_feeds_into(self, source_node, target_node, graph) -> bool:
        """Check if source_node feeds into target_node."""
        # This is a simplified check - in practice we'd need more sophisticated
        # dependency analysis. For now, assume sequential execution order.
        source_idx = None
        target_idx = None

        nodes_list = list(graph.nodes())
        for i, node in enumerate(nodes_list):
            if node.id == source_node.id:
                source_idx = i
            elif node.id == target_node.id:
                target_idx = i

        return source_idx is not None and target_idx is not None and source_idx < target_idx

    def _detect_by_structure(self, graph) -> Optional[PatternMatch]:
        """Detect attention by analyzing operation sequence."""

        nodes_list = list(graph.nodes())

        for i, node in enumerate(nodes_list):
            # Look for softmax (characteristic of attention)
            if 'softmax' in node.operation.lower():
                # Check if previous op is matmul (Q@K.T)
                if i > 0:
                    prev_node = nodes_list[i-1]
                    if 'matmul' in prev_node.operation.lower():
                        # Check if next op is matmul (@V)
                        if i < len(nodes_list) - 1:
                            next_node = nodes_list[i+1]
                            if 'matmul' in next_node.operation.lower():
                                # Found attention pattern!
                                return PatternMatch(
                                    pattern_name='attention',
                                    confidence=0.80,
                                    matched_nodes=[
                                        prev_node.id,
                                        node.id,
                                        next_node.id
                                    ],
                                    operation_sequence=[
                                        'matmul', 'softmax', 'matmul'
                                    ],
                                    optimization_hints={
                                        'can_use_flash_attention': True
                                    },
                                    metadata={
                                        'detection_method': 'structural'
                                    }
                                )

        return None


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
class LazyDAGConvolutionMatcher(PatternPlugin):
    """
    Detect convolution patterns in LazyDAG.
    
    DEPRECATED: This class is no longer registered. Use AdvancedVisionPattern
    which now supports both NetworkX matching and metadata hints.
    This class is kept for reference only.
    """
    name = "convolution"
    expected_operations = frozenset({'conv2d', 'conv1d', 'conv3d'})

    def match(self, graph) -> Optional[PatternMatch]:
        """Detect convolutional patterns."""
        warnings.warn(
            "LazyDAGConvolutionMatcher is deprecated and no longer registered. "
            "Use AdvancedVisionPattern instead, which supports both NetworkX matching "
            "and metadata hints. See docs/REFACTORING_MIGRATION_GUIDE.md",
            DeprecationWarning,
            stacklevel=2
        )

        conv_nodes = []
        for node in graph.nodes():
            if 'conv' in node.operation.lower():
                conv_nodes.append(node.id)

        if conv_nodes:
            # Check for common conv patterns in metadata
            for node_id in conv_nodes:
                node = graph.get_node(node_id)
                if hasattr(node, 'metadata'):
                    hints = node.metadata.get('pattern_hints', {})
                    if hints.get('likely_pattern') == 'conv_bn_relu':
                        return PatternMatch(
                            pattern_name='convolution',
                            confidence=0.85,
                            matched_nodes=conv_nodes,
                            operation_sequence=['conv2d'],
                            optimization_hints={
                                'can_fuse_conv_bn': True,
                                'memory_bandwidth_sensitive': True
                            },
                            metadata={
                                'detection_method': 'metadata_hints',
                                'modality': 'vision'
                            }
                        )

            # Fallback: basic convolution detection
            return PatternMatch(
                pattern_name='convolution',
                confidence=0.75,
                matched_nodes=conv_nodes,
                operation_sequence=['conv2d'],
                optimization_hints={
                    'memory_bandwidth_sensitive': True
                },
                metadata={
                    'detection_method': 'operation_analysis',
                    'modality': 'vision'
                }
            )

        return None


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
