"""Advanced pattern implementations using NetworkX for better accuracy and performance."""

from __future__ import annotations

from typing import List, Optional, Dict, Set
import networkx as nx

from ...core.graph_interface import Graph
from .base import PatternMatch, PatternPlugin
from typing import Union
from djinn.frontend.semantic.graph_utils import (
    graph_to_networkx, find_attention_pattern, find_conv_activation_pattern,
    find_mlp_pattern, find_embedding_pattern, analyze_graph_complexity,
    track_performance
)


class AdvancedLLMPattern(PatternPlugin):
    """Advanced LLM pattern detection using NetworkX subgraph matching."""

    @property
    def name(self) -> str:
        return "llm"

    # Hierarchical index metadata - core operations for LLM patterns (no aten:: prefix)
    expected_operations = frozenset({
        "matmul", "softmax"
    })
    min_nodes = 5
    max_nodes = 1000
    allows_cycles = False
    max_fanout = 50

    @track_performance
    def match(self, graph) -> Optional[PatternMatch]:
        """Detect LLM patterns using sophisticated graph analysis.
        
        Uses multiple strategies:
        1. Metadata hints (fast, for LazyDAG graphs)
        2. NetworkX subgraph matching (accurate, for all graph types)
        3. Structural analysis (fallback)
        """
        # Strategy 1: Check for metadata hints (fast path for LazyDAG)
        attention_nodes = []
        for node in graph.nodes():
            if hasattr(node, 'metadata'):
                metadata = getattr(node, 'metadata', {})
                if isinstance(metadata, dict):
                    hints = metadata.get('pattern_hints', {})
                    if hints.get('likely_pattern') == 'attention':
                        attention_nodes.append(node.id)
        
        # If we have metadata hints, try to expand the pattern
        if attention_nodes:
            for hinted_node_id in attention_nodes[:1]:  # Try first hint
                expanded = self._expand_pattern_from_hint(graph, hinted_node_id)
                if expanded:
                    return expanded
        
        # Strategy 2: NetworkX subgraph matching (accurate, works for all graphs)
        # Convert to NetworkX for pattern analysis (handles all graph types)
        G = graph_to_networkx(graph)

        # Find attention patterns
        attention_matches = find_attention_pattern(G)

        # Find MLP patterns (common in transformers)
        mlp_matches = find_mlp_pattern(G)

        # Analyze overall graph characteristics
        complexity = analyze_graph_complexity(G)

        # Scoring based on pattern density and characteristics
        attention_score = min(len(attention_matches) * 0.6, 1.0)
        mlp_score = min(len(mlp_matches) * 0.2, 0.8)
        complexity_score = complexity["compute_intensity"] * 0.5

        total_score = attention_score + mlp_score + complexity_score

        if total_score > 0.8 or len(attention_matches) >= 1:
            confidence = min(total_score, 0.95)
            if len(attention_matches) >= 1 and confidence < 0.85:
                confidence = 0.88
            matched_nodes = []

            # Collect matched node IDs
            for match in attention_matches[:3]:  # Limit to top 3
                matched_nodes.extend(match.values())
            for match in mlp_matches[:2]:  # Limit to top 2
                matched_nodes.extend(match.values())

            return PatternMatch(
                pattern_name=self.name,
                confidence=confidence,
                matched_nodes=list(set(matched_nodes)),  # Remove duplicates
                operation_sequence=["matmul", "softmax", "matmul"],  # Typical attention pattern
                optimization_hints={
                    "can_fuse_qkv_projection": len(attention_matches) >= 1,
                    "can_use_flash_attention": len(attention_matches) >= 1,
                    "supports_kv_cache": True,
                    "mlp_fusion_opportunity": len(mlp_matches) >= 2
                },
                metadata={
                    "attention_blocks": len(attention_matches),
                    "mlp_blocks": len(mlp_matches),
                    "compute_intensity": complexity.get("compute_intensity", 0),
                    "semantic_role": "transformer_layer"
                }
            )

        # Fallback for simpler transformer patterns
        if attention_matches or (len(mlp_matches) >= 2):
            return PatternMatch(
                pattern_name=self.name,
                confidence=0.7,
                matched_nodes=[list(match.values())[0] for match in attention_matches[:1]],
                operation_sequence=["matmul", "softmax"],
                optimization_hints={"can_use_flash_attention": True},
                metadata={"semantic_role": "attention_layer"}
            )
        
        return None
    
    def _expand_pattern_from_hint(self, graph, hinted_node_id) -> Optional[PatternMatch]:
        """Expand pattern from a node with attention hints (from LazyDAG metadata)."""
        # Find the hinted node
        hinted_node = None
        for node in graph.nodes():
            if str(node.id) == str(hinted_node_id):
                hinted_node = node
                break
        
        if not hinted_node:
            return None

        # Look backwards for softmax and matmul
        pattern_nodes = [str(hinted_node_id)]
        
        # Try to find softmax and matmul in graph structure
        # This is a simplified approach - NetworkX matching is more accurate
        # but this provides a fast path for metadata-annotated graphs
        nodes_list = list(graph.nodes())
        for i, node in enumerate(nodes_list):
            if str(node.id) == str(hinted_node_id):
                # Look for softmax before this node
                if i > 0:
                    prev_node = nodes_list[i-1]
                    if 'softmax' in prev_node.operation.lower():
                        pattern_nodes.insert(0, str(prev_node.id))
                        # Look for matmul before softmax
                        if i > 1:
                            prev_prev = nodes_list[i-2]
                            if 'matmul' in prev_prev.operation.lower():
                                pattern_nodes.insert(0, str(prev_prev.id))
                                return PatternMatch(
                                    pattern_name=self.name,
                                    confidence=0.85,
                                    matched_nodes=pattern_nodes,
                                    operation_sequence=['matmul', 'softmax', 'matmul'],
                                    optimization_hints={
                                        'can_use_flash_attention': True,
                                        'supports_kv_cache': True
                                    },
                                    metadata={'detection_method': 'metadata_hints_expanded'}
                                )
                break
        
        return None


class AdvancedVisionPattern(PatternPlugin):
    """Advanced vision pattern detection for CNNs and Vision Transformers."""

    @property
    def name(self) -> str:
        return "vision"

    # Hierarchical index metadata
    expected_operations = frozenset({
        "conv2d", "conv1d", "conv3d", "relu",
        "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d"
    })
    min_nodes = 3
    max_nodes = 500
    allows_cycles = False
    max_fanout = 20

    @track_performance
    def match(self, graph) -> Optional[PatternMatch]:
        """Detect vision patterns including CNNs and ViTs.
        
        Uses multiple strategies:
        1. Metadata hints (fast, for LazyDAG graphs)
        2. NetworkX subgraph matching (accurate, for all graph types)
        """
        # Strategy 1: Check for metadata hints (fast path for LazyDAG)
        conv_nodes = []
        for node in graph.nodes():
            if hasattr(node, 'metadata'):
                metadata = getattr(node, 'metadata', {})
                if isinstance(metadata, dict):
                    hints = metadata.get('pattern_hints', {})
                    if hints.get('likely_pattern') in ['conv_bn_relu', 'convolution']:
                        conv_nodes.append(node.id)
        
        # If we have metadata hints, return a pattern match
        if conv_nodes:
            return PatternMatch(
                pattern_name=self.name,
                confidence=0.85,
                matched_nodes=[str(nid) for nid in conv_nodes],
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
        
        # Strategy 2: NetworkX subgraph matching (accurate, works for all graphs)
        # Use NetworkX for pattern analysis
        G = graph_to_networkx(graph)
        
        # Find convolutional patterns
        conv_matches = find_conv_activation_pattern(G)
        
        # Find attention patterns (for Vision Transformers)
        attention_matches = find_attention_pattern(G)
        
        # Analyze for vision-specific operations
        vision_ops = {"aten::conv2d", "aten::conv1d", "aten::conv3d", 
                     "aten::max_pool2d", "aten::avg_pool2d", "aten::adaptive_avg_pool2d"}
        
        vision_node_count = sum(1 for _, data in G.nodes(data=True) 
                               if data.get('operation') in vision_ops)
        
        total_nodes = len(G.nodes())
        vision_ratio = vision_node_count / total_nodes if total_nodes > 0 else 0
        
        # Scoring
        conv_score = min(len(conv_matches) * 0.4, 1.0)
        vision_ops_score = vision_ratio * 0.8
        attention_score = min(len(attention_matches) * 0.2, 0.6)  # ViT component
        
        total_score = conv_score + vision_ops_score + attention_score
        
        if total_score > 0.8:
            confidence = min(total_score, 0.95)
            matched_nodes = []
            
            # Collect matched nodes
            for match in conv_matches[:5]:
                matched_nodes.extend(match.values())
            for match in attention_matches[:2]:
                matched_nodes.extend(match.values())
            
            return PatternMatch(
                pattern_name=self.name,
                confidence=confidence,
                matched_nodes=list(set(matched_nodes)),
                operation_sequence=["conv2d", "relu", "pool"] if conv_matches else ["conv2d", "relu"],
                optimization_hints={
                    "can_use_cudnn": True,
                    "supports_tensor_core": True,
                    "can_fuse_conv_bn": True,
                    "can_fuse_conv_activation": True,
                    "vision_transformer": len(attention_matches) > 0
                },
                metadata={
                    "conv_blocks": len(conv_matches),
                    "attention_blocks": len(attention_matches),
                    "vision_ops_ratio": vision_ratio,
                    "semantic_role": "vision_encoder" if len(attention_matches) == 0 else "vit_encoder"
                }
            )
        
        # Fallback for basic CNN patterns
        if conv_matches or vision_ratio > 0.3:
            return PatternMatch(
                pattern_name=self.name,
                confidence=0.72,
                matched_nodes=[list(match.values())[0] for match in conv_matches[:1]],
                operation_sequence=["conv2d", "relu"],
                optimization_hints={"can_fuse_conv_activation": True},
                metadata={"semantic_role": "vision_block"}
            )
        
        return None


class RecSysPattern(PatternPlugin):
    """Recommendation system pattern detection."""

    @property
    def name(self) -> str:
        return "recsys"

    # Hierarchical index metadata
    expected_operations = frozenset({
        "embedding", "embedding_bag", "gather",
        "index_select", "cat", "linear", "add"
    })
    min_nodes = 4
    max_nodes = 200
    allows_cycles = False
    max_fanout = 30

    @track_performance
    def match(self, graph) -> Optional[PatternMatch]:
        """Detect RecSys patterns: embeddings + MLPs."""
        # Use NetworkX for pattern analysis
        G = graph_to_networkx(graph)
        
        # Find embedding patterns
        embedding_matches = find_embedding_pattern(G)
        
        # Find MLP patterns
        mlp_matches = find_mlp_pattern(G)
        
        # Look for RecSys-specific operations
        recsys_ops = {"aten::embedding", "aten::embedding_bag", "aten::gather", 
                     "aten::index_select", "aten::cat"}
        
        recsys_node_count = sum(1 for _, data in G.nodes(data=True) 
                               if data.get('operation') in recsys_ops)
        
        total_nodes = len(G.nodes())
        recsys_ratio = recsys_node_count / total_nodes if total_nodes > 0 else 0
        
        # Scoring
        embedding_score = min(len(embedding_matches) * 0.5, 1.0)
        mlp_score = min(len(mlp_matches) * 0.3, 0.8)
        recsys_ops_score = recsys_ratio * 0.7
        
        total_score = embedding_score + mlp_score + recsys_ops_score
        
        if total_score > 0.7:
            confidence = min(total_score, 0.95)
            matched_nodes = []
            
            # Collect matched nodes
            for match in embedding_matches[:3]:
                matched_nodes.extend(match.values())
            for match in mlp_matches[:2]:
                matched_nodes.extend(match.values())
            
            return PatternMatch(
                pattern_name=self.name,
                confidence=confidence,
                matched_nodes=list(set(matched_nodes)),
                operation_sequence=["embedding", "mlp_layer", "output"],
                optimization_hints={
                    "can_fuse_embedding_table_lookup": True,
                    "supports_sparse_gradients": True,
                    "can_parallelize_embedding_lookups": True
                },
                metadata={
                    "embedding_blocks": len(embedding_matches),
                    "mlp_blocks": len(mlp_matches),
                    "recsys_ops_ratio": recsys_ratio,
                    "semantic_role": "recommendation_model"
                }
            )
        
        return None


class MultiModalPattern(PatternPlugin):
    """Multi-modal pattern detection for models combining vision and language."""

    @property
    def name(self) -> str:
        return "multimodal"

    # Hierarchical index metadata
    expected_operations = frozenset({
        "conv2d", "matmul", "softmax", "linear",
        "cat", "add", "embedding", "relu"
    })
    min_nodes = 8
    max_nodes = 1000
    allows_cycles = False
    max_fanout = 50

    @track_performance
    def match(self, graph) -> Optional[PatternMatch]:
        """Detect multi-modal patterns combining vision and language components."""
        # Convert to NetworkX for pattern analysis (handles all graph types)
        G = graph_to_networkx(graph)
        
        # Find vision patterns
        conv_matches = find_conv_activation_pattern(G)
        
        # Find language/attention patterns
        attention_matches = find_attention_pattern(G)
        
        # Find MLP patterns (fusion layers)
        mlp_matches = find_mlp_pattern(G)
        
        # Look for multi-modal specific operations
        vision_ops = {"aten::conv2d", "aten::max_pool2d", "aten::avg_pool2d"}
        language_ops = {"aten::embedding", "aten::linear"}
        fusion_ops = {"aten::cat", "aten::add", "aten::mul"}  # Common fusion operations
        
        vision_count = sum(1 for _, data in G.nodes(data=True) 
                          if data.get('operation') in vision_ops)
        language_count = sum(1 for _, data in G.nodes(data=True) 
                            if data.get('operation') in language_ops)
        fusion_count = sum(1 for _, data in G.nodes(data=True) 
                          if data.get('operation') in fusion_ops)
        
        total_nodes = len(G.nodes())
        if total_nodes == 0:
            return None
        
        # Multi-modal requires both vision and language components
        has_vision = (vision_count > 0 or len(conv_matches) > 0)
        has_language = (language_count > 0 or len(attention_matches) > 0)
        has_fusion = fusion_count > 0
        
        if has_vision and has_language:
            # Scoring based on component balance and fusion
            vision_score = min(len(conv_matches) * 0.3, 0.8)
            language_score = min(len(attention_matches) * 0.3, 0.8)
            fusion_score = min(fusion_count / total_nodes, 0.4)
            mlp_score = min(len(mlp_matches) * 0.2, 0.6)
            
            total_score = vision_score + language_score + fusion_score + mlp_score
            
            if total_score > 0.7:
                confidence = min(total_score, 0.95)
                matched_nodes = []
                
                # Collect representative nodes from each modality
                for match in conv_matches[:2]:
                    matched_nodes.extend(match.values())
                for match in attention_matches[:2]:
                    matched_nodes.extend(match.values())
                for match in mlp_matches[:1]:
                    matched_nodes.extend(match.values())
                
                return PatternMatch(
                    pattern_name=self.name,
                    confidence=confidence,
                    matched_nodes=list(set(matched_nodes)),
                    operation_sequence=["conv2d", "matmul", "fusion"],
                    optimization_hints={
                        "can_pipeline_modalities": True,
                        "supports_async_gpu_compute": True,
                        "fusion_opportunity": has_fusion
                    },
                    metadata={
                        "vision_blocks": len(conv_matches),
                        "language_blocks": len(attention_matches),
                        "fusion_operations": fusion_count,
                        "semantic_role": "multimodal_encoder"
                    }
                )
        
        return None


# Simple residual block pattern: conv -> bn/norm? -> relu -> conv -> add
class ResidualBlockPattern(PatternPlugin):
    @property
    def name(self) -> str:
        return "residual_block"

    # Hierarchical index metadata
    expected_operations = frozenset({
        "conv2d", "add", "relu"
    })
    min_nodes = 3
    max_nodes = 20
    allows_cycles = False
    max_fanout = 10

    @track_performance
    def match(self, graph) -> Optional[PatternMatch]:
        G = graph_to_networkx(graph)
        # Look for nodes with operation 'aten::add' having two parents, and parents include a conv path
        for node_id in G.nodes:
            if G.nodes[node_id].get('operation') != 'aten::add':
                continue
            preds = list(G.predecessors(node_id))
            if len(preds) < 2:
                continue
            # Heuristic: one predecessor should have a conv ancestor within 3 hops
            def has_conv_ancestor(n, depth=3):
                if depth == 0:
                    return False
                for p in G.predecessors(n):
                    if G.nodes[p].get('operation') == 'aten::conv2d':
                        return True
                    if has_conv_ancestor(p, depth-1):
                        return True
                return False
            if any(has_conv_ancestor(p) for p in preds):
                return PatternMatch(
                    pattern_name=self.name,
                    confidence=0.8,
                    matched_nodes=[node_id],
                    operation_sequence=["conv2d", "conv2d", "add"],
                    optimization_hints={
                        "can_fuse_residual_add": True,
                        "supports_in_place_add": True
                    },
                    metadata={"semantic_role": "residual_block"}
                )
        return None


# Performance tracking utilities
def get_pattern_performance_stats(pattern_class) -> Dict[str, float]:
    """Get performance statistics for a pattern class."""
    if hasattr(pattern_class.match, '_performance_stats'):
        stats = pattern_class.match._performance_stats
        return {
            "avg_latency": sum(stats) / len(stats),
            "max_latency": max(stats),
            "min_latency": min(stats),
            "call_count": len(stats)
        }
    return {"avg_latency": 0.0, "max_latency": 0.0, "min_latency": 0.0, "call_count": 0}
