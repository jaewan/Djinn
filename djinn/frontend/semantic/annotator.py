"""
Graph annotator: Combines all semantic analysis to produce annotated graph.

Main public API for semantic analysis with content-addressed caching.
"""

import logging
import hashlib
from typing import Optional, Dict, List
from .pattern_registry import PatternRegistry
from .phase_detector import PhaseDetector, PhaseAnnotator
from ...scheduler.core.cost_estimator import GraphCostEstimator
from .metadata_registry import (
    get_metadata_registry, NodeMetadata, MetadataRegistry
)
from .cache import SemanticAnnotatorCache
from ...core.types import MatchingMode, ExecutionPhase

logger = logging.getLogger(__name__)


class SemanticAnnotator:
    """
    Annotates computation graphs with semantic information.

    Combines:
    1. Pattern detection (attention, convolution, KV cache)
    2. Phase detection (prefill, decode, forward)
    3. Cost estimation (FLOPs, memory, intensity)
    4. Metadata storage

    Uses content-addressed caching for optimal performance.
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_dir: str = ".genie_cache",
        max_memory_mb: int = 500
    ):
        # Use the improved PatternRegistry with hierarchical indexing
        from .pattern_registry import PatternRegistry
        self.pattern_registry = PatternRegistry()
        self.phase_detector = PhaseDetector(self.pattern_registry)
        self.phase_annotator = PhaseAnnotator(self.phase_detector)
        self.cost_estimator = GraphCostEstimator()
        self.metadata_registry = get_metadata_registry()

        # Content-addressed cache for full analysis results
        self._pattern_cache: Dict[str, List] = {}
        self._graph_cache: Dict[str, 'AnnotatedGraph'] = {}
        self._cache_size_limit = 1000  # Bound memory usage

        # Initialize legacy cache for backward compatibility
        self.legacy_cache = SemanticAnnotatorCache(
            max_memory_mb=max_memory_mb,
            cache_dir=cache_dir
        ) if enable_cache else None

        # Load persisted cache if available
        if self.legacy_cache:
            self.legacy_cache.load()

        # Metrics for content-addressed cache
        self._cache_hits = 0
        self._cache_misses = 0
    
    def annotate(self, graph) -> 'AnnotatedGraph':
        """
        Fully annotate a computation graph with content-addressed caching.

        Args:
            graph: Unified Graph (FX or LazyDAG)

        Returns:
            AnnotatedGraph with semantic metadata
        """
        # Compute content hash (structural + operations + shapes)
        graph_key = self._compute_content_hash(graph)

        # Check content-addressed cache first
        if graph_key in self._graph_cache:
            self._cache_hits += 1
            logger.debug(f"Content cache hit: {graph_key[:8]}...")
            return self._graph_cache[graph_key]

        self._cache_misses += 1

        # Full analysis (cache miss)
        result = self._analyze_graph(graph)

        # Cache result (with LRU eviction)
        self._add_to_cache(graph_key, result)

        return result

    def _compute_content_hash(self, graph) -> str:
        """
        Compute structure-based hash of graph.

        Design: Hash structure + operations + shapes (not tensor values).
        This makes structurally identical graphs hit cache even with different tensor data.
        """
        hasher = hashlib.blake2b(digest_size=16)  # Fast, good distribution

        # Hash structure (edges by node relationships)
        node_inputs = []
        for node in graph.nodes():
            # ✅ FIX: Convert all IDs to strings to avoid type comparison errors
            # Handle both node objects and string IDs in inputs
            input_ids = []
            for inp in node.inputs:
                if isinstance(inp, str):
                    input_ids.append(str(inp))
                else:
                    input_ids.append(str(getattr(inp, 'id', id(inp))))
            input_ids = sorted(input_ids)
            node_inputs.append((str(getattr(node, 'id', id(node))), node.operation, tuple(input_ids)))
        node_inputs.sort()
        hasher.update(str(node_inputs).encode())

        # Hash operations sequence
        ops = sorted([
            (str(getattr(node, 'id', id(node))), getattr(node, 'operation', 'unknown'))
            for node in graph.nodes()
        ])
        hasher.update(str(ops).encode())

        # Hash shapes (if available)
        shapes = sorted([
            (str(getattr(node, 'id', id(node))), str(getattr(node, 'shape', None)))
            for node in graph.nodes()
        ])
        hasher.update(str(shapes).encode())

        return hasher.hexdigest()

    def _add_to_cache(self, key: str, result: 'AnnotatedGraph'):
        """Add to cache with LRU eviction."""
        if len(self._graph_cache) >= self._cache_size_limit:
            # Evict oldest (simple FIFO for now; could use LRU)
            oldest_key = next(iter(self._graph_cache))
            del self._graph_cache[oldest_key]
            logger.debug(f"Cache eviction: {oldest_key[:8]}...")

        self._graph_cache[key] = result

    def _analyze_graph(self, graph) -> 'AnnotatedGraph':
        """Full semantic analysis without caching."""
        logger.info(f"Starting semantic analysis for {len(list(graph.nodes()))} nodes")

        # Pattern matching with hierarchical index and early termination
        pattern_result = self.pattern_registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)
        if not pattern_result.is_ok:
            logger.warning(f"Pattern matching failed: {pattern_result.error}")
            # Convert to old format for backward compatibility
            patterns = {}
        else:
            matches = pattern_result.unwrap()
            # Convert to old format expected by phase detector
            patterns = self._convert_matches_to_patterns(matches, graph)

        logger.debug(f"Detected patterns: {list(patterns.keys())}")

        # Phase detection
        phases = self.phase_detector.detect_phases(graph, patterns)
        logger.debug(f"Phase distribution: {self._count_phases(phases)}")

        # Cost estimation and metadata
        return self._compute_costs(graph, patterns, phases)

    def _convert_matches_to_patterns(self, matches, graph):
        """
        Convert new PatternMatch format to old Pattern format.

        CRITICAL FIX: Extract matched_nodes from PatternMatch objects.
        These nodes are required by the scheduler for co-location decisions.
        """
        from .patterns.base_matcher import Pattern

        patterns = {}
        for match in matches:
            # Extract node IDs from PatternMatch
            nodes = []
            if hasattr(match, 'matched_nodes') and match.matched_nodes:
                # matched_nodes contains node ID strings from NetworkX matching
                # Convert to dict format expected by scheduler
                nodes = [{'id': str(node_id)} for node_id in match.matched_nodes]
            else:
                # Fallback: If matched_nodes is empty, extract nodes by operation type
                # This handles cases where NetworkX matching doesn't work properly
                nodes = self._extract_nodes_by_operations(match, graph)

            # Create Pattern with actual nodes
            pattern = Pattern(
                name=match.pattern_name,
                nodes=nodes,  # ← FIXED: Now populated from match.matched_nodes or fallback
                metadata={
                    'confidence': match.confidence,
                    'optimization_hints': match.optimization_hints or {},
                    'metadata': match.metadata or {}
                }
            )

            # Group patterns by name (multiple instances of same pattern)
            patterns.setdefault(match.pattern_name, []).append(pattern)

        return patterns

    def _extract_nodes_by_operations(self, match, graph):
        """
        Improved fallback method to extract nodes by operation type when NetworkX matching fails.

        This method uses pattern-specific heuristics to avoid collecting unrelated nodes.
        It considers module hierarchy, operation sequences, and connectivity patterns.
        """
        nodes = []
        expected_ops = {
            'llm': ['aten::matmul', 'aten::softmax', 'aten::add', 'aten::transpose'],
            'vision': ['aten::conv2d', 'aten::conv1d', 'aten::conv3d', 'aten::max_pool2d', 'aten::avg_pool2d'],
            'multimodal': ['aten::matmul', 'aten::conv2d', 'aten::softmax'],
            'attention': ['aten::matmul', 'aten::softmax', 'aten::transpose'],
            'recurrent': ['aten::add', 'aten::sigmoid', 'aten::tanh', 'aten::mul'],
        }

        # Get expected operations for this pattern
        ops = expected_ops.get(match.pattern_name, [])

        if not ops:
            logger.warning(f"No expected operations defined for pattern {match.pattern_name}")
            return nodes

        # Pattern-specific extraction strategies
        if match.pattern_name == 'llm':
            nodes = self._extract_llm_nodes(graph, ops)
        elif match.pattern_name == 'vision':
            nodes = self._extract_vision_nodes(graph, ops)
        elif match.pattern_name == 'multimodal':
            nodes = self._extract_multimodal_nodes(graph, ops)
        elif match.pattern_name == 'attention':
            nodes = self._extract_attention_nodes(graph, ops)
        elif match.pattern_name == 'recurrent':
            nodes = self._extract_recurrent_nodes(graph, ops)
        else:
            # Generic fallback: use module hierarchy to group related operations
            nodes = self._extract_by_module_hierarchy(graph, ops, match.pattern_name)

        logger.info(f"Pattern {match.pattern_name} fallback: extracted {len(nodes)} nodes using improved heuristics")
        return nodes

    def _extract_llm_nodes(self, graph, ops):
        """Extract LLM-related nodes using module hierarchy and operation patterns."""
        nodes = []

        # Group nodes by module hierarchy to find transformer blocks, attention layers, etc.
        module_groups = {}
        for node in graph.nodes():
            if hasattr(node, 'operation') and node.operation in ops:
                module_path = getattr(node, 'metadata', {}).get('module_path', 'unknown')
                if module_path not in module_groups:
                    module_groups[module_path] = []
                module_groups[module_path].append(node)

        # For LLM patterns, look for attention blocks or transformer layers
        llm_indicators = ['attention', 'transformer', 'encoder', 'decoder', 'embedding']
        for module_path, group_nodes in module_groups.items():
            module_lower = module_path.lower()
            if any(indicator in module_lower for indicator in llm_indicators):
                nodes.extend([{'id': str(node.id)} for node in group_nodes])
                logger.debug(f"LLM pattern: included module {module_path} with {len(group_nodes)} nodes")

        return nodes

    def _extract_vision_nodes(self, graph, ops):
        """Extract vision-related nodes using module hierarchy."""
        nodes = []

        # Look for vision-specific modules
        vision_indicators = ['conv', 'vision', 'image', 'backbone', 'resnet', 'vgg', 'vit']
        module_groups = {}

        for node in graph.nodes():
            if hasattr(node, 'operation') and node.operation in ops:
                module_path = getattr(node, 'metadata', {}).get('module_path', 'unknown')
                if module_path not in module_groups:
                    module_groups[module_path] = []
                module_groups[module_path].append(node)

        # Include modules that contain vision indicators
        for module_path, group_nodes in module_groups.items():
            module_lower = module_path.lower()
            if any(indicator in module_lower for indicator in vision_indicators):
                nodes.extend([{'id': str(node.id)} for node in group_nodes])
                logger.debug(f"Vision pattern: included module {module_path} with {len(group_nodes)} nodes")

        return nodes

    def _extract_multimodal_nodes(self, graph, ops):
        """Extract multimodal nodes by finding cross-modal operations."""
        nodes = []

        # Look for fusion modules or cross-attention patterns
        fusion_indicators = ['fusion', 'cross', 'multimodal', 'merge', 'combine']

        for node in graph.nodes():
            if hasattr(node, 'operation') and node.operation in ops:
                module_path = getattr(node, 'metadata', {}).get('module_path', 'unknown')
                metadata = getattr(node, 'metadata', {})

                # Check for fusion indicators or cross-modal metadata
                module_lower = module_path.lower()
                if (any(indicator in module_lower for indicator in fusion_indicators) or
                    metadata.get('modality') == 'fusion' or
                    metadata.get('attention_type') == 'cross_attention'):
                    nodes.append({'id': str(node.id)})
                    logger.debug(f"Multimodal pattern: included node {node.id} in {module_path}")

        return nodes

    def _extract_attention_nodes(self, graph, ops):
        """Extract attention-related nodes."""
        nodes = []

        # Look for attention-specific modules
        attention_indicators = ['attention', 'attn', 'multihead', 'self_attn']

        for node in graph.nodes():
            if hasattr(node, 'operation') and node.operation in ops:
                module_path = getattr(node, 'metadata', {}).get('module_path', 'unknown')
                metadata = getattr(node, 'metadata', {})
                module_lower = module_path.lower()

                if (any(indicator in module_lower for indicator in attention_indicators) or
                    metadata.get('semantic_role') == 'attention' or
                    'attention' in metadata.get('attention_type', '')):
                    nodes.append({'id': str(node.id)})
                    logger.debug(f"Attention pattern: included node {node.id} in {module_path}")

        return nodes

    def _extract_recurrent_nodes(self, graph, ops):
        """Extract recurrent pattern nodes."""
        nodes = []

        # Look for recurrent modules
        recurrent_indicators = ['lstm', 'gru', 'rnn', 'recurrent']

        for node in graph.nodes():
            if hasattr(node, 'operation') and node.operation in ops:
                module_path = getattr(node, 'metadata', {}).get('module_path', 'unknown')
                metadata = getattr(node, 'metadata', {})
                module_lower = module_path.lower()

                if (any(indicator in module_lower for indicator in recurrent_indicators) or
                    metadata.get('semantic_role') == 'recurrent' or
                    metadata.get('temporal_dependency')):
                    nodes.append({'id': str(node.id)})
                    logger.debug(f"Recurrent pattern: included node {node.id} in {module_path}")

        return nodes

    def _extract_by_module_hierarchy(self, graph, ops, pattern_name):
        """Generic fallback using module hierarchy."""
        nodes = []

        # Group by module and include only the largest cohesive groups
        module_groups = {}
        for node in graph.nodes():
            if hasattr(node, 'operation') and node.operation in ops:
                module_path = getattr(node, 'metadata', {}).get('module_path', 'unknown')
                if module_path not in module_groups:
                    module_groups[module_path] = []
                module_groups[module_path].append(node)

        # Sort by group size and include only substantial groups (prevents including isolated nodes)
        sorted_groups = sorted(module_groups.items(), key=lambda x: len(x[1]), reverse=True)

        for module_path, group_nodes in sorted_groups[:3]:  # Top 3 largest groups
            if len(group_nodes) > 1:  # Only include groups with multiple related nodes
                nodes.extend([{'id': str(node.id)} for node in group_nodes])
                logger.debug(f"Generic pattern {pattern_name}: included module {module_path} with {len(group_nodes)} nodes")

        return nodes

    def _compute_topology(self, graph):
        """Compute topology-level analysis (structure only)."""
        logger.info(f"Computing topology analysis for {len(list(graph.nodes()))} nodes")

        # Use new pattern registry API
        pattern_result = self.pattern_registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)
        if pattern_result.is_ok:
            matches = pattern_result.unwrap()
            patterns = self._convert_matches_to_patterns(matches, graph)
        else:
            logger.warning(f"Pattern matching failed: {pattern_result.error}")
            patterns = {}

        logger.debug(f"Detected patterns: {list(patterns.keys())}")

        phases = self.phase_detector.detect_phases(graph, patterns)
        logger.debug(f"Phase distribution: {self._count_phases(phases)}")

        return (patterns, phases)
    
    def _compute_costs(self, graph, patterns, phases):
        """Compute shape-specific costs."""
        logger.info("Computing shape-specific costs...")
        
        for node in graph.nodes():
            phase_obj = phases.get(node.id)
            phase_value = phase_obj.value if hasattr(phase_obj, 'value') else str(phase_obj) if phase_obj else None

            # ✅ FIX: Use getattr() instead of .get() to support both dict and MetadataPlaceholder
            node_meta = getattr(node, 'metadata', {})
            metadata = NodeMetadata(
                node_id=node.id,
                phase=phase_value,
                semantic_role=getattr(node_meta, 'semantic_role', None),
                modality=getattr(node_meta, 'modality', None),
            )

            metadata.optimization_hints = getattr(node_meta, 'optimization_hints', {})

            self.metadata_registry.register_metadata(node.id, metadata)

        cost_results = self.cost_estimator.estimate_graph(graph)
        costs = cost_results['per_node']
        total_compute = cost_results['total_compute_flops']
        total_memory = cost_results['total_memory_bytes']
        total_data_movement = cost_results['total_data_movement_bytes']

        for node in graph.nodes():
            node_cost = costs.get(node.id)
            if node_cost:
                self.metadata_registry.update_metadata(node.id, {
                    'compute_flops': node_cost.compute_flops,
                    'memory_bytes': node_cost.memory_bytes,
                    'operational_intensity': node_cost.operational_intensity,
                    'data_movement_bytes': node_cost.data_movement_bytes,
                })

        graph_costs = {
            'per_node': costs,
            'total_compute_flops': total_compute,
            'total_memory_bytes': total_memory,
            'total_data_movement_bytes': total_data_movement,
            'mean_operational_intensity': (total_compute / total_memory if total_memory > 0 else 0),
        }

        logger.info(f"Cost estimation complete")
        
        return AnnotatedGraph(graph, patterns, phases, graph_costs, self.metadata_registry)

    def get_cache_stats(self) -> Optional[Dict]:
        """Get cache statistics including content-addressed cache."""
        stats = {
            'content_addressed_cache': {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_rate': (self._cache_hits / (self._cache_hits + self._cache_misses)
                           if (self._cache_hits + self._cache_misses) > 0 else 0.0),
                'cache_size': len(self._graph_cache),
                'memory_mb': self._estimate_cache_memory(),
            }
        }

        # Include legacy cache stats if available
        if self.legacy_cache:
            legacy_stats = self.legacy_cache.get_stats()
            stats['legacy_cache'] = legacy_stats

        return stats

    def _estimate_cache_memory(self) -> float:
        """Estimate cache memory usage (rough)."""
        import sys

        if not self._graph_cache:
            return 0.0

        # Sample one entry to estimate average size
        sample = next(iter(self._graph_cache.values()))
        sample_size = sys.getsizeof(sample)

        return (sample_size * len(self._graph_cache)) / (1024 * 1024)

    def save_cache(self):
        """Save cache to disk."""
        if self.cache:
            self.cache.save()

    def clear_cache(self):
        """Clear all caches."""
        if self.cache:
            self.cache.clear()

    def _count_phases(self, phases) -> dict:
        """Count nodes by phase."""
        counts = {}
        for phase in phases.values():
            phase_name = phase.value if hasattr(phase, 'value') else str(phase)
            counts[phase_name] = counts.get(phase_name, 0) + 1
        return counts


class AnnotatedGraph:
    """
    Computation graph with semantic annotations.
    
    Wrapper around a Graph that includes semantic metadata.
    """
    
    def __init__(self, base_graph, patterns, phases, costs, metadata_registry):
        self.base_graph = base_graph
        self.patterns = patterns
        self.phases = phases
        self.costs = costs
        self.metadata_registry = metadata_registry
    
    def nodes(self):
        """Get all annotated nodes."""
        return self.base_graph.nodes()
    
    def get_node(self, node_id):
        """Get node by ID."""
        return self.base_graph.get_node(node_id)
    
    def get_metadata(self, node_id) -> Optional[NodeMetadata]:
        """Get metadata for a node."""
        return self.metadata_registry.get_metadata(node_id)
    
    @property
    def backend_type(self):
        """Backend type (FX or lazy_dag)."""
        return self.base_graph.backend_type
    
    def summary(self) -> str:
        """Get human-readable summary of annotations."""
        lines = [
            f"AnnotatedGraph ({self.backend_type} backend)",
            f"  Total nodes: {len(list(self.nodes()))}",
            f"  Total compute: {self.costs['total_compute_flops']:.2e} FLOPs",
            f"  Total memory: {self.costs['total_memory_bytes']:.2e} bytes",
            f"  Mean intensity: {self.costs['mean_operational_intensity']:.2f} FLOPs/byte",
            f"Patterns detected:",
        ]
        
        for pattern_name, pattern_list in self.patterns.items():
            lines.append(f"  - {pattern_name}: {len(pattern_list)} occurrences")
        
        return "\n".join(lines)


def annotate_graph(graph) -> AnnotatedGraph:
    """
    Annotate a computation graph with semantic information.
    
    Uses advanced two-level caching for performance.
    
    Args:
        graph: Unified Graph (from djinn.get_graph())
    
    Returns:
        AnnotatedGraph with full semantic metadata
    """
    annotator = SemanticAnnotator(enable_cache=True)
    return annotator.annotate(graph)
