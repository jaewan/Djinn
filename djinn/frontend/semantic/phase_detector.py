"""
Phase detection for execution phases.

Identifies phases like:
- LLM prefill (parallel attention over sequence)
- LLM decode (sequential generation with KV cache)
- Forward propagation
- Backward propagation (future)
"""

from enum import Enum
from typing import Dict, List, Any
import logging
from ...core.types import ExecutionPhase, MatchingMode

logger = logging.getLogger(__name__)


class PhaseDetector:
    """
    Detects execution phases in computation graphs.
    
    Uses pattern matches and heuristics to classify operations into phases.
    """
    
    def __init__(self, pattern_registry):
        self.pattern_registry = pattern_registry
    
    def detect_phases(self, graph, patterns=None) -> Dict[str, ExecutionPhase]:
        """
        Detect phases for all nodes in graph.

        Args:
            graph: Computation graph
            patterns: Pre-computed patterns (optional, for compatibility)

        Returns:
            Dict mapping node_id → ExecutionPhase
        """
        phases = {}

        # If patterns not provided, get them using new API
        if patterns is None:
            pattern_result = self.pattern_registry.match_patterns(graph, mode=MatchingMode.EXHAUSTIVE)
            if pattern_result.is_ok:
                matches = pattern_result.unwrap()
                patterns = self._convert_matches_to_patterns(matches)
            else:
                logger.warning(f"Pattern matching failed in phase detection: {pattern_result.error}")
                patterns = {}

        # Phase detection logic using patterns...

        # Phase 1: Detect KV cache (indicates decode)
        if "kv_cache" in patterns and patterns["kv_cache"]:
            decode_nodes = self._mark_decode_phase(patterns["kv_cache"], graph)
            phases.update(decode_nodes)
        
        # Phase 2: Detect parallel attention (indicates prefill)
        prefill_nodes = self._mark_prefill_phase(patterns.get("attention", []), graph)
        phases.update(prefill_nodes)
        
        # Phase 3: Mark remaining as forward
        for node in graph.nodes():
            if node.id not in phases:
                phases[node.id] = ExecutionPhase.FORWARD
        
        return phases
    
    def _mark_decode_phase(self, kv_cache_patterns, graph) -> Dict[str, ExecutionPhase]:
        """Mark nodes in decode phase."""
        phases = {}
        
        for pattern in kv_cache_patterns:
            for node_dict in pattern.nodes:
                # Handle both node objects and node dictionaries
                if hasattr(node_dict, 'id'):
                    node_id = node_dict.id
                elif isinstance(node_dict, dict) and 'id' in node_dict:
                    node_id = node_dict['id']
                else:
                    continue
                phases[node_id] = ExecutionPhase.LLM_DECODE
                
                # Also mark attention nodes connected to KV cache as decode
                # (they're part of the decode loop)
        
        return phases
    
    def _mark_prefill_phase(self, attention_patterns, graph) -> Dict[str, ExecutionPhase]:
        """Mark nodes in prefill phase."""
        phases = {}
        
        for pattern in attention_patterns:
            # Check if this is parallel attention (prefill)
            if self._is_parallel_attention(pattern):
                for node_dict in pattern.nodes:
                    # Handle both node objects and node dictionaries
                    if hasattr(node_dict, 'id'):
                        node_id = node_dict.id
                    elif isinstance(node_dict, dict) and 'id' in node_dict:
                        node_id = node_dict['id']
                    else:
                        continue
                    phases[node_id] = ExecutionPhase.LLM_PREFILL
        
        return phases
    
    def _is_parallel_attention(self, pattern) -> bool:
        """Check if attention processes multiple positions in parallel."""
        # Heuristic: If batch size in sequence dimension > 1
        return True  # Simplified

    def _convert_matches_to_patterns(self, matches):
        """Convert new MatchedPattern format to old Pattern format."""
        from .patterns.base_matcher import Pattern

        patterns = {}
        for match in matches:
            # Extract node IDs from MatchedPattern
            nodes = []
            if hasattr(match, 'matched_nodes') and match.matched_nodes:
                # Convert to dict format expected by scheduler
                nodes = [{'id': str(node_id)} for node_id in match.matched_nodes]

            # Create a simple Pattern object from MatchedPattern
            pattern = Pattern(
                name=match.pattern_name,
                nodes=nodes,  # ✅ FIXED: Extract nodes from match
                metadata={
                    'confidence': match.confidence,
                    'optimization_hints': match.optimization_hints or {},
                    'metadata': match.metadata or {}
                }
            )
            patterns[match.pattern_name] = [pattern]

        return patterns


class PhaseAnnotator:
    """
    Annotates graph nodes with phase information.
    
    Stores phase annotations in node metadata.
    """
    
    def __init__(self, phase_detector):
        self.phase_detector = phase_detector
    
    def annotate(self, graph):
        """
        Annotate graph nodes with phase information.
        
        Modifies node metadata in-place.
        """
        phases = self.phase_detector.detect_phases(graph)
        
        for node in graph.nodes():
            phase = phases.get(node.id, ExecutionPhase.UNKNOWN)
            if not hasattr(node, 'metadata'):
                node.metadata = {}
            if 'phase' not in node.metadata:
                node.metadata['phase'] = phase


# Backward compatibility shims for existing code
def get_phase_detector():
    """Singleton phase detector (backward compatibility)."""
    # Return a minimal dummy implementation for backward compatibility
    class DummyPhaseDetector:
        def detect_phase(self, *args, **kwargs):
            return ExecutionPhase.FORWARD
    
    return DummyPhaseDetector()


class PhaseAwareHook:
    """Hook for phase detection (backward compatibility)."""
    def __init__(self, module_name=""):
        self.module_name = module_name
    
    def __call__(self, module, inputs, outputs):
        return outputs
