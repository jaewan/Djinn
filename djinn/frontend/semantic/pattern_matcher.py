"""Pattern matching service interface and implementations.

This module implements Refactoring #5 (Extract Pattern Matching Service).
It provides dependency injection for pattern matchers, allowing different
implementations to be plugged in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import torch.fx as fx
import logging

from djinn.core.graph_interface import Graph as DjinnGraph
from djinn.core.exceptions import Result, PatternMatchError
from .workload import MatchedPattern

logger = logging.getLogger(__name__)


class IPatternMatcher(ABC):
    """Abstract interface for pattern matching services.
    
    This interface allows different pattern matching implementations to be
    plugged into SemanticAnalyzer via dependency injection.
    
    NOTE: This wraps PatternPlugin instances internally. The PatternPlugin
    interface is the primary interface for pattern implementations. IPatternMatcher
    provides a service layer abstraction for dependency injection.
    
    Implementations:
    - NetworkXPatternMatcher: Uses NetworkX for graph pattern matching (wraps PatternRegistry)
    - SimplifiedPatternMatcher: Fast heuristic-based matching
    - CompositePatternMatcher: Combines multiple matchers
    - TorchDynamoPatternMatcher: (Future) Uses PyTorch Dynamo patterns
    """
    
    @abstractmethod
    def match_patterns(self, graph: DjinnGraph) -> Result[List[MatchedPattern]]:
        """Match patterns in the given graph.
        
        Args:
            graph: Unified Graph interface (DjinnGraph) to analyze
            
        Returns:
            Result[List[MatchedPattern]]: Matched patterns or error
        """
        pass
    
    @abstractmethod
    def get_performance_report(self) -> dict:
        """Get performance statistics for pattern matching.
        
        Returns:
            Dictionary with performance metrics
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this pattern matcher implementation."""
        pass


class NetworkXPatternMatcher(IPatternMatcher):
    """Pattern matcher using NetworkX for graph analysis.
    
    This is the default implementation that uses NetworkX for subgraph
    isomorphism matching. It wraps the existing PatternRegistry.
    """
    
    def __init__(self, pattern_registry=None):
        """Initialize NetworkX pattern matcher.
        
        Args:
            pattern_registry: Optional PatternRegistry instance. If None,
                            creates a new one with default patterns.
        """
        from .pattern_registry import PatternRegistry
        
        self._pattern_registry = pattern_registry or PatternRegistry()
        self._name = "NetworkX"
    
    def match_patterns(self, graph: DjinnGraph) -> Result[List[MatchedPattern]]:
        """Match patterns using NetworkX-based registry.
        
        Args:
            graph: Unified Graph interface (DjinnGraph)
            
        Returns:
            Result[List[MatchedPattern]]: Matched patterns or error
        """
        # ✅ PHASE 3: Pattern registry needs to be updated to accept DjinnGraph
        # For now, convert to NetworkX which the registry expects
        from .graph_utils import graph_to_networkx
        nx_graph = graph_to_networkx(graph)
        # Pass through to registry (registry will need updating, but this works for now)
        return self._pattern_registry.match_patterns(graph)
    
    def get_performance_report(self) -> dict:
        """Get performance report from pattern registry.
        
        Returns:
            Dictionary with performance statistics
        """
        return self._pattern_registry.get_performance_report()
    
    @property
    def name(self) -> str:
        """Get matcher name."""
        return self._name
    
    def register_pattern(self, pattern) -> None:
        """Register additional pattern with the registry.
        
        Args:
            pattern: PatternPlugin to register
        """
        self._pattern_registry.register_pattern(pattern)


class SimplifiedPatternMatcher(IPatternMatcher):
    """Simplified pattern matcher for quick testing.
    
    This matcher uses simple heuristics instead of complex graph matching.
    Useful for testing or when full pattern matching is too slow.
    """
    
    def __init__(self):
        """Initialize simplified matcher."""
        self._name = "Simplified"
        self._match_count = 0
        self._total_time = 0.0
    
    def match_patterns(self, graph: DjinnGraph) -> Result[List[MatchedPattern]]:
        """Match patterns using simple heuristics.
        
        Args:
            graph: Unified Graph interface (DjinnGraph)
            
        Returns:
            Result[List[MatchedPattern]]: Matched patterns based on heuristics
        """
        import time
        from .workload import WorkloadType
        
        start = time.perf_counter()
        matches = []
        
        try:
            # ✅ PHASE 3: Extract operations from unified Graph interface
            ops = [node.operation for node in graph.nodes()]
            
            # Simple heuristics for pattern detection
            op_str = ' '.join(ops).lower()
            
            # LLM pattern: matmul + softmax
            if 'matmul' in op_str or 'softmax' in op_str:
                matches.append(MatchedPattern(
                    pattern_name='llm_attention_simplified',
                    confidence=0.7,
                    metadata={'workload_type': WorkloadType.LLM}
                ))
            
            # Vision pattern: conv + relu
            if 'conv' in op_str or 'relu' in op_str:
                matches.append(MatchedPattern(
                    pattern_name='vision_conv_simplified',
                    confidence=0.7,
                    metadata={'workload_type': WorkloadType.VISION}
                ))
            
            elapsed = time.perf_counter() - start
            self._match_count += 1
            self._total_time += elapsed
            
            logger.debug(f"Simplified matcher found {len(matches)} patterns in {elapsed*1000:.1f}ms")
            
            return Result.ok(matches)
            
        except Exception as e:
            return Result.err(PatternMatchError(
                f"Simplified pattern matching failed: {e}",
                context={'graph_type': type(graph).__name__}
            ))
    
    def get_performance_report(self) -> dict:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = self._total_time / self._match_count if self._match_count > 0 else 0.0
        return {
            'matcher_name': self._name,
            'total_matches': self._match_count,
            'total_time': self._total_time,
            'avg_time_ms': avg_time * 1000
        }
    
    @property
    def name(self) -> str:
        """Get matcher name."""
        return self._name


class CompositePatternMatcher(IPatternMatcher):
    """Composite pattern matcher that tries multiple matchers.
    
    This matcher allows combining multiple pattern matching strategies.
    It tries each matcher in order and combines the results.
    """
    
    def __init__(self, matchers: List[IPatternMatcher]):
        """Initialize composite matcher.
        
        Args:
            matchers: List of pattern matchers to use
        """
        self._matchers = matchers
        self._name = f"Composite({','.join(m.name for m in matchers)})"
    
    def match_patterns(self, graph: Union[ComputationGraph, fx.GraphModule]) -> Result[List[MatchedPattern]]:
        """Match patterns using all matchers.
        
        Args:
            graph: Graph to analyze
            
        Returns:
            Result[List[MatchedPattern]]: Combined matched patterns
        """
        all_matches = []
        errors = []
        
        for matcher in self._matchers:
            result = matcher.match_patterns(graph)
            if result.is_ok:
                all_matches.extend(result.unwrap())
            else:
                errors.append(result.error)
                logger.debug(f"Matcher {matcher.name} failed: {result.error}")
        
        if all_matches:
            # Deduplicate by pattern name
            seen = set()
            unique_matches = []
            for match in all_matches:
                if match.pattern_name not in seen:
                    seen.add(match.pattern_name)
                    unique_matches.append(match)
            
            return Result.ok(unique_matches)
        elif errors:
            return Result.err(PatternMatchError(
                f"All {len(self._matchers)} matchers failed",
                context={'error_count': len(errors)}
            ))
        else:
            return Result.ok([])
    
    def get_performance_report(self) -> dict:
        """Get combined performance report.
        
        Returns:
            Dictionary with performance metrics from all matchers
        """
        reports = {}
        for matcher in self._matchers:
            reports[matcher.name] = matcher.get_performance_report()
        return {
            'matcher_name': self._name,
            'sub_matchers': reports
        }
    
    @property
    def name(self) -> str:
        """Get matcher name."""
        return self._name


# Factory function for easy creation
def create_pattern_matcher(
    matcher_type: str = 'networkx',
    **kwargs
) -> IPatternMatcher:
    """Factory function to create pattern matchers.
    
    Args:
        matcher_type: Type of matcher ('networkx', 'simplified', 'composite')
        **kwargs: Additional arguments for the matcher
        
    Returns:
        IPatternMatcher instance
        
    Examples:
        >>> matcher = create_pattern_matcher('networkx')
        >>> matcher = create_pattern_matcher('simplified')
        >>> matcher = create_pattern_matcher('composite', 
        ...                                   matchers=[networkx, simplified])
    """
    matcher_type = matcher_type.lower()
    
    if matcher_type == 'networkx':
        return NetworkXPatternMatcher(**kwargs)
    elif matcher_type == 'simplified':
        return SimplifiedPatternMatcher()
    elif matcher_type == 'composite':
        matchers = kwargs.get('matchers', [])
        if not matchers:
            # Default: NetworkX + Simplified
            matchers = [NetworkXPatternMatcher(), SimplifiedPatternMatcher()]
        return CompositePatternMatcher(matchers)
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")


# Default matcher instance
def get_default_pattern_matcher() -> IPatternMatcher:
    """Get the default pattern matcher instance.
    
    Returns:
        NetworkXPatternMatcher with default patterns
    """
    return NetworkXPatternMatcher()
