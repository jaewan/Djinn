# Frontend Semantic Analysis Package
#
# This package contains all semantic analysis and annotation components
# that enrich computation graphs with semantic information before scheduling.

from .analyzer import SemanticAnalyzer  # noqa: F401
from .annotator import (  # noqa: F401
    SemanticAnnotator,
    AnnotatedGraph,
    annotate_graph,
)
from .metadata_registry import (  # noqa: F401
    MetadataRegistry,
    NodeMetadata,
    get_metadata_registry,
)
from .pattern_registry import PatternRegistry  # noqa: F401
from .pattern_matcher import (  # noqa: F401
    IPatternMatcher,
    NetworkXPatternMatcher,
    SimplifiedPatternMatcher,
    CompositePatternMatcher,
    create_pattern_matcher,
    get_default_pattern_matcher,
)
from .phase_detector import (  # noqa: F401
    ExecutionPhase,
    PhaseDetector,
    PhaseAnnotator,
)
from .workload import (  # noqa: F401
    WorkloadProfile,
    WorkloadType,
    WorkloadClassifier,
    MatchedPattern,
    ExecutionPlan,
    PlanFragment,
)
from .hooks import HookManager  # noqa: F401
from .graph_utils import analyze_operations_advanced, track_performance  # noqa: F401

# Re-export patterns for convenience
from .patterns import *  # noqa: F401, F403

__all__ = [
    'SemanticAnalyzer',
    'SemanticAnnotator',
    'AnnotatedGraph',
    'annotate_graph',
    'MetadataRegistry',
    'NodeMetadata',
    'get_metadata_registry',
    'PatternRegistry',
    'IPatternMatcher',
    'NetworkXPatternMatcher',
    'SimplifiedPatternMatcher',
    'CompositePatternMatcher',
    'create_pattern_matcher',
    'get_default_pattern_matcher',
    'ExecutionPhase',
    'PhaseDetector',
    'PhaseAnnotator',
    'WorkloadProfile',
    'WorkloadType',
    'WorkloadClassifier',
    'MatchedPattern',
    'ExecutionPlan',
    'PlanFragment',
    'HookManager',
    'analyze_operations_advanced',
    'track_performance',
]
