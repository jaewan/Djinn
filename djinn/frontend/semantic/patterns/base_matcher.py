"""
Pattern DTO for backward compatibility.

NOTE: PatternMatcher and PatternRegistry classes have been removed.
Only Pattern class is kept as an internal DTO for PhaseDetector and AnnotatedGraph.
This will be migrated to use MatchedPattern in a future refactoring.
"""

import warnings
from dataclasses import dataclass
from typing import List, Dict, Any


# Track if deprecation warning has been shown (once per process)
_pattern_deprecation_shown = False


@dataclass
class Pattern:
    """
    Represents a detected pattern in the computation graph.
    
    DEPRECATED: This is an internal DTO used for backward compatibility.
    New code should use MatchedPattern from workload.py.
    This class will be removed once PhaseDetector and AnnotatedGraph are migrated.
    
    Migration: Use MatchedPattern from djinn.frontend.semantic.workload instead.
    """
    name: str
    nodes: List[Any]  # List of GraphNode dicts with 'id' field
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Emit deprecation warning on first instantiation."""
        global _pattern_deprecation_shown
        if not _pattern_deprecation_shown:
            warnings.warn(
                "Pattern class is deprecated and will be removed in a future version. "
                "Use MatchedPattern from djinn.frontend.semantic.workload instead. "
                "See docs/REFACTORING_MIGRATION_GUIDE.md for migration instructions.",
                DeprecationWarning,
                stacklevel=3
            )
            _pattern_deprecation_shown = True
