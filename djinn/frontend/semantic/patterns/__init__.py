"""Pattern matching for computation graphs.

NOTE: PatternMatcher classes are deprecated and unused. Only Pattern class
is kept for backward compatibility with PhaseDetector and AnnotatedGraph.
"""

from .base_matcher import Pattern

# PatternMatcher classes are deprecated and unused - removed auto-registration
# Pattern class is still needed for PhaseDetector and AnnotatedGraph compatibility

__all__ = [
    'Pattern',
]
