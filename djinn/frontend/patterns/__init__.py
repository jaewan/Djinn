"""
Pattern recognition module for Djinn.

This module provides semantic pattern recognition for identifying
common workload patterns (LLM, Vision, RecSys, MultiModal) in computation graphs.
"""

from .base import PatternPlugin, PatternMatch  # noqa: F401
from .advanced_patterns import (  # noqa: F401
    AdvancedLLMPattern,
    AdvancedVisionPattern,
    RecSysPattern,
    MultiModalPattern,
    ResidualBlockPattern,
)

__all__ = [
    "PatternPlugin",
    "PatternMatch",
    "AdvancedLLMPattern",
    "AdvancedVisionPattern",
    "RecSysPattern",
    "MultiModalPattern",
    "ResidualBlockPattern",
]