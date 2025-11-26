"""
djinn/core/semantic_hints.py

Support for explicit semantic annotations (Phase 3).

Provides:
- SemanticHints dataclass for optimization hints
- session() context manager for hint propagation
- Integration with execution path
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .types import ExecutionPhase as CoreExecutionPhase

import logging
logger = logging.getLogger(__name__)


class Priority(Enum):
    """Request priority levels."""
    BACKGROUND = 0
    NORMAL = 1
    INTERACTIVE = 2
    REALTIME = 3


@dataclass
class SemanticHints:
    """
    Semantic hints for optimization.
    
    These hints guide Djinn's execution optimizations:
    - Execution phase: prefill, decode, vision, etc.
    - Priority: background, normal, interactive, realtime
    - KV cache size: estimated KV cache size for decode phase
    - Expected tokens: number of tokens expected in decode phase
    - Session ID: session identifier for stateful operations (e.g., KV cache persistence)
    """
    phase: Optional[CoreExecutionPhase] = None
    priority: Optional[Priority] = None
    kv_cache_size_mb: Optional[float] = None
    expected_tokens: Optional[int] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {}
        if self.phase:
            result['execution_phase'] = self.phase.value
        if self.priority:
            result['priority'] = self.priority.value
        if self.kv_cache_size_mb is not None:
            result['kv_cache_size_mb'] = self.kv_cache_size_mb
        if self.expected_tokens is not None:
            result['expected_tokens'] = self.expected_tokens
        if self.session_id:
            result['session_id'] = self.session_id
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SemanticHints':
        """Create from dictionary."""
        phase = None
        if 'execution_phase' in data:
            try:
                phase = CoreExecutionPhase(data['execution_phase'])
            except (ValueError, KeyError):
                pass
        
        priority = None
        if 'priority' in data:
            try:
                priority = Priority(data['priority'])
            except (ValueError, KeyError):
                pass
        
        return cls(
            phase=phase,
            priority=priority,
            kv_cache_size_mb=data.get('kv_cache_size_mb'),
            expected_tokens=data.get('expected_tokens'),
            session_id=data.get('session_id'),
        )


# Context variable for semantic hints
_current_hints: contextvars.ContextVar[Optional[SemanticHints]] = contextvars.ContextVar(
    'semantic_hints', 
    default=None
)


class session:
    """
    Context manager for semantic hints.
    
    Usage:
        with djinn.session(phase="decode", priority="interactive"):
            # This execution uses semantic hints
            logits = model(input_ids)
    
    Args:
        phase: Execution phase ("prefill", "decode", "vision", etc.)
        priority: Request priority ("background", "normal", "interactive", "realtime")
        kv_cache_size_mb: Estimated KV cache size in MB (for decode phase)
        expected_tokens: Number of tokens expected (for decode phase)
    """
    
    def __init__(
        self,
        phase: Optional[str] = None,
        priority: Optional[str] = None,
        kv_cache_size_mb: Optional[float] = None,
        expected_tokens: Optional[int] = None,
        **kwargs
    ):
        # Map string phase to ExecutionPhase enum
        phase_enum = None
        if phase:
            # Map common string values to enum
            phase_map = {
                'prefill': CoreExecutionPhase.LLM_PREFILL,
                'decode': CoreExecutionPhase.LLM_DECODE,
                'vision': CoreExecutionPhase.VISION_ENCODING,
                'training': CoreExecutionPhase.TRAINING,
                'generic': CoreExecutionPhase.FORWARD,
                'unknown': CoreExecutionPhase.UNKNOWN,
            }
            phase_enum = phase_map.get(phase.lower())
            if phase_enum is None:
                # Try direct enum value lookup
                try:
                    phase_enum = CoreExecutionPhase(phase)
                except (ValueError, KeyError):
                    pass
        
        # Map string priority to Priority enum
        priority_enum = None
        if priority:
            try:
                priority_enum = Priority[priority.upper()]
            except (KeyError, AttributeError):
                pass
        
        self.hints = SemanticHints(
            phase=phase_enum,
            priority=priority_enum,
            kv_cache_size_mb=kv_cache_size_mb,
            expected_tokens=expected_tokens,
            **kwargs
        )
        self._token = None
    
    def __enter__(self):
        """Enter context - set semantic hints."""
        self._token = _current_hints.set(self.hints)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - clear semantic hints."""
        if self._token is not None:
            _current_hints.reset(self._token)
        return False  # Don't suppress exceptions


def get_current_hints() -> Optional[SemanticHints]:
    """
    Get current semantic hints from context.
    
    Returns:
        Current SemanticHints if set, None otherwise
    """
    return _current_hints.get()


def get_hints_dict() -> Optional[dict]:
    """
    Get current semantic hints as dictionary.
    
    Convenience function for passing hints to execution functions.
    
    Returns:
        Dictionary representation of hints, or None if no hints set
    """
    hints = get_current_hints()
    if hints:
        return hints.to_dict()
    return None


