"""
Interception control with single source of truth.

This module provides thread-safe control over when Djinn should intercept operations.
"""

from enum import Enum

from ...common.async_local import AsyncLocal

class InterceptionContext(Enum):
    """What we're doing when interception is disabled."""
    NONE = "none"  # Normal operation, intercept
    CAPTURING = "capturing"  # Capturing LazyTensors into computation graph
    CONSTRUCTION = "construction"  # LazyTensor being constructed
    MATERIALIZATION = "materialization"  # Executing operations
    PROPERTY_ACCESS = "property_access"  # Accessing tensor metadata


_interception_context = AsyncLocal("interception_context")


def should_intercept(device=None, context_override=None):
    """
    Single source of truth for interception decisions.

    Args:
        device: Target device (if available)
        context_override: Explicit context check

    Returns:
        bool: Should we intercept this operation?
    """
    # Get current interception context
    current_context = getattr(_interception_context, 'context', InterceptionContext.NONE)

    # ✅ DESIGN FIX: CAPTURING context should ENABLE interception, not disable it
    # CAPTURING is used to mark that we're building a computation graph, so we MUST intercept
    if current_context == InterceptionContext.CAPTURING:
        return True
    
    # Check if we're in a disabled context (CONSTRUCTION, MATERIALIZATION, PROPERTY_ACCESS)
    # These contexts disable interception to avoid recursion
    if current_context != InterceptionContext.NONE:
        return False

    # Check capture context (fallback check)
    from .capture import is_capturing
    if is_capturing():
        return True

    # Check device
    if device is not None:
        device_str = str(device)
        if 'remote_accelerator' in device_str or 'privateuseone' in device_str:
            return True

    return False


def disable_interception(context: InterceptionContext):
    """Context manager to temporarily disable interception."""
    class DisableContext:
        def __enter__(self):
            self.prev_context = getattr(_interception_context, 'context', InterceptionContext.NONE)
            _interception_context.context = context

        def __exit__(self, *args):
            _interception_context.context = self.prev_context

    return DisableContext()


def get_current_context():
    """Get current interception context."""
    return getattr(_interception_context, 'context', InterceptionContext.NONE)
