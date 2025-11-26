"""
Enhanced exception handling with Result type for production-grade error management.

Based on Rust's Result<T, E> pattern for explicit error handling.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Union, Optional, Callable, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Type variables for Result
T = TypeVar('T')
E = TypeVar('E')


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DjinnException(Exception):
    """Base exception for all Djinn framework errors."""
    
    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class InterceptionError(DjinnException):
    """Error during tensor interception or operation capture."""
    pass


class MaterializationError(DjinnException):
    """Error during tensor materialization (execution)."""
    pass


class SchedulingError(DjinnException):
    """Error during graph scheduling or placement."""
    pass


class ShapeInferenceError(DjinnException):
    """Error during shape inference."""
    pass


class NetworkError(DjinnException):
    """Error in network communication or remote execution."""
    pass


class DjinnInternalError(DjinnException):
    """Internal framework error (should not occur in normal operation)."""
    pass


# ============================================================================
# BACKWARD COMPATIBILITY: Legacy Exception Names
# ============================================================================

# These are kept for backward compatibility with existing code
class PatternMatchError(DjinnException):
    """Pattern matching failed (legacy name)."""
    pass


class GraphBuildError(DjinnException):
    """Graph construction failed (legacy name)."""
    pass


class OptimizationError(DjinnException):
    """Graph optimization failed (legacy name)."""
    pass


class PlacementError(DjinnException):
    """Device placement failed (legacy name)."""
    pass


class TransferTimeoutError(DjinnException):
    """Transfer timed out (legacy name)."""
    pass


class NotApplicableError(DjinnException):
    """
    Execution strategy cannot handle this tensor (legacy name).
    
    Used in fallback chains where multiple strategies are tried.
    When raised, the next strategy in the chain should be attempted.
    """
    pass


class DeviceError(DjinnException):
    """Device operation failed (legacy name)."""
    pass


class ConfigurationError(DjinnException):
    """Invalid configuration (legacy name)."""
    pass


class SemanticException(DjinnException):
    """Base for semantic analysis errors (legacy name)."""
    pass


class TransportException(DjinnException):
    """Base for transport errors (legacy name)."""
    pass


class ExecutionException(DjinnException):
    """Base for execution errors (legacy name)."""
    pass


# ============================================================================
# RESULT TYPE SYSTEM - USAGE GUIDE
# ============================================================================
#
# Djinn has TWO result handling systems (both are valid):
#
# 1. RESULT CLASS (Lines 158-210) - Simple, Backward Compatible
#    - Used by: Semantic analysis, pattern matching, optimizer
#    - Import: from djinn.core.exceptions import Result
#    - Usage: Result.ok(value), Result.err(error)
#    - Type hint: Result[T] (documents intent, not enforced at runtime)
#    - Example:
#        def match_patterns(...) -> Result[List[MatchedPattern]]:
#            return Result.ok(matches)
#    - Status: DEPRECATED - Use Ok/Err for new code
#
# 2. OK/ERR CLASSES (Lines 216-334) - Modern, Rust-inspired (PREFERRED)
#    - Used by: Scheduler, new modules
#    - Import: from djinn.core.exceptions import Ok, Err
#    - Usage: Ok(value), Err(error)
#    - Type hint: ResultType[T, E] = Union[Ok[T], Err[E]]
#    - Example:
#        def schedule(...) -> ResultType[Plan, SchedulingError]:
#            return Ok(plan)
#    - Status: PREFERRED for new code
#
# IMPORTANT: Don't mix systems in the same module. Pick one and stick with it.
#
# Migration Guide:
#   - Old: result = Result.ok(value)
#   - New: result = Ok(value)
#   - Old: if result.is_ok: value = result.unwrap()
#   - New: if result.is_ok(): value = result.unwrap()
#
# Why both? Historical reasons. Semantic module was built with Result class,
# scheduler was built with Ok/Err. Both work fine. Future: may unify to Ok/Err.
# ============================================================================

# Result class for semantic analysis (backward compatible)
class Result:
    """
    Simple Result type for explicit error handling.
    
    Used by semantic analysis module. For new code, consider using Ok/Err classes.
    
    Usage:
        result = Result.ok(value)
        if result.is_ok:
            value = result.unwrap()
    """
    
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error
    
    @staticmethod
    def ok(value):
        """Create successful result."""
        return Result(value=value)
    
    @staticmethod
    def err(error):
        """Create error result."""
        return Result(error=error)
    
    @property
    def is_ok(self):
        """Check if result is successful."""
        return self._error is None
    
    @property
    def is_err(self):
        """Check if result is error."""
        return self._error is not None
    
    @property
    def error(self):
        """Get error if present."""
        return self._error
    
    def unwrap(self):
        """Get value or raise exception."""
        if self.is_err:
            raise self._error
        return self._value
    
    def unwrap_or(self, default):
        """Get value or return default."""
        if self.is_err:
            return default
        return self._value


# ============================================================================
# RESULT TYPE (Production-Grade Error Handling)
# ============================================================================

@dataclass
class Ok(Generic[T]):
    """
    Success result containing a value.
    
    Design: Immutable, hashable, comparable.
    """
    value: T
    
    def is_ok(self) -> bool:
        """Check if result is success."""
        return True
    
    def is_err(self) -> bool:
        """Check if result is error."""
        return False
    
    def unwrap(self) -> T:
        """Extract value (panics if Err)."""
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return self.value
    
    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        """Extract value or compute default."""
        return self.value
    
    def map(self, fn: Callable[[T], Any]) -> Result:
        """Transform success value."""
        try:
            return Ok(fn(self.value))
        except Exception as e:
            return Err(e)
    
    def map_err(self, fn: Callable[[Any], Any]) -> Result:
        """Transform error (no-op for Ok)."""
        return self
    
    def and_then(self, fn: Callable[[T], Result]) -> Result:
        """Chain operations that return Result."""
        try:
            return fn(self.value)
        except Exception as e:
            return Err(e)
    
    def __str__(self) -> str:
        return f"Ok({self.value})"
    
    def __repr__(self) -> str:
        return f"Ok({repr(self.value)})"


@dataclass
class Err(Generic[E]):
    """
    Error result containing an exception.
    
    Design: Carries context for debugging.
    """
    error: E
    context: dict = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    def is_ok(self) -> bool:
        """Check if result is success."""
        return False
    
    def is_err(self) -> bool:
        """Check if result is error."""
        return True
    
    def unwrap(self) -> T:
        """Extract value (raises error)."""
        if isinstance(self.error, Exception):
            raise self.error
        else:
            raise RuntimeError(str(self.error))
    
    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return default
    
    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        """Extract value or compute default."""
        return fn()
    
    def map(self, fn: Callable[[T], Any]) -> Result:
        """Transform success value (no-op for Err)."""
        return self
    
    def map_err(self, fn: Callable[[E], Any]) -> Result:
        """Transform error."""
        try:
            new_error = fn(self.error)
            return Err(new_error, self.context)
        except Exception as e:
            return Err(e, self.context)
    
    def and_then(self, fn: Callable[[T], Result]) -> Result:
        """Chain operations (no-op for Err)."""
        return self
    
    def __str__(self) -> str:
        if self.context:
            return f"Err({self.error}) with context: {self.context}"
        return f"Err({self.error})"
    
    def __repr__(self) -> str:
        return f"Err({repr(self.error)}, context={self.context})"


# Type alias for Ok/Err union (use ResultType for type annotations with Ok/Err)
# Note: Result class (lines 128-174) is separate and used by semantic module
ResultType = Union[Ok[T], Err[E]]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ok(value: T) -> Ok[T]:
    """Create an Ok result."""
    return Ok(value)


def err(error: Exception, context: Optional[dict] = None) -> Err:
    """Create an Err result."""
    return Err(error, context)


def try_fn(fn: Callable, *args, **kwargs) -> Result:
    """
    Execute a function and wrap result in Result type.
    
    Example:
        result = try_fn(risky_operation, arg1, arg2)
        if result.is_ok():
            value = result.unwrap()
    """
    try:
        return Ok(fn(*args, **kwargs))
    except Exception as e:
        return Err(e)


async def try_async_fn(fn: Callable, *args, **kwargs) -> Result:
    """
    Execute an async function and wrap result in Result type.
    """
    try:
        return Ok(await fn(*args, **kwargs))
    except Exception as e:
        return Err(e)


# ============================================================================
# RESULT COLLECTION UTILITIES
# ============================================================================

def collect_results(results: list[Result]) -> Result[list]:
    """
    Convert Vec<Result<T>> to Result<Vec<T>>.
    
    Returns Ok if all succeed, Err if any fail.
    """
    values = []
    for result in results:
        if result.is_err():
            return result  # Return first error
        values.append(result.unwrap())
    return Ok(values)


def all_ok(results: list[Result]) -> bool:
    """Check if all results are Ok."""
    return all(r.is_ok() for r in results)


def any_err(results: list[Result]) -> bool:
    """Check if any result is Err."""
    return any(r.is_err() for r in results)

