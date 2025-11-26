"""
Standardized exceptions for Djinn server.

Provides consistent exception types and error handling across the server.
"""

from typing import Dict, Any, Optional

__all__ = [
    'DjinnServerError',
    'ModelNotFoundError',
    'ExecutionError',
    'OutOfMemoryError',
    'TimeoutError',
    'SecurityError',
    'InvalidRequestError',
]


class DjinnServerError(Exception):
    """
    Base exception for all Djinn server errors.
    
    Provides consistent error format with error codes and details.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize server error.
        
        Args:
            message: Human-readable error message
            error_code: Standard error code (e.g., 'MODEL_NOT_FOUND')
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for serialization."""
        return {
            'status': 'error',
            'error_type': self.error_code,
            'message': self.message,
            **self.details
        }
    
    def __str__(self) -> str:
        """String representation."""
        if self.details:
            return f"{self.error_code}: {self.message} ({self.details})"
        return f"{self.error_code}: {self.message}"


class ModelNotFoundError(DjinnServerError):
    """Exception raised when a model is not found."""
    
    def __init__(self, fingerprint: str):
        """
        Initialize model not found error.
        
        Args:
            fingerprint: Model fingerprint that was not found
        """
        super().__init__(
            message=f"Model {fingerprint} not found. Please register first.",
            error_code="MODEL_NOT_FOUND",
            details={
                'fingerprint': fingerprint,
                'required_action': 'register_model'
            }
        )


class ExecutionError(DjinnServerError):
    """Exception raised when model execution fails."""
    
    def __init__(self, operation: str, reason: str, traceback: Optional[str] = None):
        """
        Initialize execution error.
        
        Args:
            operation: Operation that failed
            reason: Reason for failure
            traceback: Optional traceback string
        """
        details = {
            'operation': operation,
            'reason': reason
        }
        if traceback:
            details['traceback'] = traceback
        
        super().__init__(
            message=f"Execution failed: {reason}",
            error_code="EXECUTION_FAILED",
            details=details
        )


class OutOfMemoryError(DjinnServerError):
    """Exception raised when out of memory."""
    
    def __init__(self, fingerprint: Optional[str] = None):
        """
        Initialize out of memory error.
        
        Args:
            fingerprint: Optional model fingerprint
        """
        details = {}
        if fingerprint:
            details['fingerprint'] = fingerprint
        
        super().__init__(
            message="Out of memory during execution",
            error_code="OUT_OF_MEMORY",
            details=details
        )


class TimeoutError(DjinnServerError):
    """Exception raised when operation times out."""
    
    def __init__(self, fingerprint: Optional[str] = None, timeout_seconds: Optional[float] = None):
        """
        Initialize timeout error.
        
        Args:
            fingerprint: Optional model fingerprint
            timeout_seconds: Optional timeout duration
        """
        details = {}
        if fingerprint:
            details['fingerprint'] = fingerprint
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        
        super().__init__(
            message=f"Request timed out after {timeout_seconds or 'unknown'} seconds",
            error_code="TIMEOUT",
            details=details
        )


class SecurityError(DjinnServerError):
    """Exception raised when security validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize security error.
        
        Args:
            message: Security error message
            details: Additional security details
        """
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            details=details or {}
        )


class InvalidRequestError(DjinnServerError):
    """Exception raised when request is invalid."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize invalid request error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            error_code="INVALID_REQUEST",
            details=details or {}
        )

