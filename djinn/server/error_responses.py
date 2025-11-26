"""
Standardized error response formatting for Djinn server.

Provides consistent error response format across all server components.
"""

from typing import Dict, Any, Optional
from enum import Enum

__all__ = ['ErrorCode', 'ErrorResponseBuilder']


class ErrorCode(str, Enum):
    """Standard error codes for server responses."""
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    OOM = "OUT_OF_MEMORY"
    TIMEOUT = "TIMEOUT"
    SECURITY_ERROR = "SECURITY_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"


class ErrorResponseBuilder:
    """Builder for standardized error responses."""
    
    @staticmethod
    def build(
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build standardized error response.
        
        Args:
            message: Human-readable error message
            error_code: Standard error code
            details: Additional error details
            
        Returns:
            Standardized error response dict
        """
        response = {
            'status': 'error',
            'error_type': error_code.value,
            'message': message
        }
        
        if details:
            response.update(details)
        
        return response
    
    @staticmethod
    def model_not_found(fingerprint: str, required_action: str = 'register_model') -> Dict[str, Any]:
        """Build model not found error."""
        return ErrorResponseBuilder.build(
            message=f'Model {fingerprint} not found. Please register first.',
            error_code=ErrorCode.MODEL_NOT_FOUND,
            details={
                'fingerprint': fingerprint,
                'required_action': required_action
            }
        )
    
    @staticmethod
    def execution_failed(operation: str, reason: str, traceback: Optional[str] = None) -> Dict[str, Any]:
        """Build execution failed error."""
        details = {'operation': operation, 'reason': reason}
        if traceback:
            details['traceback'] = traceback
        return ErrorResponseBuilder.build(
            message=f'Execution failed: {reason}',
            error_code=ErrorCode.EXECUTION_FAILED,
            details=details
        )
    
    @staticmethod
    def timeout(fingerprint: str, timeout_seconds: float) -> Dict[str, Any]:
        """Build timeout error."""
        return ErrorResponseBuilder.build(
            message=f'Request timed out after {timeout_seconds} seconds',
            error_code=ErrorCode.TIMEOUT,
            details={'fingerprint': fingerprint}
        )
    
    @staticmethod
    def circuit_breaker_open(retry_after_seconds: int = 60) -> Dict[str, Any]:
        """Build circuit breaker open error."""
        return ErrorResponseBuilder.build(
            message='Service temporarily unavailable due to errors',
            error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            details={'retry_after_seconds': retry_after_seconds}
        )
    
    @staticmethod
    def out_of_memory(fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """Build out of memory error."""
        details = {}
        if fingerprint:
            details['fingerprint'] = fingerprint
        return ErrorResponseBuilder.build(
            message='Out of memory during execution',
            error_code=ErrorCode.OOM,
            details=details
        )
    
    @staticmethod
    def security_error(message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build security error."""
        return ErrorResponseBuilder.build(
            message=message,
            error_code=ErrorCode.SECURITY_ERROR,
            details=details or {}
        )
    
    @staticmethod
    def invalid_request(message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build invalid request error."""
        return ErrorResponseBuilder.build(
            message=message,
            error_code=ErrorCode.INVALID_REQUEST,
            details=details or {}
        )

