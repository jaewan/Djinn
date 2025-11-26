"""
Typed metadata for operations and results.

Fixes critical remote execution routing issues by ensuring client_port is properly
included in all operation metadata and result routing.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import torch


@dataclass
class OperationMetadata:
    """Metadata for remote operation request."""
    operation: str
    result_id: str
    num_inputs: int
    input_shapes: List[tuple]
    input_dtypes: List[str]
    client_port: int  # ✅ CRITICAL: Client's listening port

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'result_id': self.result_id,
            'num_inputs': self.num_inputs,
            'input_shapes': [list(shape) for shape in self.input_shapes],
            'input_dtypes': self.input_dtypes,
            'client_port': self.client_port,  # ✅ Include in dict
        }


@dataclass
class ResultMetadata:
    """Metadata for operation result."""
    result_id: str
    original_transfer: str
    shape: tuple
    dtype: str
    size_bytes: int
    is_result: bool = True
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'result_id': self.result_id,
            'original_transfer': self.original_transfer,
            'shape': list(self.shape),
            'dtype': self.dtype,
            'size_bytes': self.size_bytes,
            'is_result': True,
            'is_error': False,
        }


@dataclass
class ErrorMetadata:
    """Metadata for operation error."""
    result_id: str
    original_transfer: str
    error_message: str
    error_type: str
    is_result: bool = True
    is_error: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'result_id': self.result_id,
            'original_transfer': self.original_transfer,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'is_result': True,
            'is_error': True,
            'shape': [],
            'dtype': 'torch.float32',
            'size_bytes': 0,
        }


def create_operation_metadata(operation: str, result_id: str,
                              inputs: List[torch.Tensor],
                              client_port: int) -> OperationMetadata:
    """Helper to create operation metadata."""
    return OperationMetadata(
        operation=operation,
        result_id=result_id,
        num_inputs=len(inputs),
        input_shapes=[tuple(t.shape) for t in inputs],
        input_dtypes=[str(t.dtype) for t in inputs],
        client_port=client_port,
    )


def create_result_metadata(result_id: str, original_transfer: str,
                           result: torch.Tensor) -> ResultMetadata:
    """Helper to create result metadata."""
    return ResultMetadata(
        result_id=result_id,
        original_transfer=original_transfer,
        shape=tuple(result.shape),
        dtype=str(result.dtype),
        size_bytes=result.numel() * result.element_size(),
    )


def create_error_metadata(result_id: str, original_transfer: str,
                          error: Exception) -> ErrorMetadata:
    """Helper to create error metadata."""
    return ErrorMetadata(
        result_id=result_id,
        original_transfer=original_transfer,
        error_message=str(error),
        error_type=type(error).__name__,
    )
