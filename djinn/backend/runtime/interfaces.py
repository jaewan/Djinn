"""
Runtime interfaces for Djinn disaggregated execution.

Defines the core abstractions for remote execution, memory management,
and transfer coordination.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import torch


class HealthStatus(Enum):
    """Health status of a remote accelerator."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MemoryView:
    """View into remote GPU memory."""
    handle: int
    size_bytes: int
    device_id: int
    is_pinned: bool = False


@dataclass
class TensorHandle:
    """Handle to a tensor on remote accelerator."""
    id: str
    shape: torch.Size
    dtype: torch.dtype
    device: str  # e.g., "gpu:0"
    memory_view: Optional[MemoryView] = None


@dataclass
class TransferRequest:
    """Request to transfer data between nodes."""
    transfer_id: str
    source_node: str
    target_node: str
    tensor_handle: TensorHandle
    metadata: Dict[str, Any]


@dataclass
class TransferResult:
    """Result of a transfer operation."""
    transfer_id: str
    success: bool
    error_message: Optional[str] = None
    tensor_handle: Optional[TensorHandle] = None


@dataclass
class PlanFragment:
    """Fragment of an execution plan for remote accelerator."""
    fragment_id: str
    operations: List[Dict[str, Any]]  # Serialized operations
    input_handles: List[TensorHandle]
    output_handles: List[TensorHandle]


class RemoteAccelerator(ABC):
    """Abstract interface for remote GPU accelerator."""

    @abstractmethod
    async def execute_plan(self, plan: PlanFragment) -> Dict[str, TensorHandle]:
        """Execute a plan fragment and return output handles."""
        pass

    @abstractmethod
    async def get_health(self) -> HealthStatus:
        """Get health status of this accelerator."""
        pass

    @abstractmethod
    async def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        pass


class ExecutionService(ABC):
    """Abstract interface for remote execution services."""

    @abstractmethod
    async def execute_operations(
        self,
        operations: List[Dict[str, Any]],
        input_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute operations on input tensors."""
        pass

    @abstractmethod
    async def transfer_tensor(
        self,
        tensor: torch.Tensor,
        target_node: str,
        metadata: Dict[str, Any]
    ) -> TransferResult:
        """Transfer tensor to target node."""
        pass

    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of this execution service."""
        pass
