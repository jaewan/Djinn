"""
Control plane data types.

Defines message types, node capabilities, and transfer request/response types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class NodeCapabilities:
    """Node capability information"""
    node_id: str
    gpu_count: int
    max_transfer_size: int = 10 * 1024 * 1024 * 1024  # 10GB default
    supported_dtypes: List[str] = field(default_factory=lambda: ['float32', 'float16', 'int32', 'int64'])
    network_bandwidth_gbps: float = 100.0  # 100 Gbps default
    memory_bandwidth_gbps: float = 900.0   # 900 GB/s default for modern GPUs
    features: List[str] = field(default_factory=lambda: ['fragmentation', 'compression', 'reliability'])
    # Optional data port for UDP plane (used by integration harness)
    data_port: int = 5556


@dataclass
class TransferRequest:
    """Transfer request information"""
    transfer_id: str
    tensor_id: str
    source_node: str
    target_node: str
    size: int
    dtype: str
    shape: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    timeout_seconds: float = 300.0  # 5 minute default timeout
    # Additional fields used by ClientHandler.handle_transfer_request
    source_gpu: int = 0
    target_gpu: int = 0
    priority: int = 1
    requires_ack: bool = True
    compression: bool = False


@dataclass
class TransferResponse:
    """Response to a transfer request."""
    transfer_id: str
    accepted: bool
    reason: str = ""
    estimated_time_seconds: Optional[float] = None
    allocated_gpu: int = 0
    data_port: int = 5556

