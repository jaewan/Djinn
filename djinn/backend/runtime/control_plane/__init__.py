"""
Control Plane TCP Server for Djinn Zero-Copy Transport

This package implements the TCP-based control plane that coordinates
zero-copy tensor transfers between nodes. It handles:

- Transfer request/response negotiation
- Node capability exchange
- Heartbeat and connection monitoring
- Transfer status tracking
- Error handling and recovery

The control plane uses JSON messages over TCP for reliability and
ease of debugging, while the data plane uses custom UDP packets
for maximum performance.
"""

from .messages import MessageType, TransferStatus, ControlMessage
from .types import NodeCapabilities, TransferRequest, TransferResponse
from .client_handler import ClientHandler
from .server import (
    ControlPlaneServer,
    get_control_server,
    start_control_server,
    stop_control_server,
)
from .client import ControlPlane
from .connection_pool import ControlPlaneConnectionPool, get_connection_pool

__all__ = [
    # Message types
    'MessageType',
    'TransferStatus',
    'ControlMessage',
    # Data types
    'NodeCapabilities',
    'TransferRequest',
    'TransferResponse',
    # Server components
    'ControlPlaneServer',
    'ClientHandler',
    'ControlPlane',
    # Connection pooling
    'ControlPlaneConnectionPool',
    'get_connection_pool',
    # Global server functions
    'get_control_server',
    'start_control_server',
    'stop_control_server',
]

