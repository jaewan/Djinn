"""
Transport layer for Djinn.

Provides various transport implementations for remote tensor execution:
- TCP: Reliable fallback with connection pooling
- DPDK: Zero-copy GPU-to-GPU transfers (Phase 2)
"""

from .base import Transport
from .tcp_transport import TCPTransport
from .dpdk_transport import DPDKTransport
from .connection_pool import ConnectionPool, Connection

__all__ = [
    'Transport',
    'TCPTransport',
    'DPDKTransport',
    'ConnectionPool',
    'Connection'
]
