"""
PHASE 4.1: Network Monitoring with Socket Instrumentation

This module implements application-level network monitoring by intercepting
socket operations. This allows accurate tracking of network transfers without
requiring kernel-level eBPF or special privileges.

Architecture:
1. Monkey-patch socket.send() and socket.recv()
2. Track bytes transferred per socket
3. Aggregate by device pair (GPU-to-GPU, host-to-GPU, etc.)
4. Provide statistics and latency estimates

Expected impact: Fix network monitoring (currently 0.0 MB) → accurate measurements
"""

import socket
import threading
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

from ..common.async_local import AsyncLocal

logger = logging.getLogger(__name__)


@dataclass
class NetworkTransfer:
    """Represents a single network transfer event."""
    source: str  # Source device/host
    destination: str  # Destination device/host
    bytes_transferred: int  # Number of bytes
    latency_ms: float  # Transfer latency in milliseconds
    timestamp: float = field(default_factory=time.time)


@dataclass
class NetworkStats:
    """Aggregated network statistics."""
    total_bytes: int = 0
    total_transfers: int = 0
    total_latency_ms: float = 0.0
    
    # Per-device pair stats
    transfer_by_pair: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    bytes_by_pair: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    
    # Latency statistics
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Performance metrics
    peak_bandwidth_mbps: float = 0.0
    avg_bandwidth_mbps: float = 0.0
    
    def __str__(self) -> str:
        """Human-readable statistics summary."""
        return f"""
Network Statistics:
  Total bytes: {self.total_bytes / 1024 / 1024:.1f} MB
  Total transfers: {self.total_transfers}
  Avg latency: {self.avg_latency_ms:.2f}ms
  Min latency: {self.min_latency_ms:.2f}ms
  Max latency: {self.max_latency_ms:.2f}ms
  Avg bandwidth: {self.avg_bandwidth_mbps:.1f} Mbps
  Peak bandwidth: {self.peak_bandwidth_mbps:.1f} Mbps
  
Transfer breakdown:
"""  + "\n".join(
            f"  {src} → {dst}: {self.bytes_by_pair.get((src, dst), 0) / 1024 / 1024:.1f} MB ({self.transfer_by_pair.get((src, dst), 0)} transfers)"
            for src, dst in sorted(self.bytes_by_pair.keys())
        )


class NetworkMonitor:
    """
    Monitors network transfers via socket instrumentation.
    
    PHASE 4.1 Strategy:
    
    1. Hook socket.send() and socket.recv()
    2. Track bytes transferred
    3. Infer device pairs from network flow context
    4. Aggregate statistics
    5. Provide latency estimates
    
    This enables accurate measurement of network overhead without
    requiring kernel-level eBPF or special privileges.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize network monitor.
        
        Args:
            enabled: Whether to enable monitoring
        """
        self.enabled = enabled
        self.transfers: List[NetworkTransfer] = []
        self.stats = NetworkStats()
        self._lock = threading.Lock()
        self._original_send = socket.socket.send
        self._original_recv = socket.socket.recv
        self._original_sendall = socket.socket.sendall
        self._monitoring_active = False
        
        # Device context (async-aware to handle multi-threaded + async scenarios)
        self._context = AsyncLocal("network_monitor_context")
    
    def start_monitoring(self):
        """Enable network monitoring by monkey-patching socket operations."""
        if not self.enabled or self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Monkey-patch socket operations
        socket.socket.send = self._monitored_send
        socket.socket.recv = self._monitored_recv
        socket.socket.sendall = self._monitored_sendall
        
        logger.info("Network monitoring started")
    
    def stop_monitoring(self):
        """Disable network monitoring and restore original socket operations."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        
        # Restore original socket operations
        socket.socket.send = self._original_send
        socket.socket.recv = self._original_recv
        socket.socket.sendall = self._original_sendall
        
        logger.info("Network monitoring stopped")
    
    def set_device_context(self, source: str, destination: str):
        """
        Set the current device context for transfers.
        
        Used to label transfers with source/destination devices.
        Example: set_device_context("cuda:0", "cuda:1")
        
        Args:
            source: Source device identifier
            destination: Destination device identifier
        """
        self._context.source = source
        self._context.destination = destination
    
    def clear_device_context(self):
        """Clear the device context."""
        self._context.source = None
        self._context.destination = None
    
    def _monitored_send(self, data: bytes, flags: int = 0) -> int:
        """Wrapped socket.send() with monitoring."""
        if not self._monitoring_active:
            return self._original_send(self, data, flags)
        
        start_time = time.perf_counter()
        try:
            bytes_sent = self._original_send(self, data, flags)
        except Exception as e:
            logger.debug(f"Socket send error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Record transfer
        source = getattr(self._context, 'source', 'unknown')
        destination = getattr(self._context, 'destination', 'unknown')
        
        transfer = NetworkTransfer(
            source=source,
            destination=destination,
            bytes_transferred=bytes_sent,
            latency_ms=latency_ms
        )
        
        self._record_transfer(transfer)
        return bytes_sent
    
    def _monitored_recv(self, bufsize: int, flags: int = 0) -> bytes:
        """Wrapped socket.recv() with monitoring."""
        if not self._monitoring_active:
            return self._original_recv(self, bufsize, flags)
        
        start_time = time.perf_counter()
        try:
            data = self._original_recv(self, bufsize, flags)
        except Exception as e:
            logger.debug(f"Socket recv error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Record transfer
        source = getattr(self._context, 'source', 'unknown')
        destination = getattr(self._context, 'destination', 'unknown')
        
        transfer = NetworkTransfer(
            source=source,
            destination=destination,
            bytes_transferred=len(data),
            latency_ms=latency_ms
        )
        
        self._record_transfer(transfer)
        return data
    
    def _monitored_sendall(self, data: bytes, flags: int = 0) -> None:
        """Wrapped socket.sendall() with monitoring."""
        if not self._monitoring_active:
            return self._original_sendall(self, data, flags)
        
        start_time = time.perf_counter()
        try:
            self._original_sendall(self, data, flags)
        except Exception as e:
            logger.debug(f"Socket sendall error: {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Record transfer
        source = getattr(self._context, 'source', 'unknown')
        destination = getattr(self._context, 'destination', 'unknown')
        
        transfer = NetworkTransfer(
            source=source,
            destination=destination,
            bytes_transferred=len(data),
            latency_ms=latency_ms
        )
        
        self._record_transfer(transfer)
    
    def _record_transfer(self, transfer: NetworkTransfer):
        """Record a network transfer event."""
        with self._lock:
            self.transfers.append(transfer)
            
            # Update aggregate statistics
            self.stats.total_bytes += transfer.bytes_transferred
            self.stats.total_transfers += 1
            self.stats.total_latency_ms += transfer.latency_ms
            
            # Update per-pair statistics
            pair = (transfer.source, transfer.destination)
            self.stats.transfer_by_pair[pair] += 1
            self.stats.bytes_by_pair[pair] += transfer.bytes_transferred
            
            # Update latency statistics
            self.stats.min_latency_ms = min(self.stats.min_latency_ms, transfer.latency_ms)
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, transfer.latency_ms)
            
            # Update average
            if self.stats.total_transfers > 0:
                self.stats.avg_latency_ms = self.stats.total_latency_ms / self.stats.total_transfers
            
            # Estimate bandwidth
            if transfer.latency_ms > 0:
                bandwidth_mbps = (transfer.bytes_transferred / 1024 / 1024) / (transfer.latency_ms / 1000)
                self.stats.peak_bandwidth_mbps = max(self.stats.peak_bandwidth_mbps, bandwidth_mbps)
    
    def get_statistics(self) -> NetworkStats:
        """Get current network statistics."""
        with self._lock:
            if self.stats.total_transfers > 0:
                self.stats.avg_bandwidth_mbps = (self.stats.total_bytes / 1024 / 1024) / (self.stats.total_latency_ms / 1000) if self.stats.total_latency_ms > 0 else 0
            return self.stats
    
    def reset_statistics(self):
        """Clear all statistics."""
        with self._lock:
            self.transfers = []
            self.stats = NetworkStats()
    
    def get_transfers(self) -> List[NetworkTransfer]:
        """Get all recorded transfers."""
        with self._lock:
            return self.transfers.copy()


# Global instance
_global_network_monitor: Optional[NetworkMonitor] = None


def get_network_monitor(enabled: bool = True) -> NetworkMonitor:
    """
    Get the global network monitor instance.
    
    Args:
        enabled: Whether monitoring should be enabled
        
    Returns:
        Global NetworkMonitor instance
    """
    global _global_network_monitor
    if _global_network_monitor is None:
        _global_network_monitor = NetworkMonitor(enabled=enabled)
    return _global_network_monitor


class NetworkMonitoringContext:
    """Context manager for network monitoring."""
    
    def __init__(self, source: str = "host", destination: str = "gpu"):
        """
        Initialize monitoring context.
        
        Args:
            source: Source device
            destination: Destination device
        """
        self.source = source
        self.destination = destination
        self.monitor = get_network_monitor()
    
    def __enter__(self):
        """Start monitoring."""
        self.monitor.start_monitoring()
        self.monitor.set_device_context(self.source, self.destination)
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring."""
        self.monitor.clear_device_context()
        self.monitor.stop_monitoring()
