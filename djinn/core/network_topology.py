"""
Network topology management for cost-aware scheduling.

Bridges the coordinator's network information with the semantic analysis
and scheduling components for accurate cost estimation.
"""

import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkDevice:
    """Information about a network device."""
    node_id: str
    bandwidth_gbps: float  # Peak bandwidth in Gbps
    latency_ms: float      # Base latency in milliseconds
    device_type: str       # 'gpu', 'cpu', 'storage', etc.
    compute_tflops: float  # Compute capability (TFLOPS)
    memory_gb: float       # Memory capacity (GB)


@dataclass
class NetworkLink:
    """Information about a network link between devices."""
    source_id: str
    dest_id: str
    bandwidth_gbps: float  # Available bandwidth
    latency_ms: float      # Measured latency
    congestion_factor: float = 1.0  # Multiplier for congestion


class NetworkTopologyManager:
    """
    Manages network topology information for cost-aware scheduling.

    Integrates with the coordinator to get real network measurements
    and provides this information to the cost estimator and scheduler.
    """

    def __init__(self):
        self.devices: Dict[str, NetworkDevice] = {}
        self.links: Dict[Tuple[str, str], NetworkLink] = {}
        self._lock = threading.RLock()

        # Coordinator integration
        self._coordinator = None
        self._last_update = 0
        self._update_interval = 5.0  # Update every 5 seconds

    def register_device(self, device: NetworkDevice):
        """Register a network device."""
        with self._lock:
            self.devices[device.node_id] = device
            logger.debug(f"Registered device: {device.node_id}")

    def update_link_info(self, source_id: str, dest_id: str,
                        bandwidth_gbps: float, latency_ms: float,
                        congestion_factor: float = 1.0):
        """Update link information between devices."""
        with self._lock:
            link_key = (source_id, dest_id)
            self.links[link_key] = NetworkLink(
                source_id=source_id,
                dest_id=dest_id,
                bandwidth_gbps=bandwidth_gbps,
                latency_ms=latency_ms,
                congestion_factor=congestion_factor
            )
            logger.debug(f"Updated link: {source_id} -> {dest_id} ({bandwidth_gbps} Gbps, {latency_ms}ms)")

    def get_device(self, node_id: str) -> Optional[NetworkDevice]:
        """Get device information."""
        with self._lock:
            return self.devices.get(node_id)

    def get_link(self, source_id: str, dest_id: str) -> Optional[NetworkLink]:
        """Get link information between devices."""
        with self._lock:
            # Try both directions (assume symmetric for now)
            link_key = (source_id, dest_id)
            reverse_key = (dest_id, source_id)

            if link_key in self.links:
                return self.links[link_key]
            elif reverse_key in self.links:
                return self.links[reverse_key]

            return None

    def get_bandwidth(self, source_id: str, dest_id: str) -> float:
        """Get available bandwidth between devices."""
        link = self.get_link(source_id, dest_id)
        if link:
            # Apply congestion factor
            return link.bandwidth_gbps * (1.0 / link.congestion_factor)
        else:
            # Default bandwidth estimate
            source_device = self.get_device(source_id)
            dest_device = self.get_device(dest_id)

            if source_device and dest_device:
                # Estimate based on device types
                if source_device.device_type == 'gpu' and dest_device.device_type == 'gpu':
                    return 100.0  # High-speed GPU interconnect
                else:
                    return 10.0   # Standard Ethernet

            return 10.0  # Default fallback

    def get_latency(self, source_id: str, dest_id: str) -> float:
        """Get latency between devices."""
        if source_id == dest_id:
            return 0.1  # Local device latency

        link = self.get_link(source_id, dest_id)
        if link:
            return link.latency_ms

        # Estimate based on device locations
        source_device = self.get_device(source_id)
        dest_device = self.get_device(dest_id)

        if source_device and dest_device:
            # Estimate based on device types and assumed topology
            if source_device.device_type == 'gpu' and dest_device.device_type == 'gpu':
                return 0.5  # Fast GPU-to-GPU
            else:
                return 2.0  # Cross-device

        return 1.0  # Default latency

    def estimate_transfer_time(self, bytes_to_transfer: float,
                              source_id: str, dest_id: str) -> float:
        """Estimate transfer time in milliseconds."""
        bandwidth_gbps = self.get_bandwidth(source_id, dest_id)

        if bandwidth_gbps <= 0:
            return float('inf')

        # Convert to bytes per millisecond
        bandwidth_bytes_per_sec = (bandwidth_gbps * 1e9) / 8
        bandwidth_bytes_per_ms = bandwidth_bytes_per_sec / 1000

        if bandwidth_bytes_per_ms > 0:
            return bytes_to_transfer / bandwidth_bytes_per_ms

        return float('inf')

    def estimate_queueing_delay(self, source_id: str, dest_id: str,
                               queue_depth: int = 1) -> float:
        """Estimate queueing delay based on network congestion and device load."""
        base_latency = self.get_latency(source_id, dest_id)

        # Get destination device load (simplified model)
        dest_device = self.get_device(dest_id)
        if dest_device:
            # Estimate processing time based on compute capability
            processing_time_per_op = 1000.0 / dest_device.compute_tflops  # ms per TFLOP

            # Queueing delay = base latency + queue_depth * processing_time
            queueing_delay = base_latency + (queue_depth * processing_time_per_op * 0.001)  # Convert to ms
            return queueing_delay

        return base_latency

    def integrate_with_coordinator(self, coordinator):
        """Integrate with Djinn coordinator for live network data."""
        self._coordinator = coordinator
        logger.info("Integrated with coordinator for live network data")

    def update_from_coordinator(self, force: bool = False):
        """Update network information from coordinator."""
        if not self._coordinator:
            return

        current_time = time.time()
        if not force and (current_time - self._last_update) < self._update_interval:
            return  # No update needed

        try:
            # Query coordinator for network statistics
            # This would integrate with the coordinator's monitoring
            network_stats = self._coordinator.get_network_stats()

            for node_id, stats in network_stats.items():
                if node_id not in self.devices:
                    # Create device entry from coordinator data
                    device = NetworkDevice(
                        node_id=node_id,
                        bandwidth_gbps=stats.get('bandwidth_gbps', 10.0),
                        latency_ms=stats.get('latency_ms', 1.0),
                        device_type=stats.get('device_type', 'gpu'),
                        compute_tflops=stats.get('compute_tflops', 10.0),
                        memory_gb=stats.get('memory_gb', 16.0)
                    )
                    self.register_device(device)

                # Update link information if available
                for other_node_id, link_stats in stats.get('links', {}).items():
                    self.update_link_info(
                        node_id, other_node_id,
                        link_stats.get('bandwidth_gbps', 10.0),
                        link_stats.get('latency_ms', 1.0),
                        link_stats.get('congestion_factor', 1.0)
                    )

            self._last_update = current_time
            logger.debug("Updated network topology from coordinator")

        except Exception as e:
            logger.warning(f"Failed to update from coordinator: {e}")

    def get_topology_summary(self) -> Dict[str, Any]:
        """Get summary of current network topology."""
        with self._lock:
            return {
                'num_devices': len(self.devices),
                'num_links': len(self.links),
                'devices': {node_id: {
                    'type': device.device_type,
                    'bandwidth_gbps': device.bandwidth_gbps,
                    'latency_ms': device.latency_ms,
                    'compute_tflops': device.compute_tflops,
                    'memory_gb': device.memory_gb
                } for node_id, device in self.devices.items()},
                'links': {f"{link.source_id}->{link.dest_id}": {
                    'bandwidth_gbps': link.bandwidth_gbps,
                    'latency_ms': link.latency_ms,
                    'congestion_factor': link.congestion_factor
                } for link in self.links.values()}
            }


# Global network topology manager
_global_network_topology: Optional[NetworkTopologyManager] = None
_topology_lock = threading.Lock()


def get_network_topology() -> NetworkTopologyManager:
    """Get the global network topology manager."""
    global _global_network_topology
    if _global_network_topology is None:
        with _topology_lock:
            if _global_network_topology is None:
                _global_network_topology = NetworkTopologyManager()

    return _global_network_topology


def initialize_network_topology():
    """Initialize the global network topology manager."""
    global _global_network_topology
    with _topology_lock:
        _global_network_topology = NetworkTopologyManager()
