"""
Capability Provider - Detects and advertises server capabilities.

Discovers:
- Available GPUs and their properties
- Network interfaces and speeds
- Memory capacity
- Supported transport protocols
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUCapabilities:
    """Information about a single GPU."""
    index: int
    name: str
    total_memory_gb: float
    compute_capability: str
    driver_version: str


@dataclass
class NetworkCapabilities:
    """Network interface information."""
    interface: str
    speed_gbps: int
    supports_rdma: bool
    supports_dpdk: bool
    ip_address: str
    mac_address: str


@dataclass
class ServerCapabilities:
    """Complete server capabilities."""
    gpu_count: int
    gpu_indices: List[int]
    gpus: List[GPUCapabilities]
    networks: List[NetworkCapabilities]
    total_memory_gb: float
    supported_transports: List[str]
    hostname: str
    node_id: str


class CapabilityProvider:
    """Discovers and provides server capabilities."""

    @classmethod
    def discover(cls) -> ServerCapabilities:
        """Discover all server capabilities."""
        logger.info("Discovering server capabilities...")

        # Discover GPUs
        gpus = cls._discover_gpus()
        logger.info(f"Found {len(gpus)} GPUs")

        # Discover network interfaces
        networks = cls._discover_networks()
        logger.info(f"Found {len(networks)} network interfaces")

        # Get system memory
        total_memory_gb = cls._get_system_memory_gb()

        # Determine supported transports
        supported_transports = cls._get_supported_transports(networks)

        # Get hostname
        hostname = cls._get_hostname()

        capabilities = ServerCapabilities(
            gpu_count=len(gpus),
            gpu_indices=[gpu.index for gpu in gpus],
            gpus=gpus,
            networks=networks,
            total_memory_gb=total_memory_gb,
            supported_transports=supported_transports,
            hostname=hostname,
            node_id=hostname  # Use hostname as node_id by default
        )

        logger.info(f"Capabilities discovered: {capabilities}")
        return capabilities

    @classmethod
    def _discover_gpus(cls) -> List[GPUCapabilities]:
        """Discover available GPUs using nvidia-ml-py."""
        try:
            import pynvml

            # Initialize NVML
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Get GPU name
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')

                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory_gb = mem_info.total / (1024**3)

                    # Get compute capability
                    major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                    minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                    compute_capability = f"{major}.{minor}"

                    # Get driver version
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode('utf-8')

                    gpu = GPUCapabilities(
                        index=i,
                        name=name,
                        total_memory_gb=total_memory_gb,
                        compute_capability=compute_capability,
                        driver_version=driver_version
                    )
                    gpus.append(gpu)

                except Exception as e:
                    logger.warning(f"Failed to query GPU {i}: {e}")
                    continue

            pynvml.nvmlShutdown()
            return gpus

        except ImportError:
            logger.warning("pynvml not available, no GPU discovery")
            return []
        except Exception as e:
            logger.error(f"GPU discovery failed: {e}")
            return []

    @classmethod
    def _discover_networks(cls) -> List[NetworkCapabilities]:
        """Discover network interfaces."""
        networks = []

        try:
            # Use ip command to get interface info
            result = subprocess.run(['ip', 'addr', 'show'],
                                  capture_output=True, text=True, check=True)

            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('2:') or line.startswith('3:'):  # Ethernet interfaces
                    # Parse interface info (simplified)
                    parts = line.split()
                    if len(parts) >= 2:
                        interface = parts[1].rstrip(':')

                        # Try to get interface speed using ethtool
                        try:
                            speed_result = subprocess.run(
                                ['ethtool', interface],
                                capture_output=True, text=True, check=True
                            )

                            speed_gbps = 1  # Default 1Gbps
                            for speed_line in speed_result.stdout.split('\n'):
                                if 'Speed:' in speed_line:
                                    speed_str = speed_line.split(':')[1].strip()
                                    if '1000' in speed_str:
                                        speed_gbps = 1
                                    elif '10000' in speed_str:
                                        speed_gbps = 10
                                    elif '25000' in speed_str:
                                        speed_gbps = 25
                                    elif '40000' in speed_str:
                                        speed_gbps = 40
                                    elif '100000' in speed_str:
                                        speed_gbps = 100
                                    break
                        except:
                            speed_gbps = 1

                        # Check for RDMA support (simplified check)
                        supports_rdma = cls._check_rdma_support(interface)

                        # Check for DPDK support (simplified check)
                        supports_dpdk = cls._check_dpdk_support(interface)

                        # Get IP address
                        ip_address = cls._get_interface_ip(interface)

                        # Get MAC address
                        mac_address = cls._get_interface_mac(interface)

                        if ip_address and mac_address:
                            network = NetworkCapabilities(
                                interface=interface,
                                speed_gbps=speed_gbps,
                                supports_rdma=supports_rdma,
                                supports_dpdk=supports_dpdk,
                                ip_address=ip_address,
                                mac_address=mac_address
                            )
                            networks.append(network)

        except Exception as e:
            logger.error(f"Network discovery failed: {e}")

        # If no interfaces found, add localhost as fallback
        if not networks:
            logger.warning("No network interfaces found, using localhost")
            networks.append(NetworkCapabilities(
                interface="lo",
                speed_gbps=1,
                supports_rdma=False,
                supports_dpdk=False,
                ip_address="127.0.0.1",
                mac_address="00:00:00:00:00:00"
            ))

        return networks

    @classmethod
    def _check_rdma_support(cls, interface: str) -> bool:
        """Check if interface supports RDMA."""
        try:
            # Check for RDMA devices
            result = subprocess.run(['ibstat'], capture_output=True, text=True)
            return 'CA type' in result.stdout or 'Rate' in result.stdout
        except:
            return False

    @classmethod
    def _check_dpdk_support(cls, interface: str) -> bool:
        """Check if interface supports DPDK."""
        # This is a simplified check - in practice would need more sophisticated detection
        try:
            # Check if DPDK is installed and interface can be used
            result = subprocess.run(['dpdk-devbind', '--status'],
                                  capture_output=True, text=True)
            return interface in result.stdout or 'Network devices' in result.stdout
        except:
            return False

    @classmethod
    def _get_interface_ip(cls, interface: str) -> Optional[str]:
        """Get IP address for interface."""
        try:
            result = subprocess.run(['ip', 'addr', 'show', interface],
                                  capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if 'inet ' in line and 'scope global' in line:
                    return line.split()[1].split('/')[0]
        except:
            pass
        return None

    @classmethod
    def _get_interface_mac(cls, interface: str) -> Optional[str]:
        """Get MAC address for interface."""
        try:
            result = subprocess.run(['ip', 'link', 'show', interface],
                                  capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if 'link/ether' in line:
                    return line.split()[1]
        except:
            pass
        return None

    @classmethod
    def _get_system_memory_gb(cls) -> float:
        """Get total system memory in GB."""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)  # Convert KB to GB
        except:
            pass
        return 16.0  # Default assumption

    @classmethod
    def _get_supported_transports(cls, networks: List[NetworkCapabilities]) -> List[str]:
        """Determine which transport protocols are supported."""
        transports = ['tcp']  # TCP always available

        # Check for DPDK support
        has_dpdk = any(net.supports_dpdk for net in networks)
        if has_dpdk:
            transports.append('dpdk')

        # Check for RDMA support
        has_rdma = any(net.supports_rdma for net in networks)
        if has_rdma:
            transports.append('rdma')

        return transports

    @classmethod
    def _get_hostname(cls) -> str:
        """Get system hostname."""
        return os.uname().nodename if hasattr(os, 'uname') else 'localhost'
