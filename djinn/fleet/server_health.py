"""
Server Health Tracker for Djinn Fleet Coordinator.

Tracks server health, load metrics, and latency across the fleet.
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ServerHealthMetrics:
    """Health metrics for a server."""
    server_address: str
    last_heartbeat: float
    is_healthy: bool = True
    
    # Load metrics
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    active_requests: int = 0
    
    # Latency metrics
    avg_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    
    # Metadata
    max_staleness: float = 60.0  # 60 seconds max staleness
    
    def is_stale(self) -> bool:
        """Check if metrics are stale."""
        return time.time() - self.last_heartbeat > self.max_staleness
    
    def get_memory_utilization(self) -> float:
        """Get memory utilization (0.0 to 1.0)."""
        if self.memory_total_gb == 0:
            return 0.0
        return self.memory_used_gb / self.memory_total_gb
    
    def get_load_score(self) -> float:
        """
        Get overall load score (0.0 to 1.0, higher = more loaded).
        
        Combines memory utilization, GPU utilization, and active requests.
        """
        memory_score = self.get_memory_utilization()
        gpu_score = self.gpu_utilization / 100.0  # Convert to 0-1
        request_score = min(self.active_requests / 100.0, 1.0)  # Cap at 100 requests
        
        # Weighted average
        return (memory_score * 0.4 + gpu_score * 0.4 + request_score * 0.2)


class ServerHealthTracker:
    """
    Tracks server health across the fleet.
    
    Features:
    - Heartbeat monitoring
    - Load metrics tracking
    - Latency tracking
    - Stale tolerance (60s max staleness)
    - Thread-safe operations
    """
    
    def __init__(self, max_staleness: float = 60.0):
        """
        Initialize server health tracker.
        
        Args:
            max_staleness: Maximum staleness in seconds (default: 60s)
        """
        import asyncio
        self.max_staleness = max_staleness
        self._servers: Dict[str, ServerHealthMetrics] = {}
        self._lock = asyncio.Lock()  # ✅ Thread safety
        logger.info(f"✅ ServerHealthTracker initialized (max_staleness={max_staleness}s)")
    
    async def update_health(self, server_address: str, metrics: Dict):
        """
        Update health metrics for a server (thread-safe).
        
        Args:
            server_address: Server address (e.g., "server-1:5556")
            metrics: Health metrics dict with keys:
                - memory_used_gb: float
                - memory_total_gb: float
                - gpu_utilization: float (0-100)
                - active_requests: int
                - latency_ms: float (optional)
        """
        async with self._lock:  # ✅ Thread safety
            if server_address not in self._servers:
                self._servers[server_address] = ServerHealthMetrics(
                    server_address=server_address,
                    last_heartbeat=time.time()
                )
            
            server = self._servers[server_address]
            server.last_heartbeat = time.time()
            server.is_healthy = True
            
            # Update metrics
            server.memory_used_gb = metrics.get('memory_used_gb', server.memory_used_gb)
            server.memory_total_gb = metrics.get('memory_total_gb', server.memory_total_gb)
            server.gpu_utilization = metrics.get('gpu_utilization', server.gpu_utilization)
            server.active_requests = metrics.get('active_requests', server.active_requests)
            
            # Update latency
            if 'latency_ms' in metrics:
                server.last_latency_ms = metrics['latency_ms']
                # Update average latency (exponential moving average)
                if server.avg_latency_ms == 0:
                    server.avg_latency_ms = metrics['latency_ms']
                else:
                    alpha = 0.3  # Smoothing factor
                    server.avg_latency_ms = (
                        alpha * metrics['latency_ms'] + 
                        (1 - alpha) * server.avg_latency_ms
                    )
            
            logger.debug(
                f"📊 Updated health for {server_address}: "
                f"memory={server.get_memory_utilization():.1%}, "
                f"gpu={server.gpu_utilization:.1f}%, "
                f"load={server.get_load_score():.2f}"
            )
    
    async def mark_unhealthy(self, server_address: str):
        """
        Mark a server as unhealthy (thread-safe).
        
        Args:
            server_address: Server address
        """
        async with self._lock:  # ✅ Thread safety
            if server_address in self._servers:
                self._servers[server_address].is_healthy = False
                logger.warning(f"⚠️  Marked server {server_address} as unhealthy")
    
    async def is_healthy(self, server_address: str) -> bool:
        """
        Check if a server is healthy (thread-safe).
        
        Args:
            server_address: Server address
            
        Returns:
            True if server is healthy and not stale, False otherwise
        """
        async with self._lock:  # ✅ Thread safety
            if server_address not in self._servers:
                return False
            
            server = self._servers[server_address]
            
            # Check if stale
            if server.is_stale():
                logger.debug(f"⚠️  Server {server_address} metrics are stale")
                return False
            
            # Check if marked unhealthy
            if not server.is_healthy:
                return False
            
            return True
    
    async def get_load_metrics(self, server_address: str) -> Optional[Dict]:
        """
        Get load metrics for a server (thread-safe).
        
        Args:
            server_address: Server address
            
        Returns:
            Dict with load metrics, or None if server not found
        """
        async with self._lock:  # ✅ Thread safety
            if server_address not in self._servers:
                return None
            
            server = self._servers[server_address]
            
            return {
                'memory_utilization': server.get_memory_utilization(),
                'memory_used_gb': server.memory_used_gb,
                'memory_total_gb': server.memory_total_gb,
                'gpu_utilization': server.gpu_utilization,
                'active_requests': server.active_requests,
                'load_score': server.get_load_score(),
                'avg_latency_ms': server.avg_latency_ms,
                'last_heartbeat': server.last_heartbeat,
                'is_stale': server.is_stale(),
            }
    
    async def filter_healthy(self, servers: List[str]) -> List[str]:
        """
        Filter list of servers to only healthy ones (thread-safe).
        
        Args:
            servers: List of server addresses
            
        Returns:
            List of healthy server addresses
        """
        # Cleanup stale entries when filtering (lazy cleanup)
        await self.cleanup_stale()
        
        healthy = []
        for server in servers:
            if await self.is_healthy(server):
                healthy.append(server)
            else:
                logger.debug(f"⚠️  Filtered out unhealthy/stale server: {server}")
        return healthy
    
    async def get_all_servers(self) -> List[str]:
        """Get all tracked server addresses (thread-safe)."""
        async with self._lock:
            return list(self._servers.keys())
    
    async def get_healthy_servers(self) -> List[str]:
        """Get all healthy server addresses (thread-safe)."""
        return await self.filter_healthy(await self.get_all_servers())
    
    async def cleanup_stale(self):
        """Remove stale server entries (thread-safe)."""
        async with self._lock:  # ✅ Thread safety
            stale_servers = [
                addr for addr, server in self._servers.items()
                if server.is_stale()
            ]
            for addr in stale_servers:
                logger.debug(f"🗑️  Removing stale server: {addr}")
                del self._servers[addr]

