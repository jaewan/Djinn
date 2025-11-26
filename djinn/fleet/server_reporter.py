"""
Server Health Reporter for Djinn Fleet Coordinator.

Reports server health metrics to the global fleet coordinator.
Runs as a background task in the Djinn server.
"""

import logging
import asyncio
import time
import socket
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ServerHealthReporter:
    """
    Reports server health metrics to the global fleet coordinator.
    
    Features:
    - Periodic health reporting (heartbeat)
    - Automatic server registration
    - Metrics collection from server state
    - Graceful error handling
    """
    
    def __init__(
        self,
        server_address: str,
        global_coordinator=None,
        report_interval: float = 30.0
    ):
        """
        Initialize server health reporter.
        
        Args:
            server_address: Server address (e.g., "server-1:5556")
            global_coordinator: GlobalFleetCoordinator instance (optional)
            report_interval: Health report interval in seconds (default: 30s)
        """
        self.server_address = server_address
        self.global_coordinator = global_coordinator
        self.report_interval = report_interval
        self._reporting_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        logger.info(f"✅ ServerHealthReporter initialized for {server_address}")
    
    async def start(self):
        """Start health reporting background task."""
        if self._is_running:
            logger.warning("Health reporter already running")
            return
        
        if not self.global_coordinator:
            logger.debug("No global coordinator, skipping health reporting")
            return
        
        # Register server with global coordinator
        try:
            await self._register_server()
        except Exception as e:
            logger.warning(f"⚠️  Failed to register server: {e}")
        
        # Start background reporting task
        self._is_running = True
        self._reporting_task = asyncio.create_task(self._reporting_loop())
        logger.info(f"✅ Started health reporting (interval={self.report_interval}s)")
    
    async def stop(self):
        """Stop health reporting."""
        self._is_running = False
        
        if self._reporting_task:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                pass
        
        # Unregister server
        if self.global_coordinator:
            try:
                await self.global_coordinator.unregister_server(self.server_address)
            except Exception as e:
                logger.warning(f"⚠️  Failed to unregister server: {e}")
        
        logger.info("✅ Stopped health reporting")
    
    async def _register_server(self):
        """Register server with global coordinator."""
        if not self.global_coordinator:
            return
        
        # Get server capabilities
        capabilities = await self._get_server_capabilities()
        
        await self.global_coordinator.register_server(
            server_address=self.server_address,
            capabilities=capabilities
        )
        
        logger.info(f"✅ Registered server {self.server_address} with global coordinator")
    
    async def _reporting_loop(self):
        """Background loop for periodic health reporting."""
        while self._is_running:
            try:
                await asyncio.sleep(self.report_interval)
                
                if not self._is_running:
                    break
                
                # Collect and report metrics
                metrics = await self._collect_metrics()
                await self._report_health(metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Health reporting error: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
    
    async def _collect_metrics(self) -> Dict:
        """
        Collect health metrics from server state.
        
        Returns:
            Dict with health metrics:
                - memory_used_gb: float
                - memory_total_gb: float
                - gpu_utilization: float (0-100)
                - active_requests: int
                - latency_ms: float (optional)
        """
        metrics = {
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'gpu_utilization': 0.0,
            'active_requests': 0
        }
        
        try:
            # Try to get metrics from memory manager
            from ...backend.runtime.gpu_memory import get_memory_manager
            try:
                memory_manager = get_memory_manager()
                if memory_manager:
                    # Get GPU memory stats
                    gpu_stats = memory_manager.get_gpu_memory_stats()
                    if gpu_stats:
                        metrics['memory_used_gb'] = gpu_stats.get('used_gb', 0.0)
                        metrics['memory_total_gb'] = gpu_stats.get('total_gb', 0.0)
                        metrics['gpu_utilization'] = gpu_stats.get('utilization_percent', 0.0)
            except Exception as e:
                logger.debug(f"Could not get memory metrics: {e}")
            
            # Try to get active request count from server state
            try:
                from ...server.server_state import get_server_state
                server_state = get_server_state()
                if server_state:
                    metrics['active_requests'] = server_state.get_active_request_count()
            except Exception as e:
                logger.debug(f"Could not get request count: {e}")
            
        except Exception as e:
            logger.debug(f"Error collecting metrics: {e}")
        
        return metrics
    
    async def _report_health(self, metrics: Dict):
        """Report health metrics to global coordinator."""
        if not self.global_coordinator:
            return
        
        try:
            await self.global_coordinator.update_server_health(
                server_address=self.server_address,
                metrics=metrics
            )
            
            logger.debug(
                f"📊 Reported health: "
                f"memory={metrics.get('memory_used_gb', 0):.1f}/{metrics.get('memory_total_gb', 0):.1f}GB, "
                f"gpu={metrics.get('gpu_utilization', 0):.1f}%, "
                f"requests={metrics.get('active_requests', 0)}"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to report health: {e}")
    
    async def _get_server_capabilities(self) -> Dict:
        """
        Get server capabilities for registration.
        
        Returns:
            Dict with server capabilities:
                - memory_total_gb: float
                - gpu_count: int
                - etc.
        """
        capabilities = {
            'memory_total_gb': 0.0,
            'gpu_count': 0
        }
        
        try:
            # Try to get GPU memory stats
            from ...backend.runtime.gpu_memory import get_memory_manager
            try:
                memory_manager = get_memory_manager()
                if memory_manager:
                    gpu_stats = memory_manager.get_gpu_memory_stats()
                    if gpu_stats:
                        capabilities['memory_total_gb'] = gpu_stats.get('total_gb', 0.0)
                        capabilities['gpu_count'] = gpu_stats.get('gpu_count', 0)
            except Exception as e:
                logger.debug(f"Could not get capabilities: {e}")
        except Exception as e:
            logger.debug(f"Error getting capabilities: {e}")
        
        return capabilities

