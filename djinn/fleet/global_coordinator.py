"""
Global Fleet Coordinator for Djinn.

Centralized coordinator for fleet-wide optimization:
- Routes requests to best server in fleet
- Coordinates model cache across servers
- Load balances across fleet
- Tracks server health

Design: Stateless service (can be replicated for HA).
"""

import logging
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass

from .model_registry import GlobalModelRegistry
from .server_health import ServerHealthTracker
from .load_balancer import SemanticLoadBalancer

logger = logging.getLogger(__name__)


@dataclass
class FleetStatus:
    """Fleet status information."""
    total_servers: int
    healthy_servers: int
    total_models: int
    total_cache_entries: int
    avg_load_score: float


class GlobalFleetCoordinator:
    """
    Centralized coordinator for fleet-wide optimization.
    
    Provides:
    - Intelligent request routing
    - Global model registry
    - Server health tracking
    - Load balancing
    
    Design: Stateless service (can be replicated for HA).
    """
    
    def __init__(
        self,
        model_registry: Optional[GlobalModelRegistry] = None,
        health_tracker: Optional[ServerHealthTracker] = None,
        load_balancer: Optional[SemanticLoadBalancer] = None,
        redis_url: Optional[str] = None
    ):
        """
        Initialize global fleet coordinator.
        
        Args:
            model_registry: Optional GlobalModelRegistry instance
            health_tracker: Optional ServerHealthTracker instance
            load_balancer: Optional SemanticLoadBalancer instance
            redis_url: Optional Redis URL for model registry
        """
        # Initialize components
        self.model_registry = model_registry or GlobalModelRegistry(redis_url=redis_url)
        self.health_tracker = health_tracker or ServerHealthTracker()
        self.load_balancer = load_balancer or SemanticLoadBalancer()
        
        logger.info("✅ GlobalFleetCoordinator initialized")
    
    async def route_request(
        self,
        fingerprint: str,
        inputs: Dict,
        hints: Optional[Dict] = None,
        load_weights: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Route request to best server in fleet.
        
        Args:
            fingerprint: Model fingerprint
            inputs: Input tensors dict (for size estimation)
            hints: Optional semantic hints (execution_phase, etc.)
            load_weights: Optional load balancing weights
            
        Returns:
            Best server address (e.g., "server-3:5558")
        """
        logger.debug(f"🎯 Routing request for model {fingerprint[:16]}...")
        
        # 1. Check global registry: which servers have this model cached?
        cached_servers = await self.model_registry.find_cached_servers(fingerprint)
        
        if cached_servers:
            # Filter healthy servers
            healthy_cached = await self.health_tracker.filter_healthy(cached_servers)
            
            if healthy_cached:
                # 2. Select best server from cached servers
                # Pre-compute cache map for efficiency
                has_cache_map = {server: True for server in healthy_cached}
                
                best_server = await self.load_balancer.select_best_server(
                    candidates=healthy_cached,
                    semantic_hints=hints,
                    load_weights=load_weights,
                    health_tracker=self.health_tracker,
                    model_registry=self.model_registry,
                    fingerprint=fingerprint,
                    has_cache_map=has_cache_map
                )
                
                logger.info(
                    f"✅ Routed to cached server: {best_server} "
                    f"(from {len(healthy_cached)} healthy cached servers)"
                )
                return best_server
            else:
                logger.warning(
                    f"⚠️  Model cached but no healthy servers, "
                    f"selecting server for registration"
                )
        
        # 3. No cache hit - select server for registration
        all_servers = await self.health_tracker.get_healthy_servers()
        
        if not all_servers:
            raise RuntimeError("No healthy servers available in fleet")
        
        # Estimate model size from inputs (rough estimate)
        model_size_gb = self._estimate_model_size(inputs)
        
        target_server = await self.load_balancer.select_for_registration(
            model_size_gb=model_size_gb,
            semantic_hints=hints,
            all_servers=all_servers,
            health_tracker=self.health_tracker
        )
        
        # 4. Register in global registry (async, non-blocking)
        # Track task to prevent leaks
        if not hasattr(self, '_background_tasks'):
            self._background_tasks = set()
        
        task = asyncio.create_task(
            self._register_model_safe(fingerprint, target_server)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        logger.info(
            f"✅ Routed to server for registration: {target_server} "
            f"(model_size={model_size_gb:.2f}GB)"
        )
        
        return target_server
    
    async def register_server(
        self,
        server_address: str,
        capabilities: Optional[Dict] = None
    ):
        """
        Register a server in the fleet.
        
        Args:
            server_address: Server address (e.g., "server-1:5556")
            capabilities: Optional server capabilities dict
        """
        # Initial health update (server is healthy)
        await self.health_tracker.update_health(server_address, {
            'memory_used_gb': 0.0,
            'memory_total_gb': capabilities.get('memory_total_gb', 0.0) if capabilities else 0.0,
            'gpu_utilization': 0.0,
            'active_requests': 0
        })
        
        logger.info(f"✅ Registered server: {server_address}")
    
    async def unregister_server(self, server_address: str):
        """
        Unregister a server from the fleet.
        
        Args:
            server_address: Server address
        """
        # Mark as unhealthy
        await self.health_tracker.mark_unhealthy(server_address)
        
        # Unregister all models from this server
        await self.model_registry.unregister_server(server_address)
        
        logger.info(f"✅ Unregistered server: {server_address}")
    
    async def update_server_health(
        self,
        server_address: str,
        metrics: Dict
    ):
        """
        Update server health metrics.
        
        Args:
            server_address: Server address
            metrics: Health metrics dict
        """
        await self.health_tracker.update_health(server_address, metrics)
    
    async def get_fleet_status(self) -> FleetStatus:
        """
        Get fleet status information.
        
        Returns:
            FleetStatus object
        """
        all_servers = await self.health_tracker.get_all_servers()
        healthy_servers = await self.health_tracker.get_healthy_servers()
        
        # Get all cached models
        all_models = await self.model_registry.get_all_cached_models()
        total_models = len(all_models)
        total_cache_entries = sum(len(servers) for servers in all_models.values())
        
        # Calculate average load
        load_scores = []
        for server in healthy_servers:
            metrics = await self.health_tracker.get_load_metrics(server)
            if metrics:
                load_scores.append(metrics.get('load_score', 0.0))
        
        avg_load = sum(load_scores) / len(load_scores) if load_scores else 0.0
        
        return FleetStatus(
            total_servers=len(all_servers),
            healthy_servers=len(healthy_servers),
            total_models=total_models,
            total_cache_entries=total_cache_entries,
            avg_load_score=avg_load
        )
    
    def _estimate_model_size(self, inputs: Dict) -> float:
        """
        Estimate model size from inputs (rough estimate).
        
        Args:
            inputs: Input tensors dict
            
        Returns:
            Estimated model size in GB
        """
        # Rough estimate: assume model is 10x input size
        # This is a placeholder - can be enhanced with actual model size
        total_input_size = 0.0
        for value in inputs.values():
            if hasattr(value, 'numel'):
                total_input_size += value.numel() * value.element_size()
        
        # Convert to GB and multiply by 10 (rough estimate)
        model_size_gb = (total_input_size * 10) / (1024 ** 3)
        
        # Cap at reasonable range
        return min(max(model_size_gb, 0.1), 100.0)  # 0.1GB to 100GB
    
    async def _register_model_safe(self, fingerprint: str, server_address: str):
        """Safely register model in background (with error handling)."""
        try:
            await self.model_registry.register_model(fingerprint, server_address)
        except Exception as e:
            logger.error(f"❌ Background model registration failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

