"""
Semantic Load Balancer for Djinn Fleet Coordinator.

Intelligently selects best server based on:
- Cache availability (prefer servers with cached models)
- Load metrics (memory, GPU utilization, active requests)
- Latency
- Semantic hints (execution phase, etc.)
"""

import logging
import random
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .server_health import ServerHealthTracker
    from .model_registry import GlobalModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ServerCandidate:
    """Candidate server for load balancing."""
    server_address: str
    has_cache: bool = False
    load_score: float = 0.0  # 0.0 = no load, 1.0 = fully loaded
    latency_ms: float = 0.0
    semantic_score: float = 0.0  # Semantic match score (0.0 to 1.0)
    
    def get_total_score(self, weights: Dict[str, float]) -> float:
        """
        Get total score for server selection (lower is better).
        
        Args:
            weights: Dict with keys 'cache', 'load', 'latency', 'semantic'
            
        Returns:
            Total score (lower is better)
        """
        # Cache preference (higher is better, so invert)
        cache_score = 0.0 if self.has_cache else 1.0
        
        # Normalize latency (assume 100ms is "high" latency)
        latency_score = min(self.latency_ms / 100.0, 1.0)
        
        # Combine scores (weighted)
        total = (
            cache_score * weights.get('cache', 0.3) +
            self.load_score * weights.get('load', 0.3) +
            latency_score * weights.get('latency', 0.2) +
            (1.0 - self.semantic_score) * weights.get('semantic', 0.2)  # Invert semantic (higher is better)
        )
        
        return total


class SemanticLoadBalancer:
    """
    Semantic load balancer for fleet-wide request routing.
    
    Features:
    - Prefer servers with cached models
    - Load-aware selection (prevent hot spots)
    - Latency-aware routing
    - Semantic hint consideration
    """
    
    def __init__(self):
        """Initialize semantic load balancer."""
        logger.info("✅ SemanticLoadBalancer initialized")
    
    async def select_best_server(
        self,
        candidates: List[str],
        semantic_hints: Optional[Dict] = None,
        load_weights: Optional[Dict[str, float]] = None,
        health_tracker: Optional['ServerHealthTracker'] = None,
        model_registry: Optional['GlobalModelRegistry'] = None,
        fingerprint: Optional[str] = None,
        has_cache_map: Optional[Dict[str, bool]] = None
    ) -> str:
        """
        Select best server from candidates.
        
        Args:
            candidates: List of candidate server addresses
            semantic_hints: Optional semantic hints (execution_phase, etc.)
            load_weights: Optional weights for load balancing
                (default: {'cache': 0.3, 'load': 0.3, 'latency': 0.2, 'semantic': 0.2})
            health_tracker: Optional ServerHealthTracker for load metrics
            model_registry: Optional GlobalModelRegistry for cache checking
            fingerprint: Optional model fingerprint for cache checking
            
        Returns:
            Best server address
        """
        if not candidates:
            raise ValueError("No candidate servers provided")
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Default weights
        weights = load_weights or {
            'cache': 0.3,
            'load': 0.3,
            'latency': 0.2,
            'semantic': 0.2
        }
        
        # Build candidate list with scores
        server_candidates = []
        for server in candidates:
            candidate = ServerCandidate(server_address=server)
            
            # Check cache availability
            # Use has_cache_map if provided (pre-computed for efficiency)
            if has_cache_map is not None:
                candidate.has_cache = has_cache_map.get(server, False)
            else:
                # Fallback: assume no cache if not provided (safe default)
                candidate.has_cache = False
            
            # Get load metrics (async)
            if health_tracker:
                try:
                    metrics = await health_tracker.get_load_metrics(server)
                    if metrics:
                        candidate.load_score = metrics.get('load_score', 0.5)
                        candidate.latency_ms = metrics.get('avg_latency_ms', 0.0)
                    else:
                        candidate.load_score = 0.5
                        candidate.latency_ms = 0.0
                except Exception as e:
                    logger.debug(f"Failed to get load metrics for {server}: {e}")
                    # Fallback to default
                    candidate.load_score = 0.5
                    candidate.latency_ms = 0.0
            
            # Semantic score (placeholder - can be enhanced)
            candidate.semantic_score = self._compute_semantic_score(server, semantic_hints)
            
            server_candidates.append(candidate)
        
        # Select best server (lowest total score)
        best = min(server_candidates, key=lambda c: c.get_total_score(weights))
        
        logger.info(
            f"🎯 Selected server {best.server_address} "
            f"(load={best.load_score:.2f}, latency={best.latency_ms:.1f}ms, "
            f"semantic={best.semantic_score:.2f})"
        )
        
        return best.server_address
    
    async def select_with_backpressure(
        self,
        candidates: List[str],
        semantic_hints: Optional[Dict] = None,
        health_tracker: Optional['ServerHealthTracker'] = None,
        max_load_threshold: float = 0.9
    ) -> str:
        """
        Select server with backpressure (avoid overloaded servers).
        
        Args:
            candidates: List of candidate server addresses
            semantic_hints: Optional semantic hints
            health_tracker: Optional ServerHealthTracker
            max_load_threshold: Maximum load threshold (0.0 to 1.0)
            
        Returns:
            Selected server address
        """
        if not candidates:
            raise ValueError("No candidate servers provided")
        
        # Filter by load threshold
        available = []
        for server in candidates:
            if health_tracker:
                metrics = await health_tracker.get_load_metrics(server)
                if metrics:
                    load_score = metrics.get('load_score', 0.0)
                    if load_score < max_load_threshold:
                        available.append(server)
                    else:
                        logger.debug(
                            f"⚠️  Server {server} overloaded "
                            f"(load={load_score:.2f} > {max_load_threshold})"
                        )
                else:
                    available.append(server)  # No metrics, assume available
            else:
                available.append(server)
        
        # If no servers available, use original candidates (best effort)
        if not available:
            logger.warning("⚠️  No servers below load threshold, using best effort")
            available = candidates
        
        # Select from available servers
        return await self.select_best_server(
            candidates=available,
            semantic_hints=semantic_hints,
            health_tracker=health_tracker
        )
    
    async def select_for_registration(
        self,
        model_size_gb: float,
        semantic_hints: Optional[Dict] = None,
        all_servers: Optional[List[str]] = None,
        health_tracker: Optional['ServerHealthTracker'] = None
    ) -> str:
        """
        Select server for model registration.
        
        Args:
            model_size_gb: Estimated model size in GB
            semantic_hints: Optional semantic hints
            all_servers: List of all available servers
            health_tracker: Optional ServerHealthTracker
            
        Returns:
            Selected server address
        """
        if not all_servers:
            raise ValueError("No servers available for registration")
        
        # Filter servers with enough memory
        candidates = []
        for server in all_servers:
            if health_tracker:
                metrics = await health_tracker.get_load_metrics(server)
                if metrics:
                    memory_utilization = metrics.get('memory_utilization', 0.0)
                    memory_total_gb = metrics.get('memory_total_gb', 0.0)
                    
                    # Check if server has enough free memory
                    free_memory_gb = memory_total_gb * (1.0 - memory_utilization)
                    if free_memory_gb >= model_size_gb * 1.2:  # 20% buffer
                        candidates.append(server)
                    else:
                        logger.debug(
                            f"⚠️  Server {server} insufficient memory "
                            f"(free={free_memory_gb:.1f}GB < required={model_size_gb * 1.2:.1f}GB)"
                        )
                else:
                    candidates.append(server)  # No metrics, assume available
            else:
                candidates.append(server)
        
        # If no servers have enough memory, use all servers (best effort)
        if not candidates:
            logger.warning("⚠️  No servers with sufficient memory, using best effort")
            candidates = all_servers
        
        # Select server with lowest load
        return await self.select_best_server(
            candidates=candidates,
            semantic_hints=semantic_hints,
            health_tracker=health_tracker,
            load_weights={'cache': 0.0, 'load': 0.6, 'latency': 0.2, 'semantic': 0.2}
        )
    
    def _compute_semantic_score(self, server: str, semantic_hints: Optional[Dict]) -> float:
        """
        Compute semantic match score for a server.
        
        Args:
            server: Server address
            semantic_hints: Optional semantic hints
            
        Returns:
            Semantic score (0.0 to 1.0, higher is better)
        """
        if not semantic_hints:
            return 0.5  # Neutral score
        
        # Placeholder implementation
        # Can be enhanced with:
        # - Execution phase matching (prefill vs decode)
        # - GPU type matching (compute vs memory optimized)
        # - Network topology awareness
        
        return 0.5

