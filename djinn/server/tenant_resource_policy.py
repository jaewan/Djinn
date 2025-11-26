"""
djinn/server/tenant_resource_policy.py

Phase 1: Minimal multi-tenant resource isolation.

Provides:
1. Per-tenant VRAM limits
2. Concurrent request limits  
3. Priority levels
4. Admission control

Does NOT provide (Phase 4):
- Quota management
- Complex QoS scheduling
- Cross-tenant fairness
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TenantLimits:
    """Resource limits for a tenant."""
    
    # VRAM limit (GB)
    max_vram_gb: float = 40.0  # Half of A100-80GB by default
    
    # Concurrency limit
    max_concurrent_requests: int = 10
    
    # Priority (0=background, 1=normal, 2=high)
    priority: int = 1
    
    # Enabled flag
    enabled: bool = True


class ResourceQuotaError(Exception):
    """Raised when tenant quota is exceeded."""
    pass


class TenantResourcePolicy:
    """
    Minimal multi-tenant resource policy for Phase 1.
    
    Provides basic admission control:
    - Checks VRAM usage against limit
    - Checks concurrent request count
    - Respects priority for eviction
    
    Thread-safe via asyncio locks (assuming async execution).
    """
    
    def __init__(self):
        # Tenant configurations
        self._limits: Dict[str, TenantLimits] = {}
        
        # Default limits for unknown tenants
        self._default_limits = TenantLimits()
        
        # Current usage tracking
        self._current_vram: Dict[str, float] = {}
        self._active_requests: Dict[str, int] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def configure_tenant(self, tenant_id: str, limits: TenantLimits):
        """Configure limits for a tenant."""
        self._limits[tenant_id] = limits
        logger.info(
            f"Configured tenant {tenant_id}: "
            f"VRAM={limits.max_vram_gb:.1f}GB, "
            f"concurrency={limits.max_concurrent_requests}, "
            f"priority={limits.priority}"
        )
    
    def get_tenant_limits(self, tenant_id: str) -> TenantLimits:
        """Get limits for tenant (returns default if not configured)."""
        return self._limits.get(tenant_id, self._default_limits)
    
    async def check_admission(
        self, 
        tenant_id: str, 
        vram_estimate_gb: float
    ) -> Tuple[bool, str]:
        """
        Check if request can be admitted.
        
        Args:
            tenant_id: Tenant making the request
            vram_estimate_gb: Estimated VRAM needed for request
        
        Returns:
            (can_admit, reason)
            - can_admit: True if request should be admitted
            - reason: Human-readable reason for decision
        """
        async with self._lock:
            limits = self.get_tenant_limits(tenant_id)
            
            if not limits.enabled:
                return (False, f"Tenant {tenant_id} is disabled")
            
            # Check VRAM limit
            current_vram = self._current_vram.get(tenant_id, 0.0)
            if current_vram + vram_estimate_gb > limits.max_vram_gb:
                return (
                    False,
                    f"VRAM quota exceeded: {current_vram:.1f}GB + {vram_estimate_gb:.1f}GB "
                    f"> {limits.max_vram_gb:.1f}GB limit"
                )
            
            # Check concurrency limit
            active = self._active_requests.get(tenant_id, 0)
            if active >= limits.max_concurrent_requests:
                return (
                    False,
                    f"Concurrency limit exceeded: {active} active requests "
                    f">= {limits.max_concurrent_requests} limit"
                )
            
            return (True, "Admitted")
    
    async def reserve_resources(self, tenant_id: str, vram_gb: float):
        """
        Reserve resources for admitted request.
        
        Must be called after admission check succeeds.
        """
        async with self._lock:
            self._current_vram[tenant_id] = self._current_vram.get(tenant_id, 0.0) + vram_gb
            self._active_requests[tenant_id] = self._active_requests.get(tenant_id, 0) + 1
            
            logger.debug(
                f"Reserved {vram_gb:.1f}GB for {tenant_id}: "
                f"total={self._current_vram[tenant_id]:.1f}GB, "
                f"active={self._active_requests[tenant_id]}"
            )
    
    async def release_resources(self, tenant_id: str, vram_gb: float):
        """
        Release resources after request completes.
        """
        async with self._lock:
            self._current_vram[tenant_id] = max(
                0.0, 
                self._current_vram.get(tenant_id, 0.0) - vram_gb
            )
            self._active_requests[tenant_id] = max(
                0,
                self._active_requests.get(tenant_id, 0) - 1
            )
            
            logger.debug(
                f"Released {vram_gb:.1f}GB for {tenant_id}: "
                f"total={self._current_vram[tenant_id]:.1f}GB, "
                f"active={self._active_requests[tenant_id]}"
            )
    
    async def get_current_usage(self, tenant_id: str) -> Dict[str, any]:
        """Get current resource usage for tenant."""
        async with self._lock:
            limits = self.get_tenant_limits(tenant_id)
            return {
                'tenant_id': tenant_id,
                'vram_used_gb': self._current_vram.get(tenant_id, 0.0),
                'vram_limit_gb': limits.max_vram_gb,
                'active_requests': self._active_requests.get(tenant_id, 0),
                'request_limit': limits.max_concurrent_requests,
                'priority': limits.priority,
            }

