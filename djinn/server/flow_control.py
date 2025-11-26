"""
Flow control for TCP server.

Implements credit-based backpressure to prevent server overload.
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FlowControlStats:
    """Statistics for flow control."""
    total_credits: int = 0
    available_credits: int = 0
    blocked_requests: int = 0
    total_acquired: int = 0
    total_released: int = 0


class FlowController:
    """
    Credit-based flow control for TCP server.
    
    Prevents server overload by limiting concurrent requests/data transfer.
    
    Features:
    - Credit-based backpressure
    - Per-client credit tracking
    - Automatic credit recovery
    - Statistics tracking
    """
    
    def __init__(self, max_credits: int = 100, credit_recovery_rate: float = 1.0):
        """
        Initialize flow controller.
        
        Args:
            max_credits: Maximum credits available (default: 100)
            credit_recovery_rate: Credits recovered per second (default: 1.0)
        """
        self.max_credits = max_credits
        self.credit_recovery_rate = credit_recovery_rate
        self._credits = asyncio.Semaphore(max_credits)
        self._lock = asyncio.Lock()
        self._stats = FlowControlStats(
            total_credits=max_credits,
            available_credits=max_credits
        )
        self._last_recovery = time.time()
        
        # Per-client credit tracking (optional)
        self._client_credits: Dict[str, int] = {}
        self._max_credits_per_client = max_credits // 4  # 25% max per client
        
        logger.info(f"FlowController initialized: max_credits={max_credits}, recovery_rate={credit_recovery_rate}/s")
    
    async def acquire(self, size: int = 1, client_id: Optional[str] = None, timeout: float = 5.0) -> bool:
        """
        Acquire credits for sending data.
        
        Args:
            size: Number of credits to acquire (default: 1)
            client_id: Optional client identifier for per-client tracking
            timeout: Maximum time to wait for credits (default: 5.0s)
            
        Returns:
            True if credits acquired, False if timeout
        """
        # Check per-client limit
        if client_id:
            async with self._lock:
                client_credits = self._client_credits.get(client_id, 0)
                if client_credits + size > self._max_credits_per_client:
                    logger.debug(f"Client {client_id} exceeded credit limit ({client_credits + size}/{self._max_credits_per_client})")
                    self._stats.blocked_requests += 1
                    return False
        
        # Try to acquire credits
        try:
            # Acquire credits (size times)
            for _ in range(size):
                await asyncio.wait_for(
                    self._credits.acquire(),
                    timeout=timeout / size if size > 0 else timeout
                )
            
            # Update statistics
            async with self._lock:
                self._stats.available_credits -= size
                self._stats.total_acquired += size
                if client_id:
                    self._client_credits[client_id] = self._client_credits.get(client_id, 0) + size
            
            logger.debug(f"Acquired {size} credits (available: {self._stats.available_credits}/{self.max_credits})")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout acquiring {size} credits (timeout={timeout}s)")
            self._stats.blocked_requests += 1
            return False
    
    async def release(self, size: int = 1, client_id: Optional[str] = None):
        """
        Release credits after data sent/processed.
        
        Args:
            size: Number of credits to release (default: 1)
            client_id: Optional client identifier for per-client tracking
        """
        # Release credits
        for _ in range(size):
            self._credits.release()
        
        # Update statistics
        async with self._lock:
            self._stats.available_credits += size
            self._stats.total_released += size
            if client_id:
                current = self._client_credits.get(client_id, 0)
                self._client_credits[client_id] = max(0, current - size)
                if self._client_credits[client_id] == 0:
                    self._client_credits.pop(client_id, None)
        
        logger.debug(f"Released {size} credits (available: {self._stats.available_credits}/{self.max_credits})")
    
    def get_stats(self) -> Dict:
        """Get flow control statistics."""
        async def _get_stats():
            async with self._lock:
                return {
                    'max_credits': self.max_credits,
                    'available_credits': self._stats.available_credits,
                    'blocked_requests': self._stats.blocked_requests,
                    'total_acquired': self._stats.total_acquired,
                    'total_released': self._stats.total_released,
                    'active_clients': len(self._client_credits),
                }
        # Return sync version for easy access
        return {
            'max_credits': self.max_credits,
            'available_credits': self._credits._value,  # Access semaphore value
            'blocked_requests': self._stats.blocked_requests,
            'total_acquired': self._stats.total_acquired,
            'total_released': self._stats.total_released,
        }
    
    async def reset_client_credits(self, client_id: str):
        """Reset credits for a specific client (e.g., on disconnect)."""
        async with self._lock:
            if client_id in self._client_credits:
                credits = self._client_credits.pop(client_id)
                # Release credits back to pool
                for _ in range(credits):
                    self._credits.release()
                logger.debug(f"Reset {credits} credits for client {client_id}")


# Global flow controller instance (singleton)
_global_flow_controller: Optional[FlowController] = None


def get_flow_controller(max_credits: int = 100, credit_recovery_rate: float = 1.0) -> FlowController:
    """Get or create global flow controller instance."""
    global _global_flow_controller
    if _global_flow_controller is None:
        _global_flow_controller = FlowController(max_credits, credit_recovery_rate)
    return _global_flow_controller

