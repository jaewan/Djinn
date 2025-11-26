"""
Connection pool for control plane client connections.

Optimizes performance by reusing TCP connections instead of creating
new ones for each transfer negotiation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """Pooled connection wrapper."""
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    target: str
    created_at: float
    last_used: float
    use_count: int = 0
    error_count: int = 0

    def is_healthy(self, max_age: float = 300.0, max_idle: float = 60.0) -> bool:
        """Check if connection is still usable."""
        now = time.time()
        age = now - self.created_at
        idle = now - self.last_used
        
        return (
            not self.writer.is_closing() and
            self.error_count < 3 and
            age < max_age and  # Max 5 min age
            idle < max_idle    # Max 1 min idle
        )


class ControlPlaneConnectionPool:
    """
    Connection pool for control plane client connections.
    
    Features:
    - Per-target connection caching
    - Automatic health checking
    - Connection reuse with keepalive
    - Graceful error handling
    - Performance statistics
    """
    
    def __init__(self, max_per_target: int = 3, max_idle_seconds: float = 60.0):
        """
        Initialize connection pool.
        
        Args:
            max_per_target: Maximum connections per target
            max_idle_seconds: Close idle connections after this time
        """
        self.max_per_target = max_per_target
        self.max_idle_seconds = max_idle_seconds
        self._pools: Dict[str, deque] = {}
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'created': 0,
            'reused': 0,
            'closed': 0,
            'errors': 0,
            'health_check_fails': 0,
        }
    
    async def acquire(self, target: str) -> PooledConnection:
        """
        Get or create connection to target.
        
        Args:
            target: Target address (host:port or host)
            
        Returns:
            PooledConnection ready to use
        """
        # Normalize target
        if ':' not in target:
            target = f"{target}:5555"
        
        async with self._lock:
            if target not in self._pools:
                self._pools[target] = deque(maxlen=self.max_per_target)
        
        pool = self._pools[target]
        
        # Try to reuse existing connection
        while pool:
            try:
                conn = pool.popleft()
                if conn.is_healthy(max_idle=self.max_idle_seconds):
                    conn.last_used = time.time()
                    conn.use_count += 1
                    self._stats['reused'] += 1
                    logger.debug(f"Reusing connection to {target} (uses: {conn.use_count})")
                    return conn
                else:
                    # Connection unhealthy, close it
                    await self._close_connection(conn)
                    self._stats['health_check_fails'] += 1
            except IndexError:
                break
        
        # Create new connection
        return await self._create_connection(target)
    
    async def release(self, conn: PooledConnection):
        """
        Return connection to pool.
        
        Args:
            conn: Connection to return
        """
        if not conn.is_healthy(max_idle=self.max_idle_seconds):
            await self._close_connection(conn)
            return
        
        async with self._lock:
            pool = self._pools.get(conn.target)
            if pool and len(pool) < self.max_per_target:
                pool.append(conn)
                logger.debug(f"Returned connection to pool for {conn.target}")
            else:
                await self._close_connection(conn)
    
    async def _create_connection(self, target: str) -> PooledConnection:
        """Create new connection to target."""
        if ':' in target:
            host, port_str = target.rsplit(':', 1)
            port = int(port_str)
        else:
            host = target
            port = 5555
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0
            )
            
            conn = PooledConnection(
                reader=reader,
                writer=writer,
                target=target,
                created_at=time.time(),
                last_used=time.time(),
                use_count=0,
                error_count=0
            )
            
            self._stats['created'] += 1
            logger.debug(f"Created new connection to {target}")
            return conn
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to create connection to {target}: {e}")
            raise
    
    async def _close_connection(self, conn: PooledConnection):
        """Close connection and update stats."""
        try:
            if not conn.writer.is_closing():
                conn.writer.close()
                await conn.writer.wait_closed()
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")
        finally:
            self._stats['closed'] += 1
    
    async def cleanup_idle(self):
        """Cleanup idle connections."""
        now = time.time()
        to_remove = []
        
        async with self._lock:
            for target, pool in list(self._pools.items()):
                # Check each connection in pool
                for conn in list(pool):
                    if not conn.is_healthy(max_idle=self.max_idle_seconds):
                        pool.remove(conn)
                        to_remove.append(conn)
        
        # Close removed connections
        for conn in to_remove:
            await self._close_connection(conn)
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} idle connections")
    
    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics."""
        return self._stats.copy()
    
    async def close_all(self):
        """Close all connections in pool."""
        async with self._lock:
            all_conns = []
            for pool in self._pools.values():
                all_conns.extend(pool)
                pool.clear()
        
        for conn in all_conns:
            await self._close_connection(conn)
        
        logger.info("Closed all connections in pool")


# Global connection pool instance
_global_pool: Optional[ControlPlaneConnectionPool] = None

def get_connection_pool() -> ControlPlaneConnectionPool:
    """Get global connection pool instance."""
    global _global_pool
    if _global_pool is None:
        _global_pool = ControlPlaneConnectionPool()
    return _global_pool

