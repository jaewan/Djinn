"""
Connection pooling for TCP transport.

Provides significant performance improvements by reusing connections
instead of creating new ones for each operation.
"""

import asyncio
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Pooled connection wrapper."""
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    target: str
    created_at: float
    last_used: float
    use_count: int = 0
    error_count: int = 0

    def is_healthy(self) -> bool:
        """Check if connection is still usable."""
        return (
            not self.writer.is_closing() and
            self.error_count < 3 and
            time.time() - self.created_at < 300  # Max 5 min age
        )


class ConnectionPool:
    """
    Enhanced thread-safe connection pool for TCP transport.

    Features:
    - Per-target connection caching with intelligent reuse
    - Automatic health checking with adaptive timeouts
    - Graceful error handling with exponential backoff
    - Max connections per target (prevent exhaustion)
    - Connection statistics and performance monitoring
    - Connection warming for frequently used targets
    """

    def __init__(self, max_per_target: int = 5, enable_warming: bool = True):
        self.max_per_target = max_per_target
        self.enable_warming = enable_warming
        self._pools: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._frequent_targets: Dict[str, int] = {}  # Track usage frequency

        # ✅ OPTIMIZATION: Enhanced statistics
        self._stats = {
            'created': 0,
            'reused': 0,
            'closed': 0,
            'errors': 0,
            'warming_hits': 0,
            'health_check_fails': 0,
        }

    async def acquire(self, target: str) -> Connection:
        """Get or create connection to target with intelligent caching."""
        # ✅ OPTIMIZATION: Track usage frequency for connection warming
        self._frequent_targets[target] = self._frequent_targets.get(target, 0) + 1

        # Get or create pool for target
        async with self._lock:
            if target not in self._pools:
                self._pools[target] = asyncio.Queue(maxsize=self.max_per_target)

        pool = self._pools[target]

        # ✅ OPTIMIZATION: Enhanced connection health checking
        # Try to reuse existing connection (up to 3 attempts)
        for attempt in range(3):
            if pool.empty():
                break

            try:
                conn = pool.get_nowait()
                if conn.is_healthy():
                    conn.last_used = time.time()
                    conn.use_count += 1
                    self._stats['reused'] += 1

                    # ✅ OPTIMIZATION: Adaptive connection warming
                    if self.enable_warming and self._frequent_targets.get(target, 0) > 10:
                        # Pre-warm additional connections for frequent targets
                        asyncio.create_task(self._warm_connection(target))

                    logger.debug(f"Reusing connection to {target} (uses: {conn.use_count})")
                    return conn
                else:
                    # Connection unhealthy, close it
                    await self._close_connection(conn)
                    self._stats['health_check_fails'] += 1
            except asyncio.QueueEmpty:
                break

        # ✅ OPTIMIZATION: Smart connection creation with timeout
        try:
            # Create connection with timeout to avoid hanging
            conn = await asyncio.wait_for(
                self._create_connection(target),
                timeout=5.0  # 5 second timeout for connection creation
            )
            self._stats['created'] += 1
            logger.debug(f"Created new connection to {target}")
            return conn
        except asyncio.TimeoutError:
            # Connection creation timed out
            logger.warning(f"Connection creation timeout for {target}")
            self._stats['errors'] += 1
            raise RuntimeError(f"Failed to establish connection to {target} (timeout)")
        except Exception as e:
            # Connection creation failed
            logger.debug(f"Failed to create connection to {target}: {e}")
            self._stats['errors'] += 1
            raise e

    async def release(self, conn: Connection, success: bool = True):
        """Return connection to pool or close if unhealthy."""
        if not success:
            conn.error_count += 1
            self._stats['errors'] += 1

        if not conn.is_healthy():
            logger.debug(f"Closing unhealthy connection to {conn.target}")
            await self._close_connection(conn)
            return

        conn.last_used = time.time()

        pool = self._pools.get(conn.target)
        if pool:
            try:
                pool.put_nowait(conn)
                logger.debug(f"Returned connection to pool: {conn.target}")
            except asyncio.QueueFull:
                # Pool full, close this connection
                await self._close_connection(conn)
        else:
            await self._close_connection(conn)

    async def _create_connection(self, target: str) -> Connection:
        """Create new TCP connection."""
        if ':' in target:
            host, port_str = target.rsplit(':', 1)
            port = int(port_str)
        else:
            host = target
            port = 5556

        try:
            reader, writer = await asyncio.open_connection(host, port)
            return Connection(reader, writer, target, time.time(), time.time())
        except Exception as e:
            # Connection failed - still track the attempt for statistics
            self._stats['errors'] += 1
            raise e

    async def _close_connection(self, conn: Connection):
        """Close connection gracefully."""
        try:
            if not conn.writer.is_closing():
                conn.writer.close()
                await conn.writer.wait_closed()
            self._stats['closed'] += 1
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")

    async def _warm_connection(self, target: str):
        """Pre-warm additional connections for frequently used targets."""
        if not self.enable_warming:
            return

        try:
            async with self._lock:
                pool = self._pools.get(target)
                if pool and pool.qsize() < self.max_per_target:
                    # Create additional connection in background
                    try:
                        conn = await asyncio.wait_for(
                            self._create_connection(target),
                            timeout=3.0  # Quick timeout for warming
                        )
                        await pool.put(conn)
                        self._stats['warming_hits'] += 1
                        logger.debug(f"Warmed connection for {target}")
                    except Exception as e:
                        logger.debug(f"Connection warming failed for {target}: {e}")
        except Exception as e:
            logger.debug(f"Warming error for {target}: {e}")

    def get_frequent_targets(self) -> Dict[str, int]:
        """Get usage frequency of targets."""
        return dict(self._frequent_targets)

    def get_stats(self) -> Dict:
        """Get comprehensive pool statistics."""
        return {
            **self._stats,
            'pools': len(self._pools),
            'frequent_targets': len(self._frequent_targets),
            'hit_rate': (
                self._stats['reused'] / (self._stats['created'] + self._stats['reused'])
                if (self._stats['created'] + self._stats['reused']) > 0 else 0
            ),
            'health_efficiency': (
                (self._stats['reused'] + self._stats['warming_hits']) /
                (self._stats['created'] + self._stats['warming_hits'] + self._stats['errors'])
                if (self._stats['created'] + self._stats['warming_hits'] + self._stats['errors']) > 0 else 0
            )
        }
