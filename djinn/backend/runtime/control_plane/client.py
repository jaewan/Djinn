"""
Control plane client implementation.

Simplified wrapper for ControlPlaneServer providing client-side API for coordinator.
Optimized with connection pooling and performance metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional

from .connection_pool import get_connection_pool
from .messages import ControlMessage, MessageType
from .server import ControlPlaneServer
from .types import NodeCapabilities

logger = logging.getLogger(__name__)


class ControlPlane:
    """
    Simplified wrapper for ControlPlaneServer.

    Provides client-side API for coordinator.
    Optimized with connection pooling for better performance.
    """
    def __init__(self, node_id: str, port: int = 5555):
        self.node_id = node_id
        self.port = port
        self.server = ControlPlaneServer(node_id, '0.0.0.0', port)
        self._connection_pool = get_connection_pool()
        
        # Performance metrics
        self._metrics = {
            'negotiations_total': 0,
            'negotiations_success': 0,
            'negotiations_failed': 0,
            'total_time_seconds': 0.0,
            'connection_reuses': 0,
        }

    async def start(self):
        """Start control plane server."""
        await self.server.start()

    async def stop(self):
        """Stop control plane server."""
        await self.server.stop()

    def get_capabilities(self) -> Optional[NodeCapabilities]:
        """Get capabilities from the underlying server."""
        if self.server:
            return self.server.get_capabilities()
        return None

    async def negotiate_transfer(
        self,
        transfer_id: str,
        target: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Negotiate transfer with target (client-side).

        Uses connection pooling for better performance.
        Opens connection, sends TRANSFER_REQUEST, waits for response.
        """
        start_time = time.perf_counter()
        self._metrics['negotiations_total'] += 1
        
        # Normalize target
        if ':' not in target:
            target = f"{target}:{self.port}"
        
        conn = None
        try:
            # Get connection from pool (reuses existing if available)
            conn = await self._connection_pool.acquire(target)
            
            # Track if we reused a connection
            if conn.use_count > 1:
                self._metrics['connection_reuses'] += 1
            
            # Send transfer request
            request = ControlMessage(
                type=MessageType.TRANSFER_REQUEST,
                sender=self.node_id,
                timestamp=time.time(),
                message_id=str(uuid.uuid4()),
                payload={
                    'transfer_id': transfer_id,
                    'tensor_id': transfer_id,  # Use transfer_id as tensor_id for now
                    'source_node': self.node_id,
                    'target_node': target,
                    'size': metadata['size_bytes'],
                    'dtype': metadata['dtype'],
                    'shape': metadata['shape'],
                    'metadata': metadata
                }
            )

            # Send message
            message_str = request.to_json()
            message_data = message_str.encode('utf-8')
            length_data = len(message_data).to_bytes(4, 'big')
            conn.writer.write(length_data + message_data)
            await conn.writer.drain()

            # Wait for response
            length_data = await conn.reader.readexactly(4)
            message_length = int.from_bytes(length_data, 'big')
            message_data = await conn.reader.readexactly(message_length)
            message_str = message_data.decode('utf-8')
            response = ControlMessage.from_json(message_str)

            # Check if accepted
            accepted = response.payload.get('accepted', False)
            
            # Update metrics
            elapsed = time.perf_counter() - start_time
            self._metrics['total_time_seconds'] += elapsed
            
            if accepted:
                self._metrics['negotiations_success'] += 1
            else:
                self._metrics['negotiations_failed'] += 1
                conn.error_count += 1
            
            # Return connection to pool (or close if unhealthy)
            await self._connection_pool.release(conn)
            conn = None
            
            return accepted

        except Exception as e:
            self._metrics['negotiations_failed'] += 1
            if conn:
                conn.error_count += 1
                await self._connection_pool.release(conn)
            logger.error(f"Negotiation failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self._metrics.copy()
        pool_stats = self._connection_pool.get_stats()
        
        if metrics['negotiations_total'] > 0:
            metrics['avg_time_ms'] = (metrics['total_time_seconds'] / metrics['negotiations_total']) * 1000
            metrics['success_rate'] = (metrics['negotiations_success'] / metrics['negotiations_total']) * 100
            metrics['reuse_rate'] = (metrics['connection_reuses'] / metrics['negotiations_total']) * 100
        
        metrics['connection_pool'] = pool_stats
        return metrics
    
    async def cleanup(self):
        """Cleanup idle connections in pool."""
        await self._connection_pool.cleanup_idle()

