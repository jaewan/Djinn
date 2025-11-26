"""
Control plane server implementation.

TCP server for coordinating zero-copy tensor transfers between nodes.
Handles client connections, transfer requests, and lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import uuid
from typing import Dict, List, Optional, Callable, Any

from .client_handler import ClientHandler
from .messages import ControlMessage, MessageType
from .types import NodeCapabilities, TransferRequest, TransferResponse

logger = logging.getLogger(__name__)


class ControlPlaneServer:
    """TCP server for control plane coordination"""
    
    def __init__(self, node_id: str, host: str = '0.0.0.0', port: int = 5555):
        self.node_id = node_id
        self.host = host
        self.port = port
        
        # Server state
        self.server: Optional[asyncio.Server] = None
        self.clients: Dict[str, ClientHandler] = {}
        self.active_transfers: Dict[str, TransferRequest] = {}
        self.start_time = time.time()
        self.total_transfers = 0
        
        # Configuration
        self.max_concurrent_transfers = 100
        self.heartbeat_timeout = 60.0  # seconds
        self.message_timeout = 30.0    # seconds
        self.max_message_size = 1024 * 1024  # 1MB
        
        # Callbacks for transfer events
        self.transfer_callbacks: Dict[str, List[Callable]] = {
            'request': [],
            'ready': [],
            'start': [],
            'complete': [],
            'error': [],
            'cancel': []
        }
        
        # Server capabilities
        self.capabilities = NodeCapabilities(
            node_id=node_id,
            gpu_count=1,  # Will be updated based on actual hardware
            max_transfer_size=10 * 1024 * 1024 * 1024,  # 10GB
            supported_dtypes=['float32', 'float16', 'int32', 'int64', 'uint8'],
            network_bandwidth_gbps=100.0,
            memory_bandwidth_gbps=900.0,
            features=['fragmentation', 'reliability', 'compression', 'heartbeat']
        )
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.server_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self._metrics = {
            'messages_processed': 0,
            'transfers_processed': 0,
            'heartbeats_sent': 0,
            'clients_connected': 0,
            'clients_disconnected': 0,
        }
    
    async def start(self):
        """Start the control plane server (non-blocking)."""
        try:
            # Create server
            self.server = await asyncio.start_server(
                self.handle_client,
                self.host,
                self.port
            )

            # Begin accepting connections without blocking this coroutine
            await self.server.start_serving()

            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            self.cleanup_task = asyncio.create_task(self.cleanup_loop())

            addr = self.server.sockets[0].getsockname()
            logger.info(f"Control plane server started on {addr[0]}:{addr[1]}")

        except Exception as e:
            logger.error(f"Failed to start control plane server: {e}")
            raise
    
    async def stop(self):
        """Stop the control plane server"""
        logger.info("Stopping control plane server...")
        
        # Cancel background tasks gracefully
        pending_tasks = []
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            pending_tasks.append(self.heartbeat_task)
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            pending_tasks.append(self.cleanup_task)

        if pending_tasks:
            # Wait for tasks to handle cancellation, with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=2.0  # 2 second timeout to avoid hanging
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for background tasks to cancel")
            except Exception as e:
                logger.debug(f"Error during task cancellation: {e}")
        
        # Clear task references
        self.heartbeat_task = None
        self.cleanup_task = None
        
        # Close all client connections
        for client in list(self.clients.values()):
            client.connected = False
            try:
                client.writer.close()
                await client.writer.wait_closed()
            except:
                pass
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Control plane server stopped")

    def get_capabilities(self) -> NodeCapabilities:
        """Return server capabilities (compat with tests)."""
        return self.capabilities
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle new client connection"""
        client_handler = ClientHandler(reader, writer, self)
        await client_handler.handle_connection()
    
    def register_client(self, client_id: str, handler: ClientHandler):
        """Register a new client"""
        self.clients[client_id] = handler
        self._metrics['clients_connected'] += 1
        logger.info(f"Registered client: {client_id}")
    
    def unregister_client(self, client_id: str):
        """Unregister a client"""
        if client_id in self.clients:
            del self.clients[client_id]
            self._metrics['clients_disconnected'] += 1
            logger.info(f"Unregistered client: {client_id}")
    
    async def process_transfer_request(self, request: TransferRequest, 
                                     client: ClientHandler) -> TransferResponse:
        """Process a transfer request"""
        try:
            # Validate request
            if request.size > self.capabilities.max_transfer_size:
                return TransferResponse(
                    transfer_id=request.transfer_id,
                    accepted=False,
                    reason=f"Transfer size {request.size} exceeds maximum {self.capabilities.max_transfer_size}"
                )
            
            if request.dtype not in self.capabilities.supported_dtypes:
                return TransferResponse(
                    transfer_id=request.transfer_id,
                    accepted=False,
                    reason=f"Unsupported dtype: {request.dtype}"
                )
            
            if len(self.active_transfers) >= self.max_concurrent_transfers:
                return TransferResponse(
                    transfer_id=request.transfer_id,
                    accepted=False,
                    reason="Maximum concurrent transfers reached"
                )
            
            # Estimate transfer time (simple model)
            estimated_time = request.size / (self.capabilities.network_bandwidth_gbps * 1024**3 / 8)
            
            # Accept transfer
            self.active_transfers[request.transfer_id] = request
            self.total_transfers += 1
            self._metrics['transfers_processed'] += 1
            
            # Call request callbacks
            for callback in self.transfer_callbacks['request']:
                try:
                    await callback(request)
                except Exception as e:
                    logger.error(f"Transfer request callback error: {e}")
            
            return TransferResponse(
                transfer_id=request.transfer_id,
                accepted=True,
                reason="Transfer accepted",
                estimated_time_seconds=estimated_time,
                allocated_gpu=request.target_gpu,
                data_port=5556  # UDP port for data transfer
            )
            
        except Exception as e:
            logger.error(f"Error processing transfer request: {e}")
            return TransferResponse(
                transfer_id=request.transfer_id,
                accepted=False,
                reason=f"Processing error: {e}"
            )
    
    async def handle_transfer_ready(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer ready notification"""
        logger.info(f"Transfer {transfer_id} is ready")
        for callback in self.transfer_callbacks['ready']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer ready callback error: {e}")
    
    async def handle_transfer_start(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer start notification"""
        logger.info(f"Transfer {transfer_id} started")
        for callback in self.transfer_callbacks['start']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer start callback error: {e}")
    
    async def handle_transfer_complete(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer completion"""
        if transfer_id in self.active_transfers:
            del self.active_transfers[transfer_id]
        
        logger.info(f"Transfer {transfer_id} completed")
        for callback in self.transfer_callbacks['complete']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer complete callback error: {e}")
    
    async def handle_transfer_error(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer error"""
        if transfer_id in self.active_transfers:
            del self.active_transfers[transfer_id]
        
        error_msg = payload.get('reason', 'Unknown error')
        logger.error(f"Transfer {transfer_id} failed: {error_msg}")
        
        for callback in self.transfer_callbacks['error']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer error callback error: {e}")
    
    async def handle_transfer_cancel(self, transfer_id: str, payload: Dict[str, Any]):
        """Handle transfer cancellation"""
        if transfer_id in self.active_transfers:
            del self.active_transfers[transfer_id]
        
        logger.info(f"Transfer {transfer_id} cancelled")
        for callback in self.transfer_callbacks['cancel']:
            try:
                await callback(transfer_id, payload)
            except Exception as e:
                logger.error(f"Transfer cancel callback error: {e}")
    
    def add_transfer_callback(self, event: str, callback: Callable):
        """Add a callback for transfer events"""
        if event in self.transfer_callbacks:
            self.transfer_callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    async def heartbeat_loop(self):
        """
        Background heartbeat loop (optimized).
        
        Optimizations:
        - Batch heartbeat messages (reuse same message for all clients)
        - Skip if no clients connected
        - Use gather for parallel sends
        """
        # Pre-create heartbeat message template (reused for all clients)
        try:
            while True:
                try:
                    # Use shorter sleep intervals to check for cancellation more frequently
                    # This allows tasks to respond to cancellation faster
                    for _ in range(30):  # 30 * 1 second = 30 seconds total
                        await asyncio.sleep(1)
                        # Check if we've been cancelled
                        if self.heartbeat_task and self.heartbeat_task.cancelled():
                            raise asyncio.CancelledError()
                    
                    # Skip if no clients
                    if not self.clients:
                        continue
                    
                    # Create heartbeat message once (reused for all)
                    heartbeat_msg = ControlMessage(
                        type=MessageType.HEARTBEAT,
                        sender=self.node_id,
                        timestamp=time.time(),
                        message_id=str(uuid.uuid4()),
                        payload={'timestamp': time.time()}
                    )
                    
                    # Send to all clients in parallel (optimized)
                    clients_list = list(self.clients.values())
                    tasks = [self._send_heartbeat_to_client(client, heartbeat_msg) 
                            for client in clients_list]
                    
                    # Use gather for parallel execution
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Count successful heartbeats
                    successful = sum(1 for r in results if r is True)
                    self._metrics['heartbeats_sent'] += successful
                    
                    if successful < len(clients_list):
                        logger.debug(f"Sent {successful}/{len(clients_list)} heartbeats")
                    
                except asyncio.CancelledError:
                    logger.debug("Heartbeat loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Heartbeat loop error: {e}")
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled (outer)")
            raise
    
    async def _send_heartbeat_to_client(self, client: ClientHandler, 
                                       message: ControlMessage) -> bool:
        """Send heartbeat to a single client (helper for parallel execution)."""
        try:
            await client.send_message(message)
            return True
        except Exception as e:
            logger.debug(f"Failed to send heartbeat to {client.client_id}: {e}")
            return False
    
    async def cleanup_loop(self):
        """Background cleanup loop"""
        try:
            while True:
                try:
                    # Use shorter sleep intervals to check for cancellation more frequently
                    # This allows tasks to respond to cancellation faster
                    for _ in range(60):  # 60 * 1 second = 60 seconds total
                        await asyncio.sleep(1)
                        # Check if we've been cancelled
                        if self.cleanup_task and self.cleanup_task.cancelled():
                            raise asyncio.CancelledError()
                    
                    current_time = time.time()
                    
                    # Check for timed out clients
                    disconnected_clients = []
                    for client_id, client in self.clients.items():
                        if current_time - client.last_heartbeat > self.heartbeat_timeout:
                            disconnected_clients.append(client_id)
                    
                    # Remove timed out clients
                    for client_id in disconnected_clients:
                        logger.warning(f"Removing timed out client: {client_id}")
                        client = self.clients[client_id]
                        client.connected = False
                        self.unregister_client(client_id)
                    
                    # Check for stale transfers
                    stale_transfers = []
                    for transfer_id, transfer in self.active_transfers.items():
                        # Simple timeout check - in practice, this would be more sophisticated
                        if current_time - transfer.timeout_seconds > 3600:  # 1 hour default timeout
                            stale_transfers.append(transfer_id)
                    
                    # Remove stale transfers
                    for transfer_id in stale_transfers:
                        logger.warning(f"Removing stale transfer: {transfer_id}")
                        del self.active_transfers[transfer_id]
                    
                except asyncio.CancelledError:
                    logger.debug("Cleanup loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Cleanup loop error: {e}")
        except asyncio.CancelledError:
            logger.debug("Cleanup loop cancelled (outer)")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server performance metrics."""
        metrics = self._metrics.copy()
        metrics.update({
            'active_clients': len(self.clients),
            'active_transfers': len(self.active_transfers),
            'total_transfers': self.total_transfers,
            'uptime_seconds': time.time() - self.start_time,
        })
        return metrics


# Global server instance
_control_server: Optional[ControlPlaneServer] = None

def get_control_server(node_id: str = None) -> ControlPlaneServer:
    """Get global control server instance"""
    global _control_server
    if _control_server is None:
        if node_id is None:
            node_id = f"genie-node-{socket.gethostname()}"
        _control_server = ControlPlaneServer(node_id)
    return _control_server

async def start_control_server(node_id: str = None, host: str = '0.0.0.0', port: int = 5555):
    """Start the global control server"""
    server = get_control_server(node_id)
    server.host = host
    server.port = port
    await server.start()

async def stop_control_server():
    """Stop the global control server"""
    global _control_server
    if _control_server:
        await _control_server.stop()
        _control_server = None

