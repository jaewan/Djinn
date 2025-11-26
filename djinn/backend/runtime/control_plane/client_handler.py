"""
Client connection handler for control plane server.

Handles individual client connections, message processing, and lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict
from typing import Optional, Set

from .messages import ControlMessage, MessageType
from .types import NodeCapabilities, TransferRequest, TransferResponse

logger = logging.getLogger(__name__)


class ClientHandler:
    """Handles individual client connections"""
    
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, 
                 server: 'ControlPlaneServer'):
        self.reader = reader
        self.writer = writer
        self.server = server
        self.client_id: Optional[str] = None
        self.capabilities: Optional[NodeCapabilities] = None
        self.last_heartbeat = time.time()
        self.connected = True
        self.pending_transfers: Set[str] = set()
        
        # Get client address
        peername = writer.get_extra_info('peername')
        self.client_address = f"{peername[0]}:{peername[1]}" if peername else "unknown"
        
        logger.info(f"New client connection from {self.client_address}")
    
    async def handle_connection(self):
        """Main connection handling loop"""
        try:
            # Send welcome message
            await self.send_hello()
            
            # Process messages
            while self.connected:
                try:
                    message = await self.receive_message()
                    if message:
                        await self.process_message(message)
                    else:
                        break  # Connection closed
                except asyncio.TimeoutError:
                    # Check for heartbeat timeout
                    if time.time() - self.last_heartbeat > self.server.heartbeat_timeout:
                        logger.warning(f"Client {self.client_id} heartbeat timeout")
                        break
                except Exception as e:
                    logger.error(f"Error processing message from {self.client_id}: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Connection error with {self.client_address}: {e}")
        
        finally:
            await self.cleanup()
    
    async def send_hello(self):
        """Send initial hello message"""
        hello_msg = ControlMessage(
            type=MessageType.HELLO,
            sender=self.server.node_id,
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
            payload={
                'server_version': '1.0',
                'supported_features': ['transfer_coordination', 'heartbeat', 'status_monitoring'],
                'max_concurrent_transfers': self.server.max_concurrent_transfers
            }
        )
        await self.send_message(hello_msg)
    
    async def receive_message(self) -> Optional[ControlMessage]:
        """Receive and parse a message"""
        try:
            # Read message length (4 bytes, big endian)
            length_data = await asyncio.wait_for(
                self.reader.readexactly(4), 
                timeout=self.server.message_timeout
            )
            message_length = int.from_bytes(length_data, 'big')
            
            if message_length > self.server.max_message_size:
                raise ValueError(f"Message too large: {message_length} bytes")
            
            # Read message data
            message_data = await asyncio.wait_for(
                self.reader.readexactly(message_length),
                timeout=self.server.message_timeout
            )
            
            # Parse JSON message
            message_str = message_data.decode('utf-8')
            return ControlMessage.from_json(message_str)
            
        except asyncio.IncompleteReadError:
            # Connection closed
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    async def send_message(self, message: ControlMessage):
        """Send a message to the client"""
        try:
            message_str = message.to_json()
            message_data = message_str.encode('utf-8')
            
            # Send length prefix (4 bytes, big endian)
            length_data = len(message_data).to_bytes(4, 'big')
            self.writer.write(length_data + message_data)
            await self.writer.drain()
            
            logger.debug(f"Sent {message.type.name} to {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error sending message to {self.client_id}: {e}")
            self.connected = False
    
    async def process_message(self, message: ControlMessage):
        """Process received message"""
        self.last_heartbeat = time.time()
        
        logger.debug(f"Received {message.type.name} from {message.sender}")
        
        # Update client ID if not set
        if not self.client_id:
            self.client_id = message.sender
            self.server.register_client(self.client_id, self)
        
        # Dispatch message by type
        handler_map = {
            MessageType.CAPABILITY_EXCHANGE: self.handle_capability_exchange,
            MessageType.HEARTBEAT: self.handle_heartbeat,
            MessageType.TRANSFER_REQUEST: self.handle_transfer_request,
            MessageType.TRANSFER_READY: self.handle_transfer_ready,
            MessageType.TRANSFER_START: self.handle_transfer_start,
            MessageType.TRANSFER_COMPLETE: self.handle_transfer_complete,
            MessageType.TRANSFER_ERROR: self.handle_transfer_error,
            MessageType.TRANSFER_CANCEL: self.handle_transfer_cancel,
            MessageType.STATUS_REQUEST: self.handle_status_request,
            MessageType.NODE_LIST_REQUEST: self.handle_node_list_request,
            MessageType.GOODBYE: self.handle_goodbye,
        }
        
        handler = handler_map.get(message.type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"Unknown message type: {message.type}")
    
    async def handle_capability_exchange(self, message: ControlMessage):
        """Handle capability exchange message"""
        try:
            cap_data = message.payload
            self.capabilities = NodeCapabilities(
                node_id=message.sender,
                gpu_count=cap_data.get('gpu_count', 1),
                max_transfer_size=cap_data.get('max_transfer_size', 10 * 1024 * 1024 * 1024),
                supported_dtypes=cap_data.get('supported_dtypes', ['float32']),
                network_bandwidth_gbps=cap_data.get('network_bandwidth_gbps', 100.0),
                memory_bandwidth_gbps=cap_data.get('memory_bandwidth_gbps', 900.0),
                features=cap_data.get('features', [])
            )
            
            logger.info(f"Received capabilities from {self.client_id}: "
                       f"{self.capabilities.gpu_count} GPUs, "
                       f"{self.capabilities.network_bandwidth_gbps} Gbps network")
            
            # Send our capabilities back
            response = ControlMessage(
                type=MessageType.CAPABILITY_EXCHANGE,
                sender=self.server.node_id,
                timestamp=time.time(),
                message_id=str(uuid.uuid4()),
                payload=asdict(self.server.capabilities)
            )
            await self.send_message(response)
            
        except Exception as e:
            logger.error(f"Error handling capability exchange: {e}")
    
    async def handle_heartbeat(self, message: ControlMessage):
        """Handle heartbeat message"""
        # Heartbeat received, update timestamp
        self.last_heartbeat = time.time()
        
        # Send heartbeat response
        response = ControlMessage(
            type=MessageType.HEARTBEAT,
            sender=self.server.node_id,
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
            payload={'status': 'alive', 'timestamp': time.time()}
        )
        await self.send_message(response)
    
    async def handle_transfer_request(self, message: ControlMessage):
        """Handle transfer request"""
        try:
            req_data = message.payload
            transfer_request = TransferRequest(
                transfer_id=req_data['transfer_id'],
                tensor_id=req_data['tensor_id'],
                source_node=req_data['source_node'],
                target_node=req_data['target_node'],
                size=req_data['size'],
                dtype=req_data['dtype'],
                shape=req_data['shape'],
                source_gpu=req_data.get('source_gpu', 0),
                target_gpu=req_data.get('target_gpu', 0),
                priority=req_data.get('priority', 1),
                timeout_seconds=req_data.get('timeout_seconds', 300.0),
                requires_ack=req_data.get('requires_ack', True),
                compression=req_data.get('compression', False),
                metadata=req_data.get('metadata', {})
            )
            
            # Process transfer request
            response = await self.server.process_transfer_request(transfer_request, self)
            
            # Send response
            response_msg = ControlMessage(
                type=MessageType.TRANSFER_READY if response.accepted else MessageType.TRANSFER_ERROR,
                sender=self.server.node_id,
                timestamp=time.time(),
                message_id=str(uuid.uuid4()),
                payload=asdict(response)
            )
            await self.send_message(response_msg)
            
            if response.accepted:
                self.pending_transfers.add(transfer_request.transfer_id)
            
        except Exception as e:
            logger.error(f"Error handling transfer request: {e}")
            error_response = TransferResponse(
                transfer_id=req_data.get('transfer_id', 'unknown'),
                accepted=False,
                reason=f"Processing error: {e}"
            )
            error_msg = ControlMessage(
                type=MessageType.TRANSFER_ERROR,
                sender=self.server.node_id,
                timestamp=time.time(),
                message_id=str(uuid.uuid4()),
                payload=asdict(error_response)
            )
            await self.send_message(error_msg)
    
    async def handle_transfer_ready(self, message: ControlMessage):
        """Handle transfer ready notification"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            await self.server.handle_transfer_ready(transfer_id, message.payload)
    
    async def handle_transfer_start(self, message: ControlMessage):
        """Handle transfer start notification"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            await self.server.handle_transfer_start(transfer_id, message.payload)
    
    async def handle_transfer_complete(self, message: ControlMessage):
        """Handle transfer completion"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            self.pending_transfers.discard(transfer_id)
            await self.server.handle_transfer_complete(transfer_id, message.payload)
    
    async def handle_transfer_error(self, message: ControlMessage):
        """Handle transfer error"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            self.pending_transfers.discard(transfer_id)
            await self.server.handle_transfer_error(transfer_id, message.payload)
    
    async def handle_transfer_cancel(self, message: ControlMessage):
        """Handle transfer cancellation"""
        transfer_id = message.payload.get('transfer_id')
        if transfer_id:
            self.pending_transfers.discard(transfer_id)
            await self.server.handle_transfer_cancel(transfer_id, message.payload)
    
    async def handle_status_request(self, message: ControlMessage):
        """Handle status request"""
        status = {
            'node_id': self.server.node_id,
            'uptime_seconds': time.time() - self.server.start_time,
            'active_connections': len(self.server.clients),
            'active_transfers': len(self.server.active_transfers),
            'total_transfers': self.server.total_transfers,
            'capabilities': asdict(self.server.capabilities)
        }
        
        response = ControlMessage(
            type=MessageType.STATUS_RESPONSE,
            sender=self.server.node_id,
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
            payload=status
        )
        await self.send_message(response)
    
    async def handle_node_list_request(self, message: ControlMessage):
        """Handle node list request"""
        nodes = []
        for client_id, client_handler in self.server.clients.items():
            if client_handler.capabilities:
                nodes.append({
                    'node_id': client_id,
                    'address': client_handler.client_address,
                    'capabilities': asdict(client_handler.capabilities),
                    'connected_time': time.time() - client_handler.last_heartbeat,
                    'pending_transfers': len(client_handler.pending_transfers)
                })
        
        response = ControlMessage(
            type=MessageType.NODE_LIST_RESPONSE,
            sender=self.server.node_id,
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
            payload={'nodes': nodes}
        )
        await self.send_message(response)
    
    async def handle_goodbye(self, message: ControlMessage):
        """Handle goodbye message"""
        logger.info(f"Client {self.client_id} sent goodbye")
        self.connected = False
    
    async def cleanup(self):
        """Cleanup client connection"""
        try:
            if self.client_id:
                self.server.unregister_client(self.client_id)
                logger.info(f"Client {self.client_id} disconnected")
            
            self.writer.close()
            await self.writer.wait_closed()
            
        except Exception as e:
            logger.error(f"Error during client cleanup: {e}")

