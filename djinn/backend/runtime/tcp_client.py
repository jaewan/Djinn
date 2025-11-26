"""
TCP-based client for remote execution.

Replaces HTTP with persistent TCP connections.
Uses length-prefixed framing for binary safety.
Implements connection pooling for efficiency.
"""

import asyncio
import struct
import torch
import json
import logging
import time
import pickle
from typing import Optional, Dict, Any
import io

logger = logging.getLogger(__name__)


class TCPRemoteExecutionClient:
    """
    Client for executing operations on remote server via TCP.
    
    Uses persistent connections with connection pooling.
    Implements length-prefixed message framing.
    """

    def __init__(self, host: str = "localhost", port: int = 5556, timeout: float = 30.0):
        """
        Initialize TCP client.

        Args:
            host: Server host (default: localhost)
            port: Server port (default: 5556)
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # Connection cache
        self._connection = None
        self._last_activity = 0
        self._connection_timeout = 60.0  # Close idle connections after 60s
        
        # Statistics
        self.stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'total_time_seconds': 0.0,
            'connection_reuses': 0,
        }

        logger.info(f"Created TCPRemoteExecutionClient: {host}:{port}")

    async def _get_connection(self):
        """Get or create TCP connection with pooling."""
        now = time.time()
        
        # Check if existing connection is still valid
        if self._connection is not None:
            try:
                # Check if connection is still open
                reader, writer = self._connection
                if not writer.is_closing():
                    # Check idle timeout
                    if now - self._last_activity < self._connection_timeout:
                        self.stats['connection_reuses'] += 1
                        self._last_activity = now
                        return reader, writer
                    else:
                        logger.info("Closing idle connection")
                        writer.close()
                        await writer.wait_closed()
                        self._connection = None
            except Exception as e:
                logger.warning(f"Connection check failed: {e}")
                self._connection = None
        
        # Create new connection
        try:
            logger.info(f"Opening TCP connection to {self.host}:{self.port}")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout
            )
            # ✅ PHASE 3: Apply TCP optimizations for high-performance transfer
            import socket
            sock = writer.get_extra_info('socket')
            if sock:
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
                    try:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_WINDOW_CLAMP, 64 * 1024 * 1024)
                    except (AttributeError, OSError):
                        pass
                except Exception as e:
                    logger.warning(f"Failed to optimize TCP socket: {e}")
            self._connection = (reader, writer)
            self._last_activity = now
            return reader, writer
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to {self.host}:{self.port}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def _send_message(self, writer, message_type: int, data: bytes) -> None:
        """Send length-prefixed message."""
        # Format: [1 byte: type][4 bytes: length][N bytes: data]
        message = struct.pack('>BI', message_type, len(data)) + data
        writer.write(message)
        await writer.drain()
        self.stats['total_bytes_sent'] += len(message)

    async def _recv_message(self, reader) -> tuple[int, bytes]:
        """Receive length-prefixed message."""
        # Read type and length
        header = await asyncio.wait_for(
            reader.readexactly(5),
            timeout=self.timeout
        )
        msg_type, length = struct.unpack('>BI', header)
        
        # Read data
        data = await asyncio.wait_for(
            reader.readexactly(length),
            timeout=self.timeout
        )
        self.stats['total_bytes_received'] += len(header) + len(data)
        return msg_type, data

    async def execute_subgraph(self,
                              subgraph_request: Dict[str, Any],
                              input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Execute subgraph on remote server.

        Args:
            subgraph_request: Computation graph specification
            input_data: Input tensors

        Returns:
            Result tensor on CPU
        """
        start_time = time.time()
        self.stats['requests_total'] += 1

        try:
            logger.info(f"Executing subgraph: {len(subgraph_request['operations'])} operations")

            # Get connection
            reader, writer = await self._get_connection()

            # Create coordinator message format
            # The server expects subgraph_data to be a dict with a 'subgraph' key
            message = {
                'type': 0x03,  # EXECUTE_SUBGRAPH (coordinator format)
                'subgraph_data': {
                    'subgraph': subgraph_request  # Nest under 'subgraph' key as server expects
                },
                'input_data': {
                    k: pickle.dumps(v) for k, v in input_data.items()
                },
                'timeout': self.timeout
            }

            # Send pickled message
            message_data = pickle.dumps(message)
            await self._send_message(writer, 0x03, message_data)

            # Receive result (type=0x04: RESULT)
            msg_type, result_data = await self._recv_message(reader)
            logger.info(f"Received message type: {msg_type}, data length: {len(result_data)}")

            if msg_type == 0x05:  # ERROR
                try:
                    error_msg = result_data.decode()
                    logger.error(f"Server error: {error_msg}")
                except UnicodeDecodeError:
                    logger.error(f"Server error (binary data): {result_data[:100]}")
                    error_msg = f"Binary error data: {len(result_data)} bytes"
                raise RuntimeError(f"Remote execution failed: {error_msg}")

            if msg_type != 0x04:
                raise RuntimeError(f"Unexpected message type: {msg_type}")

            # Deserialize result
            try:
                if result_data.startswith(b"NUMPY001"):
                    from ..core.serialization import deserialize_tensor
                    result = deserialize_tensor(result_data)
                    logger.debug("Deserialized result using NumPy format")
                else:
                    from ..core.serialization import deserialize_tensor
                    result = deserialize_tensor(result_data)
                    logger.debug("Deserialized result using optimized deserializer")
            except Exception as e:
                logger.warning(f"Deserialization error: {e}, falling back")
                result = torch.load(io.BytesIO(result_data))

            # Update stats
            elapsed = time.time() - start_time
            self.stats['requests_success'] += 1
            self.stats['total_time_seconds'] += elapsed

            logger.info(f"✅ Subgraph execution: {len(subgraph_request['operations'])} ops → "
                       f"{result.shape} in {elapsed:.3f}s")

            return result

        except Exception as e:
            self.stats['requests_failed'] += 1
            logger.error(f"Subgraph execution failed: {e}")
            self._connection = None  # Close connection on error
            raise

    async def execute(self,
                     operation: str,
                     tensor: torch.Tensor,
                     timeout: float = 30.0) -> torch.Tensor:
        """
        Execute single operation on remote server.

        Args:
            operation: Operation name (e.g., 'aten::add')
            tensor: Input tensor
            timeout: Request timeout

        Returns:
            Result tensor on CPU
        """
        start_time = time.time()
        self.stats['requests_total'] += 1

        try:
            logger.debug(f"Executing {operation} on {tensor.shape}")

            # Get connection
            reader, writer = await self._get_connection()

            # Serialize request
            request = json.dumps({
                'operation': operation,
            }).encode()

            # Serialize tensor
            tensor_buffer = io.BytesIO()
            torch.save(tensor, tensor_buffer)
            tensor_data = tensor_buffer.getvalue()

            message_data = request + tensor_data

            # Send request (type=0x02: EXECUTE_OPERATION)
            await self._send_message(writer, 0x02, message_data)

            # Receive result (type=0x04: RESULT)
            msg_type, result_data = await self._recv_message(reader)

            if msg_type == 0x05:  # ERROR
                error_msg = result_data.decode()
                raise RuntimeError(f"Remote execution failed: {error_msg}")

            if msg_type != 0x04:
                raise RuntimeError(f"Unexpected message type: {msg_type}")

            # Deserialize result
            try:
                if result_data.startswith(b"NUMPY001"):
                    from ..core.serialization import deserialize_tensor
                    result = deserialize_tensor(result_data)
                else:
                    from ..core.serialization import deserialize_tensor
                    result = deserialize_tensor(result_data)
            except Exception as e:
                logger.warning(f"Deserialization error: {e}")
                result = torch.load(io.BytesIO(result_data))

            # Update stats
            elapsed = time.time() - start_time
            self.stats['requests_success'] += 1
            self.stats['total_time_seconds'] += elapsed

            logger.debug(f"Executed {operation}: {tensor.shape} → {result.shape} in {elapsed:.3f}s")

            return result

        except Exception as e:
            self.stats['requests_failed'] += 1
            logger.error(f"Operation execution failed: {e}")
            self._connection = None
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()
        if stats['requests_total'] > 0:
            stats['avg_request_time_ms'] = (stats['total_time_seconds'] / stats['requests_total']) * 1000
            stats['success_rate'] = (stats['requests_success'] / stats['requests_total']) * 100

        return stats

    async def close(self):
        """Close connection."""
        if self._connection is not None:
            reader, writer = self._connection
            writer.close()
            await writer.wait_closed()
            self._connection = None
            logger.info("TCP connection closed")


# Backward compatibility alias
RemoteExecutionClient = TCPRemoteExecutionClient
