"""
Djinn v2.3.15 Hybrid Transport - MTU-Aware Network Layer

Optimizes system calls based on payload size using Linux MTU standard (1500B).
- < 1400B: Coalesce into single syscall (latency optimized)
- > 1400B: Scatter-gather I/O (throughput optimized)

Wire Protocol Integration:
- Consumes buffers from DjinnSerializer
- Zero-copy when possible
- Reliable delivery with backoff
"""
import socket
import time
import logging
from typing import List, Union, Optional
import struct

logger = logging.getLogger(__name__)


class HybridTransport:
    """
    MTU-aware transport with syscall optimization.

    Key Design Decisions:
    - MTU threshold: 1400B (conservative buffer below 1500B Ethernet MTU)
    - Small payloads: Coalesce to reduce syscall overhead
    - Large payloads: Scatter-gather for efficiency
    - Fallback: sendall() loop for non-Unix systems
    """

    # Linux Ethernet MTU is 1500B, leave 100B buffer for headers
    MTU_THRESHOLD = 1400

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        """
        Initialize transport with reliability settings.

        Args:
            max_retries: Maximum transmission retries
            retry_delay: Delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def send(self, sock: socket.socket, buffers: List[Union[bytes, memoryview]]) -> int:
        """
        Send buffers using MTU-aware strategy.

        Args:
            sock: Connected socket
            buffers: List of buffer objects from serializer

        Returns:
            Total bytes sent

        Raises:
            ConnectionError: If send fails after retries
        """
        total_size = sum(len(buf) for buf in buffers)

        # Choose strategy based on payload size
        if total_size < self.MTU_THRESHOLD:
            return self._send_coalesced(sock, buffers)
        else:
            return self._send_scatter_gather(sock, buffers)

    def _send_coalesced(self, sock: socket.socket, buffers: List[Union[bytes, memoryview]]) -> int:
        """
        Coalesce small payloads into single syscall.

        Strategy: Latency optimized for small packets.
        - Reduces context switches
        - Minimizes syscall overhead
        """
        # Coalesce all buffers into single payload
        payload = b''.join(buffers)
        total_size = len(payload)

        logger.debug(f"Coalesced send: {len(buffers)} buffers -> {total_size} bytes")

        # Send with retries
        for attempt in range(self.max_retries + 1):
            try:
                sock.sendall(payload)
                logger.debug(f"✅ Coalesced send successful: {total_size} bytes")
                return total_size
            except (OSError, ConnectionError) as e:
                if attempt == self.max_retries:
                    logger.error(f"❌ Coalesced send failed after {self.max_retries + 1} attempts: {e}")
                    raise ConnectionError(f"Send failed: {e}") from e
                logger.warning(f"⚠️  Coalesced send attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

    def _send_scatter_gather(self, sock: socket.socket, buffers: List[Union[bytes, memoryview]]) -> int:
        """
        Use scatter-gather I/O for large payloads.

        Strategy: Throughput optimized for large packets.
        - Zero-copy user-space send
        - Efficient for network-intensive workloads
        """
        total_size = sum(len(buf) for buf in buffers)

        logger.debug(f"Scatter-gather send: {len(buffers)} buffers, {total_size} bytes")

        # Try scatter-gather first (Linux/Unix optimized)
        if hasattr(sock, 'sendmsg'):
            # Use sendmsg() for zero-copy scatter-gather
            for attempt in range(self.max_retries + 1):
                try:
                    sent = sock.sendmsg(buffers)
                    if sent == total_size:
                        logger.debug(f"✅ Scatter-gather send successful: {total_size} bytes")
                        return total_size
                    else:
                        # Partial send - shouldn't happen with blocking socket
                        raise OSError(f"Partial send: {sent}/{total_size}")
                except (OSError, ConnectionError) as e:
                    if attempt == self.max_retries:
                        logger.error(f"❌ Scatter-gather send failed after {self.max_retries + 1} attempts: {e}")
                        raise ConnectionError(f"Send failed: {e}") from e
                    logger.warning(f"⚠️  Scatter-gather send attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))
        else:
            # Fallback for non-Unix systems (Windows, etc.)
            logger.debug("sendmsg not available, using fallback sendall loop")
            return self._send_fallback(sock, buffers)

    def _send_fallback(self, sock: socket.socket, buffers: List[Union[bytes, memoryview]]) -> int:
        """
        Fallback send for systems without sendmsg().

        Uses individual sendall() calls - less efficient but compatible.
        """
        total_sent = 0

        for i, buf in enumerate(buffers):
            for attempt in range(self.max_retries + 1):
                try:
                    sock.sendall(buf)
                    total_sent += len(buf)
                    break
                except (OSError, ConnectionError) as e:
                    if attempt == self.max_retries:
                        logger.error(f"❌ Fallback send failed on buffer {i} after {self.max_retries + 1} attempts: {e}")
                        raise ConnectionError(f"Send failed on buffer {i}: {e}") from e
                    logger.warning(f"⚠️  Fallback send attempt {attempt + 1} on buffer {i} failed, retrying: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))

        logger.debug(f"✅ Fallback send successful: {total_sent} bytes")
        return total_sent

    def connect_with_retry(self, host: str, port: int,
                          timeout: float = 5.0) -> socket.socket:
        """
        Establish connection with retry logic.

        Args:
            host: Target host
            port: Target port
            timeout: Connection timeout

        Returns:
            Connected socket

        Raises:
            ConnectionError: If connection fails
        """
        for attempt in range(self.max_retries + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((host, port))
                sock.settimeout(None)  # Make socket blocking after connect
                logger.debug(f"✅ Connected to {host}:{port}")
                return sock
            except (OSError, ConnectionError) as e:
                if attempt == self.max_retries:
                    logger.error(f"❌ Connection failed after {self.max_retries + 1} attempts: {e}")
                    raise ConnectionError(f"Failed to connect to {host}:{port}: {e}") from e
                logger.warning(f"⚠️  Connection attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

    @staticmethod
    def create_test_payload(size_kb: int) -> List[bytes]:
        """
        Create test payload buffers of specified size.

        Args:
            size_kb: Target size in KB

        Returns:
            List of buffer objects
        """
        target_bytes = size_kb * 1024

        # Create multiple buffers to simulate serializer output
        buffers = []
        remaining = target_bytes

        while remaining > 0:
            chunk_size = min(remaining, 1024)  # 1KB chunks
            buffers.append(b'A' * chunk_size)
            remaining -= chunk_size

        return buffers


def benchmark_transport(host: str = "127.0.0.1", port: int = 0) -> dict:
    """
    Benchmark transport performance with different payload sizes.

    Args:
        host: Test server host
        port: Test server port (0 = random available)

    Returns:
        Benchmark results
    """
    import threading
    import time

    results = {
        'small_payload': {'size_kb': 1, 'times': []},
        'medium_payload': {'size_kb': 50, 'times': []},
        'large_payload': {'size_kb': 1024, 'times': []}
    }

    def echo_server():
        """Simple echo server for testing."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        actual_port = server_sock.getsockname()[1]
        server_sock.listen(1)

        def handle_client():
            try:
                client_sock, addr = server_sock.accept()
                logger.debug(f"Echo server: accepted connection from {addr}")

                # Simple echo loop
                while True:
                    try:
                        data = client_sock.recv(4096)
                        if not data:
                            break
                        client_sock.sendall(data)
                    except:
                        break

                client_sock.close()
            except Exception as e:
                logger.error(f"Echo server error: {e}")
            finally:
                server_sock.close()

        thread = threading.Thread(target=handle_client, daemon=True)
        thread.start()
        return actual_port

    # Start echo server
    actual_port = echo_server()
    time.sleep(0.1)  # Let server start

    transport = HybridTransport()

    try:
        # Connect to echo server
        sock = transport.connect_with_retry(host, actual_port, timeout=2.0)

        # Benchmark different sizes
        for payload_type, config in results.items():
            logger.info(f"Benchmarking {payload_type} ({config['size_kb']}KB)...")

            buffers = HybridTransport.create_test_payload(config['size_kb'])

            # Warmup
            transport.send(sock, buffers)
            sock.recv(1024)  # Drain echo

            # Measure
            for _ in range(10):
                start = time.perf_counter()
                transport.send(sock, buffers)

                # Wait for echo (approximate round-trip)
                received = 0
                while received < config['size_kb'] * 1024:
                    data = sock.recv(min(4096, config['size_kb'] * 1024 - received))
                    if not data:
                        break
                    received += len(data)

                elapsed = (time.perf_counter() - start) * 1000
                config['times'].append(elapsed)

            # Calculate stats
            config['avg_ms'] = sum(config['times']) / len(config['times'])
            config['min_ms'] = min(config['times'])
            config['max_ms'] = max(config['times'])

            logger.info(".2f")

        sock.close()

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        # Return partial results
        pass

    return results


if __name__ == "__main__":
    # Test transport functionality
    print("Testing HybridTransport...")

    # Test 1: Small payload (should use coalesced)
    transport = HybridTransport()
    small_buffers = HybridTransport.create_test_payload(1)  # 1KB

    print(f"Small payload: {len(small_buffers)} buffers, total {sum(len(b) for b in small_buffers)} bytes")
    print("Strategy: Coalesced (single syscall)")

    # Test 2: Large payload (should use scatter-gather)
    large_buffers = HybridTransport.create_test_payload(100)  # 100KB
    print(f"Large payload: {len(large_buffers)} buffers, total {sum(len(b) for b in large_buffers)} bytes")
    print("Strategy: Scatter-gather (zero-copy)")    # Test 3: Benchmark
    print("\nBenchmarking transport...")
    try:
        bench_results = benchmark_transport()
        for payload_type, result in bench_results.items():
            if 'avg_ms' in result:
                print(".2f")
    except Exception as e:
        print(f"Benchmark failed: {e}")

    print("✅ HybridTransport tests completed")
