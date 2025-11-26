"""
Cache Query Protocol: Client-side cache query for GPU disaggregation.

Before sending tensors to the server, queries which identifiers are already cached.
This is the CRITICAL optimization that enables 99.8% network reduction.

Protocol:
1. Client → Server: CACHE_QUERY with list of tensor IDs
2. Server → Client: Response with cached IDs
3. Client filters tensors: only send missing IDs

Performance:
- Query latency: ~5-10ms (vs ~6000ms for transferring 5.8GB)
- Query size: ~1-5KB (list of IDs)
- Response size: ~1-5KB (list of cached IDs)

This is Phase 1, Component 2 of the enhancement plan.
"""

import asyncio
import json
import logging
import time
import threading
from typing import Set, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheQueryResult:
    """Result of cache query."""
    cached_identifiers: Set[str]  # IDs server has cached
    missing_identifiers: Set[str]  # IDs server needs
    query_time_ms: float  # Time taken for query
    
    @property
    def cache_hit_rate(self) -> float:
        """Percentage of identifiers that were cached."""
        total = len(self.cached_identifiers) + len(self.missing_identifiers)
        return 100 * len(self.cached_identifiers) / total if total > 0 else 0.0


class CacheQueryClient:
    """
    Client-side cache query protocol.
    
    Before sending tensors, queries server to see what's cached.
    This is the CRITICAL optimization that enables network reduction.
    
    Thread-safe singleton for process-wide cache state tracking.
    """
    
    _instance: Optional['CacheQueryClient'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_client()
        return cls._instance
    
    def _init_client(self):
        """Initialize client state."""
        # Map server_address → cached IDs (local cache state)
        self._cache_state: Dict[str, Set[str]] = {}
        
        # Thread safety - use RLock for re-entrant locking
        self._state_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'queries': 0,
            'total_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'local_cache_hits': 0,
            'server_queries': 0,
            'errors': 0
        }
    
    async def query_cached_identifiers(
        self,
        server_address: str,
        tensor_identifiers: Set[str],
        use_local_cache: bool = True,
        timeout: float = 5.0
    ) -> CacheQueryResult:
        """
        Query server for cached tensor identifiers.
        
        Args:
            server_address: Server to query (e.g., "localhost:5556")
            tensor_identifiers: Set of tensor identifiers to check
            use_local_cache: If True, check local cache first (optimization)
            timeout: Timeout for server query in seconds
        
        Returns:
            CacheQueryResult with cached/missing IDs
        """
        start_time = time.perf_counter()
        
        with self._state_lock:
            self.stats['queries'] += 1
        
        # Optimization: Check local cache first
        if use_local_cache:
            with self._state_lock:
                if server_address in self._cache_state:
                    local_cached = self._cache_state[server_address] & tensor_identifiers
                    local_missing = tensor_identifiers - local_cached
                    
                    # If we have high confidence in local cache (>90% cached locally)
                    # we can skip the RPC query
                    if len(tensor_identifiers) > 0 and len(local_cached) / len(tensor_identifiers) > 0.9:
                        logger.debug(
                            f"Using local cache state for {server_address}: "
                            f"{len(local_cached)}/{len(tensor_identifiers)} cached"
                        )
                        
                        query_time = (time.perf_counter() - start_time) * 1000
                        with self._state_lock:
                            self.stats['total_time_ms'] += query_time
                            self.stats['local_cache_hits'] += 1
                        
                        return CacheQueryResult(
                            cached_identifiers=local_cached,
                            missing_identifiers=local_missing,
                            query_time_ms=query_time
                        )
        
        # Make RPC query to server
        try:
            with self._state_lock:
                self.stats['server_queries'] += 1
            
            cached_ids = await self._query_server(server_address, tensor_identifiers, timeout)
            missing_ids = tensor_identifiers - cached_ids
            
            # Update local cache state
            with self._state_lock:
                if server_address not in self._cache_state:
                    self._cache_state[server_address] = set()
                self._cache_state[server_address].update(cached_ids)
                
                # Update stats
                self.stats['cache_hits'] += len(cached_ids)
                self.stats['cache_misses'] += len(missing_ids)
            
            query_time = (time.perf_counter() - start_time) * 1000
            with self._state_lock:
                self.stats['total_time_ms'] += query_time
            
            logger.info(
                f"Cache query: {len(cached_ids)}/{len(tensor_identifiers)} cached "
                f"(hit rate: {100*len(cached_ids)/len(tensor_identifiers):.1f}%, "
                f"query time: {query_time:.1f}ms)"
            )
            
            return CacheQueryResult(
                cached_identifiers=cached_ids,
                missing_identifiers=missing_ids,
                query_time_ms=query_time
            )
        
        except Exception as e:
            logger.warning(f"Cache query failed: {e}, assuming nothing cached")
            query_time = (time.perf_counter() - start_time) * 1000
            
            with self._state_lock:
                self.stats['total_time_ms'] += query_time
                self.stats['errors'] += 1
            
            # On error, assume nothing is cached (safe fallback)
            return CacheQueryResult(
                cached_identifiers=set(),
                missing_identifiers=tensor_identifiers,
                query_time_ms=query_time
            )
    
    async def _query_server(
        self, 
        server_address: str, 
        tensor_identifiers: Set[str],
        timeout: float = 5.0
    ) -> Set[str]:
        """
        Make RPC call to server to query cache.
        
        Protocol:
        - Message type: 0x04 (CACHE_QUERY)
        - Format: JSON with {'identifiers': [...]}
        - Response: JSON with {'cached_identifiers': [...]}
        
        Args:
            server_address: Server address (e.g., "localhost:5556")
            tensor_identifiers: Set of identifiers to check
            timeout: Connection timeout in seconds
        
        Returns:
            Set of cached identifiers
        """
        host, port_str = server_address.split(':')
        port = int(port_str)
        
        # Establish connection
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
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
                logger.debug(f"Failed to optimize TCP socket: {e}")
        
        try:
            # Prepare query message
            query_message = {
                'identifiers': list(tensor_identifiers)
            }
            query_json = json.dumps(query_message).encode('utf-8')
            
            # Send query
            # Protocol: [message_type (1 byte)][length (8 bytes)][data]
            # Use 8 bytes to match server's _recv_message protocol (supports large messages)
            message_type = (0x04).to_bytes(1, 'big')  # CACHE_QUERY
            length = len(query_json).to_bytes(8, 'big')  # FIX: Use 8 bytes to match server
            
            writer.write(message_type)
            writer.write(length)
            writer.write(query_json)
            await writer.drain()
            
            # Receive response
            # Protocol: [message_type (1 byte)][length (8 bytes)][data]
            # Match server's _send_message protocol (8-byte length for large messages)
            msg_type_response = await asyncio.wait_for(
                reader.readexactly(1),
                timeout=timeout
            )
            response_length_bytes = await asyncio.wait_for(
                reader.readexactly(8),  # FIX: Use 8 bytes to match server protocol
                timeout=timeout
            )
            response_length = int.from_bytes(response_length_bytes, 'big')
            
            response_data = await asyncio.wait_for(
                reader.readexactly(response_length),
                timeout=timeout
            )
            
            response = json.loads(response_data.decode('utf-8'))
            cached_ids = set(response.get('cached_identifiers', []))
            
            return cached_ids
        
        finally:
            writer.close()
            await writer.wait_closed()
    
    def invalidate_cache(self, server_address: Optional[str] = None):
        """
        Invalidate local cache state.
        
        Args:
            server_address: If specified, invalidate only for this server.
                           If None, invalidate all.
        """
        with self._state_lock:
            if server_address:
                self._cache_state.pop(server_address, None)
            else:
                self._cache_state.clear()
            logger.debug(f"Cache invalidated for {server_address or 'all servers'}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get client statistics."""
        with self._state_lock:
            stats = self.stats.copy()
            
            # Calculate averages
            if stats['queries'] > 0:
                stats['avg_query_time_ms'] = stats['total_time_ms'] / stats['queries']
            else:
                stats['avg_query_time_ms'] = 0.0
            
            return stats


# Global singleton accessor
_cache_query_client: Optional[CacheQueryClient] = None
_cache_query_lock = threading.Lock()


def get_cache_query_client() -> CacheQueryClient:
    """Get the global cache query client."""
    global _cache_query_client
    if _cache_query_client is None:
        with _cache_query_lock:
            if _cache_query_client is None:
                _cache_query_client = CacheQueryClient()
    return _cache_query_client

