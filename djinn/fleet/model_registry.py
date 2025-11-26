"""
Global Model Registry for Djinn Fleet Coordinator.

Tracks which servers have which models cached across the fleet.
Supports Redis backend (primary) and in-memory fallback.
"""

import logging
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheEntry:
    """Entry in global model registry."""
    fingerprint: str
    server_address: str
    registered_at: float
    ttl: float = 3600.0  # 1 hour default TTL
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.registered_at > self.ttl


class GlobalModelRegistry:
    """
    Global model registry tracking which servers have which models cached.
    
    Supports multiple backends:
    - Redis (primary, for production)
    - In-memory (fallback, for testing)
    
    Design: Eventually consistent with TTL-based expiration.
    """
    
    def __init__(self, backend: Optional[str] = None, redis_url: Optional[str] = None):
        """
        Initialize global model registry.
        
        Args:
            backend: Backend type ('redis', 'memory', or None for auto-detect)
            redis_url: Redis connection URL (if using Redis backend)
        """
        self.backend_type = backend or self._detect_backend()
        self.redis_url = redis_url
        
        if self.backend_type == 'redis':
            self._init_redis_backend()
        else:
            self._init_memory_backend()
        
        logger.info(f"✅ GlobalModelRegistry initialized with {self.backend_type} backend")
    
    def _detect_backend(self) -> str:
        """Auto-detect available backend."""
        try:
            import redis
            return 'redis'
        except ImportError:
            logger.debug("Redis not available, using in-memory backend")
            return 'memory'
    
    def _init_redis_backend(self):
        """Initialize Redis backend."""
        try:
            # Try aioredis first (async Redis client)
            try:
                import aioredis
                self._use_aioredis = True
                # Will be initialized in async context
                self._redis_url = self.redis_url or 'redis://localhost:6379/0'
                self.redis_client = None  # Will be set in async_init_redis
                logger.info("✅ Will use aioredis (async Redis client)")
                return
            except ImportError:
                # Fallback to sync redis with run_in_executor
                import redis
                from redis import ConnectionPool
                self._use_aioredis = False
                
                # Use connection pool for efficiency
                self._redis_pool = ConnectionPool.from_url(
                    self.redis_url or 'redis://localhost:6379/0',
                    max_connections=10,
                    decode_responses=True
                )
                self.redis_client = redis.Redis(connection_pool=self._redis_pool)
                
                # Test connection (sync, but only during init)
                self.redis_client.ping()
                logger.info("✅ Redis backend connected (sync redis with executor)")
        except Exception as e:
            logger.warning(f"⚠️  Redis connection failed: {e}, falling back to memory")
            self.backend_type = 'memory'
            self._init_memory_backend()
    
    async def _ensure_redis_client(self):
        """Ensure Redis client is initialized (for aioredis)."""
        if self.backend_type != 'redis':
            return
        
        if hasattr(self, '_use_aioredis') and self._use_aioredis:
            if self.redis_client is None:
                import aioredis
                self.redis_client = await aioredis.from_url(
                    self._redis_url,
                    max_connections=10,
                    decode_responses=True
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("✅ Async Redis client initialized")
    
    def _init_memory_backend(self):
        """Initialize in-memory backend."""
        import asyncio
        # fingerprint -> Set[server_address]
        self._registry: Dict[str, Set[str]] = defaultdict(set)
        # server_address -> Set[fingerprint] (for cleanup)
        self._server_models: Dict[str, Set[str]] = defaultdict(set)
        # Entry metadata for TTL
        self._entries: Dict[tuple, ModelCacheEntry] = {}
        # Thread safety lock for in-memory operations
        self._lock = asyncio.Lock()
        logger.info("✅ In-memory backend initialized")
    
    async def find_cached_servers(self, fingerprint: str) -> List[str]:
        """
        Find all servers that have this model cached.
        
        Args:
            fingerprint: Model fingerprint
            
        Returns:
            List of server addresses (e.g., ["server-1:5556", "server-2:5556"])
        """
        if self.backend_type == 'redis':
            return await self._find_cached_servers_redis(fingerprint)
        else:
            return await self._find_cached_servers_memory(fingerprint)
    
    async def _find_cached_servers_redis(self, fingerprint: str) -> List[str]:
        """Find cached servers using Redis backend."""
        await self._ensure_redis_client()
        
        try:
            key = f"djinn:model:{fingerprint}"
            
            # Use async or executor based on client type
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                # Async Redis (aioredis)
                servers = await self.redis_client.smembers(key)
                servers = list(servers) if servers else []
            else:
                # Sync Redis (use executor to avoid blocking)
                import asyncio
                loop = asyncio.get_event_loop()
                servers = await loop.run_in_executor(
                    None,
                    lambda: list(self.redis_client.smembers(key))
                )
            
            if not servers:
                return []
            
            current_time = time.time()
            valid_servers = []
            expired_servers = []
            
            # Batch read all entry metadata
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                # Async: Use pipeline
                pipe = self.redis_client.pipeline()
                for server in servers:
                    entry_key = f"djinn:entry:{fingerprint}:{server}"
                    pipe.hgetall(entry_key)
                results = await pipe.execute()
            else:
                # Sync: Use executor
                import asyncio
                loop = asyncio.get_event_loop()
                pipe = self.redis_client.pipeline()
                for server in servers:
                    entry_key = f"djinn:entry:{fingerprint}:{server}"
                    pipe.hgetall(entry_key)
                results = await loop.run_in_executor(None, pipe.execute)
            
            # Process results
            for server, entry_data in zip(servers, results):
                if entry_data:
                    registered_at = float(entry_data.get('registered_at', 0))
                    ttl = float(entry_data.get('ttl', 3600.0))
                    if current_time - registered_at < ttl:
                        valid_servers.append(server)
                    else:
                        # Expired, mark for removal
                        expired_servers.append(server)
            
            # Batch unregister expired entries
            if expired_servers:
                tasks = [
                    self.unregister_model(fingerprint, server)
                    for server in expired_servers
                ]
                # Use gather with return_exceptions to not fail on individual errors
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return valid_servers
        except Exception as e:
            logger.error(f"❌ Redis query failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def _find_cached_servers_memory(self, fingerprint: str) -> List[str]:
        """Find cached servers using in-memory backend (thread-safe)."""
        async with self._lock:
            # Lazy cleanup: only cleanup if we access expired entries
            # This is more efficient than cleaning on every query
            servers = list(self._registry.get(fingerprint, set()))
            
            # Filter out expired entries (lazy cleanup)
            valid_servers = []
            expired_entries = []
            for server in servers:
                entry = self._entries.get((fingerprint, server))
                if entry and not entry.is_expired():
                    valid_servers.append(server)
                elif entry and entry.is_expired():
                    # Expired, mark for removal
                    expired_entries.append((fingerprint, server))
            
            # Remove expired entries
            for fp, server in expired_entries:
                self._registry[fp].discard(server)
                self._server_models[server].discard(fp)
                self._entries.pop((fp, server), None)
            
            return valid_servers
    
    async def register_model(self, fingerprint: str, server_address: str, ttl: float = 3600.0):
        """
        Register a model on a server.
        
        Args:
            fingerprint: Model fingerprint
            server_address: Server address (e.g., "server-1:5556")
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        if self.backend_type == 'redis':
            await self._register_model_redis(fingerprint, server_address, ttl)
        else:
            await self._register_model_memory(fingerprint, server_address, ttl)
        
        logger.debug(f"📝 Registered model {fingerprint[:16]}... on {server_address}")
    
    async def _register_model_redis(self, fingerprint: str, server_address: str, ttl: float):
        """Register model using Redis backend."""
        await self._ensure_redis_client()
        
        try:
            key = f"djinn:model:{fingerprint}"
            entry_key = f"djinn:entry:{fingerprint}:{server_address}"
            server_key = f"djinn:server:{server_address}"
            
            entry_data = {
                'registered_at': str(time.time()),
                'ttl': str(ttl),
            }
            
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                # Async Redis
                pipe = self.redis_client.pipeline()
                pipe.sadd(key, server_address)
                pipe.hset(entry_key, mapping=entry_data)
                pipe.expire(entry_key, int(ttl))
                pipe.sadd(server_key, fingerprint)
                await pipe.execute()
            else:
                # Sync Redis (use executor)
                import asyncio
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: (
                    self.redis_client.sadd(key, server_address),
                    self.redis_client.hset(entry_key, mapping=entry_data),
                    self.redis_client.expire(entry_key, int(ttl)),
                    self.redis_client.sadd(server_key, fingerprint)
                ))
        except Exception as e:
            logger.error(f"❌ Redis registration failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    async def _register_model_memory(self, fingerprint: str, server_address: str, ttl: float):
        """Register model using in-memory backend (thread-safe)."""
        async with self._lock:
            self._registry[fingerprint].add(server_address)
            self._server_models[server_address].add(fingerprint)
            
            # Store entry metadata
            entry = ModelCacheEntry(
                fingerprint=fingerprint,
                server_address=server_address,
                registered_at=time.time(),
                ttl=ttl
            )
            self._entries[(fingerprint, server_address)] = entry
    
    async def unregister_model(self, fingerprint: str, server_address: str):
        """
        Unregister a model from a server.
        
        Args:
            fingerprint: Model fingerprint
            server_address: Server address
        """
        if self.backend_type == 'redis':
            await self._unregister_model_redis(fingerprint, server_address)
        else:
            await self._unregister_model_memory(fingerprint, server_address)
        
        logger.debug(f"🗑️  Unregistered model {fingerprint[:16]}... from {server_address}")
    
    async def _unregister_model_redis(self, fingerprint: str, server_address: str):
        """Unregister model using Redis backend."""
        await self._ensure_redis_client()
        
        try:
            key = f"djinn:model:{fingerprint}"
            entry_key = f"djinn:entry:{fingerprint}:{server_address}"
            server_key = f"djinn:server:{server_address}"
            
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                # Async Redis
                pipe = self.redis_client.pipeline()
                pipe.srem(key, server_address)
                pipe.delete(entry_key)
                pipe.srem(server_key, fingerprint)
                await pipe.execute()
            else:
                # Sync Redis (use executor)
                import asyncio
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: (
                    self.redis_client.srem(key, server_address),
                    self.redis_client.delete(entry_key),
                    self.redis_client.srem(server_key, fingerprint)
                ))
        except Exception as e:
            logger.error(f"❌ Redis unregistration failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    async def _unregister_model_memory(self, fingerprint: str, server_address: str):
        """Unregister model using in-memory backend (thread-safe)."""
        async with self._lock:
            self._registry[fingerprint].discard(server_address)
            self._server_models[server_address].discard(fingerprint)
            self._entries.pop((fingerprint, server_address), None)
    
    async def unregister_server(self, server_address: str):
        """
        Unregister all models from a server (when server goes down).
        
        Args:
            server_address: Server address
        """
        if self.backend_type == 'redis':
            await self._unregister_server_redis(server_address)
        else:
            await self._unregister_server_memory(server_address)
        
        logger.info(f"🗑️  Unregistered all models from server {server_address}")
    
    async def _unregister_server_redis(self, server_address: str):
        """Unregister server using Redis backend."""
        await self._ensure_redis_client()
        
        try:
            server_key = f"djinn:server:{server_address}"
            
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                # Async Redis
                fingerprints = await self.redis_client.smembers(server_key)
                fingerprints = list(fingerprints) if fingerprints else []
            else:
                # Sync Redis (use executor)
                import asyncio
                loop = asyncio.get_event_loop()
                fingerprints = await loop.run_in_executor(
                    None,
                    lambda: list(self.redis_client.smembers(server_key))
                )
            
            # Unregister all models
            if fingerprints:
                tasks = [
                    self.unregister_model(fingerprint, server_address)
                    for fingerprint in fingerprints
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Remove server key
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                await self.redis_client.delete(server_key)
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self.redis_client.delete(server_key))
        except Exception as e:
            logger.error(f"❌ Redis server unregistration failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    async def _unregister_server_memory(self, server_address: str):
        """Unregister server using in-memory backend (thread-safe)."""
        async with self._lock:
            fingerprints = list(self._server_models.get(server_address, set()))
        
        # Unregister models (outside lock to avoid deadlock)
        for fingerprint in fingerprints:
            await self.unregister_model(fingerprint, server_address)
        
        # Remove server entry (inside lock)
        async with self._lock:
            self._server_models.pop(server_address, None)
    
    async def get_all_cached_models(self) -> Dict[str, List[str]]:
        """
        Get all cached models and their servers.
        
        Returns:
            Dict mapping fingerprint to list of server addresses
        """
        if self.backend_type == 'redis':
            return await self._get_all_cached_models_redis()
        else:
            return await self._get_all_cached_models_memory()
    
    async def _get_all_cached_models_redis(self) -> Dict[str, List[str]]:
        """Get all cached models using Redis backend."""
        await self._ensure_redis_client()
        
        try:
            result = {}
            pattern = "djinn:model:*"
            
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                # Async Redis: Collect keys
                keys = []
                async for key in self.redis_client.scan_iter(match=pattern):
                    keys.append(key)
                
                if not keys:
                    return {}
                
                # Use pipeline for batch operations
                pipe = self.redis_client.pipeline()
                for key in keys:
                    pipe.smembers(key)
                results = await pipe.execute()
            else:
                # Sync Redis (use executor)
                import asyncio
                loop = asyncio.get_event_loop()
                keys = await loop.run_in_executor(
                    None,
                    lambda: list(self.redis_client.scan_iter(match=pattern))
                )
                
                if not keys:
                    return {}
                
                pipe = self.redis_client.pipeline()
                for key in keys:
                    pipe.smembers(key)
                results = await loop.run_in_executor(None, pipe.execute)
            
            # Process results
            for key, servers in zip(keys, results):
                fingerprint = key.replace("djinn:model:", "")
                server_list = list(servers) if isinstance(servers, (set, list)) else [servers]
                if server_list:
                    result[fingerprint] = server_list
            
            return result
        except Exception as e:
            logger.error(f"❌ Redis query failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {}
    
    async def _get_all_cached_models_memory(self) -> Dict[str, List[str]]:
        """Get all cached models using in-memory backend (thread-safe)."""
        async with self._lock:
            self._cleanup_expired()
            return {
                fingerprint: list(servers)
                for fingerprint, servers in self._registry.items()
                if servers
            }
    
    def _cleanup_expired(self):
        """Clean up expired entries (in-memory backend only, must be called within lock)."""
        expired = []
        for (fingerprint, server), entry in self._entries.items():
            if entry.is_expired():
                expired.append((fingerprint, server))
        
        for fingerprint, server in expired:
            self._registry[fingerprint].discard(server)
            self._server_models[server].discard(fingerprint)
            self._entries.pop((fingerprint, server), None)
    
    async def close(self):
        """
        Close connections and cleanup resources.
        
        Should be called when shutting down the registry.
        """
        if self.backend_type == 'redis':
            if hasattr(self, '_use_aioredis') and self._use_aioredis:
                if self.redis_client:
                    await self.redis_client.close()
                    logger.info("✅ Closed async Redis client")
            elif hasattr(self, '_redis_pool'):
                # Close connection pool (sync redis)
                self._redis_pool.disconnect()
                logger.info("✅ Closed Redis connection pool")
        
        logger.info("✅ GlobalModelRegistry closed")

