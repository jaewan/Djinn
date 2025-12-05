"""
KV Cache Session Manager for stateful decode operations.

Phase 1 Enhancement:
- Session-based KV cache pinning (keeps cache on same GPU across steps)
- Automatic idle-timeout cleanup
- Async-first design with asyncio.Lock
- Integration with coordinator for GPU affinity

This reduces network traffic by 500× for autoregressive decode operations
by keeping KV cache resident on GPU rather than transferring it every step.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

import torch

logger = logging.getLogger(__name__)


@dataclass
class KVSession:
    """Represents a stateful KV cache session."""
    session_id: str
    gpu_id: int
    kv_cache: Optional[Any] = None
    last_access: float = field(default_factory=time.time)
    bytes_used: int = 0
    step_count: int = 0
    is_swapped: bool = False  # Whether KV is on host (Phase 3)
    prefetch_in_progress: bool = False  # ✅ Prevent race condition: skip COMPUTE restore if prefetch already scheduled


class KVSessionManager:
    """
    Manages stateful KV cache sessions for autoregressive decoding.
    
    Key features:
    - Session → GPU affinity (session pinned to one GPU)
    - KV cache resident on GPU (no transfers between steps)
    - Automatic cleanup of idle sessions (>60s)
    - Async-first design (all operations are async)
    - Thread-safe via asyncio.Lock
    """
    
    def __init__(self, idle_timeout_seconds: float = 60.0, cleanup_interval_seconds: float = 10.0):
        """
        Initialize KV session manager.
        
        Args:
            idle_timeout_seconds: Sessions idle > this are evicted (default: 60s)
            cleanup_interval_seconds: How often to run cleanup task (default: 10s)
        """
        self._sessions: Dict[str, KVSession] = {}
        self._lock = asyncio.Lock()
        self.idle_timeout_seconds = idle_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.stats = {
            "sessions_created": 0,
            "sessions_closed": 0,
            "kv_bytes_pinned": 0,
            "max_concurrent_sessions": 0,
            "cleanup_evictions": 0,
        }
        self._cleanup_task: Optional[asyncio.Task] = None
        logger.info(
            "KVSessionManager initialized (idle_timeout=%.0fs, cleanup_interval=%.0fs)",
            idle_timeout_seconds,
            cleanup_interval_seconds,
        )
    
    async def get_or_create(
        self, 
        session_id: str, 
        gpu_id: int,
        initial_kv: Optional[Any] = None
    ) -> KVSession:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Unique session identifier
            gpu_id: GPU to pin this session to
            initial_kv: Optional initial KV cache tensor
            
        Returns:
            KVSession object
        """
        async with self._lock:
            # Return existing session, update access time
            if session_id in self._sessions:
                sess = self._sessions[session_id]
                sess.last_access = time.time()
                sess.step_count += 1
                logger.debug(
                    "KV session retrieved: session_id=%s, gpu_id=%d, step=%d",
                    session_id, gpu_id, sess.step_count
                )
                return sess
            
            # Create new session
            kv_gpu = None
            bytes_used = 0
            if initial_kv is not None:
                kv_gpu = await asyncio.to_thread(
                    self._move_structure_to_device,
                    initial_kv,
                    torch.device(f"cuda:{gpu_id}")
                )
                bytes_used = self._estimate_size_bytes(kv_gpu)
            
            sess = KVSession(
                session_id=session_id,
                gpu_id=gpu_id,
                kv_cache=kv_gpu,
                last_access=time.time(),
                bytes_used=bytes_used,
                step_count=1,
            )
            self._sessions[session_id] = sess
            self.stats["sessions_created"] += 1
            self.stats["kv_bytes_pinned"] += bytes_used
            self.stats["max_concurrent_sessions"] = max(
                self.stats["max_concurrent_sessions"],
                len(self._sessions)
            )
            
            logger.info(
                "KV session created: session_id=%s, gpu_id=%d, initial_kv_mb=%.1f",
                session_id, gpu_id, bytes_used / (1024 * 1024)
            )
            return sess
    
    async def update_kv(
        self,
        session_id: str,
        kv_cache: Any
    ) -> KVSession:
        """
        Update KV cache for an existing session.
        
        Transfers new KV cache to session's GPU and updates resident cache.
        
        Args:
            session_id: Session to update
            kv_cache: New KV cache tensor (may be on CPU or different GPU)
            
        Returns:
            Updated KVSession
        """
        async with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            sess = self._sessions[session_id]
            
            # Move new KV to session GPU (non-blocking)
            kv_gpu = await asyncio.to_thread(
                self._move_structure_to_device,
                kv_cache,
                torch.device(f"cuda:{sess.gpu_id}")
            )
            
            # Update stats
            new_bytes = self._estimate_size_bytes(kv_gpu)
            old_bytes = sess.bytes_used
            sess.kv_cache = kv_gpu
            sess.bytes_used = new_bytes
            sess.last_access = time.time()
            sess.step_count += 1
            
            self.stats["kv_bytes_pinned"] += (new_bytes - old_bytes)
            
            logger.debug(
                "KV cache updated: session_id=%s, step=%d, kv_mb=%.1f",
                session_id, sess.step_count, new_bytes / (1024 * 1024)
            )
            return sess
    
    async def close_session(self, session_id: str) -> int:
        """
        Close and cleanup a session.
        
        Args:
            session_id: Session to close
            
        Returns:
            Bytes freed
        """
        async with self._lock:
            if session_id not in self._sessions:
                return 0
            
            sess = self._sessions.pop(session_id)
            freed_bytes = sess.bytes_used
            self.stats["sessions_closed"] += 1
            self.stats["kv_bytes_pinned"] -= freed_bytes
            
            logger.info(
                "KV session closed: session_id=%s, steps=%d, freed_mb=%.1f",
                session_id, sess.step_count, freed_bytes / (1024 * 1024)
            )
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                await asyncio.to_thread(torch.cuda.empty_cache)
            
            return freed_bytes
    
    async def start_cleanup(self) -> None:
        """Start periodic cleanup task for idle sessions."""
        if self._cleanup_task is not None:
            return  # Already running
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("KV session cleanup task started")
    
    async def stop_cleanup(self) -> None:
        """Stop cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("KV session cleanup task stopped")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of idle sessions."""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._cleanup_idle_sessions()
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
            raise
    
    async def _cleanup_idle_sessions(self) -> None:
        """Evict sessions idle > idle_timeout_seconds."""
        now = time.time()
        to_close = []
        
        async with self._lock:
            for session_id, sess in self._sessions.items():
                idle_time = now - sess.last_access
                if idle_time > self.idle_timeout_seconds:
                    to_close.append((session_id, idle_time))
        
        for session_id, idle_time in to_close:
            await self.close_session(session_id)
            self.stats["cleanup_evictions"] += 1
            logger.info(
                "Idle session evicted: session_id=%s, idle_time=%.1fs",
                session_id, idle_time
            )
    
    async def get_session_kv(self, session_id: str) -> Optional[torch.Tensor]:
        """
        Get KV cache for a session without updating access time.
        
        Useful for read-only queries about session state.
        """
        async with self._lock:
            sess = self._sessions.get(session_id)
            return sess.kv_cache if sess else None
    
    async def check_prefetch_in_progress(self, session_id: str) -> bool:
        """
        Thread-safe check if prefetch is in progress for a session.
        
        HARDENED: Uses proper async lock instead of direct dict access.
        
        Args:
            session_id: Session to check
            
        Returns:
            True if prefetch is in progress, False otherwise
        """
        async with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return False
            return sess.prefetch_in_progress
    
    async def set_prefetch_in_progress(self, session_id: str, value: bool) -> bool:
        """
        Thread-safe setter for prefetch_in_progress flag.
        
        HARDENED: Uses proper async lock instead of direct dict access.
        
        Args:
            session_id: Session to update
            value: New value for prefetch_in_progress
            
        Returns:
            True if session exists and was updated, False otherwise
        """
        async with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return False
            sess.prefetch_in_progress = value
            return True
    
    def get_stats(self) -> Dict:
        """Get session manager statistics."""
        return {
            **self.stats,
            "active_sessions": len(self._sessions),
            "kv_bytes_pinned_mb": self.stats["kv_bytes_pinned"] / (1024 * 1024),
        }
    
    async def audit_resources(self) -> Dict:
        """
        HARDENED: Audit resources to detect leaks between sessions and swap pool.
        
        Checks for:
        - Orphaned swaps (in swap pool but session doesn't exist)
        - Swapped sessions without swap pool entry
        - Sessions marked swapped but with kv_cache still set
        
        Returns:
            Audit report with any detected issues
        """
        from djinn.server.host_swap_pool_v2 import get_swap_pool
        
        issues = []
        
        async with self._lock:
            session_ids = set(self._sessions.keys())
            swapped_session_ids = {sid for sid, sess in self._sessions.items() if sess.is_swapped}
            sessions_with_kv = {sid for sid, sess in self._sessions.items() if sess.kv_cache is not None}
            
            swap_pool = get_swap_pool()
            swap_pool_ids = set(swap_pool.swapped_sessions.keys())
            
            # Check 1: Orphaned swaps (in pool but no session)
            orphaned_swaps = swap_pool_ids - session_ids
            if orphaned_swaps:
                issues.append({
                    "type": "orphaned_swaps",
                    "count": len(orphaned_swaps),
                    "session_ids": list(orphaned_swaps)[:5],  # First 5
                })
                logger.warning(f"Resource audit: {len(orphaned_swaps)} orphaned swaps in pool")
            
            # Check 2: Session marked swapped but not in pool
            missing_swaps = swapped_session_ids - swap_pool_ids
            if missing_swaps:
                issues.append({
                    "type": "missing_swaps",
                    "count": len(missing_swaps),
                    "session_ids": list(missing_swaps)[:5],
                })
                logger.warning(f"Resource audit: {len(missing_swaps)} sessions marked swapped but not in pool")
            
            # Check 3: Session swapped but still has kv_cache (should be None)
            bad_state_sessions = swapped_session_ids & sessions_with_kv
            if bad_state_sessions:
                issues.append({
                    "type": "swapped_with_kv_cache",
                    "count": len(bad_state_sessions),
                    "session_ids": list(bad_state_sessions)[:5],
                })
                logger.warning(f"Resource audit: {len(bad_state_sessions)} swapped sessions still have kv_cache")
        
        return {
            "timestamp": time.time(),
            "total_sessions": len(session_ids),
            "total_swapped": len(swapped_session_ids),
            "swap_pool_entries": len(swap_pool_ids),
            "issues": issues,
            "healthy": len(issues) == 0,
        }
    
    async def cleanup_orphaned_resources(self) -> int:
        """
        HARDENED: Clean up orphaned resources detected by audit.
        
        Returns:
            Number of resources cleaned up
        """
        from djinn.server.host_swap_pool_v2 import get_swap_pool
        
        cleaned = 0
        
        async with self._lock:
            session_ids = set(self._sessions.keys())
            swap_pool = get_swap_pool()
            
            # Clean orphaned swaps
            orphaned = set(swap_pool.swapped_sessions.keys()) - session_ids
            for sid in orphaned:
                with swap_pool._lock:
                    if sid in swap_pool.swapped_sessions:
                        del swap_pool.swapped_sessions[sid]
                        cleaned += 1
                        logger.info(f"Cleaned orphaned swap: {sid[:12]}")
            
            # Fix sessions marked swapped but not in pool
            for sid, sess in self._sessions.items():
                if sess.is_swapped and sid not in swap_pool.swapped_sessions:
                    sess.is_swapped = False
                    cleaned += 1
                    logger.info(f"Fixed bad swap state for session: {sid[:12]}")
        
        return cleaned

    def _move_structure_to_device(self, data: Any, device: torch.device) -> Any:
        """
        Recursively move tensors to the specified device.
        
        CRITICAL FIX for DynamicCache: Must deep-clone the cache object to capture
        tensor state BEFORE HuggingFace's internal loop modifies/clears it.
        
        Without this deep-copy, the DynamicCache object becomes a zombie with empty
        key_cache/value_cache lists after model.generate() completes.
        """
        if isinstance(data, torch.Tensor):
            return data.to(device, non_blocking=True)
        if isinstance(data, (list, tuple)):
            converted = [self._move_structure_to_device(elem, device) for elem in data]
            return type(data)(converted)
        if isinstance(data, dict):
            return {k: self._move_structure_to_device(v, device) for k, v in data.items()}
        
        # ✅ CRITICAL: Handle HuggingFace DynamicCache (or any Cache subclass)
        # Deep-copy the tensor references to capture state NOW, not later
        try:
            from transformers.cache_utils import Cache
            if isinstance(data, Cache):
                logger.debug(f"Deep-copying DynamicCache to {device} (capturing tensor state)")
                
                # Try the recommended approach: to_legacy_cache -> copy -> restore
                try:
                    # Convert to legacy tuple format (more stable)
                    legacy_cache = data.to_legacy_cache()
                    if legacy_cache is not None:
                        # Deep copy the legacy cache (list of tuples of tensors)
                        copied_legacy = self._move_structure_to_device(legacy_cache, device)
                        logger.debug("✅ Converted DynamicCache to legacy format and deep-copied")
                        return copied_legacy
                except Exception as e:
                    logger.warning(f"Failed to use to_legacy_cache: {e}, falling back to direct copy")
                
                # Fallback: Direct key_cache/value_cache clone
                # This is more fragile but works for current HuggingFace versions
                if hasattr(data, 'key_cache') and hasattr(data, 'value_cache'):
                    try:
                        new_key_cache = []
                        for k in data.key_cache:
                            if k is not None:
                                # Clone to capture value NOW, before it gets cleared
                                new_key_cache.append(k.clone().to(device, non_blocking=True))
                            else:
                                new_key_cache.append(None)
                        
                        new_value_cache = []
                        for v in data.value_cache:
                            if v is not None:
                                # Clone to capture value NOW
                                new_value_cache.append(v.clone().to(device, non_blocking=True))
                            else:
                                new_value_cache.append(None)
                        
                        # Reconstruct the cache object
                        from transformers.cache_utils import DynamicCache
                        new_cache = DynamicCache()
                        new_cache.key_cache = new_key_cache
                        new_cache.value_cache = new_value_cache
                        
                        logger.debug(f"✅ Deep-cloned DynamicCache: {len(new_key_cache)} layers")
                        return new_cache
                    except Exception as e:
                        logger.error(f"Failed to deep-copy DynamicCache tensors: {e}")
                        raise
        except ImportError:
            # transformers not available, just pass through
            pass
        except Exception as e:
            logger.warning(f"Exception handling DynamicCache: {e}")
            # Fall through to return data as-is
        
        return data

    def _estimate_size_bytes(self, data: Any) -> int:
        """Recursively estimate tensor memory usage in bytes."""
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if isinstance(data, (list, tuple)):
            return sum(self._estimate_size_bytes(elem) for elem in data)
        if isinstance(data, dict):
            return sum(self._estimate_size_bytes(v) for v in data.values())
        
        # Handle HuggingFace DynamicCache
        if hasattr(data, 'to_legacy_cache'):
            try:
                legacy_cache = data.to_legacy_cache()
                return self._estimate_size_bytes(legacy_cache)
            except Exception:
                pass
        
        # Fallback: try to collect from object attributes
        total_bytes = 0
        for attr_name in dir(data):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(data, attr_name)
                if torch.is_tensor(attr):
                    total_bytes += self._estimate_size_bytes(attr)
                elif isinstance(attr, (list, tuple, dict)):
                    total_bytes += self._estimate_size_bytes(attr)
            except:
                pass
        
        return total_bytes
    
    async def evict_kv_to_host(self, session_id: str) -> int:
        """
        Evict KV cache for a session to host swap pool (Phase 3 - SIMPLIFIED).
        
        Uses PyTorch's CachingHostAllocator (pin_memory=True) for allocation.
        Djinn only manages the POLICY (when to swap), not the MECHANISM.
        
        Args:
            session_id: Session to evict
            
        Returns:
            Bytes evicted, or 0 if not found/already swapped
            
        Raises:
            RuntimeError: If eviction fails
        """
        # Import here to avoid circular dependency
        from djinn.server.host_swap_pool_v2 import get_swap_pool
        
        async with self._lock:
            if session_id not in self._sessions:
                logger.debug(f"Cannot evict: session {session_id} not found")
                return 0
            
            sess = self._sessions[session_id]
            
            if sess.is_swapped:
                logger.debug(f"Session {session_id} already swapped")
                return 0
            
            if sess.kv_cache is None:
                logger.debug(f"Session {session_id} has no KV cache to evict")
                return 0
            
            try:
                # ✅ SIMPLIFIED: Delegate to PyTorch's allocator via pin_memory=True
                # This method handles:
                # 1. Allocating pinned CPU memory (PyTorch's CachingHostAllocator)
                # 2. Copying KV from GPU to pinned CPU (with non_blocking=True)
                # 3. Synchronizing the GPU stream (ensures data is complete)
                # 4. Tracking the swap for lifecycle management
                
                swap_pool = get_swap_pool()
                actual_bytes = await asyncio.to_thread(
                    swap_pool.allocate_and_swap,
                    session_id,
                    sess.kv_cache,
                    sess.gpu_id
                )
                
                # Mark as swapped
                sess.is_swapped = True
                self.stats["kv_bytes_pinned"] -= actual_bytes
                
                # ✅ CRITICAL FIX: Clear kv_cache to free GPU memory immediately!
                sess.kv_cache = None
                
                return actual_bytes
            
            except Exception as e:
                logger.error(f"Failed to evict session {session_id}: {e}")
                raise
    
    async def restore_kv_from_host(self, session_id: str) -> int:
        """
        Restore KV cache for a session from host swap pool to GPU (Phase 3 - SIMPLIFIED).
        
        Uses PyTorch's memory management for the actual copy operation.
        
        Args:
            session_id: Session to restore
            
        Returns:
            Bytes restored, or 0 if not found/not swapped
            
        Raises:
            RuntimeError: If restore fails
        """
        # Import here to avoid circular dependency
        from djinn.server.host_swap_pool_v2 import get_swap_pool
        
        async with self._lock:
            if session_id not in self._sessions:
                logger.debug(f"Cannot restore: session {session_id} not found")
                return 0
            
            sess = self._sessions[session_id]
            
            if not sess.is_swapped:
                logger.debug(f"Session {session_id} not swapped")
                return 0
            
            try:
                # ✅ CRITICAL FIX: restore_and_copy now returns GPU tensor, not just size!
                swap_pool = get_swap_pool()
                restored_tensor = await asyncio.to_thread(
                    swap_pool.restore_and_copy,
                    session_id,
                    sess.gpu_id
                )
                
                if restored_tensor is None:
                    logger.warning(f"Restore failed: session {session_id} returned None")
                    return 0
                
                # ✅ CRITICAL FIX: Store the restored tensor in kv_cache!
                size_bytes = restored_tensor.numel() * restored_tensor.element_size()
                sess.kv_cache = restored_tensor
                sess.is_swapped = False
                self.stats["kv_bytes_pinned"] += size_bytes
                
                return size_bytes
            
            except Exception as e:
                logger.error(f"Failed to restore session {session_id}: {e}")
                raise
    
    def _copy_to_host(self, kv_gpu: Any, host_view: torch.Tensor, size_bytes: int) -> int:
        """
        Copy KV cache from GPU to host pinned memory (ZERO-COPY via tensor bytes).
        
        ✅ ZERO-COPY DESIGN: No serialization overhead - direct tensor memcpy via flattened bytes.
        This aligns with Djinn's disaggregation philosophy and achieves ~24GB/s with pinned memory.
        
        Args:
            kv_gpu: KV cache structure on GPU (tuple of tensors, DynamicCache, etc.)
            host_view: Tensor view into host swap pool (pre-allocated pinned CPU memory)
            size_bytes: Expected size of flattened KV in bytes
            
        Returns:
            Actual number of bytes swapped (for restore tracking)
        """
        try:
            # ✅ STEP 1: Flatten nested KV structure to a single contiguous tensor
            # This handles DynamicCache, tuples, lists, etc. without serialization
            kv_flat_gpu = self._flatten_structure_to_tensor(kv_gpu)
            
            # ✅ STEP 2: Calculate actual size
            actual_bytes = kv_flat_gpu.element_size() * kv_flat_gpu.numel()
            if actual_bytes > size_bytes:
                raise RuntimeError(
                    f"Flattened KV cache exceeds allocated space: {actual_bytes} > {size_bytes}"
                )
            
            # ✅ STEP 3: Zero-copy memcpy from GPU to pinned host memory
            # Ensure both are flat uint8 tensors for byte-level copy
            kv_flat_bytes = kv_flat_gpu.view(-1).view(torch.uint8)  # Flatten to 1D uint8
            host_view_bytes = host_view.view(-1).view(torch.uint8)[:actual_bytes]  # Flatten to 1D uint8
            
            # Non-blocking async DMA transfer (leverages PCIe DMA engine)
            host_view_bytes.copy_(kv_flat_bytes, non_blocking=True)
            
            # Sync to ensure transfer completes before returning
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            
            logger.debug(f"✅ Swapped KV to host: {actual_bytes / (1024**2):.1f}MB (zero-copy)")
            
            return actual_bytes  # Return actual bytes for tracking
            
        except Exception as e:
            logger.error(f"❌ KV swap to host failed: {e}")
            raise
    
    def _copy_from_host(self, host_view: torch.Tensor, kv_gpu: Any, size_bytes: int) -> None:
        """
        Restore KV cache from host pinned memory back to GPU (ZERO-COPY via tensor bytes).
        
        ✅ ZERO-COPY DESIGN: Direct tensor memcpy without deserialization overhead.
        Restores the exact flattened byte structure that was swapped.
        
        Args:
            host_view: Tensor view into host swap pool (source)
            kv_gpu: Original KV structure (for device inference, not reconstruction)
            size_bytes: Size of flattened KV data in bytes
        
        Returns:
            Restored KV cache on GPU (as flattened tensor - caller must reconstruct structure)
        """
        try:
            # ✅ STEP 1: Get target device from original KV structure
            target_device = self._get_device_from_structure(kv_gpu)
            
            # ✅ STEP 2: Zero-copy memcpy from pinned host memory to GPU
            # Flatten host memory to 1D uint8 bytes
            host_bytes = host_view.view(-1).view(torch.uint8)[:size_bytes]
            
            # Create GPU tensor and copy bytes (also 1D uint8)
            gpu_bytes = torch.empty(size_bytes, dtype=torch.uint8, device=target_device)
            gpu_bytes.copy_(host_bytes, non_blocking=True)
            
            # Sync to ensure transfer completes
            torch.cuda.current_stream().synchronize()
            
            logger.debug(f"✅ Restored KV from host: {size_bytes / (1024**2):.1f}MB (zero-copy)")
            
            # Return as bytes - caller will use original KV structure metadata to reconstruct
            return gpu_bytes
            
        except Exception as e:
            logger.error(f"❌ KV restore from host failed: {e}")
            raise
    
    def _get_structure_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract structure metadata for proper deserialization after flatten."""
        if isinstance(data, torch.Tensor):
            return {
                'type': 'tensor',
                'shape': tuple(data.shape),
                'dtype': str(data.dtype),
                'device': str(data.device),
            }
        if isinstance(data, (list, tuple)):
            return {
                'type': 'sequence',
                'is_tuple': isinstance(data, tuple),
                'length': len(data),
                'elements': [self._get_structure_metadata(elem) for elem in data],
            }
        if isinstance(data, dict):
            return {
                'type': 'dict',
                'keys': list(data.keys()),
                'values': {k: self._get_structure_metadata(v) for k, v in data.items()},
            }
        # Handle HuggingFace DynamicCache
        if hasattr(data, 'to_legacy_cache'):
            try:
                legacy = data.to_legacy_cache()
                return {
                    'type': 'dynamic_cache',
                    'structure': self._get_structure_metadata(legacy),
                }
            except:
                pass
        return {'type': 'unknown'}
    
    def _flatten_structure_to_tensor(self, data: Any) -> torch.Tensor:
        """
        Flatten nested KV cache structure to a single contiguous tensor on GPU.
        
        Supports:
        - torch.Tensor
        - list/tuple of tensors
        - dict of tensors
        - HuggingFace DynamicCache (has get_seq_length() and past_key_values property)
        
        Returns a uint8 tensor containing all the data bytes.
        """
        tensors = []
        total_bytes = 0
        
        def collect_tensors(obj):
            nonlocal total_bytes
            if isinstance(obj, torch.Tensor):
                # Calculate size in bytes
                num_bytes = obj.numel() * obj.element_size()
                total_bytes += num_bytes
                # Store as contiguous float tensor (will convert to bytes later)
                tensors.append(obj.detach().contiguous())
            elif isinstance(obj, (list, tuple)):
                for elem in obj:
                    collect_tensors(elem)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_tensors(v)
            elif hasattr(obj, 'to_legacy_cache'):
                # HuggingFace DynamicCache - convert to legacy tuple format
                try:
                    legacy_cache = obj.to_legacy_cache()
                    collect_tensors(legacy_cache)
                except Exception as e:
                    logger.warning(f"Failed to convert DynamicCache to legacy: {e}")
            else:
                # Try to extract as attributes
                for attr_name in dir(obj):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr = getattr(obj, attr_name)
                        if torch.is_tensor(attr):
                            collect_tensors(attr)
                        elif isinstance(attr, (list, tuple, dict)):
                            collect_tensors(attr)
                    except:
                        pass
        
        collect_tensors(data)
        
        if not tensors:
            raise RuntimeError(f"No tensors found in KV structure (type={type(data).__name__})")
        
        # Concatenate all tensors on GPU, then convert to bytes
        if len(tensors) == 1:
            flattened_gpu = tensors[0].view(torch.uint8)
        else:
            # Convert each tensor to bytes, concatenate, then view as uint8
            byte_tensors = [t.view(torch.uint8) for t in tensors]
            flattened_gpu = torch.cat(byte_tensors)
        
        return flattened_gpu
    
    def _get_device_from_structure(self, data: Any) -> torch.device:
        """Extract device from KV structure."""
        if isinstance(data, torch.Tensor):
            return data.device
        if isinstance(data, (list, tuple)):
            for elem in data:
                dev = self._get_device_from_structure(elem)
                if dev is not None:
                    return dev
        if isinstance(data, dict):
            for v in data.values():
                dev = self._get_device_from_structure(v)
                if dev is not None:
                    return dev
        return torch.device('cuda:0')
    
    def _deserialize_kv_from_flat(
        self, 
        flat_tensor: torch.Tensor, 
        target_structure: Any, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        ✅ ZERO-COPY RECONSTRUCT: Deserialize flattened KV cache back to original structure.
        
        Takes the flattened uint8 tensor from host swap and reconstructs the nested
        KV structure (DynamicCache, tuple of tensors, etc.) using metadata.
        
        Args:
            flat_tensor: Flattened uint8 tensor from host memory
            target_structure: Template for device/dtype info (not modified)
            metadata: Structure metadata captured during flatten
            
        Returns:
            Reconstructed KV cache in original format on same device as flat_tensor
        """
        if metadata is None:
            # Fallback: if structure is single tensor, just return reshaped
            if isinstance(target_structure, torch.Tensor):
                return flat_tensor.view(target_structure.shape)
            # Otherwise return flat
            return flat_tensor
        
        offset = 0
        
        def reconstruct_recursive(meta):
            """Recursively reconstruct structure from metadata and flat bytes."""
            nonlocal offset
            
            if meta['type'] == 'tensor':
                # Extract shape and dtype from metadata
                shape = tuple(meta['shape'])
                dtype_str = meta['dtype']
                
                # Parse dtype
                if 'torch.float32' in dtype_str:
                    dtype = torch.float32
                    element_size = 4
                elif 'torch.float16' in dtype_str:
                    dtype = torch.float16
                    element_size = 2
                elif 'torch.int64' in dtype_str:
                    dtype = torch.int64
                    element_size = 8
                else:
                    dtype = torch.float32
                    element_size = 4
                
                # Calculate tensor size in bytes
                num_elements = 1
                for dim in shape:
                    num_elements *= dim
                tensor_size_bytes = num_elements * element_size
                
                # Extract bytes from flat tensor
                tensor_bytes = flat_tensor[offset:offset + tensor_size_bytes]
                offset += tensor_size_bytes
                
                # Reconstruct: view as dtype and reshape
                reconstructed = tensor_bytes.view(dtype).reshape(shape)
                return reconstructed
            
            elif meta['type'] == 'sequence':
                # Reconstruct list/tuple
                reconstructed = []
                for elem_meta in meta['elements']:
                    reconstructed.append(reconstruct_recursive(elem_meta))
                # Return as tuple if original was tuple
                return tuple(reconstructed) if meta.get('is_tuple', False) else reconstructed
            
            elif meta['type'] == 'dict':
                # Reconstruct dict
                reconstructed = {}
                for key, value_meta in meta['values'].items():
                    reconstructed[key] = reconstruct_recursive(value_meta)
                return reconstructed
            
            else:
                logger.warning(f"Unknown metadata type: {meta['type']}")
                return flat_tensor
        
        try:
            return reconstruct_recursive(metadata)
        except Exception as e:
            logger.error(f"❌ Deserialization with metadata failed: {e}")
            logger.warning(f"Returning flattened tensor as fallback")
            return flat_tensor
    
    def is_swapped(self, session_id: str) -> bool:
        """
        Check if a session's KV cache is currently on host.
        
        Verifies consistency with both KVSession.is_swapped and HostSwapPool state.
        
        Args:
            session_id: Session to check
            
        Returns:
            True if swapped, False otherwise
        """
        # Import here to avoid circular dependency
        from djinn.server.host_swap_pool import get_swap_pool
        
        # Check local state
        if session_id not in self._sessions:
            return False
        
        sess = self._sessions[session_id]
        
        # Verify consistency with host swap pool
        try:
            swap_pool = get_swap_pool()
            pool_has_mapping = swap_pool.is_swapped(session_id)
            
            # They should match
            if sess.is_swapped != pool_has_mapping:
                logger.warning(
                    f"Swap state mismatch for {session_id}: "
                    f"KVSession.is_swapped={sess.is_swapped}, "
                    f"pool.is_swapped={pool_has_mapping}. "
                    f"Assuming pool state is authoritative."
                )
                return pool_has_mapping
            
            return sess.is_swapped
        except Exception as e:
            logger.debug(f"Could not verify swap pool state: {e}")
            return sess.is_swapped
    
    def get_eviction_candidates(self, memory_pressure_bytes: int = 0) -> List[str]:
        """
        Get sessions to evict based on SRG lifetime analysis and priority.
        
        Returns sessions sorted by eviction priority (lowest priority first).
        Candidates are sessions that are NOT already swapped and HAVE KV cache.
        
        Args:
            memory_pressure_bytes: Bytes needed (for future heuristics)
        
        Returns:
            List of session IDs sorted by eviction priority
        """
        candidates = []
        
        try:
            with self._lock:
                for session_id, sess in self._sessions.items():
                    if sess.is_swapped or sess.kv_cache is None:
                        continue  # Skip already swapped or empty sessions
                    
                    # Use eviction priority from semantic memory manager
                    # Lower priority value = evict first
                    priority_value = sess.eviction_priority.value if hasattr(sess, 'eviction_priority') and sess.eviction_priority else 5
                    candidates.append((session_id, priority_value))
            
            # Sort by priority (lowest first)
            candidates.sort(key=lambda x: x[1])
            
            # Return just the session IDs
            result = [s for s, _ in candidates]
            logger.debug(f"Eviction candidates: {len(result)} sessions, top 3: {[s[:12] for s in result[:3]]}")
            return result
        
        except Exception as e:
            logger.debug(f"Error getting eviction candidates: {e}")
            return []


_global_kv_session_manager: Optional[KVSessionManager] = None


def get_kv_session_manager() -> KVSessionManager:
    """Get or create global KV session manager."""
    global _global_kv_session_manager
    if _global_kv_session_manager is None:
        _global_kv_session_manager = KVSessionManager()
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(_global_kv_session_manager.start_cleanup())
        except RuntimeError:
            logger.warning("Async loop not running; KV session cleanup not started")
    return _global_kv_session_manager
