"""
Model Switch Coordinator: Orchestrates model switching with concurrency safety.

Coordinates:
- Model residency tracking (which models are in GPU)
- Eviction decisions (which model to evict when space needed)
- Swap/restore operations (moving weights between GPU and host)
- Concurrency control (per-model locks + GPU operation semaphore)

Architecture (OPTIMIZED FOR CONCURRENT AGENTS):
┌──────────────────────────────────────────────────────────────┐
│ ModelSwitchCoordinator                                       │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ LRU Tracking│  │ Eviction     │  │ Concurrency      │  │
│  │ (access     │  │ Policy       │  │ (Per-model locks │  │
│  │  times)     │  │ (victim      │  │  + GPU semaphore)│  │
│  └─────────────┘  │  selection)  │  └──────────────────┘  │
│                    └──────────────┘                         │
└──────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐
│ Ring Buffer   │  │ Swap Pool       │  │ Eviction Policy    │
│ (GPU)         │  │ (Host RAM)      │  │ (LRU)              │
└───────────────┘  └─────────────────┘  └────────────────────┘

Key optimizations:
1. Per-model locks: Allow concurrent switches to DIFFERENT models
2. Request coalescing: Multiple agents for same model share one switch
3. GPU semaphore: Limit concurrent GPU operations (prevent OOM)
4. Fast path: Zero-cost check if model already resident
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field

from djinn.backend.runtime.ring_buffer import WeightRingBuffer, ModelState
from djinn.server.model_weight_swap_pool import ModelWeightSwapPool
from djinn.interfaces.model_eviction_policy import (
    ModelEvictionPolicy,
    ModelEvictionCandidate,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelAccessRecord:
    """Track model access for LRU."""
    model_id: str
    last_access: float
    access_count: int = 0


@dataclass
class PendingSwitch:
    """Track pending switch requests for coalescing."""
    model_id: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Optional[bool] = None
    waiters: int = 0


class ModelSwitchCoordinator:
    """
    Orchestrates model switching with optimized concurrency.
    
    Key features:
    - Per-model locks: Allow concurrent switches to different models
    - Request coalescing: Multiple agents for same model share one switch
    - GPU semaphore: Limit concurrent GPU operations
    - Fast path: Zero-cost check if model already resident
    """
    
    def __init__(
        self,
        ring_buffer: WeightRingBuffer,
        swap_pool: ModelWeightSwapPool,
        eviction_policy: ModelEvictionPolicy,
        max_concurrent_switches: int = 2  # Allow 2 concurrent switches
    ):
        """
        Initialize coordinator.
        
        Args:
            ring_buffer: WeightRingBuffer for GPU storage
            swap_pool: ModelWeightSwapPool for host storage
            eviction_policy: Policy for selecting victims
            max_concurrent_switches: Max concurrent GPU switch operations
        """
        self.ring_buffer = ring_buffer
        self.swap_pool = swap_pool
        self.eviction_policy = eviction_policy
        
        # Access tracking for LRU
        self.access_records: Dict[str, ModelAccessRecord] = {}
        self._access_lock = asyncio.Lock()
        
        # Per-model switch locks (allow concurrent switches to different models)
        self._model_switch_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects _model_switch_locks dict
        
        # Pending switches for request coalescing
        self._pending_switches: Dict[str, PendingSwitch] = {}
        self._pending_lock = asyncio.Lock()
        
        # GPU operation semaphore (limit concurrent GPU work)
        self._gpu_semaphore = asyncio.Semaphore(max_concurrent_switches)
        
        # Eviction lock (only one eviction decision at a time)
        self._eviction_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'switches_performed': 0,
            'evictions_triggered': 0,
            'restorations_performed': 0,
            'total_switch_latency_ms': 0.0,
            'coalesced_requests': 0,
            'fast_path_hits': 0,
        }
        
        logger.info(f"✅ ModelSwitchCoordinator initialized (max_concurrent={max_concurrent_switches})")
    
    async def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        """Get or create per-model lock."""
        async with self._locks_lock:
            if model_id not in self._model_switch_locks:
                self._model_switch_locks[model_id] = asyncio.Lock()
            return self._model_switch_locks[model_id]
    
    async def ensure_model_resident(self, model_id: str) -> bool:
        """
        Ensure model is resident in GPU ring buffer.
        
        If model is swapped, restores it (may evict other models).
        If model is already resident, this is a no-op.
        
        OPTIMIZED: Uses per-model locks and request coalescing to allow
        concurrent switches to different models.
        
        Args:
            model_id: Model to ensure is resident
            
        Returns:
            True if model is resident, False if failed
        """
        overall_start = time.perf_counter()
        
        # Fast path: check if already resident (no lock needed)
        if self.ring_buffer.is_model_resident(model_id):
            self.stats['fast_path_hits'] += 1
            await self.record_model_access(model_id)
            return True
        
        # Check if there's already a pending switch for this model
        async with self._pending_lock:
            if model_id in self._pending_switches:
                # Coalesce: wait for existing switch to complete
                pending = self._pending_switches[model_id]
                pending.waiters += 1
                self.stats['coalesced_requests'] += 1
                logger.debug(f"Coalescing request for {model_id[:16]}... ({pending.waiters} waiters)")
        
        # Check again after potentially waiting for pending
        async with self._pending_lock:
            if model_id in self._pending_switches:
                pending = self._pending_switches[model_id]
                # Wait for the switch to complete
                await pending.event.wait()
                pending.waiters -= 1
                # Clean up if we're the last waiter
                if pending.waiters == 0 and model_id in self._pending_switches:
                    del self._pending_switches[model_id]
                return pending.result if pending.result is not None else False
        
        # Register pending switch
        async with self._pending_lock:
            # Double-check no one else started
            if model_id in self._pending_switches:
                pending = self._pending_switches[model_id]
                pending.waiters += 1
                await pending.event.wait()
                pending.waiters -= 1
                if pending.waiters == 0:
                    del self._pending_switches[model_id]
                return pending.result if pending.result is not None else False
            
            # Register our switch
            pending = PendingSwitch(model_id=model_id)
            self._pending_switches[model_id] = pending
        
        # Get per-model lock (allows concurrent switches to different models)
        model_lock = await self._get_model_lock(model_id)
        
        profile = {
            'lock_wait_ms': 0,
            'eviction_ms': 0,
            'restoration_ms': 0,
        }
        
        try:
            lock_wait_start = time.perf_counter()
            async with model_lock:
                profile['lock_wait_ms'] = (time.perf_counter() - lock_wait_start) * 1000
                
                # Double-check after acquiring lock
                if self.ring_buffer.is_model_resident(model_id):
                    await self.record_model_access(model_id)
                    pending.result = True
                    pending.event.set()
                    return True
                
                # Check if model is in swap pool
                if not self.swap_pool.has_model(model_id):
                    logger.error(f"Model {model_id[:16]}... not in swap pool")
                    pending.result = False
                    pending.event.set()
                    return False
                
                # Get model size
                model_size = self.swap_pool.get_model_size(model_id)
                
                # Acquire GPU semaphore (limit concurrent GPU operations)
                async with self._gpu_semaphore:
                    # Make space if needed (serialize eviction decisions)
                    eviction_start = time.perf_counter()
                    eviction_count = 0
                    async with self._eviction_lock:
                        while not self.ring_buffer.can_fit_model(model_size):
                            victim_id = await self._select_and_evict_victim(model_size)
                            if victim_id is None:
                                logger.error(
                                    f"Cannot make room for {model_id[:16]}... "
                                    f"(need {model_size / 1024**3:.1f}GB)"
                                )
                                pending.result = False
                                pending.event.set()
                                return False
                            eviction_count += 1
                    profile['eviction_ms'] = (time.perf_counter() - eviction_start) * 1000
                    
                    # CRITICAL: Wait for all evictions to complete before restoration
                    # to avoid PCIe bandwidth contention between GPU->CPU and CPU->GPU
                    if eviction_count > 0:
                        evict_sync_start = time.perf_counter()
                        await asyncio.to_thread(self.swap_pool.synchronize_all_evictions)
                        profile['evict_sync_ms'] = (time.perf_counter() - evict_sync_start) * 1000
                    else:
                        profile['evict_sync_ms'] = 0.0
                    
                    # Restore model (now with full PCIe bandwidth)
                    restoration_start = time.perf_counter()
                    success = await self._restore_model(model_id)
                    profile['restoration_ms'] = (time.perf_counter() - restoration_start) * 1000
                    
                    if success:
                        await self.record_model_access(model_id)
                    
                    # Log detailed profile
                    total_ms = (time.perf_counter() - overall_start) * 1000
                    evict_sync = profile.get('evict_sync_ms', 0.0)
                    logger.info(
                        f"Model switch {model_id[:16]}...: {total_ms:.1f}ms total "
                        f"(lock_wait={profile['lock_wait_ms']:.1f}ms, "
                        f"eviction={profile['eviction_ms']:.1f}ms [{eviction_count} models], "
                        f"evict_sync={evict_sync:.1f}ms, "
                        f"restoration={profile['restoration_ms']:.1f}ms)"
                    )
                    
                    # Store profile for stats
                    if not hasattr(self, 'switch_profiles'):
                        self.switch_profiles = []
                    self.switch_profiles.append(profile)
                    
                    pending.result = success
                    pending.event.set()
                    return success
                    
        except Exception as e:
            logger.error(f"Switch to {model_id[:16]}... failed: {e}")
            pending.result = False
            pending.event.set()
            return False
        finally:
            # Cleanup pending entry if we're the owner
            async with self._pending_lock:
                if model_id in self._pending_switches and self._pending_switches[model_id].waiters == 0:
                    del self._pending_switches[model_id]
    
    async def _select_and_evict_victim(self, bytes_needed: int) -> Optional[str]:
        """
        Select a victim model and evict it.
        
        Args:
            bytes_needed: Bytes that need to be freed
            
        Returns:
            model_id of evicted model, or None if no victim available
        """
        # Build candidate list
        candidates = []
        
        async with self._access_lock:
            for model_id in self.ring_buffer.get_resident_models():
                # Get access record
                record = self.access_records.get(model_id)
                last_access = record.last_access if record else 0.0
                
                # Get model size
                reg = self.ring_buffer.registrations.get(model_id)
                if not reg:
                    continue
                
                candidate = ModelEvictionCandidate(
                    model_id=model_id,
                    last_access=last_access,
                    size_bytes=reg.total_bytes,
                    is_active=False,  # TODO: Track active executions
                    priority=0
                )
                candidates.append(candidate)
        
        if not candidates:
            logger.warning("No eviction candidates available")
            return None
        
        # Select victim using policy
        victim_id = self.eviction_policy.select_victim(candidates, bytes_needed)
        
        if victim_id is None:
            logger.warning("Policy returned no victim")
            return None
        
        # Evict the victim
        success = await self._evict_model(victim_id)
        if not success:
            return None
        
        self.stats['evictions_triggered'] += 1
        return victim_id
    
    async def _evict_model(self, model_id: str) -> bool:
        """
        Evict a model from ring buffer to swap pool using DIRECT streaming.
        Falls back to per-tensor method if direct fails (e.g., wrapped models).
        
        Args:
            model_id: Model to evict
            
        Returns:
            True if successful, False otherwise
        """
        # PROFILING: Track eviction timing
        evict_start = time.perf_counter()
        
        try:
            # OPTIMIZED: Try direct stream from ring buffer to pinned buffer (single bulk copy)
            bytes_swapped = await asyncio.to_thread(
                self.swap_pool.evict_model_direct,
                model_id,
                self.ring_buffer,
                0  # gpu_device
            )
            
            if bytes_swapped == 0:
                # Direct method failed (e.g., wrapped model), fall back to per-tensor method
                logger.info(f"Direct eviction not available for {model_id[:16]}..., using fallback")
                
                # Get weights and use old method
                weights = self.ring_buffer.get_model_weights_from_buffer(model_id)
                if not weights:
                    logger.error(f"Cannot get weights for {model_id[:16]}...")
                    return False
                
                # Evict from ring buffer first
                bytes_freed = self.ring_buffer.evict_model(model_id)
                if bytes_freed == 0:
                    logger.error(f"Ring buffer eviction returned 0 bytes for {model_id[:16]}...")
                    return False
                
                # Move to swap pool using old method
                bytes_swapped = await asyncio.to_thread(
                    self.swap_pool.evict_model_to_host,
                    model_id,
                    weights,
                    0  # gpu_device
                )
                
                if bytes_swapped == 0:
                    logger.error(f"Swap pool rejected {model_id[:16]}...")
                    return False
                
                total_ms = (time.perf_counter() - evict_start) * 1000
                bandwidth_gbps = (bytes_freed / 1024**3) / (total_ms / 1000) if total_ms > 0 else 0
                logger.info(
                    f"✅ Evicted model {model_id[:16]}... (FALLBACK): "
                    f"{bytes_freed / 1024**3:.1f}GB freed from GPU in {total_ms:.1f}ms "
                    f"({bandwidth_gbps:.1f} GB/s)"
                )
                return True
            
            # Direct method succeeded
            # Mark model as evicted in ring buffer (just updates state)
            bytes_freed = self.ring_buffer.evict_model(model_id)
            
            if bytes_freed == 0:
                logger.error(f"Ring buffer eviction returned 0 bytes for {model_id[:16]}...")
                return False
            
            total_ms = (time.perf_counter() - evict_start) * 1000
            bandwidth_gbps = (bytes_freed / 1024**3) / (total_ms / 1000) if total_ms > 0 else 0
            logger.info(
                f"✅ Evicted model {model_id[:16]}... (DIRECT): "
                f"{bytes_freed / 1024**3:.1f}GB freed from GPU in {total_ms:.1f}ms "
                f"({bandwidth_gbps:.1f} GB/s)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evict {model_id[:16]}...: {e}", exc_info=True)
            return False
    
    async def _restore_model(self, model_id: str) -> bool:
        """
        Restore a model from swap pool to ring buffer using DIRECT streaming.
        
        Args:
            model_id: Model to restore
            
        Returns:
            True if successful, False otherwise
        """
        # PROFILING: Track restoration timing
        restore_start = time.perf_counter()
        
        try:
            # OPTIMIZED: Direct stream from pinned buffer to ring buffer (single bulk copy)
            bytes_restored = await asyncio.to_thread(
                self.swap_pool.restore_model_direct,
                model_id,
                self.ring_buffer,
                self.ring_buffer.device
            )
            
            if bytes_restored == 0:
                logger.error(f"Direct restoration failed for {model_id[:16]}...")
                return False
            
            restore_ms = (time.perf_counter() - restore_start) * 1000
            bandwidth_gbps = (bytes_restored / 1024**3) / (restore_ms / 1000) if restore_ms > 0 else 0
            
            self.stats['restorations_performed'] += 1
            self.stats['total_switch_latency_ms'] += restore_ms
            
            logger.info(
                f"✅ Restored model {model_id[:16]}... (DIRECT): "
                f"{bytes_restored / 1024**3:.1f}GB in {restore_ms:.1f}ms "
                f"({bandwidth_gbps:.1f} GB/s)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore {model_id[:16]}...: {e}", exc_info=True)
            return False
    
    async def record_model_access(self, model_id: str) -> None:
        """
        Record that a model was accessed (for LRU tracking).
        
        Args:
            model_id: Model that was accessed
        """
        async with self._access_lock:
            if model_id not in self.access_records:
                self.access_records[model_id] = ModelAccessRecord(
                    model_id=model_id,
                    last_access=time.time(),
                    access_count=1
                )
            else:
                record = self.access_records[model_id]
                record.last_access = time.time()
                record.access_count += 1
    
    def get_active_model(self) -> Optional[str]:
        """
        Get currently active model ID.
        
        Returns:
            Active model ID, or None if no active model
        """
        return self.ring_buffer.active_model_id
    
    async def switch_to_model(self, model_id: str) -> float:
        """
        Switch to a specific model, ensuring it's resident.
        
        This is a convenience method that combines ensure_resident with
        latency measurement.
        
        Args:
            model_id: Model to switch to
            
        Returns:
            Switch latency in milliseconds, or -1 if failed
        """
        start = time.perf_counter()
        
        success = await self.ensure_model_resident(model_id)
        
        if not success:
            return -1.0
        
        # Update active model
        self.ring_buffer.active_model_id = model_id
        self.stats['switches_performed'] += 1
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        return latency_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        avg_switch_latency = 0.0
        if self.stats['switches_performed'] > 0:
            avg_switch_latency = (
                self.stats['total_switch_latency_ms'] / 
                self.stats['switches_performed']
            )
        
        return {
            **self.stats,
            'avg_switch_latency_ms': avg_switch_latency,
            'tracked_models': len(self.access_records),
            'ring_buffer_stats': self.ring_buffer.get_stats(),
            'swap_pool_stats': self.swap_pool.get_stats(),
        }

