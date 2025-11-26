"""
Session Manager: Distributed Garbage Collection for v2.3

Implements: Session-scoped distributed GC with heartbeat monitoring

Problem:
- Heap allocations (RemoteRefs) can leak if client crashes
- Need automatic cleanup when client disconnects
- Need reference counting to prevent use-after-free

Solution:
- Each session has a lease with heartbeat
- All heap allocations registered to session
- Lease timeout → automatic cleanup
- Reference counting for safety

Architecture:
SessionManager
  └─ sessions[session_id]
       ├─ last_heartbeat: time
       ├─ lease_timeout: float
       ├─ refs: set of heap allocations
       └─ reference_count: dict
"""

import logging
import threading
import time
import uuid
from typing import Dict, Set, Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from enum import Enum

from djinn.backend.runtime.unified_vmu import UnifiedVMU, get_vmu
from djinn.server.memory_metrics import get_metrics as get_memory_metrics

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle states."""
    ACTIVE = "active"
    IDLE = "idle"
    DEAD = "dead"
    CLEANED = "cleaned"


@dataclass
class SessionLease:
    """Lease for a session."""
    session_id: str
    created_at: float
    last_heartbeat: float
    timeout_secs: float = 300.0  # 5 minute default
    refs: Set[str] = field(default_factory=set)  # Tensor reference IDs
    reference_count: Dict[str, int] = field(default_factory=dict)
    state: SessionState = SessionState.ACTIVE
    
    @property
    def is_alive(self) -> bool:
        """Check if session lease is still valid."""
        elapsed = time.time() - self.last_heartbeat
        return elapsed < self.timeout_secs
    
    @property
    def is_expired(self) -> bool:
        """Check if session lease has expired."""
        return not self.is_alive


class SessionManager:
    """
    Manages session leases and distributed garbage collection.
    
    Responsibilities:
    - Track session leases
    - Monitor heartbeats
    - Reference counting
    - Automatic cleanup on timeout
    - Memory tracking per session
    """
    
    def __init__(
        self,
        vmu: Optional[UnifiedVMU] = None,
        heartbeat_timeout_secs: float = 300.0,
        monitor_interval_secs: float = 5.0,
    ):
        """
        Initialize session manager.
        
        Args:
            vmu: Unified VMU for memory tracking
            heartbeat_timeout_secs: Session timeout
        """
        self.vmu = vmu or self._try_get_vmu()
        self.metrics = get_memory_metrics()
        self._default_session_bytes = 0
        if self.vmu:
            self._default_session_bytes = getattr(self.vmu, "default_session_arena_bytes", 0)
        self.heartbeat_timeout_secs = heartbeat_timeout_secs
        self.sessions: Dict[str, SessionLease] = {}
        self.lock = threading.Lock()
        self.cleanup_callbacks: List[Callable[[str], None]] = []
        self.monitor_interval_secs = monitor_interval_secs
        
        # Statistics
        self.stats = {
            'sessions_created': 0,
            'sessions_killed': 0,
            'refs_registered': 0,
            'refs_released': 0,
            'memory_freed_mb': 0.0,
        }
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(
            f"✅ SessionManager initialized (timeout={heartbeat_timeout_secs}s)"
        )

    def register_cleanup_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register a callback invoked when a session is killed/cleaned.

        Args:
            callback: Callable that accepts session_id.
        """
        if not callable(callback):
            raise ValueError("cleanup callback must be callable")

        with self.lock:
            if callback in self.cleanup_callbacks:
                return
            self.cleanup_callbacks.append(callback)
    
    def create_session(
        self,
        max_session_bytes: Optional[int] = None,
        *,
        model_kv_bytes: Optional[int] = None,
        expected_tokens: Optional[int] = None,
    ) -> str:
        """
        Create new session.
        
        Returns:
            session_id
        """
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        with self.lock:
            self.sessions[session_id] = SessionLease(
                session_id=session_id,
                created_at=time.time(),
                last_heartbeat=time.time(),
                timeout_secs=self.heartbeat_timeout_secs
            )
            self.stats['sessions_created'] += 1

        try:
            self._reserve_session_arena(
                session_id,
                max_session_bytes,
                model_kv_bytes=model_kv_bytes,
                expected_tokens=expected_tokens,
            )
        except Exception as exc:
            logger.error("Failed to reserve session arena for %s: %s", session_id, exc)
            with self.lock:
                self.sessions.pop(session_id, None)
                self.stats['sessions_created'] -= 1
            raise

        self._record_session_created()
        logger.debug(f"Created session: {session_id}")
        return session_id
    
    def heartbeat(self, session_id: str) -> bool:
        """
        Record heartbeat for session.
        
        Args:
            session_id: Session ID
        
        Returns:
            True if session is alive, False if expired/unknown
        """
        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"Heartbeat for unknown session: {session_id}")
                return False
            
            lease = self.sessions[session_id]
            
            if lease.is_expired:
                logger.warning(f"Heartbeat for expired session: {session_id}")
                return False
            
            # Update heartbeat
            lease.last_heartbeat = time.time()
            return True
    
    def register_ref(self, session_id: str, ref_id: str, size_bytes: int = 0) -> bool:
        """
        Register heap allocation with session.
        
        Args:
            session_id: Session ID
            ref_id: Reference ID (tensor ID)
            size_bytes: Size for tracking
        
        Returns:
            True if registered successfully
        """
        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"Register ref for unknown session: {session_id}")
                return False
            
            lease = self.sessions[session_id]
            
            if lease.is_expired:
                logger.warning(f"Register ref for expired session: {session_id}")
                return False
            
            lease.refs.add(ref_id)
            lease.reference_count[ref_id] = lease.reference_count.get(ref_id, 0) + 1
            self.stats['refs_registered'] += 1
            
            logger.debug(
                f"Registered ref {ref_id} in {session_id} "
                f"(total refs: {len(lease.refs)})"
            )
            
            return True
    
    def release_ref(self, session_id: str, ref_id: str) -> bool:
        """
        Release heap allocation.
        
        Args:
            session_id: Session ID
            ref_id: Reference ID to release
        
        Returns:
            True if released successfully
        """
        with self.lock:
            if session_id not in self.sessions:
                return False
            
            lease = self.sessions[session_id]
            
            if ref_id not in lease.refs:
                return False
            
            # Decrement reference count
            lease.reference_count[ref_id] -= 1
            
            if lease.reference_count[ref_id] <= 0:
                # Actually remove ref
                lease.refs.discard(ref_id)
                del lease.reference_count[ref_id]
                self.stats['refs_released'] += 1
                
                logger.debug(f"Released ref {ref_id}")
            
            return True
    
    def kill_session(self, session_id: str) -> int:
        """
        Kill session and cleanup all refs.
        
        Args:
            session_id: Session to kill
        
        Returns:
            Number of refs cleaned up
        """
        callbacks: List[Callable[[str], None]] = []
        with self.lock:
            if session_id not in self.sessions:
                return 0

            lease = self.sessions[session_id]

            # Release all refs
            count = len(lease.refs)

            logger.info(
                f"Killing session {session_id}: "
                f"cleaning {count} refs"
            )

            # Mark as dead
            lease.state = SessionState.DEAD
            # Increment refs_released counter for all refs being cleaned up
            self.stats['refs_released'] += count
            lease.refs.clear()
            lease.reference_count.clear()

            # Can optionally keep for audit trail or delete immediately
            # For now, just mark as dead
            lease.state = SessionState.CLEANED

            self.stats['sessions_killed'] += 1
            callbacks = list(self.cleanup_callbacks)

        for callback in callbacks:
            try:
                callback(session_id)
            except Exception as exc:
                logger.error(
                    f"Session cleanup callback failed for {session_id}: {exc}"
                )

        if self.vmu:
            try:
                self.vmu.free_session_data(session_id)
            except Exception as exc:
                logger.error("Failed to free VMU arena for session %s: %s", session_id, exc)

        self._record_session_closed(time.time() - lease.created_at)

        return count
    
    def _monitor_loop(self):
        """Background thread monitoring heartbeats."""
        while True:
            try:
                time.sleep(self.monitor_interval_secs)
                
                expired_sessions = []
                
                # Find expired sessions (hold lock briefly)
                with self.lock:
                    for session_id, lease in list(self.sessions.items()):
                        if lease.is_expired and lease.state == SessionState.ACTIVE:
                            expired_sessions.append(session_id)
                
                # Kill expired sessions WITHOUT holding lock
                # (kill_session acquires lock internally)
                for session_id in expired_sessions:
                    self.kill_session(session_id)
            
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session."""
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            lease = self.sessions[session_id]
            
            return {
                'session_id': session_id,
                'state': lease.state.value,
                'created_at': lease.created_at,
                'last_heartbeat': lease.last_heartbeat,
                'is_alive': lease.is_alive,
                'refs_count': len(lease.refs),
                'timeout_secs': lease.timeout_secs,
            }
    
    def get_stats(self) -> Dict:
        """Get session manager statistics."""
        with self.lock:
            active_count = sum(
                1 for lease in self.sessions.values()
                if lease.state == SessionState.ACTIVE
            )
            total_refs = sum(
                len(lease.refs) for lease in self.sessions.values()
            )
        
        return {
            **self.stats,
            'active_sessions': active_count,
            'total_sessions': len(self.sessions),
            'total_refs': total_refs,
        }
    
    def cleanup_all(self) -> int:
        """Cleanup all sessions (for shutdown)."""
        with self.lock:
            sessions_to_kill = list(self.sessions.keys())
        
        total_refs = 0
        for session_id in sessions_to_kill:
            total_refs += self.kill_session(session_id)

        with self.lock:
            self.sessions.clear()
        
        self._update_session_metrics()
        logger.info(f"Cleaned up all sessions ({total_refs} refs)")
        return total_refs

    def _active_session_count(self) -> int:
        with self.lock:
            return sum(1 for lease in self.sessions.values() if lease.state == SessionState.ACTIVE)

    def _record_session_created(self) -> None:
        if not self.metrics:
            return
        self.metrics.record_session_created()
        self._update_session_metrics()

    def _record_session_closed(self, lifetime_secs: float) -> None:
        if not self.metrics:
            return
        self.metrics.record_session_closed(lifetime_secs)
        self._update_session_metrics()

    def _update_session_metrics(self) -> None:
        if not self.metrics:
            return
        active = self._active_session_count()
        self.metrics.set_active_sessions(active)
        self._update_kv_pinned_bytes()

    def _update_kv_pinned_bytes(self) -> None:
        if not (self.metrics and self.vmu):
            return
        try:
            vmu_metrics = self.vmu.get_metrics()
            # Update existing KV pinned bytes metric
            self.metrics.set_pinned_kv_bytes(vmu_metrics.data_reserved_bytes)
            # Update new VMU metrics
            self.metrics.set_vmu_text_metrics(
                vmu_metrics.text_used_bytes,
                vmu_metrics.text_capacity_bytes
            )
            self.metrics.set_vmu_data_metrics(
                vmu_metrics.data_reserved_bytes,
                vmu_metrics.data_capacity_bytes,
                vmu_metrics.data_internal_waste_bytes,
                vmu_metrics.data_external_gap_bytes
            )
            self.metrics.set_vmu_stack_metrics(
                vmu_metrics.stack_allocated_bytes,
                vmu_metrics.stack_capacity_bytes,
                vmu_metrics.stack_reset_count
            )
            self.metrics.set_vmu_session_metrics(
                vmu_metrics.active_sessions,
                vmu_metrics.models_loaded
            )
        except Exception as exc:
            logger.debug(f"Failed to update VMU metrics: {exc}")

    def _try_get_vmu(self) -> Optional[UnifiedVMU]:
        try:
            return get_vmu()
        except Exception as exc:
            logger.warning(f"SessionManager could not initialize VMU: {exc}")
            return None

    def _estimate_session_bytes(
        self,
        max_session_bytes: Optional[int],
        model_kv_bytes: Optional[int] = None,
        expected_tokens: Optional[int] = None,
    ) -> int:
        if max_session_bytes:
            return max_session_bytes
        estimate = self._default_session_bytes
        if model_kv_bytes and expected_tokens:
            estimate = int(model_kv_bytes * expected_tokens * 1.2)
        elif model_kv_bytes:
            estimate = int(model_kv_bytes * 1.2)
        return max(estimate, self._default_session_bytes)

    def _reserve_session_arena(
        self,
        session_id: str,
        max_session_bytes: Optional[int],
        model_kv_bytes: Optional[int] = None,
        expected_tokens: Optional[int] = None,
    ) -> None:
        if not self.vmu:
            return
        size = self._estimate_session_bytes(max_session_bytes, model_kv_bytes, expected_tokens)
        if size <= 0:
            return
        self.vmu.reserve_session_arena(session_id, size)


# Global session manager instance
_global_session_manager: Optional[SessionManager] = None


def get_session_manager(vmu=None) -> SessionManager:
    """Get or create global session manager."""
    global _global_session_manager
    
    if _global_session_manager is None:
        _global_session_manager = SessionManager(vmu=vmu)
    
    return _global_session_manager

