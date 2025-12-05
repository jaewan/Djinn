"""
Semantic Idle Detector: Proactive idle session detection for Phase 3 Semantic Scheduler.

Purpose:
- Track per-session GPU activity timestamps
- Detect sessions idle > idle_threshold_seconds (default: 1.0s)
- Emit SESSION_IDLE events to trigger swap-to-host of KV cache
- Enable parking lot solution: free GPU compute during agent think time

Architecture:
- SemanticActivityTracker: Singleton that monitors all sessions
- Background thread polls idle detection every check_interval_seconds
- Hooks into server.execute_model() to update activity timestamps
- Integrates with session_manager and host_swap_pool

Key Insight:
Unlike blind LRU eviction, semantic idle detection knows when to swap:
- Agent enters "Act" (tool wait) phase: gap in LazyTensor operations
- SemanticActivityTracker detects this pattern, triggers swap
- Agent resumes "Reflect": session restored from host, latency budget ~333ms (8GB @ 24GB/s)

This enables 50 agents to share 60GB GPU VRAM by time-sharing:
- Reason phase: KV cache on GPU
- Act phase: KV cache on host (frees ~2GB per agent * 50 = 100GB potential)
- Reflect phase: KV cache restored, execution continues
"""

import logging
import threading
import time
import asyncio
from typing import Dict, Optional, Set, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SessionIdleState(Enum):
    """Session activity state machine."""
    ACTIVE = "active"          # Recent GPU operation
    IDLE = "idle"              # Idle > idle_threshold_seconds
    SWAPPED = "swapped"        # KV on host, awaiting restore
    DEAD = "dead"              # Session terminated


@dataclass
class SessionActivity:
    """Per-session activity tracking."""
    session_id: str
    created_at: float
    last_op_time: float  # time of last GPU operation
    state: SessionIdleState = SessionIdleState.ACTIVE
    idle_detected_at: Optional[float] = None  # Time when first marked idle
    swap_event_count: int = 0  # Number of swap events
    restore_event_count: int = 0  # Number of restore events
    total_idle_time_seconds: float = 0.0  # Cumulative idle time
    received_signal: bool = False  # True if client sent signal_phase() for this session
    
    @property
    def idle_duration(self) -> float:
        """Seconds since last operation."""
        return time.time() - self.last_op_time
    
    @property
    def is_idle(self) -> bool:
        """Check if currently idle."""
        return self.state in (SessionIdleState.IDLE, SessionIdleState.SWAPPED)


class SemanticActivityTracker:
    """
    Monitors session activity and detects idle sessions for proactive KV swapping.
    
    Responsible for:
    1. Tracking per-session GPU operation timestamps
    2. Background idle detection (configurable threshold)
    3. Emitting idle/resume events to swap pool (async-aware)
    4. Providing metrics for observability
    
    Thread-safe singleton pattern. Async callbacks are wrapped and run in thread pool.
    """
    
    def __init__(
        self,
        idle_threshold_seconds: float = 1.0,
        check_interval_seconds: float = 0.1,
        enabled: bool = True
    ):
        """
        Initialize the activity tracker.
        
        Args:
            idle_threshold_seconds: Mark session idle after this duration (default: 1.0s)
            check_interval_seconds: Polling interval for idle detection (default: 0.1s)
            enabled: Whether to enable semantic idle detection (default: True)
        """
        self.idle_threshold_seconds = idle_threshold_seconds
        self.check_interval_seconds = check_interval_seconds
        self.enabled = enabled
        
        self._sessions: Dict[str, SessionActivity] = {}
        self._lock = threading.Lock()
        
        # Callbacks for idle/resume events - support both sync and async
        self._idle_callbacks: Set[Callable[[str], Optional[Awaitable[None]]]] = set()
        self._resume_callbacks: Set[Callable[[str], Optional[Awaitable[None]]]] = set()
        
        # Event loop reference for running async callbacks
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Background monitor thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "sessions_tracked": 0,
            "idle_detections": 0,
            "resume_detections": 0,
            "total_idle_seconds": 0.0,
            "callback_errors": 0,
        }
        
        logger.info(
            "SemanticActivityTracker initialized: "
            f"idle_threshold={idle_threshold_seconds}s, "
            f"check_interval={check_interval_seconds}s, "
            f"enabled={enabled}"
        )
    
    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Start the background idle detection thread.

        Args:
            event_loop: Optional asyncio event loop for running async callbacks.
                       If not provided, will try to get the running loop.
        """
        logger.info(f"🔧 SemanticActivityTracker.start() called with enabled={self.enabled}")

        if not self.enabled:
            logger.info("Semantic idle detector disabled")
            return

        if self._monitor_thread is not None:
            logger.warning("Activity tracker already started")
            return
        
        # Try to capture the event loop
        try:
            if event_loop is None:
                try:
                    event_loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop, will try to get it later during callbacks
                    pass
            self._event_loop = event_loop
        except Exception as e:
            logger.debug(f"Could not capture event loop: {e}")
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SemanticIdleDetector"
        )
        self._monitor_thread.start()
        logger.info("SemanticActivityTracker monitor thread started")
    
    def stop(self):
        """Stop the background monitor thread."""
        if self._monitor_thread is None:
            return
        
        self._stop_event.set()
        self._monitor_thread.join(timeout=5.0)
        self._monitor_thread = None
        logger.info("SemanticActivityTracker monitor thread stopped")
    
    def register_session(self, session_id: str) -> None:
        """
        Register a new session for activity tracking.
        
        Called when a session is created.
        
        Args:
            session_id: Unique session identifier
        """
        with self._lock:
            if session_id in self._sessions:
                logger.debug(f"Session {session_id} already tracked")
                return
            
            now = time.time()
            self._sessions[session_id] = SessionActivity(
                session_id=session_id,
                created_at=now,
                last_op_time=now,
            )
            self.stats["sessions_tracked"] += 1
            logger.debug(f"Registered session for tracking: {session_id}")
    
    def record_operation(self, session_id: str) -> None:
        """
        Record GPU operation for a session (update timestamp).
        
        Called on every execute_model() call. If session was idle, mark as ACTIVE.
        
        Args:
            session_id: Session that performed operation
        """
        if not self.enabled:
            return
        
        with self._lock:
            if session_id not in self._sessions:
                # Auto-register if not yet tracked
                self.register_session(session_id)
                return
            
            activity = self._sessions[session_id]
            was_idle = activity.is_idle
            
            # Update timestamp and state
            activity.last_op_time = time.time()
            
            # Track cumulative idle time if resuming
            if was_idle and activity.idle_detected_at is not None:
                idle_duration = activity.last_op_time - activity.idle_detected_at
                activity.total_idle_time_seconds += idle_duration
                activity.idle_detected_at = None
            
            resume_callbacks_to_run = []
            if activity.state != SessionIdleState.ACTIVE:
                activity.state = SessionIdleState.ACTIVE
                activity.restore_event_count += 1
                self.stats["resume_detections"] += 1
                logger.debug(f"Session {session_id} resumed (was idle)")
                resume_callbacks_to_run = list(self._resume_callbacks)
        
        # Run resume callbacks (outside lock)
        for callback in resume_callbacks_to_run:
            self._invoke_callback(callback, session_id, "resume")
    
    def unregister_session(self, session_id: str) -> None:
        """
        Unregister a session (called on session cleanup).
        
        Args:
            session_id: Session to stop tracking
        """
        with self._lock:
            if session_id in self._sessions:
                activity = self._sessions[session_id]
                activity.state = SessionIdleState.DEAD
                del self._sessions[session_id]
                logger.debug(f"Unregistered session: {session_id}")
    
    def mark_signal_managed(self, session_id: str) -> None:
        """Mark session as managed via signal_phase() (not idle timeout)."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].received_signal = True
                logger.debug(f"Session {session_id[:12]} marked as signal-managed")
    
    def register_idle_callback(self, callback: Callable[[str], Optional[Awaitable[None]]]) -> None:
        """
        Register callback to invoke when session becomes idle.
        
        Supports both sync and async callbacks:
        - Sync: def idle_callback(session_id: str) -> None
        - Async: async def idle_callback(session_id: str) -> None
        
        Args:
            callback: Function(session_id) -> None or Coroutine
        """
        self._idle_callbacks.add(callback)
    
    def register_resume_callback(self, callback: Callable[[str], Optional[Awaitable[None]]]) -> None:
        """
        Register callback to invoke when session resumes from idle.
        
        Supports both sync and async callbacks:
        - Sync: def resume_callback(session_id: str) -> None
        - Async: async def resume_callback(session_id: str) -> None
        
        Args:
            callback: Function(session_id) -> None or Coroutine
        """
        self._resume_callbacks.add(callback)
    
    def get_session_state(self, session_id: str) -> Optional[SessionActivity]:
        """
        Get activity state for a session.
        
        Args:
            session_id: Session to query
            
        Returns:
            SessionActivity or None if not tracked
        """
        with self._lock:
            return self._sessions.get(session_id)
    
    def get_idle_sessions(self) -> list[str]:
        """
        Get list of currently idle session IDs.
        
        Returns:
            List of session_ids in IDLE or SWAPPED state
        """
        with self._lock:
            return [
                sid for sid, activity in self._sessions.items()
                if activity.is_idle
            ]
    
    def _invoke_callback(
        self, 
        callback: Callable[[str], Optional[Awaitable[None]]], 
        session_id: str,
        callback_type: str
    ) -> None:
        """
        Invoke a callback, handling both sync and async functions.
        
        Args:
            callback: Sync or async callable
            session_id: Session ID to pass to callback
            callback_type: "idle" or "resume" for logging
        """
        try:
            result = callback(session_id)
            
            # Check if result is a coroutine (async callback)
            if asyncio.iscoroutine(result):
                # Try to run async callback
                try:
                    # Try current event loop
                    if self._event_loop and not self._event_loop.is_closed():
                        asyncio.run_coroutine_threadsafe(result, self._event_loop)
                    else:
                        # Try to get running loop
                        try:
                            loop = asyncio.get_running_loop()
                            asyncio.run_coroutine_threadsafe(result, loop)
                        except RuntimeError:
                            # No running loop, create new one in thread
                            asyncio.run(result)
                except Exception as e:
                    logger.error(f"{callback_type.upper()} callback error: {e}")
                    self.stats["callback_errors"] += 1
        except Exception as e:
            logger.error(f"{callback_type.upper()} callback failed: {e}")
            self.stats["callback_errors"] += 1
    
    def _monitor_loop(self) -> None:
        """Background thread: detect idle sessions and emit events."""
        logger.info("🔄 Idle detection monitor loop started")
        
        while not self._stop_event.is_set():
            try:
                self._detect_idle_sessions()
            except Exception as e:
                logger.error(f"Error in idle detection loop: {e}")
            
            # Sleep for check interval
            self._stop_event.wait(self.check_interval_seconds)
        
        logger.debug("Idle detection monitor loop stopped")
    
    def _detect_idle_sessions(self) -> None:
        """
        Detect newly idle sessions and emit callbacks.
        
        Called periodically from monitor thread.
        """
        now = time.time()
        to_mark_idle = []
        
        with self._lock:
            for session_id, activity in self._sessions.items():
                # Skip if already idle
                if activity.is_idle:
                    continue
                
                # ✅ Phase 4: Skip sessions managed via signal_phase() (fallback only for legacy clients)
                if activity.received_signal:
                    continue  # This session is managed by explicit signals, not timeout
                
                # Check if idle threshold exceeded
                idle_duration = now - activity.last_op_time
                if idle_duration >= self.idle_threshold_seconds:
                    to_mark_idle.append((session_id, activity))
        
        # Emit idle events (outside lock to avoid deadlock)
        for session_id, activity in to_mark_idle:
            with self._lock:
                # Double-check still active
                if session_id in self._sessions and not self._sessions[session_id].is_idle:
                    self._sessions[session_id].state = SessionIdleState.IDLE
                    self._sessions[session_id].idle_detected_at = now
                    self._sessions[session_id].swap_event_count += 1
                    self.stats["idle_detections"] += 1
            
            logger.debug(f"Detected idle session: {session_id}")
            
            # Emit callbacks
            for callback in self._idle_callbacks:
                self._invoke_callback(callback, session_id, "idle")
    
    def get_stats(self) -> Dict:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary of stats
        """
        with self._lock:
            active_count = sum(
                1 for a in self._sessions.values()
                if a.state == SessionIdleState.ACTIVE
            )
            idle_count = sum(
                1 for a in self._sessions.values()
                if a.is_idle
            )
            total_idle_seconds = sum(
                a.total_idle_time_seconds for a in self._sessions.values()
            )
            
            return {
                **self.stats,
                "active_sessions": active_count,
                "idle_sessions": idle_count,
                "total_tracked_sessions": len(self._sessions),
                "total_idle_seconds": total_idle_seconds,
            }


# Global singleton instance
_global_activity_tracker: Optional[SemanticActivityTracker] = None


def get_activity_tracker(
    idle_threshold_seconds: float = 1.0,
    check_interval_seconds: float = 0.1,
    enabled: bool = True,
) -> SemanticActivityTracker:
    """Get or create global activity tracker singleton."""
    global _global_activity_tracker
    
    if _global_activity_tracker is None:
        _global_activity_tracker = SemanticActivityTracker(
            idle_threshold_seconds=idle_threshold_seconds,
            check_interval_seconds=check_interval_seconds,
            enabled=enabled,
        )
    
    return _global_activity_tracker

