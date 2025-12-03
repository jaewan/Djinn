"""
Breakpoint Manager: Coordinate pause/resume at layer boundaries.

Purpose:
- Register breakpoints at specific layer indices
- Manage pause/resume state machine per session
- Trigger checkpoint/restore operations via ActivationCheckpointer
- Coordinate with GPU scheduler to yield/reclaim GPU resources

Architecture:
- Per-session breakpoint state tracking (RUNNING, PAUSED, RESUMING)
- Breakpoint hooks installed via BreakpointHookManager
- Checkpoint/restore coordinated with execution engine
- Integrates with SemanticIdleDetector (breakpoint = forced idle)
"""

import logging
import threading
import time
from typing import Dict, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BreakpointState(Enum):
    """Breakpoint state machine."""
    RUNNING = "running"              # Execution in progress
    PAUSED = "paused"                # Paused at breakpoint, checkpoint saved
    RESUMING = "resuming"            # Restoring checkpoint, about to resume
    COMPLETED = "completed"          # Breakpoint reached, no more execution


@dataclass
class SessionBreakpoint:
    """Per-session breakpoint configuration."""
    session_id: str
    layer_index: int                 # Layer to pause at
    state: BreakpointState = BreakpointState.RUNNING
    checkpoint_id: Optional[str] = None
    paused_at: Optional[float] = None
    resumed_at: Optional[float] = None
    pause_duration: float = 0.0
    
    @property
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self.state in (BreakpointState.PAUSED, BreakpointState.RESUMING)


class BreakpointManager:
    """
    Manages breakpoint state and coordinates checkpoint/restore operations.
    
    Responsible for:
    1. Register breakpoints per session
    2. Track pause/resume state transitions
    3. Coordinate with ActivationCheckpointer for persistence
    4. Integrate with execution engine for interruption
    5. Provide metrics for evaluation
    """
    
    def __init__(self):
        """Initialize breakpoint manager."""
        self._lock = threading.Lock()
        
        # Breakpoint registration: session_id -> SessionBreakpoint
        self.breakpoints: Dict[str, SessionBreakpoint] = {}
        
        # Callbacks
        self._pause_callbacks: Set[Callable[[str], None]] = set()
        self._resume_callbacks: Set[Callable[[str], None]] = set()
        self._checkpoint_ready_callbacks: Set[Callable[[str, str], None]] = set()
        
        # Statistics
        self.stats = {
            "breakpoints_registered": 0,
            "pauses_triggered": 0,
            "resumes_triggered": 0,
            "total_pause_duration_seconds": 0.0,
            "checkpoint_errors": 0,
            "restore_errors": 0,
        }
        
        logger.info("✅ BreakpointManager initialized")
    
    def register_breakpoint(
        self,
        session_id: str,
        layer_index: int
    ) -> SessionBreakpoint:
        """
        Register a breakpoint for a session.
        
        Args:
            session_id: Session identifier
            layer_index: Layer to pause at
        
        Returns:
            SessionBreakpoint configuration
        
        Raises:
            ValueError: If layer_index invalid or breakpoint already registered
        """
        if layer_index < 0:
            raise ValueError(f"Invalid layer_index: {layer_index}")
        
        with self._lock:
            if session_id in self.breakpoints:
                raise ValueError(f"Breakpoint already registered for session {session_id}")
            
            breakpoint = SessionBreakpoint(
                session_id=session_id,
                layer_index=layer_index
            )
            self.breakpoints[session_id] = breakpoint
            self.stats["breakpoints_registered"] += 1
            
            logger.info(
                f"Registered breakpoint: session_id={session_id}, layer={layer_index}"
            )
            
            return breakpoint
    
    def trigger_pause(
        self,
        session_id: str,
        checkpoint_id: str
    ) -> SessionBreakpoint:
        """
        Trigger pause at breakpoint (called by hook when breakpoint layer reached).
        
        Args:
            session_id: Session to pause
            checkpoint_id: ID of checkpoint just created
        
        Returns:
            Updated SessionBreakpoint
        
        Raises:
            KeyError: If session/breakpoint not found
        """
        with self._lock:
            if session_id not in self.breakpoints:
                raise KeyError(f"No breakpoint for session {session_id}")
            
            breakpoint = self.breakpoints[session_id]
            
            if breakpoint.state != BreakpointState.RUNNING:
                logger.warning(
                    f"trigger_pause called on session {session_id} in state {breakpoint.state}"
                )
                return breakpoint
            
            # Update state
            breakpoint.state = BreakpointState.PAUSED
            breakpoint.checkpoint_id = checkpoint_id
            breakpoint.paused_at = time.time()
            self.stats["pauses_triggered"] += 1
            
            logger.info(
                f"Breakpoint paused: session_id={session_id}, "
                f"layer={breakpoint.layer_index}, "
                f"checkpoint_id={checkpoint_id}"
            )
        
        # Emit callbacks (outside lock)
        for callback in self._pause_callbacks:
            try:
                callback(session_id)
            except Exception as e:
                logger.error(f"Pause callback failed: {e}")
        
        # Emit checkpoint ready callbacks
        for callback in self._checkpoint_ready_callbacks:
            try:
                callback(session_id, checkpoint_id)
            except Exception as e:
                logger.error(f"Checkpoint ready callback failed: {e}")
        
        return breakpoint
    
    def trigger_resume(self, session_id: str) -> SessionBreakpoint:
        """
        Trigger resume from pause (called by execution engine when resuming).
        
        Args:
            session_id: Session to resume
        
        Returns:
            Updated SessionBreakpoint
        
        Raises:
            KeyError: If session/breakpoint not found
        """
        with self._lock:
            if session_id not in self.breakpoints:
                raise KeyError(f"No breakpoint for session {session_id}")
            
            breakpoint = self.breakpoints[session_id]
            
            if breakpoint.state != BreakpointState.PAUSED:
                logger.warning(
                    f"trigger_resume called on session {session_id} in state {breakpoint.state}"
                )
                return breakpoint
            
            # Update state and track pause duration
            breakpoint.state = BreakpointState.RESUMING
            breakpoint.resumed_at = time.time()
            
            if breakpoint.paused_at is not None:
                pause_duration = breakpoint.resumed_at - breakpoint.paused_at
                breakpoint.pause_duration = pause_duration
                self.stats["total_pause_duration_seconds"] += pause_duration
            
            self.stats["resumes_triggered"] += 1
            
            logger.info(
                f"Breakpoint resuming: session_id={session_id}, "
                f"pause_duration={breakpoint.pause_duration*1000:.1f}ms"
            )
        
        # Emit callbacks (outside lock)
        for callback in self._resume_callbacks:
            try:
                callback(session_id)
            except Exception as e:
                logger.error(f"Resume callback failed: {e}")
        
        return breakpoint
    
    def mark_resumed(self, session_id: str) -> SessionBreakpoint:
        """
        Mark session as fully resumed and ready to continue execution.
        
        Args:
            session_id: Session that has resumed
        
        Returns:
            Updated SessionBreakpoint
        """
        with self._lock:
            if session_id not in self.breakpoints:
                raise KeyError(f"No breakpoint for session {session_id}")
            
            breakpoint = self.breakpoints[session_id]
            
            if breakpoint.state == BreakpointState.RESUMING:
                breakpoint.state = BreakpointState.RUNNING
                logger.debug(f"Session {session_id} marked as resumed")
            
            return breakpoint
    
    def get_breakpoint(self, session_id: str) -> Optional[SessionBreakpoint]:
        """Get breakpoint for a session."""
        with self._lock:
            return self.breakpoints.get(session_id)
    
    def is_paused(self, session_id: str) -> bool:
        """Check if session is currently paused."""
        with self._lock:
            bp = self.breakpoints.get(session_id)
            return bp.is_paused if bp else False
    
    def register_pause_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register callback to invoke when session pauses.
        
        Callback signature: callback(session_id: str) -> None
        
        Args:
            callback: Function to call on pause
        """
        self._pause_callbacks.add(callback)
    
    def register_resume_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register callback to invoke when session resumes.
        
        Callback signature: callback(session_id: str) -> None
        
        Args:
            callback: Function to call on resume
        """
        self._resume_callbacks.add(callback)
    
    def register_checkpoint_ready_callback(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """
        Register callback to invoke when checkpoint is ready.
        
        Callback signature: callback(session_id: str, checkpoint_id: str) -> None
        
        Args:
            callback: Function to call when checkpoint created
        """
        self._checkpoint_ready_callbacks.add(callback)
    
    def unregister_breakpoint(self, session_id: str) -> Optional[SessionBreakpoint]:
        """
        Unregister a breakpoint (called when session ends).
        
        Args:
            session_id: Session to unregister
        
        Returns:
            Removed SessionBreakpoint or None
        """
        with self._lock:
            breakpoint = self.breakpoints.pop(session_id, None)
            if breakpoint:
                logger.debug(f"Unregistered breakpoint for session {session_id}")
            return breakpoint
    
    def get_stats(self) -> Dict[str, Any]:
        """Get breakpoint manager statistics."""
        with self._lock:
            active_paused = sum(
                1 for bp in self.breakpoints.values()
                if bp.is_paused
            )
            
            return {
                **self.stats,
                "active_breakpoints": len(self.breakpoints),
                "currently_paused": active_paused,
                "avg_pause_duration_ms": (
                    (self.stats["total_pause_duration_seconds"] * 1000 / self.stats["pauses_triggered"])
                    if self.stats["pauses_triggered"] > 0 else 0.0
                ),
            }
    
    def clear(self) -> None:
        """Clear all breakpoints (e.g., on shutdown)."""
        with self._lock:
            cleared_count = len(self.breakpoints)
            self.breakpoints.clear()
            logger.info(f"Cleared breakpoint manager ({cleared_count} breakpoints)")


# Global singleton instance
_global_breakpoint_manager: Optional[BreakpointManager] = None
_manager_lock = threading.Lock()


def get_breakpoint_manager() -> BreakpointManager:
    """Get or create global breakpoint manager singleton."""
    global _global_breakpoint_manager
    
    if _global_breakpoint_manager is None:
        with _manager_lock:
            if _global_breakpoint_manager is None:
                _global_breakpoint_manager = BreakpointManager()
    
    return _global_breakpoint_manager

