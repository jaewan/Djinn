"""
djinn/core/registration_state_machine.py

Thread-safe state machine for model registration lifecycle.

This implements Phase 0 of the redesign plan: Registration State Machine Specification.
"""

import asyncio
import contextvars
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class State(Enum):
    """Model registration states."""
    UNKNOWN = auto()      # Never seen before
    GRAPH_ONLY = auto()   # Using graph execution
    REGISTERING = auto()  # Registration in progress
    REGISTERED = auto()   # Registration complete, cache available


@dataclass
class CallStatistics:
    """Statistics for a model fingerprint."""
    total_calls: int = 0
    first_call_time: float = 0.0
    last_call_time: float = 0.0
    registration_attempts: int = 0
    registration_failures: int = 0
    
    def call_rate_per_second(self) -> float:
        """Compute current call rate."""
        if self.total_calls <= 1:
            return 0.0
        
        elapsed = time.time() - self.first_call_time
        if elapsed < 0.001:
            return 0.0
        
        return self.total_calls / elapsed


@dataclass
class RegistrationConfig:
    """Configuration for registration behavior."""
    
    # Threshold for triggering registration
    call_count_threshold: int = 3
    
    # Frequency threshold (calls per second)
    min_call_rate: float = 1.0 / 60.0  # At least 1 per minute
    
    # Idle threshold (seconds since last call)
    max_idle_time: float = 300.0  # 5 minutes
    
    # Retry policy
    max_registration_attempts: int = 3
    retry_backoff_seconds: list = field(default_factory=lambda: [1.0, 5.0, 30.0])
    
    # Failure behavior
    permanent_fallback_on_failure: bool = True
    
    # Load-aware registration
    defer_registration_if_high_load: bool = True
    high_load_threshold: float = 0.9  # 90% GPU utilization


# Context variable for detecting re-entrant registration
_in_registration_context = contextvars.ContextVar('in_registration', default=None)


class RegistrationStateMachine:
    """
    Thread-safe state machine for model registration lifecycle.
    
    Responsibilities:
    1. Track state per fingerprint (UNKNOWN → GRAPH_ONLY → REGISTERING → REGISTERED)
    2. Decide execution path (cache vs graph) based on state
    3. Trigger registration when conditions met
    4. Handle registration failures with retry/backoff
    5. Prevent deadlocks via re-entrancy detection
    
    Thread safety:
    - Per-fingerprint locks for state transitions
    - Separate registration locks to prevent deadlock
    - Atomic state updates
    
    Example usage:
        state_machine = RegistrationStateMachine()
        state_machine.set_registration_callback(my_register_function)
        
        # For each request:
        path = await state_machine.get_execution_path(fingerprint)
        
        if path == 'cache':
            result = execute_via_cache(fingerprint, inputs)
        else:
            result = execute_via_graph(model, inputs)
    """
    
    def __init__(self, config: Optional[RegistrationConfig] = None):
        self.config = config or RegistrationConfig()
        
        # State tracking
        self._states: Dict[str, State] = {}
        self._call_stats: Dict[str, CallStatistics] = defaultdict(CallStatistics)
        
        # Locking (separate locks to prevent deadlock)
        self._state_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._registration_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Background tasks
        self._registration_tasks: Dict[str, asyncio.Task] = {}
        
        # Callbacks
        self._registration_callback: Optional[Callable] = None
        self._metrics_callback: Optional[Callable] = None
    
    def set_registration_callback(self, callback: Callable):
        """Set callback for actual registration work."""
        self._registration_callback = callback
    
    def set_metrics_callback(self, callback: Callable):
        """Set callback for metrics reporting."""
        self._metrics_callback = callback
    
    async def get_execution_path(self, fingerprint: str) -> str:
        """
        Determine execution path for this request.
        
        Returns:
            'cache' - Use cached model execution
            'graph' - Use graph execution
        
        Side effects:
            - May trigger background registration
            - Updates call statistics
            - Updates state machine
        """
        # Check if we're in a registration context (prevent re-entrance)
        current_registration = _in_registration_context.get()
        if current_registration == fingerprint:
            # We're registering this model, don't try to use it
            return 'graph'
        
        async with self._state_locks[fingerprint]:
            state = self._states.get(fingerprint, State.UNKNOWN)
            stats = self._call_stats[fingerprint]
            
            # Update statistics
            now = time.time()
            if stats.total_calls == 0:
                stats.first_call_time = now
            stats.last_call_time = now
            stats.total_calls += 1
            
            # Report metrics (observability integration)
            try:
                from .observability import record_registration_state
                record_registration_state(fingerprint, state.name)
            except Exception as e:
                logger.debug(f"Failed to record registration state metric: {e}")
            if self._metrics_callback:
                try:
                    await self._metrics_callback('call', fingerprint, state, stats)
                except Exception as e:
                    logger.warning(f"Metrics callback failed: {e}")
            
            # State-based path selection
            if state == State.REGISTERED:
                return 'cache'
            
            # Initialize state if unknown
            if state == State.UNKNOWN:
                self._states[fingerprint] = State.GRAPH_ONLY
                logger.debug(f"Fingerprint {fingerprint[:8]}: UNKNOWN → GRAPH_ONLY")
                state = State.GRAPH_ONLY
            
            # Check if should trigger registration
            if state == State.GRAPH_ONLY:
                should_register = self._should_trigger_registration(
                    fingerprint, stats
                )
                
                if should_register:
                    # Transition to REGISTERING
                    self._states[fingerprint] = State.REGISTERING
                    logger.info(
                        f"Fingerprint {fingerprint[:8]}: GRAPH_ONLY → REGISTERING "
                        f"(calls={stats.total_calls}, rate={stats.call_rate_per_second():.2f}/s)"
                    )
                    
                    # Start background registration
                    task = asyncio.create_task(
                        self._register_model_safe(fingerprint)
                    )
                    self._registration_tasks[fingerprint] = task
            
            # Always return graph for GRAPH_ONLY and REGISTERING
            return 'graph'
    
    def _should_trigger_registration(
        self, 
        fingerprint: str, 
        stats: CallStatistics
    ) -> bool:
        """
        Decide if registration should be triggered.
        
        Criteria:
        1. Call count threshold met
        2. Call rate sufficient (not too cold)
        3. Not idle for too long
        4. System not under high load (optional)
        """
        # Already attempted too many times?
        if stats.registration_attempts >= self.config.max_registration_attempts:
            return False
        
        # Volume threshold
        if stats.total_calls < self.config.call_count_threshold:
            return False
        
        # Frequency threshold
        call_rate = stats.call_rate_per_second()
        # If we have very few calls or calls happened instantly, rate might be 0
        # In that case, if we've met the count threshold, we should proceed
        if call_rate > 0 and call_rate < self.config.min_call_rate:
            logger.debug(
                f"Fingerprint {fingerprint[:8]}: Call rate too low "
                f"({call_rate:.4f}/s < {self.config.min_call_rate}/s), "
                f"deferring registration"
            )
            return False
        
        # Idle threshold
        time_since_last = time.time() - stats.last_call_time
        if time_since_last > self.config.max_idle_time:
            logger.debug(
                f"Fingerprint {fingerprint[:8]}: Idle for {time_since_last:.1f}s, "
                f"deferring registration"
            )
            return False
        
        # Load threshold (optional)
        if self.config.defer_registration_if_high_load:
            gpu_util = self._get_gpu_utilization()
            if gpu_util > self.config.high_load_threshold:
                logger.debug(
                    f"System under high load (GPU util={gpu_util:.1%}), "
                    f"deferring registration for {fingerprint[:8]}"
                )
                return False
        
        return True
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization (0-1)."""
        try:
            # This is a placeholder - need actual implementation
            # Could use nvidia-ml-py3 for real utilization
            # For now, return moderate load
            return 0.5  # Default: assume moderate load
        except Exception:
            return 0.5
    
    async def _register_model_safe(self, fingerprint: str):
        """
        Background registration with re-entrancy protection and retry logic.
        
        This method:
        1. Prevents deadlocks via separate registration lock
        2. Disables Djinn interception during registration
        3. Handles failures with exponential backoff
        4. Updates state machine atomically
        """
        stats = self._call_stats[fingerprint]
        should_retry = False
        backoff = None
        
        async with self._registration_locks[fingerprint]:
            # Mark this context as "in registration"
            token = _in_registration_context.set(fingerprint)
            
            try:
                # Increment attempt counter
                stats.registration_attempts += 1
                attempt = stats.registration_attempts - 1
                
                logger.info(
                    f"Starting registration for {fingerprint[:8]} "
                    f"(attempt {attempt + 1}/{self.config.max_registration_attempts})"
                )
                
                # Disable Djinn interception during registration
                # (prevents recursive calls to state machine)
                with self._disable_djinn_interception():
                    if self._registration_callback:
                        await self._registration_callback(fingerprint)
                    else:
                        raise RuntimeError("No registration callback set")
                
                # Success! Transition to REGISTERED
                async with self._state_locks[fingerprint]:
                    self._states[fingerprint] = State.REGISTERED
                    logger.info(
                        f"Registration succeeded for {fingerprint[:8]}, "
                        f"REGISTERING → REGISTERED"
                    )
                
                # Report success metrics (observability integration)
                try:
                    from .observability import record_registration_state
                    record_registration_state(fingerprint, State.REGISTERED.name)
                except Exception as obs_e:
                    logger.debug(f"Failed to record registration success metric: {obs_e}")
                
                # Report success metrics (legacy callback)
                if self._metrics_callback:
                    try:
                        await self._metrics_callback(
                            'registration_success', 
                            fingerprint, 
                            State.REGISTERED,
                            stats
                        )
                    except Exception as e:
                        logger.warning(f"Metrics callback failed: {e}")
                
            except Exception as e:
                # Registration failed
                stats.registration_failures += 1
                logger.error(
                    f"Registration failed for {fingerprint[:8]} "
                    f"(attempt {stats.registration_attempts}): {e}"
                )
                
                # Report failure metrics (observability integration)
                try:
                    from .observability import record_registration_failure, record_registration_state
                    record_registration_failure(fingerprint, str(e))
                    record_registration_state(fingerprint, State.GRAPH_ONLY.name)
                except Exception as obs_e:
                    logger.debug(f"Failed to record registration failure metric: {obs_e}")
                
                # Report failure metrics (legacy callback)
                if self._metrics_callback:
                    try:
                        await self._metrics_callback(
                            'registration_failure',
                            fingerprint,
                            State.REGISTERING,
                            stats
                        )
                    except Exception as e:
                        logger.warning(f"Metrics callback failed: {e}")
                
                # Check if we should retry (while still holding lock for atomic check)
                should_retry = stats.registration_attempts < self.config.max_registration_attempts
                backoff = None
                if should_retry:
                    backoff = self.config.retry_backoff_seconds[
                        min(attempt, len(self.config.retry_backoff_seconds) - 1)
                    ]
                    logger.info(
                        f"Will retry registration for {fingerprint[:8]} in {backoff}s"
                    )
                    
                    # Transition back to GRAPH_ONLY temporarily
                    async with self._state_locks[fingerprint]:
                        self._states[fingerprint] = State.GRAPH_ONLY
                else:
                    # Exhausted retries
                    async with self._state_locks[fingerprint]:
                        if self.config.permanent_fallback_on_failure:
                            # Permanent fallback to graph execution
                            self._states[fingerprint] = State.GRAPH_ONLY
                            logger.error(
                                f"Registration exhausted for {fingerprint[:8]}, "
                                f"permanently falling back to graph execution"
                            )
                        else:
                            # Could raise error or transition to special FAILED state
                            # For now, just stay in GRAPH_ONLY
                            self._states[fingerprint] = State.GRAPH_ONLY
            
            finally:
                # Clear registration context
                _in_registration_context.reset(token)
                
                # Clean up task reference
                if fingerprint in self._registration_tasks:
                    del self._registration_tasks[fingerprint]
        
        # Schedule retry OUTSIDE the lock to avoid deadlock
        if should_retry and backoff is not None:
            await asyncio.sleep(backoff)
            # Recursive retry (will acquire lock again)
            await self._register_model_safe(fingerprint)
    
    def _disable_djinn_interception(self):
        """
        Context manager to disable Djinn interception.
        
        During registration, we may need to run the model locally
        to extract metadata or validate. We don't want these calls
        to be intercepted and sent to Djinn recursively.
        """
        # This will be implemented in the frontend interception layer
        # Placeholder for now
        class DisableInterception:
            def __enter__(self):
                # Set flag to disable interception
                return self
            
            def __exit__(self, *args):
                # Re-enable interception
                pass
        
        return DisableInterception()
    
    def get_state(self, fingerprint: str) -> State:
        """Get current state for fingerprint (for monitoring)."""
        return self._states.get(fingerprint, State.UNKNOWN)
    
    def get_statistics(self, fingerprint: str) -> CallStatistics:
        """Get call statistics for fingerprint (for monitoring)."""
        return self._call_stats[fingerprint]
    
    async def force_registration(self, fingerprint: str):
        """
        Force immediate registration (for explicit user registration).
        
        This bypasses all heuristics and immediately transitions to REGISTERING.
        """
        # Check current state (without holding lock during wait)
        async with self._state_locks[fingerprint]:
            current_state = self._states.get(fingerprint, State.UNKNOWN)
            
            if current_state == State.REGISTERED:
                logger.info(f"Fingerprint {fingerprint[:8]} already registered")
                return
            
            if current_state == State.REGISTERING:
                logger.info(f"Fingerprint {fingerprint[:8]} registration in progress")
                # Get task reference
                task = self._registration_tasks.get(fingerprint)
                if task:
                    # Release lock before waiting
                    pass
                else:
                    return
            else:
                # Transition to REGISTERING
                self._states[fingerprint] = State.REGISTERING
                logger.info(f"Forcing registration for {fingerprint[:8]}")
                
                # Start registration immediately
                task = asyncio.create_task(
                    self._register_model_safe(fingerprint)
                )
                self._registration_tasks[fingerprint] = task
        
        # Wait for completion (outside lock to avoid deadlock)
        if task:
            try:
                await task
            except Exception as e:
                logger.error(f"Force registration failed for {fingerprint[:8]}: {e}")
                raise

