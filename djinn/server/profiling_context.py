"""
Profiling Context Manager for Djinn Components

Provides thread-safe, low-overhead profiling hooks that can be used
throughout Djinn's codebase to measure actual execution phases.

Usage:
    from djinn.server.profiling_context import get_profiler, record_phase
    
    # In any component:
    with record_phase('serialization'):
        result = serialize_tensor(tensor)
    
    # Or manually:
    profiler = get_profiler()
    if profiler:
        start = time.perf_counter()
        # ... do work ...
        duration = (time.perf_counter() - start) * 1000
        profiler.record_phase('network_c2s', duration)
"""

import time
import threading
import logging
from typing import Optional, Dict, List
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PhaseTiming:
    """Timing for a single phase."""
    phase_name: str
    duration_ms: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)


class ProfilingContext:
    """
    Thread-safe profiling context for collecting phase timings.
    
    This is a lightweight wrapper that can be used throughout Djinn
    to collect actual timing measurements.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.phases: List[PhaseTiming] = []
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
    
    def start(self):
        """Start profiling session."""
        if self.enabled:
            self._start_time = time.perf_counter()
            with self._lock:
                self.phases.clear()
    
    def record_phase(self, phase_name: str, duration_ms: float, metadata: Optional[Dict] = None):
        """Record a phase timing."""
        if not self.enabled:
            return
        
        timing = PhaseTiming(
            phase_name=phase_name,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.phases.append(timing)
    
    def get_phases(self) -> List[PhaseTiming]:
        """Get all recorded phases."""
        with self._lock:
            return self.phases.copy()
    
    def get_total_time(self) -> float:
        """Get total profiling time."""
        if not self.phases:
            return 0.0
        return sum(p.duration_ms for p in self.phases)
    
    def get_phase_dict(self) -> Dict[str, float]:
        """Get phases as a dictionary."""
        result = {}
        with self._lock:
            for phase in self.phases:
                result[phase.phase_name] = phase.duration_ms
        return result


# Global profiling context (thread-safe)
_global_profiler: Optional[ProfilingContext] = None
_profiler_lock = threading.Lock()


def set_profiler(profiler: Optional[ProfilingContext]):
    """Set the global profiler instance."""
    global _global_profiler
    with _profiler_lock:
        _global_profiler = profiler


def get_profiler() -> Optional[ProfilingContext]:
    """Get the global profiler instance."""
    with _profiler_lock:
        return _global_profiler


@contextmanager
def record_phase(phase_name: str, metadata: Optional[Dict] = None):
    """
    Context manager to record a phase timing.
    
    Usage:
        with record_phase('serialization'):
            result = serialize_tensor(tensor)
    
    Note: If profiling is disabled (profiler is None or enabled=False),
    this context manager has near-zero overhead (just a function call check).
    """
    profiler = get_profiler()
    if not profiler or not profiler.enabled:
        yield
        return
    
    # Only measure time if profiling is enabled
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        profiler.record_phase(phase_name, duration_ms, metadata)

