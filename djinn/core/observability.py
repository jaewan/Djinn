"""
djinn/core/observability.py

Basic metrics collection and tracing for Phase 2.

Provides:
- Prometheus metrics (if available)
- Request tracing for debugging
- Graceful degradation if prometheus_client not installed
"""

import time
import logging
import contextvars
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

# Context variable for request tracing
_current_trace = contextvars.ContextVar('current_trace', default=None)

# Try to import Prometheus, but don't fail if not available
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not available. Metrics will be disabled. "
        "Install with: pip install prometheus-client"
    )
    
    # Create dummy classes for graceful degradation
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, value=1):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, value):
            pass
        def inc(self, value=1):
            pass
        def dec(self, value=1):
            pass

# Metrics (will be no-ops if Prometheus not available)
if PROMETHEUS_AVAILABLE:
    model_calls_total = Counter(
        'djinn_model_calls_total',
        'Total model execution calls',
        ['fingerprint', 'tenant_id', 'path']
    )
    
    model_latency_seconds = Histogram(
        'djinn_model_latency_seconds',
        'Model execution latency',
        ['fingerprint', 'tenant_id', 'path'],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    )
    
    registration_status = Gauge(
        'djinn_model_registration_status',
        'Model registration state (0=UNKNOWN, 1=GRAPH_ONLY, 2=REGISTERING, 3=REGISTERED)',
        ['fingerprint', 'state']
    )
    
    registration_failures_total = Counter(
        'djinn_registration_failures_total',
        'Total registration failures',
        ['fingerprint', 'reason']
    )
    
    vram_used_bytes = Gauge(
        'djinn_vram_used_bytes',
        'VRAM usage in bytes',
        ['tenant_id', 'segment']
    )
    
    active_sessions = Gauge(
        'djinn_active_sessions',
        'Number of active sessions',
        ['tenant_id']
    )
else:
    # Dummy metrics for graceful degradation
    model_calls_total = Counter()
    model_latency_seconds = Histogram()
    registration_status = Gauge()
    registration_failures_total = Counter()
    vram_used_bytes = Gauge()
    active_sessions = Gauge()


class RequestTrace:
    """Per-request tracing for debugging."""
    
    def __init__(self, fingerprint: str, tenant_id: str):
        import uuid
        self.trace_id = str(uuid.uuid4())[:8]
        self.fingerprint = fingerprint
        self.tenant_id = tenant_id
        self.events = []
        self.start_time = time.time()
    
    def log_event(self, event_name: str, **metadata):
        """Log an event in the trace."""
        self.events.append({
            'timestamp': time.time() - self.start_time,
            'event': event_name,
            'metadata': metadata,
        })
        logger.debug(
            f"Trace {self.trace_id}: {event_name} "
            f"(fingerprint={self.fingerprint[:8]}, tenant={self.tenant_id})"
        )
    
    def __enter__(self):
        # Set as current trace
        token = _current_trace.set(self)
        self._token = token
        self.log_event('trace_started')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is not None:
            self.log_event('trace_error', error_type=str(exc_type), error=str(exc_val))
        else:
            self.log_event('trace_completed', duration_ms=duration_ms)
        
        # Reset context
        _current_trace.reset(self._token)
        
        # Log complete trace if there are events
        if len(self.events) > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Trace {self.trace_id} completed: {len(self.events)} events, {duration_ms:.1f}ms")
    
    def dump(self) -> str:
        """Dump trace for debugging."""
        return json.dumps({
            'trace_id': self.trace_id,
            'fingerprint': self.fingerprint,
            'tenant_id': self.tenant_id,
            'duration_ms': (time.time() - self.start_time) * 1000,
            'events': self.events,
        }, indent=2)


def get_current_trace() -> Optional[RequestTrace]:
    """Get current request trace (if any)."""
    return _current_trace.get()


def record_model_call(fingerprint: str, tenant_id: str, path: str, latency_seconds: float):
    """Record a model execution call."""
    model_calls_total.labels(
        fingerprint=fingerprint[:16],  # Truncate for Prometheus label limits
        tenant_id=tenant_id,
        path=path
    ).inc()
    
    model_latency_seconds.labels(
        fingerprint=fingerprint[:16],
        tenant_id=tenant_id,
        path=path
    ).observe(latency_seconds)


def record_registration_state(fingerprint: str, state: str):
    """Record model registration state."""
    # Map state names to numeric values for Gauge
    state_map = {
        'UNKNOWN': 0,
        'GRAPH_ONLY': 1,
        'REGISTERING': 2,
        'REGISTERED': 3,
    }
    state_value = state_map.get(state, 0)
    
    registration_status.labels(
        fingerprint=fingerprint[:16],
        state=state
    ).set(state_value)


def record_registration_failure(fingerprint: str, reason: str):
    """Record a registration failure."""
    registration_failures_total.labels(
        fingerprint=fingerprint[:16],
        reason=reason[:50]  # Truncate reason for label limits
    ).inc()


def record_vram_usage(tenant_id: str, segment: str, bytes_used: int):
    """Record VRAM usage."""
    vram_used_bytes.labels(
        tenant_id=tenant_id,
        segment=segment
    ).set(bytes_used)


def record_active_sessions(tenant_id: str, count: int):
    """Record number of active sessions."""
    active_sessions.labels(tenant_id=tenant_id).set(count)

