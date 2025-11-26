"""
Prometheus Metrics for Semantic Memory Management.

Phase 3 Enhancement: Production Monitoring

Tracks all memory management operations:
- GPU cache hits/misses and evictions
- KV cache session lifecycle
- Lifetime-based eviction decisions
- Phase-aware memory budget allocations
- Recomputation vs storage decisions
- Memory pressure events and recovery
"""

import logging
from typing import Dict, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Provide dummy implementations
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, amount=1, labels=None):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, value, labels=None):
            pass
        def inc(self, amount=1, labels=None):
            pass
        def dec(self, amount=1, labels=None):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, value, labels=None):
            pass
    
    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, value, labels=None):
            pass

logger = logging.getLogger(__name__)


# ============================================================================
# GPU CACHE METRICS
# ============================================================================

cache_hits = Counter(
    'genie_cache_hits_total',
    'Total number of GPU cache hits',
    ['model_id']
)

cache_misses = Counter(
    'genie_cache_misses_total',
    'Total number of GPU cache misses',
    ['model_id']
)

cache_evictions = Counter(
    'genie_cache_evictions_total',
    'Total number of GPU cache evictions',
    ['eviction_reason']  # 'lru_count', 'memory_pressure'
)

cache_memory_bytes = Gauge(
    'genie_cache_memory_bytes',
    'Current GPU cache memory usage in bytes',
    ['model_id']
)

cache_models_loaded = Gauge(
    'genie_cache_models_loaded',
    'Current number of models loaded in cache'
)

cache_memory_freed_bytes = Summary(
    'genie_cache_memory_freed_bytes',
    'Memory freed per eviction event in bytes'
)

cache_eviction_latency_ms = Histogram(
    'genie_cache_eviction_latency_ms',
    'Time to complete cache eviction in milliseconds',
    buckets=(1, 5, 10, 50, 100, 500, 1000, 5000)
)

cache_hit_rate = Gauge(
    'genie_cache_hit_rate_percent',
    'Current cache hit rate percentage'
)


# ============================================================================
# KV SESSION METRICS
# ============================================================================

kv_sessions_created = Counter(
    'genie_kv_sessions_created_total',
    'Total number of KV sessions created'
)

kv_sessions_closed = Counter(
    'genie_kv_sessions_closed_total',
    'Total number of KV sessions closed'
)

kv_sessions_active = Gauge(
    'genie_kv_sessions_active',
    'Current number of active KV sessions'
)

kv_cache_pinned_bytes = Gauge(
    'genie_kv_cache_pinned_bytes',
    'Total KV cache pinned to GPUs in bytes'
)

kv_session_updates = Counter(
    'genie_kv_session_updates_total',
    'Total number of KV cache updates in sessions'
)

kv_cleanup_evictions = Counter(
    'genie_kv_cleanup_evictions_total',
    'Total number of idle KV sessions evicted'
)

kv_session_lifetime_seconds = Summary(
    'genie_kv_session_lifetime_seconds',
    'Lifetime of KV sessions in seconds'
)


# ============================================================================
# LIFETIME-BASED EVICTION METRICS (PHASE 2)
# ============================================================================

lifetime_analyses = Counter(
    'genie_lifetime_analyses_total',
    'Total number of graph lifetime analyses performed'
)

lifetime_early_evictions = Counter(
    'genie_lifetime_early_evictions_total',
    'Total number of proactive (early) evictions based on lifetime'
)

lifetime_memory_saved_bytes = Summary(
    'genie_lifetime_memory_saved_bytes',
    'Memory saved per early eviction in bytes'
)

lifetime_false_retentions = Counter(
    'genie_lifetime_false_retentions_total',
    'Tensors retained past their last consumer (should be 0)'
)


# ============================================================================
# PHASE-AWARE MEMORY METRICS (PHASE 2)
# ============================================================================

phase_switches = Counter(
    'genie_phase_switches_total',
    'Total number of execution phase transitions',
    ['from_phase', 'to_phase']
)

phase_budget_violations = Counter(
    'genie_phase_budget_violations_total',
    'Allocations that exceeded phase budget',
    ['phase', 'category']
)

phase_memory_allocated_bytes = Gauge(
    'genie_phase_memory_allocated_bytes',
    'Memory allocated per category in current phase',
    ['phase', 'category']
)

phase_memory_utilization_percent = Gauge(
    'genie_phase_memory_utilization_percent',
    'Memory utilization percentage per category in phase',
    ['phase', 'category']
)


# ============================================================================
# RECOMPUTATION VS STORAGE METRICS (PHASE 2)
# ============================================================================

recompute_decisions = Counter(
    'genie_recompute_decisions_total',
    'Total number of "recompute instead of cache" decisions'
)

storage_decisions = Counter(
    'genie_storage_decisions_total',
    'Total number of "cache instead of recompute" decisions'
)

decision_latency_us = Histogram(
    'genie_decision_latency_us',
    'Latency of recomputation vs storage decisions in microseconds',
    ['decision_type'],  # 'cache', 'recompute'
    buckets=(1, 10, 100, 1000, 10000, 100000)
)


# ============================================================================
# MEMORY PRESSURE METRICS (PHASE 3)
# ============================================================================

memory_pressure_events = Counter(
    'genie_memory_pressure_events_total',
    'Total number of high memory pressure events detected',
    ['severity']  # 'warning', 'critical'
)

memory_reclaimed_bytes = Summary(
    'genie_memory_reclaimed_bytes',
    'Memory reclaimed per pressure recovery event'
)

pressure_recovery_latency_ms = Histogram(
    'genie_pressure_recovery_latency_ms',
    'Time to recover from memory pressure in milliseconds',
    buckets=(10, 50, 100, 500, 1000, 5000, 10000)
)

current_memory_utilization_percent = Gauge(
    'genie_memory_utilization_percent',
    'Current GPU memory utilization percentage'
)

memory_available_bytes = Gauge(
    'genie_memory_available_bytes',
    'Available GPU memory in bytes'
)


# ============================================================================
# ADAPTIVE BUDGET METRICS (PHASE 3)
# ============================================================================

adaptive_budget_updates = Counter(
    'genie_adaptive_budget_updates_total',
    'Total number of budget adjustments made',
    ['phase', 'category']
)

budget_efficiency_score = Gauge(
    'genie_budget_efficiency_score',
    'Efficiency score of current budget allocation (0-100)',
    ['phase']
)

learned_optimal_budgets = Gauge(
    'genie_learned_optimal_budgets_percent',
    'Learned optimal budget allocation percentage',
    ['phase', 'category']
)


# ============================================================================
# VMU METRICS (PHASE 4)
# ============================================================================

vmu_text_used_bytes = Gauge(
    'genie_vmu_text_used_bytes',
    'VMU Text segment used bytes'
)

vmu_text_capacity_bytes = Gauge(
    'genie_vmu_text_capacity_bytes',
    'VMU Text segment capacity bytes'
)

vmu_data_reserved_bytes = Gauge(
    'genie_vmu_data_reserved_bytes',
    'VMU Data segment reserved bytes'
)

vmu_data_capacity_bytes = Gauge(
    'genie_vmu_data_capacity_bytes',
    'VMU Data segment capacity bytes'
)

vmu_data_internal_waste_bytes = Gauge(
    'genie_vmu_data_internal_waste_bytes',
    'VMU Data segment internal waste bytes'
)

vmu_data_external_gap_bytes = Gauge(
    'genie_vmu_data_external_gap_bytes',
    'VMU Data segment external gap bytes'
)

vmu_stack_allocated_bytes = Gauge(
    'genie_vmu_stack_allocated_bytes',
    'VMU Stack segment allocated bytes'
)

vmu_stack_capacity_bytes = Gauge(
    'genie_vmu_stack_capacity_bytes',
    'VMU Stack segment capacity bytes'
)

vmu_stack_reset_count = Gauge(
    'genie_vmu_stack_reset_count',
    'VMU Stack segment reset count'
)

vmu_active_sessions = Gauge(
    'genie_vmu_active_sessions',
    'VMU active sessions count'
)

vmu_models_loaded = Gauge(
    'genie_vmu_models_loaded',
    'VMU models loaded count'
)


# ============================================================================
# METRICS COLLECTION HELPER
# ============================================================================

class MetricsCollector:
    """Centralized metrics collection for memory management."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.prometheus_enabled = PROMETHEUS_AVAILABLE
        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "Prometheus not available. Install with: pip install prometheus-client"
            )
        logger.info("MetricsCollector initialized (prometheus=%s)", self.prometheus_enabled)
    
    # GPU Cache Metrics
    def record_cache_hit(self, model_id: str) -> None:
        """Record a cache hit."""
        if self.prometheus_enabled:
            cache_hits.labels(model_id=model_id).inc()
    
    def record_cache_miss(self, model_id: str) -> None:
        """Record a cache miss."""
        if self.prometheus_enabled:
            cache_misses.labels(model_id=model_id).inc()
    
    def record_cache_eviction(self, reason: str, freed_bytes: int, latency_ms: float) -> None:
        """Record a cache eviction event."""
        if self.prometheus_enabled:
            cache_evictions.labels(eviction_reason=reason).inc()
            cache_memory_freed_bytes.observe(freed_bytes)
            cache_eviction_latency_ms.observe(latency_ms)
    
    def set_cache_memory(self, model_id: str, bytes_used: int) -> None:
        """Set current cache memory for a model."""
        if self.prometheus_enabled:
            cache_memory_bytes.labels(model_id=model_id).set(bytes_used)
    
    def set_models_loaded(self, count: int) -> None:
        """Set current number of models loaded."""
        if self.prometheus_enabled:
            cache_models_loaded.set(count)
    
    def set_cache_hit_rate(self, percent: float) -> None:
        """Set current cache hit rate."""
        if self.prometheus_enabled:
            cache_hit_rate.set(percent)
    
    # KV Session Metrics
    def record_session_created(self) -> None:
        """Record KV session creation."""
        if self.prometheus_enabled:
            kv_sessions_created.inc()
    
    def record_session_closed(self, lifetime_seconds: float) -> None:
        """Record KV session closure."""
        if self.prometheus_enabled:
            kv_sessions_closed.inc()
            kv_session_lifetime_seconds.observe(lifetime_seconds)
    
    def set_active_sessions(self, count: int) -> None:
        """Set current number of active sessions."""
        if self.prometheus_enabled:
            kv_sessions_active.set(count)
    
    def set_pinned_kv_bytes(self, bytes_pinned: int) -> None:
        """Set total KV cache pinned."""
        if self.prometheus_enabled:
            kv_cache_pinned_bytes.set(bytes_pinned)
    
    def record_session_update(self) -> None:
        """Record KV cache update in session."""
        if self.prometheus_enabled:
            kv_session_updates.inc()
    
    def record_cleanup_eviction(self) -> None:
        """Record idle session eviction."""
        if self.prometheus_enabled:
            kv_cleanup_evictions.inc()
    
    # Lifetime-Based Eviction Metrics
    def record_lifetime_analysis(self) -> None:
        """Record graph lifetime analysis."""
        if self.prometheus_enabled:
            lifetime_analyses.inc()
    
    def record_early_eviction(self, memory_saved_bytes: int) -> None:
        """Record proactive eviction based on lifetime."""
        if self.prometheus_enabled:
            lifetime_early_evictions.inc()
            lifetime_memory_saved_bytes.observe(memory_saved_bytes)
    
    def record_false_retention(self) -> None:
        """Record tensor retained past its lifetime (error condition)."""
        if self.prometheus_enabled:
            lifetime_false_retentions.inc()
    
    # Phase-Aware Memory Metrics
    def record_phase_switch(self, from_phase: str, to_phase: str) -> None:
        """Record execution phase transition."""
        if self.prometheus_enabled:
            phase_switches.labels(from_phase=from_phase, to_phase=to_phase).inc()
    
    def record_budget_violation(self, phase: str, category: str) -> None:
        """Record budget violation."""
        if self.prometheus_enabled:
            phase_budget_violations.labels(phase=phase, category=category).inc()
    
    def set_phase_memory_allocated(self, phase: str, category: str, bytes_allocated: int) -> None:
        """Set memory allocated for phase category."""
        if self.prometheus_enabled:
            phase_memory_allocated_bytes.labels(phase=phase, category=category).set(bytes_allocated)
    
    def set_phase_utilization(self, phase: str, category: str, percent: float) -> None:
        """Set memory utilization for phase category."""
        if self.prometheus_enabled:
            phase_memory_utilization_percent.labels(phase=phase, category=category).set(percent)
    
    # Recomputation Metrics
    def record_recompute_decision(self, latency_us: float) -> None:
        """Record 'recompute' decision."""
        if self.prometheus_enabled:
            recompute_decisions.inc()
            decision_latency_us.labels(decision_type='recompute').observe(latency_us)
    
    def record_storage_decision(self, latency_us: float) -> None:
        """Record 'cache' decision."""
        if self.prometheus_enabled:
            storage_decisions.inc()
            decision_latency_us.labels(decision_type='cache').observe(latency_us)
    
    # Memory Pressure Metrics
    def record_memory_pressure_event(self, severity: str) -> None:
        """Record memory pressure event."""
        if self.prometheus_enabled:
            memory_pressure_events.labels(severity=severity).inc()
    
    def record_pressure_recovery(self, memory_reclaimed_bytes: int, latency_ms: float) -> None:
        """Record recovery from memory pressure."""
        if self.prometheus_enabled:
            memory_reclaimed_bytes.observe(memory_reclaimed_bytes)
            pressure_recovery_latency_ms.observe(latency_ms)
    
    def set_memory_utilization(self, percent: float, available_bytes: int) -> None:
        """Set current memory utilization."""
        if self.prometheus_enabled:
            current_memory_utilization_percent.set(percent)
            memory_available_bytes.set(available_bytes)
    
    # Adaptive Budget Metrics
    def record_budget_update(self, phase: str, category: str) -> None:
        """Record adaptive budget adjustment."""
        if self.prometheus_enabled:
            adaptive_budget_updates.labels(phase=phase, category=category).inc()
    
    def set_budget_efficiency(self, phase: str, score: float) -> None:
        """Set efficiency score (0-100)."""
        if self.prometheus_enabled:
            budget_efficiency_score.labels(phase=phase).set(min(100, max(0, score)))
    
    def set_learned_budget(self, phase: str, category: str, percent: float) -> None:
        """Set learned optimal budget percentage."""
        if self.prometheus_enabled:
            learned_optimal_budgets.labels(phase=phase, category=category).set(percent)

    # VMU Metrics
    def set_vmu_text_metrics(self, used_bytes: int, capacity_bytes: int) -> None:
        """Set VMU Text segment metrics."""
        if self.prometheus_enabled:
            vmu_text_used_bytes.set(used_bytes)
            vmu_text_capacity_bytes.set(capacity_bytes)

    def set_vmu_data_metrics(self, reserved_bytes: int, capacity_bytes: int,
                           internal_waste_bytes: int, external_gap_bytes: int) -> None:
        """Set VMU Data segment metrics."""
        if self.prometheus_enabled:
            vmu_data_reserved_bytes.set(reserved_bytes)
            vmu_data_capacity_bytes.set(capacity_bytes)
            vmu_data_internal_waste_bytes.set(internal_waste_bytes)
            vmu_data_external_gap_bytes.set(external_gap_bytes)

    def set_vmu_stack_metrics(self, allocated_bytes: int, capacity_bytes: int, reset_count: int) -> None:
        """Set VMU Stack segment metrics."""
        if self.prometheus_enabled:
            vmu_stack_allocated_bytes.set(allocated_bytes)
            vmu_stack_capacity_bytes.set(capacity_bytes)
            vmu_stack_reset_count.set(reset_count)

    def set_vmu_session_metrics(self, active_sessions: int, models_loaded: int) -> None:
        """Set VMU session and model metrics."""
        if self.prometheus_enabled:
            vmu_active_sessions.set(active_sessions)
            vmu_models_loaded.set(models_loaded)


# Global metrics collector instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def export_metrics_to_json(filepath: str) -> None:
    """
    Export current metrics to a JSON file for evaluation scripts.

    This is useful for evaluation scripts that need to capture VMU and memory
    metrics without setting up a full Prometheus scraping infrastructure.

    Args:
        filepath: Path to write the JSON metrics file
    """
    import json
    from datetime import datetime

    metrics = get_metrics()
    if not metrics.prometheus_enabled:
        logger.warning("Prometheus not available, cannot export metrics")
        return

    try:
        # Import prometheus_client to get current metric values
        from prometheus_client import CollectorRegistry, generate_latest
        from prometheus_client.core import REGISTRY

        # Generate the latest metrics in Prometheus format
        metrics_data = generate_latest(REGISTRY).decode('utf-8')

        # Parse the Prometheus format into a simple dict
        # This is a basic parser - for production, consider using prometheus_client.parser
        parsed_metrics = {}
        current_metric = None
        current_help = None

        for line in metrics_data.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                if line.startswith('# HELP '):
                    parts = line.split(' ', 2)
                    if len(parts) >= 2:
                        current_metric = parts[1]
                        current_help = parts[2] if len(parts) > 2 else ""
                continue

            if current_metric and line:
                # Simple parsing - assumes no labels for now
                parts = line.split(' ')
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        parsed_metrics[current_metric] = {
                            'value': value,
                            'help': current_help or ""
                        }
                    except (ValueError, IndexError):
                        continue

        # Create export structure
        export_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': parsed_metrics
        }

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported metrics to {filepath}")

    except Exception as exc:
        logger.error(f"Failed to export metrics to {filepath}: {exc}")
        # Don't raise - evaluation scripts shouldn't fail if metrics export fails
