"""
Performance Monitor for optimizations (Tensor Registry & SRG Fusion).

Tracks:
- Registry cache hit rates and bytes saved
- Fusion overhead and pattern identification
- End-to-end latencies
- Memory usage
- Error rates
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class LatencyBucket:
    """Track latency percentiles."""
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    count: int = 0


@dataclass
class RegistryMetrics:
    """Metrics for SmartTensorRegistry."""
    hit_rate_percent: float = 0.0
    bytes_saved_total: int = 0
    bytes_saved_per_request_avg: float = 0.0
    avg_lookup_ms: float = 0.0
    cache_misses: int = 0
    cache_hits: int = 0
    evictions: int = 0
    models_cached: int = 0
    version_conflicts: int = 0


@dataclass
class FusionMetrics:
    """Metrics for SRGFusionCompiler."""
    attention_blocks_identified: int = 0
    conv_blocks_identified: int = 0
    no_fusion_blocks: int = 0
    avg_grouping_ms: float = 0.0
    total_ops_grouped: int = 0
    savings_percent: float = 0.0


@dataclass
class EndToEndMetrics:
    """End-to-end execution metrics."""
    requests_total: int = 0
    requests_with_cache_hit: int = 0
    requests_with_fusion: int = 0
    latency_baseline: LatencyBucket = field(default_factory=LatencyBucket)
    latency_with_optimizations: LatencyBucket = field(default_factory=LatencyBucket)
    errors: int = 0


class PerformanceMonitor:
    """
    Tracks optimization performance across:
    - Registry cache hits/misses
    - Fusion pattern identification
    - Overhead measurement
    - End-to-end latencies
    """

    def __init__(self):
        self.registry_metrics = RegistryMetrics()
        self.fusion_metrics = FusionMetrics()
        self.end_to_end_metrics = EndToEndMetrics()
        
        # For latency tracking
        self._latency_samples_baseline: list = []
        self._latency_samples_optimized: list = []
        
        self.enabled = True
        self.start_time = datetime.now()

    def record_registry_hit(self, bytes_saved: int, lookup_ms: float):
        """Record a cache hit."""
        if not self.enabled:
            return
        
        self.registry_metrics.cache_hits += 1
        self.registry_metrics.bytes_saved_total += bytes_saved
        self.registry_metrics.avg_lookup_ms = (
            (self.registry_metrics.avg_lookup_ms * 
             (self.registry_metrics.cache_hits - 1) + lookup_ms) /
            self.registry_metrics.cache_hits
        )
        self.end_to_end_metrics.requests_with_cache_hit += 1

    def record_registry_miss(self, lookup_ms: float):
        """Record a cache miss."""
        if not self.enabled:
            return
        
        self.registry_metrics.cache_misses += 1
        self.registry_metrics.avg_lookup_ms = (
            (self.registry_metrics.avg_lookup_ms * 
             (self.registry_metrics.cache_misses - 1) + lookup_ms) /
            self.registry_metrics.cache_misses
        )

    def record_registry_eviction(self, models_cached: int):
        """Record an LRU eviction."""
        if not self.enabled:
            return
        
        self.registry_metrics.evictions += 1
        self.registry_metrics.models_cached = models_cached

    def record_registry_version_conflict(self):
        """Record a version conflict (cache invalidation)."""
        if not self.enabled:
            return
        
        self.registry_metrics.version_conflicts += 1

    def record_registry_state(self, models_cached: int, total_cached_models: int):
        """Update registry state metrics."""
        if not self.enabled:
            return
        
        self.registry_metrics.models_cached = models_cached

    def record_fusion_grouping(
        self,
        attention_blocks: int,
        conv_blocks: int,
        no_fusion_blocks: int,
        total_ops: int,
        grouping_ms: float
    ):
        """Record fusion grouping results."""
        if not self.enabled:
            return
        
        self.fusion_metrics.attention_blocks_identified += attention_blocks
        self.fusion_metrics.conv_blocks_identified += conv_blocks
        self.fusion_metrics.no_fusion_blocks += no_fusion_blocks
        self.fusion_metrics.total_ops_grouped += total_ops
        
        # Update average grouping time
        total_previous = (
            self.fusion_metrics.attention_blocks_identified +
            self.fusion_metrics.conv_blocks_identified +
            self.fusion_metrics.no_fusion_blocks - attention_blocks - conv_blocks - no_fusion_blocks
        )
        if total_previous > 0:
            self.fusion_metrics.avg_grouping_ms = (
                (self.fusion_metrics.avg_grouping_ms * total_previous + grouping_ms) /
                (total_previous + 1)
            )
        else:
            self.fusion_metrics.avg_grouping_ms = grouping_ms
        
        self.end_to_end_metrics.requests_with_fusion += 1

    def record_end_to_end_request(self, latency_ms: float, optimizations_enabled: bool = False):
        """Record end-to-end request latency."""
        if not self.enabled:
            return
        
        self.end_to_end_metrics.requests_total += 1
        
        if optimizations_enabled:
            self._latency_samples_optimized.append(latency_ms)
        else:
            self._latency_samples_baseline.append(latency_ms)

    def record_error(self, error_type: str):
        """Record an error during execution."""
        if not self.enabled:
            return
        
        self.end_to_end_metrics.errors += 1
        logger.warning(f"Optimization error recorded: {error_type}")

    def _calculate_percentiles(self, samples: list) -> LatencyBucket:
        """Calculate latency percentiles from samples."""
        if not samples:
            return LatencyBucket()
        
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        def percentile(p: float) -> float:
            idx = max(0, int(n * p / 100) - 1)
            return sorted_samples[idx]
        
        return LatencyBucket(
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            mean_ms=sum(samples) / n,
            min_ms=min(samples),
            max_ms=max(samples),
            count=n
        )

    def get_registry_metrics(self) -> Dict:
        """Get registry metrics snapshot."""
        total_requests = self.registry_metrics.cache_hits + self.registry_metrics.cache_misses
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (self.registry_metrics.cache_hits / total_requests) * 100
        
        bytes_per_request = 0.0
        if self.registry_metrics.cache_hits > 0:
            bytes_per_request = self.registry_metrics.bytes_saved_total / self.registry_metrics.cache_hits
        
        return {
            'hit_rate_percent': round(hit_rate, 2),
            'total_hits': self.registry_metrics.cache_hits,
            'total_misses': self.registry_metrics.cache_misses,
            'bytes_saved_total': self.registry_metrics.bytes_saved_total,
            'bytes_saved_per_hit_avg': round(bytes_per_request, 2),
            'avg_lookup_ms': round(self.registry_metrics.avg_lookup_ms, 4),
            'evictions': self.registry_metrics.evictions,
            'models_cached': self.registry_metrics.models_cached,
            'version_conflicts': self.registry_metrics.version_conflicts,
        }

    def get_fusion_metrics(self) -> Dict:
        """Get fusion compiler metrics snapshot."""
        total_blocks = (
            self.fusion_metrics.attention_blocks_identified +
            self.fusion_metrics.conv_blocks_identified +
            self.fusion_metrics.no_fusion_blocks
        )
        
        return {
            'attention_blocks': self.fusion_metrics.attention_blocks_identified,
            'conv_blocks': self.fusion_metrics.conv_blocks_identified,
            'no_fusion_blocks': self.fusion_metrics.no_fusion_blocks,
            'total_blocks': total_blocks,
            'total_ops_grouped': self.fusion_metrics.total_ops_grouped,
            'avg_grouping_ms': round(self.fusion_metrics.avg_grouping_ms, 4),
        }

    def get_end_to_end_metrics(self) -> Dict:
        """Get end-to-end metrics snapshot."""
        baseline_latency = self._calculate_percentiles(self._latency_samples_baseline)
        optimized_latency = self._calculate_percentiles(self._latency_samples_optimized)
        
        speedup_p50 = 1.0
        if baseline_latency.p50_ms > 0:
            speedup_p50 = baseline_latency.p50_ms / optimized_latency.p50_ms
        
        return {
            'total_requests': self.end_to_end_metrics.requests_total,
            'requests_with_cache_hit': self.end_to_end_metrics.requests_with_cache_hit,
            'requests_with_fusion': self.end_to_end_metrics.requests_with_fusion,
            'errors': self.end_to_end_metrics.errors,
            'baseline_latency': {
                'p50_ms': round(baseline_latency.p50_ms, 2),
                'p95_ms': round(baseline_latency.p95_ms, 2),
                'p99_ms': round(baseline_latency.p99_ms, 2),
                'mean_ms': round(baseline_latency.mean_ms, 2),
                'min_ms': round(baseline_latency.min_ms, 2),
                'max_ms': round(baseline_latency.max_ms, 2),
                'samples': baseline_latency.count,
            },
            'optimized_latency': {
                'p50_ms': round(optimized_latency.p50_ms, 2),
                'p95_ms': round(optimized_latency.p95_ms, 2),
                'p99_ms': round(optimized_latency.p99_ms, 2),
                'mean_ms': round(optimized_latency.mean_ms, 2),
                'min_ms': round(optimized_latency.min_ms, 2),
                'max_ms': round(optimized_latency.max_ms, 2),
                'samples': optimized_latency.count,
            },
            'speedup_p50': round(speedup_p50, 2),
        }

    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': round(uptime, 2),
            'enabled': self.enabled,
            'registry': self.get_registry_metrics(),
            'fusion': self.get_fusion_metrics(),
            'end_to_end': self.get_end_to_end_metrics(),
        }

    def to_json(self) -> str:
        """Serialize metrics to JSON."""
        return json.dumps(self.get_summary(), indent=2)

    def reset(self):
        """Reset all metrics."""
        self.registry_metrics = RegistryMetrics()
        self.fusion_metrics = FusionMetrics()
        self.end_to_end_metrics = EndToEndMetrics()
        self._latency_samples_baseline = []
        self._latency_samples_optimized = []
        self.start_time = datetime.now()
        logger.info("Performance monitor metrics reset")

    def disable(self):
        """Disable monitoring (for performance-sensitive operations)."""
        self.enabled = False

    def enable(self):
        """Re-enable monitoring."""
        self.enabled = True
