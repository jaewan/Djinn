"""
Production Metrics for Djinn System

Comprehensive metrics collection for monitoring:
- Performance metrics (latency, throughput)
- Resource metrics (memory, network)
- Reliability metrics (errors, retries)
- Optimization metrics (cache hits, reduction factors)

Integrates with Prometheus for monitoring dashboards.
"""

import time
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    count: int = 0
    total_time_ms: float = 0.0
    errors: int = 0
    
    # Phase classification
    materialization_triggers: int = 0
    reduction_operations: int = 0
    compute_operations: int = 0
    
    # Network metrics (if applicable)
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.count if self.count > 0 else 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total * 100 if total > 0 else 0
    
    @property
    def error_rate(self) -> float:
        return self.errors / self.count * 100 if self.count > 0 else 0


class MetricsCollector:
    """
    Centralized metrics collection for Djinn.
    
    Thread-safe collection of performance, resource, and reliability metrics.
    """
    
    def __init__(self):
        self.operations: Dict[str, OperationMetrics] = defaultdict(
            lambda: OperationMetrics(operation_name="unknown")
        )
        self.lock = threading.Lock()
        
        # Performance metrics
        self.total_execution_time_ms = 0.0
        self.total_materializations = 0
        self.total_operations = 0
        
        # Resource metrics
        self.total_memory_mb = 0.0
        self.peak_memory_mb = 0.0
        self.network_bytes_sent = 0
        self.network_bytes_received = 0
        
        # Reliability metrics
        self.total_errors = 0
        self.total_retries = 0
        
        # Optimization metrics
        self.reduction_factor_total = 0.0
        self.reduction_operations_count = 0
        
        # Timing
        self.start_time = time.time()
    
    def record_operation(self, operation_name: str, time_ms: float, 
                        error: bool = False, phase: str = None):
        """Record an operation execution."""
        with self.lock:
            metrics = self.operations[operation_name]
            metrics.operation_name = operation_name
            metrics.count += 1
            metrics.total_time_ms += time_ms
            
            if error:
                metrics.errors += 1
                self.total_errors += 1
            
            if phase == 'materialization_trigger':
                metrics.materialization_triggers += 1
            elif phase == 'reduction':
                metrics.reduction_operations += 1
                self.reduction_operations_count += 1
            elif phase == 'compute':
                metrics.compute_operations += 1
            
            self.total_operations += 1
            self.total_execution_time_ms += time_ms
    
    def record_network_transfer(self, bytes_sent: int = 0, bytes_received: int = 0):
        """Record network transfer metrics."""
        with self.lock:
            self.network_bytes_sent += bytes_sent
            self.network_bytes_received += bytes_received
    
    def record_cache_hit(self, operation_name: str):
        """Record a cache hit."""
        with self.lock:
            self.operations[operation_name].cache_hits += 1
    
    def record_cache_miss(self, operation_name: str):
        """Record a cache miss."""
        with self.lock:
            self.operations[operation_name].cache_misses += 1
    
    def record_memory(self, memory_mb: float):
        """Record current memory usage."""
        with self.lock:
            self.total_memory_mb = memory_mb
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
    
    def record_reduction_factor(self, factor: float):
        """Record network transfer reduction factor."""
        with self.lock:
            self.reduction_factor_total += factor
    
    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        with self.lock:
            uptime_s = time.time() - self.start_time
            
            # Calculate averages
            avg_operation_time = self.total_execution_time_ms / self.total_operations \
                if self.total_operations > 0 else 0
            
            avg_reduction_factor = self.reduction_factor_total / self.reduction_operations_count \
                if self.reduction_operations_count > 0 else 0
            
            return {
                'uptime_seconds': uptime_s,
                'total_operations': self.total_operations,
                'average_operation_time_ms': avg_operation_time,
                'throughput_ops_per_sec': self.total_operations / uptime_s if uptime_s > 0 else 0,
                'total_execution_time_ms': self.total_execution_time_ms,
                'total_errors': self.total_errors,
                'error_rate_pct': self.total_errors / self.total_operations * 100 \
                    if self.total_operations > 0 else 0,
                'network_bytes_sent': self.network_bytes_sent,
                'network_bytes_received': self.network_bytes_received,
                'network_bytes_total': self.network_bytes_sent + self.network_bytes_received,
                'peak_memory_mb': self.peak_memory_mb,
                'reduction_operations': self.reduction_operations_count,
                'average_reduction_factor': avg_reduction_factor,
                'operation_breakdown': {
                    op_name: {
                        'count': metrics.count,
                        'avg_time_ms': metrics.avg_time_ms,
                        'cache_hit_rate_pct': metrics.cache_hit_rate,
                        'error_rate_pct': metrics.error_rate,
                    }
                    for op_name, metrics in self.operations.items()
                }
            }
    
    def print_summary(self):
        """Print formatted metrics summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("PHASE 5: PRODUCTION METRICS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Uptime: {summary['uptime_seconds']:.1f}s")
        print(f"   Total operations: {summary['total_operations']}")
        print(f"   Average latency: {summary['average_operation_time_ms']:.2f}ms")
        print(f"   Throughput: {summary['throughput_ops_per_sec']:.2f} ops/sec")
        
        print(f"\n💾 Resource Metrics:")
        print(f"   Network sent: {summary['network_bytes_sent'] / 1024 / 1024:.1f} MB")
        print(f"   Network received: {summary['network_bytes_received'] / 1024 / 1024:.1f} MB")
        print(f"   Peak memory: {summary['peak_memory_mb']:.1f} MB")
        
        print(f"\n🔧 Reliability Metrics:")
        print(f"   Total errors: {summary['total_errors']}")
        print(f"   Error rate: {summary['error_rate_pct']:.2f}%")
        
        print(f"\n⚡ Optimization Metrics:")
        print(f"   Reduction operations: {summary['reduction_operations']}")
        print(f"   Average reduction factor: {summary['average_reduction_factor']:.0f}x")
        
        if summary['operation_breakdown']:
            print(f"\n📈 Per-Operation Breakdown:")
            for op_name, op_metrics in list(summary['operation_breakdown'].items())[:10]:
                print(f"   {op_name}:")
                print(f"     Count: {op_metrics['count']}")
                print(f"     Avg time: {op_metrics['avg_time_ms']:.2f}ms")
                print(f"     Cache hit rate: {op_metrics['cache_hit_rate_pct']:.1f}%")


# Global metrics instance
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics


def record_operation(operation_name: str, time_ms: float, 
                    error: bool = False, phase: str = None):
    """Record an operation for metrics."""
    _metrics.record_operation(operation_name, time_ms, error, phase)


def record_network_transfer(bytes_sent: int = 0, bytes_received: int = 0):
    """Record network transfer for metrics."""
    _metrics.record_network_transfer(bytes_sent, bytes_received)

