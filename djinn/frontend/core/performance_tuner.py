"""
Phase 7B: Performance Tuning

Automatic performance optimization for Djinn:
- Profile operation execution
- Detect bottlenecks
- Auto-tune thresholds
- Recommend optimizations

Learns from profiling data to continuously improve performance.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from collections import defaultdict
import threading


@dataclass
class OperationProfile:
    """Profile data for a single operation."""
    operation_name: str
    total_time_ms: float = 0.0
    execution_count: int = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    remote_executions: int = 0
    local_executions: int = 0
    
    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.execution_count if self.execution_count > 0 else 0
    
    @property
    def remote_ratio(self) -> float:
        total = self.remote_executions + self.local_executions
        return self.remote_executions / total if total > 0 else 0
    
    def update(self, time_ms: float, is_remote: bool = False):
        """Update profile with new execution data."""
        self.total_time_ms += time_ms
        self.execution_count += 1
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        
        if is_remote:
            self.remote_executions += 1
        else:
            self.local_executions += 1


class PerformanceTuner:
    """
    Automatic performance tuning based on profiling.
    
    Monitors actual execution times and adjusts:
    - Remote execution thresholds
    - Cache size limits
    - Operation classification
    """
    
    def __init__(self):
        self.profiles: Dict[str, OperationProfile] = defaultdict(
            lambda: OperationProfile(operation_name="unknown")
        )
        self.lock = threading.Lock()
        self.profiling_enabled = True
        
        # Tuning parameters (can be auto-adjusted)
        self.remote_execution_threshold_bytes = 100_000
        self.cache_size_limit = 1000
        self.batch_size_threshold = 32
        
        # Statistics
        self.total_profile_points = 0
        self.optimization_improvements = 0
    
    def record_operation(self, operation_name: str, time_ms: float,
                        is_remote: bool = False):
        """Record operation execution profile."""
        if not self.profiling_enabled:
            return
        
        with self.lock:
            profile = self.profiles[operation_name]
            profile.operation_name = operation_name
            profile.update(time_ms, is_remote)
            self.total_profile_points += 1
    
    def get_bottlenecks(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Identify slowest operations.
        
        Returns list of (operation_name, total_time_ms) sorted by time.
        """
        with self.lock:
            bottlenecks = [
                (op_name, profile.total_time_ms)
                for op_name, profile in self.profiles.items()
                if profile.execution_count > 0
            ]
        
        return sorted(bottlenecks, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify operations that could be optimized.
        
        Returns suggestions for:
        - Remote execution candidates
        - Local execution candidates
        - Fusible operation pairs
        """
        opportunities = []
        
        with self.lock:
            for op_name, profile in self.profiles.items():
                if profile.execution_count < 5:
                    continue  # Need enough data
                
                avg_time = profile.avg_time_ms
                
                # Opportunity 1: Slow operations executing locally
                if avg_time > 10.0 and profile.remote_ratio < 0.1:
                    opportunities.append({
                        'type': 'remote_candidate',
                        'operation': op_name,
                        'avg_time_ms': avg_time,
                        'reason': f'Slow operation ({avg_time:.1f}ms) mostly local'
                    })
                
                # Opportunity 2: Overhead-heavy remote operations
                if avg_time > 5.0 and profile.remote_ratio > 0.9:
                    opportunities.append({
                        'type': 'local_candidate',
                        'operation': op_name,
                        'avg_time_ms': avg_time,
                        'reason': f'High overhead operation ({avg_time:.1f}ms) always remote'
                    })
                
                # Opportunity 3: Highly variable execution time
                if profile.max_time_ms > profile.min_time_ms * 5:
                    opportunities.append({
                        'type': 'inconsistent',
                        'operation': op_name,
                        'avg_time_ms': avg_time,
                        'min_ms': profile.min_time_ms,
                        'max_ms': profile.max_time_ms,
                        'reason': f'Highly variable ({profile.min_time_ms:.1f}-{profile.max_time_ms:.1f}ms)'
                    })
        
        return sorted(opportunities, key=lambda x: x.get('avg_time_ms', 0), reverse=True)
    
    def recommend_threshold_adjustment(self, operation_name: str) -> Optional[int]:
        """
        Recommend remote execution threshold for an operation.
        
        Based on: execution time, data transfer cost, computation cost.
        """
        with self.lock:
            profile = self.profiles.get(operation_name)
            if not profile or profile.execution_count < 10:
                return None
            
            avg_time = profile.avg_time_ms
            
            # If operation is consistently slow locally, increase threshold
            if avg_time > 20.0 and profile.remote_ratio < 0.5:
                # Recommend more remote execution
                return int(self.remote_execution_threshold_bytes * 0.5)
            
            # If operation is overhead-heavy remotely, decrease threshold
            if avg_time > 5.0 and profile.remote_ratio > 0.8:
                # Recommend less remote execution
                return int(self.remote_execution_threshold_bytes * 2.0)
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance tuning summary."""
        with self.lock:
            total_time = sum(p.total_time_ms for p in self.profiles.values())
            
            fastest_ops = sorted(
                [(n, p.avg_time_ms) for n, p in self.profiles.items()],
                key=lambda x: x[1]
            )[:5]
            
            slowest_ops = sorted(
                [(n, p.avg_time_ms) for n, p in self.profiles.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                'total_profile_points': self.total_profile_points,
                'total_execution_time_ms': total_time,
                'unique_operations': len(self.profiles),
                'fastest_operations': dict(fastest_ops),
                'slowest_operations': dict(slowest_ops),
                'optimization_improvements': self.optimization_improvements,
                'current_thresholds': {
                    'remote_execution_bytes': self.remote_execution_threshold_bytes,
                    'cache_size': self.cache_size_limit,
                    'batch_size': self.batch_size_threshold,
                }
            }
    
    def print_summary(self):
        """Print formatted performance tuning summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("PHASE 7B: PERFORMANCE TUNING SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Profiling Data:")
        print(f"   Total measurements: {summary['total_profile_points']}")
        print(f"   Unique operations: {summary['unique_operations']}")
        print(f"   Total execution time: {summary['total_execution_time_ms']:.1f}ms")
        
        print(f"\n⚡ Fastest Operations:")
        for op, time_ms in list(summary['fastest_operations'].items())[:3]:
            print(f"   {op}: {time_ms:.2f}ms")
        
        print(f"\n🐌 Slowest Operations:")
        for op, time_ms in list(summary['slowest_operations'].items())[:3]:
            print(f"   {op}: {time_ms:.2f}ms")
        
        print(f"\n🔧 Current Thresholds:")
        thresholds = summary['current_thresholds']
        print(f"   Remote execution: {thresholds['remote_execution_bytes']} bytes")
        print(f"   Cache size: {thresholds['cache_size']}")
        print(f"   Batch size: {thresholds['batch_size']}")
        
        opportunities = self.get_optimization_opportunities()
        if opportunities:
            print(f"\n💡 Optimization Opportunities ({len(opportunities)} found):")
            for opp in opportunities[:5]:
                print(f"   - {opp['type']}: {opp['operation']}")
                print(f"     {opp['reason']}")


# Global performance tuner instance
_performance_tuner = PerformanceTuner()


def get_performance_tuner() -> PerformanceTuner:
    """Get global performance tuner."""
    return _performance_tuner


def record_operation_profile(operation_name: str, time_ms: float, is_remote: bool = False):
    """Record operation execution profile."""
    _performance_tuner.record_operation(operation_name, time_ms, is_remote)

