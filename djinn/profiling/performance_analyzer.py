"""
Performance analyzer for Djinn profiling data.

Provides high-level analysis, visualization, and optimization recommendations
based on collected profiling measurements.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes profiling data to identify performance bottlenecks and optimization opportunities.
    """

    def __init__(self, profiler):
        self.profiler = profiler
        self.baselines = {}

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks with severity levels."""
        if not self.profiler.component_timings:
            return {'error': 'No measurements available'}

        bottlenecks = {
            'critical': [],  # >50% of total time
            'high': [],      # 20-50% of total time
            'medium': [],    # 10-20% of total time
            'low': []        # <10% of total time
        }

        total_time = sum(sum(times) for times in self.profiler.component_timings.values())
        
        if total_time == 0:
            return {'error': 'No timing data collected'}

        for component, timings in self.profiler.component_timings.items():
            component_total = sum(timings)
            percentage = (component_total / total_time) * 100
            
            bottleneck_info = {
                'component': component,
                'percentage': percentage,
                'time_ms': component_total,
                'count': len(timings),
            }

            if percentage > 50:
                bottlenecks['critical'].append(bottleneck_info)
            elif percentage > 20:
                bottlenecks['high'].append(bottleneck_info)
            elif percentage > 10:
                bottlenecks['medium'].append(bottleneck_info)
            else:
                bottlenecks['low'].append(bottleneck_info)

        return bottlenecks

    def generate_optimization_plan(self) -> Dict[str, Any]:
        """Generate prioritized optimization plan based on profiling data."""
        bottlenecks = self.identify_bottlenecks()
        
        if 'error' in bottlenecks:
            return bottlenecks

        optimization_plan = {
            'priority_order': [],
            'estimated_impact': {},
            'implementation_effort': {},
        }

        # Prioritize based on impact
        all_bottlenecks = (
            bottlenecks.get('critical', []) +
            bottlenecks.get('high', []) +
            bottlenecks.get('medium', []) +
            bottlenecks.get('low', [])
        )

        # Sort by percentage impact (descending)
        all_bottlenecks.sort(key=lambda x: x['percentage'], reverse=True)

        for i, bottleneck in enumerate(all_bottlenecks):
            optimization_plan['priority_order'].append({
                'rank': i + 1,
                'component': bottleneck['component'],
                'percentage': bottleneck['percentage'],
                'time_ms': bottleneck['time_ms'],
            })

        return optimization_plan

    def generate_performance_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for performance dashboard visualization."""
        if not self.profiler.component_timings:
            return {'error': 'No measurements available'}

        bottlenecks = self.identify_bottlenecks()
        
        if 'error' in bottlenecks:
            return bottlenecks

        total_time = sum(sum(times) for times in self.profiler.component_timings.values())

        # Prepare data for visualization
        dashboard_data = {
            'summary': {
                'total_components': len(self.profiler.component_timings),
                'total_time_ms': total_time,
                'avg_component_time_ms': total_time / len(self.profiler.component_timings) if self.profiler.component_timings else 0,
            },

            'component_breakdown': [
                {
                    'component': component,
                    'time_ms': sum(times),
                    'count': len(times),
                    'avg_ms': np.mean(times),
                    'severity': 'critical' if (sum(times) / total_time * 100) > 50 else
                               'high' if (sum(times) / total_time * 100) > 20 else
                               'medium' if (sum(times) / total_time * 100) > 10 else 'low'
                }
                for component, times in self.profiler.component_timings.items()
            ],

            'optimization_opportunities': [
                {
                    'rank': i + 1,
                    'component': b['component'],
                    'time_ms': b['time_ms'],
                    'percentage': b['percentage'],
                }
                for i, b in enumerate(sorted(
                    (b for blist in bottlenecks.values() if isinstance(blist, list) for b in blist),
                    key=lambda x: x['percentage'],
                    reverse=True
                )[:5])
            ]
        }

        return dashboard_data

    def export_to_json(self, filename: str):
        """Export analysis results to JSON file."""
        try:
            analysis = {
                'bottleneck_analysis': self.identify_bottlenecks(),
                'optimization_plan': self.generate_optimization_plan(),
                'dashboard_data': self.generate_performance_dashboard_data(),
                'timestamp': time.time(),
                'measurements_count': sum(
                    len(times) for times in self.profiler.component_timings.values()
                )
            }

            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)

            logger.info(f"Performance analysis exported to {filename}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to export analysis: {e}")
            return None
