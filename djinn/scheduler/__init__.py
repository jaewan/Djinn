"""
Scheduler: Semantic-driven execution planning.

This module implements the scheduler stage of Djinn's 3-stage pipeline:
1. Frontend (capture intent) → 2. Scheduler (optimize placement) → 3. Backend (execute)

Key components:
- Scheduler: Main orchestrator for execution planning
- CostEstimator: Performance and resource cost modeling
- Placement: Device assignment strategies
- ExecutionSchedule: Structured execution plan
"""

# Import from scheduler components
from .core.scheduling import (
    Scheduler,
    ExecutionSchedule,
    SchedulingStrategy,
    SchedulingGroup,
)
from .core.cost_estimator import (
    CostEstimator,
    GraphCostEstimator,
    NetworkTopology,
    CostEstimate,
)
from .strategies.placement import PlacementEngine, PlacementPlan, PlacementDecision

__all__ = [
    'Scheduler',
    'ExecutionSchedule',
    'SchedulingStrategy',
    'SchedulingGroup',
    'CostEstimator',
    'GraphCostEstimator',
    'NetworkTopology',
    'CostEstimate',
    'PlacementEngine',
    'PlacementPlan',
    'PlacementDecision',
]
