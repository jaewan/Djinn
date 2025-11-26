from __future__ import annotations

from typing import List, Dict, Any
from uuid import uuid4

from djinn.core.graph_interface import Graph as DjinnGraph
from djinn.core.computation_graph_adapter import ComputationGraphAdapter
from .workload import ExecutionPlan, PlanFragment, WorkloadProfile


class Planner:
    """Minimal execution planner skeleton.

    Phase 1 planner emits a single-fragment plan for the whole graph with
    a local CPU placement. Downstream runtime can consume this and execute
    via existing fallback paths.
    """

    def __init__(self) -> None:
        pass

    def generate_plan(self, graph: DjinnGraph, profile: WorkloadProfile) -> ExecutionPlan:
        fragment = self._build_whole_graph_fragment(graph)
        placement: Dict[str, Any] = {fragment.fragment_id: self._choose_placement(profile)}
        transfers: List[Dict[str, Any]] = []
        feature_flags: Dict[str, bool] = {
            "overlap_io": False,
            "micro_batching": False,
        }
        return ExecutionPlan(
            plan_id=str(uuid4()),
            fragments=[fragment],
            placement=placement,
            transfers=transfers,
            feature_flags=feature_flags,
        )

    def _build_whole_graph_fragment(self, graph: DjinnGraph) -> PlanFragment:
        fragment_id = f"frag_{uuid4().hex[:8]}"
        # ✅ PHASE 3: Use unified Graph interface
        # Entry points are root nodes (no inputs)
        inputs = [node.id for node in graph.get_roots()]
        # Outputs are leaf nodes (no outputs)
        outputs = [node.id for node in graph.get_leaves()]
        # For backward compatibility, store graph reference
        # (PlanFragment may expect ComputationGraph, adapter handles this)
        subgraph = graph
        return PlanFragment(
            fragment_id=fragment_id,
            subgraph=graph,
            inputs=inputs,
            outputs=outputs,
        )

    def _choose_placement(self, profile: WorkloadProfile) -> Dict[str, Any]:
        # Phase 1: always local CPU placement; future phases can map by workload
        return {
            "device": "cpu:0",
            "node": "local",
            "workload": str(profile.workload_type.value),
        }


