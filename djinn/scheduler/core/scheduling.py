"""Scheduling strategies for semantic-aware execution.

This module implements scheduling strategies for different workload types
using the unified graph interface.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ...core.types import ExecutionPhase
from .cost_estimator import NetworkTopology
from ...core.graph_interface import Graph

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Types of scheduling strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    DYNAMIC = "dynamic"
    PRIORITY = "priority"


@dataclass
class SchedulingGroup:
    """Group of nodes to be scheduled together."""
    group_id: str
    nodes: List[str]
    strategy: SchedulingStrategy
    priority: int = 0
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExecutionSchedule:
    """Execution schedule for a computation graph."""
    stages: List[List[SchedulingGroup]] = field(default_factory=list)
    node_to_stage: Dict[str, int] = field(default_factory=dict)
    node_to_group: Dict[str, str] = field(default_factory=dict)
    total_stages: int = 0
    strategy: SchedulingStrategy = SchedulingStrategy.SEQUENTIAL
    metadata: Dict = field(default_factory=dict)


class Scheduler:
    """Scheduler for semantic-aware execution planning."""

    def __init__(self, network_topology=None, cost_estimator=None):
        self._scheduling_cache = {}
        self._stats = defaultdict(int)
        self.network_topology = network_topology or NetworkTopology()
        self.cost_estimator = cost_estimator

        # Device information (integrated with network topology)
        self.devices = {}  # device_id -> Device info
        self._network_manager = None
    
    def create_schedule(self, graph,
                        optimization_plan: Optional[Dict] = None) -> ExecutionSchedule:
        """Create execution schedule for a graph.

        Args:
            graph: Graph to schedule using unified graph interface
            optimization_plan: Optional optimization plan with placement hints

        Returns:
            ExecutionSchedule with stages and groups
        """
        if self.cost_estimator is None:
            # Fallback to basic scheduling if no cost estimator
            return self._create_basic_schedule(graph, optimization_plan)

        # Use cost-aware scheduling
        return self._create_cost_aware_schedule(graph, optimization_plan)

    def _create_basic_schedule(self, graph,
                              optimization_plan: Optional[Dict] = None) -> ExecutionSchedule:
        """Fallback basic scheduling when no cost estimator available."""
        # Analyze graph dependencies
        dependencies = self._analyze_dependencies(graph)

        # Identify scheduling groups based on metadata
        groups = self._identify_scheduling_groups(graph, optimization_plan)

        # Create execution stages
        stages = self._create_execution_stages(groups, dependencies)

        # Build schedule
        schedule = ExecutionSchedule(
            stages=stages,
            total_stages=len(stages)
        )

        # Fill node mappings
        for stage_idx, stage_groups in enumerate(stages):
            for group in stage_groups:
                for node_name in group.nodes:
                    schedule.node_to_stage[node_name] = stage_idx
                    schedule.node_to_group[node_name] = group.group_id

        # Determine overall strategy
        schedule.strategy = self._determine_strategy(groups)

        # Add metadata
        schedule.metadata['total_nodes'] = len(list(graph.nodes()))
        schedule.metadata['total_groups'] = len(groups)

        logger.info(f"Created basic schedule with {schedule.total_stages} stages and {len(groups)} groups")

        return schedule

    def _create_cost_aware_schedule(self, graph,
                                   optimization_plan: Optional[Dict] = None) -> ExecutionSchedule:
        """Cost-aware scheduling using semantic metadata and cost estimates."""

        # Step 1: Estimate costs for all nodes
        costs = self.cost_estimator.estimate_graph(graph)
        logger.info(f"Estimated costs: {costs['total_compute_flops']:.2e} FLOPs, "
                   f"{costs['total_transfer_time_ms']:.2f}ms transfer, "
                   f"{costs['total_queueing_delay_ms']:.2f}ms queueing")

        # Step 2: Integrate with optimizer if available
        if optimization_plan and hasattr(optimization_plan, 'get_optimization_plan'):
            # Convert optimizer plan to our format
            optimizer_plan = optimization_plan.get_optimization_plan()
            if optimizer_plan:
                logger.info("Using optimization plan from semantic optimizer")
                return self._schedule_with_optimizer_plan(graph, optimizer_plan, costs)

        # Fallback to cost-based scheduling
        return self._schedule_with_cost_optimization(graph, optimization_plan, costs)

    def _schedule_with_optimizer_plan(self, graph,
                                    optimizer_plan: 'OptimizationPlan',
                                    costs: Dict) -> ExecutionSchedule:
        """Schedule using optimization plan from semantic optimizer."""

        # Use optimizer's placement decisions
        placement_plan = optimizer_plan.node_placement

        # Use optimizer's co-location groups
        colocation_groups = optimizer_plan.colocation_groups

        # Create scheduling groups based on optimizer plan
        groups = []
        processed_nodes = set()

        # Process co-location groups first (highest priority)
        for group_id, node_names in colocation_groups.items():
            if node_names:
                # Determine strategy based on group type
                strategy = SchedulingStrategy.SEQUENTIAL
                if 'parallel' in group_id.lower():
                    strategy = SchedulingStrategy.PARALLEL
                elif 'pipeline' in group_id.lower():
                    strategy = SchedulingStrategy.PIPELINE

                group = SchedulingGroup(
                    group_id=group_id,
                    nodes=node_names,
                    strategy=strategy,
                    priority=10,  # High priority for optimizer groups
                    metadata={'optimizer_group': True, 'device': placement_plan.get(node_names[0])}
                )
                groups.append(group)
                processed_nodes.update(node_names)

        # Process pipeline stages
        for stage_idx, stage_nodes in enumerate(optimizer_plan.pipeline_stages):
            if stage_nodes:
                group = SchedulingGroup(
                    group_id=f"pipeline_stage_{stage_idx}",
                    nodes=stage_nodes,
                    strategy=SchedulingStrategy.PIPELINE,
                    priority=8,
                    metadata={'pipeline_stage': stage_idx}
                )
                groups.append(group)
                processed_nodes.update(stage_nodes)

        # Process parallel branches
        for branch_idx, (branch1, branch2) in enumerate(optimizer_plan.parallel_branches):
            # First branch
            if branch1:
                group1 = SchedulingGroup(
                    group_id=f"parallel_branch_{branch_idx}_a",
                    nodes=branch1,
                    strategy=SchedulingStrategy.PARALLEL,
                    priority=7
                )
                groups.append(group1)
                processed_nodes.update(branch1)

            # Second branch
            if branch2:
                group2 = SchedulingGroup(
                    group_id=f"parallel_branch_{branch_idx}_b",
                    nodes=branch2,
                    strategy=SchedulingStrategy.PARALLEL,
                    priority=7
                )
                groups.append(group2)
                processed_nodes.update(branch2)

        # Handle remaining nodes
        for node in graph.graph.nodes:
            if node.name not in processed_nodes:
                group = SchedulingGroup(
                    group_id=f"default_{node.name}",
                    nodes=[node.name],
                    strategy=SchedulingStrategy.SEQUENTIAL,
                    priority=0,
                    metadata={'device': placement_plan.get(node.name)}
                )
                groups.append(group)

        # Create stages using dependencies
        dependencies = self._analyze_dependencies(graph)
        stages = self._create_cost_aware_stages(groups, dependencies, placement_plan)

        # Build schedule
        schedule = ExecutionSchedule(stages=stages, total_stages=len(stages))

        # Fill mappings
        for stage_idx, stage_groups in enumerate(stages):
            for group in stage_groups:
                for node_name in group.nodes:
                    schedule.node_to_stage[node_name] = stage_idx
                    schedule.node_to_group[node_name] = group.group_id

        # Set strategy based on optimizer plan
        if optimizer_plan.pipeline_stages:
            schedule.strategy = SchedulingStrategy.PIPELINE
        elif any('parallel' in g.group_id.lower() for g in groups):
            schedule.strategy = SchedulingStrategy.PARALLEL
        else:
            schedule.strategy = SchedulingStrategy.SEQUENTIAL

        # Add metadata
        schedule.metadata.update({
            'optimizer_plan': True,
            'total_nodes': len(list(graph.graph.nodes)),
            'total_groups': len(groups),
            'total_compute_flops': costs['total_compute_flops'],
            'total_memory_bytes': costs['total_memory_bytes'],
            'total_transfer_time_ms': costs['total_transfer_time_ms'],
            'total_queueing_delay_ms': costs['total_queueing_delay_ms'],
            'estimated_latency_ms': self._estimate_total_latency(costs, placement_plan),
            'placement_plan': placement_plan,
            'optimizations_applied': [opt.value for opt in optimizer_plan.optimizations],
        })

        logger.info(f"Created optimizer-guided schedule: {schedule.total_stages} stages, "
                   f"optimizations: {[opt.value for opt in optimizer_plan.optimizations]}")

        return schedule

    def _schedule_with_cost_optimization(self, graph,
                                       optimization_plan: Optional[Dict],
                                       costs: Dict) -> ExecutionSchedule:
        """Schedule using cost-based optimization."""

        # Step 2: Analyze dependencies and identify co-location requirements
        dependencies = self._analyze_dependencies(graph)
        colocation_groups = self._identify_colocation_groups(graph, optimization_plan)

        # Step 3: Create placement plan using cost-based optimization
        placement_plan = self._create_placement_plan(graph, costs, colocation_groups)

        # Step 4: Create scheduling groups based on placement and optimization
        groups = self._identify_scheduling_groups_with_placement(graph, optimization_plan, placement_plan)

        # Step 5: Create execution stages respecting dependencies and placement
        stages = self._create_cost_aware_stages(groups, dependencies, placement_plan)

        # Step 6: Build schedule with cost metadata
        schedule = ExecutionSchedule(
            stages=stages,
            total_stages=len(stages)
        )

        # Fill node mappings
        for stage_idx, stage_groups in enumerate(stages):
            for group in stage_groups:
                for node_name in group.nodes:
                    schedule.node_to_stage[node_name] = stage_idx
                    schedule.node_to_group[node_name] = group.group_id

        # Determine overall strategy based on costs
        schedule.strategy = self._determine_cost_aware_strategy(groups, costs)

        # Add comprehensive metadata
        # Use unified Graph interface
        total_nodes = len(list(graph.nodes()))

        schedule.metadata.update({
            'total_nodes': total_nodes,
            'total_groups': len(groups),
            'total_compute_flops': costs['total_compute_flops'],
            'total_memory_bytes': costs['total_memory_bytes'],
            'total_transfer_time_ms': costs['total_transfer_time_ms'],
            'total_queueing_delay_ms': costs['total_queueing_delay_ms'],
            'estimated_latency_ms': self._estimate_total_latency(costs, placement_plan),
            'placement_plan': placement_plan,
        })

        logger.info(f"Created cost-aware schedule: {schedule.total_stages} stages, "
                   f"estimated {schedule.metadata['estimated_latency_ms']:.2f}ms latency")

        return schedule
    
    def _analyze_dependencies(self, graph) -> Dict[str, Set[str]]:
        """Analyze node dependencies in the graph.

        Args:
            graph: Graph using unified graph interface

        Returns:
            Dictionary mapping node names to their dependencies
        """
        dependencies = defaultdict(set)

        # Use unified Graph interface
        for node in graph.nodes():
            node_id = node.id
            for inp in node.inputs:
                # Only add dependency if input is a node with an id (LazyTensor or NodeProtocol)
                if hasattr(inp, 'id'):
                    dependencies[node_id].add(inp.id)

        return dependencies
    
    def _identify_scheduling_groups(self, graph,
                                   optimization_plan: Optional[Dict]) -> List[SchedulingGroup]:
        """Identify scheduling groups based on node metadata.

        Args:
            graph: Graph using unified graph interface
            optimization_plan: Optional optimization plan

        Returns:
            List of SchedulingGroups
        """
        groups = []
        processed_nodes = set()
        
        # 1. Groups from optimization plan
        if optimization_plan:
            # Co-location groups
            if hasattr(optimization_plan, 'colocation_groups'):
                for group_id, node_names in optimization_plan.colocation_groups.items():
                    group = SchedulingGroup(
                        group_id=f"coloc_{group_id}",
                        nodes=node_names,
                        strategy=SchedulingStrategy.SEQUENTIAL,
                        priority=10
                    )
                    groups.append(group)
                    processed_nodes.update(node_names)
            
            # Pipeline stages
            if hasattr(optimization_plan, 'pipeline_stages'):
                for stage_idx, stage_nodes in enumerate(optimization_plan.pipeline_stages):
                    group = SchedulingGroup(
                        group_id=f"pipeline_stage_{stage_idx}",
                        nodes=stage_nodes,
                        strategy=SchedulingStrategy.PIPELINE,
                        priority=5
                    )
                    groups.append(group)
                    processed_nodes.update(stage_nodes)
            
            # Parallel branches
            if hasattr(optimization_plan, 'parallel_branches'):
                for branch_idx, (branch1, branch2) in enumerate(optimization_plan.parallel_branches):
                    # First branch
                    group1 = SchedulingGroup(
                        group_id=f"parallel_branch_{branch_idx}_a",
                        nodes=branch1,
                        strategy=SchedulingStrategy.PARALLEL,
                        priority=7
                    )
                    groups.append(group1)
                    processed_nodes.update(branch1)
                    
                    # Second branch
                    group2 = SchedulingGroup(
                        group_id=f"parallel_branch_{branch_idx}_b",
                        nodes=branch2,
                        strategy=SchedulingStrategy.PARALLEL,
                        priority=7
                    )
                    groups.append(group2)
                    processed_nodes.update(branch2)
        
        # 2. Groups from node metadata
        parallel_groups = defaultdict(list)
        fusion_groups = defaultdict(list)

        # Use unified Graph interface
        nodes = graph.nodes()

        for node in nodes:
            # Get node name and metadata for unified graph interface
            node_name = node.id
            node_meta = node.metadata

            if node_name in processed_nodes:
                continue

            # Check for parallel group metadata
            if 'parallel_group' in node_meta:
                group_name = node_meta['parallel_group']
                parallel_groups[group_name].append(node_name)

            # Check for fusion group metadata
            elif 'fusion_group' in node_meta:
                group_name = node_meta['fusion_group']
                fusion_groups[group_name].append(node_name)

            # Check for priority operations
            elif node_meta.get('priority', 0) >= 8:
                # High priority operations get their own group
                group = SchedulingGroup(
                    group_id=f"priority_{node_name}",
                    nodes=[node_name],
                    strategy=SchedulingStrategy.PRIORITY,
                    priority=node_meta['priority']
                )
                groups.append(group)
                processed_nodes.add(node_name)
        
        # Create groups for parallel operations
        for group_name, nodes in parallel_groups.items():
            if nodes:
                group = SchedulingGroup(
                    group_id=f"parallel_{group_name}",
                    nodes=nodes,
                    strategy=SchedulingStrategy.PARALLEL,
                    priority=6
                )
                groups.append(group)
                processed_nodes.update(nodes)
        
        # Create groups for fusion operations
        for group_name, nodes in fusion_groups.items():
            if nodes:
                group = SchedulingGroup(
                    group_id=f"fusion_{group_name}",
                    nodes=nodes,
                    strategy=SchedulingStrategy.SEQUENTIAL,
                    priority=5
                )
                groups.append(group)
                processed_nodes.update(nodes)
        
        # 3. Default groups for remaining nodes
        # Use unified Graph interface
        remaining_nodes = [node for node in graph.nodes()
                         if node.id not in processed_nodes]

        for node in remaining_nodes:
            # Get node name for unified graph interface
            node_name = node.id

            # Create individual group
            group = SchedulingGroup(
                group_id=f"default_{node_name}",
                nodes=[node_name],
                strategy=SchedulingStrategy.SEQUENTIAL,
                priority=0
            )
            groups.append(group)
        
        return groups
    
    def _create_execution_stages(self, groups: List[SchedulingGroup], 
                                dependencies: Dict[str, Set[str]]) -> List[List[SchedulingGroup]]:
        """Create execution stages respecting dependencies.
        
        Args:
            groups: List of scheduling groups
            dependencies: Node dependency map
            
        Returns:
            List of stages, each containing groups that can execute in parallel
        """
        # Build group dependencies
        group_deps = defaultdict(set)
        node_to_group = {}
        
        for group in groups:
            for node in group.nodes:
                node_to_group[node] = group.group_id
        
        for group in groups:
            for node in group.nodes:
                for dep in dependencies.get(node, set()):
                    dep_group = node_to_group.get(dep)
                    if dep_group and dep_group != group.group_id:
                        group_deps[group.group_id].add(dep_group)
        
        # Sort groups by priority
        sorted_groups = sorted(groups, key=lambda g: -g.priority)
        
        # Create stages using topological sort with priority
        stages = []
        scheduled = set()
        remaining = {g.group_id: g for g in sorted_groups}
        
        while remaining:
            # Find groups that can be scheduled
            ready = []
            for group_id, group in remaining.items():
                deps = group_deps[group_id]
                if deps.issubset(scheduled):
                    ready.append(group)
            
            if not ready:
                # Break cycle by scheduling highest priority remaining
                ready = [next(iter(remaining.values()))]
                logger.warning(f"Breaking dependency cycle with group {ready[0].group_id}")
            
            # Group ready nodes by strategy
            stage_groups = []
            parallel_groups = []
            
            for group in ready:
                if group.strategy == SchedulingStrategy.PARALLEL:
                    parallel_groups.append(group)
                else:
                    stage_groups.append(group)
            
            # Add parallel groups to same stage if possible
            if parallel_groups:
                stage_groups.extend(parallel_groups)
            
            if stage_groups:
                stages.append(stage_groups)
                for group in stage_groups:
                    scheduled.add(group.group_id)
                    del remaining[group.group_id]
        
        return stages
    
    def _determine_strategy(self, groups: List[SchedulingGroup]) -> SchedulingStrategy:
        """Determine overall scheduling strategy.
        
        Args:
            groups: List of scheduling groups
            
        Returns:
            Overall SchedulingStrategy
        """
        strategy_counts = defaultdict(int)
        
        for group in groups:
            strategy_counts[group.strategy] += len(group.nodes)
        
        if not strategy_counts:
            return SchedulingStrategy.SEQUENTIAL
        
        # Return most common strategy
        return max(strategy_counts.items(), key=lambda x: x[1])[0]

    def _identify_colocation_groups(self, graph, optimization_plan: Optional[Dict]) -> Dict[str, List[str]]:
        """Identify nodes that should be co-located based on semantic metadata."""
        colocation_groups = {}

        # Use unified Graph interface
        nodes = graph.nodes()

        for node in nodes:
            # Get node name and metadata for unified graph interface
            node_name = node.id
            node_meta = node.metadata
            is_operation = True  # Assume all nodes in unified graph are operations

            if is_operation:
                # Check for KV cache co-location (decode phase)
                if node_meta.get('semantic_role') == 'kv_cache':
                    kv_cache_id = node_meta.get('kv_cache_id', node_name)
                    if kv_cache_id not in colocation_groups:
                        colocation_groups[kv_cache_id] = []
                    colocation_groups[kv_cache_id].append(node_name)

                # Check for attention + KV cache co-location
                if node_meta.get('semantic_role') == 'attention':
                    kv_cache_id = node_meta.get('kv_cache_id')
                    if kv_cache_id and kv_cache_id in colocation_groups:
                        colocation_groups[kv_cache_id].append(node_name)

        return colocation_groups

    def _create_placement_plan(self, graph, costs: Dict, colocation_groups: Dict) -> Dict[str, str]:
        """Create placement plan using cost-based optimization."""
        placement_plan = {}

        # Simple greedy placement: assign to device with lowest estimated cost
        available_devices = list(self.devices.keys()) if self.devices else ['local']

        # Use unified Graph interface
        nodes = graph.nodes()

        for node in nodes:
            # Get node name for unified graph interface
            node_name = node.id
            is_operation = True  # Assume all nodes in unified graph are operations

            if is_operation:
                # Check if node is in a co-location group
                assigned_device = None
                for group_id, group_nodes in colocation_groups.items():
                    if node_name in group_nodes:
                        # Use same device as other nodes in the group
                        for other_node in group_nodes:
                            if other_node in placement_plan:
                                assigned_device = placement_plan[other_node]
                                break
                        break

                if not assigned_device:
                    # Assign to device with lowest cost
                    node_cost = costs['per_node'].get(node_name)
                    if node_cost:
                        # Choose device based on compute vs transfer tradeoff
                        if node_cost.operational_intensity > 10:  # Compute-bound
                            # Prefer local execution for compute-bound ops
                            assigned_device = available_devices[0]
                        else:  # Memory or transfer-bound
                            # Prefer remote execution to avoid local memory pressure
                            assigned_device = available_devices[-1] if len(available_devices) > 1 else available_devices[0]
                    else:
                        assigned_device = available_devices[0]

                placement_plan[node_name] = assigned_device

        return placement_plan

    def _identify_scheduling_groups_with_placement(self, graph,
                                                  optimization_plan: Optional[Dict],
                                                  placement_plan: Dict) -> List[SchedulingGroup]:
        """Create scheduling groups considering placement decisions."""
        groups = []
        processed_nodes = set()

        # Group nodes by device placement
        device_groups = defaultdict(list)
        for node_name, device in placement_plan.items():
            device_groups[device].append(node_name)

        # Create groups for each device
        for device, node_names in device_groups.items():
            if node_names:
                group = SchedulingGroup(
                    group_id=f"device_{device}",
                    nodes=node_names,
                    strategy=SchedulingStrategy.SEQUENTIAL,
                    priority=5,
                    metadata={'device': device}
                )
                groups.append(group)
                processed_nodes.update(node_names)

        # Handle any remaining nodes
        # Use unified Graph interface
        remaining_nodes = [node for node in graph.nodes()
                         if node.id not in processed_nodes]

        for node in remaining_nodes:
            # Get node name for unified graph interface
            node_name = node.id

            group = SchedulingGroup(
                group_id=f"default_{node_name}",
                nodes=[node_name],
                strategy=SchedulingStrategy.SEQUENTIAL,
                priority=0
            )
            groups.append(group)

        return groups

    def _create_cost_aware_stages(self, groups: List[SchedulingGroup],
                                 dependencies: Dict[str, Set[str]],
                                 placement_plan: Dict) -> List[List[SchedulingGroup]]:
        """Create stages considering both dependencies and placement."""
        # Build group dependencies
        group_deps = defaultdict(set)
        node_to_group = {}

        for group in groups:
            for node in group.nodes:
                node_to_group[node] = group.group_id

        for group in groups:
            for node in group.nodes:
                for dep in dependencies.get(node, set()):
                    dep_group = node_to_group.get(dep)
                    if dep_group and dep_group != group.group_id:
                        group_deps[group.group_id].add(dep_group)

        # Sort groups by priority and dependencies
        sorted_groups = sorted(groups, key=lambda g: -g.priority)

        # Create stages using topological sort
        stages = []
        scheduled = set()
        remaining = {g.group_id: g for g in sorted_groups}

        while remaining:
            ready = []
            for group_id, group in remaining.items():
                deps = group_deps[group_id]
                if deps.issubset(scheduled):
                    ready.append(group)

            if not ready:
                # Break cycle by scheduling highest priority remaining
                ready = [next(iter(remaining.values()))]
                logger.warning(f"Breaking dependency cycle with group {ready[0].group_id}")

            if ready:
                stages.append(ready)
                for group in ready:
                    scheduled.add(group.group_id)
                    del remaining[group.group_id]

        return stages

    def _determine_cost_aware_strategy(self, groups: List[SchedulingGroup], costs: Dict) -> SchedulingStrategy:
        """Determine strategy based on cost characteristics."""
        total_transfer = costs.get('total_transfer_time_ms', 0)
        total_compute = costs.get('total_compute_flops', 0)

        if total_transfer > total_compute * 0.001:  # Transfer dominates
            return SchedulingStrategy.PIPELINE
        elif len(groups) > 10:  # Many groups suggest parallelism opportunities
            return SchedulingStrategy.PARALLEL
        else:
            return SchedulingStrategy.SEQUENTIAL

    def _estimate_total_latency(self, costs: Dict, placement_plan: Dict) -> float:
        """Estimate total execution latency."""
        compute_time = costs.get('total_compute_flops', 0) / 1e12  # Rough estimate: 1 TFLOP = 1ms
        transfer_time = costs.get('total_transfer_time_ms', 0)
        queueing_delay = costs.get('total_queueing_delay_ms', 0)

        return compute_time + transfer_time + queueing_delay

    def register_device(self, device_id: str, compute_tflops: float, memory_gb: float, network_gbps: float):
        """Register device information for cost estimation."""
        self.devices[device_id] = {
            'compute_tflops': compute_tflops,
            'memory_gb': memory_gb,
            'network_gbps': network_gbps
        }

        # Update network topology manager
        self.network_topology.register_node(device_id, network_gbps, 1.0)  # 1ms latency default

        # Also update the global network topology manager
        try:
            from ..core.network_topology import get_network_topology
            network_manager = get_network_topology()
            from ..core.network_topology import NetworkDevice
            device = NetworkDevice(
                node_id=device_id,
                bandwidth_gbps=network_gbps,
                latency_ms=1.0,
                device_type='gpu',
                compute_tflops=compute_tflops,
                memory_gb=memory_gb
            )
            network_manager.register_device(device)
        except ImportError:
            pass  # Network topology manager not available