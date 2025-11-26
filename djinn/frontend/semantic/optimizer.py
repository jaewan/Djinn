"""Semantic-aware optimization engine for Djinn.

This module implements the optimization strategies from the HotNets'25 paper,
including LLM decode co-location, CNN pipeline scheduling, and multi-modal
fusion optimizations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.fx as fx
from collections import defaultdict

from ...core.types import ExecutionPhase, MemoryPattern
from .workload import WorkloadType, WorkloadProfile
from ..core.exceptions import Result, OptimizationError

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimizations that can be applied."""
    KV_CACHE_COLOCATION = "kv_cache_colocation"
    PREFILL_PARALLELIZATION = "prefill_parallelization"
    CNN_PIPELINING = "cnn_pipelining"
    MULTIMODAL_FUSION = "multimodal_fusion"
    RECOMPUTATION = "recomputation"
    MEMORY_OPTIMIZATION = "memory_optimization"


@dataclass
class OptimizationPlan:
    """Plan for applying optimizations to a graph."""
    optimizations: List[OptimizationType] = field(default_factory=list)
    node_placement: Dict[str, str] = field(default_factory=dict)  # node_name -> device
    colocation_groups: Dict[str, List[str]] = field(default_factory=dict)  # group_id -> node_names
    pipeline_stages: List[List[str]] = field(default_factory=list)  # stages of node_names
    parallel_branches: List[Tuple[List[str], List[str]]] = field(default_factory=list)  # branch pairs
    recompute_nodes: Set[str] = field(default_factory=set)
    fusion_points: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticOptimizer:
    """Semantic-aware optimizer implementing paper's strategies.
    
    This optimizer analyzes FX graphs with semantic metadata and applies
    workload-specific optimizations as described in Section 3.2 of the paper.
    """
    
    def __init__(self, enable_all: bool = True):
        """Initialize semantic optimizer.
        
        Args:
            enable_all: Whether to enable all optimizations by default
        """
        self.enabled_optimizations = {
            OptimizationType.KV_CACHE_COLOCATION: enable_all,
            OptimizationType.PREFILL_PARALLELIZATION: enable_all,
            OptimizationType.CNN_PIPELINING: enable_all,
            OptimizationType.MULTIMODAL_FUSION: enable_all,
            OptimizationType.RECOMPUTATION: enable_all,
            OptimizationType.MEMORY_OPTIMIZATION: enable_all
        }
        
        self._optimization_stats = defaultdict(int)
        self._cache = {}
    
    def optimize(self, graph: fx.GraphModule, profile: WorkloadProfile) -> Result[Tuple[fx.GraphModule, OptimizationPlan]]:
        """Apply semantic optimizations to a graph.
        
        Args:
            graph: FX GraphModule to optimize
            profile: Workload profile with semantic information
            
        Returns:
            Result[Tuple[fx.GraphModule, OptimizationPlan]]: Optimized graph and plan, or error
        """
        try:
            # Validate inputs
            if graph is None:
                return Result.err(OptimizationError(
                    "Graph cannot be None",
                    context={'profile': profile.workload_type.value if profile else 'None'}
                ))
            
            if profile is None:
                return Result.err(OptimizationError(
                    "Profile cannot be None",
                    context={'graph_nodes': len(list(graph.graph.nodes)) if graph else 0}
                ))
            
            # Create optimization plan
            plan = self._create_optimization_plan(graph, profile)
            
            # Apply optimizations based on workload type
            if profile.workload_type == WorkloadType.LLM:
                graph = self._apply_llm_optimizations(graph, plan)
            elif profile.workload_type == WorkloadType.VISION:
                graph = self._apply_vision_optimizations(graph, plan)
            elif profile.workload_type == WorkloadType.MULTIMODAL:
                graph = self._apply_multimodal_optimizations(graph, plan)
            
            # Apply general optimizations
            graph = self._apply_general_optimizations(graph, plan)
            
            # Update statistics
            for opt_type in plan.optimizations:
                self._optimization_stats[opt_type.value] += 1
            
            logger.info(f"Applied {len(plan.optimizations)} optimizations to graph")
            
            return Result.ok((graph, plan))
            
        except Exception as e:
            return Result.err(OptimizationError(
                f"Optimization failed: {e}",
                context={
                    'workload_type': profile.workload_type.value if profile else 'unknown',
                    'error': str(e)
                },
                inner_exception=e
            ))
    
    def _create_optimization_plan(self, graph: fx.GraphModule, profile: WorkloadProfile) -> OptimizationPlan:
        """Create an optimization plan based on graph analysis.
        
        Args:
            graph: FX GraphModule to analyze
            profile: Workload profile
            
        Returns:
            OptimizationPlan with planned optimizations
        """
        plan = OptimizationPlan()
        
        # Analyze graph for optimization opportunities
        kv_cache_nodes = self._find_kv_cache_nodes(graph)
        attention_nodes = self._find_attention_nodes(graph)
        conv_nodes = self._find_conv_nodes(graph)
        fusion_candidates = self._find_fusion_candidates(graph)
        
        # Plan optimizations based on findings
        if kv_cache_nodes and self.enabled_optimizations[OptimizationType.KV_CACHE_COLOCATION]:
            plan.optimizations.append(OptimizationType.KV_CACHE_COLOCATION)
            # Group KV cache operations for co-location
            plan.colocation_groups["kv_cache"] = [n.name for n in kv_cache_nodes]
        
        # For LLM workloads, if we have attention nodes, enable parallelization
        if profile.workload_type == WorkloadType.LLM and attention_nodes:
            if self.enabled_optimizations[OptimizationType.PREFILL_PARALLELIZATION]:
                plan.optimizations.append(OptimizationType.PREFILL_PARALLELIZATION)
                # Create parallel branches from attention nodes
                mid = max(1, len(attention_nodes) // 2)
                plan.parallel_branches.append(
                    ([n.name for n in attention_nodes[:mid]], 
                     [n.name for n in attention_nodes[mid:]])
                )
        
        # For Vision workloads, enable pipelining if we have any conv nodes
        if profile.workload_type == WorkloadType.VISION and conv_nodes:
            if self.enabled_optimizations[OptimizationType.CNN_PIPELINING]:
                plan.optimizations.append(OptimizationType.CNN_PIPELINING)
                # Create pipeline stages for CNN layers
                plan.pipeline_stages = self._create_cnn_pipeline_stages(conv_nodes)
        
        # For Multi-modal, check for fusion opportunities
        if profile.workload_type == WorkloadType.MULTIMODAL:
            # Even without explicit fusion candidates, enable fusion optimization
            if self.enabled_optimizations[OptimizationType.MULTIMODAL_FUSION]:
                plan.optimizations.append(OptimizationType.MULTIMODAL_FUSION)
                plan.fusion_points = [n.name for n in fusion_candidates] if fusion_candidates else []
        
        # Always enable memory optimization for any workload
        if self.enabled_optimizations[OptimizationType.MEMORY_OPTIMIZATION]:
            plan.optimizations.append(OptimizationType.MEMORY_OPTIMIZATION)
        
        # Add metadata
        plan.metadata["workload_type"] = profile.workload_type.value
        plan.metadata["total_nodes"] = len(list(graph.graph.nodes))
        
        return plan
    
    def _apply_llm_optimizations(self, graph: fx.GraphModule, plan: OptimizationPlan) -> fx.GraphModule:
        """Apply LLM-specific optimizations.
        
        Implements:
        1. Decode co-location with KV cache
        2. Prefill parallelization
        
        Args:
            graph: FX GraphModule to optimize
            plan: Optimization plan
            
        Returns:
            Optimized graph
        """
        logger.debug("Applying LLM optimizations")
        
        # 1. KV Cache Co-location
        if OptimizationType.KV_CACHE_COLOCATION in plan.optimizations:
            kv_cache_group = plan.colocation_groups.get("kv_cache", [])
            if kv_cache_group:
                # Mark nodes for co-location
                for node in graph.graph.nodes:
                    if node.name in kv_cache_group:
                        if 'semantic' not in node.meta:
                            node.meta['semantic'] = {}
                        node.meta['placement_hint'] = 'kv_cache_device'
                        node.meta['colocation_group'] = 'kv_cache'
                        node.meta['priority'] = 10  # High priority
                
                logger.info(f"Co-located {len(kv_cache_group)} KV cache operations")
        
        # 2. Prefill Parallelization
        if OptimizationType.PREFILL_PARALLELIZATION in plan.optimizations:
            for branch1, branch2 in plan.parallel_branches:
                # Mark branches for parallel execution
                for node in graph.graph.nodes:
                    if node.name in branch1:
                        node.meta['parallel_group'] = 'prefill_branch_1'
                        node.meta['can_parallelize'] = True
                    elif node.name in branch2:
                        node.meta['parallel_group'] = 'prefill_branch_2'
                        node.meta['can_parallelize'] = True
                
                logger.info(f"Parallelized prefill with {len(branch1)} + {len(branch2)} operations")
        
        # 3. Attention optimization - fuse QKV projections if possible
        self._optimize_attention_blocks(graph)
        
        return graph
    
    def _apply_vision_optimizations(self, graph: fx.GraphModule, plan: OptimizationPlan) -> fx.GraphModule:
        """Apply vision-specific optimizations.
        
        Implements:
        1. CNN pipeline scheduling
        2. Conv-BN-ReLU fusion
        
        Args:
            graph: FX GraphModule to optimize
            plan: Optimization plan
            
        Returns:
            Optimized graph
        """
        logger.debug("Applying vision optimizations")
        
        # 1. CNN Pipeline Scheduling
        if OptimizationType.CNN_PIPELINING in plan.optimizations:
            for stage_idx, stage in enumerate(plan.pipeline_stages):
                # Mark nodes for pipeline stage
                for node in graph.graph.nodes:
                    if node.name in stage:
                        node.meta['pipeline_stage'] = stage_idx
                        node.meta['pipeline_enabled'] = True
                        # Earlier stages get higher priority
                        node.meta['priority'] = len(plan.pipeline_stages) - stage_idx
                
            logger.info(f"Created {len(plan.pipeline_stages)} CNN pipeline stages")
        
        # 2. Conv-BN-ReLU fusion
        self._fuse_conv_bn_relu(graph)
        
        # 3. Optimize pooling operations
        self._optimize_pooling(graph)
        
        return graph
    
    def _apply_multimodal_optimizations(self, graph: fx.GraphModule, plan: OptimizationPlan) -> fx.GraphModule:
        """Apply multi-modal optimizations.
        
        Implements:
        1. Parallel modality processing
        2. JIT fusion at fusion points
        
        Args:
            graph: FX GraphModule to optimize
            plan: Optimization plan
            
        Returns:
            Optimized graph
        """
        logger.debug("Applying multi-modal optimizations")
        
        # 1. Identify modality branches
        vision_branch, text_branch = self._identify_modality_branches(graph)
        
        if vision_branch and text_branch:
            # Mark for parallel execution
            for node in vision_branch:
                node.meta['modality'] = 'vision'
                node.meta['parallel_group'] = 'vision_branch'
                node.meta['can_parallelize'] = True
            
            for node in text_branch:
                node.meta['modality'] = 'text'
                node.meta['parallel_group'] = 'text_branch'
                node.meta['can_parallelize'] = True
            
            logger.info(f"Parallelized vision ({len(vision_branch)}) and text ({len(text_branch)}) branches")
        
        # 2. JIT transfer at fusion points
        if OptimizationType.MULTIMODAL_FUSION in plan.optimizations:
            for fusion_point in plan.fusion_points:
                for node in graph.graph.nodes:
                    if node.name == fusion_point:
                        node.meta['is_fusion_point'] = True
                        node.meta['jit_transfer'] = True
                        node.meta['priority'] = 8  # High priority for fusion
                        
                        # Mark predecessors for early computation
                        for pred in node.all_input_nodes:
                            pred.meta['feeds_fusion'] = True
                            pred.meta['early_compute'] = True
            
            logger.info(f"Marked {len(plan.fusion_points)} fusion points for JIT transfer")
        
        return graph
    
    def _apply_general_optimizations(self, graph: fx.GraphModule, plan: OptimizationPlan) -> fx.GraphModule:
        """Apply general optimizations applicable to all workloads.
        
        Args:
            graph: FX GraphModule to optimize
            plan: Optimization plan
            
        Returns:
            Optimized graph
        """
        logger.debug("Applying general optimizations")
        
        # 1. Memory optimization - identify recomputable nodes
        if self.enabled_optimizations[OptimizationType.MEMORY_OPTIMIZATION]:
            recomputable = self._identify_recomputable_nodes(graph)
            for node in recomputable:
                node.meta['can_recompute'] = True
                node.meta['recompute_cost'] = self._estimate_recompute_cost(node)
                if node.meta['recompute_cost'] < 0.5:  # Low cost threshold
                    plan.recompute_nodes.add(node.name)
        
        # 2. Dead code elimination
        self._eliminate_dead_code(graph)
        
        # 3. Common subexpression elimination
        self._eliminate_common_subexpressions(graph)
        
        return graph
    
    def _find_kv_cache_nodes(self, graph: fx.GraphModule) -> List[fx.Node]:
        """Find nodes related to KV cache operations."""
        kv_nodes = []
        
        for node in graph.graph.nodes:
            if node.op == 'call_function':
                # Check semantic metadata
                if 'semantic' in node.meta:
                    metadata = node.meta['semantic']
                    if hasattr(metadata, 'kv_cache_related') and metadata.kv_cache_related:
                        kv_nodes.append(node)
                
                # Check operation name - look for concatenation which might be cache update
                op_name = str(node.target).lower()
                if any(kv in op_name for kv in ['cache', 'past_key', 'past_value']):
                    kv_nodes.append(node)
                # Also check for cat operations that might be cache concatenation
                elif 'cat' in op_name:
                    # Check if this looks like cache concatenation (has zeros as input)
                    for inp in node.all_input_nodes:
                        if 'zeros' in str(inp.target).lower():
                            kv_nodes.append(node)
                            break
        
        return kv_nodes
    
    def _find_attention_nodes(self, graph: fx.GraphModule) -> List[fx.Node]:
        """Find attention-related nodes."""
        attention_nodes = []
        
        for node in graph.graph.nodes:
            if node.op == 'call_function':
                op_name = str(node.target).lower()
                if any(attn in op_name for attn in ['attention', 'matmul', 'softmax']):
                    attention_nodes.append(node)
                
                # Check semantic metadata
                if 'semantic' in node.meta:
                    metadata = node.meta['semantic']
                    if hasattr(metadata, 'semantic_role') and metadata.semantic_role:
                        if 'attention' in str(metadata.semantic_role).lower():
                            attention_nodes.append(node)
        
        return attention_nodes
    
    def _find_conv_nodes(self, graph: fx.GraphModule) -> List[fx.Node]:
        """Find convolution nodes."""
        conv_nodes = []
        
        for node in graph.graph.nodes:
            if node.op == 'call_function':
                op_name = str(node.target).lower()
                if 'conv' in op_name:
                    conv_nodes.append(node)
        
        return conv_nodes
    
    def _find_fusion_candidates(self, graph: fx.GraphModule) -> List[fx.Node]:
        """Find multi-modal fusion candidates."""
        fusion_nodes = []
        
        for node in graph.graph.nodes:
            if node.op == 'call_function':
                # Check for concatenation or addition of different modalities
                op_name = str(node.target).lower()
                if any(fusion_op in op_name for fusion_op in ['cat', 'concat', 'add', 'fusion']):
                    # Check if inputs come from different modalities
                    input_modalities = set()
                    for inp in node.all_input_nodes:
                        if 'modality' in inp.meta:
                            input_modalities.add(inp.meta['modality'])
                    
                    if len(input_modalities) > 1:
                        fusion_nodes.append(node)
                
                # Check semantic metadata
                if 'semantic' in node.meta:
                    metadata = node.meta['semantic']
                    if hasattr(metadata, 'execution_phase'):
                        if metadata.execution_phase == ExecutionPhase.MULTIMODAL_FUSION:
                            fusion_nodes.append(node)
        
        return fusion_nodes
    
    def _is_prefill_phase(self, node: fx.Node) -> bool:
        """Check if node is in prefill phase."""
        if 'semantic' in node.meta:
            metadata = node.meta['semantic']
            if hasattr(metadata, 'execution_phase'):
                return metadata.execution_phase == ExecutionPhase.PREFILL
        return False
    
    def _create_cnn_pipeline_stages(self, conv_nodes: List[fx.Node]) -> List[List[str]]:
        """Create pipeline stages for CNN layers."""
        stages = []
        stage_size = max(1, len(conv_nodes) // 3)  # Create 3 stages by default
        
        for i in range(0, len(conv_nodes), stage_size):
            stage = [node.name for node in conv_nodes[i:i+stage_size]]
            if stage:
                stages.append(stage)
        
        return stages
    
    def _optimize_attention_blocks(self, graph: fx.GraphModule):
        """Optimize attention blocks by fusing QKV projections."""
        # Find QKV projection patterns
        for node in graph.graph.nodes:
            if node.op == 'call_function' and 'linear' in str(node.target).lower():
                # Check if this is part of QKV projections
                if 'semantic' in node.meta:
                    role = node.meta['semantic'].semantic_role if hasattr(node.meta['semantic'], 'semantic_role') else None
                    if role and any(proj in str(role).lower() for proj in ['query', 'key', 'value']):
                        node.meta['can_fuse'] = True
                        node.meta['fusion_group'] = 'qkv_projection'
    
    def _fuse_conv_bn_relu(self, graph: fx.GraphModule):
        """Fuse Conv-BatchNorm-ReLU patterns."""
        # This would use FX's pattern matching to find and fuse patterns
        # For now, just mark eligible nodes
        conv_nodes = self._find_conv_nodes(graph)
        
        for conv_node in conv_nodes:
            # Check if followed by BN and ReLU
            users = list(conv_node.users)
            if users:
                next_node = users[0]
                if 'norm' in str(next_node.target).lower():
                    next_users = list(next_node.users)
                    if next_users and 'relu' in str(next_users[0].target).lower():
                        # Mark for fusion
                        conv_node.meta['fusion_group'] = 'conv_bn_relu'
                        next_node.meta['fusion_group'] = 'conv_bn_relu'
                        next_users[0].meta['fusion_group'] = 'conv_bn_relu'
    
    def _optimize_pooling(self, graph: fx.GraphModule):
        """Optimize pooling operations."""
        for node in graph.graph.nodes:
            if node.op == 'call_function' and 'pool' in str(node.target).lower():
                # Mark for optimization
                node.meta['can_optimize'] = True
                node.meta['optimization_type'] = 'pooling'
    
    def _identify_modality_branches(self, graph: fx.GraphModule) -> Tuple[List[fx.Node], List[fx.Node]]:
        """Identify vision and text branches in multi-modal graph."""
        vision_nodes = []
        text_nodes = []
        
        for node in graph.graph.nodes:
            # Check semantic metadata for modality
            if 'semantic' in node.meta:
                metadata = node.meta['semantic']
                if hasattr(metadata, 'data_lineage') and metadata.data_lineage:
                    modality = metadata.data_lineage.modality
                    if modality == 'vision':
                        vision_nodes.append(node)
                    elif modality == 'text':
                        text_nodes.append(node)
            
            # Heuristic based on operation type
            op_name = str(node.target).lower() if node.op == 'call_function' else ''
            if 'conv' in op_name or 'pool' in op_name:
                vision_nodes.append(node)
            elif 'embedding' in op_name or 'lstm' in op_name or 'gru' in op_name:
                text_nodes.append(node)
        
        return vision_nodes, text_nodes
    
    def _identify_recomputable_nodes(self, graph: fx.GraphModule) -> List[fx.Node]:
        """Identify nodes that can be recomputed instead of stored."""
        recomputable = []
        
        for node in graph.graph.nodes:
            if node.op == 'call_function':
                op_name = str(node.target).lower()
                # Cheap operations that can be recomputed
                if any(cheap_op in op_name for cheap_op in ['relu', 'dropout', 'reshape', 'view', 'transpose']):
                    recomputable.append(node)
        
        return recomputable
    
    def _estimate_recompute_cost(self, node: fx.Node) -> float:
        """Estimate the cost of recomputing a node (0-1, lower is cheaper)."""
        if 'semantic' in node.meta:
            metadata = node.meta['semantic']
            if hasattr(metadata, 'compute_intensity'):
                # Normalize compute intensity to 0-1 range
                return min(1.0, metadata.compute_intensity / 10.0)
        
        # Default heuristic based on operation
        op_name = str(node.target).lower()
        if any(cheap in op_name for cheap in ['relu', 'dropout', 'reshape']):
            return 0.1
        elif any(medium in op_name for medium in ['norm', 'softmax']):
            return 0.5
        else:
            return 0.8
    
    def _eliminate_dead_code(self, graph: fx.GraphModule):
        """Remove dead code from graph."""
        # Mark nodes with no users and not outputs
        dead_nodes = []
        output_nodes = set()
        
        for node in graph.graph.nodes:
            if node.op == 'output':
                # Track output dependencies
                def add_output_deps(n):
                    if isinstance(n, fx.Node):
                        output_nodes.add(n)
                        for inp in n.all_input_nodes:
                            add_output_deps(inp)
                
                for arg in node.args:
                    add_output_deps(arg)
        
        for node in graph.graph.nodes:
            if node.op in ['call_function', 'call_method', 'call_module']:
                if not node.users and node not in output_nodes:
                    dead_nodes.append(node)
                    node.meta['dead_code'] = True
        
        if dead_nodes:
            logger.info(f"Identified {len(dead_nodes)} dead code nodes")
    
    def _eliminate_common_subexpressions(self, graph: fx.GraphModule):
        """Eliminate common subexpressions."""
        # Track expressions by their signature
        expr_map = {}
        
        for node in graph.graph.nodes:
            if node.op == 'call_function':
                # Create signature for the expression
                sig = (node.target, tuple(str(arg) for arg in node.args))
                
                if sig in expr_map:
                    # Found duplicate
                    node.meta['duplicate_of'] = expr_map[sig].name
                    node.meta['can_eliminate'] = True
                else:
                    expr_map[sig] = node
    
    def get_optimization_stats(self) -> Dict[str, int]:
        """Get optimization statistics.
        
        Returns:
            Dictionary of optimization type to count
        """
        return dict(self._optimization_stats)
    
    def reset_stats(self):
        """Reset optimization statistics."""
        self._optimization_stats.clear()
        self._cache.clear()


class AdaptiveOptimizer(SemanticOptimizer):
    """Adaptive optimizer that learns from execution feedback.
    
    This optimizer extends SemanticOptimizer with the ability to adapt
    optimization strategies based on runtime performance feedback.
    """
    
    def __init__(self):
        super().__init__(enable_all=True)
        self.performance_history = []
        self.optimization_effectiveness = defaultdict(float)
    
    def optimize_with_feedback(self, graph: fx.GraphModule, profile: WorkloadProfile,
                              previous_perf: Optional[Dict[str, float]] = None) -> Result[Tuple[fx.GraphModule, OptimizationPlan]]:
        """Optimize with performance feedback.
        
        Args:
            graph: FX GraphModule to optimize
            profile: Workload profile
            previous_perf: Previous performance metrics
            
        Returns:
            Result[Tuple[fx.GraphModule, OptimizationPlan]]: Optimized graph and plan, or error
        """
        # Update effectiveness based on previous performance
        if previous_perf:
            self._update_effectiveness(previous_perf)
        
        # Adjust enabled optimizations based on effectiveness
        self._adjust_optimizations()
        
        # Apply optimizations (returns Result)
        return self.optimize(graph, profile)
    
    def _update_effectiveness(self, perf_metrics: Dict[str, float]):
        """Update optimization effectiveness based on performance."""
        # Simple effectiveness tracking
        latency = perf_metrics.get('latency', 1.0)
        throughput = perf_metrics.get('throughput', 1.0)
        
        # Score based on improvement
        score = throughput / max(latency, 0.001)
        
        self.performance_history.append({
            'score': score,
            'metrics': perf_metrics,
            'optimizations': list(self._optimization_stats.keys())
        })
        
        # Update effectiveness for each optimization
        for opt in self._optimization_stats:
            self.optimization_effectiveness[opt] = score
    
    def _adjust_optimizations(self):
        """Adjust which optimizations are enabled based on effectiveness."""
        # Disable ineffective optimizations
        threshold = 0.5
        
        for opt_type in OptimizationType:
            effectiveness = self.optimization_effectiveness.get(opt_type.value, 1.0)
            self.enabled_optimizations[opt_type] = effectiveness > threshold
