"""
SmartSubgraphBuilder: Cost-aware subgraph fragmentation.

Phase 2 implementation of the network enhancement plan.
This extends the basic SubgraphBuilder with intelligent fragmentation
based on memory constraints, network costs, and compute efficiency.

Key features:
- Memory-aware fragmentation for large graphs
- Cost-based decision making (compute vs transfer)
- Mixed local/remote execution optimization
- Dynamic fragmentation based on runtime constraints

Usage:
    builder = SmartSubgraphBuilder(memory_limit_gb=8, network_gbps=100)
    fragments = builder.build_with_fragmentation(target_tensor)
"""

import logging
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from ..subgraph_builder import SubgraphBuilder, RemoteSubgraph
from ...frontend.core.lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


@dataclass
class FragmentationConfig:
    """Configuration for smart fragmentation."""
    memory_limit_gb: float = 8.0  # Max GPU memory per fragment
    network_gbps: float = 100.0   # Network bandwidth
    compute_tflops: float = 10.0  # GPU compute capacity
    fragmentation_threshold: float = 0.8  # Fragment when > 80% of memory limit
    prefer_local_compute: bool = True     # Prefer local execution when possible


@dataclass
class CostEstimate:
    """Cost estimate for a subgraph fragment."""
    compute_cost_ms: float        # Compute time in milliseconds
    memory_usage_gb: float        # Peak memory usage in GB
    transfer_cost_ms: float       # Network transfer time in milliseconds
    total_cost_ms: float          # Total estimated time
    operations_count: int         # Number of operations
    efficiency_score: float       # Operations per GB of memory


@dataclass
class SubgraphFragment:
    """A fragment of a computation graph."""
    operations: List[LazyTensor]
    input_tensors: Dict[int, LazyTensor]
    output_tensor: LazyTensor
    cost_estimate: CostEstimate
    execution_mode: str  # 'local' or 'remote'
    dependencies: List['SubgraphFragment']  # Fragments this depends on


class SmartSubgraphBuilder(SubgraphBuilder):
    """
    Cost-aware subgraph builder with intelligent fragmentation.

    Extends basic SubgraphBuilder with:
    - Memory constraint awareness
    - Cost-based fragmentation
    - Mixed local/remote execution
    - Dynamic optimization
    """

    def __init__(self, config: FragmentationConfig):
        super().__init__()
        self.config = config
        self.memory_estimator = MemoryEstimator()
        self.cost_calculator = CostCalculator(config)

    def build_with_fragmentation(self, target_tensor: LazyTensor) -> List[SubgraphFragment]:
        """
        Build subgraph fragments with intelligent cost-based decisions.

        Algorithm:
        1. Build full subgraph
        2. Estimate costs and memory usage
        3. If too large, fragment at optimal boundaries
        4. Assign execution modes (local vs remote)
        5. Return list of fragments in execution order
        """
        # Handle None input gracefully
        if target_tensor is None:
            logger.debug("No target tensor provided for fragmentation")
            return []

        # Build initial subgraph
        subgraph = self.build_remote_subgraph(target_tensor)

        if subgraph is None:
            logger.debug("No remote subgraph to fragment")
            return []

        # Estimate costs
        full_cost = self.cost_calculator.estimate_subgraph_cost(subgraph)

        logger.info(f"Full subgraph: {len(subgraph.operations)} ops, "
                   f"memory: {full_cost.memory_usage_gb:.2f}GB, "
                   f"cost: {full_cost.total_cost_ms:.2f}ms")

        # Check if fragmentation is needed
        if full_cost.memory_usage_gb <= self.config.memory_limit_gb * self.config.fragmentation_threshold:
            # No fragmentation needed - execute as single fragment
            fragment = SubgraphFragment(
                operations=subgraph.operations,
                input_tensors=subgraph.input_tensors,
                output_tensor=subgraph.output_tensor,
                cost_estimate=full_cost,
                execution_mode=self._choose_execution_mode(full_cost),
                dependencies=[]
            )
            return [fragment]

        # Fragmentation needed - break into smaller pieces
        logger.info(f"Fragmenting subgraph (memory: {full_cost.memory_usage_gb:.2f}GB > "
                   f"limit: {self.config.memory_limit_gb * self.config.fragmentation_threshold:.2f}GB)")

        fragments = self._fragment_subgraph(subgraph, full_cost)
        return fragments

    def _fragment_subgraph(self, subgraph: RemoteSubgraph, full_cost: CostEstimate) -> List[SubgraphFragment]:
        """
        Fragment subgraph into smaller pieces based on cost analysis.

        Strategy:
        1. Find natural fragmentation points (layer boundaries, etc.)
        2. Estimate cost of each fragment
        3. Choose optimal cut points that minimize total cost
        4. Create fragments with dependencies
        """
        operations = subgraph.operations

        if len(operations) <= 2:
            # Too small to fragment effectively
            fragment = SubgraphFragment(
                operations=operations,
                input_tensors=subgraph.input_tensors,
                output_tensor=subgraph.output_tensor,
                cost_estimate=full_cost,
                execution_mode=self._choose_execution_mode(full_cost),
                dependencies=[]
            )
            return [fragment]

        # Find optimal fragmentation points
        cut_points = self._find_optimal_cuts(operations, full_cost)

        if not cut_points:
            # No good fragmentation points found
            fragment = SubgraphFragment(
                operations=operations,
                input_tensors=subgraph.input_tensors,
                output_tensor=subgraph.output_tensor,
                cost_estimate=full_cost,
                execution_mode=self._choose_execution_mode(full_cost),
                dependencies=[]
            )
            return [fragment]

        # Create fragments
        fragments = []
        prev_cut = 0

        for cut_idx in cut_points:
            fragment_ops = operations[prev_cut:cut_idx + 1]

            # Build input tensors for this fragment
            fragment_inputs = {}
            for op in fragment_ops:
                for inp in op.inputs:
                    if isinstance(inp, LazyTensor) and id(inp) not in fragment_inputs:
                        # Check if input is from previous fragment or external
                        if any(id(inp) in [id(prev_op) for prev_op in operations[:prev_cut]] for prev_op in operations[:prev_cut]):
                            # Input is from previous fragment - this creates dependency
                            pass  # Will be handled by dependency tracking
                        else:
                            # External input
                            fragment_inputs[id(inp)] = inp

            # Estimate cost for this fragment
            fragment_cost = self.cost_calculator.estimate_operations_cost(fragment_ops)

            fragment = SubgraphFragment(
                operations=fragment_ops,
                input_tensors=fragment_inputs,
                output_tensor=fragment_ops[-1],
                cost_estimate=fragment_cost,
                execution_mode=self._choose_execution_mode(fragment_cost),
                dependencies=[]  # Will be computed after all fragments are created
            )

            fragments.append(fragment)
            prev_cut = cut_idx + 1

        # Add remaining operations
        if prev_cut < len(operations):
            fragment_ops = operations[prev_cut:]
            fragment_inputs = {}

            for op in fragment_ops:
                for inp in op.inputs:
                    if isinstance(inp, LazyTensor) and id(inp) not in fragment_inputs:
                        if any(id(inp) in [id(prev_op) for prev_op in operations[:prev_cut]] for prev_op in operations[:prev_cut]):
                            pass  # Dependency on previous fragment
                        else:
                            fragment_inputs[id(inp)] = inp

            fragment_cost = self.cost_calculator.estimate_operations_cost(fragment_ops)

            fragment = SubgraphFragment(
                operations=fragment_ops,
                input_tensors=fragment_inputs,
                output_tensor=fragment_ops[-1],
                cost_estimate=fragment_cost,
                execution_mode=self._choose_execution_mode(fragment_cost),
                dependencies=[]
            )

            fragments.append(fragment)

        # Compute dependencies between fragments
        self._compute_fragment_dependencies(fragments, operations)

        # Sort fragments in execution order (respecting dependencies)
        sorted_fragments = self._topological_sort_fragments(fragments)

        logger.info(f"Fragmented into {len(sorted_fragments)} pieces")
        for i, frag in enumerate(sorted_fragments):
            logger.debug(f"  Fragment {i}: {len(frag.operations)} ops, "
                        f"mode={frag.execution_mode}, cost={frag.cost_estimate.total_cost_ms:.2f}ms")

        return sorted_fragments

    def _find_optimal_cuts(self, operations: List[LazyTensor], full_cost: CostEstimate) -> List[int]:
        """
        Find optimal points to cut the operation sequence.

        Strategy:
        1. Look for natural boundaries (e.g., between layers)
        2. Consider memory usage patterns
        3. Choose cuts that balance fragment sizes
        """
        cuts = []

        if len(operations) <= 3:
            return cuts  # Too small to fragment

        # Find operations with high memory usage (potential cut points)
        memory_intensive_ops = []
        for i, op in enumerate(operations):
            # Estimate memory for this operation
            mem_estimate = self.memory_estimator.estimate_operation_memory(op)

            if mem_estimate > self.config.memory_limit_gb * 0.3:  # > 30% of limit
                memory_intensive_ops.append((i, mem_estimate))

        # Look for natural layer boundaries
        layer_boundaries = self._find_layer_boundaries(operations)

        # Combine memory-intensive ops and layer boundaries
        potential_cuts = set()
        for idx, _ in memory_intensive_ops:
            potential_cuts.add(idx)

        for boundary in layer_boundaries:
            potential_cuts.add(boundary)

        # Filter cuts to ensure reasonable fragment sizes
        valid_cuts = []
        for cut_idx in sorted(potential_cuts):
            if cut_idx > 0 and cut_idx < len(operations) - 1:
                # Check fragment sizes
                left_size = cut_idx + 1
                right_size = len(operations) - cut_idx - 1

                if left_size >= 1 and right_size >= 1:
                    valid_cuts.append(cut_idx)

        return valid_cuts

    def _find_layer_boundaries(self, operations: List[LazyTensor]) -> List[int]:
        """
        Find natural boundaries between layers.

        Looks for patterns like:
        - Batch normalization after convolution
        - Residual connections
        - Module boundaries
        """
        boundaries = []

        for i in range(1, len(operations)):
            op1 = operations[i-1]
            op2 = operations[i]

            # Look for activation functions (often end layers)
            if op1.operation in ['aten::relu', 'aten::gelu', 'aten::sigmoid', 'aten::tanh']:
                boundaries.append(i-1)

            # Look for pooling operations (often end feature extraction)
            if 'pool' in op1.operation.lower():
                boundaries.append(i-1)

            # Look for module boundaries based on metadata
            if hasattr(op1, 'metadata') and hasattr(op2, 'metadata'):
                module1 = op1.metadata.get('module_path', '')
                module2 = op2.metadata.get('module_path', '')

                if module1 != module2 and module1 and module2:
                    boundaries.append(i-1)

        return boundaries

    def _compute_fragment_dependencies(self, fragments: List[SubgraphFragment], all_operations: List[LazyTensor]):
        """Compute dependencies between fragments."""
        # Build operation to fragment mapping
        op_to_fragment = {}
        for frag_idx, fragment in enumerate(fragments):
            for op in fragment.operations:
                op_to_fragment[id(op)] = frag_idx

        # Compute dependencies
        for frag_idx, fragment in enumerate(fragments):
            dependencies = set()

            for op in fragment.operations:
                for inp in op.inputs:
                    if isinstance(inp, LazyTensor):
                        inp_frag_idx = op_to_fragment.get(id(inp))
                        if inp_frag_idx is not None and inp_frag_idx != frag_idx:
                            dependencies.add(inp_frag_idx)

            fragment.dependencies = [fragments[idx] for idx in sorted(dependencies)]

    def _topological_sort_fragments(self, fragments: List[SubgraphFragment]) -> List[SubgraphFragment]:
        """Sort fragments in topological order respecting dependencies."""
        # Simple topological sort (could be optimized)
        sorted_fragments = []
        remaining = fragments.copy()
        processed = set()

        def has_unprocessed_dependencies(frag):
            for dep in frag.dependencies:
                if id(dep) not in processed:
                    return True
            return False

        while remaining:
            # Find fragments with no unprocessed dependencies
            ready_fragments = [f for f in remaining if not has_unprocessed_dependencies(f)]

            if not ready_fragments:
                # Circular dependency or error - just return remaining fragments
                logger.warning("Circular dependency detected in fragments, using original order")
                sorted_fragments.extend(remaining)
                break

            # Process ready fragments
            for frag in ready_fragments:
                sorted_fragments.append(frag)
                remaining.remove(frag)
                processed.add(id(frag))

        return sorted_fragments

    def _choose_execution_mode(self, cost: CostEstimate) -> str:
        """
        Choose between local and remote execution for a fragment.

        Decision based on:
        - Compute vs transfer cost tradeoff
        - Memory constraints
        - Network availability
        """
        # If memory usage is very high, prefer remote even if slower
        if cost.memory_usage_gb > self.config.memory_limit_gb * 0.9:
            return 'remote'

        # If transfer cost is much higher than compute cost, prefer local
        transfer_ratio = cost.transfer_cost_ms / max(cost.compute_cost_ms, 1.0)

        if self.config.prefer_local_compute and transfer_ratio > 2.0:
            return 'local'

        # Default to remote for better GPU utilization
        return 'remote'


class MemoryEstimator:
    """Estimates memory usage for operations and tensors."""

    def __init__(self):
        self.tensor_memory_cache = {}

    def estimate_operation_memory(self, operation: LazyTensor) -> float:
        """Estimate peak memory usage for an operation in GB."""
        # Get input tensor shapes and dtypes
        input_memory = 0.0

        for inp in operation.inputs:
            if isinstance(inp, LazyTensor):
                mem = self._estimate_tensor_memory(inp)
                input_memory += mem
            elif isinstance(inp, torch.Tensor):
                mem = self._estimate_tensor_memory_from_shape(inp.shape, inp.dtype)
                input_memory += mem

        # Estimate output memory
        output_memory = self._estimate_tensor_memory(operation)

        # Peak memory is max of inputs and outputs (simplified)
        peak_memory = max(input_memory, output_memory)

        # Add some overhead for intermediate computations
        overhead_factor = 1.2  # 20% overhead

        return peak_memory * overhead_factor

    def estimate_subgraph_memory(self, subgraph: RemoteSubgraph) -> float:
        """Estimate peak memory usage for entire subgraph."""
        if not subgraph.operations:
            return 0.0

        # Simulate memory usage over time
        max_memory = 0.0
        current_memory = 0.0

        # Input tensors
        for tensor in subgraph.input_tensors.values():
            mem = self._estimate_tensor_memory(tensor)
            current_memory += mem

        max_memory = max(max_memory, current_memory)

        # Operations
        for op in subgraph.operations:
            # Remove input tensors that are consumed
            for inp in op.inputs:
                if isinstance(inp, LazyTensor):
                    mem = self._estimate_tensor_memory(inp)
                    current_memory -= mem

            # Add output tensor
            output_mem = self._estimate_tensor_memory(op)
            current_memory += output_mem

            max_memory = max(max_memory, current_memory)

        return max_memory

    def _estimate_tensor_memory(self, tensor: LazyTensor) -> float:
        """Estimate memory for a LazyTensor."""
        cache_key = (tuple(tensor.shape), str(tensor.dtype))

        if cache_key in self.tensor_memory_cache:
            return self.tensor_memory_cache[cache_key]

        memory_bytes = self._estimate_tensor_memory_from_shape(tensor.shape, tensor.dtype)
        self.tensor_memory_cache[cache_key] = memory_bytes

        return memory_bytes

    def _estimate_tensor_memory_from_shape(self, shape: torch.Size, dtype: torch.dtype) -> float:
        """Estimate memory from shape and dtype."""
        if not shape:
            return 0.0

        # Calculate number of elements
        elements = 1
        for dim in shape:
            elements *= dim

        # Memory per element (bytes)
        dtype_sizes = {
            torch.float32: 4,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
        }

        bytes_per_element = dtype_sizes.get(dtype, 4)  # Default to float32

        total_bytes = elements * bytes_per_element
        total_gb = total_bytes / (1024 ** 3)  # Convert to GB

        return total_gb


class CostCalculator:
    """Calculates compute vs transfer costs for optimization decisions."""

    def __init__(self, config: FragmentationConfig):
        self.config = config

    def estimate_subgraph_cost(self, subgraph: RemoteSubgraph) -> CostEstimate:
        """Estimate total cost for a subgraph."""
        operations = subgraph.operations

        if not operations:
            return CostEstimate(0.0, 0.0, 0.0, 0.0, 0, 0.0)

        # Estimate compute cost
        compute_cost_ms = self._estimate_compute_cost(operations)

        # Estimate memory usage
        memory_estimator = MemoryEstimator()
        memory_gb = memory_estimator.estimate_subgraph_memory(subgraph)

        # Estimate transfer cost (input + output tensors)
        transfer_cost_ms = self._estimate_transfer_cost(subgraph)

        # Total cost
        total_cost_ms = compute_cost_ms + transfer_cost_ms

        # Efficiency score (operations per GB)
        efficiency = len(operations) / max(memory_gb, 0.1)  # Avoid division by zero

        return CostEstimate(
            compute_cost_ms=compute_cost_ms,
            memory_usage_gb=memory_gb,
            transfer_cost_ms=transfer_cost_ms,
            total_cost_ms=total_cost_ms,
            operations_count=len(operations),
            efficiency_score=efficiency
        )

    def estimate_operations_cost(self, operations: List[LazyTensor]) -> CostEstimate:
        """Estimate cost for a list of operations."""
        if not operations:
            return CostEstimate(0.0, 0.0, 0.0, 0.0, 0, 0.0)

        # Create a temporary subgraph for estimation
        # Find all input tensors that operations depend on
        input_tensors = {}
        for op in operations:
            for inp in op.inputs:
                if isinstance(inp, LazyTensor) and id(inp) not in input_tensors:
                    input_tensors[id(inp)] = inp

        temp_subgraph = RemoteSubgraph(
            operations=operations,
            input_tensors=input_tensors,
            output_tensor=operations[-1]
        )

        return self.estimate_subgraph_cost(temp_subgraph)

    def _estimate_compute_cost(self, operations: List[LazyTensor]) -> float:
        """Estimate compute time for operations in milliseconds."""
        total_flops = 0.0

        for op in operations:
            flops = self._estimate_operation_flops(op)
            total_flops += flops

        # Convert FLOPs to time (assuming 10 TFLOPS = 10^13 FLOPS per second)
        flops_per_second = self.config.compute_tflops * 1e12
        time_seconds = total_flops / flops_per_second
        time_ms = time_seconds * 1000

        return time_ms

    def _estimate_operation_flops(self, operation: LazyTensor) -> float:
        """Estimate FLOPs for a single operation."""
        op_name = operation.operation

        # Simple FLOP estimation based on operation type
        if 'matmul' in op_name.lower() or 'mm' in op_name.lower():
            # Matrix multiplication: 2 * M * N * K
            if operation.shape and len(operation.shape) >= 2:
                m, n = operation.shape[-2], operation.shape[-1]
                k = operation.inputs[0].shape[-1] if operation.inputs else n
                return 2.0 * m * n * k
            return 1e6  # Default

        elif 'conv' in op_name.lower():
            # Convolution: 2 * H_out * W_out * K_h * K_w * C_in * C_out
            return 1e8  # Placeholder

        elif any(x in op_name.lower() for x in ['add', 'sub', 'mul', 'div', 'relu', 'sigmoid', 'tanh', 'abs', 'neg', 'exp', 'log', 'sqrt']):
            # Element-wise: number of elements (1 FLOP per element)
            elements = 1
            if operation.shape:
                for dim in operation.shape:
                    elements *= dim
            return float(elements)

        else:
            # Default estimate
            return 1e3

    def _estimate_transfer_cost(self, subgraph: RemoteSubgraph) -> float:
        """Estimate network transfer time in milliseconds."""
        # Define dtype sizes once (fix variable scoping issue)
        dtype_sizes = {
            torch.float32: 4,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
        }

        # Input tensors
        input_bytes = 0
        for tensor in subgraph.input_tensors.values():
            if isinstance(tensor, LazyTensor):
                elements = 1
                if tensor.shape:
                    for dim in tensor.shape:
                        elements *= dim

                bytes_per_element = dtype_sizes.get(tensor.dtype, 4)
                input_bytes += elements * bytes_per_element

        # Output tensor
        output_bytes = 0
        if subgraph.output_tensor.shape:
            elements = 1
            for dim in subgraph.output_tensor.shape:
                elements *= dim

            bytes_per_element = dtype_sizes.get(subgraph.output_tensor.dtype, 4)
            output_bytes += elements * bytes_per_element

        total_bytes = input_bytes + output_bytes

        # Transfer time: bytes / bandwidth
        bandwidth_bps = self.config.network_gbps * 1e9  # Convert to bits per second
        bandwidth_Bps = bandwidth_bps / 8  # Convert to bytes per second

        transfer_time_seconds = total_bytes / bandwidth_Bps
        transfer_time_ms = transfer_time_seconds * 1000

        return transfer_time_ms
