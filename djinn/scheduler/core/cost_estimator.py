"""
Cost estimation for operations.

Estimates:
- Compute cost (FLOPs)
- Memory footprint (bytes)
- Operational intensity (FLOPs/byte)
- Data movement cost

Uses PyTorch's FlopCounterMode for accurate FLOP counting.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """Cost estimate for an operation."""
    compute_flops: float  # Floating point operations
    memory_bytes: float   # Memory footprint
    operational_intensity: float  # FLOPs per byte
    data_movement_bytes: float  # Input/output data volume

    # Network costs (NEW for Phase 4)
    transfer_time_ms: float = 0.0  # Estimated transfer time
    queueing_delay_ms: float = 0.0  # Estimated queueing delay


class NetworkTopology:
    """Network topology information for cost estimation.

    This is a simplified interface that delegates to the global NetworkTopologyManager.
    """

    def __init__(self):
        # Import here to avoid circular dependency
        from ...core.network_topology import get_network_topology
        self._manager = get_network_topology()

    def register_node(self, node_id: str, bandwidth_gbps: float, latency_ms: float):
        """Register network information for a node."""
        from ...core.network_topology import NetworkDevice
        device = NetworkDevice(
            node_id=node_id,
            bandwidth_gbps=bandwidth_gbps,
            latency_ms=latency_ms,
            device_type='gpu',  # Default assumption
            compute_tflops=10.0,  # Default assumption
            memory_gb=16.0       # Default assumption
        )
        self._manager.register_device(device)

    def get_bandwidth(self, src_node: str, dst_node: str) -> float:
        """Get bandwidth between source and destination nodes."""
        return self._manager.get_bandwidth(src_node, dst_node)

    def get_latency(self, src_node: str, dst_node: str) -> float:
        """Get latency between source and destination nodes."""
        return self._manager.get_latency(src_node, dst_node)

    def estimate_transfer_time(self, bytes_to_transfer: float, src_node: str, dst_node: str) -> float:
        """Estimate transfer time in milliseconds."""
        return self._manager.estimate_transfer_time(bytes_to_transfer, src_node, dst_node)

    def estimate_queueing_delay(self, src_node: str, dst_node: str, queue_depth: int = 1) -> float:
        """Estimate queueing delay based on network congestion."""
        return self._manager.estimate_queueing_delay(src_node, dst_node, queue_depth)

    def update_from_coordinator(self):
        """Update network information from coordinator."""
        self._manager.update_from_coordinator()


class CostEstimator:
    """
    Estimates computational costs for operations.

    Uses shape information and operation type to estimate costs.
    """

    def __init__(self, network_topology: Optional[NetworkTopology] = None):
        self.network_topology = network_topology or NetworkTopology()

    def estimate_operation(self, node) -> CostEstimate:
        """
        Estimate cost for a single operation.

        Args:
            node: GraphNode with shape and dtype information

        Returns:
            CostEstimate with compute, memory, and intensity metrics
        """
        op = node.operation.lower()

        # Get the underlying tensor if this is a node adapter
        tensor = getattr(node, 'tensor', node)

        # Dispatch to operation-specific estimator
        # ✅ FIX: Pass node (not tensor) to matmul estimator so it has access to inputs
        if 'matmul' in op or 'mm' in op or 'linear' in op:
            return self._estimate_matmul(node)
        elif 'conv' in op:
            return self._estimate_conv(tensor)
        elif 'softmax' in op and self._is_attention_context(node):
            return self._estimate_attention(tensor)
        elif any(x in op for x in ['add', 'sub', 'mul', 'div']):
            return self._estimate_elementwise(tensor)
        elif 'softmax' in op or 'sigmoid' in op:
            return self._estimate_pointwise(tensor)
        else:
            return self._estimate_generic(tensor)
    
    def _estimate_matmul(self, node) -> CostEstimate:
        """Estimate cost for matrix multiplication."""
        # Use manual estimation (FlopCounterMode not available in this PyTorch version)
        return self._estimate_matmul_fallback(node)

    def _execute_matmul_symbolic(self, node):
        """Execute matmul operation symbolically on meta device."""
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 2:
            return None

        # Create meta tensors
        meta_inputs = []
        for inp in inputs:
            shape = getattr(inp, 'shape', None)
            if shape:
                meta_inputs.append(torch.empty(shape, device='meta'))
            else:
                return None

        # Execute operation
        op = node.operation.lower()
        if 'matmul' in op or 'mm' in op:
            return torch.matmul(meta_inputs[0], meta_inputs[1])
        elif 'bmm' in op:
            return torch.bmm(meta_inputs[0], meta_inputs[1])
        else:
            return torch.mm(meta_inputs[0], meta_inputs[1])

    def _estimate_matmul_memory(self, node):
        """Estimate memory usage for matmul."""
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 2:
            return 0

        shape_a = getattr(inputs[0], 'shape', None)
        shape_b = getattr(inputs[1], 'shape', None)

        if shape_a and shape_b:
            m, k = shape_a[-2], shape_a[-1]
            n = shape_b[-1]
            # Memory: inputs + output
            dtype_bytes = 4  # float32
            return (m * k + k * n + m * n) * dtype_bytes

        return 0

    def _estimate_matmul_fallback(self, tensor) -> CostEstimate:
        """Fallback estimation when FlopCounterMode fails."""
        # Fallback to manual calculation
        inputs = getattr(tensor, 'inputs', [])
        if len(inputs) < 2:
            logger.debug(f"Matmul node has {len(inputs)} inputs, falling back to generic estimate")
            return self._estimate_generic(tensor)

        shape_a = getattr(inputs[0], 'shape', None)
        shape_b = getattr(inputs[1], 'shape', None)

        if not shape_a or not shape_b:
            logger.debug(f"Matmul inputs missing shapes: shape_a={shape_a}, shape_b={shape_b}, falling back to generic")
            return self._estimate_generic(tensor)

        if shape_a and shape_b:
            # Handle linear operations (A @ B where B is weight matrix)
            # Linear: input [B, in_features] @ weight [out_features, in_features].T
            if len(shape_a) == 2 and len(shape_b) == 2:
                # Standard linear: [B, in_features] @ [out_features, in_features]
                B, in_features = shape_a
                out_features, _ = shape_b

                # FLOPs: 2 * B * out_features * in_features (multiply-add)
                compute_flops = 2 * B * out_features * in_features

                # Memory: input + weight + output + bias (if present)
                dtype_bytes = 4
                input_bytes = B * in_features * dtype_bytes
                weight_bytes = out_features * in_features * dtype_bytes
                output_bytes = B * out_features * dtype_bytes
                memory_bytes = input_bytes + weight_bytes + output_bytes

                # Estimate transfer costs (for remote execution)
                transfer_time_ms = self.network_topology.estimate_transfer_time(
                    input_bytes + output_bytes, "local", "remote"
                )
                queueing_delay_ms = self.network_topology.estimate_queueing_delay(
                    "local", "remote", 1  # Assume queue depth 1 for now
                )

                return CostEstimate(
                    compute_flops=float(compute_flops),
                    memory_bytes=float(memory_bytes),
                    operational_intensity=compute_flops / memory_bytes if memory_bytes > 0 else 0,
                    data_movement_bytes=float(input_bytes + output_bytes),
                    transfer_time_ms=transfer_time_ms,
                    queueing_delay_ms=queueing_delay_ms
                )
            else:
                # General matmul case
                m, k = shape_a[-2], shape_a[-1]
                n = shape_b[-1]
                compute_flops = 2 * m * n * k

                dtype_bytes = 4
                input_bytes = (m * k + k * n + m * n) * dtype_bytes

                return CostEstimate(
                    compute_flops=float(compute_flops),
                    memory_bytes=float(input_bytes),
                    operational_intensity=compute_flops / input_bytes if input_bytes > 0 else 0,
                    data_movement_bytes=float(input_bytes)
                )

        return self._estimate_generic(tensor)
    
    def _estimate_conv(self, tensor) -> CostEstimate:
        """Accurate convolution cost estimation."""
        # Use manual estimation (FlopCounterMode not available in this PyTorch version)
        return self._estimate_conv_manual(tensor)

    def _execute_conv_symbolic(self, node):
        """Execute conv operation symbolically on meta device."""
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 2:
            return None

        # Create meta tensors
        meta_inputs = []
        for inp in inputs:
            shape = getattr(inp, 'shape', None)
            if shape:
                meta_inputs.append(torch.empty(shape, device='meta'))
            else:
                return None

        # Parse kwargs for conv parameters
        kwargs = getattr(node, 'kwargs', {})

        # Execute operation
        op = node.operation.lower()
        if 'conv2d' in op:
            return torch.conv2d(meta_inputs[0], meta_inputs[1], **kwargs)
        elif 'conv1d' in op:
            return torch.conv1d(meta_inputs[0], meta_inputs[1], **kwargs)
        elif 'conv3d' in op:
            return torch.conv3d(meta_inputs[0], meta_inputs[1], **kwargs)
        else:
            return torch.conv2d(meta_inputs[0], meta_inputs[1], **kwargs)

    def _estimate_conv_memory(self, node):
        """Estimate memory usage for convolution."""
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 2:
            return 0

        shape_input = getattr(inputs[0], 'shape', None)
        shape_weight = getattr(inputs[1], 'shape', None)

        if shape_input and shape_weight:
            # Input: [N, C_in, H_in, W_in]
            # Weight: [C_out, C_in, K_h, K_w]
            # Output: [N, C_out, H_out, W_out]

            N, C_in, H_in, W_in = shape_input
            C_out, _, K_h, K_w = shape_weight

            # Parse conv parameters
            kwargs = getattr(node, 'kwargs', {})
            stride = kwargs.get('stride', 1)
            if isinstance(stride, (list, tuple)):
                stride_h, stride_w = stride
            else:
                stride_h = stride_w = stride

            padding = kwargs.get('padding', 0)
            if isinstance(padding, (list, tuple)):
                pad_h, pad_w = padding
            else:
                pad_h = pad_w = padding

            # Output dimensions
            H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
            W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

            # Memory: input + weight + output
            dtype_bytes = 4  # float32
            input_bytes = N * C_in * H_in * W_in * dtype_bytes
            weight_bytes = C_out * C_in * K_h * K_w * dtype_bytes
            output_bytes = N * C_out * H_out * W_out * dtype_bytes

            return input_bytes + weight_bytes + output_bytes

        return 0

    def _estimate_conv_manual(self, node) -> CostEstimate:
        """Manual convolution cost estimation when FlopCounterMode fails."""
        # Accurate convolution cost estimation
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 2:
            return self._estimate_generic(node)

        shape_input = getattr(inputs[0], 'shape', None)
        shape_weight = getattr(inputs[1], 'shape', None)

        if not shape_input or not shape_weight:
            return self._estimate_generic(node)

        # Input: [N, C_in, H_in, W_in]
        # Weight: [C_out, C_in, K_h, K_w]
        N, C_in, H_in, W_in = shape_input
        C_out, _, K_h, K_w = shape_weight

        # Parse conv parameters
        kwargs = getattr(node, 'kwargs', {})
        stride = kwargs.get('stride', 1)
        if isinstance(stride, (list, tuple)):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride

        padding = kwargs.get('padding', 0)
        if isinstance(padding, (list, tuple)):
            pad_h, pad_w = padding
        else:
            pad_h = pad_w = padding

        # Output dimensions
        H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
        W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

        # FLOPs: 2 (MAC) * output_elements * kernel_elements * input_channels
        compute_flops = 2 * N * C_out * H_out * W_out * K_h * K_w * C_in

        # Memory: input + weight + output
        dtype_bytes = 4  # float32
        input_bytes = N * C_in * H_in * W_in * dtype_bytes
        weight_bytes = C_out * C_in * K_h * K_w * dtype_bytes
        output_bytes = N * C_out * H_out * W_out * dtype_bytes
        memory_bytes = input_bytes + weight_bytes + output_bytes

        # Estimate transfer costs (for remote execution)
        transfer_time_ms = self.network_topology.estimate_transfer_time(
            input_bytes + output_bytes, "local", "remote"
        )
        queueing_delay_ms = self.network_topology.estimate_queueing_delay(
            "local", "remote", 1
        )

        return CostEstimate(
            compute_flops=float(compute_flops),
            memory_bytes=float(memory_bytes),
            operational_intensity=compute_flops / memory_bytes if memory_bytes > 0 else 0,
            data_movement_bytes=float(input_bytes + output_bytes),
            transfer_time_ms=transfer_time_ms,
            queueing_delay_ms=queueing_delay_ms
        )
    
    def _estimate_attention(self, tensor) -> CostEstimate:
        """Accurate attention cost estimation."""
        # Use manual estimation (FlopCounterMode not available in this PyTorch version)
        return self._estimate_attention_manual(tensor)

    def _execute_attention_symbolic(self, node):
        """Execute attention operation symbolically on meta device."""
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 3:
            return None

        # Create meta tensors for Q, K, V
        meta_inputs = []
        for inp in inputs[:3]:  # Q, K, V
            shape = getattr(inp, 'shape', None)
            if shape:
                meta_inputs.append(torch.empty(shape, device='meta'))
            else:
                return None

        # Execute attention computation
        q, k, v = meta_inputs

        # Q @ K.T
        scores = torch.matmul(q, k.transpose(-2, -1))
        # Softmax
        attn = torch.softmax(scores, dim=-1)
        # @ V
        result = torch.matmul(attn, v)

        return result

    def _estimate_attention_memory(self, node):
        """Estimate memory usage for attention."""
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 3:
            return 0

        # Assume Q, K, V have same shape for simplicity
        shape = getattr(inputs[0], 'shape', None)
        if shape:
            # [B, L, H] for all Q, K, V
            B, L, H = shape

            # Memory: Q + K + V + scores + output
            dtype_bytes = 4  # float32
            qkv_bytes = 3 * B * L * H * dtype_bytes
            scores_bytes = B * L * L * dtype_bytes  # Attention scores
            output_bytes = B * L * H * dtype_bytes

            return qkv_bytes + scores_bytes + output_bytes

        return 0

    def _estimate_attention_manual(self, node) -> CostEstimate:
        """Manual attention cost estimation when FlopCounterMode fails."""
        # Attention: Q @ K.T (O(n^2)) + softmax (O(n)) + @ V (O(n^2))
        inputs = getattr(node, 'inputs', [])
        if len(inputs) < 3:
            return self._estimate_generic(node)

        # Assume Q, K, V have same shape for simplicity
        shape = getattr(inputs[0], 'shape', None)
        if not shape:
            return self._estimate_generic(node)

        # [B, L, H] for all Q, K, V
        B, L, H = shape

        # FLOPs: Q@K.T (2*B*L*L*H) + softmax (~5*B*L*L) + @V (2*B*L*L*H)
        qk_flops = 2 * B * L * L * H
        softmax_flops = B * L * L * 5  # Rough estimate
        sv_flops = 2 * B * L * L * H

        total_flops = qk_flops + softmax_flops + sv_flops

        # Memory: Q + K + V + scores + output
        dtype_bytes = 4  # float32
        qkv_bytes = 3 * B * L * H * dtype_bytes
        scores_bytes = B * L * L * dtype_bytes
        output_bytes = B * L * H * dtype_bytes
        memory_bytes = qkv_bytes + scores_bytes + output_bytes

        # Estimate transfer costs (for remote execution)
        transfer_time_ms = self.network_topology.estimate_transfer_time(
            qkv_bytes + output_bytes, "local", "remote"
        )
        queueing_delay_ms = self.network_topology.estimate_queueing_delay(
            "local", "remote", 1
        )

        return CostEstimate(
            compute_flops=float(total_flops),
            memory_bytes=float(memory_bytes),
            operational_intensity=total_flops / memory_bytes if memory_bytes > 0 else 0,
            data_movement_bytes=float(qkv_bytes + output_bytes),
            transfer_time_ms=transfer_time_ms,
            queueing_delay_ms=queueing_delay_ms
        )
    
    def _estimate_elementwise(self, tensor) -> CostEstimate:
        """Estimate cost for element-wise operations."""
        shape = getattr(tensor, 'shape', None)
        if shape:
            try:
                elements = 1
                for dim in shape:
                    elements *= int(dim)

                # Element-wise: 1 FLOP per element
                compute_flops = elements
                memory_bytes = elements * 4  # float32

                # Estimate transfer costs (for remote execution)
                transfer_time_ms = self.network_topology.estimate_transfer_time(
                    memory_bytes, "local", "remote"
                )
                queueing_delay_ms = self.network_topology.estimate_queueing_delay(
                    "local", "remote", 1
                )

                return CostEstimate(
                    compute_flops=float(compute_flops),
                    memory_bytes=float(memory_bytes),
                    operational_intensity=1.0,  # Low intensity
                    data_movement_bytes=float(memory_bytes),
                    transfer_time_ms=transfer_time_ms,
                    queueing_delay_ms=queueing_delay_ms
                )
            except (TypeError, AttributeError):
                pass

        return self._estimate_generic(tensor)
    
    def _estimate_pointwise(self, tensor) -> CostEstimate:
        """Estimate cost for pointwise operations (softmax, sigmoid, etc.)."""
        return self._estimate_elementwise(tensor)
    
    def _estimate_generic(self, tensor) -> CostEstimate:
        """Generic cost estimate (conservative)."""
        # Estimate transfer costs (for remote execution)
        transfer_time_ms = self.network_topology.estimate_transfer_time(
            1000, "local", "remote"
        )
        queueing_delay_ms = self.network_topology.estimate_queueing_delay(
            "local", "remote", 1
        )

        return CostEstimate(
            compute_flops=1000,  # Conservative estimate
            memory_bytes=1000,
            operational_intensity=1.0,
            data_movement_bytes=1000,
            transfer_time_ms=transfer_time_ms,
            queueing_delay_ms=queueing_delay_ms
        )

    def _is_attention_context(self, node) -> bool:
        """Check if softmax is in attention context (has matmul inputs/outputs)."""
        # Get the underlying tensor
        tensor = getattr(node, 'tensor', node)

        # Check if softmax has matmul as input (Q@K.T)
        for inp in getattr(tensor, 'inputs', []):
            if hasattr(inp, 'operation') and 'matmul' in inp.operation.lower():
                return True

        # Check if softmax has matmul as consumer (scores@V)
        if hasattr(node, 'get_consumers'):
            for consumer in node.get_consumers():
                if self._is_matmul_like(consumer):
                    return True

        return False

    def _is_matmul_like(self, node) -> bool:
        """Check if node is matrix multiplication."""
        op = getattr(node, 'operation', '').lower()
        return 'matmul' in op or 'mm' in op or 'bmm' in op or 'linear' in op


class GraphCostEstimator:
    """
    Estimates total cost for a computation graph.
    
    Aggregates costs of individual operations and estimates
    data movement between operations.
    """
    
    def __init__(self):
        self.operation_estimator = CostEstimator()
    
    def estimate_graph(self, graph) -> Dict[str, Any]:
        """
        Estimate cost for entire graph.
        
        Returns:
            Dict with total costs and per-node estimates
        """
        costs = {}
        total_compute = 0
        total_memory = 0
        total_data_movement = 0
        total_transfer_time = 0
        total_queueing_delay = 0

        for node in graph.nodes():
            node_cost = self.operation_estimator.estimate_operation(node)
            costs[node.id] = node_cost

            total_compute += node_cost.compute_flops
            total_memory += node_cost.memory_bytes
            total_data_movement += node_cost.data_movement_bytes
            total_transfer_time += node_cost.transfer_time_ms
            total_queueing_delay += node_cost.queueing_delay_ms

        return {
            'per_node': costs,
            'total_compute_flops': total_compute,
            'total_memory_bytes': total_memory,
            'total_data_movement_bytes': total_data_movement,
            'total_transfer_time_ms': total_transfer_time,
            'total_queueing_delay_ms': total_queueing_delay,
            'mean_operational_intensity': (total_compute / total_memory
                                          if total_memory > 0 else 0),
        }
