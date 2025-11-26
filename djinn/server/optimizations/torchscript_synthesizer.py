"""
TorchScript Synthesizer

Converts LazyTensor operation graphs to executable TorchScript modules
for TensorRT compilation and optimization.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
import io

logger = logging.getLogger(__name__)


class OperationBlock(nn.Module):
    """
    Dynamically created module from operation list.

    This is the bridge between LazyTensor ops and TorchScript.
    """

    def __init__(self, operations: List[Dict], block_id: str):
        super().__init__()
        self.operations = operations
        self.block_id = block_id

        # Extract parameters (if any)
        self._register_parameters()

    def _register_parameters(self):
        """Register any parameters found in operations."""
        for op in self.operations:
            # Check if operation uses parameters
            if 'weight' in op.get('kwargs', {}):
                weight = op['kwargs']['weight']
                if isinstance(weight, torch.Tensor):
                    self.register_parameter(
                        f"weight_{op['op_id']}",
                        nn.Parameter(weight)
                    )

    def forward(self, *inputs):
        """
        Forward pass: execute operations in sequence using proper tensor flow.

        The key insight: instead of manually dispatching operations (which TorchScript
        can't trace), we create a computational graph where tensors flow naturally
        through PyTorch operations that TorchScript can trace.
        """
        # Initialize tensor storage with input tensors
        tensors = list(inputs)  # Start with input tensors as a list

        # Execute operations in topological order
        # Each operation transforms tensors and stores result in tensors list
        for op_dict in self.operations:
            op_inputs = []
            for inp_ref in op_dict['inputs']:
                if isinstance(inp_ref, int):
                    # Integer reference to tensor in our storage
                    if 0 <= inp_ref < len(tensors):
                        op_inputs.append(tensors[inp_ref])
                    else:
                        raise ValueError(f"Input index {inp_ref} out of range")
                elif isinstance(inp_ref, str) and inp_ref in self._parameters:
                    # Parameter reference
                    op_inputs.append(self._parameters[inp_ref])
                elif isinstance(inp_ref, str) and inp_ref in self._buffers:
                    # Buffer reference
                    op_inputs.append(self._buffers[inp_ref])
                else:
                    # Pass through other types (scalars, etc.)
                    op_inputs.append(inp_ref)

            # Execute operation using PyTorch's native functions (traceable!)
            result = self._execute_traceable_operation(
                op_dict['operation'],
                op_inputs,
                op_dict.get('kwargs', {})
            )

            # Store result - operations can reference it by its index
            tensors.append(result)

        # Return final output (last operation's result)
        return tensors[-1]

    def _execute_traceable_operation(self, operation: str, inputs: List, kwargs: Dict) -> torch.Tensor:
        """
        Execute operation using PyTorch's traceable functions.

        This is the key: instead of using our custom dispatcher (which TorchScript
        can't trace), we use PyTorch's native functions that TorchScript understands.
        """
        # Map operation names to PyTorch functions
        op_name = operation.replace('aten::', '')

        # Handle common operations with their PyTorch equivalents
        if op_name == 'relu':
            return torch.nn.functional.relu(inputs[0], **kwargs)
        elif op_name == 'sigmoid':
            return torch.sigmoid(inputs[0])
        elif op_name == 'tanh':
            return torch.tanh(inputs[0])
        elif op_name == 'gelu':
            return torch.nn.functional.gelu(inputs[0], **kwargs)
        elif op_name == 'silu':
            return torch.nn.functional.silu(inputs[0])
        elif op_name == 'linear':
            return torch.nn.functional.linear(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None, **kwargs)
        elif op_name == 'matmul':
            return torch.matmul(inputs[0], inputs[1])
        elif op_name == 'add':
            return torch.add(inputs[0], inputs[1], **kwargs)
        elif op_name == 'mul':
            return torch.mul(inputs[0], inputs[1])
        elif op_name == 'conv2d':
            return torch.nn.functional.conv2d(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None, **kwargs)
        elif op_name == 'batch_norm':
            return torch.nn.functional.batch_norm(inputs[0], inputs[1], inputs[2], **kwargs)
        elif op_name == 'layer_norm':
            return torch.nn.functional.layer_norm(inputs[0], inputs[1], **kwargs)
        else:
            # Fallback: try torch.ops if available (less traceable but better than nothing)
            try:
                if hasattr(torch.ops.aten, op_name):
                    return getattr(torch.ops.aten, op_name)(*inputs, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                logger.warning(f"Operation {operation} not directly supported by TorchScript: {e}")
                # Last resort: use our dispatcher but this won't be traceable
                from djinn.frontend.core.universal_dispatcher import get_universal_dispatcher
                dispatcher = get_universal_dispatcher()
                return dispatcher.dispatch(operation, inputs, kwargs)


class TorchScriptSynthesizer:
    """
    Synthesizes TorchScript modules from operation lists.

    Workflow:
    1. Group operations into blocks (semantic or size-based)
    2. Create OperationBlock for each group
    3. Trace to TorchScript
    4. Compile to TensorRT (optional)
    """

    def __init__(self):
        self.block_cache: Dict[str, torch.jit.ScriptModule] = {}
        self.tensorrt_cache: Dict[str, Any] = {}

    def synthesize_block(
        self,
        operations: List[Dict],
        block_id: str,
        sample_inputs: List[torch.Tensor]
    ) -> torch.jit.ScriptModule:
        """
        Synthesize TorchScript module from operations.

        Args:
            operations: List of operation specs
            block_id: Unique identifier for caching
            sample_inputs: Sample inputs for tracing

        Returns:
            Traced TorchScript module
        """
        # Check cache
        if block_id in self.block_cache:
            return self.block_cache[block_id]

        try:
            # Create module
            module = OperationBlock(operations, block_id)

            # Trace to TorchScript
            traced = torch.jit.trace(module, sample_inputs)

            # Cache
            self.block_cache[block_id] = traced

            logger.info(f"Synthesized TorchScript block: {block_id} ({len(operations)} ops)")
            return traced

        except Exception as e:
            logger.warning(f"TorchScript synthesis failed: {e}")
            return None

    def compile_tensorrt(
        self,
        torchscript_module: torch.jit.ScriptModule,
        block_id: str,
        sample_inputs: List[torch.Tensor],
        use_fp16: bool = True
    ) -> Optional[Any]:
        """
        Compile TorchScript module to TensorRT.

        Args:
            torchscript_module: TorchScript module to compile
            block_id: Block identifier
            sample_inputs: Sample inputs for compilation
            use_fp16: Enable FP16 optimization

        Returns:
            Compiled TensorRT module or None if compilation failed
        """
        # Check cache
        if block_id in self.tensorrt_cache:
            return self.tensorrt_cache[block_id]

        try:
            # Try torch2trt (lighter weight)
            from torch2trt import torch2trt

            trt_module = torch2trt(
                torchscript_module,
                sample_inputs,
                fp16_mode=use_fp16,
                max_workspace_size=1 << 25  # 32MB
            )

            # Cache
            self.tensorrt_cache[block_id] = trt_module

            logger.info(f"Compiled TensorRT module: {block_id} (FP16={use_fp16})")
            return trt_module

        except ImportError:
            logger.warning("torch2trt not available, using optimized TorchScript")

            if use_fp16:
                # Convert to FP16
                half_module = torchscript_module.half()
                self.tensorrt_cache[block_id] = half_module
                return half_module

            return torchscript_module

        except Exception as e:
            logger.warning(f"TensorRT compilation failed: {e}")
            return None

    def get_cached_module(self, block_id: str) -> Optional[torch.jit.ScriptModule]:
        """Get cached TorchScript module."""
        return self.block_cache.get(block_id)

    def get_cached_tensorrt(self, block_id: str) -> Optional[Any]:
        """Get cached TensorRT module."""
        return self.tensorrt_cache.get(block_id)


# Global synthesizer
_synthesizer = TorchScriptSynthesizer()


def get_synthesizer() -> TorchScriptSynthesizer:
    return _synthesizer
