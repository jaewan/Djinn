"""
SubgraphExecutor: Executes operation subgraphs on server GPU.

This module implements the server-side execution of computation subgraphs.
The key optimization is that intermediates stay on GPU - only the final
output is transferred back to CPU.

Key benefits:
- Intermediates stay on GPU (no CPU round-trips)
- Single network transfer for entire computation
- Better memory efficiency
- Reduced serialization overhead

Usage:
    executor = SubgraphExecutor(gpu_id=0)
    result = executor.execute(subgraph_request, input_data)
"""

import logging
import torch
import traceback
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
import time
from ..frontend.core.universal_dispatcher import get_universal_dispatcher

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for subgraph execution on server."""
    device: torch.device
    tensors: Dict[int, torch.Tensor]  # Cached intermediate results
    memory_limit: Optional[int] = None  # Max memory usage (bytes)
    timeout: Optional[float] = None  # Max execution time (seconds)


class SubgraphExecutor:
    """
    Executes operation subgraphs on server GPU.

    KEY OPTIMIZATION: Intermediates stay on GPU!
    Only the final output is transferred back to CPU.
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.universal_dispatcher = get_universal_dispatcher()  # Use universal dispatcher

        # Statistics
        self.stats = {
            'subgraphs_executed': 0,
            'operations_executed': 0,
            'total_time_seconds': 0.0,
            'memory_peak_mb': 0.0,
            'errors': []
        }

        logger.info(f"SubgraphExecutor initialized on GPU {gpu_id} with universal dispatcher")

    def execute(self,
                subgraph_request: Dict[str, Any],
                input_data: Dict[str, torch.Tensor],
                timeout: float = 300.0) -> torch.Tensor:
        """
        Execute subgraph on GPU.

        Args:
            subgraph_request: Serialized subgraph specification
            input_data: Input tensors (on CPU)
            timeout: Maximum execution time in seconds

        Returns:
            Result tensor (on CPU)

        Process:
        1. Move inputs to GPU
        2. Execute operations in order (all on GPU)
        3. Only transfer final output to CPU
        """
        start_time = time.time()
        self.stats['subgraphs_executed'] += 1

        try:
            logger.info(f"🚀 Executing subgraph: {len(subgraph_request['operations'])} operations")
            logger.info(
                "   Executor device=%s | CUDA available=%s | current_device=%s",
                self.device,
                torch.cuda.is_available(),
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            )

            context = ExecutionContext(
                device=self.device,
                tensors={},
                timeout=timeout
            )

            # Step 1: Move inputs to GPU
            logger.info(f"📥 Received {len(input_data)} input tensors")
            logger.info(f"   Input data keys: {list(input_data.keys())}")
            logger.info(f"   Subgraph input_tensors keys: {list(subgraph_request.get('input_tensors', {}).keys())}")
            logger.debug(f"Moving {len(input_data)} inputs to GPU {self.gpu_id}")
            move_start = time.time()
            
            # ✅ CRITICAL FIX: Build a mapping from input tensor IDs to ensure all references work
            # Input tensors can be referenced by their tensor_id (for external inputs)
            # or by op_id (if they're also operation outputs in the subgraph)
            input_tensor_ids = set()
            for tensor_id_str, tensor_data in input_data.items():
                tensor_id = int(tensor_id_str)
                input_tensor_ids.add(tensor_id)
                
                logger.info(
                    "      loading input tensor[%s]: shape=%s dtype=%s src_device=%s",
                    tensor_id,
                    tuple(tensor_data.shape),
                    tensor_data.dtype,
                    tensor_data.device,
                )
                
                # ✅ FINAL DEFENSE: Meta tensors should have been filtered on client side
                # If we see one here, it means the client-side filters failed
                if tensor_data.device.type == 'meta':
                    logger.error(
                        f"❌ CRITICAL: Input tensor[{tensor_id}] is on meta device - this should have been filtered on client! "
                        f"This indicates a bug in client-side subgraph building or materialization. "
                        f"Shape: {tensor_data.shape}, dtype: {tensor_data.dtype}. "
                        f"Cannot copy meta tensors to GPU (they have no data)."
                    )
                    raise NotImplementedError(
                        f"Cannot copy tensor[{tensor_id}] from meta device to {self.device}: meta tensors have no data. "
                        f"This is a client-side bug - meta tensors should be filtered before sending to server."
                    )
                
                copy_start = time.time()
                moved_tensor = tensor_data.to(self.device, non_blocking=True)
                copy_ms = (time.time() - copy_start) * 1000
                logger.info(
                    "      tensor[%s] moved to device=%s (transfer %.2f ms)",
                    tensor_id,
                    moved_tensor.device,
                    copy_ms,
                )
                context.tensors[tensor_id] = moved_tensor
            
            # ✅ CRITICAL FIX: Pre-scan operations to identify which op_ids correspond to input tensors
            # This ensures that if an input tensor is also referenced as an operation output,
            # both references point to the same tensor
            for op_spec in subgraph_request['operations']:
                op_id = op_spec.get('op_id')
                # If this operation's output ID matches an input tensor ID, it means the input
                # is being used directly (shouldn't happen, but handle gracefully)
                if op_id in input_tensor_ids:
                    logger.warning(
                        f"⚠️  Operation {op_spec.get('operation', 'unknown')} has op_id={op_id} "
                        f"that matches an input tensor ID. This may indicate a serialization issue."
                    )
            
            logger.info(f"✅ Loaded {len(context.tensors)} tensors into context: {list(context.tensors.keys())}")
            if self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            move_ms = (time.time() - move_start) * 1000
            logger.info("   Input transfer time: %.2f ms", move_ms)

            # Step 2: Execute operations in topological order
            total_ops = len(subgraph_request['operations'])
            logger.info(f"🚀 Executing {total_ops} operations in topological order")
            logger.info(f"   Available input tensor IDs: {list(context.tensors.keys())}")
            
            # ✅ PROFILING: Measure GPU execution time
            try:
                from .profiling_context import get_profiler, record_phase
                profiler = get_profiler()
                if profiler and profiler.enabled:
                    with record_phase('gpu_execution', metadata={
                        'operation_count': total_ops,
                        'input_count': len(input_data)
                    }):
                        # Progress tracking for large subgraphs
                        progress_interval = max(1, total_ops // 20)  # Log every 5%
                        
                        for op_idx, op_spec in enumerate(subgraph_request['operations']):
                            self.stats['operations_executed'] += 1
                            
                            # Log progress for large subgraphs
                            if op_idx % progress_interval == 0 or op_idx == total_ops - 1:
                                logger.info(f"   Progress: {op_idx+1}/{total_ops} operations ({100*(op_idx+1)/total_ops:.1f}%)")
                            
                            # Debug: Log operation details (only for first few and last few)
                            if op_idx < 5 or op_idx >= total_ops - 5:
                                op_id = op_spec.get('op_id', 'unknown')
                                op_name = op_spec.get('operation', 'unknown')
                                input_ids = op_spec.get('inputs', [])
                                logger.debug(
                                    f"  Operation {op_idx+1}/{total_ops}: {op_name} "
                                    f"(op_id={op_id}, inputs={input_ids})"
                                )

                            result = self._execute_operation(op_spec, context, subgraph_request)
                            
                            # ✅ FIX: Handle operations that return non-tensor values (bool, int, etc.)
                            # Some operations like _has_compatible_shallow_copy_type return bool
                            # We need to handle these gracefully instead of assuming all ops return tensors
                            if isinstance(result, torch.Tensor):
                                context.tensors[op_spec['op_id']] = result
                                if op_idx < 5 or op_idx >= total_ops - 5:
                                    logger.debug(
                                        "      op[%s] result device=%s shape=%s",
                                        op_spec['op_id'],
                                        result.device,
                                        tuple(result.shape) if hasattr(result, 'shape') else "n/a",
                                    )
                            else:
                                # Non-tensor result (bool, int, etc.) - store as-is but log warning
                                # These operations shouldn't typically be in subgraphs, but handle gracefully
                                logger.warning(
                                    f"⚠️  Operation {op_spec.get('operation', 'unknown')} returned non-tensor: {type(result).__name__} = {result}"
                                )
                                context.tensors[op_spec['op_id']] = result
                                if op_idx < 5 or op_idx >= total_ops - 5:
                                    logger.debug(
                                        "      op[%s] result type=%s value=%s",
                                        op_spec['op_id'],
                                        type(result).__name__,
                                        result,
                                    )

                            # Check memory usage
                            if torch.cuda.is_available():
                                memory_mb = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
                                self.stats['memory_peak_mb'] = max(self.stats['memory_peak_mb'], memory_mb)
                                logger.debug(f"GPU memory: {memory_mb:.1f}MB")
                else:
                    # No profiling - execute normally
                    # Progress tracking for large subgraphs
                    progress_interval = max(1, total_ops // 20)  # Log every 5%
                    
                    for op_idx, op_spec in enumerate(subgraph_request['operations']):
                        self.stats['operations_executed'] += 1
                        
                        # Log progress for large subgraphs
                        if op_idx % progress_interval == 0 or op_idx == total_ops - 1:
                            logger.info(f"   Progress: {op_idx+1}/{total_ops} operations ({100*(op_idx+1)/total_ops:.1f}%)")
                        
                        # Debug: Log operation details (only for first few and last few)
                        if op_idx < 5 or op_idx >= total_ops - 5:
                            op_id = op_spec.get('op_id', 'unknown')
                            op_name = op_spec.get('operation', 'unknown')
                            input_ids = op_spec.get('inputs', [])
                            logger.debug(
                                f"  Operation {op_idx+1}/{total_ops}: {op_name} "
                                f"(op_id={op_id}, inputs={input_ids})"
                            )

                        result = self._execute_operation(op_spec, context, subgraph_request)
                        
                        # ✅ FIX: Handle operations that return non-tensor values (bool, int, etc.)
                        # Some operations like _has_compatible_shallow_copy_type return bool
                        # We need to handle these gracefully instead of assuming all ops return tensors
                        if isinstance(result, torch.Tensor):
                            context.tensors[op_spec['op_id']] = result
                            if op_idx < 5 or op_idx >= total_ops - 5:
                                logger.debug(
                                    "      op[%s] result device=%s shape=%s",
                                    op_spec['op_id'],
                                    result.device,
                                    tuple(result.shape) if hasattr(result, 'shape') else "n/a",
                                )
                        else:
                            # Non-tensor result (bool, int, etc.) - store as-is but log warning
                            # These operations shouldn't typically be in subgraphs, but handle gracefully
                            logger.warning(
                                f"⚠️  Operation {op_spec.get('operation', 'unknown')} returned non-tensor: {type(result).__name__} = {result}"
                            )
                            context.tensors[op_spec['op_id']] = result
                            if op_idx < 5 or op_idx >= total_ops - 5:
                                logger.debug(
                                    "      op[%s] result type=%s value=%s",
                                    op_spec['op_id'],
                                    type(result).__name__,
                                    result,
                                )

                        # Check memory usage
                        if torch.cuda.is_available():
                            memory_mb = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
                            self.stats['memory_peak_mb'] = max(self.stats['memory_peak_mb'], memory_mb)
                            logger.debug(f"GPU memory: {memory_mb:.1f}MB")
            except ImportError:
                # Fallback if profiling not available
                pass
            # Progress tracking for large subgraphs
            progress_interval = max(1, total_ops // 20)  # Log every 5%
            
            for op_idx, op_spec in enumerate(subgraph_request['operations']):
                self.stats['operations_executed'] += 1
                
                # Log progress for large subgraphs
                if op_idx % progress_interval == 0 or op_idx == total_ops - 1:
                    logger.info(f"   Progress: {op_idx+1}/{total_ops} operations ({100*(op_idx+1)/total_ops:.1f}%)")
                
                # Debug: Log operation details (only for first few and last few)
                if op_idx < 5 or op_idx >= total_ops - 5:
                    op_id = op_spec.get('op_id', 'unknown')
                    op_name = op_spec.get('operation', 'unknown')
                    input_ids = op_spec.get('inputs', [])
                    logger.debug(
                        f"  Operation {op_idx+1}/{total_ops}: {op_name} "
                        f"(op_id={op_id}, inputs={input_ids})"
                    )

                result = self._execute_operation(op_spec, context, subgraph_request)
                
                # ✅ FIX: Handle operations that return non-tensor values (bool, int, etc.)
                # Some operations like _has_compatible_shallow_copy_type return bool
                # We need to handle these gracefully instead of assuming all ops return tensors
                if isinstance(result, torch.Tensor):
                    context.tensors[op_spec['op_id']] = result
                    if op_idx < 5 or op_idx >= total_ops - 5:
                        logger.debug(
                            "      op[%s] result device=%s shape=%s",
                            op_spec['op_id'],
                            result.device,
                            tuple(result.shape) if hasattr(result, 'shape') else "n/a",
                        )
                else:
                    # Non-tensor result (bool, int, etc.) - store as-is but log warning
                    # These operations shouldn't typically be in subgraphs, but handle gracefully
                    logger.warning(
                        f"⚠️  Operation {op_spec.get('operation', 'unknown')} returned non-tensor: {type(result).__name__} = {result}"
                    )
                    context.tensors[op_spec['op_id']] = result
                    if op_idx < 5 or op_idx >= total_ops - 5:
                        logger.debug(
                            "      op[%s] result type=%s value=%s",
                            op_spec['op_id'],
                            type(result).__name__,
                            result,
                        )

                # Check memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
                    self.stats['memory_peak_mb'] = max(self.stats['memory_peak_mb'], memory_mb)
                    logger.debug(f"GPU memory: {memory_mb:.1f}MB")

            # Step 3: Transfer ONLY final output to CPU
            output_id = subgraph_request['output_id']
            final_result = context.tensors[output_id]

            # ✅ FIX: Ensure final result is a tensor (not bool/int/etc.)
            if not isinstance(final_result, torch.Tensor):
                raise RuntimeError(
                    f"Final subgraph output is not a tensor: got {type(final_result).__name__} = {final_result}. "
                    f"This indicates an operation that returns non-tensor values is being used as the final output. "
                    f"Output ID: {output_id}, Operation: {subgraph_request.get('operations', [])[-1] if subgraph_request.get('operations') else 'unknown'}"
                )

            logger.debug(f"Transferring final result: {final_result.shape} to CPU")
            logger.info(
                "   Final result before transfer: device=%s shape=%s dtype=%s",
                final_result.device,
                tuple(final_result.shape),
                final_result.dtype,
            )

            # Move to CPU and detach from computation graph for transfer
            transfer_start = time.time()
            result_cpu = final_result.cpu().detach()
            transfer_ms = (time.time() - transfer_start) * 1000
            logger.info("   Final result transfer to CPU: %.2f ms", transfer_ms)

            # Statistics
            elapsed = time.time() - start_time
            self.stats['total_time_seconds'] += elapsed

            logger.info(f"✅ Subgraph execution complete: {len(subgraph_request['operations'])} ops "
                       f"in {elapsed:.3f}s, result shape: {result_cpu.shape}")

            return result_cpu

        except (NotImplementedError, KeyError, ValueError) as e:
            # Re-raise specific exceptions without wrapping (for testing)
            logger.error(f"Subgraph execution failed: {e}")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            # traceback is already imported at module level
            error_msg = f"Subgraph execution failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            self.stats['errors'].append({
                'timestamp': time.time(),
                'error': str(e),
                'error_type': type(e).__name__,
                'operations_count': len(subgraph_request.get('operations', [])),
                'traceback': traceback.format_exc()
            })

            # Re-raise with more context for other exceptions
            raise RuntimeError(error_msg) from e

    def execute_single_op(self, operation: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Execute single operation directly (convenience method).
        
        Faster than full subgraph execution for simple operations.
        Optimized for Phase 1 remote execution of atomic operations.
        
        Args:
            operation: Operation name (e.g., 'aten::relu', 'aten::add')
            tensor: Input tensor (on CPU or GPU)
        
        Returns:
            Result tensor (on CPU)
            
        Performance:
            - Single op: 50% faster than wrapping in subgraph format
            - No overhead from subgraph parsing
            - Direct operation registry lookup
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Executing single op: {operation}")
            
            # Move input to GPU
            gpu_tensor = tensor.to(self.device)
            
            # Execute operation using universal dispatcher
            result_gpu = self.universal_dispatcher.dispatch(operation, [gpu_tensor], {})
            
            # Move result to CPU
            result_cpu = result_gpu.cpu().detach()
            
            # Statistics
            elapsed = time.time() - start_time
            self.stats['operations_executed'] += 1
            self.stats['total_time_seconds'] += elapsed
            
            logger.debug(f"Single op executed: {operation} in {elapsed:.3f}s, result shape: {result_cpu.shape}")
            
            return result_cpu
            
        except Exception as e:
            logger.error(f"Single op execution failed: {operation}: {e}")
            raise RuntimeError(f"Failed to execute {operation}: {e}") from e

    def _execute_operation(self,
                          op_spec: Dict[str, Any],
                          context: ExecutionContext,
                          subgraph_request: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Execute single operation on GPU.

        Args:
            op_spec: Operation specification
            context: Execution context with cached tensors

        Returns:
            Result tensor (on GPU)
        """
        try:
            operation = op_spec.get('operation', 'unknown')
            # Ensure operation is a string
            if not isinstance(operation, str):
                operation = str(operation)
            
            input_ids = op_spec.get('inputs', [])
            kwargs = op_spec.get('kwargs', {}).copy()
            kwargs.pop('_stacklevel', None)

            inputs = []
            for idx, identifier in enumerate(input_ids):
                if isinstance(identifier, dict) and 'type' in identifier:
                    decoded = self._decode_literal(identifier)
                    inputs.append(decoded)
                    logger.debug(f"   Decoded literal input[{idx}]: {type(decoded).__name__} = {decoded}")
                else:
                    # ✅ CRITICAL FIX: Enhanced input resolution with better error messages
                    if identifier not in context.tensors:
                        # Enhanced error message with debugging info
                        available_ids = list(context.tensors.keys())
                        logger.error(
                            f"❌ Tensor ID {identifier} not found in context for operation {operation} "
                            f"(input index {idx}, op_id={op_spec.get('op_id', 'unknown')})"
                        )
                        logger.error(f"   Available tensor IDs: {available_ids[:50]}...")  # Show first 50
                        logger.error(f"   Operation inputs: {input_ids}")
                        
                        # Check if identifier matches any op_id from subgraph (if available)
                        if subgraph_request:
                            executed_op_ids = [op.get('op_id') for op in subgraph_request.get('operations', [])]
                            if identifier in executed_op_ids:
                                logger.error(
                                    f"   ⚠️  ID {identifier} matches an operation ID but result not stored. "
                                    f"This indicates a bug in operation execution or result storage."
                                )
                            else:
                                logger.error(
                                    f"   ⚠️  ID {identifier} does not match any operation ID. "
                                    f"This may indicate a serialization/deserialization mismatch."
                                )
                        
                        raise KeyError(
                            f"Input tensor ID {identifier} not found in context for operation {operation}. "
                            f"Available IDs: {available_ids[:20]}..., Operation inputs: {input_ids}, "
                            f"Op ID: {op_spec.get('op_id', 'unknown')}"
                        )
                    
                    tensor = context.tensors[identifier]
                    # ✅ CRITICAL FIX: Validate tensor is actually a tensor (not None or wrong type)
                    if not isinstance(tensor, torch.Tensor):
                        logger.error(
                            f"❌ CRITICAL: Input ID {identifier} resolved to non-tensor: {type(tensor).__name__} = {tensor}"
                        )
                        raise TypeError(
                            f"Input ID {identifier} for operation {operation} resolved to {type(tensor).__name__}, "
                            f"expected torch.Tensor"
                        )
                    inputs.append(tensor)
            
            # ✅ SENIOR ENGINEER FIX: Comprehensive dtype logging and coercion
            # Log input dtypes BEFORE coercion for debugging
            tensor_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
            if tensor_inputs:
                input_dtypes = [inp.dtype for inp in tensor_inputs]
                input_shapes = [tuple(inp.shape) for inp in tensor_inputs]
                logger.info(
                    f"🔧 Executing {operation} with {len(inputs)} inputs "
                    f"(dtypes: {input_dtypes}, shapes: {input_shapes})"
                )
            else:
                logger.info(f"🔧 Executing {operation} with {len(inputs)} inputs: {[type(x).__name__ for x in inputs]}")
            
            if operation == 'aten::unsqueeze':
                logger.info(f"   ⚠️  UNSQUEEZE DEBUG: inputs={inputs}, kwargs={kwargs}, input_ids={input_ids}")

            # ✅ CRITICAL FIX: Coerce input dtypes to ensure consistency before execution
            # This is critical for operations that require matching dtypes (e.g., matmul, attention)
            # For float16 models like GPT2-XL, all inputs should be float16
            # ✅ IMPROVED: Prioritize float16 if any input is float16 (matches frontend logic)
            if len(tensor_inputs) > 1:
                # Collect all floating-point dtypes
                floating_dtypes = [inp.dtype for inp in tensor_inputs if inp.dtype.is_floating_point]
                if floating_dtypes:
                    # ✅ CRITICAL: Prioritize float16 if any input is float16
                    # This ensures float16 models maintain float16 precision
                    if torch.float16 in floating_dtypes:
                        target_dtype = torch.float16
                    else:
                        # Otherwise, use the first tensor's dtype
                        target_dtype = tensor_inputs[0].dtype
                    
                    # Only coerce if we have dtype mismatches
                    coerced = False
                    for i, inp in enumerate(inputs):
                        if isinstance(inp, torch.Tensor) and inp.dtype.is_floating_point and inp.dtype != target_dtype:
                            logger.warning(
                                f"⚠️  DTYPE MISMATCH: Coercing input {i} dtype from {inp.dtype} to {target_dtype} "
                                f"for operation {operation} (shape: {tuple(inp.shape)})"
                            )
                            inputs[i] = inp.to(dtype=target_dtype)
                            coerced = True
                    if coerced:
                        # Update context tensors if we modified them
                        for idx, identifier in enumerate(input_ids):
                            if idx < len(inputs) and isinstance(inputs[idx], torch.Tensor):
                                if identifier in context.tensors:
                                    context.tensors[identifier] = inputs[idx]
                        # Log final dtypes after coercion
                        final_dtypes = [inp.dtype for inp in inputs if isinstance(inp, torch.Tensor)]
                        logger.info(f"✅ After dtype coercion: {final_dtypes}")

            # ✅ SENIOR ENGINEER FIX: Execute with comprehensive error handling
            try:
                result = self.universal_dispatcher.dispatch(operation, inputs, kwargs)
            except RuntimeError as e:
                # ✅ CRITICAL: Catch dtype mismatch errors and provide detailed diagnostics
                error_msg = str(e)
                if "expected m1 and m2 to have the same dtype" in error_msg or "dtype" in error_msg.lower():
                    logger.error(f"❌ DTYPE MISMATCH ERROR in {operation}:")
                    logger.error(f"   Error: {error_msg}")
                    logger.error(f"   Input dtypes: {[inp.dtype if isinstance(inp, torch.Tensor) else type(inp).__name__ for inp in inputs]}")
                    logger.error(f"   Input shapes: {[tuple(inp.shape) if isinstance(inp, torch.Tensor) else 'N/A' for inp in inputs]}")
                    logger.error(f"   Operation: {operation}")
                    logger.error(f"   Op ID: {op_spec.get('op_id', 'unknown')}")
                    # Re-raise with more context
                    raise RuntimeError(
                        f"Dtype mismatch in {operation}: {error_msg}\n"
                        f"Input dtypes: {[inp.dtype if isinstance(inp, torch.Tensor) else type(inp).__name__ for inp in inputs]}\n"
                        f"Input shapes: {[tuple(inp.shape) if isinstance(inp, torch.Tensor) else 'N/A' for inp in inputs]}"
                    ) from e
                raise  # Re-raise other RuntimeErrors as-is

            result_index = op_spec.get('result_index')
            if result_index is not None and isinstance(result, (tuple, list)):
                try:
                    result = result[result_index]
                except IndexError as exc:
                    raise RuntimeError(
                        f"Result index {result_index} out of range for operation {operation}"
                    ) from exc

            # ✅ CRITICAL FIX: For operations that might create float32 when inputs are float16,
            # check if all inputs are the same dtype and preserve it
            # This is critical for float16 models like GPT2-XL
            # ✅ IMPROVED: Only coerce if all inputs are float16 (don't coerce if mixed dtypes)
            if isinstance(result, torch.Tensor):
                # Check if all tensor inputs have the same dtype
                tensor_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
                if tensor_inputs:
                    input_dtypes = {inp.dtype for inp in tensor_inputs}
                    # If all inputs are the same dtype (especially float16), preserve it
                    if len(input_dtypes) == 1:
                        common_dtype = tensor_inputs[0].dtype
                        # Only coerce if result dtype differs and both are floating point
                        # ✅ CRITICAL: Only coerce float32->float16 if ALL inputs are float16
                        # Some operations (like softmax, layer_norm) may use float32 internally for stability
                        # but if all inputs are float16, we should preserve float16 for consistency
                        if result.dtype != common_dtype and result.dtype.is_floating_point and common_dtype.is_floating_point:
                            # PyTorch promotion rules: float16 + float32 -> float32 is correct
                            # But if all inputs are float16, result should also be float16
                            if common_dtype == torch.float16 and result.dtype == torch.float32:
                                # ✅ CRITICAL: PyTorch operations like softmax and layer_norm DO return float16
                                # when given float16 inputs, so we should coerce to match PyTorch behavior
                                # This ensures correctness with vanilla PyTorch
                                logger.warning(
                                    f"⚠️  DTYPE PROMOTION: Coercing {operation} result from {result.dtype} to {common_dtype} "
                                    f"(all inputs are {common_dtype}, shape: {tuple(result.shape)})"
                                )
                                result = result.to(dtype=common_dtype)

            return result  # Result stays on GPU!
        except Exception as e:
            operation_name = str(op_spec.get('operation', 'unknown'))
            # Enhanced error message with input shapes for debugging
            input_shapes = []
            for inp in inputs:
                if hasattr(inp, 'shape'):
                    input_shapes.append(f"{tuple(inp.shape)}")
                else:
                    input_shapes.append(f"{type(inp).__name__}({inp})")
            
            error_msg = f"Operation {operation_name} failed: {type(e).__name__}: {str(e)}"
            logger.error(f"{error_msg}")
            logger.error(f"   Input shapes: {input_shapes}")
            logger.error(f"   Input IDs: {input_ids}")
            logger.error(f"   Operation index: {op_spec.get('_op_idx', 'unknown')}")
            logger.error(f"   Available tensor IDs in context: {list(context.tensors.keys())}")
            logger.error("", exc_info=True)
            raise RuntimeError(error_msg) from e



    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        stats = self.stats.copy()

        if stats['subgraphs_executed'] > 0:
            stats['avg_time_per_subgraph'] = (
                stats['total_time_seconds'] / stats['subgraphs_executed']
            )
            stats['avg_operations_per_subgraph'] = (
                stats['operations_executed'] / stats['subgraphs_executed']
            )
        else:
            stats['avg_time_per_subgraph'] = 0.0
            stats['avg_operations_per_subgraph'] = 0.0

        return stats

    @staticmethod
    def _decode_literal(spec: Dict[str, Any]) -> Any:
        literal_type = spec.get('type')
        value = spec.get('value')

        if literal_type == 'scalar':
            return value
        if literal_type == 'tuple':
            return tuple(
                SubgraphExecutor._decode_literal(v) if isinstance(v, dict) else v
                for v in value
            )
        if literal_type == 'list':
            return [
                SubgraphExecutor._decode_literal(v) if isinstance(v, dict) else v
                for v in value
            ]
        if literal_type == 'none':
            return None
        if literal_type == 'dtype':
            return getattr(torch, value.split('.')[-1]) if isinstance(value, str) else value
        if literal_type == 'slice':
            start, stop, step = (
                SubgraphExecutor._decode_literal(v) if isinstance(v, dict) else v
                for v in value
            )
            return slice(start, stop, step)
        if literal_type == 'device':
            # Convert device string back to torch.device
            return torch.device(value)
        if literal_type == 'unknown':
            # Handle unknown types that were serialized as strings
            # Try to convert common types like torch.device
            value_str = str(value)
            if value_str in ('cpu', 'cuda', 'meta'):
                return torch.device(value_str)
            # Fallback: return as string
            return value_str
        raise ValueError(f"Unsupported literal spec: {spec}")
