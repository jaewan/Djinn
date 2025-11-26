"""
Resilient Model Handler: Production-grade error handling with automatic recovery.

Features:
- Automatic retry with exponential backoff
- Circuit breaker for repeated failures
- Graceful fallback to graph execution
- Comprehensive error tracking

This is part of the redesign plan (Week 2).
"""

import asyncio
import logging
import traceback
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Optional

import torch

from .profile_registry_client import ProfileRegistryClient

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Error types for classification."""
    MODEL_NOT_FOUND = "model_not_found"
    OOM = "out_of_memory"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"
    SECURITY_ERROR = "security_error"


class ResilientModelHandler:
    """
    Production-grade error handling with automatic recovery.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for repeated failures
    - Graceful fallback to graph execution
    - Comprehensive error tracking
    """
    
    def __init__(self, gpu_id: int = 0):
        from .memory_aware_model_cache import MemoryAwareModelCache
        from .model_security import ModelSecurityValidator
        
        self.model_cache = MemoryAwareModelCache(device=f'cuda:{gpu_id}')
        self.security_validator = ModelSecurityValidator()
        
        # Fallback to graph execution
        from .subgraph_executor import SubgraphExecutor
        self.graph_executor = SubgraphExecutor(gpu_id)
        
        # Error tracking for circuit breaker
        self.error_counts = defaultdict(int)
        self.error_threshold = 5
        self.circuit_breaker_open = False
        
        # Retry configuration
        self.max_retries = 3
        self.timeout_seconds = 300.0  # ✅ FIX: Increased from 30s to 5min for model execution
        
        # Phase 0: Profile registry client (no-op for now)
        self.profile_registry_client = ProfileRegistryClient()
        
        logger.info(f"ResilientModelHandler initialized (gpu_id={gpu_id})")
    
    async def handle_request(self, request: Dict) -> Dict:
        """
        Handle request with comprehensive error recovery.
        
        Args:
            request: Request dictionary with 'type' field
        
        Returns:
            Response dictionary with 'status' and result/error
        """
        
        # Circuit breaker check
        if self.circuit_breaker_open:
            return self._circuit_breaker_response()
        
        request_type = request.get('type')
        
        try:
            if request_type == 'EXECUTE_MODEL':
                return await self._execute_with_recovery(request)
            elif request_type == 'REGISTER_MODEL':
                return await self._register_with_recovery(request)
            else:
                # Unknown request - fall back to graph
                return await self._fallback_to_graph(request)
                
        except Exception as e:
            return self._handle_fatal_error(e, request)
    
    async def _execute_with_recovery(self, request: Dict) -> Dict:
        """
        Execute with multiple recovery strategies.
        
        Implements:
        - Retry with exponential backoff
        - OOM recovery with cache clearing
        - Fallback to graph execution
        """
        
        fingerprint = request['fingerprint']
        
        for attempt in range(self.max_retries):
            try:
                # Validate inputs (security)
                self.security_validator.validate_inputs(request['inputs'])
                
                # Try model cache execution
                result = await asyncio.wait_for(
                    self._execute_model(request),
                    timeout=self.timeout_seconds
                )
                
                # Success - reset error count
                self.error_counts[fingerprint] = 0
                return result
                
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return self._timeout_response(fingerprint)
                
            except ValueError as e:
                if 'not found' in str(e):
                    # Model not registered - log and request registration
                    logger.warning(f"Model {fingerprint} not found in cache. Available models: {list(self.model_cache.models.keys())}")
                    return self._request_registration_response(fingerprint)
                raise
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # OOM - clear cache and retry
                    if self.model_cache.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    if attempt < self.max_retries - 1:
                        continue
                    # Final attempt - fall back to graph
                    return await self._fallback_to_graph(request)
                raise
                
            except Exception as e:
                # Track errors
                self.error_counts[fingerprint] += 1
                logger.error(f"Model cache execution failed (attempt {attempt+1}/{self.max_retries}): {e}")
                import traceback
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                
                # Check circuit breaker
                if self.error_counts[fingerprint] > self.error_threshold:
                    self.circuit_breaker_open = True
                    logger.error(f"Circuit breaker opened for {fingerprint}")
                
                # Last resort - graph execution
                if attempt == self.max_retries - 1:
                    logger.warning(f"All retries exhausted, falling back to graph execution")
                    return await self._fallback_to_graph(request)
        
        # Should not reach here, but handle gracefully
        return self._error_response("Max retries exceeded", ErrorType.EXECUTION_FAILED)
    
    async def _register_with_recovery(self, request: Dict) -> Dict:
        """
        Register model with security validation.
        
        Returns:
            Success or error response
        """
        
        try:
            # Security validation
            self.security_validator.validate_model_registration(
                fingerprint=request['fingerprint'],
                architecture_data=request.get('architecture_data'),
                state_dict=request.get('uncached_weights', {})
            )
            
            # Register model
            self.model_cache.register_model(
                fingerprint=request['fingerprint'],
                descriptor=request['descriptor'],
                weight_ids=request['weight_ids'],
                uncached_weights=request.get('uncached_weights', {}),
                architecture_data=request.get('architecture_data')
            )
            
            return {
                'status': 'success',
                'fingerprint': request['fingerprint'],
                'message': 'Model registered successfully'
            }
            
        except Exception as e:
            # Check if it's a security error
            from .model_security import SecurityError
            if isinstance(e, SecurityError):
                return {
                    'status': 'error',
                    'error_type': ErrorType.SECURITY_ERROR.value,
                    'message': str(e)
                }
            else:
                logger.error(f"Model registration failed: {e}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                return {
                    'status': 'error',
                    'error_type': ErrorType.EXECUTION_FAILED.value,
                    'message': str(e)
                }
    
    async def _init_model(self, request: Dict) -> Dict:
        """
        Initialize (warmup) a registered model.
        
        Returns:
            Success or error response
        """
        try:
            fingerprint = request['fingerprint']
            success = self.model_cache.init_model(fingerprint)
            
            if success:
                return {
                    'status': 'success',
                    'fingerprint': fingerprint,
                    'message': 'Model initialized successfully'
                }
            else:
                return {
                    'status': 'error',
                    'error_type': ErrorType.EXECUTION_FAILED.value,
                    'message': f'Model initialization failed for {fingerprint}'
                }
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'error_type': ErrorType.EXECUTION_FAILED.value,
                'message': str(e)
            }
    
    async def _execute_model(self, request: Dict) -> Dict:
        """
        Execute using model cache (synchronous operation wrapped in async).
        
        This runs the actual model execution in a thread pool to avoid blocking.
        """
        
        try:
            from .profiling_context import ProfilingContext, get_profiler, record_phase, set_profiler
            
            # Create a new profiler for this request (server-side)
            server_profiler = ProfilingContext(enabled=True)
            server_profiler.start()
            set_profiler(server_profiler)
            
            # Measure request handling
            import time
            execute_start = time.perf_counter()
            logger.info(f"🔍 [DIAGNOSTIC] Starting model cache execution (fingerprint: {request['fingerprint'][:16]}...)")
            
            with record_phase('model_cache_request_handling'):
                # Deserialize inputs before execution
                deserialized_inputs = self._deserialize_inputs(request['inputs'])

                # Execute directly - GPU operations release GIL, so they don't block the event loop
                # No need for thread pool overhead!
                # NOTE: For multi-tenancy support later, we may need to reintroduce thread pool
                # to isolate concurrent requests and prevent one slow request from blocking others.
                result = self.model_cache.execute(
                    fingerprint=request['fingerprint'],
                    inputs=deserialized_inputs,
                    hints=request.get('hints', {})
                )
            
            execute_time = (time.perf_counter() - execute_start) * 1000
            logger.info(f"🔍 [DIAGNOSTIC] Model cache execute() completed: {execute_time:.1f}ms")
            
            # Serialize result
            serialize_start = time.perf_counter()
            with record_phase('model_cache_result_serialization'):
                serialized_result = self._serialize_tensor(result)

            serialize_time = (time.perf_counter() - serialize_start) * 1000
            logger.info(f"🔍 [DIAGNOSTIC] Result serialization: {serialize_time:.1f}ms")
            
            # Get server-side phases
            server_phases = server_profiler.get_phase_dict()

            # Get model cache stats (including phase detection and OOM events)
            cache_stats = self.model_cache.get_stats()
            memory_status = self.model_cache.get_memory_status()

            # Build telemetry payload (Phase 0 logging)
            telemetry = {
                'profile_id': request.get('profile_id'),
                'fingerprint': request['fingerprint'],
                'execution_time_ms': execute_time,
                'input_shapes': self._extract_input_shapes(request.get('inputs', {})),
                'memory_status': memory_status,
                'cache_stats': {
                    'oom_events': cache_stats.get('oom_events', 0),
                    'phase_detections': cache_stats.get('phase_detections', {}),
                    'phase_switches': cache_stats.get('phase_switches', 0),
                    'evictions': cache_stats.get('evictions', 0),
                },
                'server_phases': server_phases,
            }

            # Log telemetry summary at INFO, full payload at DEBUG
            logger.info(
                f"📡 TELEMETRY: profile_id={telemetry.get('profile_id')}, "
                f"exec_time={telemetry['execution_time_ms']:.1f}ms, "
                f"fingerprint={telemetry['fingerprint'][:16]}..."
            )
            logger.debug(f"📡 TELEMETRY (full payload): {telemetry}")
            self.profile_registry_client.record_telemetry(telemetry)
            
            # ✅ PHASE 1: Shadow mode - query registry and log what we WOULD have done
            self._run_shadow_mode_analysis(request, telemetry)

            return {
                'status': 'success',
                'result': serialized_result,
                'execution_path': 'model_cache',
                'memory_status': memory_status,
                'server_phases': server_phases,  # Include server-side profiling
                'cache_stats': telemetry['cache_stats'],
                'telemetry': telemetry
            }
        except Exception as e:
            logger.error(f"Model cache execution failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
    
    async def _fallback_to_graph(self, request: Dict) -> Dict:
        """
        Fall back to graph execution.
        
        This provides backward compatibility when model cache fails.
        Note: This fallback requires the client to send a subgraph, which
        the model cache execution path doesn't provide. So this is mainly
        for error handling - the client should handle fallback.
        """
        
        # Model cache execution doesn't include subgraph, so we can't fallback here
        # The client should handle fallback to graph execution
        return {
            'status': 'error',
            'error_type': ErrorType.EXECUTION_FAILED.value,
            'message': 'Model cache execution failed. Client should fallback to graph execution.',
            'fallback_required': True
        }
    
    def _request_registration_response(self, fingerprint: str) -> Dict:
        """Response requesting model registration."""
        from .error_responses import ErrorResponseBuilder
        return ErrorResponseBuilder.model_not_found(fingerprint, required_action='register_model')
    
    def _timeout_response(self, fingerprint: str) -> Dict:
        """Response for timeout errors."""
        from .error_responses import ErrorResponseBuilder
        return ErrorResponseBuilder.timeout(fingerprint, self.timeout_seconds)
    
    def _circuit_breaker_response(self) -> Dict:
        """Response when circuit breaker is open."""
        from .error_responses import ErrorResponseBuilder
        return ErrorResponseBuilder.circuit_breaker_open(retry_after_seconds=60)
    
    def _handle_fatal_error(self, e: Exception, request: Dict) -> Dict:
        """Handle fatal errors."""
        logger.error(f"Fatal error handling request: {e}")
        logger.error(traceback.format_exc())
        
        from .error_responses import ErrorResponseBuilder
        operation = request.get('operation', 'unknown')
        return ErrorResponseBuilder.execution_failed(
            operation=operation,
            reason=str(e),
            traceback=traceback.format_exc()
        )
    
    def _error_response(self, message: str, error_type: ErrorType) -> Dict:
        """Generic error response."""
        from .error_responses import ErrorResponseBuilder, ErrorCode
        
        # Map internal ErrorType to ErrorCode
        error_code_map = {
            ErrorType.MODEL_NOT_FOUND: ErrorCode.MODEL_NOT_FOUND,
            ErrorType.EXECUTION_FAILED: ErrorCode.EXECUTION_FAILED,
            ErrorType.OOM: ErrorCode.OOM,
            ErrorType.TIMEOUT: ErrorCode.TIMEOUT,
            ErrorType.SECURITY_ERROR: ErrorCode.SECURITY_ERROR,
        }
        
        error_code = error_code_map.get(error_type, ErrorCode.EXECUTION_FAILED)
        return ErrorResponseBuilder.build(message=message, error_code=error_code)
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> Dict:
        """
        Safely serialize tensor using numpy binary format.
        
        Consistent with FastSerializer - uses binary format instead of list.
        """
        import numpy as np
        
        tensor_cpu = tensor.detach().cpu() if hasattr(tensor, 'detach') else tensor.cpu()
        np_array = tensor_cpu.numpy()
        
        return {
            'data': np_array.tobytes(),  # Binary numpy format
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'format': 'numpy_binary',  # Format marker
        }

    def _run_shadow_mode_analysis(self, request: Dict, telemetry: Dict) -> None:
        """
        Phase 1 Shadow Mode: Query registry and log what we WOULD have done.
        
        This runs after v1.0 execution completes and logs decisions without
        affecting production traffic.
        """
        try:
            fingerprint = request['fingerprint']
            input_shapes = telemetry.get('input_shapes', {})
            
            # Query registry for profile
            profile_dict = self.profile_registry_client.get_profile(
                model_fingerprint=fingerprint,
                input_shapes=input_shapes
            )
            
            if profile_dict:
                # Profile found - log what we would have done
                logger.info(
                    f"🔮 SHADOW MODE: Profile {profile_dict['profile_id']} found for {fingerprint[:16]}... "
                    f"(phase={profile_dict['execution_phase']}, v{profile_dict['version']})"
                )
                
                # Compare actual vs predicted
                actual_phase = telemetry.get('cache_stats', {}).get('phase_detections', {})
                predicted_phase = profile_dict['execution_phase']
                
                if actual_phase:
                    # Get most common phase from actual detections
                    most_common_phase = max(actual_phase.items(), key=lambda x: x[1])[0] if actual_phase else 'unknown'
                    
                    if most_common_phase != predicted_phase:
                        logger.warning(
                            f"🔮 SHADOW MODE: Phase mismatch! "
                            f"Predicted={predicted_phase}, Actual={most_common_phase}"
                        )
                    else:
                        logger.info(
                            f"🔮 SHADOW MODE: Phase prediction correct! "
                            f"({predicted_phase})"
                        )
                
                # Log resource budget decisions
                if profile_dict.get('resource_budget'):
                    budget = profile_dict['resource_budget']
                    logger.info(
                        f"🔮 SHADOW MODE: Would use resource budget: "
                        f"activations={budget['activations_fraction']:.1%}, "
                        f"weights={budget['weights_fraction']:.1%}, "
                        f"kv_cache={budget['kv_cache_fraction']:.1%}"
                    )
                
                # Log optimization directives
                if profile_dict.get('optimization_directives'):
                    directives = profile_dict['optimization_directives']
                    logger.info(
                        f"🔮 SHADOW MODE: Would apply optimizations: {', '.join(directives)}"
                    )
            else:
                # No profile found
                logger.info(
                    f"🔮 SHADOW MODE: No profile found for {fingerprint[:16]}... "
                    f"(would use default v1.0 behavior)"
                )
        
        except Exception as e:
            # Shadow mode should never affect production
            logger.debug(f"Shadow mode analysis failed (non-critical): {e}")
    
    def _extract_input_shapes(self, inputs: Dict[str, Any]) -> Dict[str, Optional[list]]:
        """Extract input shapes for telemetry (handles serialized inputs)."""
        shapes: Dict[str, Optional[list]] = {}
        for key, value in inputs.items():
            if isinstance(value, dict) and 'shape' in value:
                shapes[key] = value['shape']
            elif hasattr(value, 'shape'):
                shapes[key] = list(value.shape)
            else:
                shapes[key] = None
        return shapes
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (for testing or manual recovery)."""
        self.circuit_breaker_open = False
        self.error_counts.clear()
        logger.info("Circuit breaker reset")
    
    def _deserialize_inputs(self, inputs: Dict) -> Dict:
        """
        Deserialize inputs from network format to tensors.

        This mirrors the deserialization logic in MemoryAwareModelCache.
        """
        logger.info(f"🔄 _deserialize_inputs called with: {list(inputs.keys()) if inputs else 'EMPTY'}")

        deserialized = {}

        for key, value in inputs.items():
            if isinstance(value, dict) and 'data' in value:
                # This is a serialized tensor from _serialize_inputs
                logger.info(f"🔄 Deserializing tensor: {key}")
                deserialized[key] = self._deserialize_tensor(value)
            else:
                # Pass through other values (scalars, etc.)
                logger.info(f"🔄 Passing through non-tensor: {key} = {value}")
                deserialized[key] = value

        logger.info(f"🔄 Deserialized result keys: {list(deserialized.keys()) if deserialized else 'EMPTY'}")
        return deserialized

    def _deserialize_tensor(self, tensor_dict: Dict) -> torch.Tensor:
        """Deserialize a single tensor from dict format."""
        import numpy as np

        # Handle binary numpy format (from _serialize_inputs)
        if isinstance(tensor_dict.get('data'), bytes):
            # Binary format - use frombuffer to reconstruct numpy array
            dtype_str = tensor_dict.get('dtype', 'float32')
            if dtype_str.startswith('torch.'):
                # Convert 'torch.float32' -> 'float32'
                dtype_str = dtype_str.replace('torch.', '')

            # Map torch dtypes to numpy dtypes
            dtype_map = {
                'float32': np.float32,
                'float16': np.float16,
                'int64': np.int64,
                'int32': np.int32,
                'bool': np.bool_,
            }
            numpy_dtype = dtype_map.get(dtype_str, np.float32)

            # Reconstruct numpy array from bytes
            shape = tensor_dict['shape']
            data_bytes = tensor_dict['data']
            np_array = np.frombuffer(data_bytes, dtype=numpy_dtype).reshape(shape)

            # Convert to torch tensor
            tensor = torch.from_numpy(np_array)
            return tensor
        else:
            # Legacy format or other types
            raise ValueError(f"Unsupported tensor format: {tensor_dict}")

    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
            'circuit_breaker_open': self.circuit_breaker_open,
            'error_counts': dict(self.error_counts),
            'model_cache_stats': self.model_cache.get_stats()
        }

