"""
OptimizationExecutor: Wraps SubgraphExecutor with Tensor Registry and Fusion Compiler.

Integrates:
- SmartTensorRegistry for weight/KV cache caching
- SRGFusionCompiler for pattern-based operation grouping
- PerformanceMonitor for tracking optimization overhead

This executor is a drop-in replacement for SubgraphExecutor that adds:
1. Check registry for cached model weights
2. Apply fusion compiler for operation grouping
3. Execute optimized subgraph
4. Track metrics for monitoring
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch

from ..subgraph_executor import SubgraphExecutor
from ..performance_monitor import PerformanceMonitor
from .tensor_registry import SmartTensorRegistry
from .fusion_compiler import SRGFusionCompiler
from ...config import get_config

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStats:
    """Statistics for optimization execution."""
    registry_hit: bool = False
    registry_lookup_ms: float = 0.0
    fusion_applied: bool = False
    fusion_grouping_ms: float = 0.0
    tensorrt_compiled: bool = False
    tensorrt_used: bool = False
    execution_ms: float = 0.0
    transfer_ms: float = 0.0
    total_ms: float = 0.0


class OptimizationExecutor:
    """
    Wraps SubgraphExecutor with optimization components.
    
    Flow:
    1. Check if model weights are in tensor registry (cache hit = skip transfer)
    2. Apply fusion compiler to group operations by pattern
    3. Execute optimized subgraph
    4. Track metrics for monitoring
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Core executor
        self.executor = SubgraphExecutor(gpu_id=gpu_id)
        
        # Optimization components
        self.registry: Optional[SmartTensorRegistry] = None
        self.compiler: Optional[SRGFusionCompiler] = None
        self.memory_manager: Optional['PhaseAwareMemoryManager'] = None
        self.pressure_handler: Optional['MemoryPressureHandler'] = None
        self.tensorrt_compiler: Optional['TensorRTCompiler'] = None
        self.monitor = PerformanceMonitor()

        # TensorRT cache for compiled modules
        self.tensorrt_cache: Dict[str, Any] = {}
        
        # Configuration
        self.config = get_config()
        self.opt_config = self.config.optimization
        
        # Initialize optimization components based on config
        self._initialize_optimizations()
        
        logger.info(
            f"OptimizationExecutor initialized on GPU {gpu_id} | "
            f"Registry: {self.opt_config.enable_tensor_registry} | "
            f"Fusion: {self.opt_config.enable_srg_fusion}"
        )

    def _initialize_optimizations(self):
        """Initialize tensor registry and fusion compiler based on config."""
        if self.opt_config.enable_tensor_registry:
            self.registry = SmartTensorRegistry(
                max_cached_models=self.opt_config.tensor_registry_max_models,
                max_bytes_per_model=self.opt_config.tensor_registry_max_bytes_per_model,
                max_total_bytes=self.opt_config.tensor_registry_max_total_bytes
            )
            logger.info("✓ SmartTensorRegistry initialized")
        
        if self.opt_config.enable_srg_fusion:
            self.compiler = SRGFusionCompiler(
                enable_torchscript=self.opt_config.enable_fusion_torchscript,
                enable_compilation=self.opt_config.enable_fusion_compilation
            )
            logger.info("✓ SRGFusionCompiler initialized")

        # ✅ Week 3: Initialize PhaseAwareMemoryManager
        try:
            # Get GPU memory info
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024 * 1024)
                from ..semantic_memory_manager import PhaseAwareMemoryManager
                self.memory_manager = PhaseAwareMemoryManager(total_gpu_memory_mb=gpu_memory_mb)
                logger.info("✓ PhaseAwareMemoryManager initialized")
            else:
                logger.info("⚠️ CUDA not available, skipping PhaseAwareMemoryManager")
        except Exception as e:
            logger.warning(f"Failed to initialize PhaseAwareMemoryManager: {e}")

        # ✅ Week 3: Initialize MemoryPressureHandler
        try:
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024 * 1024)
                from ..memory_pressure_handler import MemoryPressureHandler
                self.pressure_handler = MemoryPressureHandler(
                    total_gpu_memory_mb=gpu_memory_mb,
                    warning_threshold_percent=80.0,
                    critical_threshold_percent=95.0
                )
                logger.info("✓ MemoryPressureHandler initialized")
            else:
                logger.info("⚠️ CUDA not available, skipping MemoryPressureHandler")
        except Exception as e:
            logger.warning(f"Failed to initialize MemoryPressureHandler: {e}")

        # ✅ Week 3: Initialize TensorRTCompiler
        try:
            from .tensorrt_compiler import TensorRTCompiler
            self.tensorrt_compiler = TensorRTCompiler(compilation_threshold=10)  # Lower threshold for testing
            logger.info("✓ TensorRTCompiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorRTCompiler: {e}")

    async def execute(
        self,
        subgraph_request: Dict[str, Any],
        input_data: Dict[str, torch.Tensor],
        model_id: Optional[str] = None,
        model_version: int = 0,
        timeout: float = 300.0
    ) -> Tuple[torch.Tensor, OptimizationStats]:
        """
        Execute subgraph with optimizations.
        
        Args:
            subgraph_request: Serialized subgraph specification
            input_data: Input tensors (on CPU)
            model_id: Model ID for registry caching (e.g., "gpt2", "bert")
            model_version: Model version for cache invalidation
            timeout: Maximum execution time in seconds
        
        Returns:
            (result_tensor, stats)
        """
        stats = OptimizationStats()
        total_start = time.time()
        
        logger.info(f"🎯 OptimizationExecutor.execute: {len(subgraph_request.get('operations', []))} operations, {len(input_data)} inputs, timeout={timeout}s")
        
        try:
            # Step 1: Check tensor registry for cached model weights
            if self.registry and model_id:
                stats.registry_hit, input_data = await self._check_registry(
                    model_id=model_id,
                    input_data=input_data,
                    model_version=model_version
                )
                stats.registry_lookup_ms = (time.time() - total_start) * 1000
                
                if stats.registry_hit:
                    logger.debug(f"✓ Registry cache hit for {model_id}")
                    self.monitor.record_registry_hit(
                        bytes_saved=sum(t.numel() * t.element_size() for t in input_data.values()),
                        lookup_ms=stats.registry_lookup_ms
                    )
                else:
                    logger.debug(f"✗ Registry cache miss for {model_id}")
                    self.monitor.record_registry_miss(stats.registry_lookup_ms)
            
            # Step 2: Apply fusion compiler for operation grouping
            fusion_start = time.time()
            optimized_request = subgraph_request
            
            if self.compiler:
                semantic_metadata = subgraph_request.get('semantic_metadata', {})
                if semantic_metadata:
                    blocks = self.compiler.fuse_subgraph(
                        operations=subgraph_request.get('operations', []),
                        semantic_metadata=semantic_metadata
                    )
                    stats.fusion_applied = len(blocks) > 0
                    stats.fusion_grouping_ms = (time.time() - fusion_start) * 1000
                    
                    if stats.fusion_applied:
                        compiler_stats = self.compiler.get_stats()
                        logger.debug(
                            f"Fusion grouping: {compiler_stats['attention_blocks_identified']} attention, "
                            f"{compiler_stats['conv_blocks_identified']} conv blocks"
                        )
                        self.monitor.record_fusion_grouping(
                            attention_blocks=compiler_stats.get('attention_blocks_identified', 0),
                            conv_blocks=compiler_stats.get('conv_blocks_identified', 0),
                            no_fusion_blocks=compiler_stats.get('no_fusion_blocks', 0),
                            total_ops=len(subgraph_request.get('operations', [])),
                            grouping_ms=stats.fusion_grouping_ms
                        )
                        # Note: Could store blocks in optimized_request for later use
                        # when actual kernel fusion (Tier 2/3) is implemented
            
            # Step 3: Apply TensorRT compilation for repeated executions
            tensorrt_start = time.time()
            if self.tensorrt_compiler and model_id:
                # Check if we should compile to TensorRT
                profile = self.tensorrt_compiler.get_profile(model_id)
                if profile and profile.should_compile_tensorrt():
                    logger.info(f"Compiling {model_id} to TensorRT for optimization")
                    try:
                        # Step 3a: Synthesize TorchScript from operations
                        from djinn.server.torchscript_synthesizer import get_synthesizer
                        synthesizer = get_synthesizer()

                        operations = optimized_request.get('operations', [])
                        sample_inputs = list(input_data.values())

                        torchscript_module = synthesizer.synthesize_block(
                            operations=operations,
                            block_id=model_id,
                            sample_inputs=sample_inputs
                        )

                        if torchscript_module is None:
                            logger.warning(f"TorchScript synthesis failed for {model_id}")
                            stats.tensorrt_compiled = False
                        else:
                            # Step 3b: Compile TorchScript to TensorRT
                            trt_module = self.tensorrt_compiler.try_compile_tensorrt(
                                block_id=model_id,
                                torchscript_module=torchscript_module,
                                sample_input=sample_inputs[0] if sample_inputs else None,
                                use_fp16=True
                            )

                            if trt_module:
                                # Cache the compiled module
                                self.tensorrt_cache[model_id] = trt_module
                                stats.tensorrt_compiled = True
                                logger.info(f"✓ TensorRT compilation completed for {model_id}")
                            else:
                                stats.tensorrt_compiled = False

                    except Exception as e:
                        logger.warning(f"TensorRT compilation failed: {e}")
                        stats.tensorrt_compiled = False
                else:
                    stats.tensorrt_compiled = False
            else:
                stats.tensorrt_compiled = False

            tensorrt_ms = (time.time() - tensorrt_start) * 1000

            # Step 4: Execute subgraph (with all optimizations applied)
            exec_start = time.time()

            # Check if we have a cached TensorRT module (hot path)
            if model_id and model_id in self.tensorrt_cache:
                logger.info(f"🔥 Using cached TensorRT module for {model_id}")
                trt_module = self.tensorrt_cache[model_id]

                # Prepare inputs for TensorRT
                gpu_inputs = []
                for tensor in input_data.values():
                    if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                        gpu_inputs.append(tensor.cuda())
                    else:
                        gpu_inputs.append(tensor)

                # Execute with TensorRT
                with torch.no_grad():
                    result = trt_module(*gpu_inputs)

                stats.tensorrt_used = True
                logger.info(f"🚀 TensorRT execution completed for {model_id}")

            else:
                # Regular execution
                result = self.executor.execute(
                    subgraph_request=optimized_request,
                    input_data=input_data,
                    timeout=timeout
                )
                stats.tensorrt_used = False

            stats.execution_ms = (time.time() - exec_start) * 1000

            # Add TensorRT time to execution time if compilation happened
            if stats.tensorrt_compiled:
                stats.execution_ms += tensorrt_ms
            
            # Step 4: Track end-to-end metrics
            stats.total_ms = (time.time() - total_start) * 1000
            self.monitor.record_end_to_end_request(
                latency_ms=stats.total_ms,
                optimizations_enabled=(stats.registry_hit or stats.fusion_applied)
            )
            
            logger.debug(
                f"Execution complete: {stats.total_ms:.2f}ms total | "
                f"Registry: {stats.registry_lookup_ms:.2f}ms | "
                f"Fusion: {stats.fusion_grouping_ms:.2f}ms | "
                f"Exec: {stats.execution_ms:.2f}ms"
            )
            
            return result, stats
        
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error in optimized execution: {error_msg}", exc_info=True)
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.monitor.record_error(type(e).__name__)
            # Re-raise with proper formatting instead of falling back
            # This ensures the error is properly propagated
            raise RuntimeError(f"Optimized execution failed: {error_msg}") from e

    async def _check_registry(
        self,
        model_id: str,
        input_data: Dict[str, torch.Tensor],
        model_version: int
    ) -> Tuple[bool, Dict[str, torch.Tensor]]:
        """
        Check tensor registry for cached model weights.
        
        Returns:
            (hit, updated_input_data)
            - hit: True if found in cache
            - updated_input_data: With cached tensors substituted if hit
        """
        if not self.registry:
            return False, input_data
        
        registry_start = time.time()
        hit = False
        updated_data = dict(input_data)
        
        # Check each input tensor against registry
        for tensor_name, tensor in input_data.items():
            # Skip if already cached or not a weight-like tensor
            if not self._is_cacheable_tensor(tensor_name):
                continue
            
            check_start = time.time()
            needs_transfer, handle = await self.registry.check_and_register(
                model_id=model_id,
                tensor_name=tensor_name,
                tensor=tensor,
                model_version=model_version
            )
            
            if not needs_transfer and handle:
                # Cache hit - could use remote handle here
                # For now, tensor already available locally
                hit = True
                logger.debug(f"✓ Cache hit: {model_id}/{tensor_name}")
        
        lookup_ms = (time.time() - registry_start) * 1000
        return hit, updated_data

    @staticmethod
    def _is_cacheable_tensor(tensor_name: str) -> bool:
        """Check if tensor is worth caching (weights, not activations)."""
        # Cache weights and KV cache, skip activations/gradients
        cacheable_patterns = ['weight', 'kv_cache', 'cache', 'embedding']
        ephemeral_patterns = ['activation', 'hidden', 'logits', 'gradient', 'grad']
        
        name_lower = tensor_name.lower()
        
        # Don't cache if matches ephemeral patterns
        for pattern in ephemeral_patterns:
            if pattern in name_lower:
                return False
        
        # Cache if matches cacheable patterns
        for pattern in cacheable_patterns:
            if pattern in name_lower:
                return True
        
        # Conservative: default to caching if not explicitly ephemeral
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics snapshot."""
        return self.monitor.get_summary()

    def get_registry_metrics(self) -> Dict[str, Any]:
        """Get registry-specific metrics."""
        return self.monitor.get_registry_metrics()

    def get_fusion_metrics(self) -> Dict[str, Any]:
        """Get fusion compiler-specific metrics."""
        return self.monitor.get_fusion_metrics()

    async def invalidate_model_cache(self, model_id: str):
        """Invalidate all cached tensors for a model."""
        if self.registry:
            await self.registry.invalidate_model(model_id)
            logger.info(f"Invalidated cache for model: {model_id}")

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.monitor.reset()
        logger.info("Performance metrics reset")
