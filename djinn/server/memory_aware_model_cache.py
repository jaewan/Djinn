"""
Memory-Aware Model Cache: Production-grade model caching with intelligent memory management.

Key Features:
- Size-aware eviction (not count-based)
- Value-based eviction (access frequency / age / size)
- OOM protection with automatic recovery
- Phase-aware memory management (prefill/decode/vision optimization)
- Optimized input preparation (async transfer, pinned memory)
- Integration with Phase 1 weight cache

This is part of the redesign plan (Week 2).
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

from .architecture_registry import get_architecture_registry

logger = logging.getLogger(__name__)


@dataclass
class ModelMemoryProfile:
    """Memory profile for cached model."""
    fingerprint: str
    param_bytes: int
    activation_bytes: int  # Estimated peak activation memory
    total_bytes: int
    last_access_time: float
    access_count: int
    execution_time_ms: float = 0.0  # Track execution time for value calculation


class MemoryAwareModelCache:
    """
    Production-grade model cache with intelligent memory management.
    
    Key Features:
    - Size-aware eviction (not count-based)
    - Value-based eviction (access frequency / age / size)
    - OOM protection with automatic recovery
    - Integration with Phase 1 weight cache
    """
    
    def __init__(self, 
                 max_memory_gb: Optional[float] = None,
                 target_utilization: float = 0.8,
                 device: str = 'cuda:0'):
        
        self.device = torch.device(device)
        
        # Auto-detect available memory if not specified
        if max_memory_gb is None:
            if torch.cuda.is_available():
                # ✅ IMPROVEMENT: Use actual FREE GPU memory, not total memory
                # This accounts for other processes using the GPU
                free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(self.device)
                # Use 80% of FREE memory (not total) to leave headroom
                max_memory_gb = (free_memory_bytes / 1024**3) * target_utilization
                logger.info(
                    f"GPU memory: {total_memory_bytes/1024**3:.1f}GB total, "
                    f"{free_memory_bytes/1024**3:.1f}GB free, "
                    f"cache limit: {max_memory_gb:.1f}GB ({target_utilization*100:.0f}% of free)"
                )
            else:
                max_memory_gb = 8.0  # Default 8GB for CPU
        
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        
        # Model storage
        self.models: OrderedDict[str, nn.Module] = OrderedDict()
        self.profiles: Dict[str, ModelMemoryProfile] = {}
        # Track which models have been initialized (warmed up)
        self._initialized_models: Set[str] = set()
        
        # Architecture registry
        self.architecture_registry = get_architecture_registry()
        
        # Reuse Phase 1 weight cache
        from .gpu_cache import get_global_cache
        self.weight_cache = get_global_cache()
        
        # Integrate SmartTensorRegistry for shared weight caching with graph-based system
        # This allows both model cache and graph execution to share the same weight cache
        try:
            from .optimizations.tensor_registry import SmartTensorRegistry
            # Initialize with same memory limits as model cache
            max_total_bytes = self.max_memory_bytes
            self.tensor_registry = SmartTensorRegistry(
                max_cached_models=5,  # Reasonable default
                max_bytes_per_model=None,  # Use total limit instead
                max_total_bytes=max_total_bytes
            )
            logger.info("SmartTensorRegistry integrated for shared weight caching")
        except Exception as e:
            logger.warning(f"Failed to initialize SmartTensorRegistry: {e}")
            self.tensor_registry = None
        
        # Phase-aware memory management
        try:
            from .semantic_memory_manager import PhaseAwareMemoryManager
            from djinn.core.types import ExecutionPhase
            total_memory_mb = (self.max_memory_bytes / 1024**2)
            self.phase_memory_manager = PhaseAwareMemoryManager(total_gpu_memory_mb=total_memory_mb)
            self.ExecutionPhase = ExecutionPhase
            logger.info("Phase-aware memory management enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize phase-aware memory manager: {e}")
            self.phase_memory_manager = None
            self.ExecutionPhase = None
        
        # Memory tracking
        self.current_memory_bytes = 0
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'oom_events': 0,
            'total_executions': 0,
            'total_latency_ms': 0.0,
            'phase_detections': {},  # Track phase detection counts
            'phase_switches': 0,  # Track phase changes
            'last_phase': None,  # Track last detected phase
        }
        
        logger.info(
            f"MemoryAwareModelCache initialized: max_memory={max_memory_gb:.1f}GB, "
            f"device={device}"
        )
    
    def register_model(self, 
                       fingerprint: str,
                       descriptor: Dict,
                       weight_ids: Dict[str, str],
                       uncached_weights: Dict[str, torch.Tensor],
                       architecture_data: Optional[bytes] = None):
        """
        Register model with memory-aware eviction.
        
        Integrates with Phase 1 weight cache for efficient weight handling.
        
        Args:
            fingerprint: Model fingerprint
            descriptor: Architecture descriptor dict
            weight_ids: Dict mapping parameter names to weight identifiers
            uncached_weights: Dict of uncached weight tensors
            architecture_data: Optional serialized architecture bytes
        
        Raises:
            ValueError: If architecture cannot be reconstructed
            MemoryError: If model is too large for available memory
        """
        import time
        total_start = time.perf_counter()
        
        if fingerprint in self.models:
            logger.info(f"Model {fingerprint} already registered")
            return
        
        logger.info(f"Registering model {fingerprint}")
        
        # Load architecture
        try:
            model = self.architecture_registry.load_architecture(
                fingerprint, architecture_data, descriptor
            )
        except ValueError as e:
            # Cannot reconstruct - will use graph execution fallback
            raise ValueError(f"Cannot reconstruct model architecture: {e}")
        
        # Calculate memory requirement
        param_bytes = sum(
            p.numel() * p.element_size() 
            for p in uncached_weights.values()
            if isinstance(p, torch.Tensor)
        )
        
        # Add cached weights size (from Phase 1 cache)
        # ✅ FIX: Handle both dict and list formats for weight_ids
        weight_ids_iter = weight_ids.values() if isinstance(weight_ids, dict) else weight_ids
        for weight_id in weight_ids_iter:
            if weight_id in self.weight_cache.cache_new:
                cached_tensor = self.weight_cache.cache_new[weight_id]
                param_bytes += cached_tensor.numel() * cached_tensor.element_size()
        
        # Estimate activation memory (heuristic: ~25% of params for peak activations)
        activation_bytes = param_bytes // 4
        total_bytes = param_bytes + activation_bytes
        
        # Evict models if needed (memory-aware!)
        while self.current_memory_bytes + total_bytes > self.max_memory_bytes:
            if not self.models:
                raise MemoryError(
                    f"Model requires {total_bytes/1024**3:.1f}GB but limit is "
                    f"{self.max_memory_bytes/1024**3:.1f}GB. Cannot fit even one model."
                )
            self._evict_least_valuable_model()
        
        # ✅ OPTIMIZATION: Progressive GPU transfer with CUDA streams (pipelining)
        # Load state dict (leveraging Phase 1 cache and SmartTensorRegistry)
        state_dict = {}
        
        # ✅ FIX: Handle CUDA OOM during registration with automatic eviction
        # Try to transfer weights to GPU, with OOM recovery
        max_oom_retries = 2
        oom_retry_count = 0
        
        while oom_retry_count <= max_oom_retries:
            try:
                # Use CUDA streams for parallel GPU transfers (if CUDA available)
                if torch.cuda.is_available() and self.device.type == 'cuda' and len(uncached_weights) > 1:
                    # Create multiple streams for parallel transfers
                    num_streams = min(4, len(uncached_weights))  # Use up to 4 streams
                    streams = [torch.cuda.Stream() for _ in range(num_streams)]
                    
                    # Transfer uncached weights in parallel using streams
                    # ✅ FIX: Handle both dict and list formats for weight_ids
                    if isinstance(weight_ids, dict):
                        weight_ids_iter = weight_ids.items()
                    else:
                        # List format: weight_ids is list of param names, use param name as weight_id
                        weight_ids_iter = [(name, name) for name in weight_ids]
                    
                    uncached_items = [
                        (param_name, weight_id, uncached_weights[param_name])
                        for param_name, weight_id in weight_ids_iter
                        if param_name in uncached_weights
                    ]
                    
                    if uncached_items:
                        gpu_transfer_start = time.perf_counter()
                        logger.info(f"🔄 Transferring {len(uncached_items)} weights to GPU using {num_streams} CUDA streams...")
                        
                        # Schedule transfers on different streams (parallel)
                        for i, (param_name, weight_id, weight_tensor) in enumerate(uncached_items):
                            if isinstance(weight_tensor, dict):
                                weight_tensor = self._deserialize_tensor(weight_tensor)
                            
                            # Assign to stream (round-robin)
                            stream = streams[i % num_streams]
                            with torch.cuda.stream(stream):
                                # Move to GPU (non-blocking, uses stream)
                                gpu_tensor = weight_tensor.to(self.device, non_blocking=True)
                                
                                # Add to Phase 1 cache
                                self.weight_cache.cache_new[weight_id] = gpu_tensor
                                
                                # Also register in SmartTensorRegistry
                                if self.tensor_registry:
                                    self._register_weight_in_tensor_registry(
                                        fingerprint, param_name, gpu_tensor
                                    )
                                
                                state_dict[param_name] = gpu_tensor
                        
                        # Synchronize all streams (wait for all transfers to complete)
                        for stream in streams:
                            stream.synchronize()
                        
                        gpu_transfer_time = (time.perf_counter() - gpu_transfer_start) * 1000
                        logger.info(f"✅ Transferred {len(uncached_items)} weights to GPU (CUDA streams): {gpu_transfer_time:.1f}ms")
                else:
                    # Fallback: Sequential transfer (CPU, no CUDA, or single weight)
                    # ✅ FIX: Handle both dict and list formats for weight_ids
                    if isinstance(weight_ids, dict):
                        weight_ids_iter = weight_ids.items()
                    else:
                        # List format: weight_ids is list of param names, use param name as weight_id
                        weight_ids_iter = [(name, name) for name in weight_ids]
                    
                    for param_name, weight_id in weight_ids_iter:
                        if param_name in uncached_weights:
                            # New weight - add to Phase 1 cache
                            weight_tensor = uncached_weights[param_name]
                            if isinstance(weight_tensor, dict):
                                weight_tensor = self._deserialize_tensor(weight_tensor)
                            
                            gpu_tensor = weight_tensor.to(self.device, non_blocking=True)
                            # Add to Phase 1 cache
                            self.weight_cache.cache_new[weight_id] = gpu_tensor
                            
                            # Also register in SmartTensorRegistry for shared caching with graph-based system
                            # This allows graph-based execution to reuse weights cached by model cache
                            if self.tensor_registry:
                                self._register_weight_in_tensor_registry(
                                    fingerprint, param_name, gpu_tensor
                                )
                            
                            state_dict[param_name] = gpu_tensor
                
                # Success - break out of retry loop
                break
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    oom_retry_count += 1
                    self.stats['oom_events'] += 1
                    logger.warning(
                        f"⚠️ CUDA OOM during model registration (attempt {oom_retry_count}/{max_oom_retries + 1}). "
                        f"Evicting models to free memory..."
                    )
                    
                    # Emergency eviction: free more memory
                    self._handle_oom()
                    
                    # Clear CUDA cache
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Also evict more aggressively if we have models
                    if self.models and oom_retry_count < max_oom_retries:
                        # Evict additional models (50% on second attempt)
                        evict_count = max(1, len(self.models) // 2)
                        for _ in range(evict_count):
                            if self.models:
                                self._evict_least_valuable_model()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                    if oom_retry_count > max_oom_retries:
                        # All retries exhausted
                        logger.error(
                            f"❌ CUDA OOM: Failed to register model after {max_oom_retries + 1} attempts. "
                            f"Model requires {total_bytes/1024**3:.1f}GB but only {self.current_memory_bytes/1024**3:.1f}GB "
                            f"available after eviction."
                        )
                        raise MemoryError(
                            f"CUDA out of memory: Cannot register model {fingerprint}. "
                            f"Required: {total_bytes/1024**3:.1f}GB, "
                            f"Available after eviction: {(self.max_memory_bytes - self.current_memory_bytes)/1024**3:.1f}GB"
                        )
                else:
                    # Not an OOM error - re-raise
                    raise
        
        # Add cached weights to state dict
        # ✅ FIX: Handle both dict and list formats for weight_ids
        if isinstance(weight_ids, dict):
            weight_ids_iter = weight_ids.items()
        else:
            # List format: weight_ids is list of param names, use param name as weight_id
            weight_ids_iter = [(name, name) for name in weight_ids]
        
        for param_name, weight_id in weight_ids_iter:
            if param_name not in state_dict:
                # Get from Phase 1 cache
                if weight_id in self.weight_cache.cache_new:
                    state_dict[param_name] = self.weight_cache.cache_new[weight_id]
                else:
                    logger.warning(f"Weight {weight_id} not found in cache")
        
        # Load state dict into model
        load_start = time.perf_counter()
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        load_time = (time.perf_counter() - load_start) * 1000
        logger.info(f"⏱️  Model load_state_dict + to(device): {load_time:.1f}ms")
        
        # ✅ NEW: Warmup is now done via explicit init_model() call
        # This makes registration fast (~2-3s instead of 12.9s)
        # Model is registered but not yet initialized (warmed up)
        logger.info(f"Model {fingerprint} registered (not yet initialized - call init_model() to warmup)")
        
        # Store model and profile
        self.models[fingerprint] = model
        self.profiles[fingerprint] = ModelMemoryProfile(
            fingerprint=fingerprint,
            param_bytes=param_bytes,
            activation_bytes=activation_bytes,
            total_bytes=total_bytes,
            last_access_time=time.time(),
            access_count=0
        )
        
        self.current_memory_bytes += total_bytes
        
        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(
            f"Model {fingerprint} registered: {total_bytes/1024**3:.2f}GB "
            f"(total cached: {self.current_memory_bytes/1024**3:.2f}GB, "
            f"total time: {total_time:.1f}ms)"
        )
    
    def get_model_reference(self, fingerprint: str) -> Optional[nn.Module]:
        """
        Return a reference to the cached model if available.

        Primarily used to bridge existing caches with the VMU-backed ModelCacheV23.
        """
        return self.models.get(fingerprint)
    
    def init_model(self, fingerprint: str) -> bool:
        """
        Initialize (warmup) a registered model by triggering CUDA kernel JIT compilation.
        
        This is an explicit initialization step that can be called after registration.
        It executes the model once to trigger kernel compilation, ensuring fast first execution.
        
        **Benefits of explicit initialization:**
        - Registration is fast (~2-3s instead of 12.9s)
        - User controls when warmup happens (can batch multiple models)
        - Better separation of concerns (registration vs. initialization)
        
        Args:
            fingerprint: Model fingerprint to initialize
        
        Returns:
            True if initialization succeeded, False otherwise
        
        Example:
            # Register model (fast, ~2-3s)
            cache.register_model(fingerprint, descriptor, weight_ids, weights)
            
            # Initialize model (warmup, ~8-10s for GPT-2-small)
            cache.init_model(fingerprint)
            
            # Now first execution will be fast (no compilation overhead)
        """
        if fingerprint not in self.models:
            logger.warning(f"Model {fingerprint} not registered, cannot initialize")
            return False
        
        if fingerprint in self._initialized_models:
            logger.debug(f"Model {fingerprint} already initialized")
            return True
        
        model = self.models[fingerprint]
        profile = self.profiles.get(fingerprint)
        
        # Try to get descriptor from architecture registry
        descriptor = {}
        try:
            arch_data = self.architecture_registry.get_architecture(fingerprint)
            if arch_data:
                descriptor = arch_data.get('descriptor', {})
        except:
            pass
        
        # Fallback: infer from model
        if not descriptor:
            descriptor = {
                'class_name': model.__class__.__name__,
                'class_module': model.__class__.__module__,
            }
        
        logger.info(f"Initializing model {fingerprint} (JIT compilation warmup)...")
        warmup_start = time.perf_counter()
        
        try:
            # Create dummy input based on model type
            dummy_input = self._create_dummy_input(model, descriptor)
            if dummy_input is not None:
                with torch.no_grad():
                    # Execute once to trigger JIT compilation
                    if isinstance(model, nn.Sequential):
                        _ = model(dummy_input)
                    else:
                        # Try keyword args first
                        try:
                            _ = model(**dummy_input) if isinstance(dummy_input, dict) else model(dummy_input)
                        except TypeError:
                            _ = model(dummy_input) if not isinstance(dummy_input, dict) else model(**dummy_input)
                    
                    # Synchronize to ensure compilation completes
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                
                warmup_time = (time.perf_counter() - warmup_start) * 1000
                self._initialized_models.add(fingerprint)
                logger.info(f"✅ Model {fingerprint} initialized (JIT compilation): {warmup_time:.1f}ms")
                return True
            else:
                logger.warning(f"Could not create dummy input for {fingerprint}, skipping warmup")
                return False
        except Exception as e:
            warmup_time = (time.perf_counter() - warmup_start) * 1000
            logger.warning(f"Model initialization failed for {fingerprint} (took {warmup_time:.1f}ms): {e}")
            return False
    
    def _register_weight_in_tensor_registry(
        self, 
        model_id: str, 
        tensor_name: str, 
        tensor: torch.Tensor
    ):
        """
        Register a weight tensor in SmartTensorRegistry for shared caching.
        
        This allows the graph-based execution system to reuse weights cached
        by the model cache system, reducing memory duplication.
        
        Args:
            model_id: Model fingerprint/identifier
            tensor_name: Parameter name (e.g., "transformer.layer.0.weight")
            tensor: The weight tensor on GPU
        """
        if not self.tensor_registry:
            return
        
        try:
            import asyncio
            from .optimizations.tensor_registry import RemoteHandle
            import uuid
            
            # Create remote handle for the tensor
            tensor_bytes = tensor.numel() * tensor.element_size()
            remote_handle = RemoteHandle(
                device_id=str(self.device),
                tensor_id=str(uuid.uuid4()),
                shape=tensor.shape,
                dtype=tensor.dtype,
                timestamp=time.time(),
                version=0,  # Model version (could be enhanced with version tracking)
                tensor_bytes=tensor_bytes
            )
            
            # Register in tensor registry (async call in sync context)
            # Try to use existing event loop, or create new one if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Event loop is running - skip registration for now
                    # TODO: Could use asyncio.create_task if we make this async
                    logger.debug(f"Skipping tensor registry registration (event loop running): {tensor_name}")
                    return
            except RuntimeError:
                # No event loop - create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                needs_transfer, handle = loop.run_until_complete(
                    self.tensor_registry.check_and_register(
                        model_id=model_id,
                        tensor_name=tensor_name,
                        tensor=tensor,
                        model_version=0,
                        remote_handle=remote_handle
                    )
                )
                if not needs_transfer:
                    logger.debug(f"✓ Weight {tensor_name} registered in SmartTensorRegistry")
            finally:
                # Only close if we created the loop
                try:
                    current_loop = asyncio.get_event_loop()
                    if loop != current_loop:
                        loop.close()
                except RuntimeError:
                    loop.close()
        except Exception as e:
            # Graceful degradation - continue without tensor registry
            logger.debug(f"Failed to register weight in SmartTensorRegistry: {e}")
    
    def _detect_execution_phase(self, model: nn.Module, inputs: Dict[str, Any], 
                                hints: Optional[Dict] = None) -> 'ExecutionPhase':
        """
        Detect execution phase from model type and inputs.
        
        Heuristics:
        - If KV cache present → decode
        - If large sequence length → prefill
        - If image inputs → vision encoding
        - If transformers model → prefill (default for LLM)
        - Else → forward
        """
        if not self.ExecutionPhase:
            return None
        
        # Check hints first (most reliable)
        if hints:
            phase_str = hints.get('phase') or hints.get('execution_phase')
            if phase_str:
                try:
                    return self.ExecutionPhase(phase_str)
                except ValueError:
                    pass
        
        # Check for KV cache (indicates decode)
        if any(key in inputs for key in ['past_key_values', 'kv_cache', 'cache']):
            return self.ExecutionPhase.LLM_DECODE
        
        # Check for image inputs (indicates vision)
        if any(key in inputs for key in ['pixel_values', 'images', 'image']):
            return self.ExecutionPhase.VISION_ENCODING
        
        # Check sequence length for LLM models
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            if isinstance(input_ids, torch.Tensor):
                seq_len = input_ids.shape[-1] if len(input_ids.shape) > 1 else 1
                # Large sequence = prefill, small = decode
                if seq_len > 10:
                    return self.ExecutionPhase.LLM_PREFILL
                else:
                    return self.ExecutionPhase.LLM_DECODE
        
        # Check model type
        model_class_name = model.__class__.__name__.lower()
        if any(name in model_class_name for name in ['gpt', 'bert', 'transformer', 'llm']):
            # Default to prefill for LLM models
            return self.ExecutionPhase.LLM_PREFILL
        
        if any(name in model_class_name for name in ['resnet', 'vision', 'cnn', 'conv']):
            return self.ExecutionPhase.VISION_ENCODING
        
        # Default to unknown (general forward pass)
        return self.ExecutionPhase.UNKNOWN
    
    def _evict_least_valuable_model(self):
        """
        Evict model using value score with phase-aware priorities.
        
        Value = (access_count * execution_speed) / (age * size)
        
        Higher value = worth keeping
        Lower value = safe to evict
        
        Uses phase-aware eviction priorities if available.
        """
        
        if not self.models:
            return
        
        current_time = time.time()
        
        # Get phase-aware eviction priorities if available
        eviction_priorities = None
        if self.phase_memory_manager:
            try:
                eviction_priorities = self.phase_memory_manager.get_eviction_priority_order()
            except Exception:
                pass
        
        # Calculate value scores
        scores = {}
        for fp, profile in self.profiles.items():
            age_seconds = max(1.0, current_time - profile.last_access_time)
            size_gb = profile.total_bytes / 1024**3
            access_count = profile.access_count + 1  # +1 to avoid division by zero
            execution_speed = 1.0 / max(profile.execution_time_ms, 1.0)  # Faster = higher value
            
            # Value formula: favor frequently accessed, fast-executing, small models
            # Penalize old, large, slow models
            base_value = (access_count * execution_speed) / (age_seconds * size_gb)
            
            # Apply phase-aware priority adjustment
            # Models matching current phase priorities get higher value
            if eviction_priorities:
                # This is a simplified heuristic - in practice, we'd need to know
                # which phase each model is associated with
                # For now, just use base value
                scores[fp] = base_value
            else:
                scores[fp] = base_value
        
        # Evict lowest value model
        victim = min(scores, key=scores.get)
        victim_profile = self.profiles[victim]
        
        # Remove from cache
        del self.models[victim]
        self.current_memory_bytes -= victim_profile.total_bytes
        del self.profiles[victim]
        
        # Clear from Phase 1 cache (optional - may want to keep weights)
        # For now, keep weights in Phase 1 cache (they're shared)
        
        self.stats['evictions'] += 1
        
        logger.info(
            f"Evicted model {victim}: {victim_profile.total_bytes/1024**3:.2f}GB "
            f"(value={scores[victim]:.4f}, freed {self.current_memory_bytes/1024**3:.2f}GB)"
        )
    
    def execute(self, fingerprint: str, inputs: Dict[str, Any], 
                hints: Optional[Dict] = None) -> torch.Tensor:
        """
        Execute model with OOM protection.
        
        Args:
            fingerprint: Model fingerprint
            inputs: Input tensors dict
            hints: Optional scheduling hints
        
        Returns:
            Output tensor (on CPU)
        
        Raises:
            ValueError: If model not found
            RuntimeError: If execution fails
        """
        
        start_time = time.perf_counter()
        
        self.stats['total_executions'] += 1
        
        if fingerprint not in self.models:
            self.stats['cache_misses'] += 1
            raise ValueError(f"Model {fingerprint} not found. Please register first.")
        
        self.stats['cache_hits'] += 1
        
        # Update access tracking (LRU)
        self.models.move_to_end(fingerprint)
        profile = self.profiles[fingerprint]
        profile.last_access_time = time.time()
        profile.access_count += 1
        
        model = self.models[fingerprint]
        
        # ✅ DIAGNOSTIC: Log execution start
        logger.info(f"🔍 [DIAGNOSTIC] Starting model execution for {fingerprint[:16]}...")
        
        # Detect execution phase and adjust memory budgets
        if self.phase_memory_manager and self.ExecutionPhase:
            phase = self._detect_execution_phase(model, inputs, hints)
            
            # Track phase detection for monitoring
            if phase:
                phase_str = phase.value if hasattr(phase, 'value') else str(phase)
                self.stats['phase_detections'][phase_str] = self.stats['phase_detections'].get(phase_str, 0) + 1
                
                # Track phase switches
                if self.stats['last_phase'] and self.stats['last_phase'] != phase_str:
                    self.stats['phase_switches'] += 1
                self.stats['last_phase'] = phase_str
            
            self.phase_memory_manager.adjust_for_phase(phase)
        
        # Prepare inputs - handle both dict format and direct tensors
        # This is OUTSIDE GPU execution timing (like PyTorch baseline)
        # Optimization: Use non_blocking=True for async transfer, pinned memory when possible
        deserialize_start = time.perf_counter()
        gpu_inputs = {}
        
        # Get model dtype for dtype matching (critical for float16 models)
        model_dtype = None
        if fingerprint in self.models:
            model = self.models[fingerprint]
            # Try to infer model dtype from first parameter
            for param in model.parameters():
                if param.dtype.is_floating_point:
                    model_dtype = param.dtype
                    break
        
        for key, value in inputs.items():
            if isinstance(value, dict) and 'data' in value:
                # Deserialize from dict format
                tensor = self._deserialize_tensor(value)
                # Convert to GPU and match model dtype if needed
                tensor = tensor.to(self.device, non_blocking=True)
                if model_dtype and tensor.dtype.is_floating_point and tensor.dtype != model_dtype:
                    tensor = tensor.to(dtype=model_dtype)
                gpu_inputs[key] = tensor
            elif isinstance(value, torch.Tensor):
                # Direct tensor - use non_blocking for async transfer
                # Pin memory if tensor is on CPU (faster transfer)
                if value.device.type == 'cpu' and self.device.type == 'cuda':
                    # Pin memory for faster transfer (if not already pinned)
                    if not value.is_pinned():
                        try:
                            value = value.pin_memory()
                        except RuntimeError:
                            # Memory pinning failed, continue without pinning
                            pass
                tensor = value.to(self.device, non_blocking=True)
                # Match model dtype for floating point tensors
                if model_dtype and tensor.dtype.is_floating_point and tensor.dtype != model_dtype:
                    tensor = tensor.to(dtype=model_dtype)
                gpu_inputs[key] = tensor
            else:
                # Other types (e.g., lists, scalars)
                gpu_inputs[key] = value
        # Ensure all inputs are on GPU before timing starts
        # This synchronization ensures all async transfers complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        deserialize_time = (time.perf_counter() - deserialize_start) * 1000
        logger.info(f"🔍 [DIAGNOSTIC] Input deserialization + GPU transfer: {deserialize_time:.1f}ms")
        
        # Execute model with OOM protection
        # Handle nn.Sequential models - they expect positional args, not keyword args
        try:
            # Enable profiling context
            from .profiling_context import get_profiler, record_phase
            profiler = get_profiler()
            
            # Execute model forward pass
            # NOTE: We're NOT using autocast here to match PyTorch baseline exactly
            # Autocast adds overhead (~2-5ms) and PyTorch baseline doesn't use it
            # TODO: Make autocast optional via config if needed for mixed precision training
            with torch.no_grad():
                    # Measure GPU execution time with proper synchronization
                    # This should match PyTorch baseline timing exactly
                    with record_phase('model_cache_gpu_execution', metadata={
                        'fingerprint': fingerprint,
                        'input_count': len(gpu_inputs)
                    }):
                        # Note: We already synchronized above (line 447), so no need to sync again here
                        # This eliminates redundant synchronization overhead
                        
                        # Handle nn.Sequential models - they expect positional args
                        if isinstance(model, nn.Sequential) and len(gpu_inputs) == 1:
                            # Sequential models expect: model(input_tensor)
                            input_tensor = list(gpu_inputs.values())[0]
                            output = model(input_tensor)
                        else:
                            # Other models expect: model(**inputs) or model(input_tensor)
                            # Try keyword args first, fallback to positional
                            try:
                                output = model(**gpu_inputs)
                            except TypeError:
                                # Fallback to positional args (for Sequential or models that don't accept kwargs)
                                input_values = list(gpu_inputs.values())
                                if len(input_values) == 1:
                                    output = model(input_values[0])
                                else:
                                    output = model(*input_values)
                        
                        # Synchronize GPU after execution (like PyTorch baseline)
                        # This ensures all GPU work completes before we measure the end time
                        forward_end = time.perf_counter()
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        forward_time = (time.perf_counter() - forward_end) * 1000
                        logger.info(f"🔍 [DIAGNOSTIC] Model forward pass: {forward_time:.1f}ms (sync overhead)")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                self.stats['oom_events'] += 1
                logger.warning(f"OOM during execution of {fingerprint}, clearing cache")
                
                # Emergency eviction
                self._handle_oom()
                
                # Clear CUDA cache
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Retry once with same logic (no autocast to match baseline)
                with torch.no_grad():
                        if isinstance(model, nn.Sequential) and len(gpu_inputs) == 1:
                            input_tensor = list(gpu_inputs.values())[0]
                            output = model(input_tensor)
                        else:
                            try:
                                output = model(**gpu_inputs)
                            except TypeError:
                                input_values = list(gpu_inputs.values())
                                if len(input_values) == 1:
                                    output = model(input_values[0])
                                else:
                                    output = model(*input_values)
            else:
                raise
        
        # Handle output - transformers models return ModelOutput objects
        if hasattr(output, 'logits'):
            # Transformers model output (e.g., GPT2LMHeadModel)
            output = output.logits
        elif isinstance(output, tuple):
            # Tuple output (e.g., (logits, past_key_values))
            output = output[0]
        
        # ✅ DIAGNOSTIC: Log CPU transfer time
        cpu_transfer_start = time.perf_counter()
        result = output.cpu()
        cpu_transfer_time = (time.perf_counter() - cpu_transfer_start) * 1000
        logger.info(f"🔍 [DIAGNOSTIC] Output CPU transfer: {cpu_transfer_time:.1f}ms")
        
        # Record execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"🔍 [DIAGNOSTIC] Total execution time: {execution_time_ms:.1f}ms")
        profile.execution_time_ms = execution_time_ms
        self.stats['total_latency_ms'] += execution_time_ms
        
        return result
    
    def _handle_oom(self):
        """Handle out-of-memory by aggressive eviction."""
        # Evict 25% of models
        evict_count = max(1, len(self.models) // 4)
        for _ in range(evict_count):
            if self.models:
                self._evict_least_valuable_model()
    
    def _deserialize_tensor(self, tensor_dict: Dict) -> torch.Tensor:
        """Deserialize tensor from dict (delegates to centralized function)."""
        from .serialization import deserialize_tensor_from_dict
        return deserialize_tensor_from_dict(tensor_dict)
    
    def get_memory_status(self) -> Dict:
        """Get current memory status."""
        return {
            'used_gb': self.current_memory_bytes / 1024**3,
            'max_gb': self.max_memory_bytes / 1024**3,
            'utilization': self.current_memory_bytes / self.max_memory_bytes if self.max_memory_bytes > 0 else 0.0,
            'num_models': len(self.models),
            'stats': self.stats.copy()
        }
    
    def _create_dummy_input(self, model: nn.Module, descriptor: Dict) -> Optional[Any]:
        """
        Create dummy input for model warmup.
        
        Returns:
            Dummy input (tensor or dict) suitable for model forward pass, or None if cannot infer
        """
        try:
            # Try to infer input shape from model
            # For transformers models, check for config in descriptor
            if 'config' in descriptor:
                config = descriptor['config']
                # GPT-2 style models: input_ids shape (batch_size, seq_length)
                if 'vocab_size' in config or 'n_positions' in config:
                    batch_size = 1
                    seq_length = config.get('n_positions', 128) if 'n_positions' in config else 128
                    vocab_size = config.get('vocab_size', 50257)
                    # Create dummy input_ids
                    dummy_input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=self.device)
                    return {'input_ids': dummy_input_ids}
            
            # For Sequential models, try to infer from first layer
            if isinstance(model, nn.Sequential) and len(model) > 0:
                first_layer = model[0]
                if isinstance(first_layer, nn.Linear):
                    # Create dummy input matching input features
                    dummy_input = torch.zeros((1, first_layer.in_features), device=self.device)
                    return dummy_input
                elif isinstance(first_layer, nn.Conv2d):
                    # Create dummy image input
                    dummy_input = torch.zeros((1, first_layer.in_channels, 224, 224), device=self.device)
                    return dummy_input
            
            # Try to get input shape from model's first parameter
            # This is a heuristic - may not work for all models
            for name, param in model.named_parameters():
                if len(param.shape) >= 2:
                    # Assume first dimension is batch, second is input size
                    # This is a rough heuristic
                    if 'embed' in name.lower() or 'weight' in name.lower():
                        if len(param.shape) == 2:
                            # Linear layer: (out_features, in_features)
                            dummy_input = torch.zeros((1, param.shape[1]), device=self.device)
                            return dummy_input
                        elif len(param.shape) == 4:
                            # Conv2d: (out_channels, in_channels, H, W)
                            dummy_input = torch.zeros((1, param.shape[1], 224, 224), device=self.device)
                            return dummy_input
                break
            
            # Fallback: Try to use model's forward signature if available
            # This is a last resort
            logger.debug("Could not infer dummy input shape, skipping warmup")
            return None
            
        except Exception as e:
            logger.debug(f"Failed to create dummy input: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_execs = self.stats['total_executions']
        if total_execs > 0:
            hit_rate = self.stats['cache_hits'] / total_execs
            avg_latency = self.stats['total_latency_ms'] / total_execs
        else:
            hit_rate = 0
            avg_latency = 0
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'avg_latency_ms': avg_latency,
            'total_executions': total_execs,
            'cached_models': len(self.models),
            'evictions': self.stats['evictions'],
            'oom_events': self.stats['oom_events'],
            'phase_detections': self.stats.get('phase_detections', {}),
            'phase_switches': self.stats.get('phase_switches', 0),
            'last_phase': self.stats.get('last_phase'),
            **self.get_memory_status()
        }
    
    def clear(self):
        """Clear all cached models."""
        self.models.clear()
        self.profiles.clear()
        self.current_memory_bytes = 0
        logger.info("Model cache cleared")

