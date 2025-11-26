"""
Phase-Aware Execution Engine for Semantic-Driven Optimization.

Implements different execution strategies for LLM prefill vs decode phases:

LLM Prefill (compute-bound, parallel attention):
  - Large batch size (maximize parallelism)
  - Mixed precision (FP16/INT8 where possible)
  - Aggressive fusion (reduce memory bandwidth)
  - Higher GPU utilization target (85-95%)

LLM Decode (memory-bound, sequential generation):
  - Small batch size (minimize KV cache memory)
  - Full precision (FP32 for numerical stability)
  - KV cache pinning (prevent eviction)
  - Incremental computation (reuse previous values)

Vision & Multimodal:
  - Workload-specific optimizations
  - Feature map management
  - Cross-modal fusion strategies
"""

import logging
import torch
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from djinn.core.types import ExecutionPhase, DataResidency
from ..semantic_memory_manager import (
    PhaseAwareMemoryManager,
    LifetimeBasedEvictor,
    EvictionPriority
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStrategy:
    """Strategy parameters for a specific execution phase."""
    phase: ExecutionPhase
    batch_size: int
    use_mixed_precision: bool
    enable_fusion: bool
    pin_kv_cache: bool
    incremental_compute: bool
    target_gpu_utilization: float  # 0-1
    max_gpu_memory_utilization: float  # 0-1
    description: str


class ExecutionPhaseStrategy:
    """Base class for phase-specific execution strategies."""
    
    def __init__(self, phase: ExecutionPhase):
        self.phase = phase
        self.stats = {
            'executions': 0,
            'total_time_ms': 0.0,
            'total_memory_bytes': 0,
        }
    
    def get_strategy(self) -> ExecutionStrategy:
        """Get execution strategy for this phase."""
        raise NotImplementedError
    
    def prepare_for_execution(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for phase-specific execution."""
        return inputs
    
    def post_execution_cleanup(self) -> None:
        """Clean up after execution."""
        pass
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            **self.stats,
            'avg_time_ms': self.stats['total_time_ms'] / max(1, self.stats['executions']),
            'avg_memory_bytes': self.stats['total_memory_bytes'] / max(1, self.stats['executions']),
        }


class PrefillExecutionStrategy(ExecutionPhaseStrategy):
    """Prefill phase (parallel attention over sequence): compute-bound optimization."""
    
    def __init__(self):
        super().__init__(ExecutionPhase.LLM_PREFILL)
        self.fusion_compiler = None  # Will be set during initialization
    
    def get_strategy(self) -> ExecutionStrategy:
        """
        Prefill is compute-bound:
        - Maximize parallelism with large batch size
        - Use mixed precision (FP16) to reduce memory and increase throughput
        - Aggressive fusion to reduce memory bandwidth
        - Target high GPU utilization (90%)
        """
        return ExecutionStrategy(
            phase=ExecutionPhase.LLM_PREFILL,
            batch_size=32,  # Large batch for parallelism
            use_mixed_precision=True,  # FP16 is safe here
            enable_fusion=True,  # Aggressive fusion
            pin_kv_cache=False,  # KV cache not critical in prefill
            incremental_compute=False,  # Full computation
            target_gpu_utilization=0.90,
            max_gpu_memory_utilization=0.70,  # Conservative to avoid OOM
            description="Prefill: Compute-bound, large batch, mixed precision, aggressive fusion"
        )
    
    def prepare_for_execution(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare for prefill execution."""
        prepared = {}
        
        # Batch multiple sequences together if needed
        for key, tensor in inputs.items():
            if key == 'input_ids' and len(tensor.shape) == 2:
                # Ensure batch size matches target (pad or split if needed)
                target_batch = self.get_strategy().batch_size
                current_batch = tensor.shape[0]
                
                if current_batch < target_batch:
                    # Pad batch
                    padding = target_batch - current_batch
                    padded = torch.nn.functional.pad(tensor, (0, 0, 0, padding), value=0)
                    prepared[key] = padded
                elif current_batch > target_batch:
                    # Use target batch (process in chunks)
                    prepared[key] = tensor[:target_batch]
                else:
                    prepared[key] = tensor
            else:
                prepared[key] = tensor
        
        logger.debug(f"Prefill execution prepared: batch_size={prepared.get('input_ids', torch.tensor([])).shape[0] if 'input_ids' in prepared else 'unknown'}")
        return prepared
    
    def post_execution_cleanup(self) -> None:
        """Cleanup after prefill execution."""
        # Could trigger aggressive cache eviction if needed
        pass


class DecodeExecutionStrategy(ExecutionPhaseStrategy):
    """Decode phase (sequential generation with KV cache): memory-bound optimization."""
    
    def __init__(self):
        super().__init__(ExecutionPhase.LLM_DECODE)
        self.pinned_tensors: Dict[str, torch.Tensor] = {}
    
    def get_strategy(self) -> ExecutionStrategy:
        """
        Decode is memory-bound:
        - Small batch size (minimize KV cache memory)
        - Full precision (FP32 for numerical stability)
        - KV cache pinning (prevent eviction)
        - Incremental computation (only compute new token)
        - Target moderate GPU utilization (40-50%, decode is sequential)
        """
        return ExecutionStrategy(
            phase=ExecutionPhase.LLM_DECODE,
            batch_size=4,  # Small batch due to memory constraints
            use_mixed_precision=False,  # FP32 for stability
            enable_fusion=False,  # Fusion less beneficial for small batches
            pin_kv_cache=True,  # CRITICAL: Keep KV cache
            incremental_compute=True,  # Only new token
            target_gpu_utilization=0.45,  # Decode is sequential
            max_gpu_memory_utilization=0.60,  # Protect KV cache space
            description="Decode: Memory-bound, small batch, FP32, KV cache pinning"
        )
    
    def prepare_for_execution(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare for decode execution."""
        prepared = {}
        
        # Ensure batch size is small for decode
        for key, tensor in inputs.items():
            if key == 'input_ids' and len(tensor.shape) == 2:
                # Take only first token of each sequence (incremental)
                prepared[key] = tensor[:, -1:]  # Last token only
            elif key == 'past_key_values':
                # Preserve KV cache as-is
                prepared[key] = tensor
            else:
                prepared[key] = tensor
        
        logger.debug(f"Decode execution prepared: batch_size={prepared.get('input_ids', torch.tensor([])).shape[0] if 'input_ids' in prepared else 'unknown'}")
        return prepared
    
    def pin_kv_cache(self, kv_cache: torch.Tensor, priority: EvictionPriority = EvictionPriority.CRITICAL) -> None:
        """Pin KV cache to prevent eviction during decode."""
        # Mark tensor as critical to prevent eviction
        # Handle both single tensor and list of tensors (KV pairs)
        if isinstance(kv_cache, list):
            logger.debug(f"KV cache pinned: {len(kv_cache)} tensors, priority={priority.name}")
            self.pinned_tensors['kv_cache'] = kv_cache
        else:
            logger.debug(f"KV cache pinned: {kv_cache.shape}, priority={priority.name}")
            self.pinned_tensors['kv_cache'] = kv_cache
    
    def unpin_kv_cache(self) -> None:
        """Unpin KV cache after decode completion."""
        self.pinned_tensors.clear()
        logger.debug("KV cache unpinned")


class VisionExecutionStrategy(ExecutionPhaseStrategy):
    """Vision encoding phase: memory bandwidth optimization."""
    
    def __init__(self):
        super().__init__(ExecutionPhase.VISION_ENCODING)
    
    def get_strategy(self) -> ExecutionStrategy:
        """
        Vision is bandwidth-bound:
        - Medium batch size (balance parallelism and memory)
        - Mixed precision (FP16 safe)
        - Conv fusion (reduce memory accesses)
        - Target high utilization (85%)
        """
        return ExecutionStrategy(
            phase=ExecutionPhase.VISION_ENCODING,
            batch_size=16,  # Medium batch
            use_mixed_precision=True,
            enable_fusion=True,  # Conv fusion
            pin_kv_cache=False,
            incremental_compute=False,
            target_gpu_utilization=0.85,
            max_gpu_memory_utilization=0.70,
            description="Vision: Bandwidth-bound, medium batch, conv fusion"
        )


class PhaseAwareExecutor:
    """
    Executes computations with phase-specific optimization strategies.
    
    Adapts execution based on detected execution phase:
    - LLM Prefill: Compute-bound optimization (large batch, mixed precision)
    - LLM Decode: Memory-bound optimization (small batch, KV cache pinning)
    - Vision: Bandwidth-bound optimization (medium batch, conv fusion)
    - Multimodal: Cross-modal fusion optimization
    
    Integrates with PhaseAwareMemoryManager for semantic-driven memory management.
    """
    
    def __init__(
        self,
        total_gpu_memory_mb: float = 24000,
        enable_profiling: bool = True
    ):
        """
        Initialize phase-aware executor.
        
        Args:
            total_gpu_memory_mb: Total GPU memory in MB (default: 24GB)
            enable_profiling: Whether to profile execution
        """
        self.total_gpu_memory_mb = total_gpu_memory_mb
        self.enable_profiling = enable_profiling
        
        # Phase-specific strategies
        self.phase_strategies = {
            ExecutionPhase.LLM_PREFILL: PrefillExecutionStrategy(),
            ExecutionPhase.LLM_DECODE: DecodeExecutionStrategy(),
            ExecutionPhase.VISION_ENCODING: VisionExecutionStrategy(),
        }
        
        # Memory management
        self.memory_manager = PhaseAwareMemoryManager(total_gpu_memory_mb)
        self.lifetime_evictor = LifetimeBasedEvictor()
        
        # Statistics
        self.stats = {
            'total_executions': 0,
            'total_time_ms': 0.0,
            'phase_switches': 0,
            'current_phase': ExecutionPhase.UNKNOWN,
        }
        
        logger.info(
            "PhaseAwareExecutor initialized (GPU memory: %.0f MB, profiling: %s)",
            total_gpu_memory_mb,
            enable_profiling
        )
    
    def execute_with_phase_optimization(
        self,
        graph: Any,
        inputs: Dict[str, torch.Tensor],
        srg_nodes: Optional[List[Dict]] = None,
        srg_edges: Optional[List[Dict]] = None
    ) -> torch.Tensor:
        """
        Execute computation with phase-aware optimization.
        
        Args:
            graph: Computation graph (FX GraphModule or unified Graph interface)
            inputs: Input tensors
            srg_nodes: SRG nodes for lifetime analysis
            srg_edges: SRG edges for lifetime analysis
        
        Returns:
            Computation result
        """
        start_time = time.time()
        
        # Step 1: Detect execution phase
        phase = self._detect_phase(graph, inputs)
        
        # Step 2: Update memory strategy if phase changed
        if phase != self.stats['current_phase']:
            self.memory_manager.adjust_for_phase(phase)
            self.stats['phase_switches'] += 1
            self.stats['current_phase'] = phase
        
        # Step 3: Analyze tensor lifetimes (if SRG provided)
        if srg_nodes and srg_edges:
            self.lifetime_evictor.analyze_graph_lifetimes(srg_nodes, srg_edges)
        
        # Step 4: Execute with phase-specific strategy
        strategy = self.phase_strategies.get(phase)
        if strategy:
            logger.info(f"Executing {phase.value} phase: {strategy.get_strategy().description}")
            result = self._execute_with_strategy(graph, inputs, strategy)
        else:
            logger.warning(f"Unknown phase {phase}, using default execution")
            result = self._execute_default(graph, inputs)
        
        # Step 5: Update statistics
        execution_time_ms = (time.time() - start_time) * 1000
        self.stats['total_executions'] += 1
        self.stats['total_time_ms'] += execution_time_ms
        
        logger.debug(f"Phase {phase.value} execution completed in {execution_time_ms:.1f}ms")
        
        return result
    
    def _detect_phase(self, graph: Any, inputs: Dict[str, torch.Tensor]) -> ExecutionPhase:
        """
        Detect execution phase based on graph and inputs.
        
        Heuristics:
        - If KV cache present → decode
        - If large sequence length → prefill
        - If image inputs → vision
        - Else → forward
        """
        # Check for KV cache (indicates decode)
        if 'past_key_values' in inputs or 'kv_cache' in inputs:
            return ExecutionPhase.LLM_DECODE
        
        # Check for large sequence length (indicates prefill)
        if 'input_ids' in inputs:
            seq_len = inputs['input_ids'].shape[-1] if len(inputs['input_ids'].shape) > 1 else 1
            if seq_len > 10:  # Arbitrary threshold
                return ExecutionPhase.LLM_PREFILL
        
        # Check for image inputs (indicates vision)
        if 'pixel_values' in inputs or 'images' in inputs:
            return ExecutionPhase.VISION_ENCODING
        
        # Default
        return ExecutionPhase.FORWARD
    
    def _execute_with_strategy(
        self,
        graph: Any,
        inputs: Dict[str, torch.Tensor],
        strategy: ExecutionPhaseStrategy
    ) -> torch.Tensor:
        """Execute with phase-specific strategy."""
        exec_strategy = strategy.get_strategy()
        
        # Prepare inputs for strategy
        prepared_inputs = strategy.prepare_for_execution(inputs)
        
        # Enable mixed precision if configured
        if exec_strategy.use_mixed_precision:
            with torch.amp.autocast(device_type='cuda'):
                result = self._execute_graph(graph, prepared_inputs)
        else:
            result = self._execute_graph(graph, prepared_inputs)
        
        # Pin KV cache if decode
        if exec_strategy.pin_kv_cache and 'past_key_values' in inputs:
            if hasattr(strategy, 'pin_kv_cache'):
                strategy.pin_kv_cache(inputs['past_key_values'], EvictionPriority.CRITICAL)
        
        # Cleanup
        strategy.post_execution_cleanup()
        
        return result
    
    def _execute_graph(self, graph: Any, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute computation graph."""
        # Handle FX GraphModule
        if hasattr(graph, 'graph'):
            # This is an FX GraphModule
            return graph(*inputs.values())
        
        # Handle unified Graph interface
        elif hasattr(graph, 'execute'):
            return graph.execute(inputs)
        
        # Fallback: assume callable
        else:
            return graph(**inputs)
    
    def _execute_default(self, graph: Any, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Default execution without optimization."""
        return self._execute_graph(graph, inputs)
    
    def get_stats(self) -> Dict:
        """Get executor statistics."""
        result = dict(self.stats)
        result['avg_time_ms'] = self.stats['total_time_ms'] / max(1, self.stats['total_executions'])
        
        # Add per-phase stats
        for phase, strategy in self.phase_strategies.items():
            result[f'{phase.value}_stats'] = strategy.get_stats()
        
        # Add memory management stats
        result['memory_stats'] = self.memory_manager.get_stats()
        
        return result
    
    def report_memory_usage(
        self,
        available_bytes: int,
        total_bytes: int
    ) -> None:
        """Report current memory usage for adaptive management."""
        utilization = 1.0 - (available_bytes / total_bytes) if total_bytes > 0 else 0.0
        logger.debug(f"Memory usage: {utilization*100:.1f}% ({available_bytes} free of {total_bytes})")
