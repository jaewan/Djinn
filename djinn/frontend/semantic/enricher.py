"""
Semantic enricher for LazyTensors.

Analyzes operations and produces rich semantic metadata (Refactoring #2).
"""
from typing import List, Any, Optional, Dict
import torch
from .metadata_registry import SemanticMetadata
from djinn.core.exceptions import Result, SemanticException
from djinn.core.semantic_metadata import ExecutionPhase, MemoryPattern, DataLineage
import logging
import time

logger = logging.getLogger(__name__)


class SemanticEnricher:
    """
    Enriches LazyTensors with semantic metadata.
    
    Separated from LazyTensor to follow Single Responsibility Principle.
    LazyTensor handles execution, SemanticEnricher handles semantics.
    """
    
    def __init__(self):
        self._module_context_tracker = None  # Lazy init
        self._phase_detector = None  # Lazy init
        self._last_debug_log = 0 # Rate-limiting for debug logs
    
    def enrich(self, operation: str, inputs: List[Any], kwargs: Dict,
               shape: Optional[torch.Size], dtype: Optional[torch.dtype]) -> Result[SemanticMetadata]:
        """
        Create semantic metadata for an operation.
        
        Args:
            operation: Operation name (e.g., "aten::matmul")
            inputs: Operation inputs
            kwargs: Operation keyword arguments
            shape: Inferred tensor shape
            dtype: Inferred tensor dtype
            
        Returns:
            Result[SemanticMetadata]: Enriched metadata
        """
        try:
            metadata = SemanticMetadata(
                operation_type=operation,
                tensor_shape=tuple(shape) if shape else None,
                dtype=str(dtype) if dtype else None,
                
                # Enrich with semantic analysis
                semantic_role=self._infer_semantic_role(operation, inputs, kwargs),
                model_module=self._get_module_context(),
                execution_phase=self._detect_phase(operation, inputs),
                data_lineage=self._track_lineage(inputs),
                
                # Memory pattern analysis
                memory_pattern=self._analyze_memory_pattern(operation, inputs),
                compute_intensity=self._estimate_compute_intensity(operation, shape),
                kv_cache_related=self._is_kv_cache_operation(operation, kwargs),
                
                # Additional context
                layer_depth=self._get_layer_depth(),
                is_activation=self._is_activation(operation, inputs),
                
                # Scheduling hints
                can_parallelize=self._can_parallelize(operation, inputs),
                priority=self._calculate_priority(operation),
                colocation_group=self._infer_colocation_group(operation, inputs)
            )
            
            return Result.ok(metadata)
        
        except Exception as e:
            if time.time() - self._last_debug_log > 10:  # 10 seconds
                logger.debug(f"Error: {e}")
                self._last_debug_log = time.time()
            return Result.err(SemanticException(
                f"Semantic enrichment failed for {operation}: {e}",
                context={'operation': operation, 'inputs_count': len(inputs)}
            ))
    
    def _infer_semantic_role(self, operation: str, inputs: List[Any], 
                            kwargs: Dict) -> Optional[str]:
        """
        Infer the semantic role of this operation.
        
        Examples:
            - matmul with 3D inputs → "attention_score_computation"
            - softmax → "softmax_activation"
            - linear → "linear_projection"
        """
        op_name = operation.split("::")[-1]
        
        # Check module context first
        try:
            if self._module_context_tracker is None:
                from .module_context import get_module_context_tracker
                self._module_context_tracker = get_module_context_tracker()
            
            context = self._module_context_tracker.get_current_context()
            if context:
                return self._module_context_tracker.infer_semantic_role(operation, context)
        except ImportError:
            pass  # Module context not available
        
        # Fallback heuristics
        if "matmul" in op_name and len(inputs) >= 2:
            # Check if attention-like (3D tensors)
            if self._looks_like_attention(inputs):
                return "attention_score_computation"
            return "matrix_multiplication"
        
        # Activation functions
        if op_name in ["relu", "gelu", "sigmoid", "tanh", "softmax"]:
            return f"{op_name}_activation"
        
        # Normalization
        if "norm" in op_name or "layer_norm" in op_name:
            return "normalization"
        
        # Projection
        if op_name == "linear":
            return "linear_projection"
        
        return None
    
    def _looks_like_attention(self, inputs: List[Any]) -> bool:
        """Check if operation looks like attention (3D+ tensors)."""
        for inp in inputs[:2]:  # Check first two inputs
            if hasattr(inp, 'shape') and len(inp.shape) >= 3:
                return True
        return False
    
    def _get_module_context(self) -> Optional[str]:
        """Get current module execution context."""
        try:
            if self._module_context_tracker is None:
                from .module_context import get_module_context_tracker
                self._module_context_tracker = get_module_context_tracker()
            
            context = self._module_context_tracker.get_current_context()
            return context.module_path if context else None
        except ImportError:
            return None
    
    def _detect_phase(self, operation: str, inputs: List[Any]) -> Optional[str]:
        """Detect execution phase (prefill/decode/etc)."""
        try:
            if self._phase_detector is None:
                from .phase_detector import get_phase_detector
                self._phase_detector = get_phase_detector()
            
            phase = self._phase_detector.detect_phase(operation, inputs, {})
            if hasattr(phase, 'value'):
                return phase.value if hasattr(phase, 'value') else str(phase)
            return str(phase) if phase else None
        except ImportError:
            return None
    
    def _track_lineage(self, inputs: List[Any]) -> Optional[dict]:
        lineage = {
            'source_tensors': [],
            'source_modules': [],
            'transformation_chain': [],
            'modality': None
        }
        
        # Track from inputs
        for inp in inputs:
            if hasattr(inp, 'id'):  # LazyTensor
                lineage['source_tensors'].append(inp.id)
                
                # Propagate lineage from inputs
                if hasattr(inp, 'metadata'):
                    inp_metadata = inp.metadata
                    if inp_metadata and hasattr(inp_metadata, 'data_lineage') and inp_metadata.data_lineage:
                        lineage['source_modules'].extend(getattr(inp_metadata.data_lineage, 'source_modules', []))
                        lineage['modality'] = getattr(inp_metadata.data_lineage, 'modality', None)
        
        return lineage if lineage['source_tensors'] else None
    
    def _analyze_memory_pattern(self, operation: str, inputs: List[Any]) -> str:
        """Analyze memory access pattern."""
        op_name = operation.split("::")[-1]
        
        # KV cache is persistent
        if self._is_kv_cache_operation(operation, {}):
            return "persistent"
        
        # Convolution and matmul are streaming
        if op_name in ["conv2d", "conv1d", "matmul", "mm", "bmm"]:
            return "streaming"
        
        # Random operations
        if "dropout" in op_name or "random" in op_name:
            return "random"
        
        return "streaming"  # Default
    
    def _estimate_compute_intensity(self, operation: str, 
                                    shape: Optional[torch.Size]) -> float:
        """
        Estimate compute intensity (FLOPs per byte).
        
        Higher values = more compute-bound.
        """
        op_name = operation.split("::")[-1]
        
        if op_name in ["matmul", "mm", "bmm"]:
            return 10.0  # High intensity
        elif op_name in ["conv2d", "conv1d"]:
            return 8.0
        elif op_name in ["relu", "sigmoid", "tanh"]:
            return 1.0  # Low intensity
        
        return 5.0  # Medium default
    
    def _is_kv_cache_operation(self, operation: str, kwargs: Dict) -> bool:
        """Check if operation is KV cache related."""
        op_name = operation.lower()
        
        # Check operation name
        if any(kv in op_name for kv in ['cache', 'past_key', 'past_value']):
            return True
        
        # Check kwargs
        if any(kv in str(kwargs).lower() for kv in ['cache', 'past']):
            return True
        
        return False
    
    def _get_layer_depth(self) -> Optional[int]:
        """Get layer depth from module context."""
        try:
            if self._module_context_tracker is None:
                from .module_context import get_module_context_tracker
                self._module_context_tracker = get_module_context_tracker()
            
            context = self._module_context_tracker.get_current_context()
            return context.layer_depth if context else None
        except ImportError:
            return None
    
    def _is_activation(self, operation: str, inputs: List[Any]) -> bool:
        """Check if this produces an activation tensor vs weight."""
        # Check if any input is a parameter (would be a weight operation)
        for inp in inputs:
            if isinstance(inp, torch.nn.Parameter):
                return False
        
        # Activation functions produce activations
        op_name = operation.split("::")[-1]
        if op_name in ["relu", "gelu", "sigmoid", "tanh", "softmax", "dropout"]:
            return True
        
        # Default to activation for dynamic operations
        return True
    
    def _can_parallelize(self, operation: str, inputs: List[Any]) -> bool:
        """Check if operation can be parallelized."""
        op_name = operation.split("::")[-1]
        
        # Most operations can be parallelized
        if op_name in ["matmul", "conv2d", "relu", "add", "mul"]:
            return True
        
        # Sequential operations cannot
        if "lstm" in op_name or "gru" in op_name:
            return False
        
        return True
    
    def _calculate_priority(self, operation: str) -> int:
        """
        Calculate scheduling priority (0-10).
        
        Higher priority = should execute earlier.
        """
        op_name = operation.split("::")[-1]
        
        # Critical path operations
        if op_name in ["matmul", "conv2d"]:
            return 7
        
        # Normal priority
        if op_name in ["add", "mul", "relu"]:
            return 5
        
        # Low priority (can be recomputed)
        if op_name in ["reshape", "view", "transpose"]:
            return 2
        
        return 5  # Default medium
    
    def _infer_colocation_group(self, operation: str, inputs: List[Any]) -> Optional[str]:
        """Infer colocation group for this operation."""
        # KV cache operations should be colocated
        if self._is_kv_cache_operation(operation, {}):
            return "kv_cache"
        
        # Could add more heuristics here
        return None


# Global instance
_enricher = None

def get_semantic_enricher() -> SemanticEnricher:
    """Get global semantic enricher instance."""
    global _enricher
    if _enricher is None:
        _enricher = SemanticEnricher()
    return _enricher

