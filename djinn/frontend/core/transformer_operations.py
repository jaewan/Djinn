"""
Phase 7A: Advanced Transformer Operations

Implements specialized handling for common transformer operations:
- Layer normalization
- Attention mechanisms  
- Activation functions (GELU, SiLU)
- Sequence operations (dropout, masking)

This module ensures Djinn can correctly handle transformer models
by providing semantic understanding of these high-level operations.
"""

from typing import Optional, Tuple, Any, Dict
import torch
from enum import Enum


class TransformerOpType(Enum):
    """Classification of transformer operations."""
    NORMALIZATION = "normalization"      # Layer norm, batch norm
    ACTIVATION = "activation"            # GELU, ReLU, SiLU, etc
    ATTENTION = "attention"              # Multi-head attention
    SEQUENCE = "sequence"                # Dropout, masking, padding
    PROJECTION = "projection"            # Linear projections
    EMBEDDING = "embedding"              # Token/positional embeddings


class TransformerOperationClassifier:
    """Classify transformer operations for specialized handling."""
    
    # Normalization operations
    NORMALIZATION_OPS = {
        'layer_norm', 'layernorm',
        'batch_norm', 'batchnorm',
        'group_norm', 'groupnorm',
        'instance_norm', 'instancenorm',
        'rms_norm', 'rmsnorm',
    }
    
    # Activation functions
    ACTIVATION_OPS = {
        'gelu', 'gelu_new', 'gelu_python',
        'relu', 'relu6',
        'silu', 'swish',
        'mish',
        'elu', 'selu',
        'hard_swish', 'hard_sigmoid',
        'leaky_relu',
    }
    
    # Attention operations
    ATTENTION_OPS = {
        'attention', 'self_attention', 'multihead_attention',
        'scaled_dot_product_attention',
        'flash_attention', 'attention_v2',
    }
    
    # Sequence operations
    SEQUENCE_OPS = {
        'dropout',
        'pad', 'pad_sequence',
        'mask', 'apply_mask',
        'attention_mask',
    }
    
    # Projection operations
    PROJECTION_OPS = {
        'linear', 'dense',
        'projection', 'project',
        'embedding', 'token_embedding', 'positional_embedding',
    }
    
    @classmethod
    def classify(cls, op_name: str) -> Optional[TransformerOpType]:
        """Classify a transformer operation."""
        op_lower = op_name.lower()
        
        # Check in specific order (longer patterns first to avoid substring matches)
        # Sequence ops should be checked before attention (mask/padding patterns)
        if any(op in op_lower for op in cls.SEQUENCE_OPS):
            return TransformerOpType.SEQUENCE
        
        if any(op in op_lower for op in cls.ATTENTION_OPS):
            return TransformerOpType.ATTENTION
        
        if any(op in op_lower for op in cls.NORMALIZATION_OPS):
            return TransformerOpType.NORMALIZATION
        
        if any(op in op_lower for op in cls.ACTIVATION_OPS):
            return TransformerOpType.ACTIVATION
        
        if any(op in op_lower for op in cls.PROJECTION_OPS):
            return TransformerOpType.PROJECTION
        
        return None


class TransformerOperationOptimizer:
    """
    Optimize execution of transformer operations.
    
    Determines best execution strategy (local vs remote) based on:
    - Operation type
    - Input size
    - Computation/communication ratio
    """
    
    def __init__(self, network_bandwidth_gbps: float = 10.0):
        self.network_bandwidth_bytes_per_sec = network_bandwidth_gbps * 1e9 / 8
    
    def should_execute_remotely(self, op_type: TransformerOpType,
                               input_shapes: Tuple[Tuple, ...],
                               operation_name: str = None) -> bool:
        """
        Decide whether to execute a transformer operation remotely.
        
        Strategy:
        - NORMALIZATION: Local (low compute cost, small data)
        - ACTIVATION: Local (element-wise, minimal overhead)
        - ATTENTION: Remote if >100K tokens (high compute)
        - SEQUENCE: Local (data dependent, can't defer)
        - PROJECTION: Remote if >100K input (linear, high compute)
        """
        try:
            # Estimate input size
            input_size = self._estimate_input_size(input_shapes)
            
            if op_type == TransformerOpType.NORMALIZATION:
                # Normalization is cheap, keep local
                return False
            
            elif op_type == TransformerOpType.ACTIVATION:
                # Activations are element-wise, keep local
                return False
            
            elif op_type == TransformerOpType.ATTENTION:
                # Attention is expensive, execute remotely if large
                # Assume batch size > 1 or seq_len > 512
                return input_size > 100_000  # >100K elements
            
            elif op_type == TransformerOpType.SEQUENCE:
                # Sequence ops often depend on data, keep local
                return False
            
            elif op_type == TransformerOpType.PROJECTION:
                # Projections (linear layers) benefit from remote execution
                return input_size > 100_000
            
            return False
        
        except Exception:
            return False
    
    def _estimate_input_size(self, input_shapes: Tuple[Tuple, ...]) -> int:
        """Estimate total input size in elements."""
        total = 0
        for shape in input_shapes:
            if isinstance(shape, (tuple, list)):
                size = 1
                for dim in shape:
                    if isinstance(dim, int):
                        size *= dim
                total += size
        return total


class TransformerSemantics:
    """
    Semantic information about transformer operations.
    
    Helps LazyTensor understand:
    - What operations preserve semantics
    - Which operations require immediate execution
    - How to fuse operations
    """
    
    # Operations that preserve tensor semantics
    SHAPE_PRESERVING = {
        'gelu', 'relu', 'silu', 'mish',
        'layer_norm', 'dropout',
        'add', 'mul', 'div',
    }
    
    # Operations that change semantics significantly
    SEMANTICS_CHANGING = {
        'softmax', 'argmax', 'topk',
        'attention',  # Changes meaning of elements
        'embedding',  # Maps indices to embeddings
    }
    
    # Fusible operation patterns
    FUSIBLE_PATTERNS = {
        # Common transformer patterns
        ('layer_norm', 'gelu'): 'layer_norm_gelu',
        ('linear', 'gelu'): 'linear_gelu',
        ('softmax', 'matmul'): 'softmax_matmul',
        ('dropout', 'add'): 'dropout_residual',
    }
    
    @classmethod
    def is_shape_preserving(cls, op_name: str) -> bool:
        """Check if operation preserves tensor shape."""
        op_lower = op_name.lower()
        return any(op in op_lower for op in cls.SHAPE_PRESERVING)
    
    @classmethod
    def can_fuse(cls, op1: str, op2: str) -> bool:
        """Check if two operations can be fused."""
        key = (op1.lower(), op2.lower())
        return key in cls.FUSIBLE_PATTERNS


# Global optimizer instance
_transformer_optimizer = TransformerOperationOptimizer()


def get_transformer_optimizer() -> TransformerOperationOptimizer:
    """Get global transformer operation optimizer."""
    return _transformer_optimizer


def classify_transformer_op(op_name: str) -> Optional[TransformerOpType]:
    """Classify a transformer operation."""
    return TransformerOperationClassifier.classify(op_name)


def should_execute_transformer_op_remotely(op_type: TransformerOpType,
                                          input_shapes: Tuple[Tuple, ...]) -> bool:
    """Decide whether to execute a transformer operation remotely."""
    return _transformer_optimizer.should_execute_remotely(op_type, input_shapes)

