"""
Operation Classification System for Hybrid Execution Model

This module implements the three-class operation taxonomy that enables
hybrid execution: MATERIALIZATION_TRIGGER, REDUCTION_OPERATION, and
COMPUTE_OPERATION. This is the core of the improvement strategy outlined
in enhancement_plan.md.

Phase 6A Enhancement: Context-aware classification for operations with
context-dependent semantics (e.g., sum with/without dim parameter).

REFACTORING: Consolidated from materialization_control.py to provide
single source of truth for operation classification.
"""

from enum import Enum
from typing import Optional, Any, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class OperationClass(Enum):
    """
    Fundamental operation classification for execution strategy.
    
    Each operation falls into one of three categories that determine how
    it should be executed:
    """
    # Must execute immediately and return concrete values
    MATERIALIZATION_TRIGGER = "materialization_trigger"
    
    # Should execute where data resides to minimize transfer
    REDUCTION_OPERATION = "reduction_operation"  
    
    # Standard deferred execution
    COMPUTE_OPERATION = "compute_operation"
    
    # Operations that must materialize due to data-dependent output shapes
    SHAPE_DEPENDENT = "shape_dependent"
    
    # Operations that return tuples (need special handling)
    TUPLE_RETURNING = "tuple_returning"


class OperationClassifier:
    """
    Production-ready operation classifier with context-aware semantics.
    
    Phase 6A Enhancement: Now handles context-dependent operations where
    the return type and behavior depend on parameters (e.g., sum with/without dim).
    
    This classifier determines how each operation should be executed based
    on its semantics and expected return type. Conservative classification
    ensures correctness at the expense of some optimization opportunities.
    """
    
    # Operations that MUST materialize (return non-Tensor types)
    # Merged from materialization_control.PYTHON_PROTOCOL_OPS
    MATERIALIZATION_TRIGGERS: Set[str] = {
        # Boolean returns (control flow)
        'all', 'any', '__bool__', '__nonzero__',
        
        # Scalar returns  
        'item', '__int__', '__float__', '__index__',
        
        # Python type conversions
        'tolist', 'numpy',
        
        # Shape/device queries that need concrete values
        'size', 'numel', 'dim', 'is_cuda', 'get_device',
        
        # Explicit materialization
        'cpu', 'cuda', 'to',  # device transfers
        
        # Comparison operations that might be used for control flow
        '__lt__', '__le__', '__gt__', '__ge__', '__eq__', '__ne__',
        'equal',
    }
    
    # Python Protocol Operations (subset of MATERIALIZATION_TRIGGERS)
    # Kept for backward compatibility with materialization_control.py
    PYTHON_PROTOCOL_OPS: Set[str] = {
        # Boolean conversion (required by Python if/while)
        '__bool__',
        '__nonzero__',  # Python 2 compat
        
        # Numeric conversions (required by Python int()/float())
        '__int__',
        '__float__',
        '__index__',     # Required for indexing: list[tensor]
        
        # Data extraction (required for scalar access)
        'item',          # tensor.item() → Python number
        'tolist',        # tensor.tolist() → Python list
        'numpy',         # tensor.numpy() → numpy array
    }
    
    # Operations that dramatically reduce data size
    REDUCTION_OPERATIONS = {
        # Arg reductions (input: [N, M], output: [N] or scalar)
        'argmax', 'argmin', 'argsort',
        
        # Statistical reductions
        'sum', 'mean', 'std', 'var', 'prod',
        'max', 'min',  # when returning values only
        
        # Selection operations
        'topk', 'kthvalue', 'mode', 'median',
    }
    
    # Operations with data-dependent output shapes (MUST materialize)
    SHAPE_DEPENDENT_OPERATIONS = {
        'nonzero',      # Shape depends on data values
        'unique',       # Output size unknown until execution
        'where',        # Output size data-dependent
        'masked_select',# Size depends on mask values
    }
    
    # Operations that return tuples (need special handling)
    # Merged from materialization_control.TUPLE_RETURNING_OPS
    TUPLE_RETURNING_OPERATIONS: Dict[str, Optional[int]] = {
        # Splitting operations (variable number of outputs)
        'split': None,      # Variable number of splits
        'chunk': None,      # Variable number of chunks
        'unbind': None,     # Variable number of unbind results
        'hsplit': None,     # Variable number of splits
        'vsplit': None,     # Variable number of splits
        'dsplit': None,     # Variable number of splits
        'tensor_split': None,  # Variable number of splits
        
        # Operations returning (values, indices)
        'topk': 2,          # Returns (values, indices)
        'sort': 2,          # Returns (values, indices)
        'kthvalue': 2,      # Returns (values, indices)
        'median': 2,        # Returns (values, indices) sometimes
        'mode': 2,          # Returns (values, indices)
        
        # Matrix decomposition
        'qr': 2,            # Returns (Q, R)
        'svd': 3,           # Returns (U, S, V)
        'eig': 2,           # Returns (eigenvalues, eigenvectors)
        'slogdet': 2,       # Returns (sign, logabsdet)
    }
    
    # Tuple-returning operations as a set (for backward compatibility)
    # Kept for backward compatibility with materialization_control.py
    TUPLE_RETURNING_OPS: Set[str] = {
        # Splitting operations
        'split', 'chunk', 'unbind', 'hsplit', 'vsplit', 'dsplit', 'tensor_split',
        # Operations returning (values, indices)
        'topk', 'sort', 'kthvalue', 'median', 'mode',
        # Matrix decomposition
        'qr', 'svd', 'eig', 'slogdet',
    }
    
    # Context-dependent operations: behavior changes based on parameters
    CONTEXT_DEPENDENT_OPERATIONS = {
        'sum': {
            'returns_scalar': ['dim'],  # If 'dim' is None or not provided, returns scalar
            'materialization_if_scalar': True,
        },
        'mean': {
            'returns_scalar': ['dim'],
            'materialization_if_scalar': True,
        },
        'std': {
            'returns_scalar': ['dim'],
            'materialization_if_scalar': True,
        },
        'var': {
            'returns_scalar': ['dim'],
            'materialization_if_scalar': True,
        },
        'max': {
            'returns_tuple_sometimes': True,  # Returns (values, indices) if dim is specified
        },
        'min': {
            'returns_tuple_sometimes': True,
        },
    }
    
    @classmethod
    def classify(cls, operation: str, return_type: Optional[type] = None, 
                 args: Optional[Tuple] = None, kwargs: Optional[Dict] = None) -> OperationClass:
        """
        Classify operation based on name, return type, and execution context.
        
        Phase 6A Enhancement: Now supports context-dependent classification.
        
        Args:
            operation: Operation name (e.g., 'argmax', 'item', 'add')
            return_type: Expected return type (for future enhancement)
            args: Positional arguments passed to the operation
            kwargs: Keyword arguments passed to the operation
        
        Returns:
            OperationClass indicating how to execute this operation
        """
        # Normalize operation name (remove aten:: prefix if present)
        op_name = operation
        if '::' in op_name:
            op_name = op_name.split('::')[-1]
        
        # Materialization triggers take precedence
        if op_name in cls.MATERIALIZATION_TRIGGERS:
            return OperationClass.MATERIALIZATION_TRIGGER
        
        # Shape-dependent operations (must materialize)
        if op_name in cls.SHAPE_DEPENDENT_OPERATIONS:
            return OperationClass.SHAPE_DEPENDENT
        
        # Tuple-returning operations (special handling)
        if op_name in cls.TUPLE_RETURNING_OPERATIONS:
            return OperationClass.TUPLE_RETURNING
        
        # Context-dependent operations (analyze parameters)
        if op_name in cls.CONTEXT_DEPENDENT_OPERATIONS:
            return cls._classify_context_dependent(op_name, args, kwargs)
        
        # Check if it's a reduction
        if op_name in cls.REDUCTION_OPERATIONS:
            return OperationClass.REDUCTION_OPERATION
            
        # Default: normal compute operation
        return OperationClass.COMPUTE_OPERATION
    
    @classmethod
    def _classify_context_dependent(cls, operation: str, args: Optional[Tuple] = None,
                                   kwargs: Optional[Dict] = None) -> OperationClass:
        """
        Classify context-dependent operations based on their arguments.
        
        For example:
        - tensor.sum() -> MATERIALIZATION_TRIGGER (returns scalar)
        - tensor.sum(dim=0) -> REDUCTION_OPERATION (returns tensor)
        """
        kwargs = kwargs or {}
        context = cls.CONTEXT_DEPENDENT_OPERATIONS.get(operation, {})
        
        # Check if operation returns scalar
        if 'returns_scalar' in context:
            # If 'dim' is not in kwargs or is None, it returns scalar
            if 'dim' not in kwargs or kwargs.get('dim') is None:
                if context.get('materialization_if_scalar'):
                    return OperationClass.MATERIALIZATION_TRIGGER
        
        # Check if operation returns tuple sometimes
        if context.get('returns_tuple_sometimes'):
            if 'dim' in kwargs or (args and len(args) > 0):
                return OperationClass.TUPLE_RETURNING
        
        # Default to reduction for these operations
        return OperationClass.REDUCTION_OPERATION
    
    @classmethod
    def is_materialization_trigger(cls, operation: str) -> bool:
        """Check if operation is a materialization trigger."""
        op_name = operation
        if '::' in op_name:
            op_name = op_name.split('::')[-1]
        return op_name in cls.MATERIALIZATION_TRIGGERS
    
    @classmethod
    def is_reduction_operation(cls, operation: str) -> bool:
        """Check if operation is a reduction."""
        op_name = operation
        if '::' in op_name:
            op_name = op_name.split('::')[-1]
        return op_name in cls.REDUCTION_OPERATIONS
    
    @classmethod
    def is_shape_dependent(cls, operation: str) -> bool:
        """Check if operation has data-dependent output shape."""
        op_name = operation
        if '::' in op_name:
            op_name = op_name.split('::')[-1]
        return op_name in cls.SHAPE_DEPENDENT_OPERATIONS
    
    @classmethod
    def is_tuple_returning(cls, operation: str) -> bool:
        """Check if operation returns a tuple."""
        op_name = operation
        if '::' in op_name:
            op_name = op_name.split('::')[-1]
        return op_name in cls.TUPLE_RETURNING_OPERATIONS
    
    @classmethod
    def get_tuple_size(cls, operation: str) -> Optional[int]:
        """Get the expected tuple size for tuple-returning operations."""
        op_name = operation
        if '::' in op_name:
            op_name = op_name.split('::')[-1]
        return cls.TUPLE_RETURNING_OPERATIONS.get(op_name)



