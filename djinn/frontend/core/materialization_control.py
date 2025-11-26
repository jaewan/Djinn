"""
Materialization Control - Determines when LazyTensor must materialize.

ARCHITECTURE:
  Operations that require immediate materialization:
  
  1. PYTHON_PROTOCOL_OPS: Python language requires concrete values
     - __bool__, __int__, __float__, item(), etc.
     - No choice - Python won't accept LazyTensor
  
  2. Everything else: LAZY by default (~2000 operations)
     - Tuple operations (split, chunk, etc.) return LazyTuple
     - All other operations return LazyTensor
"""

from typing import Set, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Category 1: Python Protocol Operations (PERMANENT)
# ============================================================================

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


# ============================================================================
# Category 2: Tuple Operations (for reference - handled by LazyTuple)
# ============================================================================

TUPLE_RETURNING_OPS: Set[str] = {
    # Splitting operations
    'split',
    'chunk',
    'unbind',
    'hsplit',
    'vsplit',
    'dsplit',
    'tensor_split',
    
    # Operations returning (values, indices)
    'topk',
    'sort',
    'kthvalue',
    'median',
    'mode',
    
    # Matrix decomposition
    'qr',
    'svd',
    'eig',
    'slogdet',
}


# ============================================================================
# Configuration
# ============================================================================

class MaterializationConfig:
    """Global configuration for materialization behavior."""
    pass


_config = MaterializationConfig()


def get_config() -> MaterializationConfig:
    """Get global materialization config."""
    return _config


# ============================================================================
# Decision Logic
# ============================================================================

def should_materialize_immediately(func_name: str) -> Tuple[bool, str]:
    """
    Determine if operation requires immediate materialization.
    
    Args:
        func_name: Operation name (e.g., 'item', 'matmul', 'split')
    
    Returns:
        (should_materialize: bool, reason: str)
    
    Examples:
        >>> should_materialize_immediately('item')
        (True, 'python_protocol')
        
        >>> should_materialize_immediately('split')
        (False, 'lazy_tuple')  # Returns LazyTuple
        
        >>> should_materialize_immediately('matmul')
        (False, 'lazy_default')  # Returns LazyTensor
    """
    # Python protocols (always materialize - Python requires concrete values)
    if func_name in PYTHON_PROTOCOL_OPS:
        return True, 'python_protocol'
    
    # Everything else is lazy by default
    # Tuple operations return LazyTuple, others return LazyTensor
    return False, 'lazy_default'

