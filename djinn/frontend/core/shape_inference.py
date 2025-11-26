"""
Phase 6B: Shape Inference System

Tracks tensor shapes through the lazy DAG without materializing.
Infers output shapes for operations based on inputs, enabling
control flow that depends on tensor shapes.

Key idea: Lazy shape evaluation - only materialize if we can't infer the shape.
"""

from typing import Optional, Tuple, List, Union, Any
import numpy as np
import torch


class ShapeInference:
    """
    Infers output shapes for tensor operations without materialization.
    
    For shape-dependent operations (nonzero, unique, etc.), still requires
    materialization, but this module makes that decision explicit.
    """
    
    # Operations and their shape transformation rules
    SHAPE_RULES = {
        # Identity operations (shape unchanged)
        'transpose': lambda x, *dims: _transpose_shape(x, dims),
        'permute': lambda x, *dims: _permute_shape(x, dims),
        'contiguous': lambda x: x,
        'clone': lambda x: x,
        'detach': lambda x: x,
        
        # Reshape operations
        'reshape': lambda x, shape: tuple(shape) if isinstance(shape, (list, tuple)) else (shape,),
        'view': lambda x, shape: tuple(shape) if isinstance(shape, (list, tuple)) else (shape,),
        'unsqueeze': lambda x, dim: _unsqueeze_shape(x, dim),
        'squeeze': lambda x, dim=None: _squeeze_shape(x, dim),
        'flatten': lambda x, start=0, end=-1: _flatten_shape(x, start, end),
        
        # Repeat operations
        'repeat': lambda x, *sizes: _repeat_shape(x, sizes),
        'tile': lambda x, *dims: _tile_shape(x, dims),
        
        # Concatenation and stacking
        'cat': lambda shapes, dim: _cat_shape(shapes, dim),
        'stack': lambda shapes, dim: _stack_shape(shapes, dim),
        
        # Selection and indexing
        'narrow': lambda x, dim, start, length: _narrow_shape(x, dim, length),
        'index_select': lambda x, dim, indices: _index_select_shape(x, dim, indices),
        'gather': lambda x, dim, indices: _gather_shape(x, dim, indices.shape),
        'scatter': lambda x, dim, indices, src: x,  # Output shape = input shape
        
        # Slicing (handled via __getitem__)
        '__getitem__': lambda x, key: x,  # Conservative: assume same shape
        
        # Reduction without dim (scalar output)
        'sum_scalar': lambda x: (),
        'mean_scalar': lambda x: (),
        'std_scalar': lambda x: (),
        'var_scalar': lambda x: (),
        'max_scalar': lambda x: (),
        'min_scalar': lambda x: (),
        'prod_scalar': lambda x: (),
        'all_scalar': lambda x: (),
        'any_scalar': lambda x: (),
        
        # Reduction with dim (removes dimension)
        'sum': lambda x, dim=None, keepdim=False: _reduction_shape(x, dim, keepdim),
        'mean': lambda x, dim=None, keepdim=False: _reduction_shape(x, dim, keepdim),
        'std': lambda x, dim=None, keepdim=False: _reduction_shape(x, dim, keepdim),
        'var': lambda x, dim=None, keepdim=False: _reduction_shape(x, dim, keepdim),
        'max': lambda x, dim=None, keepdim=False: _reduction_shape(x, dim, keepdim),
        'min': lambda x, dim=None, keepdim=False: _reduction_shape(x, dim, keepdim),
        'prod': lambda x, dim=None, keepdim=False: _reduction_shape(x, dim, keepdim),
        
        # Argmax/argmin
        'argmax': lambda x, dim=None, keepdim=False: _argmax_shape(x, dim, keepdim),
        'argmin': lambda x, dim=None, keepdim=False: _argmax_shape(x, dim, keepdim),
        
        # Top-k and sorting
        'topk': lambda x, k, dim=None: _topk_shape(x, k, dim),
        'sort': lambda x, dim=None: _sort_shape(x, dim),
        'argsort': lambda x, dim=None: _sort_shape(x, dim),
        
        # Matrix operations
        'matmul': lambda x, y: _matmul_shape(x, y),
        '__matmul__': lambda x, y: _matmul_shape(x, y),
        'mm': lambda x, y: _mm_shape(x, y),
        'bmm': lambda x, y: _bmm_shape(x, y),
        'dot': lambda x, y: (),  # Returns scalar
        
        # Broadcasting operations
        '__add__': lambda x, y: _broadcast_shape(x, y),
        '__sub__': lambda x, y: _broadcast_shape(x, y),
        '__mul__': lambda x, y: _broadcast_shape(x, y),
        '__truediv__': lambda x, y: _broadcast_shape(x, y),
        '__floordiv__': lambda x, y: _broadcast_shape(x, y),
        '__mod__': lambda x, y: _broadcast_shape(x, y),
        '__pow__': lambda x, y: _broadcast_shape(x, y),
        # Also support non-dunder names (for aten::add, etc.)
        'add': lambda x, y: _broadcast_shape(x, y),
        'sub': lambda x, y: _broadcast_shape(x, y),
        'mul': lambda x, y: _broadcast_shape(x, y),
        'div': lambda x, y: _broadcast_shape(x, y),
        'truediv': lambda x, y: _broadcast_shape(x, y),
        'floordiv': lambda x, y: _broadcast_shape(x, y),
        'mod': lambda x, y: _broadcast_shape(x, y),
        'pow': lambda x, y: _broadcast_shape(x, y),
        # In-place operations (add_, sub_, etc.) - same shape as input
        'add_': lambda x, y: _broadcast_shape(x, y),
        'sub_': lambda x, y: _broadcast_shape(x, y),
        'mul_': lambda x, y: _broadcast_shape(x, y),
        'div_': lambda x, y: _broadcast_shape(x, y),
        'truediv_': lambda x, y: _broadcast_shape(x, y),
        'floordiv_': lambda x, y: _broadcast_shape(x, y),
        'mod_': lambda x, y: _broadcast_shape(x, y),
        'pow_': lambda x, y: _broadcast_shape(x, y),
        
        # Factory functions (no input shapes, shape determined by arguments)
        'arange': lambda *args, **kwargs: _arange_shape(*args, **kwargs),
        'linspace': lambda *args, **kwargs: _linspace_shape(*args, **kwargs),
        'logspace': lambda *args, **kwargs: _logspace_shape(*args, **kwargs),
        'zeros': lambda *args, **kwargs: _zeros_shape(*args, **kwargs),
        'ones': lambda *args, **kwargs: _ones_shape(*args, **kwargs),
        'empty': lambda *args, **kwargs: _empty_shape(*args, **kwargs),
        'randn': lambda *args, **kwargs: _randn_shape(*args, **kwargs),
        'rand': lambda *args, **kwargs: _rand_shape(*args, **kwargs),
        # Tensor factory methods (new_ones, new_zeros, etc.) - shape from first arg
        'new_ones': lambda self, *args, **kwargs: _new_tensor_shape(self, *args, **kwargs),
        'new_zeros': lambda self, *args, **kwargs: _new_tensor_shape(self, *args, **kwargs),
        'new_empty': lambda self, *args, **kwargs: _new_tensor_shape(self, *args, **kwargs),
        
        # Element-wise operations (preserve shape)
        'abs': lambda x: x,
        'sqrt': lambda x: x,
        'exp': lambda x: x,
        'log': lambda x: x,
        'sin': lambda x: x,
        'cos': lambda x: x,
        'tan': lambda x: x,
        'sigmoid': lambda x: x,
        'tanh': lambda x: x,
        
        # Normalization and activation
        'batch_norm': lambda x, *args: x,
        'layer_norm': lambda x, *args: x,
        'group_norm': lambda x, *args: x,
        'instance_norm': lambda x, *args: x,
        'softmax': lambda x, dim=-1: x,  # Output shape = input shape
        'log_softmax': lambda x, dim=-1: x,
        'gelu': lambda x: x,
        'silu': lambda x: x,
        'dropout': lambda x, *args: x if isinstance(x, tuple) and len(x) > 0 else x,  # Preserve shape, but ensure it's a tuple
        
        # Transformer-specific operations (Phase 7)
        'attention': lambda q, k, v: _attention_shape(q, k, v),  # Multi-head attention
        'scaled_dot_product_attention': lambda q, k, v: _attention_shape(q, k, v),
    }
    
    @classmethod
    def infer_shape(cls, operation: str, input_shapes: Union[Tuple, List[Tuple]], 
                   *args, **kwargs) -> Optional[Tuple]:
        """
        Infer output shape for an operation.
        
        Args:
            operation: Operation name
            input_shapes: Shape(s) of input tensor(s)
            *args: Positional arguments to the operation
            **kwargs: Keyword arguments to the operation
        
        Returns:
            Output shape as tuple, or None if shape can't be inferred
        """
        # Normalize operation name
        op_name = operation
        if '::' in op_name:
            op_name = op_name.split('::')[-1]
        
        # If not in rules, return None (can't infer)
        if op_name not in cls.SHAPE_RULES:
            return None
        
        try:
            rule = cls.SHAPE_RULES[op_name]
            
            # Factory functions (no input shapes, shape determined by arguments)
            factory_ops = ['arange', 'linspace', 'logspace', 'zeros', 'ones', 'empty', 'randn', 'rand']
            if op_name in factory_ops:
                # Factory functions don't have input shapes, only args/kwargs
                result_shape = rule(*args, **kwargs)
                return result_shape if result_shape else None
            
            # input_shapes is a tuple of input shapes
            # For single-input operations, extract the first shape
            # For multi-input operations (cat, stack, matmul, mm, bmm, add, mul, etc.), pass all shapes
            multi_input_ops = ['cat', 'stack', 'matmul', '__matmul__', 'mm', 'bmm',
                               '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__',
                               'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow',
                               'add_', 'sub_', 'mul_', 'div_', 'truediv_', 'floordiv_', 'mod_', 'pow_']
            
            if op_name in multi_input_ops:
                # Multiple input shapes
                if len(input_shapes) == 1:
                    # Single shape provided - check if second arg is a scalar
                    # For broadcasting ops (add, mul, etc.), scalar should be treated as empty shape
                    broadcasting_ops = ('__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__',
                                      'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow',
                                      'add_', 'sub_', 'mul_', 'div_', 'truediv_', 'floordiv_', 'mod_', 'pow_')
                    if op_name in broadcasting_ops:
                        # Broadcasting operation - if second arg is scalar, use empty shape
                        if len(args) > 0:
                            second_arg = args[0]
                            # Check if second arg is a scalar (Python int/float or 0-dimensional tensor)
                            is_scalar = False
                            if not isinstance(second_arg, (tuple, list, torch.Tensor)) and not hasattr(second_arg, 'shape'):
                                # Python scalar (int/float)
                                is_scalar = True
                            elif hasattr(second_arg, 'shape') and len(second_arg.shape) == 0:
                                # 0-dimensional tensor (scalar tensor)
                                is_scalar = True
                            
                            if is_scalar:
                                # Second argument is a scalar, treat as empty shape for broadcasting
                                return rule(input_shapes[0], ())
                        # Try with single shape (might work for some operations)
                        return rule(input_shapes[0], *args, **kwargs)
                    else:
                        # Non-broadcasting multi-input op - try with single shape
                        return rule(input_shapes[0], *args, **kwargs)
                elif len(input_shapes) == 2:
                    # Two shapes (matmul, add, etc.)
                    return rule(input_shapes[0], input_shapes[1], *args, **kwargs)
                else:
                    # More than 2 shapes
                    return rule(input_shapes, *args, **kwargs)
            else:
                # Single input shape for most operations
                result_shape = rule(input_shapes[0], *args, **kwargs)
                
                # Check for invalid scalar shapes for shape-preserving operations
                # These operations should never return scalar if input wasn't scalar
                shape_preserving_ops = ['dropout', 'relu', 'gelu', 'silu', 'softmax', 'log_softmax',
                                       'layer_norm', 'batch_norm', 'instance_norm', 'group_norm']
                if op_name in shape_preserving_ops:
                    input_shape = input_shapes[0]
                    # If input has non-scalar shape but result is scalar, return None to trigger materialization
                    if isinstance(input_shape, tuple) and len(input_shape) > 0:
                        if isinstance(result_shape, tuple) and len(result_shape) == 0:
                            # Invalid: shape-preserving op returned scalar for non-scalar input
                            return None  # Trigger materialization to get actual shape
                
                return result_shape
        
        except Exception as e:
            # If inference fails, return None
            # Debug: log the exception for troubleshooting
            if op_name in ('add', '__add__', 'sub', '__sub__', 'mul', '__mul__'):
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Shape inference failed for {op_name}: {e}")
            return None


# Shape transformation helper functions

def _transpose_shape(shape: Tuple, dims: Tuple) -> Tuple:
    """Compute shape after transpose."""
    if len(dims) == 0:
        return tuple(reversed(shape))
    dim0, dim1 = dims[0], dims[1] if len(dims) > 1 else None
    if dim1 is None:
        return shape
    shape_list = list(shape)
    shape_list[dim0], shape_list[dim1] = shape_list[dim1], shape_list[dim0]
    return tuple(shape_list)


def _permute_shape(shape: Tuple, dims: Tuple) -> Tuple:
    """Compute shape after permute."""
    return tuple(shape[d] for d in dims[0] if isinstance(dims[0], (list, tuple)))


def _unsqueeze_shape(shape: Tuple, dim: int) -> Tuple:
    """Compute shape after unsqueeze."""
    shape_list = list(shape)
    shape_list.insert(dim, 1)
    return tuple(shape_list)


def _squeeze_shape(shape: Tuple, dim: Optional[int] = None) -> Tuple:
    """Compute shape after squeeze."""
    if dim is None:
        return tuple(s for s in shape if s != 1)
    if shape[dim] == 1:
        return shape[:dim] + shape[dim+1:]
    return shape


def _flatten_shape(shape: Tuple, start: int = 0, end: int = -1) -> Tuple:
    """Compute shape after flatten."""
    if end == -1:
        end = len(shape) - 1
    size = np.prod(shape[start:end+1])
    return shape[:start] + (int(size),) + shape[end+1:]


def _reduction_shape(shape: Tuple, dim: Optional[int] = None, keepdim: bool = False) -> Tuple:
    """Compute shape after reduction (sum, mean, etc.)."""
    if dim is None:
        return () if not keepdim else tuple(1 for _ in shape)
    if keepdim:
        shape_list = list(shape)
        shape_list[dim] = 1
        return tuple(shape_list)
    return shape[:dim] + shape[dim+1:]


def _argmax_shape(shape: Tuple, dim: Optional[int] = None, keepdim: bool = False) -> Tuple:
    """Compute shape after argmax/argmin."""
    if dim is None:
        return ()  # Returns scalar
    return _reduction_shape(shape, dim, keepdim)


def _topk_shape(shape: Tuple, k: int, dim: Optional[int] = None) -> Tuple:
    """Compute shape after topk (returns values, indices)."""
    if dim is None:
        dim = -1
    shape_list = list(shape)
    shape_list[dim] = k
    return tuple(shape_list)


def _sort_shape(shape: Tuple, dim: Optional[int] = None) -> Tuple:
    """Compute shape after sort/argsort."""
    return shape  # Output shape = input shape


def _cat_shape(shapes: List[Tuple], dim: int) -> Optional[Tuple]:
    """Compute shape after concatenation."""
    if not shapes:
        return None
    # All shapes must match except on concat dimension
    shape_list = list(shapes[0])
    for shape in shapes[1:]:
        if len(shape) != len(shape_list):
            return None  # Incompatible ranks
        for i, (s1, s2) in enumerate(zip(shape_list, shape)):
            if i == dim:
                shape_list[i] += s2
            elif s1 != s2:
                return None  # Incompatible shapes
    return tuple(shape_list)


def _stack_shape(shapes: List[Tuple], dim: int) -> Optional[Tuple]:
    """Compute shape after stacking."""
    if not shapes:
        return None
    # All shapes must match
    for shape in shapes[1:]:
        if shape != shapes[0]:
            return None
    shape_list = list(shapes[0])
    shape_list.insert(dim, len(shapes))
    return tuple(shape_list)


def _narrow_shape(shape: Tuple, dim: int, length: int) -> Tuple:
    """Compute shape after narrow (slicing)."""
    shape_list = list(shape)
    shape_list[dim] = length
    return tuple(shape_list)


def _index_select_shape(shape: Tuple, dim: int, indices_shape: Tuple) -> Tuple:
    """Compute shape after index_select."""
    shape_list = list(shape)
    shape_list[dim] = indices_shape[0]  # Assume 1D indices
    return tuple(shape_list)


def _gather_shape(shape: Tuple, dim: int, indices_shape: Tuple) -> Tuple:
    """Compute shape after gather."""
    # Output shape matches indices shape
    return indices_shape


def _matmul_shape(x: Tuple, y: Tuple) -> Optional[Tuple]:
    """Compute shape after matrix multiplication."""
    # Handle various cases: (n), (n, k), (b, n, k), etc.
    if len(x) == 1 and len(y) == 1:
        return ()  # Dot product
    if len(x) == 1:
        return y[:-2] + (y[-1],)
    if len(y) == 1:
        return x[:-1]
    if len(x) == 2 and len(y) == 2:
        return (x[0], y[1])
    if len(x) >= 2 and len(y) >= 2:
        # Batch matrix multiply
        return x[:-1] + (y[-1],)
    return None


def _mm_shape(x: Tuple, y: Tuple) -> Optional[Tuple]:
    """Compute shape after mm (2D only)."""
    if len(x) == 2 and len(y) == 2:
        return (x[0], y[1])
    return None


def _bmm_shape(x: Tuple, y: Tuple) -> Optional[Tuple]:
    """Compute shape after bmm (batched 2D)."""
    if len(x) == 3 and len(y) == 3 and x[0] == y[0]:
        return (x[0], x[1], y[2])
    return None


def _broadcast_shape(x: Tuple, y: Tuple) -> Optional[Tuple]:
    """Compute shape after broadcasting."""
    # Simple broadcasting: align right and match dimensions
    max_len = max(len(x), len(y))
    x_padded = (1,) * (max_len - len(x)) + x
    y_padded = (1,) * (max_len - len(y)) + y
    
    result = []
    for sx, sy in zip(x_padded, y_padded):
        if sx == sy:
            result.append(sx)
        elif sx == 1:
            result.append(sy)
        elif sy == 1:
            result.append(sx)
        else:
            return None  # Incompatible shapes
    return tuple(result)


def _repeat_shape(shape: Tuple, sizes: Tuple) -> Tuple:
    """Compute shape after repeat."""
    # repeat(*sizes) repeats each dimension by the corresponding size
    # [2, 3].repeat(2, 1) -> [4, 3]
    if not isinstance(sizes, (tuple, list)):
        sizes = (sizes,)
    
    result = []
    for i, s in enumerate(shape):
        if i < len(sizes):
            result.append(s * sizes[i])
        else:
            result.append(s)
    return tuple(result)


def _tile_shape(shape: Tuple, dims: Tuple) -> Tuple:
    """Compute shape after tile."""
    # tile(*dims) tiles the shape
    # Similar to repeat but dims are aligned left instead of right
    if not isinstance(dims, (tuple, list)):
        dims = (dims,)
    
    # Align to right first
    max_len = max(len(shape), len(dims))
    shape_padded = (1,) * (max_len - len(shape)) + shape
    dims_padded = (1,) * (max_len - len(dims)) + dims
    
    result = []
    for s, d in zip(shape_padded, dims_padded):
        result.append(s * d)
    return tuple(result)


def _attention_shape(q_shape: Tuple, k_shape: Tuple, v_shape: Tuple) -> Tuple:
    """
    Compute shape after multi-head attention.
    
    Attention output shape matches query input shape:
    Query: [batch, seq_len, hidden] 
    Key: [batch, seq_len, hidden]
    Value: [batch, seq_len, hidden]
    Output: [batch, seq_len, hidden]
    
    For multi-head: [batch, num_heads, seq_len, head_dim]
    Output: [batch, num_heads, seq_len, head_dim]
    """
    # Output shape = query shape
    return q_shape


# Factory function shape inference helpers

def _arange_shape(*args, **kwargs) -> Tuple:
    """
    Infer shape for torch.arange(start, end, step, ...).
    
    Shape is 1D: (length,) where length = ceil((end - start) / step)
    
    Note: torch.arange always returns a 1D tensor, even for scalar inputs.
    """
    if len(args) == 0:
        return ()  # No arguments, can't infer
    
    # Helper to extract int value from tensor or scalar
    def _extract_int(value):
        """Extract integer value from tensor or scalar."""
        if isinstance(value, (int, float)):
            return int(value)
        elif isinstance(value, torch.Tensor):
            # If it's a tensor, extract the value
            # Handle both scalar tensors and 1-element tensors
            if value.numel() == 1:
                return int(value.item())
            else:
                # Multi-element tensor - can't use as arange argument
                return None
        elif hasattr(value, 'item'):  # LazyTensor or similar
            try:
                return int(value.item())
            except:
                return None
        return None
    
    # torch.arange(end) -> shape (end,)
    if len(args) == 1:
        end = _extract_int(args[0])
        if end is not None:
            return (max(0, end),)
    
    # torch.arange(start, end, step) -> shape (ceil((end-start)/step),)
    if len(args) >= 2:
        start = _extract_int(args[0])
        end = _extract_int(args[1])
        step = _extract_int(args[2]) if len(args) >= 3 else 1
        
        if start is None or end is None:
            return ()  # Can't infer
        
        if step is None:
            step = 1
        
        if step == 0:
            return ()  # Invalid step
        
        length = max(0, int((end - start) / step))
        return (length,)
    
    return ()  # Can't infer


def _linspace_shape(*args, **kwargs) -> Tuple:
    """Infer shape for torch.linspace(start, end, steps)."""
    if len(args) >= 3:
        steps = args[2]
        if isinstance(steps, (int, float)):
            return (int(steps),)
    return ()  # Can't infer


def _logspace_shape(*args, **kwargs) -> Tuple:
    """Infer shape for torch.logspace(start, end, steps)."""
    if len(args) >= 3:
        steps = args[2]
        if isinstance(steps, (int, float)):
            return (int(steps),)
    return ()  # Can't infer


def _zeros_shape(*args, **kwargs) -> Tuple:
    """Infer shape for torch.zeros(*size)."""
    if 'size' in kwargs:
        size = kwargs['size']
        if isinstance(size, (tuple, list)):
            return tuple(size)
        elif isinstance(size, int):
            return (size,)
    elif len(args) > 0:
        # torch.zeros(d1, d2, ...) or torch.zeros((d1, d2, ...))
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        else:
            return tuple(args)
    return ()  # Can't infer


def _ones_shape(*args, **kwargs) -> Tuple:
    """Infer shape for torch.ones(*size)."""
    return _zeros_shape(*args, **kwargs)  # Same as zeros


def _empty_shape(*args, **kwargs) -> Tuple:
    """Infer shape for torch.empty(*size)."""
    return _zeros_shape(*args, **kwargs)  # Same as zeros


def _randn_shape(*args, **kwargs) -> Tuple:
    """Infer shape for torch.randn(*size)."""
    return _zeros_shape(*args, **kwargs)  # Same as zeros


def _rand_shape(*args, **kwargs) -> Tuple:
    """Infer shape for torch.rand(*size)."""
    return _zeros_shape(*args, **kwargs)  # Same as zeros


def _new_tensor_shape(self, *args, **kwargs) -> Tuple:
    """
    Infer shape for tensor.new_ones(size), tensor.new_zeros(size), etc.
    
    Args:
        self: The tensor (LazyTensor) - not used for shape, but needed for signature
        *args: Size argument(s) - can be int, tuple, or list
        **kwargs: Optional dtype, device, etc.
    
    Returns:
        Shape tuple
    """
    if len(args) == 0:
        return ()  # No size argument, can't infer
    
    size_arg = args[0]
    
    # Handle different size argument types
    if isinstance(size_arg, (tuple, list)):
        # Size is a tuple/list: new_ones((2, 3)) -> shape (2, 3)
        # Handle empty tuple: new_ones(()) -> shape () (scalar)
        if len(size_arg) == 0:
            return ()
        return tuple(int(s) for s in size_arg)
    elif isinstance(size_arg, (int, float)):
        # Size is a single int: new_ones(5) -> shape (5,)
        return (int(size_arg),)
    elif isinstance(size_arg, torch.Tensor):
        # Size is a tensor: convert to tuple
        if size_arg.numel() == 0:
            return ()  # Empty tensor -> scalar
        elif size_arg.numel() == 1:
            return (int(size_arg.item()),)
        else:
            return tuple(int(s) for s in size_arg.tolist())
    else:
        return ()  # Can't infer
