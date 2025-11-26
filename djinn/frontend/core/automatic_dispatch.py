"""
Automatic Dispatch System for Shape Inference (Graph Construction Phase)

Purpose: Infer shapes/dtypes during graph construction (capture phase)
Input: LazyTensors
Output: LazyTensor with inferred metadata
When: Called from __torch_dispatch__ during operation capture

This module uses PyTorch's meta tensor system to automatically infer shapes,
dtypes, and other metadata for operations without manual intervention.

Key Features:
1. Automatic shape inference via meta tensors
2. Handles ALL PyTorch operations (2000+)
3. No manual handlers needed (except truly special cases)
4. Scalable and maintainable

Architecture:
    LazyTensor → Meta Tensor → PyTorch Op → Meta Result → LazyTensor

Key Distinction:
- This is a TOOL used by shape inference, not shape inference itself
- Used during graph construction (capture phase)
- Converts LazyTensors → meta tensors → calls PyTorch → infers shape/dtype
- Returns new LazyTensor with inferred metadata

See also: universal_dispatcher.py (execution phase)
"""

import torch
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class AutomaticDispatch:
    """
    Automatic Dispatch System for Shape Inference (Graph Construction Phase)

    Purpose: Infer shapes/dtypes during graph construction (capture phase)
    Input: LazyTensors
    Output: LazyTensor with inferred metadata
    When: Called from __torch_dispatch__ during operation capture

    This module uses PyTorch's meta tensor system to automatically infer shapes,
    dtypes, and other metadata for operations without manual intervention.

    Key Distinction:
    - This is a TOOL used by shape inference, not shape inference itself
    - Used during graph construction (capture phase)
    - Converts LazyTensors → meta tensors → calls PyTorch → infers shape/dtype
    - Returns new LazyTensor with inferred metadata

    See also: universal_dispatcher.py (execution phase)
    """
    
    # Operations that require special handling (cannot use meta tensors)
    SPECIAL_OPS = {
        # Materialization operations
        'aten::item',
        'aten::numpy',
        'aten::to_numpy',
        'aten::cpu',
        'aten::cuda',
        'aten::to',
        
        # Operations that return non-tensors
        'aten::size',
        'aten::numel',
        'aten::dim',
        'aten::is_contiguous',
        
        # Operations with side effects
        'aten::print',
        'aten::save',
        'aten::load',
        
        # ✅ FIX: Operations that fail with meta tensors
        # softmax returns empty shape when called with meta tensors
        'aten::softmax',
        'aten::_softmax',
        
        # ✅ Week 3-4: Tuple-returning operations (handled by LazyTuple)
        'aten::split',
        'aten::chunk',
        'aten::unbind',
        'aten::hsplit',
        'aten::vsplit',
        'aten::dsplit',
        'aten::tensor_split',
        'aten::topk',
        'aten::sort',
        'aten::kthvalue',
        'aten::median',
        'aten::mode',
        'aten::qr',
        'aten::svd',
        'aten::eig',
        'aten::slogdet',
    }
    
    @classmethod
    def should_use_automatic_dispatch(cls, func: Callable, args: Tuple, kwargs: Dict) -> bool:
        """
        Determine if we should use automatic dispatch for this operation.
        
        Returns:
            bool: True if automatic dispatch should be used, False if special handling needed
        """
        # Get operation name
        op_name = cls._get_op_name(func)
        
        # Check if it's a special operation
        if op_name in cls.SPECIAL_OPS:
            return False
        
        # Check if function name suggests special handling
        if hasattr(func, '__name__'):
            func_name = func.__name__
            # Materialization operations
            if func_name in ('item', 'numpy', 'tolist', '__len__', '__bool__', '__int__', '__float__'):
                return False
            # ✅ Week 3-4: Tuple-returning operations (handled by LazyTuple)
            if func_name in ('split', 'chunk', 'unbind', 'hsplit', 'vsplit', 'dsplit', 'tensor_split',
                           'topk', 'sort', 'kthvalue', 'median', 'mode', 'qr', 'svd', 'eig', 'slogdet'):
                return False
        
        return True
    
    @classmethod
    def dispatch(cls, func: Callable, args: Tuple, kwargs: Dict, lazy_tensor_class: type) -> Any:
        """
        Automatically dispatch operation using meta tensors.
        
        Args:
            func: PyTorch function to call
            args: Function arguments (may contain LazyTensors)
            kwargs: Function keyword arguments (may contain LazyTensors)
            lazy_tensor_class: LazyTensor class for creating result
            
        Returns:
            LazyTensor or tuple of LazyTensors with inferred metadata
        """
        try:
            # Step 1: Convert all LazyTensors to meta tensors
            meta_args, arg_mapping = cls._to_meta_tensors(args)
            meta_kwargs, kwarg_mapping = cls._to_meta_tensors(kwargs)
            
            # Step 2: Call function with meta tensors (automatic shape inference!)
            with torch.device('meta'):
                meta_result = func(*meta_args, **meta_kwargs)
            
            # Step 3: Convert meta result back to LazyTensor(s)
            result = cls._from_meta_result(
                meta_result,
                func,
                args,
                kwargs,
                lazy_tensor_class,
                arg_mapping,
                kwarg_mapping
            )
            
            return result
            
        except Exception as e:
            # If automatic dispatch fails, log and return None (caller will use fallback)
            logger.debug(f"Automatic dispatch failed for {cls._get_op_name(func)}: {e}")
            return None
    
    @classmethod
    def _to_meta_tensors(cls, obj: Any) -> Tuple[Any, Dict[int, Any]]:
        """
        Convert LazyTensors to meta tensors recursively.
        
        Args:
            obj: Object to convert (can be tensor, list, tuple, dict, etc.)
            
        Returns:
            Tuple of (converted_obj, mapping) where mapping tracks original LazyTensors
        """
        mapping = {}
        
        def convert(x, path=()):
            if type(x).__name__ == 'LazyTensor':
                # Convert LazyTensor to meta tensor
                obj_id = id(x)
                mapping[obj_id] = x
                
                # Get shape, dtype, device from LazyTensor
                shape = x.shape if hasattr(x, 'shape') else torch.Size([])
                dtype = x.dtype if hasattr(x, 'dtype') else torch.float32
                
                # ✅ FIX: If shape is empty, this LazyTensor hasn't been inferred yet
                # We can't create a valid meta tensor, so return None to signal failure
                if not shape or len(shape) == 0:
                    # Check if this is a scalar operation (which legitimately has empty shape)
                    operation = object.__getattribute__(x, '_operation') if hasattr(x, '_operation') else ''
                    if operation not in ('aten::tensor', 'aten::scalar'):
                        logger.debug(f"Cannot convert LazyTensor to meta: empty shape for {operation}")
                        raise ValueError(f"LazyTensor has uninitialized shape for operation {operation}")
                
                # Create meta tensor with same metadata
                # Disable interception to avoid detach() issues
                from .interception_control import disable_interception, InterceptionContext
                with disable_interception(InterceptionContext.CONSTRUCTION):
                    # Handle scalar tensors (shape = ())
                    if len(shape) == 0:
                        meta_tensor = torch.empty((), dtype=dtype, device='meta')
                    else:
                        meta_tensor = torch.empty(shape, dtype=dtype, device='meta')
                return meta_tensor
                
            elif isinstance(x, torch.Tensor):
                # Regular tensor - convert to meta
                return torch.empty(x.shape, dtype=x.dtype, device='meta')
                
            elif isinstance(x, (list, tuple)):
                converted = [convert(item, path + (i,)) for i, item in enumerate(x)]
                return type(x)(converted)
                
            elif isinstance(x, dict):
                return {k: convert(v, path + (k,)) for k, v in x.items()}
                
            else:
                # Non-tensor types (scalars, strings, etc.) - return as-is
                return x
        
        converted = convert(obj)
        return converted, mapping
    
    @classmethod
    def _from_meta_result(
        cls,
        meta_result: Any,
        func: Callable,
        original_args: Tuple,
        original_kwargs: Dict,
        lazy_tensor_class: type,
        arg_mapping: Dict[int, Any],
        kwarg_mapping: Dict[int, Any]
    ) -> Any:
        """
        Convert meta tensor result back to LazyTensor(s).
        
        Args:
            meta_result: Result from calling func with meta tensors
            func: Original function
            original_args: Original arguments (with LazyTensors)
            original_kwargs: Original kwargs (with LazyTensors)
            lazy_tensor_class: LazyTensor class
            arg_mapping: Mapping of LazyTensor IDs to original LazyTensors
            kwarg_mapping: Mapping of LazyTensor IDs to original LazyTensors
            
        Returns:
            LazyTensor or structure of LazyTensors matching meta_result structure
        """
        op_name = cls._get_op_name(func)
        
        def convert(meta_obj):
            if isinstance(meta_obj, torch.Tensor):
                # Meta tensor result - convert to LazyTensor
                
                # Infer device from input LazyTensors
                inferred_device = cls._infer_device(original_args, original_kwargs)
                
                # Get metadata from meta tensor
                shape = meta_obj.shape
                dtype = meta_obj.dtype
                
                # ✅ DEBUG: Log shape for softmax
                if 'softmax' in op_name:
                    logger.info(f"🔍 Automatic dispatch for {op_name}: meta_obj.shape = {shape}")
                
                # Create LazyTensor with inferred metadata
                # ✅ FIX: Use correct import path for metadata_capture
                try:
                    from ..semantic.metadata_capture import get_metadata_capture
                    metadata = get_metadata_capture().capture_metadata(
                        operation=op_name,
                        inputs=list(original_args),
                        kwargs=original_kwargs
                    )
                except (ImportError, Exception):
                    # Fallback: use MetadataPlaceholder if metadata capture fails
                    from ...core.metadata import MetadataPlaceholder
                    metadata = MetadataPlaceholder(
                        operation=op_name,
                        inputs=tuple(original_args),
                        kwargs=original_kwargs
                    )
                
                return lazy_tensor_class(
                    operation=op_name,
                    inputs=list(original_args),
                    kwargs=original_kwargs,
                    shape=shape,
                    dtype=dtype,
                    device=inferred_device,
                    metadata=metadata
                )
                
            elif isinstance(meta_obj, (list, tuple)):
                converted = [convert(item) for item in meta_obj]
                return type(meta_obj)(converted)
                
            elif isinstance(meta_obj, dict):
                return {k: convert(v) for k, v in meta_obj.items()}
                
            else:
                # Non-tensor result (scalar, None, etc.)
                return meta_obj
        
        return convert(meta_result)
    
    @classmethod
    def _infer_device(cls, args: Tuple, kwargs: Dict):
        """
        Infer device from input arguments.
        
        ✅ FIX: Returns original device string (e.g., 'remote_accelerator:0')
        to preserve user-visible device, not the internal mapping.
        
        Args:
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            str or torch.device: Inferred device (prefers string to preserve remote_accelerator)
        """
        # Check kwargs first
        if 'device' in kwargs:
            device = kwargs['device']
            if isinstance(device, str):
                return device  # Return string to preserve remote_accelerator
            return device
        
        # Check args for LazyTensors
        def find_device(obj):
            if type(obj).__name__ == 'LazyTensor':
                # ✅ FIX: Check _original_device first to preserve remote_accelerator
                if hasattr(obj, '_original_device') and obj._original_device:
                    original = obj._original_device
                    # Return as string to preserve remote_accelerator
                    if isinstance(original, str):
                        return original
                    elif isinstance(original, torch.device):
                        device_str = str(original)
                        # If it's remote_accelerator, return as string
                        if 'remote_accelerator' in device_str or 'privateuseone' in device_str:
                            return device_str
                        return original
                    return original
                # Fallback to device property
                if hasattr(obj, 'device') and obj.device:
                    device_obj = obj.device
                    if isinstance(device_obj, torch.device):
                        device_str = str(device_obj)
                        # If it's remote_accelerator, return as string
                        if 'remote_accelerator' in device_str or 'privateuseone' in device_str:
                            return device_str
                        return device_obj
                    return device_obj
            elif isinstance(obj, torch.Tensor):
                return str(obj.device)  # Return string for consistency
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    device = find_device(item)
                    if device:
                        return device
            elif isinstance(obj, dict):
                for value in obj.values():
                    device = find_device(value)
                    if device:
                        return device
            return None
        
        # Search args
        for arg in args:
            device = find_device(arg)
            if device:
                return device
        
        # Search kwargs
        for value in kwargs.values():
            device = find_device(value)
            if device:
                return device
        
        # Default to CPU (as string)
        return 'cpu'
    
    @classmethod
    def _get_op_name(cls, func: Callable) -> str:
        """
        Get normalized operation name from function.
        
        Args:
            func: PyTorch function
            
        Returns:
            str: Normalized operation name (e.g., 'aten::add')
        """
        if hasattr(func, '__name__'):
            name = func.__name__
            
            # Handle torch.ops.aten.* format
            if hasattr(func, '__module__') and 'torch._ops' in func.__module__:
                return f'aten::{name}'
            
            # Handle torch.nn.functional.* format
            if hasattr(func, '__module__') and 'torch.nn.functional' in func.__module__:
                return f'aten::{name}'
            
            # Default format
            return f'aten::{name}'
        
        # Fallback
        return str(func)


def create_automatic_dispatcher() -> AutomaticDispatch:
    """
    Factory function to create an AutomaticDispatch instance.
    
    Returns:
        AutomaticDispatch: Configured dispatcher
    """
    return AutomaticDispatch()


# Global dispatcher instance
_global_dispatcher = None


def get_automatic_dispatcher() -> AutomaticDispatch:
    """
    Get the global automatic dispatcher instance.
    
    Returns:
        AutomaticDispatch: Global dispatcher
    """
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = create_automatic_dispatcher()
    return _global_dispatcher

