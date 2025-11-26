"""
LazyTuple: Tuple of LazyTensors with deferred unpacking.

This is the PROPER solution to tuple-returning operations.
Replaces eager materialization from Week 1-2.

Design:
- Inherits from tuple (standard Python type)
- Elements are LazyTensors
- Materialization happens on demand (when unpacked or accessed)
- Transparent to user (works like normal tuple)

Performance:
- No network transfer until elements used
- Can optimize across multiple elements
- Enables fusion opportunities

Example:
    x = torch.randn(1000, device='remote_accelerator:0')
    a, b = x.split(500)  # Returns LazyTuple (no execution)
    
    result = a + b       # Still lazy
    final = result.cpu() # ONE execution, minimal transfer
"""

from typing import Any, Iterator, Union, List, Optional, Dict
import torch
import logging

logger = logging.getLogger(__name__)


class LazyTuple(tuple):
    """
    Tuple of LazyTensors with deferred materialization.
    
    Behaves like Python tuple but preserves laziness until elements accessed.
    """
    
    def __new__(cls, elements, operation_metadata=None):
        """
        Create LazyTuple.
        
        Args:
            elements: List/tuple of LazyTensors
            operation_metadata: Optional metadata about source operation
        """
        instance = super().__new__(cls, elements)
        instance._operation_metadata = operation_metadata or {}
        instance._materialized_cache = {}
        return instance
    
    def __repr__(self):
        """User-friendly repr."""
        count = len(self)
        types = [type(e).__name__ for e in self]
        return f"LazyTuple({count} elements: {types})"
    
    def __getitem__(self, index):
        """
        Access element by index.
        
        Returns LazyTensor (still lazy), NOT materialized.
        """
        element = super().__getitem__(index)
        
        # If already concrete, return as-is
        if not isinstance(element, torch.Tensor) or type(element).__name__ != 'LazyTensor':
            return element
        
        # Return LazyTensor (maintains laziness)
        logger.debug(f"LazyTuple[{index}] accessed (still lazy)")
        return element
    
    def materialize(self):
        """
        Materialize entire tuple.
        
        Returns:
            Regular Python tuple of concrete tensors
        """
        from .lazy_tensor import LazyTensor
        
        logger.debug(f"Materializing LazyTuple ({len(self)} elements)")
        
        concrete_elements = []
        for i, element in enumerate(self):
            if isinstance(element, LazyTensor):
                concrete = element.materialize()
                concrete_elements.append(concrete)
            else:
                concrete_elements.append(element)
        
        return tuple(concrete_elements)
    
    def __iter__(self):
        """
        Iterate over elements.
        
        Returns LazyTensors (still lazy).
        """
        return super().__iter__()
    
    @classmethod
    def from_split(cls, base_tensor, split_size_or_sections, dim=0):
        """
        Create LazyTuple from split operation.
        
        Args:
            base_tensor: LazyTensor to split
            split_size_or_sections: Size of each chunk (int) or list of sizes
            dim: Dimension to split along
        
        Returns:
            LazyTuple of split chunks (lazy)
        """
        from .lazy_tensor import LazyTensor
        
        # Handle both split_size (int) and sections (list)
        if isinstance(split_size_or_sections, (list, tuple)):
            # List of sizes - one chunk per size
            num_chunks = len(split_size_or_sections)
            chunks = []
            for i, size in enumerate(split_size_or_sections):
                chunk = LazyTensor(
                    operation='aten::split',
                    inputs=[base_tensor],
                    kwargs={
                        'split_size_or_sections': split_size_or_sections,
                        'dim': dim,
                        '_chunk_index': i,
                        '_total_chunks': num_chunks
                    },
                    shape=cls._infer_chunk_shape(base_tensor, size, dim),
                    dtype=base_tensor.dtype if hasattr(base_tensor, 'dtype') else torch.float32,
                    device=base_tensor.device if hasattr(base_tensor, 'device') else None
                )
                chunks.append(chunk)
        else:
            # Single split_size - calculate number of chunks
            split_size = split_size_or_sections
            total_size = base_tensor.shape[dim] if hasattr(base_tensor, 'shape') else None
            
            if total_size is None:
                # Can't infer - create placeholder (will be resolved during materialization)
                num_chunks = 1  # Conservative estimate
            else:
                num_chunks = (total_size + split_size - 1) // split_size
            
            # Create LazyTensor for each chunk
            chunks = []
            for i in range(num_chunks):
                chunk = LazyTensor(
                    operation='aten::split',
                    inputs=[base_tensor],
                    kwargs={
                        'split_size_or_sections': split_size,
                        'dim': dim,
                        '_chunk_index': i,
                        '_total_chunks': num_chunks
                    },
                    shape=cls._infer_chunk_shape(base_tensor, split_size, dim),
                    dtype=base_tensor.dtype if hasattr(base_tensor, 'dtype') else torch.float32,
                    device=base_tensor.device if hasattr(base_tensor, 'device') else None
                )
                chunks.append(chunk)
        
        metadata = {
            'source_operation': 'split',
            'split_size_or_sections': split_size_or_sections,
            'dim': dim,
            'num_chunks': len(chunks)
        }
        
        return cls(chunks, operation_metadata=metadata)
    
    @staticmethod
    def _infer_chunk_shape(base_tensor, chunk_size, dim):
        """Infer shape of a chunk from split operation."""
        if not hasattr(base_tensor, 'shape'):
            return torch.Size([])
        
        shape = list(base_tensor.shape)
        if dim < len(shape):
            shape[dim] = chunk_size
        return torch.Size(shape)
    
    @classmethod
    def from_chunk(cls, base_tensor, chunks, dim=0):
        """
        Create LazyTuple from chunk operation.
        
        Args:
            base_tensor: LazyTensor to chunk
            chunks: Number of chunks
            dim: Dimension to chunk along
        
        Returns:
            LazyTuple of chunks (lazy)
        """
        from .lazy_tensor import LazyTensor
        
        total_size = base_tensor.shape[dim] if hasattr(base_tensor, 'shape') else None
        if total_size is None:
            # Can't infer - create placeholder
            chunk_size = 1
            num_chunks = chunks
        else:
            chunk_size = (total_size + chunks - 1) // chunks
            num_chunks = chunks
        
        chunk_list = []
        for i in range(num_chunks):
            chunk = LazyTensor(
                operation='aten::chunk',
                inputs=[base_tensor],
                kwargs={
                    'chunks': chunks,
                    'dim': dim,
                    '_chunk_index': i,
                    '_total_chunks': num_chunks
                },
                shape=cls._infer_chunk_shape(base_tensor, chunk_size, dim),
                dtype=base_tensor.dtype if hasattr(base_tensor, 'dtype') else torch.float32,
                device=base_tensor.device if hasattr(base_tensor, 'device') else None
            )
            chunk_list.append(chunk)
        
        metadata = {
            'source_operation': 'chunk',
            'chunks': chunks,
            'dim': dim,
            'num_chunks': num_chunks
        }
        
        return cls(chunk_list, operation_metadata=metadata)
    
    @classmethod
    def from_unbind(cls, base_tensor, dim=0):
        """
        Create LazyTuple from unbind operation.
        
        Args:
            base_tensor: LazyTensor to unbind
            dim: Dimension to unbind along
        
        Returns:
            LazyTuple of unbound tensors (lazy)
        """
        from .lazy_tensor import LazyTensor
        
        total_size = base_tensor.shape[dim] if hasattr(base_tensor, 'shape') else None
        if total_size is None:
            num_chunks = 1  # Conservative estimate
        else:
            num_chunks = total_size
        
        unbind_list = []
        for i in range(num_chunks):
            # Infer shape: remove dimension at dim
            if hasattr(base_tensor, 'shape'):
                shape = list(base_tensor.shape)
                shape.pop(dim)
                inferred_shape = torch.Size(shape)
            else:
                inferred_shape = torch.Size([])
            
            chunk = LazyTensor(
                operation='aten::unbind',
                inputs=[base_tensor],
                kwargs={
                    'dim': dim,
                    '_chunk_index': i,
                    '_total_chunks': num_chunks
                },
                shape=inferred_shape,
                dtype=base_tensor.dtype if hasattr(base_tensor, 'dtype') else torch.float32,
                device=base_tensor.device if hasattr(base_tensor, 'device') else None
            )
            unbind_list.append(chunk)
        
        metadata = {
            'source_operation': 'unbind',
            'dim': dim,
            'num_chunks': num_chunks
        }
        
        return cls(unbind_list, operation_metadata=metadata)


# ============================================================================
# Helper Functions
# ============================================================================

def is_lazy_tuple(obj) -> bool:
    """Check if object is LazyTuple."""
    return isinstance(obj, LazyTuple)

