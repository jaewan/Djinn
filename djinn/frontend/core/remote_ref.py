"""
RemoteRef: Lazy Output References for bandwidth-efficient remote execution

Implements the v2.3 architecture feature: "Lazy Output References"

Problem Solved:
- LLM models return multiple outputs: (logits, hidden_states, kv_cache)
- Transferring all outputs wastes gigabytes of bandwidth
- User often only needs the logits
- Hidden states are intermediate artifacts

Solution:
- Server returns RemoteRef objects instead of concrete tensors
- Refs contain tensor ID, shape, dtype (tiny, bytes of data)
- Data only transferred if user explicitly accesses it
- Lazy materialization on demand

Bandwidth Savings:
- Without RemoteRef: Transfer 1.3GB logits + 10GB hidden_states = 11.3GB
- With RemoteRef: Transfer 100 bytes of refs = ~99.9% reduction
- User calls ref.to('cuda:0') only for needed tensors

Example:
    output = model(input_ids)  # Returns RemoteRef
    print(output.shape)        # No transfer (metadata only)
    logits = output.to('cuda:0')  # Only now transferred
"""

import logging
import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class RemoteRefMetadata:
    """Metadata about remote tensor (no actual data)."""
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: str = "cuda:0"
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate size from shape and dtype if not provided."""
        if self.size_bytes == 0 and self.shape:
            num_elements = 1
            for dim in self.shape:
                if dim > 0:
                    num_elements *= dim
            element_size = torch.empty(0, dtype=self.dtype).element_size()
            self.size_bytes = num_elements * element_size


class RemoteRef:
    """
    Lazy reference to a tensor on remote server.
    
    Instead of transferring tensor data immediately, server sends:
    - Tensor ID
    - Shape
    - Dtype
    - Device
    
    Actual tensor data only transferred on demand (lazy materialization).
    
    Usage:
        # Server returns RemoteRef instead of tensor
        ref = model(input_ids)  # RemoteRef (bytes of data)
        
        # Access metadata without transfer
        print(ref.shape)   # Tensor shape
        print(ref.dtype)   # Data type
        
        # Materialize tensor on demand
        tensor = ref.cpu()  # Triggers network transfer
        tensor = ref.to('cuda:0')  # Transfer to specific device
    
    Benefits:
    - Zero bandwidth for unused outputs
    - Transparent interface (works like torch.Tensor)
    - Enables selective materialization
    """
    
    def __init__(self, metadata: RemoteRefMetadata, fetch_fn=None):
        """
        Initialize remote reference.
        
        Args:
            metadata: RemoteRefMetadata with tensor info
            fetch_fn: Callable to fetch tensor from server (async)
        """
        self.metadata = metadata
        self.fetch_fn = fetch_fn
        self._cached_tensor: Optional[torch.Tensor] = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Tensor shape (no transfer)."""
        return self.metadata.shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Tensor dtype (no transfer)."""
        return self.metadata.dtype
    
    @property
    def device(self) -> str:
        """Remote device location."""
        return self.metadata.device
    
    @property
    def size_bytes(self) -> int:
        """Estimated size in bytes (no transfer)."""
        return self.metadata.size_bytes
    
    def __repr__(self) -> str:
        """String representation showing lazy status."""
        cached = "cached" if self._cached_tensor is not None else "lazy"
        return (
            f"RemoteRef({cached}, shape={self.shape}, dtype={self.dtype}, "
            f"size={self.size_bytes / 1024**2:.1f}MB)"
        )
    
    async def _fetch_async(self) -> torch.Tensor:
        """Fetch tensor from remote server (async)."""
        if self._cached_tensor is not None:
            return self._cached_tensor
        
        if self.fetch_fn is None:
            raise RuntimeError(
                f"Cannot fetch RemoteRef {self.metadata.tensor_id}: "
                f"no fetch function provided"
            )
        
        logger.debug(f"Fetching tensor {self.metadata.tensor_id} from server")
        
        # Call fetch function (async)
        tensor = await self.fetch_fn(self.metadata.tensor_id)
        
        # Cache for future access
        self._cached_tensor = tensor
        
        logger.debug(
            f"✅ Fetched tensor {self.metadata.tensor_id}: "
            f"{tensor.shape} {tensor.dtype}"
        )
        
        return tensor
    
    def _fetch_sync(self) -> torch.Tensor:
        """Fetch tensor from remote server (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            should_close = True
        else:
            should_close = False
        
        try:
            tensor = loop.run_until_complete(self._fetch_async())
        finally:
            if should_close:
                loop.close()
        
        return tensor
    
    def cpu(self) -> torch.Tensor:
        """Materialize tensor on CPU."""
        tensor = self._fetch_sync()
        return tensor.cpu()
    
    def cuda(self, device: Optional[int] = None) -> torch.Tensor:
        """Materialize tensor on CUDA."""
        tensor = self._fetch_sync()
        if device is None:
            return tensor.cuda()
        else:
            return tensor.cuda(device)
    
    def to(self, device: str) -> torch.Tensor:
        """Materialize tensor on specified device."""
        tensor = self._fetch_sync()
        return tensor.to(device=device)
    
    def numpy(self):
        """Convert to numpy array."""
        tensor = self._fetch_sync()
        return tensor.detach().cpu().numpy()
    
    def item(self):
        """Extract scalar value."""
        tensor = self._fetch_sync()
        return tensor.item()
    
    # Support tensor operations on materialized version
    def __add__(self, other):
        """Addition operator."""
        tensor = self._fetch_sync()
        return tensor + other
    
    def __mul__(self, other):
        """Multiplication operator."""
        tensor = self._fetch_sync()
        return tensor * other
    
    def __getitem__(self, key):
        """Indexing operator."""
        tensor = self._fetch_sync()
        return tensor[key]
    
    def __len__(self):
        """Length (first dimension)."""
        return self.shape[0] if self.shape else 0


class RemoteRefTuple:
    """
    Tuple-like container for RemoteRefs.
    
    Enables unpacking of multiple outputs without materializing all:
    
        output_refs = model(input)  # RemoteRefTuple((ref1, ref2, ref3))
        logits, hidden, kv = output_refs  # Refs, not tensors
        
        # Only materialize logits when needed
        logits_tensor = logits.cpu()
    """
    
    def __init__(self, refs: Tuple[RemoteRef, ...]):
        """Initialize tuple of RemoteRefs."""
        self.refs = tuple(refs)
    
    def __len__(self) -> int:
        """Number of refs in tuple."""
        return len(self.refs)
    
    def __getitem__(self, index: int) -> RemoteRef:
        """Get ref at index."""
        return self.refs[index]
    
    def __iter__(self):
        """Iterate over refs."""
        return iter(self.refs)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RemoteRefTuple({self.refs})"
    
    def materialize(self, indices=None) -> Tuple[torch.Tensor, ...]:
        """
        Materialize specified refs (or all if indices=None).
        
        Enables selective materialization to save bandwidth.
        
        Args:
            indices: List of indices to materialize, or None for all
        
        Returns:
            Tuple of tensors
        """
        if indices is None:
            indices = range(len(self.refs))
        
        tensors = []
        for i in indices:
            ref = self.refs[i]
            tensor = ref._fetch_sync()
            tensors.append(tensor)
        
        return tuple(tensors)


def create_remote_ref(tensor_id: str, shape: Tuple[int, ...], dtype: torch.dtype,
                      fetch_fn=None) -> RemoteRef:
    """
    Factory function to create RemoteRef.
    
    Args:
        tensor_id: Unique identifier for tensor on server
        shape: Tensor shape
        dtype: Data type
        fetch_fn: Async function to fetch tensor from server
    
    Returns:
        RemoteRef instance
    """
    metadata = RemoteRefMetadata(
        tensor_id=tensor_id,
        shape=shape,
        dtype=dtype
    )
    return RemoteRef(metadata, fetch_fn=fetch_fn)


def create_remote_ref_tuple(tensor_ids: Tuple[str, ...],
                           shapes: Tuple[Tuple[int, ...], ...],
                           dtypes: Tuple[torch.dtype, ...],
                           fetch_fn=None) -> RemoteRefTuple:
    """
    Factory function to create RemoteRefTuple.
    
    Args:
        tensor_ids: Tuple of tensor IDs on server
        shapes: Tuple of tensor shapes
        dtypes: Tuple of data types
        fetch_fn: Async function to fetch tensors from server
    
    Returns:
        RemoteRefTuple instance
    """
    refs = tuple(
        RemoteRef(
            RemoteRefMetadata(tensor_id=tid, shape=s, dtype=dt),
            fetch_fn=fetch_fn
        )
        for tid, s, dt in zip(tensor_ids, shapes, dtypes)
    )
    return RemoteRefTuple(refs)

