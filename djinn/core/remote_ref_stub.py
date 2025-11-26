"""
RemoteRefStub: Lightweight stub for lazy materialization (Output Skeletonization).

Moved from frontend to core to avoid circular imports between frontend and server.
"""

import logging
from typing import Tuple, Optional
import torch

logger = logging.getLogger(__name__)


# Global tensor registry for storing tensors in stubs
_tensor_registry = {}
_registry_lock = None  # Will be initialized if needed


def _init_registry_lock():
    """Initialize registry lock if needed."""
    global _registry_lock
    if _registry_lock is None:
        import threading
        _registry_lock = threading.Lock()


def store_tensor(ref_id: str, tensor: torch.Tensor) -> None:
    """
    Store tensor in global registry.
    
    Args:
        ref_id: Unique reference ID
        tensor: Tensor to store
    """
    _init_registry_lock()
    
    with _registry_lock:
        _tensor_registry[ref_id] = tensor
        logger.debug(f"Stored tensor {ref_id} in registry ({tensor.numel() * tensor.element_size() / 1024**2:.1f}MB)")


def retrieve_tensor(ref_id: str) -> Optional[torch.Tensor]:
    """
    Retrieve tensor from global registry.
    
    Args:
        ref_id: Reference ID
    
    Returns:
        Tensor if found, None otherwise
    """
    _init_registry_lock()
    
    with _registry_lock:
        tensor = _tensor_registry.get(ref_id)
        if tensor is not None:
            logger.debug(f"Retrieved tensor {ref_id} from registry")
        return tensor


def release_tensor(ref_id: str) -> bool:
    """
    Release tensor from registry.
    
    Args:
        ref_id: Reference ID
    
    Returns:
        True if tensor was found and released
    """
    _init_registry_lock()
    
    with _registry_lock:
        if ref_id in _tensor_registry:
            size_mb = _tensor_registry[ref_id].numel() * _tensor_registry[ref_id].element_size() / 1024**2
            del _tensor_registry[ref_id]
            logger.debug(f"Released tensor {ref_id} from registry ({size_mb:.1f}MB)")
            return True
        return False


class RemoteRefStub:
    """
    Lightweight stub for lazy materialization (Output Skeletonization).
    
    Used by HybridExecutor to return full output structure with RemoteRefStubs
    instead of concrete tensors. Preserves API transparency.
    
    Example:
        # Server-side
        output = model(inputs)  # Dict[str, Tensor]
        
        # Skeletonize
        stub_output = {
            'logits': RemoteRefStub('ref_1', (1, 768, 50257), torch.float32),
            'hidden': RemoteRefStub('ref_2', (1, 768, 1024), torch.float32),
        }
        
        # Return stub_output to client
        
        # Client-side
        logits = stub_output['logits'].to('cuda:0')  # Triggers fetch
    """
    
    def __init__(self, 
                 ref_id: str,
                 shape: Tuple[int, ...],
                 dtype: torch.dtype):
        """
        Initialize RemoteRefStub.
        
        Args:
            ref_id: Unique reference ID on server
            shape: Tensor shape
            dtype: Data type
        """
        self.ref_id = ref_id
        self.shape = shape
        self.dtype = dtype
    
    def __del__(self):
        """Best-effort cleanup notification on client."""
        try:
            from ..backend.runtime.initialization import get_runtime_state
            state = get_runtime_state()
            
            if state and state.coordinator:
                # Notify server that this ref is being released
                state.coordinator.notify_release(self.ref_id)
                logger.debug(f"Cleanup notification sent for {self.ref_id}")
            
            # Also release from local registry
            release_tensor(self.ref_id)
        except Exception as e:
            logger.debug(f"Cleanup failed: {e}")
    
    def to(self, device: str) -> torch.Tensor:
        """
        Materialize tensor on specified device.
        
        Triggers fetch from server heap (or local registry if available).
        
        Args:
            device: Target device (e.g., 'cuda:0', 'cpu')
        
        Returns:
            Materialized tensor
        """
        logger.debug(f"RemoteRefStub.to({device}): Attempting to materialize {self.ref_id}")
        
        # First try local registry
        tensor = retrieve_tensor(self.ref_id)
        
        if tensor is not None:
            logger.info(f"✅ Found {self.ref_id} in local registry, moving to {device}")
            return tensor.to(device=device)
        
        # Fall back to coordinator fetch
        logger.debug(f"Not found in local registry, attempting coordinator fetch for {self.ref_id}")
        try:
            from ..backend.runtime.initialization import get_runtime_state
            state = get_runtime_state()
            
            if not state or not state.coordinator:
                logger.warning(f"Coordinator not available, checking registry again...")
                # Try once more in case it was just added
                tensor = retrieve_tensor(self.ref_id)
                if tensor is not None:
                    logger.info(f"✅ Found {self.ref_id} on retry, moving to {device}")
                    return tensor.to(device=device)
                raise RuntimeError("Coordinator not available for fetching")
            
            # Fetch tensor from server
            logger.debug(f"Fetching {self.ref_id} from coordinator...")
            tensor = state.coordinator.fetch_tensor(self.ref_id)
            logger.info(f"✅ Fetched {self.ref_id} from coordinator, moving to {device}")
        except Exception as e:
            logger.error(f"❌ Failed to materialize RemoteRefStub {self.ref_id}: {e}")
            raise
        
        # Move to target device
        return tensor.to(device=device)
    
    def cpu(self) -> torch.Tensor:
        """Materialize on CPU."""
        return self.to('cpu')
    
    def cuda(self, device: Optional[int] = None) -> torch.Tensor:
        """Materialize on CUDA."""
        if device is None:
            return self.to('cuda:0')
        else:
            return self.to(f'cuda:{device}')
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RemoteRefStub(ref_id={self.ref_id}, shape={self.shape}, "
            f"dtype={self.dtype})"
        )
    
    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()

