"""
Mixed Placement Strategy for Hybrid Execution

Handles operations with mixed local/remote tensor inputs by determining
the most efficient execution location based on data transfer costs.

Key insight: When an operation has inputs on different devices, we should
execute where most data resides to minimize network transfers.
"""

from enum import Enum
from typing import List, Tuple, Optional
import math


class DataLocation(Enum):
    """Where a tensor currently resides."""
    LOCAL = "local"          # On CPU
    REMOTE = "remote"        # On remote GPU
    UNKNOWN = "unknown"      # Can't determine


class PlacementStrategy:
    """
    Determines optimal execution location for operations with mixed inputs.
    """
    
    # Threshold: switch to remote if 60% of data is there
    REMOTE_EXECUTION_THRESHOLD = 0.6
    
    @classmethod
    def get_tensor_location(cls, tensor) -> DataLocation:
        """Determine where a tensor resides."""
        from .lazy_tensor import LazyTensor
        
        # Check if it's a LazyTensor (remote)
        if isinstance(tensor, LazyTensor):
            return DataLocation.REMOTE
        
        # Check if it has device information
        if hasattr(tensor, 'device'):
            device_str = str(tensor.device)
            if 'cpu' in device_str.lower():
                return DataLocation.LOCAL
            elif 'cuda' in device_str.lower() or 'remote' in device_str.lower():
                return DataLocation.REMOTE
        
        return DataLocation.UNKNOWN
    
    @classmethod
    def estimate_tensor_size(cls, tensor) -> int:
        """Estimate tensor size in bytes."""
        try:
            if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                return tensor.numel() * tensor.element_size()
            return 0
        except Exception:
            return 0
    
    @classmethod
    def should_execute_remotely(cls, inputs: List) -> bool:
        """
        Determine if operation should execute remotely given mixed input locations.
        
        Strategy:
        1. Sum up data sizes at each location
        2. Execute where most data resides (>60% threshold)
        3. Conservative: if uncertain, execute locally
        
        Args:
            inputs: List of tensors (potentially mixed locations)
        
        Returns:
            True if execution should be remote
        """
        if not inputs:
            return False
        
        # Track data at each location
        local_size = 0
        remote_size = 0
        
        for inp in inputs:
            try:
                location = cls.get_tensor_location(inp)
                size = cls.estimate_tensor_size(inp)
                
                if location == DataLocation.LOCAL:
                    local_size += size
                elif location == DataLocation.REMOTE:
                    remote_size += size
                # Unknown: ignore for now
            
            except Exception:
                # If we can't determine, skip
                pass
        
        total_size = local_size + remote_size
        if total_size == 0:
            # No data size info available, execute locally (conservative)
            return False
        
        # Calculate remote data percentage
        remote_percentage = remote_size / total_size
        
        # Execute remotely if most data is there
        return remote_percentage >= cls.REMOTE_EXECUTION_THRESHOLD
    
    @classmethod
    def plan_data_transfers(cls, inputs: List) -> Tuple[bool, List]:
        """
        Plan which inputs need to be transferred before execution.
        
        Returns:
            (should_execute_remotely: bool, inputs_to_transfer: List of indices)
        """
        should_remote = cls.should_execute_remotely(inputs)
        inputs_to_transfer = []
        
        for i, inp in enumerate(inputs):
            location = cls.get_tensor_location(inp)
            
            if should_remote and location == DataLocation.LOCAL:
                # Need to transfer local tensor to remote
                inputs_to_transfer.append(i)
            elif not should_remote and location == DataLocation.REMOTE:
                # Need to transfer remote tensor to local
                inputs_to_transfer.append(i)
        
        return should_remote, inputs_to_transfer

