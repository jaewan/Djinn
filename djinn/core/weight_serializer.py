"""
Direct binary weight serialization (optimized path).

This module provides fast serialization of weights to the direct binary protocol,
avoiding JSON overhead and intermediate dict structures.
"""
import struct
import numpy as np
import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def serialize_weights_binary(weights: Dict[str, torch.Tensor]) -> bytes:
    """
    Serialize model weights to direct binary format.
    
    Protocol:
    [4 bytes: num_weights]
    [for each weight:
        [4 bytes: name_len] [name_bytes]
        [4 bytes: shape_len] [shape: 4*shape_len bytes]
        [4 bytes: dtype_len] [dtype_bytes]
        [8 bytes: data_len] [data_bytes]
    ]
    
    Args:
        weights: Dictionary mapping weight names to tensors
        
    Returns:
        Binary data containing serialized weights
    """
    result = bytearray()
    
    # Write number of weights
    result.extend(struct.pack('>I', len(weights)))
    
    total_bytes = 0
    for name, tensor in weights.items():
        # Ensure tensor is on CPU and contiguous
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Serialize name
        name_bytes = name.encode('utf-8')
        result.extend(struct.pack('>I', len(name_bytes)))
        result.extend(name_bytes)
        
        # Serialize shape
        shape = tuple(tensor.shape)
        result.extend(struct.pack('>I', len(shape)))
        result.extend(struct.pack(f'>{len(shape)}I', *shape))
        
        # Serialize dtype
        dtype_str = str(tensor.dtype)
        dtype_bytes = dtype_str.encode('utf-8')
        result.extend(struct.pack('>I', len(dtype_bytes)))
        result.extend(dtype_bytes)
        
        # Serialize data
        tensor_bytes = tensor.numpy().tobytes()
        result.extend(struct.pack('>Q', len(tensor_bytes)))
        result.extend(tensor_bytes)
        total_bytes += len(tensor_bytes)
    
    logger.debug(f"Serialized {len(weights)} weights, total size: {total_bytes / (1024*1024):.2f} MB")
    return bytes(result)

