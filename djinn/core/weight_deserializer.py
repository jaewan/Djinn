"""
Direct binary weight deserialization (optimized path).

This module provides fast deserialization of weights from the direct binary protocol,
avoiding JSON overhead and intermediate dict structures.
"""
import struct
import numpy as np
import torch
from typing import Dict


def deserialize_weights_binary(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Deserialize direct binary format weights.
    
    Protocol:
    [4 bytes: num_weights]
    [for each weight:
        [4 bytes: name_len] [name_bytes]
        [4 bytes: shape_len] [shape: 4*shape_len bytes]
        [4 bytes: dtype_len] [dtype_bytes]
        [8 bytes: data_len] [data_bytes]
    ]
    
    Args:
        data: Binary data containing serialized weights
        
    Returns:
        Dictionary mapping weight names to tensors
    """
    offset = 0
    
    if len(data) < 4:
        raise ValueError(f"Invalid binary weights data: too short ({len(data)} bytes)")
    
    num_weights = struct.unpack_from('>I', data, offset)[0]
    offset += 4
    
    weights = {}
    for _ in range(num_weights):
        # Unpack: name
        if len(data) < offset + 4:
            raise ValueError(f"Invalid binary weights data: missing name length at offset {offset}")
        name_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        
        if len(data) < offset + name_len:
            raise ValueError(f"Invalid binary weights data: missing name at offset {offset}")
        name = data[offset:offset+name_len].decode('utf-8')
        offset += name_len
        
        # Unpack: shape
        if len(data) < offset + 4:
            raise ValueError(f"Invalid binary weights data: missing shape length at offset {offset}")
        shape_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        
        if len(data) < offset + 4 * shape_len:
            raise ValueError(f"Invalid binary weights data: missing shape at offset {offset}")
        shape = struct.unpack_from(f'>{shape_len}I', data, offset)
        offset += 4 * shape_len
        
        # Unpack: dtype
        if len(data) < offset + 4:
            raise ValueError(f"Invalid binary weights data: missing dtype length at offset {offset}")
        dtype_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        
        if len(data) < offset + dtype_len:
            raise ValueError(f"Invalid binary weights data: missing dtype at offset {offset}")
        dtype_str = data[offset:offset+dtype_len].decode('utf-8')
        offset += dtype_len
        
        # Unpack: data
        if len(data) < offset + 8:
            raise ValueError(f"Invalid binary weights data: missing data length at offset {offset}")
        data_len = struct.unpack_from('>Q', data, offset)[0]
        offset += 8
        
        if len(data) < offset + data_len:
            raise ValueError(f"Invalid binary weights data: missing data at offset {offset} (expected {data_len} bytes)")
        weight_data = data[offset:offset+data_len]
        offset += data_len
        
        # Reconstruct tensor
        # Map PyTorch dtype strings to numpy dtypes
        dtype_map = {
            'torch.float32': np.float32,
            'torch.float64': np.float64,
            'torch.float16': np.float16,
            'torch.int32': np.int32,
            'torch.int64': np.int64,
            'torch.int16': np.int16,
            'torch.int8': np.int8,
            'torch.uint8': np.uint8,
        }
        
        # Handle both 'torch.float32' and 'float32' formats
        if dtype_str.startswith('torch.'):
            numpy_dtype = dtype_map.get(dtype_str, np.float32)
        else:
            # Try direct mapping
            numpy_dtype = getattr(np, dtype_str, np.float32)
        
        np_array = np.frombuffer(weight_data, dtype=numpy_dtype)
        np_array = np_array.reshape(shape)
        # Make writable copy to avoid PyTorch warning
        np_array = np_array.copy()
        tensor = torch.from_numpy(np_array)
        weights[name] = tensor
    
    return weights

