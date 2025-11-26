"""
Device mapping utilities for remote accelerator support.

Maps remote device strings (e.g., 'remote_accelerator:0') to local devices
(e.g., 'cuda:0' or 'cpu').
"""

import torch
from typing import Any, Optional

__all__ = ['DeviceMapper']


class DeviceMapper:
    """Maps remote device strings to local devices."""
    
    # Tensor creation operations that accept device kwarg
    CREATION_OPS = {
        "randn", "rand", "randint",
        "zeros", "ones", "empty", "full", "empty_strided",
        "arange", "linspace", "logspace",
    }
    
    @staticmethod
    def map_remote_to_local(device: Any) -> str:
        """
        Map remote_accelerator:N to cuda:N or cpu.
        
        Args:
            device: Device string, torch.device, or None
            
        Returns:
            Local device string (e.g., 'cuda:0' or 'cpu')
        """
        if device is None:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if isinstance(device, str):
            if 'remote_accelerator' in device:
                device_idx = device.split(':')[-1]
                try:
                    device_idx_int = int(device_idx)
                    if device_idx_int < torch.cuda.device_count():
                        return f'cuda:{device_idx}'
                    else:
                        return "cpu"
                except (ValueError, IndexError):
                    return "cpu"
            return device
        
        if hasattr(device, 'type') and device.type in ('remote_accelerator', 'privateuseone'):
            try:
                if device.index < torch.cuda.device_count():
                    return f'cuda:{device.index}'
                else:
                    return "cpu"
            except (AttributeError, IndexError):
                return "cpu"
        
        return "cpu"
    
    @staticmethod
    def should_include_device_kwarg(op_name: str) -> bool:
        """
        Check if operation accepts device kwarg.
        
        Only tensor creation operations accept device kwarg.
        Most other operations don't accept it.
        
        Args:
            op_name: Operation name (e.g., 'aten::randn' or 'randn')
            
        Returns:
            True if operation accepts device kwarg
        """
        base_name = op_name.replace("aten::", "")
        return base_name in DeviceMapper.CREATION_OPS
    
    @staticmethod
    def clean_device_kwarg(op_name: str, kwargs: dict) -> dict:
        """
        Clean device kwarg for operation dispatch.
        
        - For creation ops: Map remote device to local device
        - For other ops: Remove device kwarg (not accepted)
        
        Args:
            op_name: Operation name
            kwargs: Operation kwargs (will be modified in place)
            
        Returns:
            Cleaned kwargs dict
        """
        kwargs = kwargs.copy() if kwargs else {}
        
        if DeviceMapper.should_include_device_kwarg(op_name):
            # Creation ops: map device
            device = kwargs.get("device")
            kwargs["device"] = DeviceMapper.map_remote_to_local(device)
        else:
            # Non-creation ops: remove device kwarg
            kwargs.pop("device", None)
        
        return kwargs

