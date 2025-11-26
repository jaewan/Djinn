"""
PyTorch Device Compatibility Layer for Djinn.

Provides PyTorch-compatible device semantics by patching nn.Module.to()
to automatically convert model weights to LazyTensors when 'remote_accelerator'
device is requested. This bridges PyTorch's device model with Djinn's
LazyTensor-based computation graph capture.
"""

import os
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


_ENABLE_PRIVATEUSE1 = os.environ.get("DJINN_ENABLE_PRIVATEUSE1_DEVICE", "").lower() in {"1", "true", "yes"}


def _register_device_python_fallback():
    """
    Python fallback for device registration when C++ backend is unavailable.
    
    Uses PyTorch's PrivateUse1 backend system to register remote_accelerator
    as a valid device type. This allows PyTorch functions to accept
    device='remote_accelerator:0' without validation errors.
    """
    if _ENABLE_PRIVATEUSE1 and hasattr(torch.utils, 'rename_privateuse1_backend'):
        try:
            torch.utils.rename_privateuse1_backend("remote_accelerator")
            torch.utils.generate_methods_for_privateuse1_backend(
                for_tensor=True, for_module=True, for_packed_sequence=True, for_storage=False
            )
            logger.info("✅ Registered remote_accelerator device via PrivateUse1 backend")
            return True
        except Exception as e:
            logger.warning(f"⚠️  PrivateUse1 backend registration failed: {e}")
            logger.warning(
                "   Falling back to internal device shim (pin_memory will remain available). "
                "Set DJINN_ENABLE_PRIVATEUSE1_DEVICE=1 only if you provide a custom backend implementation."
            )
            return False

    logger.info(
        "ℹ️  Skipping PrivateUse1 backend rename (ensures pinned memory works out of the box). "
        "Set DJINN_ENABLE_PRIVATEUSE1_DEVICE=1 to opt into official backend registration."
    )
    return False


class RemoteAcceleratorSupport:
    """Enables remote_accelerator device for PyTorch models."""

    _initialized = False
    _original_module_to = None

    @classmethod
    def initialize(cls):
        """Initialize remote_accelerator device support."""
        if cls._initialized:
            return

        # Try C++ backend first, fall back to Python
        try:
            from .. import _C
            _C.register_remote_accelerator_device()
            logger.info("✅ C++ device backend registered")
        except (ImportError, AttributeError):
            logger.info("ℹ️  C++ backend unavailable, using Python fallback")
            _register_device_python_fallback()

        # Store original method
        cls._original_module_to = nn.Module.to

        # Patch Module.to()
        def patched_to(self, *args, **kwargs):
            print(f"PATCHED_TO called on {self.__class__.__name__} with args={args} kwargs={kwargs}")
            # Check if this is a remote_accelerator request before parsing
            device_arg = None
            if args and isinstance(args[0], (str, torch.device)):
                device_arg = args[0]
            elif 'device' in kwargs:
                device_arg = kwargs['device']

            if device_arg is not None:
                device_str = str(device_arg)

                # Check if remote_accelerator is requested
                if 'remote_accelerator' in device_str:
                    logger.info(f"Converting model {self.__class__.__name__} to remote_accelerator")

                    # Import here to avoid circular dependency
                    from djinn.frontend.core.lazy_tensor import LazyTensor

                    # Convert all parameters and buffers to LazyTensors
                    lazy_params = {}
                    lazy_buffers = {}

                    # ✅ DESIGN FIX: Move tensors to CPU during LazyTensor conversion
                    # For GPU disaggregation, models should stay on CPU (not GPU)
                    # This eliminates unnecessary GPU→CPU copy during registration
                    has_gpu_params = any(p.device.type == 'cuda' for p in self.parameters())
                    if has_gpu_params:
                        logger.info(
                            "Model is on GPU - moving to CPU for GPU disaggregation. "
                            "For best performance, load models on CPU and move directly to 'remote_accelerator:0'"
                        )
                    
                    for name, param in self.named_parameters():
                        # ✅ DESIGN FIX: Move to CPU before converting to LazyTensor
                        # This ensures models stay on CPU (correct for GPU disaggregation)
                        # and eliminates GPU→CPU copy during registration
                        if param.device.type == 'cuda':
                            param_cpu = param.cpu()
                            logger.debug(f"Moving parameter {name} from GPU to CPU (required for GPU disaggregation)")
                        else:
                            param_cpu = param
                        
                        # ✅ ELEGANT FIX: Explicitly preserve dtype during conversion
                        # This ensures model parameters maintain their dtype (e.g., float16 for GPT2-XL)
                        lazy_params[name] = LazyTensor.tensor(
                            param_cpu.detach(),
                            dtype=param.dtype,  # Preserve dtype from original param
                            device='cpu'  # ✅ FIX: Explicitly CPU (correct for GPU disaggregation)
                        )

                    for name, buffer in self.named_buffers():
                        # ✅ DESIGN FIX: Move buffers to CPU too
                        if buffer.device.type == 'cuda':
                            buffer_cpu = buffer.cpu()
                            logger.debug(f"Moving buffer {name} from GPU to CPU (required for GPU disaggregation)")
                        else:
                            buffer_cpu = buffer
                        
                        # ✅ ELEGANT FIX: Explicitly preserve dtype for buffers too
                        lazy_buffers[name] = LazyTensor.tensor(
                            buffer_cpu.detach(),
                            dtype=buffer.dtype,  # Preserve dtype from original buffer
                            device='cpu'  # ✅ FIX: Explicitly CPU (correct for GPU disaggregation)
                        )

                    # Replace parameters and buffers
                    for name, lazy_param in lazy_params.items():
                        # ✅ FIX: Create new Parameter with LazyTensor data
                        # PyTorch's Parameter.set_data() doesn't accept LazyTensor directly
                        # So we need to create a new Parameter object
                        param = dict(self.named_parameters())[name]
                        new_param = nn.Parameter(lazy_param, requires_grad=param.requires_grad)
                        # Use setattr to replace the parameter in the module
                        parts = name.split('.')
                        if len(parts) == 1:
                            setattr(self, name, new_param)
                        else:
                            # Navigate to the parent module
                            parent = self
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            setattr(parent, parts[-1], new_param)

                    for name, lazy_buffer in lazy_buffers.items():
                        # ✅ FIX: Buffers can be set directly, but let's be safe
                        buffer = dict(self.named_buffers())[name]
                        parts = name.split('.')
                        if len(parts) == 1:
                            setattr(self, name, lazy_buffers[name])
                        else:
                            # Navigate to the parent module
                            parent = self
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            setattr(parent, parts[-1], lazy_buffers[name])

                    # Track conversion
                    param_count = len(lazy_params)
                    buffer_count = len(lazy_buffers)
                    logger.info(f"Converted {param_count} parameters and {buffer_count} buffers to LazyTensors")

                    # Return self (device conversion complete)
                    return self

            # For all other devices, use original implementation
            # Handle the case where PyTorch might reject our custom device string
            try:
                if 'cuda' in device_str:
                    logger.info(f"Delegating {self.__class__.__name__}.to({device_str}) to original implementation")
                return cls._original_module_to(self, *args, **kwargs)
            except RuntimeError as e:
                if 'remote_accelerator' in str(e):
                    # This is our custom device, handle it
                    logger.warning(f"PyTorch rejected remote_accelerator device: {e}")
                    # Return self unchanged - the device conversion is handled at tensor level
                    return self
                else:
                    # Re-raise for actual errors
                    raise

        # Apply the patch
        nn.Module.to = patched_to
        cls._initialized = True

        logger.info("✅ Remote accelerator device support initialized")


def setup():
    """Setup device compatibility layer."""
    RemoteAcceleratorSupport.initialize()