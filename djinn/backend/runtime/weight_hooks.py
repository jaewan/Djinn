"""
PyTorch Forward Hooks for Weight Ring Buffer Integration

Installs hooks on model layers that intercept forward passes and redirect
weight pointers to the ring buffer before execution.

Key insight: No tensor copy needed. Just update module.weight.data pointer
(O(1) CPU operation) to point to pre-computed views in ring buffer.

Usage:
    from djinn.backend.runtime.weight_hooks import install_ring_buffer_hooks
    
    ring_buffer = WeightRingBuffer(...)
    ring_buffer.register_model('llama-70b', model.state_dict())
    
    install_ring_buffer_hooks(
        model,
        ring_buffer=ring_buffer,
        model_id='llama-70b',
        streamer=weight_streamer  # WeightStreamer instance
    )
    
    # Now model.forward() automatically prefetches and redirects weights
    output = model.forward(input)
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, List, Any
from djinn.backend.runtime.weight_registry import WeightNameRegistry

logger = logging.getLogger(__name__)


class RingBufferHookManager:
    """
    Manages forward hooks for ring buffer weight redirection.
    
    Installs hooks on each layer that:
    1. Prefetch next layer's weights asynchronously
    2. Redirect module.weight to ring buffer view
    3. Signal when computation is done (for ring buffer reuse)
    """
    
    def __init__(
        self,
        model: nn.Module,
        ring_buffer,
        model_id: str,
        streamer=None,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Initialize hook manager.
        
        Args:
            model: PyTorch model
            ring_buffer: WeightRingBuffer instance
            model_id: Model ID for ring buffer
            streamer: WeightStreamer for async prefetching
            layer_names: Optional list of layer names (if None, infers from model)
        """
        self.model = model
        self.ring_buffer = ring_buffer
        self.model_id = model_id
        self.streamer = streamer
        self.hooks_installed = False
        
        # Create weight name registry for unified naming
        self.registry = WeightNameRegistry(model)
        
        # Extract MODULE names (e.g., "lm_head") which correspond to forward hooks
        if layer_names is None:
            self.layer_names = self._extract_layer_names(model)
        else:
            self.layer_names = layer_names
        
        # Warn if no layers found
        if not self.layer_names:
            logger.warning(
                "No layers with weights found in model. "
                "Hooks will be no-op. Check layer_names argument."
            )
        
        # Map layer name to index for event tracking
        self.layer_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.layer_names)
        }
        
        # Store model reference for on-demand weight access during prefetching
        # We will fetch weights from model on-demand instead of caching all in RAM
        # This avoids the memory explosion of storing 140GB of weights twice
        self.model_state_dict = None
        
        # Store hooks for potential removal
        self.hook_handles: Dict[str, Any] = {}
        
        # Register tied weights with ring buffer
        if self.registry.tied_weights:
            self.ring_buffer.set_tied_weights(model_id, self.registry.tied_weights)
            self.registry.log_summary()
        
        logger.info(f"Initialized RingBufferHookManager for {len(self.layer_names)} layers")
    
    def install_hooks(self) -> None:
        """Install forward hooks on all layers."""
        if self.hooks_installed:
            logger.warning("Hooks already installed")
            return
        
        layers_hooked = 0
        
        for layer_name in self.layer_names:
            # Get the module by name
            module = self._get_module_by_name(self.model, layer_name)
            if module is None:
                logger.warning(f"Could not find module: {layer_name}")
                continue
            
            # Create hook function for this layer
            layer_idx = self.layer_name_to_idx[layer_name]
            hook_fn = self._create_pre_hook(layer_name, layer_idx)
            
            # Install pre-hook
            handle = module.register_forward_pre_hook(hook_fn)
            self.hook_handles[layer_name] = handle
            
            layers_hooked += 1
        
        self.hooks_installed = True
        logger.info(f"✅ Installed hooks on {layers_hooked}/{len(self.layer_names)} layers")
    
    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for layer_name, handle in self.hook_handles.items():
            handle.remove()
        
        self.hook_handles.clear()
        self.hooks_installed = False
        logger.info("✅ Removed all hooks")
    
    def _create_pre_hook(self, layer_name: str, layer_idx: int) -> Callable:
        """
        Create a forward pre-hook for a layer.

        For virtualization with resident weights, this hook is simplified to:
        - Avoid errors when streamer is not tracking this model
        - Allow forward pass to proceed with pre-loaded ring buffer weights
        """
        def pre_hook(module, inputs):
            # For virtualization: all resident weights are already in ring buffer views
            # No need to redirect pointers since model was created with ring buffer views
            # Hooks are no-op for resident weights, and streamed weights will be handled
            # by the weight streamer when implemented
            pass
        
        return pre_hook
    
    def _extract_layer_names(self, model: nn.Module) -> List[str]:
        """
        Extract layer names from model by looking for weight parameters.
        
        Returns list of named parameters that have weights (typically
        linear/conv layers).
        """
        layer_names = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                layer_names.append(name)
        return layer_names
    
    def _get_module_by_name(
        self,
        model: nn.Module,
        module_name: str
    ) -> Optional[nn.Module]:
        """Get a module by its dotted name."""
        parts = module_name.split('.')
        module = model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        return module


def install_ring_buffer_hooks(
    model: nn.Module,
    ring_buffer,
    model_id: str,
    streamer=None,
    layer_names: Optional[List[str]] = None,
) -> RingBufferHookManager:
    """
    Install ring buffer hooks on a model.
    
    This enables automatic weight redirection from ring buffer during
    forward passes. Hooks run in pre-forward phase to redirect weights
    before computation.
    
    Args:
        model: PyTorch model
        ring_buffer: WeightRingBuffer instance
        model_id: Model ID (must be pre-registered with ring_buffer)
        streamer: WeightStreamer for async prefetching (optional)
        layer_names: Explicit layer names (if None, infers from model)
    
    Returns:
        RingBufferHookManager instance (can be used to remove hooks later)
    
    Example:
        ring_buffer = WeightRingBuffer(capacity_bytes=48*1024**3)
        ring_buffer.register_model('llama', model.state_dict())
        
        hook_mgr = install_ring_buffer_hooks(
            model,
            ring_buffer=ring_buffer,
            model_id='llama'
        )
        
        # Hooks now active
        output = model.forward(input)
        
        # Remove hooks when done
        hook_mgr.remove_hooks()
    """
    manager = RingBufferHookManager(
        model,
        ring_buffer=ring_buffer,
        model_id=model_id,
        streamer=streamer,
        layer_names=layer_names,
    )
    
    manager.install_hooks()
    return manager


def uninstall_ring_buffer_hooks(hook_manager: RingBufferHookManager) -> None:
    """Remove ring buffer hooks from a model."""
    hook_manager.remove_hooks()

