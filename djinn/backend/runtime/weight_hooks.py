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
        """
        Install forward hooks on all layers.
        
        Uses block-granularity if blocks are available, otherwise falls back
        to parameter-granularity (legacy).
        """
        if self.hooks_installed:
            logger.warning("Hooks already installed")
            return
        
        # Check if we have blocks (new block-granularity mode)
        registration = self.ring_buffer.registrations.get(self.model_id)
        if registration and registration.blocks:
            self._install_block_hooks()
        else:
            self._install_param_hooks()
    
    def _install_block_hooks(self) -> None:
        """
        Install block-level hooks (NEW: block-granularity streaming).
        
        One hook per transformer block, installed on the first module of each block.
        """
        registration = self.ring_buffer.registrations[self.model_id]
        blocks_hooked = 0
        
        for block in registration.blocks:
            # Find the first module in this block
            first_module_name = None
            for param_name in block.param_names:
                # Extract module name from param name (remove .weight, .bias, etc.)
                parts = param_name.split('.')
                if len(parts) > 1:
                    module_name = '.'.join(parts[:-1])
                    first_module_name = module_name
                    break
            
            if first_module_name is None:
                logger.warning(f"Could not find module for block {block.block_name}")
                continue
            
            # Get the module
            module = self._get_module_by_name(self.model, first_module_name)
            if module is None:
                logger.warning(f"Could not find module: {first_module_name}")
                continue
            
            # Create block-level hook
            hook_fn = self._create_block_pre_hook(block.block_idx)
            
            # Install pre-hook
            handle = module.register_forward_pre_hook(hook_fn)
            self.hook_handles[block.block_name] = handle
            
            blocks_hooked += 1
        
        self.hooks_installed = True
        logger.info(f"✅ Installed block-level hooks on {blocks_hooked}/{len(registration.blocks)} blocks")
        
        # Pre-prefetch first few streamed blocks to avoid cold start
        if self.streamer is not None:
            streamed_blocks = [b for b in registration.blocks if not b.is_resident]
            if streamed_blocks:
                # Prefetch first 2 streamed blocks
                for i, block in enumerate(streamed_blocks[:2]):
                    try:
                        self.streamer.prefetch_block(self.model_id, block.block_idx)
                        logger.debug(f"  Pre-prefetched block {block.block_idx} ({block.block_name})")
                    except Exception as e:
                        logger.warning(f"Failed to pre-prefetch block {block.block_idx}: {e}")
                
                # Wait for first prefetch to complete
                if streamed_blocks and len(registration.streaming_slots) > 0:
                    first_block_idx = streamed_blocks[0].block_idx
                    slot_id = registration.block_to_slot.get(first_block_idx)
                    if slot_id is not None:
                        slot = registration.streaming_slots[slot_id]
                        if slot.ready_event is not None:
                            slot.ready_event.synchronize()
                            logger.debug(f"  First streamed block ready")
    
    def _install_param_hooks(self) -> None:
        """
        Install parameter-level hooks (LEGACY: param-granularity streaming).
        
        One hook per module with weights.
        """
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
        logger.info(f"✅ Installed param-level hooks on {layers_hooked}/{len(self.layer_names)} layers")
    
    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for layer_name, handle in self.hook_handles.items():
            handle.remove()
        
        self.hook_handles.clear()
        self.hooks_installed = False
        logger.info("✅ Removed all hooks")
    
    def _create_block_pre_hook(self, block_idx: int) -> Callable:
        """
        Create a forward pre-hook for a transformer block (NEW: block-granularity).
        
        This hook:
        1. Waits for the block's slot to be ready (if streamed)
        2. Binds ALL params in the block to their slot views
        3. Triggers prefetch of block_idx + 2 (2-ahead pipelining)
        """
        def pre_hook(module, inputs):
            registration = self.ring_buffer.registrations.get(self.model_id)
            if not registration:
                return
            
            if block_idx >= len(registration.blocks):
                return
            
            block = registration.blocks[block_idx]
            
            # If block is resident, all params already point to ring buffer
            if block.is_resident:
                # Trigger prefetch of next streamed block
                if self.streamer is not None:
                    # Find next streamed block
                    for next_idx in range(block_idx + 1, len(registration.blocks)):
                        next_block = registration.blocks[next_idx]
                        if not next_block.is_resident:
                            self.streamer.prefetch_block(self.model_id, next_idx)
                            break
                return
            
            # Block is streamed - get its slot
            slot_id = registration.block_to_slot.get(block_idx)
            if slot_id is None:
                logger.error(
                    f"Streamed block {block_idx} ({block.block_name}) not prefetched! "
                    f"This should not happen."
                )
                # Emergency fallback: trigger synchronous prefetch
                if self.streamer is not None:
                    self.streamer.prefetch_block(self.model_id, block_idx)
                    slot_id = registration.block_to_slot.get(block_idx)
                
                if slot_id is None:
                    logger.error(f"Failed to prefetch block {block_idx}")
                    return
            
            slot = registration.streaming_slots[slot_id]
            
            # Wait for prefetch to complete (GPU-side sync)
            if slot.ready_event is not None:
                torch.cuda.current_stream().wait_event(slot.ready_event)
            
            # Bind ALL params in this block to their slot views
            for param_name, (start_offset, end_offset) in block.param_offsets.items():
                # Get the parameter from model
                try:
                    param = self._get_param_by_name(self.model, param_name)
                    if param is None:
                        continue
                    
                    allocation = registration.layer_allocations.get(param_name)
                    if allocation is None:
                        continue
                    
                    # Create view into slot at the param's offset
                    param_view = self.ring_buffer.buffer[
                        slot.offset + start_offset : slot.offset + end_offset
                    ].view(allocation.dtype).view(allocation.shape)
                    
                    # Redirect param data
                    param.data = param_view
                    
                except Exception as e:
                    logger.error(f"Failed to bind param {param_name}: {e}")
            
            # Trigger prefetch of block_idx + 2 (2-ahead pipelining)
            if self.streamer is not None and block_idx + 2 < len(registration.blocks):
                next_block = registration.blocks[block_idx + 2]
                if not next_block.is_resident:
                    try:
                        self.streamer.prefetch_block(self.model_id, block_idx + 2)
                    except Exception as e:
                        logger.warning(f"Prefetch failed for block {block_idx + 2}: {e}")
        
        return pre_hook
    
    def _create_pre_hook(self, layer_name: str, layer_idx: int) -> Callable:
        """
        Create a forward pre-hook for a layer.
        
        This hook redirects module.weight to point to the ring buffer view
        before the forward pass executes. Handles both:
        - Resident layers: bind directly to fixed view
        - Streamed layers: wait for prefetch and bind to slot view
        
        For async pipelining, the hook also triggers prefetch of next layer.
        """
        def pre_hook(module, inputs):
            # Get model registration from ring buffer
            if self.model_id not in self.ring_buffer.registrations:
                logger.warning(f"Model {self.model_id} not registered in ring buffer")
                return
            
            registration = self.ring_buffer.registrations[self.model_id]
            
            # Find the weight parameter name for this layer
            # For most layers, it's just 'weight', but check all parameters
            weight_param_name = None
            for param_name in ['weight', 'weight_g', 'weight_v']:
                if hasattr(module, param_name) and getattr(module, param_name) is not None:
                    weight_param_name = param_name
                    break
            
            if weight_param_name is None:
                # No weight parameter to redirect
                return
            
            # Get the full parameter name (layer_name.weight)
            full_param_name = f"{layer_name}.{weight_param_name}" if layer_name else weight_param_name
            
            # Check if this weight is in the ring buffer
            if full_param_name in registration.layer_allocations:
                allocation = registration.layer_allocations[full_param_name]
                
                # Determine if this is a resident or streamed layer
                if allocation.is_resident:
                    # Resident layer: bind directly to fixed view
                    weight_view = self.ring_buffer.buffer[
                        allocation.offset : allocation.end_offset()
                    ].view(allocation.dtype).view(allocation.shape)
                else:
                    # Streamed layer: wait for prefetch and bind to slot
                    slot_id = registration.layer_to_slot.get(full_param_name)
                    if slot_id is None:
                        logger.error(
                            f"Streamed layer {full_param_name} not prefetched! "
                            f"This should not happen - prefetch should be triggered by previous layer."
                        )
                        # Emergency fallback: trigger synchronous prefetch
                        if self.streamer is not None:
                            self.streamer.prefetch_layer(
                                model_id=self.model_id,
                                layer_idx=layer_idx,
                                layer_name=full_param_name
                            )
                            slot_id = registration.layer_to_slot.get(full_param_name)
                        
                        if slot_id is None:
                            logger.error(f"Failed to prefetch {full_param_name}, skipping weight redirect")
                            return
                    
                    slot = registration.streaming_slots[slot_id]
                    
                    # Wait for prefetch to complete (GPU-side, no CPU blocking)
                    if slot.ready_event is not None:
                        torch.cuda.current_stream().wait_event(slot.ready_event)
                    
                    # Bind to slot view
                    weight_view = self.ring_buffer.buffer[
                        slot.offset : slot.offset + allocation.size_bytes
                    ].view(allocation.dtype).view(allocation.shape)
                
                # Redirect module's weight to point to ring buffer view
                # This is O(1) - just updating a pointer, not copying data
                original_weight = getattr(module, weight_param_name)
                setattr(module, weight_param_name, torch.nn.Parameter(
                    weight_view,
                    requires_grad=original_weight.requires_grad
                ))
                
                # If streamer exists, trigger prefetch of next module's weights
                # We use module names here because hooks operate on modules
                if self.streamer is not None and layer_idx + 1 < len(self.layer_names):
                    next_module_name = self.layer_names[layer_idx + 1]
                    # Async prefetch all params of next module (non-blocking)
                    # The streamer will use module_to_params mapping to find all params
                    try:
                        self.streamer.prefetch_layer(
                            model_id=self.model_id,
                            layer_idx=layer_idx + 1,
                            layer_name=next_module_name  # This is a module name
                        )
                    except Exception as e:
                        logger.warning(f"Prefetch failed for {next_module_name}: {e}")
        
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
    
    def _get_param_by_name(
        self,
        model: nn.Module,
        param_name: str
    ) -> Optional[torch.nn.Parameter]:
        """Get a parameter by its dotted name."""
        parts = param_name.split('.')
        obj = model
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        
        if isinstance(obj, torch.nn.Parameter):
            return obj
        
        return None


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

