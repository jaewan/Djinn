"""
Breakpoint Hook Manager: Install forward hooks for layer-level breakpoint interception.

Purpose:
- Install PyTorch forward hooks on model layers to intercept execution
- Checkpoint activations when reaching breakpoint layer
- Enable pause/resume at precise layer boundaries
- Reuse pattern from weight_hooks.py for consistency

Architecture:
- Pre-hook: Called before layer forward pass (setup, prefetching)
- Post-hook: Called after layer forward pass (checkpoint, state update)
- Breakpoint hook: Triggers checkpoint and pause at specified layer
- Maintains hook handles for cleanup

Integration:
- BreakpointManager provides breakpoint configuration per session
- ActivationCheckpointer persists intermediate activations
- Execution engine respects pause state from BreakpointManager
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any, Callable, Tuple
import uuid

logger = logging.getLogger(__name__)


class BreakpointHookManager:
    """
    Manages forward hooks for layer-level breakpoint interception.
    
    Installs hooks on model layers to:
    1. Track which layer is currently executing
    2. Checkpoint intermediate activations at breakpoint layer
    3. Signal pause/resume to BreakpointManager
    """
    
    def __init__(
        self,
        model: nn.Module,
        breakpoint_layer_index: int,
        session_id: str,
        activation_checkpointer=None,
        breakpoint_manager=None,
    ):
        """
        Initialize hook manager for a specific breakpoint.
        
        Args:
            model: PyTorch model to install hooks on
            breakpoint_layer_index: Layer index where to pause
            session_id: Session identifier
            activation_checkpointer: ActivationCheckpointer instance (lazy-loaded if None)
            breakpoint_manager: BreakpointManager instance (lazy-loaded if None)
        """
        self.model = model
        self.breakpoint_layer_index = breakpoint_layer_index
        self.session_id = session_id
        self.activation_checkpointer = activation_checkpointer
        self.breakpoint_manager = breakpoint_manager
        
        # Extract layers from model
        self.layers = self._extract_layers(model)
        
        if not self.layers:
            logger.warning(
                f"No layers found in model. "
                f"Breakpoint hooks may not work. Check model structure."
            )
        
        # Validate breakpoint layer index
        if breakpoint_layer_index >= len(self.layers):
            raise ValueError(
                f"Breakpoint layer {breakpoint_layer_index} out of range. "
                f"Model has {len(self.layers)} layers."
            )
        
        # Store hook handles for cleanup
        self.hook_handles: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "pre_hooks_called": 0,
            "post_hooks_called": 0,
            "breakpoint_checkpoints": 0,
            "checkpoint_errors": 0,
        }
        
        logger.info(
            f"Initialized BreakpointHookManager: "
            f"session_id={session_id}, "
            f"breakpoint_layer={breakpoint_layer_index}, "
            f"total_layers={len(self.layers)}"
        )
    
    def install_hooks(self) -> None:
        """Install breakpoint hooks on all layers."""
        for layer_idx, layer in enumerate(self.layers):
            # Pre-hook: Before layer execution
            pre_handle = layer.register_forward_pre_hook(
                self._create_pre_hook(layer_idx)
            )
            self.hook_handles[f"pre_{layer_idx}"] = pre_handle
            
            # Post-hook: After layer execution
            post_handle = layer.register_forward_hook(
                self._create_post_hook(layer_idx)
            )
            self.hook_handles[f"post_{layer_idx}"] = post_handle
        
        logger.info(f"✅ Installed {len(self.layers) * 2} hooks ({len(self.layers)} layers)")
    
    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        logger.info("✅ Removed all breakpoint hooks")
    
    def _create_pre_hook(self, layer_idx: int) -> Callable:
        """Create pre-hook for a layer."""
        def pre_hook(module: nn.Module, input: Tuple) -> None:
            """Pre-hook: called before forward pass."""
            self.stats["pre_hooks_called"] += 1
            
            # Log layer execution (debug)
            if layer_idx == self.breakpoint_layer_index:
                logger.debug(
                    f"[Session {self.session_id}] Executing breakpoint layer {layer_idx}"
                )
        
        return pre_hook
    
    def _create_post_hook(self, layer_idx: int) -> Callable:
        """Create post-hook for a layer."""
        def post_hook(
            module: nn.Module,
            input: Tuple,
            output: torch.Tensor
        ) -> None:
            """Post-hook: called after forward pass. Handles checkpoint."""
            self.stats["post_hooks_called"] += 1
            
            # Check if this is the breakpoint layer
            if layer_idx == self.breakpoint_layer_index:
                logger.debug(
                    f"[Session {self.session_id}] Breakpoint layer {layer_idx} completed, "
                    f"checkpointing..."
                )
                
                try:
                    # Get managers (lazy load if needed)
                    checkpointer = self._get_checkpointer()
                    manager = self._get_breakpoint_manager()
                    
                    # Collect activations to checkpoint
                    activations = self._collect_activations(module, input, output)
                    
                    # Create checkpoint
                    checkpoint_id = f"bp_{self.session_id}_{layer_idx}_{uuid.uuid4().hex[:8]}"
                    
                    metadata, checkpoint_time = checkpointer.checkpoint(
                        session_id=self.session_id,
                        checkpoint_id=checkpoint_id,
                        layer_index=layer_idx,
                        activations=activations,
                        device=output.device if isinstance(output, torch.Tensor) else None,
                    )
                    
                    # Trigger pause in breakpoint manager
                    manager.trigger_pause(
                        session_id=self.session_id,
                        checkpoint_id=checkpoint_id
                    )
                    
                    self.stats["breakpoint_checkpoints"] += 1
                    
                    logger.info(
                        f"✅ [Session {self.session_id}] Breakpoint checkpoint created: "
                        f"checkpoint_id={checkpoint_id}, "
                        f"size={metadata.total_bytes / 1024**2:.1f}MB, "
                        f"time={checkpoint_time*1000:.1f}ms"
                    )
                
                except Exception as e:
                    self.stats["checkpoint_errors"] += 1
                    logger.error(f"❌ Checkpoint failed: {e}", exc_info=True)
                    raise
        
        return post_hook
    
    def _collect_activations(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Collect intermediate activations to checkpoint.
        
        ✅ OPTIMIZED: Avoids GPU blocking inside forward hook
        - Returns GPU tensors (detached only)
        - Actual H2D transfer happens in ActivationCheckpointer with async stream
        - CRITICAL: Explicitly saves attention_mask for transformer continuation
        
        Args:
            module: The layer module
            input: Input tuple to the layer
            output: Output from the layer
        
        Returns:
            Dict of activation name -> tensor (still on GPU, not synced)
        """
        activations = {}
        
        # Include layer output (keep on GPU, no .cpu() to avoid hook blocking)
        if isinstance(output, torch.Tensor):
            activations["output"] = output.detach()
        elif isinstance(output, (tuple, list)):
            for i, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    activations[f"output_{i}"] = tensor.detach()
        
        # Include layer inputs (for some architectures)
        if isinstance(input, tuple) and len(input) > 0:
            for i, tensor in enumerate(input):
                if isinstance(tensor, torch.Tensor):
                    activations[f"input_{i}"] = tensor.detach()
        
        # ✅ CRITICAL: Explicitly save attention_mask from inputs
        # For transformer layers (GPT-2, etc.), input[1] is attention_mask
        # This is needed for _continue_from_layer to pass correct context to subsequent layers
        if "input_1" in activations:
            activations["attention_mask"] = activations["input_1"]
            logger.debug(f"[{self.session_id}] Saved attention_mask from input_1")
        
        # Include module buffers (hidden states, attention scores, etc.)
        for name, buffer in module.named_buffers():
            if buffer is not None and isinstance(buffer, torch.Tensor):
                activations[f"buffer_{name}"] = buffer.detach()
        
        return activations
    
    def _extract_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Extract sequential layers from model.
        
        Handles different model architectures:
        - Transformer: transformer.h or model.layers
        - Sequential: sequential layers
        - Custom: all nn.Module children
        
        Args:
            model: Model to extract layers from
        
        Returns:
            List of layer modules
        """
        layers = []
        
        # Try common transformer architectures
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # Hugging Face GPT-2 style
            layers = list(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama style
            layers = list(model.model.layers)
        elif hasattr(model, 'layers'):
            # Direct layers attribute
            layers = list(model.layers)
        elif isinstance(model, nn.Sequential):
            # Sequential model
            layers = list(model)
        elif hasattr(model, 'blocks'):
            # Vision transformer style
            layers = list(model.blocks)
        else:
            # Fallback: collect all direct nn.Module children
            # (may not work well for nested architectures)
            for name, child in model.named_children():
                if isinstance(child, nn.Module):
                    layers.append(child)
        
        if not layers:
            logger.warning(
                "Could not extract layers from model. "
                "Tried: transformer.h, model.layers, layers, sequential, blocks"
            )
        
        return layers
    
    def _get_checkpointer(self):
        """Get activation checkpointer (lazy load if needed)."""
        if self.activation_checkpointer is None:
            from djinn.server.activation_checkpointer import get_activation_checkpointer
            self.activation_checkpointer = get_activation_checkpointer()
        return self.activation_checkpointer
    
    def _get_breakpoint_manager(self):
        """Get breakpoint manager (lazy load if needed)."""
        if self.breakpoint_manager is None:
            from djinn.server.breakpoint_manager import get_breakpoint_manager
            self.breakpoint_manager = get_breakpoint_manager()
        return self.breakpoint_manager
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hook statistics."""
        return {**self.stats}


def install_breakpoint_hooks(
    model: nn.Module,
    breakpoint_layer_index: int,
    session_id: str,
    activation_checkpointer=None,
    breakpoint_manager=None,
) -> BreakpointHookManager:
    """
    Install breakpoint hooks on a model.
    
    Convenience function to create and install hooks.
    
    Args:
        model: PyTorch model
        breakpoint_layer_index: Layer to pause at
        session_id: Session identifier
        activation_checkpointer: Optional ActivationCheckpointer instance
        breakpoint_manager: Optional BreakpointManager instance
    
    Returns:
        BreakpointHookManager instance
    """
    manager = BreakpointHookManager(
        model=model,
        breakpoint_layer_index=breakpoint_layer_index,
        session_id=session_id,
        activation_checkpointer=activation_checkpointer,
        breakpoint_manager=breakpoint_manager,
    )
    manager.install_hooks()
    return manager

