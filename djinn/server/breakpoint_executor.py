"""
Breakpoint Executor: Modified execution loop with breakpoint support.

Purpose:
- Extend standard execution to support pause/resume at breakpoints
- Coordinate with BreakpointManager for state transitions
- Handle checkpoint restoration and partial execution
- Provide metrics for evaluating context switch overhead

Architecture:
- Pre-execution: Check if breakpoint configured, install hooks
- Execution: Model runs normally, hooks checkpoint at breakpoint layer
- Post-breakpoint: Wait for resume signal from manager
- Resume-execution: Restore activations, continue execution
- Post-execution: Cleanup and metrics collection
"""

import logging
import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple, Callable
import threading

logger = logging.getLogger(__name__)


class BreakpointExecutor:
    """
    Executor with breakpoint support for mid-inference pause/resume.
    
    Handles:
    1. Setting up breakpoint hooks before execution
    2. Waiting for pause signal at breakpoint
    3. Handling resume and activation restoration
    4. Tracking metrics (checkpoint/restore times, overhead)
    """
    
    def __init__(
        self,
        activation_checkpointer=None,
        breakpoint_manager=None,
    ):
        """
        Initialize breakpoint executor.
        
        Args:
            activation_checkpointer: ActivationCheckpointer instance
            breakpoint_manager: BreakpointManager instance
        """
        self.activation_checkpointer = activation_checkpointer
        self.breakpoint_manager = breakpoint_manager
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "breakpoint_executions": 0,
            "successful_pauses": 0,
            "successful_resumes": 0,
            "total_pause_time_seconds": 0.0,
            "total_restore_time_seconds": 0.0,
            "max_checkpoint_size_mb": 0.0,
            "errors": 0,
        }
        
        logger.info("✅ BreakpointExecutor initialized")
    
    def execute_with_breakpoint(
        self,
        session_id: str,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        breakpoint_layer_index: int,
        wait_for_resume: bool = True,
        resume_timeout_seconds: float = 300.0,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute model with breakpoint support.
        
        Flow:
        1. Install breakpoint hooks
        2. Run model (pauses at breakpoint layer via hook)
        3. Wait for resume signal (or timeout)
        4. Restore activations from checkpoint
        5. Continue execution
        6. Return results and metrics
        
        Args:
            session_id: Session identifier
            model: PyTorch model to execute
            inputs: Input tensors (dict)
            breakpoint_layer_index: Layer to pause at
            wait_for_resume: If True, wait for resume signal; if False, return after checkpoint
            resume_timeout_seconds: Timeout for waiting on resume signal
        
        Returns:
            (model_output, metrics_dict)
        
        Raises:
            RuntimeError: If execution or checkpoint fails
            TimeoutError: If resume timeout exceeded
        """
        from djinn.backend.runtime.breakpoint_hooks import install_breakpoint_hooks
        from djinn.server.activation_checkpointer import get_activation_checkpointer
        from djinn.server.breakpoint_manager import get_breakpoint_manager
        
        # Get managers
        checkpointer = self.activation_checkpointer or get_activation_checkpointer()
        manager = self.breakpoint_manager or get_breakpoint_manager()
        
        execution_start = time.perf_counter()
        metrics = {
            "session_id": session_id,
            "breakpoint_layer": breakpoint_layer_index,
            "checkpoint_time_ms": 0.0,
            "pause_duration_ms": 0.0,
            "restore_time_ms": 0.0,
            "total_overhead_ms": 0.0,
            "checkpoint_size_mb": 0.0,
            "model_output": None,
        }
        
        try:
            # Register breakpoint
            manager.register_breakpoint(session_id, breakpoint_layer_index)
            
            # Install breakpoint hooks
            hook_manager = install_breakpoint_hooks(
                model=model,
                breakpoint_layer_index=breakpoint_layer_index,
                session_id=session_id,
                activation_checkpointer=checkpointer,
                breakpoint_manager=manager,
            )
            
            logger.info(
                f"[{session_id}] Starting execution with breakpoint at layer {breakpoint_layer_index}"
            )
            
            # Get model device
            model_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            logger.debug(f"[{session_id}] Model device: {model_device}")
            
            # Move inputs to model device
            if isinstance(inputs, dict):
                inputs_on_device = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in inputs.items()}
            elif isinstance(inputs, torch.Tensor):
                inputs_on_device = inputs.to(model_device)
            else:
                inputs_on_device = [v.to(model_device) if isinstance(v, torch.Tensor) else v 
                                   for v in inputs]
            
            # Execute model (will pause at breakpoint via hook)
            # Handle both dict inputs (for transformers) and direct tensors (for Sequential)
            with torch.no_grad():
                if isinstance(inputs_on_device, dict):
                    model_output = model(**inputs_on_device)
                elif isinstance(inputs_on_device, torch.Tensor):
                    model_output = model(inputs_on_device)
                else:
                    model_output = model(*inputs_on_device)
            
            # Check if we hit the breakpoint
            breakpoint = manager.get_breakpoint(session_id)
            if not breakpoint or breakpoint.checkpoint_id is None:
                logger.warning(f"[{session_id}] Breakpoint not triggered, execution completed")
                metrics["model_output"] = model_output
                
                # Extract logits only for efficient serialization
                # (avoid serializing hidden_states, past_key_values, etc.)
                if isinstance(model_output, dict) and 'logits' in model_output:
                    model_output = {'logits': model_output['logits']}
                    logger.debug(f"[{session_id}] Extracted logits only for serialization efficiency")
                
                return model_output, metrics
            
            checkpoint_id = breakpoint.checkpoint_id
            checkpoint = checkpointer.get_checkpoint(checkpoint_id)
            if checkpoint:
                metrics["checkpoint_size_mb"] = checkpoint.total_bytes / 1024**2
            
            self.stats["successful_pauses"] += 1
            
            if not wait_for_resume:
                logger.info(f"[{session_id}] Breakpoint reached, returning (wait_for_resume=False)")
                metrics["model_output"] = None  # Partial execution
                metrics["checkpoint_id"] = checkpoint_id
                metrics["state"] = "paused"
                metrics["session_id"] = session_id  # Important: client needs this for resume RPC
                
                checkpoint_activation_tensor = None
                try:
                    # Restore activations to CPU so the client can inspect/modify them
                    cpu_device = torch.device('cpu')
                    activations_cpu, _ = checkpointer.restore(
                        checkpoint_id=checkpoint_id,
                        device=cpu_device,
                    )
                    logger.debug(f"[{session_id}] Restored activations keys: {list(activations_cpu.keys())}")
                    output_key = 'output' if 'output' in activations_cpu else 'output_0'
                    checkpoint_activation_tensor = activations_cpu.get(output_key)
                    if checkpoint_activation_tensor is not None:
                        logger.info(
                            f"[{session_id}] ✅ Prepared checkpoint_activation for client "
                            f"(shape={checkpoint_activation_tensor.shape}, dtype={checkpoint_activation_tensor.dtype})"
                        )
                    else:
                        logger.warning(f"[{session_id}] ❌ Failed to extract activation from key '{output_key}'")
                except Exception as activation_err:
                    logger.error(
                        f"[{session_id}] ❌ Failed to materialize checkpoint activation for client: {activation_err}",
                        exc_info=True
                    )
                
                result_payload = {
                    'checkpoint_activation': checkpoint_activation_tensor
                } if checkpoint_activation_tensor is not None else {}
                logger.info(f"[{session_id}] Returning result_payload with {len(result_payload)} keys (checkpoint_activation present: {checkpoint_activation_tensor is not None})")
                
                # Do NOT unregister breakpoint—the session remains paused until resume RPC
                metrics["awaiting_resume"] = True
                return result_payload, metrics
            
            # Wait for resume signal
            pause_start = time.perf_counter()
            logger.info(f"[{session_id}] Paused at breakpoint, waiting for resume signal...")
            
            # In a real implementation, this would wait on an event or condition variable
            # For now, we simulate a brief pause
            pause_duration = self._wait_for_resume(
                session_id=session_id,
                manager=manager,
                timeout_seconds=resume_timeout_seconds
            )
            
            pause_end = time.perf_counter()
            metrics["pause_duration_ms"] = pause_duration * 1000
            
            # Restore activations
            logger.info(f"[{session_id}] Resuming, restoring activations from checkpoint...")
            restore_start = time.perf_counter()
            
            activations, restore_time = checkpointer.restore(
                checkpoint_id=checkpoint_id,
                device=next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            )
            
            restore_end = time.perf_counter()
            metrics["restore_time_ms"] = restore_time * 1000
            
            # Update breakpoint state
            manager.trigger_resume(session_id)
            manager.mark_resumed(session_id)
            
            self.stats["successful_resumes"] += 1
            
            # Continue execution from breakpoint
            logger.info(
                f"[{session_id}] Continuing execution from layer {breakpoint_layer_index + 1}..."
            )
            
            # ✅ PROPER SEMANTICS: Continue from layer N+1 using restored activations
            # Store restored activations for _continue_from_layer to access attention_mask
            logger.info(f"[{session_id}] DEBUG: activations type: {type(activations)}")
            logger.info(f"[{session_id}] DEBUG: activations is None: {activations is None}")
            if activations is not None:
                logger.info(f"[{session_id}] DEBUG: activations keys: {list(activations.keys()) if hasattr(activations, 'keys') else 'no keys method'}")
                logger.info(f"[{session_id}] DEBUG: 'output' in activations: {'output' in activations if hasattr(activations, '__contains__') else 'no contains method'}")
                for key, value in activations.items():
                    logger.info(f"[{session_id}] DEBUG: activations['{key}'] shape: {value.shape if hasattr(value, 'shape') else type(value)}")

            if activations and ('output' in activations or 'output_0' in activations):
                try:
                    # ✅ CRITICAL: Store activations for _continue_from_layer to access attention_mask
                    self._restored_activations = activations
                    
                    # Get the intermediate activation from checkpoint
                    # GPT-2 layers return tuples, so we use 'output_0' for the main output
                    output_key = 'output' if 'output' in activations else 'output_0'
                    checkpoint_activation = activations[output_key]
                    logger.info(
                        f"[{session_id}] Restored activation shape: {checkpoint_activation.shape}, "
                        f"device: {checkpoint_activation.device}"
                    )
                    
                    # Extract layers and continue from layer_index + 1
                    layer_index = breakpoint_layer_index
                    model_output = self._continue_from_layer(
                        model=model,
                        layer_index=layer_index,
                        checkpoint_activation=checkpoint_activation,
                        session_id=session_id
                    )
                    logger.info(
                        f"[{session_id}] Continued execution from layer {layer_index + 1} "
                        f"to final output"
                    )
                    
                    # Clear stored activations after continuation
                    self._restored_activations = None
                except Exception as e:
                    logger.warning(
                        f"[{session_id}] Failed to continue from checkpoint: {e}. "
                        f"Falling back to full model execution."
                    )
                    # Clear stored activations on error
                    self._restored_activations = None
                    
                    # Fallback: run full model
                    with torch.no_grad():
                        if isinstance(inputs_on_device, dict):
                            model_output = model(**inputs_on_device)
                        elif isinstance(inputs_on_device, torch.Tensor):
                            model_output = model(inputs_on_device)
                        else:
                            model_output = model(*inputs_on_device)
            else:
                # No saved activations: run full model for reference
                logger.info(f"[{session_id}] No saved activations, running full model")
                with torch.no_grad():
                    if isinstance(inputs_on_device, dict):
                        model_output = model(**inputs_on_device)
                    elif isinstance(inputs_on_device, torch.Tensor):
                        model_output = model(inputs_on_device)
                    else:
                        model_output = model(*inputs_on_device)
            
            metrics["model_output"] = model_output
            
            # Calculate total overhead
            total_execution_time = time.perf_counter() - execution_start
            metrics["total_overhead_ms"] = (
                metrics["checkpoint_time_ms"] + 
                metrics["pause_duration_ms"] + 
                metrics["restore_time_ms"]
            )
            
            overhead_percent = (
                (metrics["total_overhead_ms"] / (total_execution_time * 1000)) * 100
                if total_execution_time > 0 else 0.0
            )
            metrics["overhead_percent"] = overhead_percent
            
            logger.info(
                f"✅ [Session {session_id}] Breakpoint execution completed:\n"
                f"  Checkpoint: {metrics['checkpoint_time_ms']:.1f}ms\n"
                f"  Pause: {metrics['pause_duration_ms']:.1f}ms\n"
                f"  Restore: {metrics['restore_time_ms']:.1f}ms\n"
                f"  Total Overhead: {metrics['total_overhead_ms']:.1f}ms ({overhead_percent:.1f}%)"
            )
            
            self.stats["breakpoint_executions"] += 1
            self.stats["total_pause_time_seconds"] += pause_duration
            self.stats["total_restore_time_seconds"] += restore_time
            self.stats["max_checkpoint_size_mb"] = max(
                self.stats["max_checkpoint_size_mb"],
                metrics["checkpoint_size_mb"]
            )
            
            # Extract logits only for efficient serialization
            # (avoid serializing hidden_states, past_key_values, etc.)
            if isinstance(model_output, dict) and 'logits' in model_output:
                model_output = {'logits': model_output['logits']}
                logger.debug(f"[{session_id}] Extracted logits only for serialization efficiency")
            
            return model_output, metrics
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"❌ Breakpoint execution failed: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup breakpoint tracking only after full resume
            if wait_for_resume:
                manager.unregister_breakpoint(session_id)
            else:
                logger.debug(f"[{session_id}] Breakpoint remains registered (awaiting resume RPC)")
            try:
                hook_manager.remove_hooks()
            except Exception as e:
                logger.warning(f"Error removing hooks: {e}")
            logger.debug(f"[{session_id}] Cleaned up breakpoint hooks")
    
    def resume_from_modified_activation(
        self,
        session_id: str,
        model: nn.Module,
        layer_index: int,
        modified_activation: torch.Tensor,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Resume execution from a paused checkpoint using a user-modified activation tensor.
        """
        from djinn.server.activation_checkpointer import get_activation_checkpointer
        from djinn.server.breakpoint_manager import get_breakpoint_manager, BreakpointState

        checkpointer = self.activation_checkpointer or get_activation_checkpointer()
        manager = self.breakpoint_manager or get_breakpoint_manager()

        breakpoint = manager.get_breakpoint(session_id)
        if breakpoint is None:
            raise RuntimeError(f"No breakpoint registered for session {session_id}")
        if breakpoint.state not in (BreakpointState.PAUSED, BreakpointState.RESUMING):
            raise RuntimeError(
                f"Session {session_id} is not paused (state={breakpoint.state}); cannot resume"
            )
        if breakpoint.checkpoint_id is None:
            raise RuntimeError(f"No checkpoint available for session {session_id}")

        checkpoint_id = breakpoint.checkpoint_id
        model_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

        activations, restore_time = checkpointer.restore(
            checkpoint_id=checkpoint_id,
            device=model_device,
        )

        output_key = 'output' if 'output' in activations else 'output_0'
        if output_key not in activations:
            raise RuntimeError(f"Checkpoint activations missing '{output_key}' for session {session_id}")

        modified_on_device = modified_activation.to(model_device)
        if activations[output_key].shape != modified_on_device.shape:
            raise ValueError(
                f"Modified activation shape {tuple(modified_on_device.shape)} "
                f"!= checkpoint shape {tuple(activations[output_key].shape)}"
            )

        activations[output_key] = modified_on_device
        self._restored_activations = activations

        # Resume semantics
        manager.trigger_resume(session_id)
        manager.mark_resumed(session_id)

        metrics: Dict[str, Any] = {
            "session_id": session_id,
            "resume_layer": layer_index,
            "steering_applied": True,
            "restore_time_ms": restore_time * 1000,
            "modified_activation_shape": list(modified_on_device.shape),
        }

        try:
            model_output = self._continue_from_layer(
                model=model,
                layer_index=layer_index,
                checkpoint_activation=modified_on_device,
                session_id=session_id,
            )
            metrics["model_output"] = "completed"
            self.stats["successful_resumes"] += 1
        finally:
            self._restored_activations = None
            try:
                checkpointer.release_checkpoint(checkpoint_id)
            except Exception as release_error:
                logger.warning(
                    f"[{session_id}] Failed to release checkpoint {checkpoint_id}: {release_error}"
                )
            try:
                manager.unregister_breakpoint(session_id)
            except Exception as unregister_err:
                logger.warning(f"[{session_id}] Failed to unregister breakpoint: {unregister_err}")

        return model_output, metrics
    
    def _continue_from_layer(
        self,
        model: nn.Module,
        layer_index: int,
        checkpoint_activation: torch.Tensor,
        session_id: str
    ) -> Any:
        """
        Continue model execution from layer_index + 1 using restored activation.
        
        ✅ CRITICAL FIX: Passes attention_mask to transformer layers for correctness
        
        Args:
            model: The model
            layer_index: Index of layer where we paused (0-based)
            checkpoint_activation: Activation tensor from layer_index's output
            session_id: Session identifier for logging
        
        Returns:
            Final model output (logits or prediction)
        
        Raises:
            RuntimeError: If unable to extract layers or continue execution
        """
        from djinn.backend.runtime.breakpoint_hooks import BreakpointHookManager
        
        logger.info(f"[{session_id}] Starting _continue_from_layer: layer_index={layer_index}, activation_shape={checkpoint_activation.shape}")
        
        # Extract layers using existing pattern
        logger.info(f"[{session_id}] Creating BreakpointHookManager for layer_index={layer_index}")
        hook_manager = BreakpointHookManager(
            model=model,
            breakpoint_layer_index=layer_index,
            session_id=session_id
        )
        layers = hook_manager.layers
        logger.info(f"[{session_id}] Extracted {len(layers) if layers else 0} layers")

        if not layers or layer_index >= len(layers):
            error_msg = (
                f"Cannot extract layers: {len(layers) if layers else 0} layers found, "
                f"breakpoint at {layer_index}"
            )
            logger.error(f"[{session_id}] {error_msg}")
            raise RuntimeError(error_msg)
        
        # ✅ CRITICAL: Get attention_mask from restored activations
        # This is saved by _collect_activations in breakpoint_hooks.py
        attention_mask = None
        if hasattr(self, '_restored_activations') and self._restored_activations:
            attention_mask = self._restored_activations.get('attention_mask')
            if attention_mask is None:
                attention_mask = self._restored_activations.get('input_1')
            if attention_mask is not None:
                logger.info(f"[{session_id}] Using saved attention_mask shape {attention_mask.shape}")
            else:
                logger.warning(f"[{session_id}] No attention_mask found in restored activations")
        else:
            logger.warning(f"[{session_id}] No restored activations available for attention_mask")
        
        # Execute layers from layer_index + 1 to end
        current_output = checkpoint_activation

        logger.info(
            f"[{session_id}] Starting continuation from layer {layer_index + 1}/{len(layers)} "
            f"with activation shape {current_output.shape}"
        )

        with torch.no_grad():
            # Pass through remaining layers with attention mask
            for i in range(layer_index + 1, len(layers)):
                layer = layers[i]
                logger.info(f"[{session_id}] Executing layer {i} ({type(layer).__name__})")
                try:
                    # ✅ CRITICAL: Pass attention_mask to transformer layers
                    # This ensures correct attention computation (not garbage output)
                    if attention_mask is not None:
                        try:
                            # Try calling with attention_mask (standard transformer API)
                            outputs = layer(current_output, attention_mask=attention_mask)
                        except TypeError:
                            # Some layers might not accept attention_mask
                            logger.debug(f"[{session_id}] Layer {i} doesn't accept attention_mask, calling without it")
                            outputs = layer(current_output)
                    else:
                        outputs = layer(current_output)
                    
                    # ✅ Handle tuple returns (GPT-2 returns (hidden_states, present_kv, ...))
                    # Extract hidden states from position 0 if tuple
                    if isinstance(outputs, tuple):
                        if len(outputs) == 0:
                            raise RuntimeError(f"Layer {i} returned empty tuple")
                        current_output = outputs[0]
                        logger.debug(f"[{session_id}] Layer {i} returned tuple, extracted hidden_states")
                    elif outputs is None:
                        raise RuntimeError(f"Layer {i} returned None")
                    else:
                        current_output = outputs
                    
                    logger.info(
                        f"[{session_id}] Layer {i} completed, output shape: {current_output.shape if hasattr(current_output, 'shape') else type(current_output)}"
                    )
                except Exception as e:
                    logger.error(
                        f"[{session_id}] Failed at layer {i}: {e}"
                    )
                    raise RuntimeError(f"Failed executing layer {i}: {e}") from e
        
        # After passing through all transformer layers, apply final layer norm and LM head
        # This is specific to GPT-2 architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            # Apply final layer norm
            current_output = model.transformer.ln_f(current_output)
            logger.debug(f"[{session_id}] Applied final layer norm")
        
        if hasattr(model, 'lm_head'):
            # Apply language model head to get logits
            current_output = model.lm_head(current_output)
            logger.debug(
                f"[{session_id}] Applied LM head, final logits shape: {current_output.shape}"
            )
        elif hasattr(model, 'score'):
            # Alternative name for LM head in some models
            current_output = model.score(current_output)
            logger.debug(f"[{session_id}] Applied score layer")
        
        # ✅ CRITICAL: Wrap output in dict to match model's expected return format
        # GPT-2 and other transformers return a dict with 'logits' key
        # This ensures compatibility with the full model execution's return format
        output_dict = {
            'logits': current_output
        }
        logger.debug(f"[{session_id}] Wrapped output in dict with 'logits' key")
        
        return output_dict

    def _wait_for_resume(
        self,
        session_id: str,
        manager,
        timeout_seconds: float
    ) -> float:
        """
        Wait for resume signal from another component.
        
        In a real system, this would wait on an event/condition variable
        that gets signaled when another job completes and GPU is available.
        
        For evaluation, this simulates realistic pause durations.
        
        Args:
            session_id: Session waiting to resume
            manager: BreakpointManager
            timeout_seconds: Maximum wait time
        
        Returns:
            Actual pause duration in seconds
        
        Raises:
            TimeoutError: If timeout exceeded
        """
        start_time = time.perf_counter()
        
        # Simulate waiting (in real implementation, this would be an event)
        # For evaluation, we wait a brief period or until signaled
        wait_time = 0.1  # 100ms default
        
        elapsed = 0.0
        while elapsed < timeout_seconds and elapsed < wait_time:
            time.sleep(0.01)
            elapsed = time.perf_counter() - start_time
        
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Resume timeout exceeded for session {session_id} "
                f"(waited {elapsed:.1f}s)"
            )
        
        return elapsed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            return {
                **self.stats,
                "avg_pause_time_ms": (
                    (self.stats["total_pause_time_seconds"] * 1000 / self.stats["successful_pauses"])
                    if self.stats["successful_pauses"] > 0 else 0.0
                ),
                "avg_restore_time_ms": (
                    (self.stats["total_restore_time_seconds"] * 1000 / self.stats["successful_resumes"])
                    if self.stats["successful_resumes"] > 0 else 0.0
                ),
            }


# Global singleton instance
_global_breakpoint_executor: Optional[BreakpointExecutor] = None
_executor_lock = threading.Lock()


def get_breakpoint_executor() -> BreakpointExecutor:
    """Get or create global breakpoint executor singleton."""
    global _global_breakpoint_executor
    
    if _global_breakpoint_executor is None:
        with _executor_lock:
            if _global_breakpoint_executor is None:
                _global_breakpoint_executor = BreakpointExecutor()
    
    return _global_breakpoint_executor

