"""
HuggingFace backend adapter: Wraps HuggingFace Transformers as a Djinn backend.

This adapter implements the SwappableState and InferenceBackend interfaces
for HuggingFace DynamicCache, enabling semantic scheduler integration.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import torch

from djinn.interfaces import SwappableState, InferenceBackend

logger = logging.getLogger(__name__)


class HuggingFaceKVCache(SwappableState):
    """
    Adapter for HuggingFace DynamicCache to support GPU<->CPU swapping.
    
    Handles conversion between DynamicCache (HF's format) and legacy tuple format
    (which can be efficiently transferred to CPU).
    """
    
    def __init__(self, cache_obj: Any):
        """
        Initialize with a HuggingFace cache object.
        
        Args:
            cache_obj: DynamicCache or legacy tuple format
        """
        self.cache_obj = cache_obj
    
    def to_host_format(self) -> Tuple[Any, Dict[str, Any]]:
        """Convert DynamicCache to CPU-transferable format."""
        try:
            from transformers.cache_utils import Cache
            
            if isinstance(self.cache_obj, Cache):
                # Convert DynamicCache to legacy tuple format
                legacy_cache = self.cache_obj.to_legacy_cache()
                if legacy_cache is not None:
                    return (
                        legacy_cache,
                        {
                            'type': 'hf_dynamic_cache',
                            'num_layers': len(legacy_cache),
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to convert to legacy cache: {e}")
        
        # Fallback: return as-is
        return (
            self.cache_obj,
            {'type': 'hf_cache', 'format': type(self.cache_obj).__name__}
        )
    
    @classmethod
    def from_host_format(cls, cpu_data: Any, metadata: Dict[str, Any]) -> 'HuggingFaceKVCache':
        """Reconstruct DynamicCache from CPU format."""
        try:
            if metadata.get('type') == 'hf_dynamic_cache' and isinstance(cpu_data, (tuple, list)):
                # Reconstruct from legacy format
                from transformers.cache_utils import DynamicCache
                restored_kv = DynamicCache.from_legacy_cache(cpu_data)
                
                # Validate
                if hasattr(restored_kv, 'get_seq_length'):
                    return cls(restored_kv)
                else:
                    logger.error(f"[CRITICAL] from_legacy_cache returned invalid type: {type(restored_kv)}")
                    raise RuntimeError("DynamicCache reconstruction failed")
        except Exception as e:
            logger.error(f"Failed to reconstruct DynamicCache: {e}", exc_info=True)
            raise
        
        # Fallback: return as-is
        return cls(cpu_data)
    
    def gpu_size_bytes(self) -> int:
        """Report GPU memory footprint of the KV cache."""
        try:
            total_bytes = 0
            if isinstance(self.cache_obj, (tuple, list)):
                # Legacy tuple format: each layer has (k_tensor, v_tensor)
                for layer_kv in self.cache_obj:
                    if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                        k_t, v_t = layer_kv
                        if isinstance(k_t, torch.Tensor):
                            total_bytes += k_t.numel() * k_t.element_size()
                        if isinstance(v_t, torch.Tensor):
                            total_bytes += v_t.numel() * v_t.element_size()
            else:
                # DynamicCache format: iterate key_cache and value_cache
                from transformers.cache_utils import Cache
                if isinstance(self.cache_obj, Cache):
                    # Iterate through key_cache and value_cache tensors
                    if hasattr(self.cache_obj, 'key_cache'):
                        for tensor in self.cache_obj.key_cache:
                            if isinstance(tensor, torch.Tensor):
                                total_bytes += tensor.numel() * tensor.element_size()
                    if hasattr(self.cache_obj, 'value_cache'):
                        for tensor in self.cache_obj.value_cache:
                            if isinstance(tensor, torch.Tensor):
                                total_bytes += tensor.numel() * tensor.element_size()
            
            return max(total_bytes, 0)
        except Exception as e:
            logger.warning(f"Could not calculate KV cache size: {e}")
            return 0


class HuggingFaceBackend(InferenceBackend):
    """
    Backend implementation for HuggingFace Transformers.
    
    NOTE: This is a reference implementation for future multi-backend support.
    Currently, Djinn uses direct model execution via hybrid_executor.py.
    This backend will be wired into the main execution path post-OSDI
    to enable vLLM and TensorRT-LLM integrations without modifying core logic.
    """
    
    def __init__(self):
        """Initialize the HuggingFace backend."""
        self.models: Dict[str, Any] = {}
        self.session_states: Dict[str, Optional[HuggingFaceKVCache]] = {}
        logger.info("HuggingFaceBackend initialized")
    
    async def execute(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        session_state: Optional[SwappableState] = None
    ) -> Tuple[Any, Optional[SwappableState]]:
        """Execute inference using HuggingFace model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded. Call load_model first.")
        
        model = self.models[model_id]
        session_id = inputs.get('_session_id')
        
        # Prepare inputs
        exec_inputs = {k: v for k, v in inputs.items() if not k.startswith('_')}
        
        # Set KV cache from session state if provided
        if session_state is not None and isinstance(session_state, HuggingFaceKVCache):
            exec_inputs['past_key_values'] = session_state.cache_obj
        
        # Execute model
        try:
            output = model(**exec_inputs)
            
            # Extract updated KV cache
            updated_state = None
            if hasattr(output, 'past_key_values') and output.past_key_values is not None:
                updated_state = HuggingFaceKVCache(output.past_key_values)
                
                # Store in session tracking
                if session_id:
                    self.session_states[session_id] = updated_state
            
            return output.logits, updated_state
        
        except Exception as e:
            logger.error(f"HuggingFace execution error: {e}", exc_info=True)
            raise
    
    async def load_model(self, model_id: str, **kwargs) -> None:
        """Load a HuggingFace model."""
        if model_id in self.models:
            logger.debug(f"Model {model_id} already loaded")
            return
        
        try:
            from transformers import AutoModelForCausalLM
            logger.info(f"Loading HuggingFace model: {model_id}")
            
            # Load with kwargs (dtype, device, etc.)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=kwargs.get('torch_dtype', 'float16'),
                device_map=kwargs.get('device_map', 'cuda'),
                **{k: v for k, v in kwargs.items() if k not in ['torch_dtype', 'device_map']}
            )
            self.models[model_id] = model
            logger.info(f"✅ Loaded: {model_id}")
        
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}", exc_info=True)
            raise
    
    def get_session_state(self, session_id: str) -> Optional[SwappableState]:
        """Get KV cache for a session."""
        return self.session_states.get(session_id)
    
    def set_session_state(self, session_id: str, state: Optional[SwappableState]) -> None:
        """Set KV cache for a session."""
        if state is None:
            self.session_states.pop(session_id, None)
        elif isinstance(state, HuggingFaceKVCache):
            self.session_states[session_id] = state
        else:
            logger.warning(f"Unexpected state type: {type(state)}")
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """Return loaded models."""
        return self.models.copy()

