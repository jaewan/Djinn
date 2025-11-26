"""
Ghost Model Interception: Hooks HuggingFace model loading to prevent weight downloads on client.

This implements the v2.3 architecture feature: "Ghost Model Interception"

Instead of downloading model weights to the client, we:
1. Intercept from_pretrained() calls
2. Forward model ID and authentication to server
3. Create a "ghost" model on client (device='meta', zero memory)
4. Return a DjinnWrapper that delegates execution to server

This achieves the v2.3 goal: "The Data never touches the Client until requested."
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)


class DjinnModelWrapper:
    """
    Wraps a ghost model created on meta device.
    
    When called, delegates execution to remote server instead of local computation.
    Tracks model fingerprint for server-side caching.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 model_id: str,
                 auth_token: Optional[str] = None,
                 remote_handle: Optional[str] = None):
        """
        Initialize wrapper around ghost model.
        
        Args:
            model: Ghost model created on meta device
            model_id: HuggingFace model identifier (e.g., "gpt2-xl")
            auth_token: HuggingFace authentication token for gated models
            remote_handle: Server-side handle for remote model
        """
        self.model = model
        self.model_id = model_id
        self.auth_token = auth_token
        self.remote_handle = remote_handle
        
        # Track that this model should use remote execution
        self._is_ghost_model = True
        
        logger.info(f"🎭 Created ghost model wrapper for {model_id}")
    
    def __call__(self, *args, **kwargs):
        """Forward call to remote executor instead of local computation."""
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Execute model on remote server."""
        from ...core.enhanced_model_manager import get_model_manager
        
        manager = get_model_manager()
        
        # Convert args to dict format for remote execution
        inputs_dict = {}
        if len(args) > 0:
            inputs_dict['input_ids'] = args[0]
        inputs_dict.update(kwargs)
        
        # Execute via model cache on server
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop - create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            manager.execute_model(self.model, inputs_dict)
        )
        
        return result
    
    def to(self, *args, **kwargs):
        """Device movement (no-op for ghost models)."""
        # Ghost models are always on meta device
        # No actual device movement needed
        logger.debug(f"Ghost model {self.model_id} .to() called - no-op")
        return self
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped model."""
        return getattr(self.model, name)


class ModelLoaderInterceptor:
    """
    Intercepts HuggingFace model loading to create ghost models.
    
    Installation:
        interceptor = ModelLoaderInterceptor()
        interceptor.install_hooks()
    
    After installation, calls to AutoModel.from_pretrained() will:
    1. Check Djinn config
    2. If enabled: create ghost model + register with server
    3. If disabled: fall back to standard HuggingFace loading
    """
    
    def __init__(self, auto_register: bool = True):
        """
        Initialize interceptor.
        
        Args:
            auto_register: Whether to automatically register ghost models with server
        """
        self.auto_register = auto_register
        self._original_from_pretrained = None
        self._hooked = False
    
    def install_hooks(self):
        """Patch HuggingFace transformers library."""
        if self._hooked:
            return
        
        try:
            from transformers import AutoModel, AutoModelForCausalLM
            
            # Store original methods
            self._original_from_pretrained = AutoModel.from_pretrained
            self._original_causal_lm = AutoModelForCausalLM.from_pretrained
            
            # Install hooks
            AutoModel.from_pretrained = self._patched_from_pretrained(AutoModel)
            AutoModelForCausalLM.from_pretrained = self._patched_from_pretrained(AutoModelForCausalLM)
            
            self._hooked = True
            logger.info("✅ Ghost Model Interception hooks installed")
            
        except ImportError:
            logger.warning("⚠️  transformers library not found - model loader interception unavailable")
    
    def _patched_from_pretrained(self, model_class):
        """Create patched version of from_pretrained."""
        original = model_class.from_pretrained
        
        @wraps(original)
        def from_pretrained_hook(model_id, **kwargs):
            """
            Intercepts HuggingFace model loading.
            
            Flow:
            1. Check if Djinn interception is enabled
            2. If yes: Create ghost model + optionally register with server
            3. If no: Fall back to standard loading
            """
            
            # Check if interception is enabled
            try:
                from djinn.config import get_config
                config = get_config()
                if not getattr(config, 'intercept_huggingface', False):
                    # Interception disabled - use standard loading
                    logger.debug(f"Model interception disabled, loading {model_id} normally")
                    return original(model_id, **kwargs)
            except (ImportError, AttributeError):
                # Config not available - use standard loading
                return original(model_id, **kwargs)
            
            logger.info(f"🎭 Intercepting model load: {model_id}")
            
            try:
                # Extract authentication token
                token = kwargs.get('token') or kwargs.get('use_auth_token')
                if isinstance(token, bool) and token:
                    # Need to get token from cache
                    try:
                        from huggingface_hub import HfFolder
                        token = HfFolder.get_token()
                    except (ImportError, AttributeError):
                        token = None
                
                # Create config for ghost model
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_id, token=token)
                    logger.debug(f"Loaded config for {model_id}")
                except Exception as e:
                    logger.error(f"Failed to load config for {model_id}: {e}")
                    # Fall back to standard loading
                    return original(model_id, **kwargs)
                
                # Create ghost model on meta device (zero memory footprint)
                try:
                    logger.debug(f"Creating ghost model on meta device")
                    with torch.device('meta'):
                        ghost_model = model_class.from_config(config)
                    
                    # Move to remote_accelerator if specified in kwargs
                    device = kwargs.get('device', 'remote_accelerator:0')
                    if isinstance(device, str) and 'remote' in device.lower():
                        ghost_model = ghost_model.to(device)
                    
                    logger.info(f"✅ Created ghost model for {model_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to create ghost model: {e}")
                    # Fall back to standard loading
                    return original(model_id, **kwargs)
                
                # Wrap ghost model
                wrapper = DjinnModelWrapper(
                    model=ghost_model,
                    model_id=model_id,
                    auth_token=token
                )
                
                # Optionally register with server for faster execution
                if self.auto_register:
                    try:
                        from ...core.enhanced_model_manager import get_model_manager
                        import asyncio
                        
                        manager = get_model_manager()
                        
                        # Register asynchronously in background
                        # (don't block model loading on registration)
                        try:
                            loop = asyncio.get_running_loop()
                            # Schedule registration as task (don't await)
                            asyncio.create_task(manager.register_model(ghost_model))
                            logger.debug("Model registration scheduled in background")
                        except RuntimeError:
                            # No running loop - skip background registration
                            logger.debug("No async loop for background registration")
                        
                    except Exception as e:
                        logger.debug(f"Could not register model: {e}")
                
                return wrapper
            
            except Exception as e:
                logger.error(f"Ghost model interception failed: {e}")
                # Fall back to standard loading
                logger.info(f"Falling back to standard loading for {model_id}")
                return original(model_id, **kwargs)
        
        return from_pretrained_hook
    
    def uninstall_hooks(self):
        """Restore original model loading."""
        if not self._hooked:
            return
        
        try:
            from transformers import AutoModel, AutoModelForCausalLM
            
            if self._original_from_pretrained:
                AutoModel.from_pretrained = self._original_from_pretrained
                AutoModelForCausalLM.from_pretrained = self._original_causal_lm
            
            self._hooked = False
            logger.info("✅ Ghost Model Interception hooks removed")
        
        except ImportError:
            pass


# Global interceptor instance
_global_interceptor: Optional[ModelLoaderInterceptor] = None


def install_model_loader_hooks():
    """Install ghost model interception hooks globally."""
    global _global_interceptor
    
    if _global_interceptor is None:
        _global_interceptor = ModelLoaderInterceptor()
    
    _global_interceptor.install_hooks()


def uninstall_model_loader_hooks():
    """Uninstall ghost model interception hooks."""
    global _global_interceptor
    
    if _global_interceptor is not None:
        _global_interceptor.uninstall_hooks()

