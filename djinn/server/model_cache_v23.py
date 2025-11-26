"""
Model Cache for v2.3 - Server-side model storage in VMU.

Stores registered models in GPU memory (VMU slab) for fast execution.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional
import threading

logger = logging.getLogger(__name__)


class ModelCacheV23:
    """
    Server-side model cache using VMU for storage.
    
    Models are loaded into GPU memory once and reused across requests.
    """
    
    def __init__(self, vmu=None):
        """
        Initialize model cache.
        
        Args:
            vmu: VMU instance for memory management
        """
        from ..backend.runtime.unified_vmu import get_vmu
        self.vmu = vmu or get_vmu()
        
        # Cache: fingerprint -> model
        self.models: Dict[str, nn.Module] = {}
        self.lock = threading.Lock()
        
        logger.info("✅ ModelCacheV23 initialized")
    
    def register_model(self, fingerprint: str, model: nn.Module, model_id: Optional[str] = None):
        """
        Register model in cache using Text Segment for weights.

        Args:
            fingerprint: Model fingerprint
            model: PyTorch model
            model_id: Optional model identifier
        """
        with self.lock:
            if fingerprint in self.models:
                logger.debug(f"Model {fingerprint[:8]} already in cache")
                return

            # Use Text Segment for model weight storage
            model_id = model_id or fingerprint

            # Extract state dict for Text Segment loading
            state_dict = model.state_dict()

            # Load weights into Text Segment (shared, read-only)
            success = self.vmu.load_model_to_text(model_id, state_dict)
            if success:
                logger.info(f"✅ Model {model_id} weights loaded into Text Segment")
            else:
                logger.info(f"ℹ️  Model {model_id} weights already in Text Segment (shared)")

            # Replace model parameters with Text Segment views
            if model_id in self.vmu.text_segment.loaded_models:
                model_data = self.vmu.text_segment.loaded_models[model_id]
                param_views = model_data['param_views']

            # Replace each parameter/buffer with VMU-backed view
                named_params = dict(model.named_parameters())
                named_buffers = dict(model.named_buffers())
                with torch.no_grad():
                    for name, view in param_views.items():
                        target = None
                        if name in named_params:
                            target = named_params[name]
                        elif name in named_buffers:
                            target = named_buffers[name]
                        if target is not None:
                            target.data = view
                            logger.debug(f"Replaced {name} with VMU-backed view")
                        else:
                            logger.warning(f"State dict key {name} not found in parameters or buffers")

                # Move any remaining buffers to VMU device (non-persistent buffers)
                for buf_name, buffer in named_buffers.items():
                    if buffer.device != self.vmu.device:
                        buffer.data = buffer.to(self.vmu.device)
                        logger.debug(f"Moved buffer {buf_name} to {self.vmu.device}")

                logger.info(f"✅ Model {model_id} parameters now backed by Text Segment")
            else:
                logger.error(f"Model {model_id} not found in Text Segment after loading")
                return

            # Verify model is on correct device
            if torch.cuda.is_available():
                model_device = next(model.parameters()).device if list(model.parameters()) else None
                if model_device != self.vmu.device:
                    logger.warning(f"Model parameters not on expected device: {model_device} vs {self.vmu.device}")
                else:
                    logger.info(f"✅ Model parameters on VMU device: {model_device}")

            # Set to eval mode
            model.eval()

            # Store in cache (model object points to Text Segment weights)
            self.models[fingerprint] = model

            # Log memory usage
            text_stats = self.vmu.text_segment.get_stats()
            logger.info(
                f"✅ Model {fingerprint[:8]} cached with Text Segment backing: "
                f"{text_stats['loaded_bytes'] / 1024**2:.1f}MB total weights"
            )
    
    def get_model(self, fingerprint: str) -> Optional[nn.Module]:
        """
        Get model from cache.
        
        Args:
            fingerprint: Model fingerprint
            
        Returns:
            Model if found, None otherwise
        """
        with self.lock:
            return self.models.get(fingerprint)
    
    def clear(self):
        """Clear all cached models."""
        with self.lock:
            self.models.clear()
            logger.info("Model cache cleared")


# Global instance
_global_cache: Optional[ModelCacheV23] = None


def get_model_cache_v23() -> ModelCacheV23:
    """Get or create global model cache v2.3."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ModelCacheV23()
    return _global_cache

