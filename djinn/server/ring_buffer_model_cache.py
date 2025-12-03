"""
Ring Buffer Model Cache: Intelligent wrapper for model caching.

Routes models to either standard cache (small models) or ring buffer (oversized models).
Enables running 140GB+ models on 60GB VRAM by streaming weights during inference.

Key Features:
- Automatic detection of oversized models (>80% VRAM)
- Transparent routing to ring buffer for large models
- Fallback to standard memory-aware cache for small models
- Manages WeightStreamer lifecycle for async prefetching
- Bus contention control: DISABLE_KV_SWAP flag ensures weights own 100% of PCIe
"""

import logging
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .memory_aware_model_cache import MemoryAwareModelCache
from ..backend.runtime.ring_buffer import WeightRingBuffer
from ..backend.runtime.weight_streamer import WeightStreamer
from ..backend.runtime.weight_hooks import RingBufferHookManager

logger = logging.getLogger(__name__)

# Bus contention control flag
# When True, weights own 100% of PCIe bus (no concurrent KV swapping)
# Useful for isolating ring buffer performance during experiments
DISABLE_KV_SWAP = os.environ.get('DJINN_DISABLE_KV_SWAP', '0') == '1'

if DISABLE_KV_SWAP:
    logger.info("🚀 DJINN_DISABLE_KV_SWAP=1: KV swapping disabled, weights own 100% of PCIe bus")


@dataclass
class RingBufferConfig:
    """Configuration for ring buffer mode."""
    enabled: bool = False
    capacity_gb: float = 48.0
    prefetch_workers: int = 1
    vram_threshold: float = 0.8  # Use ring buffer if model > 80% VRAM


class RingBufferModelCache:
    """
    Intelligent model cache that switches to ring buffer for oversized models.
    
    Strategy:
    1. Estimate model size
    2. If model > 80% available VRAM: Use ring buffer (streaming mode)
    3. Else: Use standard memory-aware cache (all-at-once)
    
    This provides transparent weight streaming without changing client code.
    """
    
    def __init__(
        self,
        device: str = 'cuda:0',
        max_vram_gb: float = 80.0,
        ring_buffer_config: Optional[RingBufferConfig] = None
    ):
        """
        Initialize ring buffer-aware model cache.
        
        Args:
            device: GPU device string ('cuda:0', etc.)
            max_vram_gb: Maximum VRAM available on this device
            ring_buffer_config: Ring buffer configuration (if None, disabled)
        """
        self.device = torch.device(device)
        self.max_vram_gb = max_vram_gb
        self.ring_buffer_config = ring_buffer_config or RingBufferConfig(enabled=False)
        
        # Standard cache for small models
        self.standard_cache = MemoryAwareModelCache(
            device=device,
            max_memory_gb=max_vram_gb
        )
        
        # Ring buffer for large models (lazy init)
        self.ring_buffer: Optional[WeightRingBuffer] = None
        self.weight_streamer: Optional[WeightStreamer] = None
        self.hook_managers: Dict[str, RingBufferHookManager] = {}
        self.models_using_ring_buffer: set = set()
        
        logger.info(
            f"RingBufferModelCache initialized "
            f"(device={device}, max_vram_gb={max_vram_gb}, "
            f"ring_buffer_enabled={self.ring_buffer_config.enabled}, "
            f"kv_swap_enabled={not DISABLE_KV_SWAP})"
        )
    
    def register_model(
        self,
        fingerprint: str,
        descriptor: Dict,
        weight_ids: Dict[str, str],
        uncached_weights: Dict[str, torch.Tensor],
        architecture_data: Optional[bytes] = None,
        model: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Register a model, choosing between standard and ring buffer storage.
        
        Args:
            fingerprint: Model fingerprint (unique ID)
            descriptor: Model descriptor
            weight_ids: Weight ID mapping
            uncached_weights: Deserialized weight tensors
            architecture_data: Serialized model architecture
            model: Optional pre-loaded PyTorch model
        
        Returns:
            Registration status dict
        """
        
        # Estimate model size
        total_weight_bytes = sum(
            w.numel() * w.element_size() 
            for w in uncached_weights.values()
            if isinstance(w, torch.Tensor)
        )
        total_weight_gb = total_weight_bytes / (1024 ** 3)
        
        # Decide: ring buffer or standard cache?
        should_use_ring_buffer = (
            self.ring_buffer_config.enabled and
            total_weight_gb > (self.max_vram_gb * self.ring_buffer_config.vram_threshold)
        )
        
        logger.info(
            f"Model {fingerprint[:16]}... size: {total_weight_gb:.1f}GB / "
            f"{self.max_vram_gb:.1f}GB VRAM "
            f"→ {'RING BUFFER' if should_use_ring_buffer else 'STANDARD CACHE'}"
        )
        
        if should_use_ring_buffer:
            return self._register_with_ring_buffer(
                fingerprint, descriptor, weight_ids, uncached_weights,
                architecture_data, model, total_weight_gb
            )
        else:
            return self._register_with_standard_cache(
                fingerprint, descriptor, weight_ids, uncached_weights,
                architecture_data, model
            )
    
    def _register_with_ring_buffer(
        self,
        fingerprint: str,
        descriptor: Dict,
        weight_ids: Dict,
        uncached_weights: Dict[str, torch.Tensor],
        architecture_data: Optional[bytes],
        model: Optional[nn.Module],
        model_size_gb: float
    ) -> Dict[str, Any]:
        """Register model using ring buffer for streaming."""
        
        try:
            # Initialize ring buffer if needed
            if self.ring_buffer is None:
                capacity_bytes = int(
                    self.ring_buffer_config.capacity_gb * 1024 ** 3
                )
                self.ring_buffer = WeightRingBuffer(
                    capacity_bytes=capacity_bytes,
                    device=self.device
                )
                logger.info(
                    f"✅ Initialized ring buffer: "
                    f"{self.ring_buffer_config.capacity_gb:.1f}GB on {self.device}"
                )
            
            # Initialize weight streamer if needed
            if self.weight_streamer is None:
                self.weight_streamer = WeightStreamer(
                    ring_buffer=self.ring_buffer,
                    device=self.device,
                    prefetch_queue_size=16
                )
                self.weight_streamer.start()
                logger.info("✅ Started weight streamer (async pipelining)")
            
            # Register model in ring buffer
            logger.info(f"Registering {fingerprint[:16]}... in ring buffer...")
            registration = self.ring_buffer.register_model(
                model_id=fingerprint,
                state_dict=uncached_weights
            )
            logger.info(
                f"✅ Ring buffer registration complete: "
                f"{len(registration.layer_names)} layers, "
                f"skip-end allocation at {registration.skip_end_offset} bytes"
            )
            
            # Install weight hooks on model
            if model is not None:
                logger.info(f"Installing weight hooks for {fingerprint[:16]}...")
                hook_manager = RingBufferHookManager(
                    model=model,
                    ring_buffer=self.ring_buffer,
                    model_id=fingerprint,
                    streamer=self.weight_streamer,
                    layer_names=registration.layer_names
                )
                hook_manager.install_hooks()
                self.hook_managers[fingerprint] = hook_manager
                self.models_using_ring_buffer.add(fingerprint)
                logger.info(f"✅ Weight hooks installed")
            
            return {
                'status': 'success',
                'method': 'ring_buffer',
                'model_size_gb': model_size_gb,
                'buffer_capacity_gb': self.ring_buffer_config.capacity_gb,
                'layers': len(registration.layer_names),
                'fingerprint': fingerprint
            }
        
        except Exception as e:
            logger.error(f"❌ Ring buffer registration failed: {e}")
            logger.info("Falling back to standard cache...")
            return self._register_with_standard_cache(
                fingerprint, descriptor, weight_ids, uncached_weights,
                architecture_data, model
            )
    
    def _register_with_standard_cache(
        self,
        fingerprint: str,
        descriptor: Dict,
        weight_ids: Dict,
        uncached_weights: Dict[str, torch.Tensor],
        architecture_data: Optional[bytes],
        model: Optional[nn.Module]
    ) -> Dict[str, Any]:
        """Register model using standard memory-aware cache."""
        
        logger.info(f"Registering {fingerprint[:16]}... with standard cache...")
        
        result = self.standard_cache.register_model(
            fingerprint=fingerprint,
            descriptor=descriptor,
            weight_ids=weight_ids,
            uncached_weights=uncached_weights,
            architecture_data=architecture_data
        )
        
        result['method'] = 'standard_cache'
        return result
    
    def execute(
        self,
        fingerprint: str,
        inputs: Dict[str, torch.Tensor],
        hints: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Execute model (ring buffer or standard).
        
        For ring buffer models: Weight hooks handle swapping during forward pass.
        For standard models: Standard execution path.
        """
        
        if fingerprint in self.models_using_ring_buffer:
            # Ring buffer model - just call forward (hooks handle weight swapping)
            logger.debug(f"Executing {fingerprint[:16]}... via ring buffer")
            
            # Get model from standard cache (it's stored there after being created)
            model = self.standard_cache.models.get(fingerprint)
            if model is None:
                raise ValueError(f"Model {fingerprint} not found in cache")
            
            with torch.no_grad():
                # Hooks will intercept and swap weights as needed
                output = model(**inputs)
            
            return output
        else:
            # Standard cache model
            logger.debug(f"Executing {fingerprint[:16]}... via standard cache")
            return self.standard_cache.execute(fingerprint, inputs, hints)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'ring_buffer_models': len(self.models_using_ring_buffer),
            'standard_cache_stats': self.standard_cache.get_stats() if hasattr(self.standard_cache, 'get_stats') else {}
        }
        
        if self.ring_buffer:
            stats['ring_buffer_stats'] = {
                'capacity_gb': self.ring_buffer_config.capacity_gb,
                'models_registered': len(self.ring_buffer.registrations),
                'total_layers': sum(
                    len(m.layer_names) for m in self.ring_buffer.registrations.values()
                )
            }
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        if self.weight_streamer:
            self.weight_streamer.stop()
        if self.ring_buffer:
            self.ring_buffer.clear()
        self.standard_cache.clear()

