"""
Model Fingerprinting: Stable, deterministic model identification.

Generates stable fingerprints based on architecture + parameter shapes (not values).
Includes collision detection for security.

Key Design Decisions:
1. Use explicit model ID if provided (for known models)
2. Otherwise compute deterministic hash from architecture + shapes
3. Track fingerprints to detect collisions (security)
4. Thread-safe for concurrent access

This is part of the redesign plan (Week 1).
"""

import hashlib
import json
import logging
import threading
from typing import Dict, Optional, Set, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelFingerprint:
    """
    Generate stable, deterministic model identifiers.
    
    Based on architecture + parameter shapes (not values).
    Includes collision detection for security.
    """
    
    # Class-level tracking for collision detection
    _registered_fingerprints: Dict[str, str] = {}  # fingerprint -> structure_hash
    _fingerprint_lock = threading.Lock()
    
    @staticmethod
    def compute(model: nn.Module, model_id: Optional[str] = None) -> str:
        """
        Generate fingerprint for model identification.
        
        Args:
            model: PyTorch model
            model_id: Optional explicit model identifier
        
        Returns:
            Stable fingerprint string (format: "djinn_v1:hexdigest")
        
        Strategy:
        1. Use explicit ID if model defines one or model_id provided
        2. Otherwise compute from architecture + shapes
        3. Check for collisions (security)
        """
        # Option 1: Model self-identifies or explicit ID provided
        if model_id:
            return f"djinn_explicit:{model_id}"
        
        if hasattr(model, '__djinn_model_id__'):
            return f"djinn_explicit:{model.__djinn_model_id__}"
        
        # Option 2: Compute deterministic fingerprint
        fingerprint_data = {
            'class_name': model.__class__.__module__ + '.' + model.__class__.__name__,
            'architecture': ModelFingerprint._get_architecture_signature(model),
            'parameter_shapes': ModelFingerprint._get_parameter_shapes(model)
        }
        
        # Create stable JSON representation
        json_str = json.dumps(fingerprint_data, sort_keys=True, default=str)
        
        # Generate hash
        hash_value = hashlib.sha256(json_str.encode()).hexdigest()
        fingerprint = f"djinn_v1:{hash_value[:16]}"
        
        # Check for collisions
        structure_hash = hashlib.sha256(json_str.encode()).hexdigest()
        ModelFingerprint._check_collision(fingerprint, structure_hash)
        
        return fingerprint
    
    @staticmethod
    def _get_architecture_signature(model: nn.Module) -> Dict[str, Any]:
        """Extract architecture signature."""
        signature = {
            'modules': [],
            'total_params': sum(p.numel() for p in model.parameters()),
            'total_buffers': sum(b.numel() for b in model.buffers())
        }
        
        # Traverse module hierarchy
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                signature['modules'].append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'config': ModelFingerprint._get_module_config(module)
                })
        
        return signature
    
    @staticmethod
    def _get_module_config(module: nn.Module) -> Dict[str, Any]:
        """Extract module configuration."""
        config: Dict[str, Any] = {}
        
        if isinstance(module, nn.Linear):
            config = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None
            }
        elif isinstance(module, nn.Conv2d):
            config = {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': tuple(module.kernel_size),
                'stride': tuple(module.stride),
                'padding': tuple(module.padding),
                'bias': module.bias is not None
            }
        elif isinstance(module, nn.MultiheadAttention):
            config = {
                'embed_dim': module.embed_dim,
                'num_heads': module.num_heads,
                'dropout': module.dropout
            }
        elif isinstance(module, nn.Embedding):
            config = {
                'num_embeddings': module.num_embeddings,
                'embedding_dim': module.embedding_dim,
                'padding_idx': module.padding_idx
            }
        elif isinstance(module, nn.LayerNorm):
            config = {
                'normalized_shape': tuple(module.normalized_shape),
                'eps': module.eps
            }
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
            config = {'type': module.__class__.__name__}
        elif isinstance(module, nn.Dropout):
            config = {'p': module.p}
        elif isinstance(module, nn.BatchNorm2d):
            config = {
                'num_features': module.num_features,
                'eps': module.eps,
                'momentum': module.momentum
            }
        
        return config
    
    @staticmethod
    def _get_parameter_shapes(model: nn.Module) -> Dict[str, list]:
        """Get shapes of all parameters."""
        shapes = {}
        for name, param in model.named_parameters():
            shapes[name] = list(param.shape)
        return shapes
    
    @staticmethod
    def _check_collision(fingerprint: str, structure_hash: str):
        """
        Check for fingerprint collisions (security issue).
        
        Raises ValueError if collision detected.
        """
        with ModelFingerprint._fingerprint_lock:
            if fingerprint in ModelFingerprint._registered_fingerprints:
                existing_hash = ModelFingerprint._registered_fingerprints[fingerprint]
                if existing_hash != structure_hash:
                    raise ValueError(
                        f"Fingerprint collision detected for {fingerprint}. "
                        "Different model structure with same fingerprint. "
                        "This could indicate a security issue or hash collision."
                    )
            else:
                ModelFingerprint._registered_fingerprints[fingerprint] = structure_hash
                logger.debug(f"Registered fingerprint: {fingerprint}")
    
    @staticmethod
    def clear_registry():
        """Clear fingerprint registry (for testing)."""
        with ModelFingerprint._fingerprint_lock:
            ModelFingerprint._registered_fingerprints.clear()
            logger.debug("Fingerprint registry cleared")
    
    @staticmethod
    def get_registry_stats() -> Dict[str, int]:
        """Get fingerprint registry statistics."""
        with ModelFingerprint._fingerprint_lock:
            return {
                'registered_count': len(ModelFingerprint._registered_fingerprints)
            }

