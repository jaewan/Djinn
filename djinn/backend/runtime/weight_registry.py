"""
Weight Name Registry for Unified Weight Streaming

Provides bidirectional mapping between module names and parameter names,
with automatic detection and handling of tied weights (e.g., lm_head sharing
weights with embedding layers in GPT-2, Llama, etc.).

This fixes the fundamental design flaw where:
- Hooks operate at MODULE level (e.g., "lm_head")
- Ring buffer operates at PARAMETER level (e.g., "transformer.wte.weight")
- No consistent naming led to "Layer not found" errors for tied weights
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightNameRegistry:
    """
    Manages unified weight naming and tied weight resolution.
    
    Provides:
    1. Module name -> parameter names mapping
    2. Parameter name -> module names mapping (handles ties)
    3. Tied weight detection and resolution
    4. Canonical weight identification
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize registry from model structure.
        
        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        
        # Module name -> list of parameter names
        self.module_to_params: Dict[str, List[str]] = {}
        
        # Parameter name -> list of module names (list because of tied weights)
        self.param_to_modules: Dict[str, List[str]] = {}
        
        # Tied weights mapping: alias -> canonical name
        self.tied_weights: Dict[str, str] = {}
        
        # Canonical parameters (remove duplicates from ties)
        self.canonical_params: Set[str] = set()
        
        # Build mappings
        self._build_mappings()
    
    def _build_mappings(self):
        """Build all internal mappings from model."""
        # First pass: map modules to parameters
        for module_name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Store this module as having a weight parameter
                param_name = f"{module_name}.weight" if module_name else "weight"
                self.module_to_params[module_name] = [param_name]
                
                # Track parameter -> module mapping
                if param_name not in self.param_to_modules:
                    self.param_to_modules[param_name] = []
                self.param_to_modules[param_name].append(module_name)
                
                # Also handle bias if present
                if hasattr(module, 'bias') and module.bias is not None:
                    bias_name = f"{module_name}.bias" if module_name else "bias"
                    if param_name not in self.module_to_params[module_name]:
                        self.module_to_params[module_name].append(bias_name)
        
        # Second pass: detect tied weights
        self._detect_tied_weights()
    
    def _detect_tied_weights(self):
        """
        Detect tied weights by comparing tensor data pointers.
        
        When two parameters share the same underlying data (e.g., lm_head.weight
        and transformer.wte.weight in GPT-2), identify which is canonical.
        """
        param_ptr_to_name: Dict[int, str] = {}
        
        for name, param in self.model.named_parameters():
            if param is None:
                continue
            
            # Skip meta tensors (data_ptr() returns 0 for all meta tensors)
            if param.device.type == 'meta':
                self.canonical_params.add(name)
                continue
            
            ptr = param.data_ptr()
            
            if ptr in param_ptr_to_name:
                # This parameter shares data with another
                canonical = param_ptr_to_name[ptr]
                
                # Mark this as an alias of the canonical parameter
                self.tied_weights[name] = canonical
                logger.debug(f"Detected tied weight: {name} -> {canonical}")
            else:
                # This is a canonical parameter
                param_ptr_to_name[ptr] = name
                self.canonical_params.add(name)
    
    def get_canonical_param(self, param_name: str) -> str:
        """
        Get canonical parameter name for a parameter (handles aliases).
        
        Args:
            param_name: Parameter name (may be an alias)
            
        Returns:
            Canonical parameter name
        """
        return self.tied_weights.get(param_name, param_name)
    
    def get_canonical_params(self) -> Set[str]:
        """Get all canonical (non-tied) parameter names."""
        return self.canonical_params.copy()
    
    def get_aliases(self, canonical_param: str) -> List[str]:
        """
        Get all aliases for a canonical parameter.
        
        Args:
            canonical_param: Canonical parameter name
            
        Returns:
            List of alias names
        """
        aliases = []
        for tied_param, canonical in self.tied_weights.items():
            if canonical == canonical_param:
                aliases.append(tied_param)
        return aliases
    
    def module_to_param(self, module_name: str, param_type: str = "weight") -> Optional[str]:
        """
        Get parameter name for a module.
        
        Args:
            module_name: Module name
            param_type: "weight" or "bias"
            
        Returns:
            Full parameter name or None
        """
        if module_name not in self.module_to_params:
            return None
        
        for param_name in self.module_to_params[module_name]:
            if param_name.endswith(param_type):
                return param_name
        
        return None
    
    def param_to_modules(self, param_name: str) -> List[str]:
        """
        Get all modules that use a parameter (may be multiple due to ties).
        
        Args:
            param_name: Parameter name
            
        Returns:
            List of module names
        """
        canonical = self.get_canonical_param(param_name)
        return self.param_to_modules.get(canonical, [])
    
    def log_summary(self):
        """Log summary of registry findings."""
        logger.info(f"Weight Registry Summary:")
        logger.info(f"  Modules with weights: {len(self.module_to_params)}")
        logger.info(f"  Canonical parameters: {len(self.canonical_params)}")
        logger.info(f"  Tied weights detected: {len(self.tied_weights)}")
        
        if self.tied_weights:
            logger.info("  Tied weight mappings:")
            for alias, canonical in self.tied_weights.items():
                logger.info(f"    {alias} -> {canonical}")


class WeightRegistryError(Exception):
    """Exception raised when weight registry operations fail."""
    pass

