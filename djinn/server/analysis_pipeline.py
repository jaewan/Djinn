"""
Analysis Pipeline: Offline service for generating PerformanceProfiles.

Phase 1 Implementation:
- Simple rule-based analysis (Phase 2 will add ML-based predictions)
- Generates profiles from model characteristics
- Validates and publishes to ProfileRegistry
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .performance_profile import (
    PerformanceProfile,
    ExecutionPhase,
    InputEnvelope,
    ResourceBudget,
    OptimizationDirective,
)
from .profile_registry import ProfileRegistry, get_profile_registry

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Offline pipeline for generating PerformanceProfiles.
    
    Phase 1: Rule-based analysis
    Phase 2: Add ML-based predictions
    Phase 3: Add cost modeling and simulation
    """
    
    def __init__(self, registry: Optional[ProfileRegistry] = None):
        """
        Initialize analysis pipeline.
        
        Args:
            registry: ProfileRegistry instance (uses global if None)
        """
        self.registry = registry or get_profile_registry()
        logger.info("AnalysisPipeline initialized")
    
    def analyze_and_register(self,
                            model: nn.Module,
                            model_fingerprint: str,
                            sample_inputs: Dict[str, torch.Tensor]) -> PerformanceProfile:
        """
        Analyze model and register profile.
        
        Args:
            model: PyTorch model
            model_fingerprint: Model fingerprint
            sample_inputs: Sample inputs for shape analysis
        
        Returns:
            Generated PerformanceProfile
        """
        logger.info(f"Analyzing model {model_fingerprint[:16]}...")
        
        # Stage 1: Capture - extract model characteristics
        characteristics = self._capture_stage(model, sample_inputs)
        
        # Stage 2: Analysis - determine execution phase and budgets
        execution_phase, resource_budget = self._analysis_stage(model, characteristics)
        
        # Stage 3: Optimization - determine directives
        optimization_directives = self._optimization_stage(model, characteristics)
        
        # Stage 4: Build profile
        profile = self._build_profile(
            model_fingerprint=model_fingerprint,
            sample_inputs=sample_inputs,
            execution_phase=execution_phase,
            resource_budget=resource_budget,
            optimization_directives=optimization_directives,
        )
        
        # Stage 5: Validation
        is_valid, errors = profile.validate()
        if not is_valid:
            raise ValueError(f"Invalid profile generated: {errors}")
        
        # Stage 6: Publish to registry
        success = self.registry.register_profile(profile)
        if not success:
            raise RuntimeError(f"Failed to register profile {profile.profile_id}")
        
        logger.info(
            f"✅ Profile {profile.profile_id} registered "
            f"(phase={execution_phase.value}, v{profile.version})"
        )
        
        return profile
    
    def _capture_stage(self, 
                      model: nn.Module,
                      sample_inputs: Dict[str, torch.Tensor]) -> Dict:
        """
        Capture stage: Extract model characteristics.
        
        Returns:
            Dict of characteristics
        """
        characteristics = {}
        
        # Model size
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        characteristics['param_count'] = param_count
        characteristics['param_size_mb'] = param_size_mb
        
        # Input shapes
        characteristics['input_shapes'] = {
            k: tuple(v.shape) for k, v in sample_inputs.items()
        }
        
        # Model type detection (simple heuristics)
        model_name = model.__class__.__name__.lower()
        model_module = model.__class__.__module__.lower() if hasattr(model.__class__, '__module__') else ''
        
        # Check for attention layers (indicates transformer/LLM)
        has_attention = any('attention' in name.lower() for name, _ in model.named_modules())
        characteristics['has_attention'] = has_attention
        
        # Detect LLM models
        if ('gpt' in model_name or 'llama' in model_name or 'transformer' in model_name or
            'gpt' in model_module or has_attention):
            characteristics['model_type'] = 'llm'
        # Detect vision models
        elif 'resnet' in model_name or 'vit' in model_name or 'conv' in model_name:
            characteristics['model_type'] = 'vision'
        else:
            characteristics['model_type'] = 'unknown'
        
        logger.debug(f"Captured characteristics: {characteristics}")
        return characteristics
    
    def _analysis_stage(self,
                       model: nn.Module,
                       characteristics: Dict) -> Tuple[ExecutionPhase, ResourceBudget]:
        """
        Analysis stage: Determine execution phase and resource budgets.
        
        Returns:
            (ExecutionPhase, ResourceBudget)
        """
        # Phase detection based on model type and input shapes
        model_type = characteristics.get('model_type', 'unknown')
        input_shapes = characteristics.get('input_shapes', {})
        
        # Detect execution phase
        if model_type == 'llm':
            # For LLMs, check sequence length to determine prefill vs decode
            # Prefill: longer sequences, parallel processing
            # Decode: short sequences (1 token), sequential
            seq_length = 1
            for shape in input_shapes.values():
                if len(shape) >= 2:
                    seq_length = max(seq_length, shape[1])
            
            if seq_length > 1:
                execution_phase = ExecutionPhase.LLM_PREFILL
                # Prefill: More activations, less KV cache
                resource_budget = ResourceBudget(
                    activations_fraction=0.6,
                    weights_fraction=0.2,
                    kv_cache_fraction=0.2,
                )
            else:
                execution_phase = ExecutionPhase.LLM_DECODE
                # Decode: Less activations, more KV cache
                resource_budget = ResourceBudget(
                    activations_fraction=0.2,
                    weights_fraction=0.2,
                    kv_cache_fraction=0.6,
                )
        elif model_type == 'vision':
            execution_phase = ExecutionPhase.VISION_ENCODING
            # Vision: More activations (feature maps), less KV cache
            resource_budget = ResourceBudget(
                activations_fraction=0.5,
                weights_fraction=0.3,
                kv_cache_fraction=0.2,
            )
        else:
            execution_phase = ExecutionPhase.UNKNOWN
            # Default balanced budget
            resource_budget = ResourceBudget(
                activations_fraction=0.4,
                weights_fraction=0.3,
                kv_cache_fraction=0.3,
            )
        
        logger.debug(
            f"Analysis: phase={execution_phase.value}, "
            f"budget=({resource_budget.activations_fraction:.1%}, "
            f"{resource_budget.weights_fraction:.1%}, "
            f"{resource_budget.kv_cache_fraction:.1%})"
        )
        
        return execution_phase, resource_budget
    
    def _optimization_stage(self,
                           model: nn.Module,
                           characteristics: Dict) -> List[OptimizationDirective]:
        """
        Optimization stage: Determine optimization directives.
        
        Returns:
            List of OptimizationDirective
        """
        directives = []
        
        # Enable fusion for models with many small ops
        if characteristics.get('has_attention', False):
            directives.append(OptimizationDirective.ENABLE_FUSION)
        
        # Enable compilation for larger models
        if characteristics.get('param_size_mb', 0) > 100:
            directives.append(OptimizationDirective.ENABLE_COMPILATION)
        
        logger.debug(f"Optimization directives: {[d.value for d in directives]}")
        return directives
    
    def _build_profile(self,
                      model_fingerprint: str,
                      sample_inputs: Dict[str, torch.Tensor],
                      execution_phase: ExecutionPhase,
                      resource_budget: ResourceBudget,
                      optimization_directives: List[OptimizationDirective]) -> PerformanceProfile:
        """
        Build PerformanceProfile from analysis results.
        
        Returns:
            PerformanceProfile
        """
        # Extract input shapes
        input_shapes = {k: tuple(v.shape) for k, v in sample_inputs.items()}
        
        # Generate profile ID
        profile_id = PerformanceProfile.generate_profile_id(
            model_fingerprint=model_fingerprint,
            input_shapes=input_shapes
        )
        
        # Create input envelope (exact match for now, Phase 2 will add ranges)
        input_envelope = InputEnvelope(
            shape_constraints={
                k: (shape, shape)  # (min, max) - exact match
                for k, shape in input_shapes.items()
            },
            dtype_constraints={
                k: str(v.dtype) for k, v in sample_inputs.items()
            }
        )
        
        # Build profile
        profile = PerformanceProfile(
            profile_id=profile_id,
            model_fingerprint=model_fingerprint,
            version=1,
            input_envelope=input_envelope,
            execution_phase=execution_phase,
            resource_budget=resource_budget,
            optimization_directives=optimization_directives,
        )
        
        return profile

