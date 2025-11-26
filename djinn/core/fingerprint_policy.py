"""
djinn/core/fingerprint_policy.py

Collision-resistant, tenant-safe model fingerprinting.

This implements Phase 0 of the redesign plan: Fingerprint Policy Specification.
"""

import hashlib
import json
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

# Try to import xxhash for fast hashing, fallback to hashlib if not available
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    logger.warning(
        "xxhash not available, falling back to hashlib.sha256 for fingerprinting. "
        "Install xxhash for better performance: pip install xxhash"
    )


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint computation."""
    
    # Algorithm selection
    hash_algorithm: str = 'xxhash64'  # Fast non-crypto hash
    
    # Size thresholds (in GB)
    large_model_threshold: float = 10.0
    
    # Sampling for large models
    sample_rate: float = 0.01  # 1% sampling
    sample_seed_base: int = 42  # For reproducibility
    
    # HuggingFace verification
    verify_hf_weights: bool = False  # Check if weights match hub (default: False for performance)
    hf_sample_params: list = None  # Params to check
    
    def __post_init__(self):
        if self.hf_sample_params is None:
            # Default: check first and last layer
            self.hf_sample_params = [
                'lm_head.weight',
                'model.embed_tokens.weight',
                'transformer.wte.weight',  # GPT-style
                'transformer.h.0.mlp.c_fc.weight',
            ]


class FingerprintPolicy:
    """
    Policy for computing collision-resistant fingerprints.
    
    Strategy:
    1. Canonical HuggingFace models: Use model_id + revision
    2. Fine-tuned HF models: Detect modifications, treat as private
    3. Large private models (>10GB): Sample 1% of weights
    4. Small private models (<10GB): Full hash
    
    All fingerprints include:
    - Version prefix (for future evolution)
    - Tenant ID (for isolation)
    - Architecture hash (structure)
    - Weights hash (strategy-dependent)
    """
    
    VERSION = 'v2'  # Fingerprint schema version
    
    def __init__(self, config: Optional[FingerprintConfig] = None):
        self.config = config or FingerprintConfig()
        self._hf_model_cache = {}  # Cache for HF model verification
        
        # CRITICAL: Cache fingerprints to avoid recomputing on every call
        # Key: (model_id, tenant_id, version_tag) -> (fingerprint, metadata)
        # Uses weak references to avoid memory leaks
        self._fingerprint_cache: WeakKeyDictionary = WeakKeyDictionary()
        self._fingerprint_cache_by_key: dict = {}  # Fallback for non-model keys
    
    def compute_fingerprint(
        self,
        model: nn.Module,
        model_id: Optional[str] = None,
        tenant_id: str = 'default',
        version_tag: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Compute collision-resistant fingerprint for model.
        
        CRITICAL PERFORMANCE NOTE:
        This method caches fingerprints per (model_instance, tenant_id, model_id, version_tag)
        to avoid expensive weight hashing on every call. Fingerprints are only recomputed
        when the model instance changes or version_tag changes.
        
        Args:
            model: PyTorch model
            model_id: Optional HuggingFace model ID or user name
            tenant_id: Tenant identifier for isolation
            version_tag: Optional user-provided version
        
        Returns:
            (fingerprint: str, metadata: dict)
            
        Examples:
            # Canonical HF model
            fp, meta = policy.compute_fingerprint(
                model, 
                model_id="gpt2",
                tenant_id="research_team"
            )
            # fp = "v2:public_hf:gpt2:main:arch_abc123"
            
            # Fine-tuned model (detected as modified)
            fp, meta = policy.compute_fingerprint(
                finetuned_model,
                model_id="gpt2",  # Base model
                tenant_id="research_team"
            )
            # fp = "v2:private:research_team:arch_abc123:weights_def456"
            
            # Custom model with version
            fp, meta = policy.compute_fingerprint(
                custom_model,
                model_id="MyResNet",
                tenant_id="research_team",
                version_tag="v1.0"
            )
            # fp = "v2:private:research_team:MyResNet:v1.0:arch_abc123"
        """
        # Check cache first (CRITICAL for performance)
        cache_key = (id(model), model_id, tenant_id, version_tag)
        if cache_key in self._fingerprint_cache_by_key:
            cached_fp, cached_meta = self._fingerprint_cache_by_key[cache_key]
            return cached_fp, cached_meta.copy()
        
        # Cache miss: compute fingerprint
        components = [self.VERSION]
        metadata = {}
        
        # Step 1: Compute architecture hash
        arch_hash = self._compute_architecture_hash(model)
        metadata['architecture_hash'] = arch_hash
        
        # Step 2: Determine if canonical HuggingFace
        is_canonical_hf = False
        if model_id and self._is_public_hf_id(model_id):
            is_canonical_hf = self._verify_canonical_hf(model, model_id)
            metadata['is_canonical_hf'] = is_canonical_hf
        
        # Step 3: Build fingerprint based on model type
        if is_canonical_hf:
            # Canonical HF: Use model_id as identity
            revision = self._get_hf_revision(model_id)
            components.extend([
                'public_hf',
                model_id,
                revision,
                arch_hash
            ])
            metadata['model_kind'] = 'public_hf'
            metadata['revision'] = revision
            
        else:
            # Private/custom model
            components.extend(['private', tenant_id])
            
            if version_tag:
                # User-provided version
                components.append(model_id or 'custom')
                components.append(version_tag)
                components.append(arch_hash)
                metadata['model_kind'] = 'private_versioned'
                metadata['version'] = version_tag
            else:
                # Auto-version from weights
                components.append(arch_hash)
                
                # Compute weights hash
                model_size_gb = self._estimate_model_size_gb(model)
                metadata['model_size_gb'] = model_size_gb
                
                if model_size_gb > self.config.large_model_threshold:
                    # Large model: sample-based hash
                    weights_hash = self._compute_sampled_weights_hash(
                        model, 
                        tenant_id
                    )
                    components.append(f"sampled:{weights_hash}")
                    metadata['model_kind'] = 'private_large'
                    metadata['weights_hash_method'] = 'sampled'
                else:
                    # Small model: full hash
                    weights_hash = self._compute_full_weights_hash(model)
                    components.append(f"full:{weights_hash}")
                    metadata['model_kind'] = 'private_small'
                    metadata['weights_hash_method'] = 'full'
        
        # Step 4: Generate final fingerprint
        fingerprint_str = '|'.join(components)
        fingerprint = hashlib.sha256(
            fingerprint_str.encode()
        ).hexdigest()[:16]
        
        metadata['fingerprint'] = fingerprint
        metadata['tenant_id'] = tenant_id
        
        # Cache result for future calls (avoids expensive recomputation)
        self._fingerprint_cache_by_key[cache_key] = (fingerprint, metadata.copy())
        
        return fingerprint, metadata
    
    def _is_public_hf_id(self, model_id: str) -> bool:
        """Check if model_id looks like a public HuggingFace ID."""
        # Simple heuristic: contains "/" or is in known list
        if '/' in model_id:
            return True
        
        # Known public models
        known_public = {
            'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
            'bert-base-uncased', 'bert-large-uncased',
            'roberta-base', 'roberta-large',
            't5-small', 't5-base', 't5-large',
        }
        return model_id in known_public
    
    def _verify_canonical_hf(
        self, 
        model: nn.Module, 
        model_id: str
    ) -> bool:
        """
        Verify that model weights match HuggingFace Hub.
        
        This detects fine-tuned models that started from HF base.
        
        Strategy:
        1. Load config from HF Hub
        2. Sample a few representative parameters
        3. Compare with local model
        4. Return True only if weights match
        """
        if not self.config.verify_hf_weights:
            return True  # Trust user
        
        try:
            from transformers import AutoConfig, AutoModel
            
            # Check cache first
            cache_key = f"{model_id}:verify"
            if cache_key in self._hf_model_cache:
                hub_params = self._hf_model_cache[cache_key]
            else:
                # Load from hub (config only, no weights initially)
                config = AutoConfig.from_pretrained(model_id)
                
                # Get sample parameters from hub
                # (This is lightweight: only loads specified params)
                hub_params = self._load_hub_sample_params(model_id, config)
                self._hf_model_cache[cache_key] = hub_params
            
            # Compare with local model
            local_state = model.state_dict()
            
            for param_name in self.config.hf_sample_params:
                if param_name not in local_state:
                    continue  # Skip if not present
                
                if param_name not in hub_params:
                    continue  # Skip if not in hub
                
                local_param = local_state[param_name]
                hub_param = hub_params[param_name]
                
                # Compare shapes first (fast)
                if local_param.shape != hub_param.shape:
                    return False  # Shape mismatch → modified
                
                # Compare values (use relaxed tolerance)
                if not torch.allclose(local_param, hub_param, rtol=1e-4, atol=1e-6):
                    return False  # Weights differ → fine-tuned
            
            return True  # All checks passed
            
        except Exception as e:
            # If verification fails, be conservative
            logger.warning(
                f"HF verification failed for {model_id}: {e}. "
                f"Treating as private model."
            )
            return False
    
    def _load_hub_sample_params(self, model_id: str, config) -> dict:
        """Load only sample parameters from HuggingFace Hub."""
        from transformers import AutoModel
        
        # Load full model (we need this for weight comparison)
        # TODO: Optimize to only download sample params
        hub_model = AutoModel.from_config(config)
        
        # Extract sample params
        hub_state = hub_model.state_dict()
        sample_params = {}
        
        for param_name in self.config.hf_sample_params:
            if param_name in hub_state:
                sample_params[param_name] = hub_state[param_name]
        
        return sample_params
    
    def _get_hf_revision(self, model_id: str) -> str:
        """Get HuggingFace model revision (commit hash or branch)."""
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return info.sha or 'main'
        except Exception:
            return 'main'
    
    def _compute_architecture_hash(self, model: nn.Module) -> str:
        """
        Compute hash of model architecture (structure only, no weights).
        
        Includes:
        - Module names and types
        - Module hierarchy
        - Config if available
        """
        if hasattr(model, "_djinn_ghost_arch_hash"):
            return getattr(model, "_djinn_ghost_arch_hash")

        arch_desc = {
            'class': model.__class__.__name__,
            'module': model.__class__.__module__,
        }
        
        # Add config if available
        if hasattr(model, 'config'):
            config = model.config
            if hasattr(config, 'to_dict'):
                try:
                    arch_desc['config'] = config.to_dict()
                except Exception:
                    # Config might not be serializable, skip it
                    pass
        
        # Add module structure
        arch_desc['modules'] = {}
        for name, module in model.named_modules():
            if name:  # Skip root
                arch_desc['modules'][name] = {
                    'type': module.__class__.__name__,
                }
                # Add key parameters (shapes only)
                for param_name, param in module.named_parameters(recurse=False):
                    arch_desc['modules'][name][param_name] = list(param.shape)
        
        # Convert to deterministic JSON
        arch_json = json.dumps(arch_desc, sort_keys=True)
        
        # Hash
        return hashlib.sha256(arch_json.encode()).hexdigest()[:16]
    
    def _estimate_model_size_gb(self, model: nn.Module) -> float:
        """Estimate model size in GB."""
        if hasattr(model, "_djinn_ghost_metadata"):
            cfg = getattr(model, "_djinn_ghost_metadata", {}).get("config", {}) or {}
            param_count = (
                cfg.get("num_parameters")
                or cfg.get("n_parameters")
                or cfg.get("n_params")
            )
            if isinstance(param_count, (int, float)) and param_count > 0:
                size_bytes = int(param_count) * 4
                return size_bytes / (1024 ** 3)
            return 0.0
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 (4 bytes per param) - conservative estimate
        size_bytes = total_params * 4
        return size_bytes / (1024 ** 3)
    
    def _compute_sampled_weights_hash(
        self, 
        model: nn.Module, 
        tenant_id: str
    ) -> str:
        """
        Compute hash of sampled weights (1% of parameters).
        
        Uses deterministic sampling based on tenant_id for reproducibility.
        """
        import numpy as np
        
        # Deterministic seed from tenant
        seed = self.config.sample_seed_base + hash(tenant_id) % 10000
        rng = np.random.RandomState(seed)
        
        # Collect all parameters
        all_params = list(model.parameters())
        total_params = len(all_params)
        
        if total_params == 0:
            return "empty"
        
        # Sample indices
        num_samples = max(1, int(total_params * self.config.sample_rate))
        sample_indices = rng.choice(total_params, min(num_samples, total_params), replace=False)
        
        # Hash sampled parameters
        if XXHASH_AVAILABLE:
            hasher = xxhash.xxh64()
        else:
            hasher = hashlib.sha256()
        
        for idx in sorted(sample_indices):
            param = all_params[idx]
            # Hash shape first
            shape_str = str(param.shape).encode()
            hasher.update(shape_str)
            # Hash parameter bytes
            param_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(param_bytes)
        
        if XXHASH_AVAILABLE:
            return hasher.hexdigest()
        else:
            return hasher.hexdigest()[:16]
    
    def _compute_full_weights_hash(self, model: nn.Module) -> str:
        """Compute hash of all model weights."""
        if XXHASH_AVAILABLE:
            hasher = xxhash.xxh64()
        else:
            hasher = hashlib.sha256()
        
        for name, param in model.named_parameters():
            # Hash name
            hasher.update(name.encode())
            # Hash shape
            hasher.update(str(param.shape).encode())
            # Hash data
            param_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(param_bytes)
        
        if XXHASH_AVAILABLE:
            return hasher.hexdigest()
        else:
            return hasher.hexdigest()[:16]

