"""
Profile Manager: API for registering and managing PerformanceProfiles.

Phase 2: Provides CLI and programmatic interface for profile management.
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .performance_profile import PerformanceProfile
from .profile_registry import get_profile_registry
from .analysis_pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)


class ProfileManager:
    """
    High-level API for profile management.
    
    Handles:
    - Profile registration (explicit)
    - Profile discovery and listing
    - Profile versioning
    - Telemetry inspection
    """
    
    def __init__(self):
        """Initialize ProfileManager."""
        self.registry = get_profile_registry()
        self.pipeline = AnalysisPipeline(registry=self.registry)
        logger.info("ProfileManager initialized")
    
    def register_model(self,
                      model: nn.Module,
                      model_fingerprint: str,
                      sample_inputs: Dict[str, torch.Tensor],
                      override: bool = False) -> PerformanceProfile:
        """
        Register a model's PerformanceProfile.
        
        This generates a profile from the model and registers it for use in v2.0.
        
        Args:
            model: PyTorch model
            model_fingerprint: Unique model identifier
            sample_inputs: Representative input tensors
            override: If True, re-generate even if profile exists
        
        Returns:
            PerformanceProfile that was registered
        
        Raises:
            ValueError: If model_fingerprint already registered and override=False
        """
        # Check if already registered
        existing = self.registry.get_profile(model_fingerprint)
        if existing and not override:
            logger.warning(
                f"Profile for {model_fingerprint[:16]}... already exists "
                f"(v{existing.version}). Use override=True to regenerate."
            )
            return existing
        
        # Generate and register profile
        logger.info(f"Registering model {model_fingerprint[:16]}...")
        profile = self.pipeline.analyze_and_register(
            model=model,
            model_fingerprint=model_fingerprint,
            sample_inputs=sample_inputs
        )
        
        logger.info(
            f"✅ Model registered: {model_fingerprint[:16]}... "
            f"(profile {profile.profile_id}, v{profile.version})"
        )
        
        return profile
    
    def get_profile(self, 
                   model_fingerprint: str,
                   input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None) -> Optional[PerformanceProfile]:
        """
        Get profile for a model.
        
        Args:
            model_fingerprint: Model identifier
            input_shapes: Optional input shapes for matching
        
        Returns:
            PerformanceProfile or None
        """
        return self.registry.get_profile(model_fingerprint, input_shapes)
    
    def list_profiles(self, model_fingerprint: Optional[str] = None) -> List[PerformanceProfile]:
        """
        List registered profiles.
        
        Args:
            model_fingerprint: Optional filter by model
        
        Returns:
            List of PerformanceProfile
        """
        return self.registry.list_profiles(model_fingerprint)
    
    def get_profile_stats(self, profile_id: Optional[str] = None) -> Dict:
        """
        Get detailed stats for a profile or all profiles.
        
        Args:
            profile_id: Optional filter by profile ID
        
        Returns:
            Statistics dict
        """
        if profile_id:
            profile = self.registry.get_profile_by_id(profile_id)
            if not profile:
                raise ValueError(f"Profile {profile_id} not found")
            
            return {
                'profile_id': profile.profile_id,
                'model_fingerprint': profile.model_fingerprint,
                'version': profile.version,
                'execution_phase': profile.execution_phase.value,
                'observations': profile.observation_count,
                'observed_latency_ms': profile.observed_latency_ms,
                'observed_memory_gb': profile.observed_memory_gb,
                'created_at': profile.created_at,
                'updated_at': profile.updated_at,
            }
        else:
            # Aggregate stats
            profiles = self.registry.list_profiles()
            return {
                'total_profiles': len(profiles),
                'by_phase': self._count_by_phase(profiles),
                'registry_stats': self.registry.get_stats(),
            }
    
    def get_telemetry(self, 
                     profile_id: Optional[str] = None,
                     limit: int = 100) -> List[Dict]:
        """
        Get telemetry for a profile.
        
        Args:
            profile_id: Profile to query
            limit: Max records to return
        
        Returns:
            List of telemetry records
        """
        return self.registry.get_telemetry(profile_id, limit)
    
    def export_profiles(self, filepath: str) -> None:
        """
        Export all profiles to JSON file.
        
        Args:
            filepath: Path to write profiles
        """
        import json
        
        profiles = self.registry.list_profiles()
        data = {
            'version': 1,
            'profiles': [p.to_dict() for p in profiles],
            'metadata': {
                'count': len(profiles),
                'registry_stats': self.registry.get_stats(),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(profiles)} profiles to {filepath}")
    
    def import_profiles(self, filepath: str, override: bool = False) -> int:
        """
        Import profiles from JSON file.
        
        Args:
            filepath: Path to read profiles
            override: If True, replace existing profiles
        
        Returns:
            Number of profiles imported
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        count = 0
        for profile_dict in data.get('profiles', []):
            profile = PerformanceProfile.from_dict(profile_dict)
            
            # Check for existing profile
            existing = self.registry.get_profile_by_id(profile.profile_id)
            if existing and not override:
                logger.debug(f"Skipping existing profile {profile.profile_id}")
                continue
            
            self.registry.register_profile(profile)
            count += 1
        
        logger.info(f"Imported {count} profiles from {filepath}")
        return count
    
    def _count_by_phase(self, profiles: List[PerformanceProfile]) -> Dict[str, int]:
        """Count profiles by execution phase."""
        counts = {}
        for profile in profiles:
            phase = profile.execution_phase.value
            counts[phase] = counts.get(phase, 0) + 1
        return counts


# Global singleton
_profile_manager: Optional[ProfileManager] = None
_manager_lock = __import__('threading').Lock()


def get_profile_manager() -> ProfileManager:
    """Get global ProfileManager instance."""
    global _profile_manager
    if _profile_manager is None:
        with _manager_lock:
            if _profile_manager is None:
                _profile_manager = ProfileManager()
    return _profile_manager

