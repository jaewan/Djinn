import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ProfileRegistryClient:
    """
    Client for the ProfileRegistry.
    
    Phase 0: No-op placeholder
    Phase 1: Connects to in-process ProfileRegistry
    Phase 2: Connects to remote ProfileRegistry service
    """

    def __init__(self, endpoint: Optional[str] = None, use_local_registry: bool = True):
        """
        Initialize ProfileRegistryClient.
        
        Args:
            endpoint: Registry endpoint URL (for remote registry)
            use_local_registry: If True, use in-process registry (Phase 1)
        """
        self.endpoint = endpoint or os.environ.get("DJINN_PROFILE_REGISTRY_URL")
        self.use_local_registry = use_local_registry
        
        # Phase 1: Use local in-process registry
        if self.use_local_registry:
            from .profile_registry import get_profile_registry
            self._registry = get_profile_registry()
            logger.info("ProfileRegistryClient initialized (local in-process registry)")
        else:
            self._registry = None
            logger.info(f"ProfileRegistryClient initialized (remote endpoint={self.endpoint})")

    def get_profile(self, 
                   model_fingerprint: str,
                   input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
                   profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get profile for model and input shapes.
        
        Args:
            model_fingerprint: Model fingerprint
            input_shapes: Input shapes dict
            profile_id: Optional explicit profile ID
        
        Returns:
            Profile dict if found, None otherwise
        """
        # Phase 1: Query local registry
        if self.use_local_registry and self._registry:
            # If explicit profile_id provided, use that
            if profile_id:
                profile = self._registry.get_profile_by_id(profile_id)
            else:
                # Otherwise query by fingerprint + shapes
                profile = self._registry.get_profile(model_fingerprint, input_shapes)
            
            if profile:
                logger.debug(
                    f"Profile found: {profile.profile_id} "
                    f"(phase={profile.execution_phase.value}, v{profile.version})"
                )
                return profile.to_dict()
            else:
                logger.debug(f"No profile found for {model_fingerprint}")
                return None
        
        # Phase 0: No-op fallback
        if profile_id:
            logger.debug(
                f"ProfileRegistryClient.get_profile called (no local registry). "
                f"Ignoring profile_id={profile_id}"
            )
        return None

    def record_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """
        Record telemetry from execution.
        
        Args:
            telemetry: Telemetry dict with profile_id, fingerprint, metrics
        """
        # Phase 1: Record to local registry
        if self.use_local_registry and self._registry:
            profile_id = telemetry.get('profile_id')
            model_fingerprint = telemetry.get('fingerprint')
            
            if model_fingerprint:
                self._registry.record_telemetry(
                    profile_id=profile_id,
                    model_fingerprint=model_fingerprint,
                    telemetry=telemetry
                )
                logger.debug(
                    f"Telemetry recorded: profile={profile_id}, "
                    f"model={model_fingerprint[:16]}..."
                )
            else:
                logger.warning("Telemetry missing model fingerprint, skipping")
            return
        
        # Phase 0: No-op fallback
        logger.debug(
            f"ProfileRegistryClient.record_telemetry called (no local registry). "
            f"Telemetry keys: {list(telemetry.keys())}"
        )

