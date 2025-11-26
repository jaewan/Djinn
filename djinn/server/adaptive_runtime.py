"""
Adaptive Runtime: v2.0 execution logic that uses PerformanceProfiles.

Phase 2: Executes models according to profiles instead of generic logic.
Can fallback to v1.0 logic gracefully if profile unavailable.
"""

import logging
import time
from typing import Dict, Optional, Any, Callable

from .performance_profile import PerformanceProfile, ExecutionPhase
from .profile_registry import get_profile_registry
from .profile_registry_client import ProfileRegistryClient

logger = logging.getLogger(__name__)


class AdaptiveRuntime:
    """
    v2.0 runtime that adapts execution to PerformanceProfile directives.
    
    Responsibilities:
    1. Query ProfileRegistry for model's profile
    2. Apply profile's resource budget directives
    3. Apply optimization directives (fusion, compilation, etc.)
    4. Track execution metrics against profile predictions
    5. Fallback gracefully if profile unavailable
    """
    
    def __init__(self, 
                 profile_registry_client: Optional[ProfileRegistryClient] = None,
                 enable_profile_application: bool = True):
        """
        Initialize AdaptiveRuntime.
        
        Args:
            profile_registry_client: Client for profile queries
            enable_profile_application: If True, actually apply profile directives
        """
        self.profile_registry_client = (
            profile_registry_client or ProfileRegistryClient(use_local_registry=True)
        )
        self.enable_profile_application = enable_profile_application
        
        # Statistics
        self.stats = {
            'profile_hits': 0,
            'profile_misses': 0,
            'profile_applications': 0,
            'profile_errors': 0,
        }
        
        logger.info(
            f"AdaptiveRuntime initialized "
            f"(profile_application={enable_profile_application})"
        )
    
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Execute request with adaptive runtime using profiles.
        
        This is the main entry point for v2.0 execution path.
        
        Args:
            request: Request dict with fingerprint, inputs, profile_id, etc.
        
        Returns:
            Response dict with result or error
        """
        import asyncio
        from .resilient_model_handler import ResilientModelHandler
        
        try:
            fingerprint = request.get('fingerprint')
            profile_id = request.get('profile_id')
            hints = request.get('hints', {})
            
            logger.info(f"📋 AdaptiveRuntime handling {fingerprint[:16]}... (profile_id={profile_id})")
            
            # Step 1: Get profile from registry
            input_shapes = {}  # Can be extracted from request if needed
            profile = self.profile_registry_client.get_profile(
                fingerprint,
                input_shapes,
                profile_id
            )
            
            if profile:
                self.stats['profile_hits'] += 1
                logger.info(f"✅ Profile found: {profile.get('profile_id', 'unknown')} (phase={profile.get('execution_phase', 'unknown')})")
            else:
                self.stats['profile_misses'] += 1
                logger.debug(f"⚠️  No profile for {fingerprint[:16]}..., using v1.0")
            
            # Step 2: Execute via v1.0 ResilientModelHandler (will use profile hints if available)
            handler = ResilientModelHandler(gpu_id=0)
            
            # Add profile hints to request for handler
            if profile:
                request['_profile'] = profile
                request['_profile_applied'] = True
                self.stats['profile_applications'] += 1
            
            # Execute the request
            response = await handler.handle_request(request)
            
            # Add execution path tracking
            if profile:
                response['_execution_path'] = 'v2_profile'
            else:
                response['_execution_path'] = 'v1_fallback'
            
            return response
            
        except Exception as e:
            logger.error(f"AdaptiveRuntime.execute_request failed: {e}", exc_info=True)
            self.stats['profile_errors'] += 1
            return {
                'status': 'error',
                'message': f'AdaptiveRuntime error: {str(e)}',
                '_execution_path': 'v1_fallback_after_error'
            }
    
    def execute_with_profile(self,
                            model_fingerprint: str,
                            input_shapes: Dict[str, tuple],
                            execute_v1_fallback: Callable[..., Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute model using profile, with v1.0 fallback.
        
        Args:
            model_fingerprint: Model fingerprint
            input_shapes: Input shapes dict
            execute_v1_fallback: Callable that executes v1.0 path
        
        Returns:
            Execution result dict with profile information
        """
        profile = None
        execution_path = 'v1_fallback'
        
        try:
            # Step 1: Query profile registry
            profile = self._get_profile(model_fingerprint, input_shapes)
            
            if profile and self.enable_profile_application:
                # Step 2: Apply profile directives
                start_time = time.perf_counter()
                hints = self._build_profile_hints(profile)
                result = self._execute_with_directives(profile, execute_v1_fallback, hints)
                exec_time = (time.perf_counter() - start_time) * 1000
                
                self.stats['profile_applications'] += 1
                execution_path = 'v2_profile'
                
                logger.info(
                    f"✅ Applied profile {profile.profile_id} "
                    f"(phase={profile.execution_phase.value}, {exec_time:.1f}ms)"
                )
                return result
            else:
                # No profile or not enabled - fall back to v1.0
                if profile:
                    logger.debug(
                        f"Profile {profile.profile_id} found but "
                        f"profile_application disabled, using v1.0"
                    )
                    self.stats['profile_hits'] += 1
                else:
                    logger.debug(f"No profile for {model_fingerprint[:16]}..., using v1.0")
                    self.stats['profile_misses'] += 1
                
                result = execute_v1_fallback()
                result['_execution_path'] = 'v1_fallback'
                return result
        
        except Exception as e:
            # Error in profile execution - fallback to v1.0 gracefully
            logger.error(
                f"Error executing with profile: {e}, falling back to v1.0",
                exc_info=True
            )
            self.stats['profile_errors'] += 1
            
            result = execute_v1_fallback()
            result['_execution_path'] = 'v1_fallback_after_error'
            return result
    
    def _get_profile(self, 
                    model_fingerprint: str,
                    input_shapes: Dict[str, tuple]) -> Optional[PerformanceProfile]:
        """
        Query ProfileRegistry for model profile.
        
        Args:
            model_fingerprint: Model fingerprint
            input_shapes: Input shapes dict
        
        Returns:
            PerformanceProfile or None
        """
        try:
            profile_dict = self.profile_registry_client.get_profile(
                model_fingerprint=model_fingerprint,
                input_shapes=input_shapes
            )
            
            if profile_dict:
                self.stats['profile_hits'] += 1
                return PerformanceProfile.from_dict(profile_dict)
            else:
                self.stats['profile_misses'] += 1
                return None
        
        except Exception as e:
            logger.error(f"Error querying profile registry: {e}")
            return None
    
    def _execute_with_directives(self,
                                profile: PerformanceProfile,
                                execute_v1_fallback: Callable[..., Dict[str, Any]],
                                hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model with profile directives applied.
        
        Phase 2: Apply resource budget constraints
        Phase 3: Apply optimization directives (fusion, compilation, etc.)
        
        Args:
            profile: PerformanceProfile with directives
            execute_v1_fallback: Fallback execution function
        
        Returns:
            Execution result
        """
        self._apply_resource_budget(profile)
        self._apply_optimizations(profile)

        result = execute_v1_fallback(hints=hints, profile_id=profile.profile_id)
        result['_profile_info'] = {
            'profile_id': profile.profile_id,
            'version': profile.version,
            'execution_phase': profile.execution_phase.value,
            'resource_budget': {
                'activations': profile.resource_budget.activations_fraction,
                'weights': profile.resource_budget.weights_fraction,
                'kv_cache': profile.resource_budget.kv_cache_fraction,
            } if profile.resource_budget else None,
            'optimizations': [d.value for d in profile.optimization_directives],
        }
        result['_execution_path'] = 'v2_profile'
        result['_profile_hints'] = hints
        return result

    def _build_profile_hints(self, profile: PerformanceProfile) -> Dict[str, Any]:
        hints: Dict[str, Any] = {
            'execution_phase': profile.execution_phase.value,
            'profile_id': profile.profile_id,
        }

        if profile.resource_budget:
            hints['resource_budget'] = {
                'activations_fraction': profile.resource_budget.activations_fraction,
                'weights_fraction': profile.resource_budget.weights_fraction,
                'kv_cache_fraction': profile.resource_budget.kv_cache_fraction,
            }

        if profile.optimization_directives:
            hints['optimization_directives'] = [d.value for d in profile.optimization_directives]

        if profile.placement_constraints:
            hints['placement_constraints'] = dict(profile.placement_constraints)

        return hints

    def _apply_resource_budget(self, profile: PerformanceProfile) -> None:
        if not profile.resource_budget:
            return

        budget = profile.resource_budget
        logger.info(
            f"Applying resource budget: "
            f"activations={budget.activations_fraction:.1%}, "
            f"weights={budget.weights_fraction:.1%}, "
            f"kv_cache={budget.kv_cache_fraction:.1%}"
        )
        self._last_applied_budget = budget

    def _apply_optimizations(self, profile: PerformanceProfile) -> None:
        if not profile.optimization_directives:
            return

        directives_str = ', '.join(d.value for d in profile.optimization_directives)
        logger.info(f"Applying optimizations: {directives_str}")
        self._last_applied_directives = [d.value for d in profile.optimization_directives]
    
    def get_stats(self) -> Dict:
        """Get AdaptiveRuntime statistics."""
        total = self.stats['profile_hits'] + self.stats['profile_misses']
        hit_rate = (
            self.stats['profile_hits'] / total 
            if total > 0 else 0.0
        )
        
        return {
            **self.stats,
            'total_profile_queries': total,
            'profile_hit_rate': hit_rate,
            'error_rate': (
                self.stats['profile_errors'] / total
                if total > 0 else 0.0
            ),
        }


# Global singleton instance
_adaptive_runtime: Optional[AdaptiveRuntime] = None
_runtime_lock = __import__('threading').Lock()


def get_adaptive_runtime() -> AdaptiveRuntime:
    """Get global AdaptiveRuntime instance."""
    global _adaptive_runtime
    if _adaptive_runtime is None:
        with _runtime_lock:
            if _adaptive_runtime is None:
                _adaptive_runtime = AdaptiveRuntime()
    return _adaptive_runtime

