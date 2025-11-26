"""
Feature Flags for v2.0 Adaptive Runtime Rollout.

Phase 2: Controls which models use v2.0 logic vs v1.0 fallback.
"""

import logging
import os
from typing import Set, Dict, Optional
import json
from threading import RLock

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Feature flag management for Phase 2 rollout.
    
    Supports:
    - Enable v2.0 for specific models
    - Global enable/disable
    - Traffic percentage routing
    - Safe defaults (v1.0 for unknown models)
    """
    
    def __init__(self):
        """Initialize feature flags from environment."""
        self._lock = RLock()
        
        # Global v2.0 enable flag
        self.v2_enabled = os.environ.get('DJINN_V2_ENABLED', 'true').lower() == 'true'
        
        # Configured models for v2.0 (space-separated fingerprints or names)
        models_str = os.environ.get('DJINN_V2_MODELS', '').strip()
        self.v2_models: Set[str] = set(m.strip() for m in models_str.split(',') if m.strip())
        
        # Traffic percentage for A/B testing (0-100)
        self.v2_traffic_percent = float(os.environ.get('DJINN_V2_TRAFFIC_PERCENT', '0'))
        
        # Emergency disable
        self.v2_emergency_disable = os.environ.get('DJINN_V2_EMERGENCY_DISABLE', 'false').lower() == 'true'
        
        # Statistics
        self.stats = {
            'v1_requests': 0,
            'v2_requests': 0,
            'v2_errors': 0,
            'v2_fallbacks': 0,
        }
        
        logger.info(
            f"FeatureFlags initialized: "
            f"v2_enabled={self.v2_enabled}, "
            f"v2_models={len(self.v2_models)}, "
            f"v2_traffic_percent={self.v2_traffic_percent}%, "
            f"emergency_disable={self.v2_emergency_disable}"
        )
    
    def should_use_v2(self, model_fingerprint: str) -> bool:
        """
        Determine if v2.0 logic should be used for a model.
        
        Decision tree:
        1. If emergency disable → v1.0
        2. If v2 not globally enabled → v1.0
        3. If model explicitly configured → check traffic routing
        4. Otherwise → v1.0
        
        Args:
            model_fingerprint: Model fingerprint
        
        Returns:
            True if v2.0 should be used, False for v1.0
        """
        with self._lock:
            # Emergency disable
            if self.v2_emergency_disable:
                logger.debug(f"v2.0 disabled (emergency), using v1.0 for {model_fingerprint[:16]}...")
                self.stats['v1_requests'] += 1
                return False
            
            # Global disable
            if not self.v2_enabled:
                logger.debug(f"v2.0 not enabled, using v1.0 for {model_fingerprint[:16]}...")
                self.stats['v1_requests'] += 1
                return False
            
            # Check if model is in v2 list
            if model_fingerprint not in self.v2_models:
                logger.debug(f"Model {model_fingerprint[:16]}... not in v2 list, using v1.0")
                self.stats['v1_requests'] += 1
                return False
            
            # Model is configured for v2 - check traffic routing
            if self.v2_traffic_percent >= 100:
                # 100% traffic to v2
                logger.debug(f"Using v2.0 for {model_fingerprint[:16]}... (100% traffic)")
                self.stats['v2_requests'] += 1
                return True
            elif self.v2_traffic_percent > 0:
                # A/B testing - deterministic hash-based routing
                import hashlib
                hash_val = int(hashlib.md5(model_fingerprint.encode()).hexdigest(), 16)
                use_v2 = (hash_val % 100) < self.v2_traffic_percent
                
                if use_v2:
                    logger.debug(
                        f"Using v2.0 for {model_fingerprint[:16]}... "
                        f"(A/B test: {self.v2_traffic_percent}%)"
                    )
                    self.stats['v2_requests'] += 1
                else:
                    logger.debug(
                        f"Using v1.0 for {model_fingerprint[:16]}... "
                        f"(A/B control: {100 - self.v2_traffic_percent}%)"
                    )
                    self.stats['v1_requests'] += 1
                
                return use_v2
            else:
                # 0% traffic to v2
                logger.debug(f"Using v1.0 for {model_fingerprint[:16]}... (0% traffic)")
                self.stats['v1_requests'] += 1
                return False
    
    def register_model(self, model_fingerprint: str) -> None:
        """Register model for v2.0 testing."""
        with self._lock:
            self.v2_models.add(model_fingerprint)
            logger.info(f"Registered model {model_fingerprint[:16]}... for v2.0 testing")
    
    def unregister_model(self, model_fingerprint: str) -> None:
        """Unregister model from v2.0 testing."""
        with self._lock:
            self.v2_models.discard(model_fingerprint)
            logger.info(f"Unregistered model {model_fingerprint[:16]}... from v2.0 testing")
    
    def set_traffic_percent(self, percent: float) -> None:
        """Update traffic percentage for A/B testing."""
        with self._lock:
            if not 0 <= percent <= 100:
                raise ValueError(f"Traffic percent must be 0-100, got {percent}")
            self.v2_traffic_percent = percent
            logger.info(f"Updated v2.0 traffic percentage to {percent}%")
    
    def emergency_disable_v2(self, reason: str = "") -> None:
        """Emergency disable v2.0 logic."""
        with self._lock:
            self.v2_emergency_disable = True
            logger.warning(f"EMERGENCY: Disabled v2.0 logic. Reason: {reason}")
    
    def emergency_enable_v2(self) -> None:
        """Re-enable v2.0 after emergency disable."""
        with self._lock:
            self.v2_emergency_disable = False
            logger.warning("Re-enabled v2.0 logic after emergency disable")
    
    def get_stats(self) -> Dict:
        """Get feature flag statistics."""
        with self._lock:
            total = self.stats['v1_requests'] + self.stats['v2_requests']
            v2_rate = (
                self.stats['v2_requests'] / total 
                if total > 0 else 0.0
            )
            return {
                **self.stats,
                'total_requests': total,
                'v2_request_rate': v2_rate,
                'v2_error_rate': (
                    self.stats['v2_errors'] / self.stats['v2_requests']
                    if self.stats['v2_requests'] > 0 else 0.0
                ),
                'v2_fallback_rate': (
                    self.stats['v2_fallbacks'] / self.stats['v2_requests']
                    if self.stats['v2_requests'] > 0 else 0.0
                ),
            }
    
    def record_v2_error(self) -> None:
        """Record v2.0 execution error."""
        with self._lock:
            self.stats['v2_errors'] += 1
    
    def record_v2_fallback(self) -> None:
        """Record v2.0 fallback to v1.0."""
        with self._lock:
            self.stats['v2_fallbacks'] += 1
    
    def get_config(self) -> Dict:
        """Get current configuration."""
        with self._lock:
            return {
                'v2_enabled': self.v2_enabled,
                'v2_models': sorted(list(self.v2_models)),
                'v2_traffic_percent': self.v2_traffic_percent,
                'v2_emergency_disable': self.v2_emergency_disable,
            }


# Global singleton instance
_feature_flags: Optional[FeatureFlags] = None
_flags_lock = __import__('threading').Lock()


def get_feature_flags() -> FeatureFlags:
    """Get global FeatureFlags instance."""
    global _feature_flags
    if _feature_flags is None:
        with _flags_lock:
            if _feature_flags is None:
                _feature_flags = FeatureFlags()
    return _feature_flags

