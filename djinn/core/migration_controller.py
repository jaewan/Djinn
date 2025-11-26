"""
Migration Controller: Gradual rollout with feature flags.

Enables safe, gradual migration from graph execution to model cache.
Supports A/B testing and feature flags.

This is part of the redesign plan (Week 3).
"""

import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MigrationStrategy(Enum):
    """Migration strategies."""
    GRAPH_ONLY = "graph_only"  # Use only graph execution (old system)
    MODEL_CACHE_ONLY = "model_cache_only"  # Use only model cache (new system)
    HYBRID = "hybrid"  # Try model cache, fallback to graph
    PERCENTAGE_ROLLOUT = "percentage_rollout"  # Percentage-based rollout


class MigrationController:
    """
    Controls gradual migration from graph execution to model cache.
    
    Features:
    - Feature flags per model/fingerprint
    - Percentage-based rollout
    - A/B testing support
    - Automatic fallback on errors
    """
    
    def __init__(self, strategy: MigrationStrategy = MigrationStrategy.HYBRID):
        """
        Initialize migration controller.
        
        Args:
            strategy: Migration strategy
        """
        self.strategy = strategy
        
        # Per-model feature flags
        self.model_flags: Dict[str, bool] = {}
        
        # Percentage rollout (0.0 to 1.0)
        self.rollout_percentage = 0.0  # Start with 0% (all graph)
        
        # Error tracking for automatic fallback
        self.error_counts: Dict[str, int] = {}
        self.max_errors_before_fallback = 3
        
        logger.info(f"MigrationController initialized with strategy: {strategy.value}")
    
    def should_use_model_cache(self, fingerprint: str) -> bool:
        """
        Determine if model cache should be used for this fingerprint.
        
        Args:
            fingerprint: Model fingerprint
        
        Returns:
            True if model cache should be used
        """
        
        # Check per-model flag (highest priority)
        if fingerprint in self.model_flags:
            return self.model_flags[fingerprint]
        
        # Check strategy
        if self.strategy == MigrationStrategy.GRAPH_ONLY:
            return False
        
        if self.strategy == MigrationStrategy.MODEL_CACHE_ONLY:
            return True
        
        if self.strategy == MigrationStrategy.HYBRID:
            # Check error count - if too many errors, use graph
            if self.error_counts.get(fingerprint, 0) >= self.max_errors_before_fallback:
                return False
            return True
        
        if self.strategy == MigrationStrategy.PERCENTAGE_ROLLOUT:
            # Hash fingerprint to get consistent assignment
            import hashlib
            hash_value = int(hashlib.md5(fingerprint.encode()).hexdigest(), 16)
            percentage = (hash_value % 100) / 100.0
            return percentage < self.rollout_percentage
        
        return False
    
    def record_error(self, fingerprint: str):
        """Record error for a fingerprint (triggers fallback)."""
        self.error_counts[fingerprint] = self.error_counts.get(fingerprint, 0) + 1
        
        if self.error_counts[fingerprint] >= self.max_errors_before_fallback:
            logger.warning(
                f"Too many errors for {fingerprint}, "
                f"switching to graph execution"
            )
    
    def record_success(self, fingerprint: str):
        """Record success (resets error count)."""
        if fingerprint in self.error_counts:
            self.error_counts[fingerprint] = 0
    
    def enable_model_cache(self, fingerprint: str):
        """Enable model cache for specific fingerprint."""
        self.model_flags[fingerprint] = True
        logger.info(f"Enabled model cache for {fingerprint}")
    
    def disable_model_cache(self, fingerprint: str):
        """Disable model cache for specific fingerprint."""
        self.model_flags[fingerprint] = False
        logger.info(f"Disabled model cache for {fingerprint}")
    
    def set_rollout_percentage(self, percentage: float):
        """
        Set rollout percentage (0.0 to 1.0).
        
        Args:
            percentage: Percentage of requests to use model cache (0.0 = 0%, 1.0 = 100%)
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("Percentage must be between 0.0 and 1.0")
        
        self.rollout_percentage = percentage
        logger.info(f"Rollout percentage set to {percentage * 100:.1f}%")
    
    def get_stats(self) -> Dict:
        """Get migration statistics."""
        return {
            'strategy': self.strategy.value,
            'rollout_percentage': self.rollout_percentage,
            'model_flags': dict(self.model_flags),
            'error_counts': dict(self.error_counts)
        }

