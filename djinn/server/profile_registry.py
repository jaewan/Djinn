"""
Profile Registry: Single source of truth for PerformanceProfiles.

Phase 1 Implementation:
- In-memory storage (Phase 2 will add persistent storage)
- Telemetry ingestion from runtime
- Simple query API
- Thread-safe operations
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

from .performance_profile import PerformanceProfile, ExecutionPhase

logger = logging.getLogger(__name__)


class _ReadWriteLock:
    """Simple reader-writer lock to reduce contention."""

    class _ReadContext:
        def __init__(self, parent: "_ReadWriteLock"):
            self._parent = parent

        def __enter__(self):
            with self._parent._cond:
                while self._parent._writer:
                    self._parent._cond.wait()
                self._parent._readers += 1

        def __exit__(self, exc_type, exc_val, exc_tb):
            with self._parent._cond:
                self._parent._readers -= 1
                if self._parent._readers == 0:
                    self._parent._cond.notify_all()

    class _WriteContext:
        def __init__(self, parent: "_ReadWriteLock"):
            self._parent = parent

        def __enter__(self):
            self._parent._cond.acquire()
            while self._parent._writer or self._parent._readers > 0:
                self._parent._cond.wait()
            self._parent._writer = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._parent._writer = False
            self._parent._cond.notify_all()
            self._parent._cond.release()

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False

    def read(self):
        return self._ReadContext(self)

    def write(self):
        return self._WriteContext(self)


class ProfileRegistry:
    """
    In-memory registry for PerformanceProfiles.
    
    Phase 1: Simple in-memory implementation
    Phase 2: Add persistent storage (SQLite/Redis)
    Phase 3: Add HA/replication
    """
    
    def __init__(self):
        """Initialize registry."""
        # Storage: profile_id -> PerformanceProfile
        self._profiles: Dict[str, PerformanceProfile] = {}
        
        # Index: model_fingerprint -> List[profile_id]
        self._by_fingerprint: Dict[str, List[str]] = defaultdict(list)
        
        # Shape-based quick lookup: (model, shape_key) -> profile_id
        self._shape_index: Dict[Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]], str] = {}
        
        # Telemetry buffer: profile_id -> deque of telemetry
        self._telemetry_maxlen = 1000
        def _buffer_factory():
            return deque(maxlen=self._telemetry_maxlen)
        self._telemetry_buffer: Dict[str, deque] = defaultdict(_buffer_factory)
        
        # Thread safety
        self._rwlock = _ReadWriteLock()
        
        # Statistics
        self.stats = {
            'profiles_stored': 0,
            'queries': 0,
            'query_hits': 0,
            'query_misses': 0,
            'telemetry_records': 0,
        }
        
        logger.info("ProfileRegistry initialized (in-memory storage)")
    
    def register_profile(self, profile: PerformanceProfile) -> bool:
        """
        Register a new profile or update existing one.
        
        Args:
            profile: PerformanceProfile to register
        
        Returns:
            True if successful, False otherwise
        """
        with self._rwlock.write():
            # Validate profile
            is_valid, errors = profile.validate()
            if not is_valid:
                logger.error(f"Invalid profile {profile.profile_id}: {errors}")
                return False
            
            # Check if profile exists (update version if so)
            if profile.profile_id in self._profiles:
                existing = self._profiles[profile.profile_id]
                profile.version = existing.version + 1
                logger.info(
                    f"Updating profile {profile.profile_id} "
                    f"(v{existing.version} -> v{profile.version})"
                )
            else:
                logger.info(f"Registering new profile {profile.profile_id} (v{profile.version})")
                self.stats['profiles_stored'] += 1
            
            # Store profile
            profile.updated_at = time.time()
            self._profiles[profile.profile_id] = profile
            
            # Update index
            if profile.profile_id not in self._by_fingerprint[profile.model_fingerprint]:
                self._by_fingerprint[profile.model_fingerprint].append(profile.profile_id)
            
            self._index_profile_shape(profile)
            
            return True
    
    def get_profile(self, 
                    model_fingerprint: str,
                    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None) -> Optional[PerformanceProfile]:
        """
        Query profile for a model and input shapes.
        
        Args:
            model_fingerprint: Model fingerprint
            input_shapes: Input shapes dict (optional)
        
        Returns:
            PerformanceProfile if found, None otherwise
        """
        self.stats['queries'] += 1
        
        shape_key = self._shape_key(model_fingerprint, input_shapes) if input_shapes else None
        
        with self._rwlock.read():
            if shape_key:
                indexed_id = self._shape_index.get(shape_key)
                if indexed_id and indexed_id in self._profiles:
                    profile = self._profiles[indexed_id]
                    self.stats['query_hits'] += 1
                    logger.debug(f"Shape index hit for {model_fingerprint}")
                    return profile

            # Get all profiles for this model
            profile_ids = self._by_fingerprint.get(model_fingerprint, [])
            
            if not profile_ids:
                self.stats['query_misses'] += 1
                logger.debug(f"No profiles found for model {model_fingerprint}")
                return None
            
            # If no input shapes provided, return first profile
            if input_shapes is None:
                profile = self._profiles[profile_ids[0]]
                self.stats['query_hits'] += 1
                logger.debug(f"Returning default profile {profile.profile_id} for {model_fingerprint}")
                return profile
            
            # Find profile matching input shapes
            for profile_id in profile_ids:
                profile = self._profiles[profile_id]
                if profile.input_envelope and profile.input_envelope.matches(input_shapes):
                    self.stats['query_hits'] += 1
                    logger.debug(
                        f"Found matching profile {profile.profile_id} "
                        f"for {model_fingerprint} with shapes {input_shapes}"
                    )
                    return profile
            
            # No exact match - return first profile as fallback
            profile = self._profiles[profile_ids[0]]
            self.stats['query_hits'] += 1
            logger.debug(
                f"No exact match for shapes {input_shapes}, "
                f"returning fallback profile {profile.profile_id}"
            )
            return profile
    
    def get_profile_by_id(self, profile_id: str) -> Optional[PerformanceProfile]:
        """Get profile by ID."""
        with self._rwlock.read():
            return self._profiles.get(profile_id)
    
    def record_telemetry(self, 
                        profile_id: Optional[str],
                        model_fingerprint: str,
                        telemetry: Dict) -> None:
        """
        Record telemetry from runtime execution.
        
        Args:
            profile_id: Profile ID used (None if no profile)
            model_fingerprint: Model fingerprint
            telemetry: Telemetry dict from execution
        """
        with self._rwlock.write():
            self.stats['telemetry_records'] += 1
            
            # Store in buffer
            key = profile_id if profile_id else f"no_profile:{model_fingerprint}"
            self._telemetry_buffer[key].append({
                'timestamp': time.time(),
                'profile_id': profile_id,
                'model_fingerprint': model_fingerprint,
                **telemetry
            })
            
            # Update profile statistics if profile exists
            if profile_id and profile_id in self._profiles:
                profile = self._profiles[profile_id]
                profile.observation_count += 1
                
                # Update observed metrics (exponential moving average)
                alpha = 0.1  # Smoothing factor
                if 'execution_time_ms' in telemetry:
                    latency = telemetry['execution_time_ms']
                    if profile.observed_latency_ms is None:
                        profile.observed_latency_ms = latency
                    else:
                        profile.observed_latency_ms = (
                            alpha * latency + (1 - alpha) * profile.observed_latency_ms
                        )
                
                if 'memory_status' in telemetry:
                    memory_gb = telemetry['memory_status'].get('used_gb', 0)
                    if profile.observed_memory_gb is None:
                        profile.observed_memory_gb = memory_gb
                    else:
                        profile.observed_memory_gb = (
                            alpha * memory_gb + (1 - alpha) * profile.observed_memory_gb
                        )
                
                latency_str = f"{profile.observed_latency_ms:.1f}" if profile.observed_latency_ms else "0"
                memory_str = f"{profile.observed_memory_gb:.2f}" if profile.observed_memory_gb else "0"
                logger.debug(
                    f"Updated profile {profile_id} telemetry: "
                    f"observations={profile.observation_count}, "
                    f"latency={latency_str}ms, "
                    f"memory={memory_str}GB"
                )
    
    def list_profiles(self, model_fingerprint: Optional[str] = None) -> List[PerformanceProfile]:
        """
        List all profiles, optionally filtered by model fingerprint.
        
        Args:
            model_fingerprint: Optional filter by model
        
        Returns:
            List of profiles
        """
        with self._rwlock.read():
            if model_fingerprint:
                profile_ids = self._by_fingerprint.get(model_fingerprint, [])
                return [self._profiles[pid] for pid in profile_ids]
            else:
                return list(self._profiles.values())
    
    def get_telemetry(self, 
                     profile_id: Optional[str] = None,
                     limit: int = 100) -> List[Dict]:
        """
        Get telemetry records.
        
        Args:
            profile_id: Optional filter by profile ID
            limit: Max records to return
        
        Returns:
            List of telemetry dicts
        """
        with self._rwlock.read():
            if profile_id:
                records = self._telemetry_buffer.get(profile_id, [])
            else:
                # Flatten all records
                records = []
                for buffer in self._telemetry_buffer.values():
                    records.extend(buffer)
                # Sort by timestamp
                records.sort(key=lambda r: r['timestamp'], reverse=True)
            
            return records[:limit]
    
    def get_stats(self) -> Dict:
        """Get registry statistics."""
        with self._rwlock.read():
            hit_rate = (
                self.stats['query_hits'] / self.stats['queries'] 
                if self.stats['queries'] > 0 else 0.0
            )
            return {
                **self.stats,
                'query_hit_rate': hit_rate,
                'total_profiles': len(self._profiles),
                'total_models': len(self._by_fingerprint),
            }
    
    def clear(self) -> None:
        """Clear all profiles and telemetry (for testing)."""
        with self._rwlock.write():
            self._profiles.clear()
            self._by_fingerprint.clear()
            self._shape_index.clear()
            self._telemetry_buffer.clear()
            self.stats = {
                'profiles_stored': 0,
                'queries': 0,
                'query_hits': 0,
                'query_misses': 0,
                'telemetry_records': 0,
            }
            logger.info("ProfileRegistry cleared")
    
    def _shape_key(self, model_fingerprint: str, input_shapes: Optional[Dict[str, Tuple[int, ...]]]) -> Optional[Tuple[str, Tuple[Tuple[str, Tuple[int, ...]], ...]]]:
        """Create normalized tuple key for shape matching."""
        if not input_shapes:
            return None
        normalized = tuple(sorted((name, tuple(shape)) for name, shape in input_shapes.items()))
        return (model_fingerprint, normalized)

    def _extract_exact_shape(self, profile: PerformanceProfile) -> Optional[Dict[str, Tuple[int, ...]]]:
        """Extract exact shape mapping when min == max."""
        if not profile.input_envelope:
            return None
        shape_map = {}
        for name, (min_shape, max_shape) in profile.input_envelope.shape_constraints.items():
            if min_shape != max_shape:
                return None
            shape_map[name] = min_shape
        return shape_map

    def _index_profile_shape(self, profile: PerformanceProfile) -> None:
        """Index profile by exact shape for fast lookup."""
        shape_map = self._extract_exact_shape(profile)
        if not shape_map:
            return
        key = self._shape_key(profile.model_fingerprint, shape_map)
        if key:
            self._shape_index[key] = profile.profile_id


# Global singleton instance
_registry_instance: Optional[ProfileRegistry] = None
_registry_lock = threading.Lock()


def get_profile_registry() -> ProfileRegistry:
    """Get global ProfileRegistry instance (singleton)."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ProfileRegistry()
    return _registry_instance

