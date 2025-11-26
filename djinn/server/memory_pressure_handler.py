"""
Memory Pressure Handler for Production Deployments.

Phase 3 Enhancement: Memory Pressure Detection and Recovery

Monitors GPU memory utilization and takes proactive action when:
- Warning threshold (80% utilization): Trigger aggressive eviction
- Critical threshold (95% utilization): Emergency eviction + recomputation preference
- OOM events: Recover via semantic-aware eviction

Uses semantic knowledge to make intelligent decisions:
- Protect critical data (KV cache during decode)
- Evict cheap activations first
- Prefer recomputation over caching under pressure
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryPressureEvent:
    """Represents a memory pressure event."""
    timestamp: float
    severity: str  # 'warning' (80%), 'critical' (95%), 'oom' (100%)
    utilization_percent: float
    available_bytes: int
    required_bytes: int
    action_taken: str


class MemoryPressureHandler:
    """
    Detects and responds to memory pressure events.
    
    Key responsibilities:
    1. Monitor memory utilization
    2. Trigger evictions when thresholds exceeded
    3. Adapt memory management strategy under pressure
    4. Recover from OOM events
    5. Track pressure events for analysis
    """
    
    def __init__(
        self,
        total_gpu_memory_mb: float,
        warning_threshold_percent: float = 80.0,
        critical_threshold_percent: float = 95.0,
        monitoring_interval_seconds: float = 1.0
    ):
        """
        Initialize memory pressure handler.
        
        Args:
            total_gpu_memory_mb: Total GPU memory in MB
            warning_threshold_percent: Trigger warning at this utilization (default: 80%)
            critical_threshold_percent: Trigger critical at this utilization (default: 95%)
            monitoring_interval_seconds: How often to check memory (default: 1s)
        """
        self.total_memory_mb = total_gpu_memory_mb
        self.warning_threshold = warning_threshold_percent / 100.0
        self.critical_threshold = critical_threshold_percent / 100.0
        self.monitoring_interval = monitoring_interval_seconds
        
        # Current state
        self.current_utilization = 0.0
        self.current_available_bytes = int(total_gpu_memory_mb * 1024 * 1024)
        
        # Callbacks for eviction
        self.eviction_callbacks: Dict[str, Callable] = {}
        
        # Pressure history
        self.pressure_events: List[MemoryPressureEvent] = []
        self.max_pressure_history = 1000
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_under_pressure = False
        self.pressure_recovery_mode = False
        
        # Statistics
        self.stats = {
            'warning_events': 0,
            'critical_events': 0,
            'oom_events': 0,
            'total_memory_freed': 0,
            'last_recovery_time_ms': 0,
        }
        
        logger.info(
            "MemoryPressureHandler initialized (total=%.0f MB, warning=%.0f%%, critical=%.0f%%)",
            total_gpu_memory_mb, warning_threshold_percent, critical_threshold_percent
        )
    
    def register_eviction_callback(
        self,
        name: str,
        callback: Callable[[str], int]
    ) -> None:
        """
        Register a callback for eviction under pressure.
        
        Args:
            name: Name of eviction source (e.g., 'gpu_cache', 'kv_sessions')
            callback: Function(severity: str) -> freed_bytes: int
                      Called with 'warning', 'critical', or 'oom'
                      Should return bytes freed
        """
        self.eviction_callbacks[name] = callback
        logger.info("Registered eviction callback: %s", name)
    
    async def start_monitoring(self) -> None:
        """Start memory pressure monitoring loop."""
        if self.monitoring_task is not None:
            return  # Already running
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Memory pressure monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop memory pressure monitoring loop."""
        if self.monitoring_task is not None:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Memory pressure monitoring stopped")
    
    async def update_memory_status(
        self,
        available_bytes: int,
        total_bytes: int
    ) -> None:
        """
        Update current memory status.
        
        Should be called regularly (e.g., every operation) to track memory.
        
        Args:
            available_bytes: Available GPU memory in bytes
            total_bytes: Total GPU memory in bytes
        """
        self.current_available_bytes = available_bytes
        utilization = 1.0 - (available_bytes / total_bytes) if total_bytes > 0 else 0.0
        self.current_utilization = utilization
        
        # Check thresholds
        await self._check_pressure_thresholds(total_bytes)
    
    async def handle_oom_event(self) -> bool:
        """
        Handle an OOM event.
        
        Attempts to free memory via emergency eviction.
        
        Returns:
            True if recovery successful, False if memory still exhausted
        """
        logger.error("OOM event detected! Attempting emergency recovery...")
        
        start_time = time.time()
        
        # Record event
        event = MemoryPressureEvent(
            timestamp=start_time,
            severity='oom',
            utilization_percent=100.0,
            available_bytes=0,
            required_bytes=self.total_memory_mb * 1024 * 1024,
            action_taken='emergency_eviction'
        )
        self.pressure_events.append(event)
        self.stats['oom_events'] += 1
        
        # Emergency eviction: call all callbacks with 'oom'
        total_freed = 0
        for name, callback in self.eviction_callbacks.items():
            try:
                freed = await asyncio.to_thread(callback, 'oom')
                total_freed += freed
                logger.warning("OOM recovery: %s freed %d bytes", name, freed)
            except Exception as e:
                logger.error("OOM recovery callback %s failed: %s", name, e)
        
        recovery_time_ms = (time.time() - start_time) * 1000
        self.stats['last_recovery_time_ms'] = recovery_time_ms
        self.stats['total_memory_freed'] += total_freed
        
        if total_freed > 0:
            logger.warning("OOM recovery: Freed %.1f MB in %.0fms", 
                          total_freed / (1024 * 1024), recovery_time_ms)
            return True
        else:
            logger.error("OOM recovery failed: Unable to free any memory!")
            return False
    
    async def _monitoring_loop(self) -> None:
        """Periodically check memory pressure."""
        try:
            while True:
                await asyncio.sleep(self.monitoring_interval)
                await self._check_pressure_thresholds(
                    self.total_memory_mb * 1024 * 1024
                )
        except asyncio.CancelledError:
            logger.info("Memory pressure monitoring cancelled")
            raise
    
    async def _check_pressure_thresholds(self, total_bytes: int) -> None:
        """Check if memory utilization exceeds thresholds."""
        utilization = 1.0 - (self.current_available_bytes / total_bytes)
        
        # Critical threshold (95%): Emergency action
        if utilization >= self.critical_threshold:
            if not self.is_under_pressure:
                await self._handle_critical_pressure(utilization, total_bytes)
        
        # Warning threshold (80%): Preventative action
        elif utilization >= self.warning_threshold:
            if not self.is_under_pressure:
                await self._handle_warning_pressure(utilization, total_bytes)
        
        # Below warning: Normal operation
        else:
            if self.is_under_pressure:
                await self._handle_pressure_recovery()
    
    async def _handle_warning_pressure(self, utilization: float, total_bytes: int) -> None:
        """Handle warning-level pressure (80% utilization)."""
        logger.warning("Memory pressure WARNING: %.1f%% utilization", utilization * 100)
        
        self.is_under_pressure = True
        self.stats['warning_events'] += 1
        
        # Trigger evictions with 'warning' severity
        total_freed = 0
        for name, callback in self.eviction_callbacks.items():
            try:
                freed = await asyncio.to_thread(callback, 'warning')
                total_freed += freed
                logger.debug("Warning eviction: %s freed %d bytes", name, freed)
            except Exception as e:
                logger.error("Warning eviction callback %s failed: %s", name, e)
        
        if total_freed > 0:
            logger.warning("Warning recovery: Freed %.1f MB", total_freed / (1024 * 1024))
            self.stats['total_memory_freed'] += total_freed
    
    async def _handle_critical_pressure(self, utilization: float, total_bytes: int) -> None:
        """Handle critical-level pressure (95% utilization)."""
        logger.error("Memory pressure CRITICAL: %.1f%% utilization", utilization * 100)
        
        self.is_under_pressure = True
        self.pressure_recovery_mode = True
        self.stats['critical_events'] += 1
        
        # Trigger evictions with 'critical' severity
        total_freed = 0
        for name, callback in self.eviction_callbacks.items():
            try:
                freed = await asyncio.to_thread(callback, 'critical')
                total_freed += freed
                logger.error("Critical eviction: %s freed %d bytes", name, freed)
            except Exception as e:
                logger.error("Critical eviction callback %s failed: %s", name, e)
        
        if total_freed > 0:
            logger.error("Critical recovery: Freed %.1f MB", total_freed / (1024 * 1024))
            self.stats['total_memory_freed'] += total_freed
        else:
            logger.error("Critical recovery: No memory freed! OOM likely!")
    
    async def _handle_pressure_recovery(self) -> None:
        """Handle recovery from memory pressure."""
        if self.is_under_pressure:
            logger.info("Memory pressure RESOLVED: Back to normal operation")
            self.is_under_pressure = False
            self.pressure_recovery_mode = False
    
    def get_pressure_statistics(self) -> Dict:
        """Get memory pressure statistics."""
        return {
            **self.stats,
            'is_under_pressure': self.is_under_pressure,
            'current_utilization_percent': self.current_utilization * 100,
            'available_bytes': self.current_available_bytes,
            'total_events': len(self.pressure_events),
        }
    
    def get_pressure_history(self, max_events: int = 100) -> List[MemoryPressureEvent]:
        """Get recent pressure events."""
        return self.pressure_events[-max_events:]
    
    def should_prefer_recomputation(self) -> bool:
        """
        Should we prefer recomputation over caching under current pressure?
        
        Returns:
            True if under critical pressure (prefer recomputation)
            False if normal operation (prefer caching)
        """
        return self.pressure_recovery_mode
    
    def get_adaptive_recomputation_threshold(self) -> float:
        """
        Get adaptive recomputation cost threshold under pressure.
        
        Returns:
            Multiplier for recomputation cost threshold
            - Normal: 1.0 (original threshold)
            - Warning: 0.8 (prefer recomputation slightly more)
            - Critical: 0.5 (strongly prefer recomputation)
        """
        if self.pressure_recovery_mode:
            return 0.5  # Critical: aggressive recomputation
        elif self.is_under_pressure:
            return 0.8  # Warning: slight preference for recomputation
        else:
            return 1.0  # Normal: original threshold
