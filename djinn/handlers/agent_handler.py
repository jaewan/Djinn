"""
Agent Phase Handler: Handles IO_WAIT and COMPUTE phases for reasoning agents.

This is the current Djinn phase handler, extracted into a pluggable interface.
It implements the semantic scheduler logic for agent workflows:
- IO_WAIT: Agent entering tool execution → evict KV cache to CPU
- COMPUTE: Agent resuming computation → restore KV cache from CPU
"""

import asyncio
import logging
from typing import Dict, Any, List
import time

from djinn.interfaces import PhaseHandler

logger = logging.getLogger(__name__)


def _handle_task_exception(task: asyncio.Task, task_name: str) -> None:
    """Log exceptions from fire-and-forget tasks."""
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(f"Background task '{task_name}' failed: {exc}", exc_info=exc)
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        pass


class AgentPhaseHandler(PhaseHandler):
    """
    Phase handler for reasoning agent workloads (ReAct, CoT, etc.).
    
    Handles two phases:
    - IO_WAIT: Long-latency I/O (tool execution, DB queries)
    - COMPUTE: GPU computation (generation, reasoning)
    
    Implements the semantic scheduler policy:
    1. On IO_WAIT: Immediately evict KV cache, schedule proactive prefetch
    2. On COMPUTE: Restore KV cache and prepare for inference
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize agent phase handler.
        
        Args:
            config: Configuration dict with:
                - prefetch_margin_ms: safety margin for prefetch timing (default: 100ms - balanced)
                - max_swapped_sessions: max concurrent swapped sessions
        """
        self.config = config or {}
        # Balanced margin: 100ms (reduced from 150ms for faster prefetch)
        # Gives restore operation enough time without being too aggressive
        self.prefetch_margin_ms = self.config.get('prefetch_margin_ms', 100)
        logger.info(f"AgentPhaseHandler initialized (prefetch_margin_ms={self.prefetch_margin_ms})")
    
    async def on_phase_signal(
        self,
        session_id: str,
        phase: str,
        hints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle phase signal for agent."""
        phase = phase.upper()
        
        if phase == 'IO_WAIT':
            return await self._handle_io_wait(session_id, hints)
        elif phase == 'COMPUTE':
            return await self._handle_compute(session_id, hints)
        else:
            return {
                'status': 'error',
                'message': f'Unknown phase: {phase}',
                'phase': phase,
                'session_id': session_id[:12]
            }
    
    def supported_phases(self) -> List[str]:
        """Return supported phases."""
        return ['IO_WAIT', 'COMPUTE']
    
    async def _handle_io_wait(
        self,
        session_id: str,
        hints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle IO_WAIT phase: agent entering I/O-bound operation.
        
        Strategy:
        1. Immediately evict KV cache to free GPU
        2. If estimated_resume_ms provided, schedule proactive prefetch
        
        This enables parking lot solution: GPU freed during tool execution,
        pre-fetches before agent resumes.
        """
        try:
            from djinn.server.multi_tenant.kv_session_manager import get_kv_session_manager
            from djinn.server.memory_metrics import get_metrics
            
            # Record signal
            metrics = get_metrics()
            metrics.record_semantic_signal('IO_WAIT')
            
            # Immediately evict KV cache
            kv_manager = get_kv_session_manager()
            if kv_manager:
                evict_task = asyncio.create_task(
                    kv_manager.evict_kv_to_host(session_id)
                )
                evict_task.add_done_callback(lambda t: _handle_task_exception(t, f"evict:{session_id[:12]}"))
                logger.info(f"IO_WAIT signal: scheduling eviction for {session_id[:12]}")
                
                # If client provided estimated resume time, schedule proactive prefetch
                estimated_resume_ms = hints.get('estimated_resume_ms')
                if estimated_resume_ms and estimated_resume_ms > 0:
                    restore_delay_ms = max(0, estimated_resume_ms - self.prefetch_margin_ms)
                    prefetch_task = asyncio.create_task(
                        self._schedule_prefetch(session_id, restore_delay_ms, kv_manager)
                    )
                    prefetch_task.add_done_callback(lambda t: _handle_task_exception(t, f"prefetch:{session_id[:12]}"))
                    logger.debug(f"Prefetch scheduled for {session_id[:12]}: "
                               f"delay={restore_delay_ms}ms (estimated_resume={estimated_resume_ms}ms)")
            
            return {
                'status': 'ok',
                'phase': 'IO_WAIT',
                'session_id': session_id[:12],
                'action': 'evicted'
            }
        
        except Exception as e:
            logger.error(f"Error handling IO_WAIT for {session_id}: {e}", exc_info=True)
            return {
                'status': 'error',
                'phase': 'IO_WAIT',
                'session_id': session_id[:12],
                'message': str(e)
            }
    
    async def _handle_compute(
        self,
        session_id: str,
        hints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle COMPUTE phase: agent resuming GPU computation.
        
        Strategy:
        1. Check if prefetch is already in progress (race condition protection)
        2. If needed, restore KV cache from host
        
        This ensures KV is ready for the next generation step.
        """
        try:
            from djinn.server.multi_tenant.kv_session_manager import get_kv_session_manager
            from djinn.server.memory_metrics import get_metrics
            
            # Record signal
            metrics = get_metrics()
            metrics.record_semantic_signal('COMPUTE')
            
            kv_manager = get_kv_session_manager()
            if kv_manager:
                # Check if prefetch already in progress
                prefetch_status = await kv_manager.check_prefetch_in_progress(session_id)
                if prefetch_status:
                    logger.info(f"COMPUTE signal: prefetch already in progress for {session_id[:12]}, skipping restore")
                    return {
                        'status': 'ok',
                        'phase': 'COMPUTE',
                        'session_id': session_id[:12],
                        'action': 'prefetch_in_progress'
                    }
                
                # Normal case: schedule restore
                restore_task = asyncio.create_task(
                    kv_manager.restore_kv_from_host(session_id)
                )
                restore_task.add_done_callback(lambda t: _handle_task_exception(t, f"restore:{session_id[:12]}"))
                logger.info(f"COMPUTE signal: scheduling restore for {session_id[:12]}")
            
            return {
                'status': 'ok',
                'phase': 'COMPUTE',
                'session_id': session_id[:12],
                'action': 'restoring'
            }
        
        except Exception as e:
            logger.error(f"Error handling COMPUTE for {session_id}: {e}", exc_info=True)
            return {
                'status': 'error',
                'phase': 'COMPUTE',
                'session_id': session_id[:12],
                'message': str(e)
            }
    
    async def _schedule_prefetch(
        self,
        session_id: str,
        delay_ms: int,
        kv_manager: Any
    ) -> None:
        """Schedule proactive KV restore ahead of expected compute resumption."""
        try:
            # Mark prefetch as in progress
            await kv_manager.set_prefetch_in_progress(session_id, True)
            
            # Wait for the specified delay
            await asyncio.sleep(delay_ms / 1000.0)
            
            # Perform restoration
            start_time = time.perf_counter()
            await kv_manager.restore_kv_from_host(session_id)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            logger.debug(f"✅ Prefetch completed: {session_id[:12]} (scheduled {delay_ms}ms ahead) "
                        f"restore_latency={latency_ms:.1f}ms")
            
            # Record metrics
            from djinn.server.memory_metrics import get_metrics
            metrics = get_metrics()
            metrics.record_semantic_prefetch(latency_ms)
            
            # Clear the prefetch_in_progress flag
            await kv_manager.set_prefetch_in_progress(session_id, False)
        
        except Exception as e:
            logger.error(f"Scheduled prefetch error for {session_id[:12]}: {e}", exc_info=True)
            # Ensure flag is cleared even on error
            try:
                await kv_manager.set_prefetch_in_progress(session_id, False)
            except Exception:
                pass

