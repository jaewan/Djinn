"""
PhaseHandler: Extensible interface for workload-specific semantic phases.

Different workloads have different execution phases:
- Reasoning agents: IO_WAIT (tool execution) and COMPUTE (generation)
- Training: FORWARD, BACKWARD, UPDATE
- Vision pipelines: PREPROCESS, ENCODE, DECODE

This interface allows workload-specific phase handling without modifying Djinn core.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class PhaseHandler(ABC):
    """
    Extensible handler for workload-specific semantic phase signals.
    
    The framework handles generic scheduling. The phase handler implements
    workload-specific policies: what to do when a session enters a given phase.
    """
    
    @abstractmethod
    async def on_phase_signal(
        self, 
        session_id: str, 
        phase: str, 
        hints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a phase signal from a client.
        
        Args:
            session_id: Session ID that sent the signal
            phase: Phase name (case-insensitive, e.g., "IO_WAIT", "COMPUTE")
            hints: Optional hints from client (e.g., estimated_resume_ms)
        
        Returns:
            Response dict with:
            - 'status': 'ok' or 'error'
            - 'phase': The phase that was processed
            - 'session_id': Session ID (for logging)
            - Other phase-specific data
        
        Example (Agent workload):
            if phase == 'IO_WAIT':
                # Agent entering tool execution → swap KV cache to CPU
                await scheduler.evict_kv(session_id)
                if hints.get('estimated_resume_ms'):
                    # Pre-fetch KV before agent wakes up
                    schedule_prefetch(session_id, hints['estimated_resume_ms'])
        """
        pass
    
    @abstractmethod
    def supported_phases(self) -> List[str]:
        """
        Return list of phases this handler supports.
        
        Used for validation and documentation.
        
        Example:
            return ["IO_WAIT", "COMPUTE", "DEBUG"]
        """
        pass

