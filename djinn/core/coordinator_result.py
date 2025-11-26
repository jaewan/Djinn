"""
Result Queue Management for Djinn Coordinator.

Handles asynchronous result delivery from remote operations.
Provides thread-safe result queues and callback handling.
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResultManager:
    """
    Manages result queues for asynchronous remote operations.
    
    Thread-safe singleton for process-wide result tracking.
    """
    
    def __init__(self):
        """Initialize result manager."""
        self._result_queues: Dict[str, asyncio.Queue] = {}
        self._result_handlers: Dict[str, Any] = {}
    
    def create_result_queue(self, result_id: str) -> asyncio.Queue:
        """
        Create result queue for operation.
        
        Args:
            result_id: Unique result identifier
            
        Returns:
            asyncio.Queue for receiving the result
        """
        result_queue = asyncio.Queue(maxsize=1)
        self._result_queues[result_id] = result_queue
        return result_queue
    
    async def handle_result_received(self, result_id: str, result: Any):
        """
        Handle result from transport (callback).
        
        Args:
            result_id: Result identifier (from metadata)
            result: torch.Tensor (success) or Exception (failure)
        """
        logger.debug(f"  Result received: {result_id}")
        logger.debug(f"  Available queues: {list(self._result_queues.keys())}")

        if result_id in self._result_queues:
            queue = self._result_queues[result_id]

            try:
                # Put result (tensor or exception) in queue (non-blocking)
                queue.put_nowait(result)

                # Log based on type
                if isinstance(result, Exception):
                    logger.debug(f"  Error delivered: {result}")
                else:
                    logger.debug(f"  Result delivered: {result.shape if hasattr(result, 'shape') else 'unknown'}")
            except asyncio.QueueFull:
                logger.warning(f"Result queue full for {result_id}")
        else:
            logger.warning(f"No queue for result {result_id}")
            logger.warning(f"Available queue keys: {list(self._result_queues.keys())}")
    
    def cleanup_queue(self, result_id: str):
        """Remove result queue after operation completes."""
        self._result_queues.pop(result_id, None)
    
    def get_active_queue_count(self) -> int:
        """Get number of active result queues."""
        return len(self._result_queues)

