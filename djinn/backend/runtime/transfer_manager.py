"""
Transfer manager for coordinating data movement between nodes.

Handles the lifecycle of transfers, including initiation, progress tracking,
and completion notification.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, List
import logging

from .interfaces import TransferRequest, TransferResult

logger = logging.getLogger(__name__)


@dataclass
class TransferFuture:
    """Future for tracking transfer completion."""
    transfer_id: str
    request: TransferRequest
    result: Optional[TransferResult] = None
    done_callbacks: List[Callable] = field(default_factory=list)

    def set_result(self, result: TransferResult):
        """Set the result and notify callbacks."""
        self.result = result
        for callback in self.done_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Transfer callback failed: {e}")

    def add_done_callback(self, callback: Callable):
        """Add callback to be called when transfer completes."""
        self.done_callbacks.append(callback)


class TransferManager:
    """Manages ongoing transfers between nodes."""

    def __init__(self):
        self.active_transfers: Dict[str, TransferFuture] = {}
        self._lock = asyncio.Lock()

    async def initiate_transfer(self, request: TransferRequest) -> TransferFuture:
        """Initiate a new transfer."""
        async with self._lock:
            transfer_id = str(uuid.uuid4())
            future = TransferFuture(transfer_id, request)
            self.active_transfers[transfer_id] = future
            return future

    async def complete_transfer(self, transfer_id: str, result: TransferResult):
        """Mark a transfer as complete."""
        async with self._lock:
            if transfer_id in self.active_transfers:
                future = self.active_transfers[transfer_id]
                future.set_result(result)
                del self.active_transfers[transfer_id]

    def get_transfer(self, transfer_id: str) -> Optional[TransferFuture]:
        """Get transfer future by ID."""
        return self.active_transfers.get(transfer_id)

    def cancel_transfer(self, transfer_id: str):
        """Cancel an ongoing transfer."""
        if transfer_id in self.active_transfers:
            del self.active_transfers[transfer_id]
