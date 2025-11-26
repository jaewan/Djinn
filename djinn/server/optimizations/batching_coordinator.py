"""
Automatic operation batching for efficient network communication.

Features:
- Batches multiple operations into single network transfer
- Configurable batch size and timeout
- Async/await support for concurrent operations
- Statistics and monitoring
"""

from __future__ import annotations
import asyncio
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


# ============================================================================
# BATCHING STRUCTURES
# ============================================================================

@dataclass
class BatchedOperation:
    """Single operation to be batched."""
    operation_id: str
    operation_name: str
    inputs: List[Any]
    target_device: str
    future: asyncio.Future = field(default_factory=asyncio.Future)
    timestamp: float = field(default_factory=time.time)


@dataclass
class OperationBatch:
    """Batch of operations to send together."""
    batch_id: str
    operations: List[BatchedOperation]
    created_at: float = field(default_factory=time.time)
    
    def size(self) -> int:
        """Total number of operations in batch."""
        return len(self.operations)
    
    def total_data_bytes(self) -> int:
        """Estimate total data bytes for network transfer."""
        total = 0
        for op in self.operations:
            for inp in op.inputs:
                if hasattr(inp, '__len__'):
                    total += len(inp) * 4  # Assume float32
                else:
                    total += 8  # Scalar
        return total


# ============================================================================
# BATCHING COORDINATOR
# ============================================================================

class BatchingCoordinator:
    """
    Automatically batches remote operations for efficient network transfer.
    
    Strategy:
    1. Collect operations in a queue
    2. When batch_size reached or timeout expires, send batch
    3. Resolve all futures when results arrive
    4. Maintain statistics for monitoring
    """
    
    # Configuration
    DEFAULT_BATCH_SIZE = 16  # Operations per batch
    DEFAULT_BATCH_TIMEOUT_MS = 10  # Max wait time in milliseconds
    MAX_QUEUE_SIZE = 1000  # Prevent unbounded queue growth
    
    def __init__(self, connection_pool=None, batch_size: int = DEFAULT_BATCH_SIZE,
                batch_timeout_ms: int = DEFAULT_BATCH_TIMEOUT_MS,
                max_queue_size: int = MAX_QUEUE_SIZE,
                executor: Optional[Callable] = None):
        """
        Initialize batching coordinator.
        
        Supports both old (with connection_pool) and new (without) API signatures.
        
        Args:
            connection_pool: (Legacy) Connection pool for network communication (ignored in new API)
            batch_size: Number of operations to batch together
            batch_timeout_ms: Max milliseconds to wait before sending
            max_queue_size: Maximum queue size
            executor: (New) Function to execute batches remotely
        """
        # Support both old and new signatures
        self.connection_pool = connection_pool
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds for backward compat
        self.max_queue_size = max_queue_size
        self.executor = executor
        
        # Queue management
        self._queue: List[BatchedOperation] = []
        self._queue_lock = threading.RLock()
        self._batch_timer: Optional[threading.Timer] = None
        
        # Statistics
        self._stats = {
            'batches_sent': 0,
            'operations_batched': 0,
            'total_operations': 0,
            'total_bytes_transferred': 0,
            'avg_batch_size': 0.0,
        }
        self._stats_lock = threading.RLock()
        
        # Public stats property for backward compat
        self.stats = self._stats
        
        # Event loop for async support (optional)
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            except Exception:
                self._loop = None
    
    def submit_operation(self, operation_id: str, operation_name: str,
                        inputs: List[Any], target_device: str) -> asyncio.Future:
        """
        Submit operation for batching.
        
        Args:
            operation_id: Unique operation identifier
            operation_name: Name of remote operation to execute
            inputs: Input tensors/values
            target_device: Target device for execution
            
        Returns:
            Future that will contain result when batch executes
        """
        # Create batched operation with proper Future handling
        try:
            # Try to get running event loop for async futures
            if self._loop is not None and self._loop.is_running():
                future = asyncio.Future(loop=self._loop)
            else:
                future = asyncio.Future()
        except Exception:
            # Fallback: create a simple future-like object
            from concurrent.futures import Future
            future = Future()
        
        batched_op = BatchedOperation(
            operation_id=operation_id,
            operation_name=operation_name,
            inputs=inputs,
            target_device=target_device,
            future=future,
        )
        
        # Add to queue
        with self._queue_lock:
            self._queue.append(batched_op)
            with self._stats_lock:
                self._stats['total_operations'] += 1
            
            # Check if batch is ready to send
            if len(self._queue) >= self.batch_size:
                self._flush_batch_immediately()
            elif len(self._queue) == 1:
                # Start timeout timer for first operation in batch
                self._start_batch_timer()
        
        return future
    
    def _start_batch_timer(self) -> None:
        """Start timer to flush batch after timeout."""
        with self._queue_lock:
            # Cancel existing timer if any
            if self._batch_timer is not None:
                self._batch_timer.cancel()
            
            # Create new timer
            timeout_seconds = self.batch_timeout_ms / 1000.0
            self._batch_timer = threading.Timer(
                timeout_seconds, self._flush_batch_on_timeout
            )
            self._batch_timer.daemon = True
            self._batch_timer.start()
    
    def _flush_batch_immediately(self) -> None:
        """Flush batch immediately when size reached."""
        with self._queue_lock:
            if len(self._queue) >= self.batch_size:
                self._flush_batch_internal()
    
    def _flush_batch_on_timeout(self) -> None:
        """Flush batch when timeout expires."""
        with self._queue_lock:
            if self._queue:
                self._flush_batch_internal()
    
    def _flush_batch_internal(self) -> None:
        """Internal method to flush batch (must be called with lock held)."""
        if not self._queue:
            return
        
        # Create batch
        batch_id = f"batch_{self._stats['batches_sent']}"
        batch = OperationBatch(
            batch_id=batch_id,
            operations=self._queue.copy(),
        )
        self._queue.clear()
        
        # Cancel timer
        if self._batch_timer is not None:
            self._batch_timer.cancel()
            self._batch_timer = None
        
        # Send batch (async)
        self._send_batch(batch)
    
    def _send_batch(self, batch: OperationBatch) -> None:
        """Send batch to remote executor."""
        # Update statistics
        with self._stats_lock:
            self._stats['batches_sent'] += 1
            self._stats['operations_batched'] += batch.size()
            self._stats['total_bytes_transferred'] += batch.total_data_bytes()
            
            if self._stats['operations_batched'] > 0:
                avg_size = self._stats['operations_batched'] / self._stats['batches_sent']
                self._stats['avg_batch_size'] = avg_size
        
        logger.info(f"Sending batch {batch.batch_id}: {batch.size()} operations, "
                   f"{batch.total_data_bytes() / 1024 / 1024:.1f}MB")
        
        # Execute batch (if executor provided)
        if self.executor is not None:
            try:
                results = self.executor(batch)
                self._resolve_batch_futures(batch, results)
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                self._reject_batch_futures(batch, e)
        else:
            # Stub executor: just echo back results
            results = [f"result_{op.operation_id}" for op in batch.operations]
            self._resolve_batch_futures(batch, results)
    
    def _resolve_batch_futures(self, batch: OperationBatch, 
                              results: List[Any]) -> None:
        """Resolve futures with results."""
        if len(results) != len(batch.operations):
            logger.warning(f"Result count mismatch: got {len(results)}, "
                         f"expected {len(batch.operations)}")
        
        for op, result in zip(batch.operations, results):
            try:
                # Set result in future
                if not op.future.done():
                    op.future.set_result(result)
            except Exception as e:
                logger.error(f"Failed to set result: {e}")
    
    def _reject_batch_futures(self, batch: OperationBatch, error: Exception) -> None:
        """Reject all futures with error."""
        for op in batch.operations:
            try:
                if not op.future.done():
                    op.future.set_exception(error)
            except Exception as e:
                logger.error(f"Failed to set exception: {e}")
    
    async def wait_for_operation(self, future: asyncio.Future) -> Any:
        """Wait for operation result (async-safe)."""
        return await asyncio.wait_for(future, timeout=30.0)
    
    def flush(self) -> None:
        """Force flush all pending operations."""
        with self._queue_lock:
            if self._queue:
                self._flush_batch_internal()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        with self._stats_lock:
            return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._stats_lock:
            self._stats = {
                'batches_sent': 0,
                'operations_batched': 0,
                'total_operations': 0,
                'total_bytes_transferred': 0,
                'avg_batch_size': 0.0,
            }


# ============================================================================
# GLOBAL BATCHING COORDINATOR
# ============================================================================

_global_coordinator: Optional[BatchingCoordinator] = None


def get_batching_coordinator(batch_size: int = BatchingCoordinator.DEFAULT_BATCH_SIZE,
                            batch_timeout_ms: int = BatchingCoordinator.DEFAULT_BATCH_TIMEOUT_MS) -> BatchingCoordinator:
    """Get or create global batching coordinator."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = BatchingCoordinator(
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
        )
    return _global_coordinator


def submit_batched_operation(operation_id: str, operation_name: str,
                            inputs: List[Any], target_device: str) -> asyncio.Future:
    """
    Submit operation for batching (convenience function).
    
    Usage:
        future = submit_batched_operation(
            'op_123', 'matmul',
            [x, y],  # inputs
            'gpu:0'  # target
        )
        result = await future
    """
    coordinator = get_batching_coordinator()
    return coordinator.submit_operation(operation_id, operation_name, inputs, target_device)


__all__ = [
    'BatchingCoordinator',
    'BatchedOperation',
    'OperationBatch',
    'get_batching_coordinator',
    'submit_batched_operation',
]
