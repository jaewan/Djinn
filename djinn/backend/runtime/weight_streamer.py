"""
Async Weight Streamer: Dual-stream pipelining for weight streaming

Implements the evaluation plan's async pipelining architecture:
- Stream A (Prefetcher): High-priority copy stream (Host → Device)
- Stream B (Compute): Kernel execution stream

Key principle: ZERO CPU blocking in the hot path. All synchronization
happens via GPU events (stream.wait_event()).

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Host Memory (pinned)        │ GPU VRAM (ring buffer)             │
│ [Layer N weights]           │ [Ring Buffer]                      │
└─────────────────────────────────────────────────────────────────┘
            ↓ (async copy via prefetch_stream)
    Layer N+1 arrives              Layer N executes (compute_stream)
            ↓                                ↓
    Event N+1 ready         ←→         Wait for Event N+1
            ↓                                ↓
    Record on prefetch_stream    compute_stream.wait_event()
                              
Timeline:
  T0: Prefetch Layer N into ring[offset0]  (Stream A)
  T0: Compute Layer N-3 using ring[offset0-3]  (Stream B)
  ...
  T1: Prefetch Layer N+1 into ring[offset1]  (Stream A)
  T1: Compute Layer N-2 using ring[offset1-2]  (Stream B, waits for Layer N+1 ready)
  
By the time compute needs a layer, prefetch is 3 layers ahead.
No CPU blocking. All coordination via GPU events.
"""

import logging
import torch
import threading
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time

logger = logging.getLogger(__name__)


class PrefetchState(Enum):
    """State of a prefetch operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    FAILED = "failed"


@dataclass
class PrefetchJob:
    """A pending weight prefetch job."""
    model_id: str
    layer_idx: int
    layer_name: str
    weights: torch.Tensor
    state: PrefetchState = PrefetchState.PENDING
    created_at: float = 0.0
    completed_at: float = 0.0
    error_msg: Optional[str] = None


class WeightStreamer:
    """
    Async dual-stream weight pipelining engine.
    
    Manages prefetching weights for next layers while GPU computes current layers.
    All GPU-GPU synchronization via events (zero CPU blocking).
    """
    
    def __init__(
        self,
        ring_buffer,  # WeightRingBuffer instance
        device: torch.device = None,
        prefetch_queue_size: int = 16,
        chunk_size_bytes: int = None,
    ):
        """
        Initialize weight streamer.
        
        Args:
            ring_buffer: WeightRingBuffer instance for storing weights
            device: CUDA device
            prefetch_queue_size: Max pending prefetch jobs
            chunk_size_bytes: Chunk size for prefetch operations (default: full layer)
        """
        if device is None:
            device = torch.device('cuda:0')
        
        self.ring_buffer = ring_buffer
        self.device = device
        self.prefetch_queue_size = prefetch_queue_size
        self.chunk_size_bytes = chunk_size_bytes or 64 * 1024 * 1024  # Default 64MB
        
        # Create high-priority prefetch stream (only on GPU devices)
        # Handle both torch.device objects and string device names
        device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
        
        if device_type == 'cuda':
            try:
                # Priority -1 is higher priority on NVIDIA GPUs
                self.prefetch_stream = torch.cuda.Stream(device=device, priority=-1)
                logger.info(f"✅ High-priority prefetch stream created on {device}")
            except Exception as e:
                logger.warning(f"Failed to create high-priority stream: {e}, using default")
                self.prefetch_stream = torch.cuda.Stream(device=device)
            
            # Compute stream is just the current stream (typically default)
            self.compute_stream = torch.cuda.current_stream(device=device)
        else:
            # For CPU device, streams are not applicable - set to None
            self.prefetch_stream = None
            self.compute_stream = None
            logger.warning("WeightStreamer on CPU device - async pipelining disabled")
        
        # Job queue and state (use deque for O(1) popleft instead of list pop(0) which is O(n))
        self.prefetch_queue: deque = deque(maxlen=prefetch_queue_size)
        self.queue_lock = threading.Lock()
        self.prefetch_thread: Optional[threading.Thread] = None
        self.should_stop = False
        
        # Statistics
        self.stats = {
            'total_prefetch_jobs': 0,
            'prefetch_success': 0,
            'prefetch_failed': 0,
            'total_bytes_prefetched': 0,
            'total_prefetch_time_ms': 0.0,
            'avg_prefetch_latency_ms': 0.0,
        }
    
    def start(self) -> None:
        """Start the background prefetch thread."""
        if self.prefetch_thread is not None:
            logger.warning("Prefetch thread already running")
            return
        
        self.should_stop = False
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="WeightStreamer-Prefetcher"
        )
        self.prefetch_thread.start()
        logger.info("✅ Weight streamer started")
    
    def stop(self, timeout_secs: float = 10.0) -> None:
        """Stop the background prefetch thread."""
        self.should_stop = True
        
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=timeout_secs)
            if self.prefetch_thread.is_alive():
                logger.warning("Prefetch thread did not stop within timeout")
            self.prefetch_thread = None
        
        logger.info("✅ Weight streamer stopped")
    
    def queue_prefetch(
        self,
        model_id: str,
        layer_idx: int,
        layer_name: str,
        weights: torch.Tensor
    ) -> PrefetchJob:
        """
        Queue a weight for prefetching.
        
        Non-blocking - returns immediately. Actual prefetch happens
        asynchronously on background thread.
        
        Args:
            model_id: Model ID
            layer_idx: Layer index in model
            layer_name: Layer name
            weights: Tensor with weights (CPU or GPU)
        
        Returns:
            PrefetchJob for tracking
        
        Raises:
            RuntimeError: If queue is full
        """
        with self.queue_lock:
            if len(self.prefetch_queue) >= self.prefetch_queue_size:
                raise RuntimeError(
                    f"Prefetch queue full ({self.prefetch_queue_size} jobs)"
                )
            
            job = PrefetchJob(
                model_id=model_id,
                layer_idx=layer_idx,
                layer_name=layer_name,
                weights=weights,
                created_at=time.time()
            )
            self.prefetch_queue.append(job)
            self.stats['total_prefetch_jobs'] += 1
            
            logger.debug(
                f"Queued prefetch: {layer_name} (job {self.stats['total_prefetch_jobs']}, "
                f"queue depth: {len(self.prefetch_queue)})"
            )
            
            return job
    
    def wait_for_layer(
        self,
        model_id: str,
        layer_idx: int
    ) -> bool:
        """
        Wait for a layer's weights to be ready in the ring buffer.
        
        This makes the compute stream wait on the layer_ready_event,
        ensuring weights have been copied to GPU before compute uses them.
        All synchronization happens on GPU (zero CPU blocking).
        
        Args:
            model_id: Model ID
            layer_idx: Layer index
        
        Returns:
            True if successful, False on error
        """
        try:
            # On CPU devices, there's no async pipelining - just return
            if self.compute_stream is None:
                return True
            
            # Get the ready event from ring buffer
            ready_event = self.ring_buffer.get_ready_event(model_id, layer_idx)
            if ready_event is None:
                return True  # No event on CPU device
            
            # Make compute stream wait on this event
            # This is GPU-side synchronization - zero CPU blocking
            self.compute_stream.wait_event(ready_event)
            
            logger.debug(f"Compute stream now waiting for layer {layer_idx} ready event")
            return True
            
        except Exception as e:
            logger.error(f"Failed to wait for layer {layer_idx}: {e}")
            return False
    
    def signal_layer_done(
        self,
        model_id: str,
        layer_idx: int
    ) -> None:
        """
        Signal that compute is done with a layer.
        
        This records an event on the compute stream, which the prefetcher
        can then wait on before reusing the ring buffer slot for the next layer.
        No-op on CPU devices.
        
        Args:
            model_id: Model ID
            layer_idx: Layer index
        """
        # On CPU devices, there's no async pipelining
        if self.compute_stream is None:
            return
        
        try:
            self.ring_buffer.record_done(model_id, layer_idx, self.compute_stream)
            logger.debug(f"Signaled layer {layer_idx} done")
        except Exception as e:
            logger.error(f"Failed to signal layer {layer_idx} done: {e}")
    
    def _prefetch_worker(self) -> None:
        """Background thread that processes prefetch jobs."""
        logger.info("Prefetch worker thread started")
        
        while not self.should_stop:
            # Get next job (O(1) with deque)
            job: Optional[PrefetchJob] = None
            with self.queue_lock:
                if self.prefetch_queue:
                    job = self.prefetch_queue.popleft()  # O(1) vs O(n) with list.pop(0)
            
            if job is None:
                # No work, sleep briefly
                time.sleep(0.001)
                continue
            
            # Process the job
            self._execute_prefetch_job(job)
    
    def _execute_prefetch_job(self, job: PrefetchJob) -> None:
        """Execute a single prefetch job with optional chunking."""
        job.state = PrefetchState.IN_PROGRESS
        start_time = time.time()
        
        try:
            # Determine if we should chunk this transfer
            total_bytes = job.weights.numel() * job.weights.element_size()
            should_chunk = total_bytes > self.chunk_size_bytes and self.chunk_size_bytes > 0
            
            if should_chunk:
                # Transfer in chunks for better pipelining
                logger.debug(
                    f"Chunking {job.layer_name}: {total_bytes / 1024**2:.1f}MB "
                    f"into {self.chunk_size_bytes / 1024**2:.1f}MB chunks"
                )
                self._execute_prefetch_chunked(job)
            else:
                # Transfer entire layer at once
                self._execute_prefetch_monolithic(job)
            
            job.state = PrefetchState.READY
            job.completed_at = time.time()
            
            elapsed_ms = (job.completed_at - job.created_at) * 1000.0
            transfer_bytes = job.weights.numel() * job.weights.element_size()
            
            self.stats['prefetch_success'] += 1
            self.stats['total_bytes_prefetched'] += transfer_bytes
            self.stats['total_prefetch_time_ms'] += elapsed_ms
            self.stats['avg_prefetch_latency_ms'] = (
                self.stats['total_prefetch_time_ms'] / self.stats['prefetch_success']
            )
            
            # Calculate throughput: bytes/ms * 1000 = bytes/s, then /1e9 for GB/s
            throughput_gbs = (transfer_bytes / max(elapsed_ms, 0.001)) * 1000 / (1024**3)
            
            logger.debug(
                f"✅ Prefetch {job.layer_name} ({transfer_bytes / 1024**2:.1f}MB): "
                f"{elapsed_ms:.1f}ms, throughput {throughput_gbs:.1f}GB/s"
            )
            
        except Exception as e:
            job.state = PrefetchState.FAILED
            job.error_msg = str(e)
            job.completed_at = time.time()
            
            self.stats['prefetch_failed'] += 1
            logger.error(
                f"❌ Prefetch {job.layer_name} failed: {e}"
            )
    
    def _execute_prefetch_monolithic(self, job: PrefetchJob) -> None:
        """Transfer entire layer at once."""
        if self.prefetch_stream is not None:
            with torch.cuda.stream(self.prefetch_stream):
                self.ring_buffer.load_layer_weights(
                    job.model_id,
                    job.layer_name,
                    job.weights,
                    stream=self.prefetch_stream
                )
                
                self.ring_buffer.record_ready(
                    job.model_id,
                    job.layer_idx,
                    self.prefetch_stream
                )
        else:
            self.ring_buffer.load_layer_weights(
                job.model_id,
                job.layer_name,
                job.weights,
                stream=None
            )
    
    def _execute_prefetch_chunked(self, job: PrefetchJob) -> None:
        """Transfer weights in chunks for better latency hiding."""
        # Reshape weight tensor to 1D for easier chunking
        flat_weights = job.weights.flatten()
        total_elements = flat_weights.numel()
        chunk_elements = self.chunk_size_bytes // flat_weights.element_size()
        
        # Transfer each chunk
        for start_idx in range(0, total_elements, chunk_elements):
            end_idx = min(start_idx + chunk_elements, total_elements)
            chunk = flat_weights[start_idx:end_idx]
            
            # For chunked transfer, we send chunks but ring buffer still expects full layer
            # This is a simplified implementation - full chunking would require ring buffer changes
            if start_idx == 0:
                # First chunk: send as usual, which will trigger allocation
                if self.prefetch_stream is not None:
                    with torch.cuda.stream(self.prefetch_stream):
                        self.ring_buffer.load_layer_weights(
                            job.model_id,
                            job.layer_name,
                            job.weights,  # Still send full weights
                            stream=self.prefetch_stream
                        )
                        self.ring_buffer.record_ready(
                            job.model_id,
                            job.layer_idx,
                            self.prefetch_stream
                        )
                else:
                    self.ring_buffer.load_layer_weights(
                        job.model_id,
                        job.layer_name,
                        job.weights,
                        stream=None
                    )
                break  # For now, simplification: send full weights anyway
            
            # TODO: Implement true chunked copy that doesn't require full layer at once
            logger.debug(f"  Chunk {start_idx//chunk_elements + 1}: {chunk.numel() * chunk.element_size() / 1024**2:.1f}MB")
    
    def get_stats(self) -> Dict:
        """Get streamer statistics."""
        with self.queue_lock:
            queue_depth = len(self.prefetch_queue)
        
        return {
            **self.stats,
            'queue_depth': queue_depth,
            'success_rate': (
                self.stats['prefetch_success'] / max(self.stats['total_prefetch_jobs'], 1)
            ),
        }

