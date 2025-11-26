"""
Semantic Batch Executor for Multi-Tenant Workloads.

Executes batches of semantically similar requests with phase-specific optimizations:
- LLM Prefill: Maximize parallelism, large batch size, mixed precision
- LLM Decode: KV cache management, incremental execution
- Vision: Efficient conv execution, intermediate feature optimization
- Mixed: Fallback to default execution

This integrates with the MultiTenantCoordinator to execute semantic groups efficiently.
"""

import asyncio
import logging
import time
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExecutionPhase(str, Enum):
    """Execution phases detected from SRG."""
    UNKNOWN = "unknown"
    LLM_PREFILL = "llm_prefill"
    LLM_DECODE = "llm_decode"
    VISION_ENCODING = "vision_encoding"
    VISION_DECODING = "vision_decoding"
    MULTIMODAL_FUSION = "multimodal_fusion"
    TRAINING = "training"


@dataclass
class ExecutionResult:
    """Result of batch execution."""
    request_id: str
    success: bool
    output: Optional[torch.Tensor] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    memory_used_mb: float = 0.0


@dataclass
class BatchExecutionStats:
    """Statistics for batch execution."""
    batch_size: int
    phase: ExecutionPhase
    total_latency_ms: float
    avg_latency_ms: float
    peak_memory_mb: float
    success_count: int
    error_count: int
    throughput_rps: float


class SemanticBatchExecutor:
    """
    Executes batches of semantically similar requests with phase-specific optimizations.
    
    Design:
    - Phase-specific execution strategies (prefill, decode, vision)
    - Memory-efficient batch processing
    - Mixed precision training where beneficial
    - KV cache management for decode
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize semantic batch executor.
        
        Args:
            device: torch device for execution
        """
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # Phase-specific execution strategies
        self.execution_strategies = {
            ExecutionPhase.LLM_PREFILL: self._execute_prefill_optimized,
            ExecutionPhase.LLM_DECODE: self._execute_decode_optimized,
            ExecutionPhase.VISION_ENCODING: self._execute_vision_optimized,
        }
        
        # Statistics tracking
        self.stats = {
            'batches_executed': 0,
            'requests_executed': 0,
            'requests_failed': 0,
            'total_latency_ms': 0.0,
            'peak_memory_mb': 0.0,
        }
        
        logger.info(f"SemanticBatchExecutor initialized on {self.device}")
    
    async def execute_batch(
        self,
        requests: List[Dict],
        phase: ExecutionPhase,
        memory_budget_mb: float,
        executor_func=None
    ) -> Tuple[List[ExecutionResult], BatchExecutionStats]:
        """
        Execute batch with phase-specific optimization.
        
        Args:
            requests: List of request dicts with 'request_id', 'inputs', etc.
            phase: Execution phase (determines strategy)
            memory_budget_mb: Available GPU memory budget
            executor_func: Optional external executor function
        
        Returns:
            (results, stats)
        """
        if not requests:
            empty_stats = BatchExecutionStats(
                batch_size=0,
                phase=phase,
                total_latency_ms=0.0,
                avg_latency_ms=0.0,
                peak_memory_mb=0.0,
                success_count=0,
                error_count=0,
                throughput_rps=0.0
            )
            return [], empty_stats
        
        batch_start = time.time()
        
        # Select strategy for this phase
        strategy = self.execution_strategies.get(
            phase,
            self._execute_default
        )
        
        # Execute batch
        try:
            results = await strategy(requests, memory_budget_mb, executor_func)
        except Exception as e:
            logger.error(f"Error in batch execution: {e}")
            # Return error results for all requests
            results = [
                ExecutionResult(
                    request_id=req['request_id'],
                    success=False,
                    error=str(e),
                    latency_ms=0.0
                )
                for req in requests
            ]
        
        # Compute statistics
        batch_latency_ms = (time.time() - batch_start) * 1000
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        stats = BatchExecutionStats(
            batch_size=len(requests),
            phase=phase,
            total_latency_ms=batch_latency_ms,
            avg_latency_ms=batch_latency_ms / len(requests) if requests else 0,
            peak_memory_mb=max([r.memory_used_mb for r in results]) if results else 0,
            success_count=len(successful),
            error_count=len(failed),
            throughput_rps=len(successful) / (batch_latency_ms / 1000) if batch_latency_ms > 0 else 0
        )
        
        # Update global statistics
        self.stats['batches_executed'] += 1
        self.stats['requests_executed'] += len(successful)
        self.stats['requests_failed'] += len(failed)
        self.stats['total_latency_ms'] += batch_latency_ms
        self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], stats.peak_memory_mb)
        
        logger.info(
            f"✅ Batch executed: {len(successful)}/{len(requests)} succeeded, "
            f"phase={phase.value}, latency={batch_latency_ms:.1f}ms"
        )
        
        return results, stats
    
    async def _execute_prefill_optimized(
        self,
        requests: List[Dict],
        memory_budget_mb: float,
        executor_func=None
    ) -> List[ExecutionResult]:
        """
        Prefill optimization: Maximum parallelism.
        
        Strategy:
        - Combine sequences into large batch
        - Use mixed precision (float16) for memory efficiency
        - Maximize parallelism across batch
        - Process all requests together
        """
        logger.debug(f"Executing PREFILL batch: {len(requests)} requests")
        
        results = []
        batch_start = time.time()
        
        try:
            # Simulate batched prefill execution
            for i, request in enumerate(requests):
                req_start = time.time()
                request_id = request['request_id']
                
                try:
                    # Placeholder: In real implementation, would:
                    # 1. Combine input sequences
                    # 2. Run model forward with mixed precision
                    # 3. Store computed attention heads for later decode
                    
                    # For now, simulate with small delay
                    await asyncio.sleep(0.01)  # 10ms per request
                    
                    if executor_func:
                        output = await executor_func(request)
                    else:
                        output = torch.randn(1, 1, 768)  # Simulated output
                    
                    req_latency_ms = (time.time() - req_start) * 1000
                    
                    results.append(ExecutionResult(
                        request_id=request_id,
                        success=True,
                        output=output,
                        latency_ms=req_latency_ms,
                        memory_used_mb=512.0  # Simulated
                    ))
                    
                except Exception as e:
                    req_latency_ms = (time.time() - req_start) * 1000
                    logger.warning(f"Error executing request {request_id}: {e}")
                    results.append(ExecutionResult(
                        request_id=request_id,
                        success=False,
                        error=str(e),
                        latency_ms=req_latency_ms
                    ))
            
            logger.info(
                f"✅ Prefill batch complete: {len([r for r in results if r.success])}/{len(requests)} "
                f"succeeded in {(time.time() - batch_start) * 1000:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Fatal error in prefill batch: {e}")
        
        return results
    
    async def _execute_decode_optimized(
        self,
        requests: List[Dict],
        memory_budget_mb: float,
        executor_func=None
    ) -> List[ExecutionResult]:
        """
        Decode optimization: KV cache management.
        
        Strategy:
        - Execute each request sequentially (dependency on previous state)
        - Ensure KV cache stays pinned to GPU
        - Use autoregressive decoding
        - Reuse cached attention heads from prefill
        """
        logger.debug(f"Executing DECODE batch: {len(requests)} requests")
        
        results = []
        batch_start = time.time()
        
        try:
            for request in requests:
                req_start = time.time()
                request_id = request['request_id']
                client_id = request.get('client_id')
                
                try:
                    # Placeholder: In real implementation, would:
                    # 1. Get KV cache for this client
                    # 2. Execute one step of autoregressive decoding
                    # 3. Update KV cache
                    # 4. Return logits for next token
                    
                    # For now, simulate with longer delay
                    await asyncio.sleep(0.02)  # 20ms per request (sequential)
                    
                    if executor_func:
                        output = await executor_func(request)
                    else:
                        output = torch.randn(1, 50257)  # Simulated logits
                    
                    req_latency_ms = (time.time() - req_start) * 1000
                    
                    results.append(ExecutionResult(
                        request_id=request_id,
                        success=True,
                        output=output,
                        latency_ms=req_latency_ms,
                        memory_used_mb=256.0  # KV cache
                    ))
                    
                except Exception as e:
                    req_latency_ms = (time.time() - req_start) * 1000
                    logger.warning(f"Error executing decode request {request_id}: {e}")
                    results.append(ExecutionResult(
                        request_id=request_id,
                        success=False,
                        error=str(e),
                        latency_ms=req_latency_ms
                    ))
            
            logger.info(
                f"✅ Decode batch complete: {len([r for r in results if r.success])}/{len(requests)} "
                f"succeeded in {(time.time() - batch_start) * 1000:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Fatal error in decode batch: {e}")
        
        return results
    
    async def _execute_vision_optimized(
        self,
        requests: List[Dict],
        memory_budget_mb: float,
        executor_func=None
    ) -> List[ExecutionResult]:
        """
        Vision optimization: Conv-heavy workloads.
        
        Strategy:
        - Batch images for efficient conv processing
        - Use tensor optimization (channels-last memory layout)
        - Fuse batch norm into conv where possible
        - Process all images together
        """
        logger.debug(f"Executing VISION batch: {len(requests)} requests")
        
        results = []
        batch_start = time.time()
        
        try:
            for request in requests:
                req_start = time.time()
                request_id = request['request_id']
                
                try:
                    # Placeholder: In real implementation, would:
                    # 1. Stack images into batch
                    # 2. Run CNN forward pass
                    # 3. Extract features or logits
                    
                    # For now, simulate
                    await asyncio.sleep(0.015)
                    
                    if executor_func:
                        output = await executor_func(request)
                    else:
                        output = torch.randn(1, 2048)  # Vision features
                    
                    req_latency_ms = (time.time() - req_start) * 1000
                    
                    results.append(ExecutionResult(
                        request_id=request_id,
                        success=True,
                        output=output,
                        latency_ms=req_latency_ms,
                        memory_used_mb=384.0
                    ))
                    
                except Exception as e:
                    req_latency_ms = (time.time() - req_start) * 1000
                    logger.warning(f"Error executing vision request {request_id}: {e}")
                    results.append(ExecutionResult(
                        request_id=request_id,
                        success=False,
                        error=str(e),
                        latency_ms=req_latency_ms
                    ))
            
            logger.info(
                f"✅ Vision batch complete: {len([r for r in results if r.success])}/{len(requests)} "
                f"succeeded in {(time.time() - batch_start) * 1000:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Fatal error in vision batch: {e}")
        
        return results
    
    async def _execute_default(
        self,
        requests: List[Dict],
        memory_budget_mb: float,
        executor_func=None
    ) -> List[ExecutionResult]:
        """
        Default execution strategy for unknown phases.
        """
        logger.debug(f"Executing DEFAULT batch: {len(requests)} requests")
        
        results = []
        
        for request in requests:
            req_start = time.time()
            request_id = request['request_id']
            
            try:
                await asyncio.sleep(0.01)
                
                if executor_func:
                    output = await executor_func(request)
                else:
                    output = torch.randn(1, 256)
                
                req_latency_ms = (time.time() - req_start) * 1000
                
                results.append(ExecutionResult(
                    request_id=request_id,
                    success=True,
                    output=output,
                    latency_ms=req_latency_ms,
                    memory_used_mb=256.0
                ))
                
            except Exception as e:
                req_latency_ms = (time.time() - req_start) * 1000
                logger.warning(f"Error in default execution: {e}")
                results.append(ExecutionResult(
                    request_id=request_id,
                    success=False,
                    error=str(e),
                    latency_ms=req_latency_ms
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get executor statistics."""
        return {
            **self.stats,
            'avg_batch_latency_ms': (
                self.stats['total_latency_ms'] / self.stats['batches_executed']
                if self.stats['batches_executed'] > 0 else 0
            )
        }
