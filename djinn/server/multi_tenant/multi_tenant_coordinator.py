"""
Multi-Tenant Coordinator for Semantic-Aware GPU Sharing.

Implements the gap between single-tenant implementation and multi-tenant vision:
- Concurrent request handling
- Per-client semantic context tracking
- Semantic-aware admission control
- Semantic eviction prioritization
- Fairness-aware resource allocation

This is the key component for demonstrating semantic-driven multi-tenancy benefits.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


class ClientPriority(Enum):
    """Client priority levels for fair resource allocation."""
    INTERACTIVE = 0  # Highest priority (VQA, chatbot)
    SERVING = 1      # Medium priority (inference serving)
    BATCH = 2        # Lowest priority (training, batch inference)


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
class ClientContext:
    """Per-client execution context with semantic metadata."""
    client_id: str
    model_id: str
    phase: ExecutionPhase
    priority: ClientPriority
    memory_usage_mb: float = 0.0
    kv_cache_size_mb: float = 0.0
    active_requests: int = 0
    slo_deadline_ms: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    cumulative_requests: int = 0
    failed_requests: int = 0
    slo_violations: int = 0
    phase_history: List[ExecutionPhase] = field(default_factory=list)


@dataclass
class AdmissionDecision:
    """Result of admission control decision."""
    admitted: bool
    reason: str
    memory_freed_mb: float = 0.0
    evicted_clients: List[str] = field(default_factory=list)


class MultiTenantCoordinator:
    """
    Core multi-tenancy coordinator leveraging semantic information.
    
    Key capabilities:
    1. Concurrent request management with per-client tracking
    2. Semantic-aware admission control (predicts memory needs from SRG)
    3. Semantic eviction (protects critical data, understands phases)
    4. Fair resource allocation (DRF-based with semantic adjustments)
    5. SLO tracking and latency management
    """
    
    def __init__(self, gpu_memory_mb: int = 22000, max_concurrent_clients: int = 10):
        """
        Initialize multi-tenant coordinator.
        
        Args:
            gpu_memory_mb: Total GPU memory available
            max_concurrent_clients: Maximum concurrent clients
        """
        self.gpu_memory_mb = gpu_memory_mb
        self.memory_allocated_mb = 0.0
        self.max_concurrent_clients = max_concurrent_clients
        
        # Client management
        self.clients: Dict[str, ClientContext] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.request_results: Dict[str, asyncio.Future] = {}
        
        # Semantic-aware memory management (reuse existing Phase 2-3 components)
        self.phase_memory_mgr = None  # Will be set from external
        self.lifetime_evictor = None
        self.recompute_decider = None
        self.kv_manager = None
        
        # Fairness tracking
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
        self.fairness_history: List[Dict] = []
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'admitted_requests': 0,
            'rejected_requests': 0,
            'evictions_performed': 0,
            'total_memory_freed_mb': 0.0,
            'start_time': time.time(),
        }
        
        logger.info(
            f"MultiTenantCoordinator initialized: {gpu_memory_mb}MB, "
            f"max_clients={max_concurrent_clients}"
        )
    
    async def submit_request(
        self,
        client_id: str,
        model_id: str,
        phase: ExecutionPhase,
        inputs: Dict,
        priority: ClientPriority = ClientPriority.SERVING,
        slo_deadline_ms: Optional[float] = None,
        estimated_memory_mb: Optional[float] = None
    ) -> str:
        """
        Submit request with semantic metadata for admission control.
        
        Args:
            client_id: Unique client identifier
            model_id: Model being executed
            phase: Execution phase (from SRG)
            inputs: Input tensors
            priority: Client priority level
            slo_deadline_ms: SLO deadline in milliseconds
            estimated_memory_mb: Estimated memory requirement (if provided)
        
        Returns:
            request_id for tracking
        """
        self.stats['total_requests'] += 1
        
        # Create or update client context
        if client_id not in self.clients:
            slo = self._compute_slo(priority, phase)
            self.clients[client_id] = ClientContext(
                client_id=client_id,
                model_id=model_id,
                phase=phase,
                priority=priority,
                slo_deadline_ms=slo_deadline_ms or slo,
                created_at=time.time(),
                last_active=time.time()
            )
            logger.info(f"✨ New client registered: {client_id} (priority={priority.name}, phase={phase.value})")
        else:
            # Update phase if changed
            old_phase = self.clients[client_id].phase
            if phase != old_phase:
                self.clients[client_id].phase_history.append(old_phase)
                self.clients[client_id].phase = phase
                logger.info(f"📊 Client {client_id} phase changed: {old_phase.value} → {phase.value}")
            
            self.clients[client_id].last_active = time.time()
        
        # Estimate memory if not provided
        if estimated_memory_mb is None:
            estimated_memory_mb = self._estimate_memory_from_inputs(inputs)
        
        # Admission control based on semantic analysis
        decision = await self._admit_request(client_id, estimated_memory_mb)
        
        if not decision.admitted:
            logger.warning(
                f"❌ Request rejected: {decision.reason} "
                f"(freed {decision.memory_freed_mb:.1f}MB)"
            )
            self.stats['rejected_requests'] += 1
            raise RuntimeError(f"Admission failed: {decision.reason}")
        
        self.stats['admitted_requests'] += 1
        
        # Create request with semantic metadata
        request_id = str(uuid.uuid4())
        request = {
            'request_id': request_id,
            'client_id': client_id,
            'model_id': model_id,
            'phase': phase,
            'priority': priority,
            'inputs': inputs,
            'estimated_memory_mb': estimated_memory_mb,
            'submitted_at': time.time(),
            'deadline_ms': slo_deadline_ms or self.clients[client_id].slo_deadline_ms
        }
        
        # Queue request
        await self.request_queue.put(request)
        self.request_results[request_id] = asyncio.Future()
        
        # Update client state
        self.clients[client_id].active_requests += 1
        self.clients[client_id].cumulative_requests += 1
        self.clients[client_id].memory_usage_mb += estimated_memory_mb
        
        logger.debug(f"✅ Request queued: {request_id} for {client_id}")
        
        return request_id
    
    async def _admit_request(
        self,
        client_id: str,
        estimated_memory_mb: float
    ) -> AdmissionDecision:
        """
        Semantic-aware admission control.
        
        Uses SRG phase metadata to make eviction decisions:
        - INTERACTIVE + DECODE: Never evict (SLO critical, KV cache vital)
        - INTERACTIVE + others: Aggressive eviction if needed
        - BATCH: Evict readily
        """
        client = self.clients[client_id]
        
        # Check if we have capacity
        if self.memory_allocated_mb + estimated_memory_mb < self.gpu_memory_mb:
            return AdmissionDecision(
                admitted=True,
                reason="Sufficient capacity available"
            )
        
        # Need to evict - use semantic eviction
        logger.info(
            f"🔄 Admission requires eviction: need {estimated_memory_mb:.1f}MB, "
            f"allocated {self.memory_allocated_mb:.1f}MB / {self.gpu_memory_mb:.1f}MB"
        )
        
        freed_mb, evicted_clients = await self._semantic_eviction(
            estimated_memory_mb,
            protect_client=client_id
        )
        
        success = freed_mb >= estimated_memory_mb
        
        return AdmissionDecision(
            admitted=success,
            reason="Made room via semantic eviction" if success else "Could not free enough memory",
            memory_freed_mb=freed_mb,
            evicted_clients=evicted_clients
        )
    
    async def _semantic_eviction(
        self,
        needed_mb: float,
        protect_client: Optional[str] = None
    ) -> tuple:
        """
        Evict based on semantic priority, not LRU.
        
        Semantic rules:
        1. Never evict INTERACTIVE clients with active requests
        2. Never evict clients in DECODE phase (KV cache critical)
        3. Evict BATCH clients first, then SERVING, then oldest INTERACTIVE
        4. Within same priority, evict clients with lowest throughput
        
        Returns: (freed_mb, evicted_client_ids)
        """
        evicted_mb = 0.0
        evicted_clients = []
        
        # Compute eviction order
        eviction_order = self._compute_eviction_order(protect_client)
        
        logger.debug(f"Semantic eviction order: {[c for c, _ in eviction_order]}")
        
        for client_id, _ in eviction_order:
            if evicted_mb >= needed_mb:
                break
            
            client = self.clients[client_id]
            
            # Semantic protection rules
            if client.priority == ClientPriority.INTERACTIVE and client.active_requests > 0:
                logger.debug(f"  ⏭️  Skipping {client_id}: INTERACTIVE with active requests")
                continue
            
            if client.phase == ExecutionPhase.LLM_DECODE:
                logger.debug(f"  ⏭️  Skipping {client_id}: in DECODE phase (KV cache critical)")
                continue
            
            # Evict this client's resources
            freed = await self._evict_client_resources(client_id)
            evicted_mb += freed
            evicted_clients.append(client_id)
            
            logger.info(
                f"🗑️  Evicted {client_id}: freed {freed:.1f}MB "
                f"(total freed: {evicted_mb:.1f}MB / {needed_mb:.1f}MB needed)"
            )
        
        self.stats['evictions_performed'] += len(evicted_clients)
        self.stats['total_memory_freed_mb'] += evicted_mb
        
        return evicted_mb, evicted_clients
    
    def _compute_eviction_order(
        self,
        protect_client: Optional[str] = None
    ) -> List[tuple]:
        """
        Compute semantic-aware eviction ordering.
        
        Score factors:
        1. Priority (BATCH < SERVING < INTERACTIVE)
        2. Phase (PREFILL can evict, DECODE should protect)
        3. Activity (idle clients should evict before active)
        4. Efficiency (low throughput clients should evict first)
        
        Returns: List of (client_id, score) tuples, lowest score first
        """
        clients_with_score = []
        
        for client_id, client in self.clients.items():
            if client_id == protect_client:
                continue
            
            score = 0.0
            
            # Priority factor (lower value = more likely to evict)
            # BATCH (2) < SERVING (1) < INTERACTIVE (0)
            score += client.priority.value * 1000
            
            # Phase factor
            if client.phase == ExecutionPhase.LLM_PREFILL:
                score += 0  # Can recompute, low cost
            elif client.phase == ExecutionPhase.LLM_DECODE:
                score += 10000  # KV cache very precious
            elif client.phase == ExecutionPhase.VISION_ENCODING:
                score += 500  # Moderate
            else:
                score += 100
            
            # Activity factor
            idle_time_sec = time.time() - client.last_active
            activity_penalty = -min(idle_time_sec * 10, 500)
            score += activity_penalty
            
            # Efficiency factor (throughput per MB)
            if client.memory_usage_mb > 0:
                throughput = client.cumulative_requests / max(1, client.memory_usage_mb)
                efficiency_bonus = -throughput * 100
                score += efficiency_bonus
            
            clients_with_score.append((client_id, score))
        
        # Sort by score (ascending = evict first)
        clients_with_score.sort(key=lambda x: x[1])
        
        return clients_with_score
    
    async def _evict_client_resources(self, client_id: str) -> float:
        """
        Evict all resources for a client.
        
        Returns: Memory freed in MB
        """
        client = self.clients[client_id]
        freed_mb = client.memory_usage_mb + client.kv_cache_size_mb
        
        # Reset client state
        client.memory_usage_mb = 0.0
        client.kv_cache_size_mb = 0.0
        
        # If we have KV manager, clear sessions
        if self.kv_manager is not None:
            try:
                await self.kv_manager.clear_client_sessions(client_id)
            except Exception as e:
                logger.warning(f"Error clearing KV sessions for {client_id}: {e}")
        
        self.memory_allocated_mb -= freed_mb
        logger.info(f"Evicted {client_id}: freed {freed_mb:.1f}MB")
        
        return freed_mb
    
    def _estimate_memory_from_inputs(self, inputs: Dict) -> float:
        """
        Estimate memory requirement from input tensors and metadata.
        
        Heuristic:
        - Input tensors: actual size
        - Intermediate activations: 3-5× input size (typical for transformers)
        - KV cache: estimated from context length
        """
        try:
            import torch
            
            total_bytes = 0
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    total_bytes += value.element_size() * value.numel()
            
            # Conservative estimate: assume 4× for intermediates
            estimated_bytes = total_bytes * 4
            estimated_mb = estimated_bytes / (1024 * 1024)
            
            return min(estimated_mb, 5000)  # Cap at 5GB
        except Exception as e:
            logger.warning(f"Error estimating memory: {e}")
            return 100.0  # Default estimate
    
    def _compute_slo(self, priority: ClientPriority, phase: ExecutionPhase) -> float:
        """Compute default SLO deadline based on priority and phase."""
        base_slos = {
            ClientPriority.INTERACTIVE: 100,   # 100ms
            ClientPriority.SERVING: 500,       # 500ms
            ClientPriority.BATCH: 5000,        # 5s
        }
        
        slo = base_slos.get(priority, 1000)
        
        # Adjust for phase
        if phase == ExecutionPhase.LLM_DECODE:
            slo *= 1.5  # Sequential, slower
        elif phase == ExecutionPhase.VISION_ENCODING:
            slo *= 0.8  # Often faster
        
        return slo
    
    async def get_next_batch(self, batch_size: int = 4) -> List[Dict]:
        """
        Get next batch of requests for execution.
        
        Groups semantically similar requests for efficient execution.
        """
        batch = []
        
        try:
            # Non-blocking: collect up to batch_size requests
            for _ in range(batch_size):
                try:
                    request = self.request_queue.get_nowait()
                    batch.append(request)
                except asyncio.QueueEmpty:
                    break
            
            if batch:
                logger.debug(f"📦 Batch ready: {len(batch)} requests")
            
            return batch
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            return []
    
    def _group_by_semantics(self, batch: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group requests by semantic similarity.
        
        Groups:
        - prefill: LLM_PREFILL phase
        - decode: LLM_DECODE phase
        - vision: VISION_* phases
        - mixed: everything else
        """
        groups = {
            'prefill': [],
            'decode': [],
            'vision': [],
            'mixed': []
        }
        
        for request in batch:
            phase = request['phase']
            
            if phase == ExecutionPhase.LLM_PREFILL:
                groups['prefill'].append(request)
            elif phase == ExecutionPhase.LLM_DECODE:
                groups['decode'].append(request)
            elif phase in [ExecutionPhase.VISION_ENCODING, ExecutionPhase.VISION_DECODING]:
                groups['vision'].append(request)
            else:
                groups['mixed'].append(request)
        
        # Filter out empty groups
        return {k: v for k, v in groups.items() if v}
    
    async def complete_request(self, request_id: str, result: object, error: Optional[str] = None):
        """Mark request as completed and wake waiting caller."""
        if request_id in self.request_results:
            future = self.request_results[request_id]
            if error:
                future.set_exception(RuntimeError(error))
            else:
                future.set_result(result)
            del self.request_results[request_id]
    
    def get_stats(self) -> Dict:
        """Get coordinator statistics."""
        return {
            **self.stats,
            'active_clients': len(self.clients),
            'queued_requests': self.request_queue.qsize(),
            'pending_results': len(self.request_results),
            'memory_allocated_mb': self.memory_allocated_mb,
            'memory_available_mb': max(0, self.gpu_memory_mb - self.memory_allocated_mb),
            'clients': {
                cid: {
                    'phase': c.phase.value,
                    'priority': c.priority.name,
                    'active_requests': c.active_requests,
                    'cumulative_requests': c.cumulative_requests,
                    'failed_requests': c.failed_requests,
                    'slo_violations': c.slo_violations,
                    'memory_usage_mb': c.memory_usage_mb,
                }
                for cid, c in self.clients.items()
            }
        }
