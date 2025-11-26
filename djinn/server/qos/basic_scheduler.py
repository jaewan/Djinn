from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, Optional


logger = logging.getLogger(__name__)


class QoSClass(str, Enum):
    """Supported QoS classes ordered from highest to lowest priority."""

    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    BATCH = "batch"

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["QoSClass"]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return None


@dataclass
class ScheduledWork:
    """Container for queued execution work."""

    coro_factory: Callable[[], Awaitable[Any]]
    future: asyncio.Future
    metadata: Dict[str, Any] = field(default_factory=dict)
    enqueued_at: float = field(default_factory=time.time)


class BasicQosScheduler:
    """
    Minimal QoS-aware scheduler.

    Provides strict priority ordering (Realtime > Interactive > Batch) with
    per-class concurrency limits and bounded total concurrency.
    """

    def __init__(
        self,
        max_concurrency: int,
        class_shares: Optional[Dict[str, float]] = None,
        escalation_delay_ms: float = 800.0,
    ):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        self._max_concurrency = max_concurrency
        self._queues: Dict[QoSClass, Deque[ScheduledWork]] = {
            qos: deque() for qos in QoSClass
        }
        self._inflight_total = 0
        self._inflight_per_class: Dict[QoSClass, int] = {qos: 0 for qos in QoSClass}
        self._lock = asyncio.Lock()

        self._class_limits = self._compute_class_limits(max_concurrency, class_shares)
        self._escalation_delay_ms = max(0.0, float(escalation_delay_ms))
        logger.info(
            "BasicQoSScheduler initialized: total_slots=%d, limits=%s",
            self._max_concurrency,
            {cls.value: limit for cls, limit in self._class_limits.items()},
        )

    @staticmethod
    def _compute_class_limits(
        max_concurrency: int, class_shares: Optional[Dict[str, float]]
    ) -> Dict[QoSClass, int]:
        """
        Convert share dictionary into concrete concurrency limits per QoS class.
        """
        if not class_shares:
            class_shares = {
                QoSClass.REALTIME.value: 0.3,
                QoSClass.INTERACTIVE.value: 0.5,
                QoSClass.BATCH.value: 0.2,
            }

        normalized: Dict[QoSClass, float] = {}
        for qos in QoSClass:
            normalized[qos] = max(
                0.0, float(class_shares.get(qos.value, 0.0))
            )

        total_weight = sum(normalized.values()) or 1.0
        limits: Dict[QoSClass, int] = {}
        assigned = 0
        for qos in QoSClass:
            weight = normalized[qos]
            limit = int(math.floor((weight / total_weight) * max_concurrency))
            limits[qos] = limit
            assigned += limit

        # Ensure at least one slot across classes by distributing the remainder
        idx = 0
        qos_order = list(QoSClass)
        while assigned < max_concurrency:
            qos = qos_order[idx % len(qos_order)]
            limits[qos] += 1
            assigned += 1
            idx += 1

        # Guarantee at least one realtime slot for latency-sensitive work
        if limits[QoSClass.REALTIME] == 0:
            limits[QoSClass.REALTIME] = 1

        return limits

    async def run(
        self,
        qos_class: QoSClass,
        coro_factory: Callable[[], Awaitable[Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Submit work for execution under the specified QoS class.
        """
        metadata = metadata or {}
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        work = ScheduledWork(coro_factory=coro_factory, future=future, metadata=metadata)

        async with self._lock:
            self._queues[qos_class].append(work)
            await self._maybe_dispatch_locked()

        return await future

    async def _maybe_dispatch_locked(self) -> None:
        while self._inflight_total < self._max_concurrency:
            next_class = self._pick_next_class_locked()
            if not next_class:
                break

            work = self._queues[next_class].popleft()
            self._inflight_total += 1
            self._inflight_per_class[next_class] += 1
            asyncio.create_task(self._execute_work(next_class, work))

    def _pick_next_class_locked(self) -> Optional[QoSClass]:
        now = time.time()
        for qos in QoSClass:
            queue = self._queues[qos]
            if not queue:
                continue
            work = queue[0]
            wait_ms = (now - work.enqueued_at) * 1000.0
            if self._inflight_per_class[qos] < self._class_limits[qos]:
                return qos
            if (
                wait_ms >= self._escalation_delay_ms
                and self._inflight_total < self._max_concurrency
            ):
                logger.debug(
                    "QoS[%s] escalation triggered after %.1fms wait",
                    qos.value,
                    wait_ms,
                )
                return qos
        # If all classes hit their per-class limits but we still have capacity,
        # relax per-class limit and schedule highest priority waiting request.
        if self._inflight_total < self._max_concurrency:
            for qos in QoSClass:
                if self._queues[qos]:
                    return qos
        return None

    async def _execute_work(self, qos_class: QoSClass, work: ScheduledWork) -> None:
        execution_start = None
        try:
            queue_latency_ms = (time.time() - work.enqueued_at) * 1000.0
            logger.debug(
                "QoS[%s] starting request_id=%s queue_latency=%.1fms",
                qos_class.value,
                work.metadata.get('request_id'),
                queue_latency_ms,
            )
            # PHASE 1.5 FIX: Store queue latency in request dict if available
            request_ref = work.metadata.get('_request_ref')
            if request_ref is not None:
                request_ref['_queue_latency_ms'] = queue_latency_ms
            # Also store in metadata for logging
            work.metadata['_queue_latency_ms'] = queue_latency_ms
            execution_start = time.time()
            result = await work.coro_factory()
            # PHASE 1.5 FIX: Ensure queue latency is in result metrics if it's a dict
            if isinstance(result, dict) and 'metrics' in result:
                result['metrics']['queue_latency_ms'] = queue_latency_ms
            if not work.future.cancelled():
                work.future.set_result(result)
        except Exception as exc:
            if not work.future.cancelled():
                work.future.set_exception(exc)
        finally:
            total_latency_ms = (time.time() - work.enqueued_at) * 1000.0
            execution_time_ms = (time.time() - execution_start) * 1000.0 if execution_start else 0.0
            logger.info(
                "QoS[%s] completed request_id=%s total_latency=%.1fms "
                "(queue=%.1fms, execution=%.1fms)",
                qos_class.value,
                work.metadata.get('request_id'),
                total_latency_ms,
                queue_latency_ms,
                execution_time_ms,
            )
            async with self._lock:
                self._inflight_total = max(0, self._inflight_total - 1)
                self._inflight_per_class[qos_class] = max(
                    0, self._inflight_per_class[qos_class] - 1
                )
                await self._maybe_dispatch_locked()

