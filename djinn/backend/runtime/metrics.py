from __future__ import annotations

import logging
from typing import Dict, Any

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, start_http_server
except Exception:  # pragma: no cover - optional dependency
    Counter = Gauge = Histogram = CollectorRegistry = None  # type: ignore
    def start_http_server(*args, **kwargs):  # type: ignore
        return None

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Prometheus metrics exporter (Python-side simplification for Phase 2.2).

    Exposes key counters/gauges for the transport layer via HTTP; if dependencies
    are not available, it becomes a no-op to keep the system running.
    """

    def __init__(self, port: int = 9095):
        self._enabled = Counter is not None and Gauge is not None and Histogram is not None
        self._port = port
        self._server_started = False
        self._registry = CollectorRegistry() if self._enabled else None

        if self._enabled:
            self.packets_total = Counter(
                "genie_packets_total", "Total packets processed", ["direction"], registry=self._registry
            )
            self.bytes_total = Counter(
                "genie_bytes_total", "Total bytes processed", ["direction"], registry=self._registry
            )
            self.transfers_active = Gauge(
                "genie_transfers_active", "Active transfers count", registry=self._registry
            )
            self.transfer_latency = Histogram(
                "genie_transfer_latency_seconds",
                "Transfer latency distribution",
                buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
                registry=self._registry,
            )
        else:
            self.packets_total = self.bytes_total = self.transfers_active = self.transfer_latency = None

    def start(self):
        if self._enabled and not self._server_started:
            try:
                start_http_server(self._port, registry=self._registry)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on :{self._port}")
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")

    def record_packet(self, direction: str, bytes_count: int):
        if not self._enabled:
            return
        try:
            self.packets_total.labels(direction=direction).inc()
            self.bytes_total.labels(direction=direction).inc(bytes_count)
        except Exception:
            pass

    def set_active_transfers(self, count: int):
        if not self._enabled:
            return
        try:
            self.transfers_active.set(count)
        except Exception:
            pass

    def observe_transfer_latency(self, seconds: float):
        if not self._enabled:
            return
        try:
            self.transfer_latency.observe(seconds)
        except Exception:
            pass

    def dump_snapshot(self) -> Dict[str, Any]:
        # Lightweight snapshot for diagnostics even if Prometheus not enabled
        return {
            "enabled": self._enabled,
            "port": self._port,
            "server_started": self._server_started,
        }


