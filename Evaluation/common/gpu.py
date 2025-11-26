"""
GPU sampling utilities shared across evaluation scripts.

The NVML helper mirrors the logic used in earlier experiments but lives in a
dedicated module so week-5 harnesses can import it without duplicating code.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import pynvml

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pynvml = None
    _NVML_AVAILABLE = False


@dataclass
class GpuSampleSummary:
    gpu_util_pct: Optional[float] = None
    mem_util_pct: Optional[float] = None
    mem_used_mb: Optional[float] = None


class NvmlSampler:
    """Background GPU utilization sampler using NVML.

    The sampler collects per-interval utilization snapshots in a background
    thread.  It is intentionally lightweight so smoke tests on the dev L4 box
    incur minimal overhead.
    """

    def __init__(self, device_index: int, interval_s: float = 0.01) -> None:
        if not _NVML_AVAILABLE:
            raise RuntimeError("pynvml not available; install nvidia-ml-py3")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: List[Dict[str, float]] = []
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "NvmlSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        pynvml.nvmlShutdown()

    def _run(self) -> None:
        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self._samples.append(
                {
                    "gpu_util_pct": float(util.gpu),
                    "mem_util_pct": float(util.memory),
                    "mem_used_mb": float(mem.used) / (1024**2),
                    "timestamp": time.perf_counter(),
                }
            )
            self._stop.wait(self.interval_s)

    def summary(self) -> GpuSampleSummary:
        if not self._samples:
            return GpuSampleSummary()
        gpu_util = [sample["gpu_util_pct"] for sample in self._samples]
        mem_util = [sample["mem_util_pct"] for sample in self._samples]
        mem_used = [sample["mem_used_mb"] for sample in self._samples]
        return GpuSampleSummary(
            gpu_util_pct=sum(gpu_util) / len(gpu_util),
            mem_util_pct=sum(mem_util) / len(mem_util),
            mem_used_mb=sum(mem_used) / len(mem_used),
        )


__all__ = ["GpuSampleSummary", "NvmlSampler", "_NVML_AVAILABLE"]


