#!/usr/bin/env python3
"""
Synthetic memory stress harness for Experiment 3.2.

Simulates concurrent sessions performing allocate/use/free cycles to validate
fragmentation and allocation latency instrumentation before hooking into real
VMU/allocator backends.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class AllocationRecord:
    session_id: int
    size_mb: float
    duration_ms: float
    start_time: float
    end_time: float


class BaseAllocator:
    def __init__(self, name: str):
        self.name = name
        self.active_allocations: List[AllocationRecord] = []

    def allocate(self, session_id: int, size_mb: float, hold_ms: float) -> AllocationRecord:
        start = time.perf_counter()
        time.sleep(0.0005)  # simulate allocation latency
        end = time.perf_counter()
        record = AllocationRecord(session_id, size_mb, hold_ms, start, end)
        self.active_allocations.append(record)
        return record

    def free(self, record: AllocationRecord):
        time.sleep(0.0002)
        self.active_allocations.remove(record)

    def fragmentation(self) -> float:
        return 0.0

    def alloc_latency_us(self) -> float:
        if not self.active_allocations:
            return 0.0
        return sum((rec.end_time - rec.start_time) for rec in self.active_allocations) / len(self.active_allocations) * 1e6


class VmuAllocator(BaseAllocator):
    def __init__(self):
        super().__init__("vmu")

    def allocate(self, session_id: int, size_mb: float, hold_ms: float) -> AllocationRecord:
        start = time.perf_counter()
        time.sleep(0.00001)
        end = time.perf_counter()
        record = AllocationRecord(session_id, size_mb, hold_ms, start, end)
        self.active_allocations.append(record)
        return record


class CudaAllocator(BaseAllocator):
    def __init__(self):
        super().__init__("cuda")

    def allocate(self, session_id: int, size_mb: float, hold_ms: float) -> AllocationRecord:
        start = time.perf_counter()
        time.sleep(0.0002 + size_mb * 0.00001)
        end = time.perf_counter()
        record = AllocationRecord(session_id, size_mb, hold_ms, start, end)
        self.active_allocations.append(record)
        return record

    def fragmentation(self) -> float:
        return min(0.35, len(self.active_allocations) * 0.02)


class TorchAllocator(BaseAllocator):
    def __init__(self):
        super().__init__("torch")

    def allocate(self, session_id: int, size_mb: float, hold_ms: float) -> AllocationRecord:
        start = time.perf_counter()
        time.sleep(0.00005 + size_mb * 0.000005)
        end = time.perf_counter()
        record = AllocationRecord(session_id, size_mb, hold_ms, start, end)
        self.active_allocations.append(record)
        return record

    def fragmentation(self) -> float:
        return min(0.2, len(self.active_allocations) * 0.01)


ALLOCATORS = {
    "vmu": VmuAllocator,
    "cuda": CudaAllocator,
    "torch": TorchAllocator,
}


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return yaml.safe_load(f)


def simulate_sessions(cfg: Dict, allocator_name: str, duration_s: float) -> Dict:
    allocator_cls = ALLOCATORS[allocator_name]
    allocator = allocator_cls()

    workload = cfg["session_workload"]
    num_sessions = workload["num_sessions"]
    ops_per_session = workload["operations_per_session"]
    alloc_sizes = workload["alloc_sizes_mb"]
    reuse_prob = workload["reuse_probability"]
    hold_ms = workload["hold_time_ms"]

    timeline = []
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < duration_s:
        for session_id in range(num_sessions):
            active = [rec for rec in allocator.active_allocations if rec.session_id == session_id]
            if active and random.random() < reuse_prob:
                continue
            if len(active) >= ops_per_session:
                oldest = active[0]
                allocator.free(oldest)
            size_mb = random.choice(alloc_sizes)
            record = allocator.allocate(session_id, size_mb, hold_ms)
            timeline.append(
                {
                    "session_id": session_id,
                    "size_mb": size_mb,
                    "alloc_latency_us": (record.end_time - record.start_time) * 1e6,
                    "timestamp": time.perf_counter(),
                    "fragmentation": allocator.fragmentation(),
                }
            )

    return {
        "allocator": allocator_name,
        "events": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allocator", choices=ALLOCATORS.keys(), required=True)
    parser.add_argument("--duration", type=int, default=10, help="Duration seconds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = simulate_sessions(cfg, args.allocator, args.duration)
    payload = {
        "allocator": args.allocator,
        "duration_s": args.duration,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        **results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved memory stress results to {args.output}")


if __name__ == "__main__":
    main()


