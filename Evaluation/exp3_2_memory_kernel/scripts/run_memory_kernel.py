#!/usr/bin/env python3
"""
Real memory-kernel stress harness for Experiment 3.2.

Runs concurrent session churn against either the Djinn VMU allocator or a
PyTorch baseline allocator, capturing allocation latency and fragmentation
metrics for the OSDI plots.
"""

from __future__ import annotations

import argparse
import json
import time
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Evaluation.common.djinn_init import check_gpu_memory
from djinn.backend.runtime.unified_vmu import UnifiedVMU

MB = 1024 * 1024


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.loads(handle.read()) if path.suffix == ".json" else _load_yaml(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open() as handle:
        return yaml.safe_load(handle)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(100.0, pct))
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return sorted(values)[idx]


class MemoryKernelRunner:
    def __init__(
        self,
        workload_cfg: Dict[str, Any],
        *,
        duration_s: float,
        device: int,
        seed: int,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the memory-kernel experiment")

        self.device_idx = device
        torch.cuda.set_device(device)
        check_gpu_memory()

        self.workload = workload_cfg
        self.duration_s = max(duration_s, 1.0)
        self.max_sessions = max(int(self.workload.get("num_sessions", 1)), 1)
        self.ops_per_session = max(int(self.workload.get("operations_per_session", 1)), 1)
        self.alloc_sizes_mb = self.workload.get("alloc_sizes_mb", [1])
        self.reuse_probability = float(self.workload.get("reuse_probability", 0.0))
        self.hold_time_ms = float(self.workload.get("hold_time_ms", 0.0))
        if self.hold_time_ms <= 0:
            raise ValueError("session_workload.hold_time_ms must be > 0 to model live compute pressure")
        self.metrics_sample_interval = max(int(self.workload.get("metrics_sample_interval", 100)), 1)
        self.min_alloc_bytes = int(min(self.alloc_sizes_mb) * MB)
        self.max_session_mb = int(
            self.workload.get(
                "max_allocation_mb",
                self.ops_per_session * max(self.alloc_sizes_mb),
            )
        )
        self.rng = random.Random(seed)
        self.max_events = self.workload.get(
            "max_events",
            self.max_sessions * self.ops_per_session * 4,
        )

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _simulate_hold_work(self) -> None:
        if self.hold_time_ms <= 0:
            return
        time.sleep(self.hold_time_ms / 1000.0)

    def _maybe_record_metrics(
        self,
        op_counter: int,
        events: List[Dict[str, Any]],
        sampler: Callable[[], Optional[Dict[str, Any]]],
    ) -> None:
        if self.metrics_sample_interval <= 0:
            return
        if op_counter % self.metrics_sample_interval != 0:
            return
        sample = sampler()
        if sample:
            events.append(sample)


class VMUMemoryKernelRunner(MemoryKernelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmu = UnifiedVMU(device_id=self.device_idx)
        data_capacity = getattr(getattr(self.vmu, "data_segment", None), "capacity", None)
        if data_capacity:
            per_session_cap = max(int(data_capacity / max(self.max_sessions * 4, 1)), MB)
            self.session_quota_bytes = min(per_session_cap, self.max_session_mb * MB)
        else:
            self.session_quota_bytes = self.max_session_mb * MB
        data_segment = getattr(self.vmu, "data_segment", None)
        self.arena_alignment = getattr(data_segment, "alignment", 256)

    def run(self) -> Dict[str, Any]:
        start = time.perf_counter()
        events: List[Dict[str, Any]] = []
        session_stats: List[Dict[str, Any]] = []
        active_sessions: Dict[str, Dict[str, Any]] = {}
        op_counter = 0

        for session_idx in range(self.max_sessions):
            session_id = f"vmu_session_{session_idx}"
            session_info = self._start_session(session_id)
            active_sessions[session_id] = session_info
            events.append(session_info["reserve_event"])

        while (time.perf_counter() - start) < self.duration_s and len(events) < self.max_events:
            for session_id in list(active_sessions.keys())[::-1]:
                session_info = active_sessions[session_id]
                arena_used = self._arena_used(session_id)
                op_counter += 1
                if session_info["allocs_done"] >= self.ops_per_session:
                    stats = self._finish_session(session_id)
                    session_stats.append(stats)
                    session_info["allocs_done"] = 0
                    session_info["bytes_allocated"] = self._arena_used(session_id)
                    self._maybe_record_metrics(op_counter, events, self._collect_vmu_metrics)
                    continue
                if (
                    session_info["bytes_allocated"] > 0
                    and self.rng.random() < self.reuse_probability
                ):
                    self._maybe_record_metrics(op_counter, events, self._collect_vmu_metrics)
                    continue
                if arena_used + self.min_alloc_bytes > self.session_quota_bytes:
                    stats = self._finish_session(session_id)
                    session_stats.append(stats)
                    session_info["allocs_done"] = 0
                    session_info["bytes_allocated"] = self._arena_used(session_id)
                    self._maybe_record_metrics(op_counter, events, self._collect_vmu_metrics)
                    continue
                allocation_event = self._allocate(session_id, session_info)
                events.append(allocation_event)
                self._simulate_hold_work()
                self._maybe_record_metrics(op_counter, events, self._collect_vmu_metrics)

        final_metrics = self.vmu.get_metrics().to_dict()
        summary = summarize_run(events, session_stats, final_metrics)
        return {
            "allocator": "vmu",
            "device": f"cuda:{self.device_idx}",
            "duration_s": time.perf_counter() - start,
            "events": events,
            "session_metrics": session_stats,
            "summary": summary,
            "final_metrics": final_metrics,
        }

    def _start_session(self, session_id: str) -> Dict[str, Any]:
        size_bytes = self.session_quota_bytes
        t0 = time.perf_counter()
        self.vmu.reserve_session_arena(session_id, size_bytes)
        latency_us = (time.perf_counter() - t0) * 1e6
        return {
            "allocs_done": 0,
            "bytes_allocated": 0,
            "reserve_event": {
                "kind": "reserve",
                "session_id": session_id,
                "size_bytes": size_bytes,
                "latency_us": latency_us,
                "timestamp": time.time(),
            },
        }

    def _allocate(self, session_id: str, session_info: Dict[str, Any]) -> Dict[str, Any]:
        arena_used = self._arena_used(session_id)
        size_mb = self.rng.choice(self.alloc_sizes_mb)
        size_bytes = int(size_mb * MB)
        if arena_used + size_bytes > self.session_quota_bytes:
            size_bytes = max(self.min_alloc_bytes, self.session_quota_bytes - arena_used)
            size_mb = size_bytes / MB
        t0 = time.perf_counter()
        offset = self.vmu.allocate_session_data(session_id, size_bytes, name=f"kv_{session_id}")
        latency_us = (time.perf_counter() - t0) * 1e6
        session_info["allocs_done"] += 1
        session_info["bytes_allocated"] = self._arena_used(session_id)
        return {
            "kind": "alloc",
            "session_id": session_id,
            "size_mb": size_mb,
            "size_bytes": size_bytes,
            "offset": offset,
            "latency_us": latency_us,
            "timestamp": time.time(),
        }

    def _finish_session(self, session_id: str) -> Dict[str, Any]:
        metrics = self.vmu.get_metrics().to_dict()
        arena = self.vmu.data_segment.sessions.get(session_id)
        if arena:
            arena.used = 0
        return {
            "session_id": session_id,
            "data_reserved_bytes": metrics["data_reserved_bytes"],
            "data_internal_waste_bytes": metrics["data_internal_waste_bytes"],
            "data_external_gap_bytes": metrics["data_external_gap_bytes"],
            "active_sessions": metrics["active_sessions"],
            "timestamp": time.time(),
        }

    def _arena_used(self, session_id: str) -> int:
        arena = self.vmu.data_segment.sessions.get(session_id)
        return arena.used if arena else 0

    def _collect_vmu_metrics(self) -> Optional[Dict[str, Any]]:
        metrics = self.vmu.get_metrics().to_dict()
        reserved = metrics.get("data_reserved_bytes", 0)
        capacity = metrics.get("data_capacity_bytes", 0)
        external_gap = metrics.get("data_external_gap_bytes", 0)
        internal_waste = metrics.get("data_internal_waste_bytes", 0)
        return {
            "kind": "metrics",
            "allocator": "vmu",
            "timestamp": time.time(),
            "data_reserved_bytes": reserved,
            "data_capacity_bytes": capacity,
            "data_external_gap_bytes": external_gap,
            "data_internal_waste_bytes": internal_waste,
            "fragmentation_external": (external_gap / reserved) if reserved else 0.0,
            "fragmentation_internal": (internal_waste / reserved) if reserved else 0.0,
            "data_utilization": (reserved / capacity) if capacity else 0.0,
            "active_sessions": metrics.get("active_sessions"),
        }


class TorchMemoryKernelRunner(MemoryKernelRunner):
    def run(self) -> Dict[str, Any]:
        start = time.perf_counter()
        events: List[Dict[str, Any]] = []
        session_stats: List[Dict[str, Any]] = []
        active_sessions: Dict[str, Dict[str, Any]] = {}
        max_session_bytes = self.max_session_mb * MB
        op_counter = 0

        for session_idx in range(self.max_sessions):
            session_id = f"torch_session_{session_idx}"
            active_sessions[session_id] = {
                "allocs_done": 0,
                "allocations": [],
                "bytes_allocated": 0,
            }

        while (time.perf_counter() - start) < self.duration_s and len(events) < self.max_events:
            for session_id in list(active_sessions.keys()):
                session_info = active_sessions[session_id]
                op_counter += 1

                if session_info["allocs_done"] >= self.ops_per_session or session_info["bytes_allocated"] >= max_session_bytes:
                    stats = self._finish_session(session_id, session_info)
                    session_stats.append(stats)
                    self._maybe_record_metrics(op_counter, events, self._collect_torch_metrics)
                    continue

                if session_info["bytes_allocated"] > 0 and self.rng.random() < self.reuse_probability:
                    self._maybe_record_metrics(op_counter, events, self._collect_torch_metrics)
                    continue

                action = "alloc"
                if session_info["allocations"]:
                    if self.rng.random() < 0.4:
                        action = "free_one"

                if action == "free_one":
                    self._free_one_allocation(session_info)
                    self._maybe_record_metrics(op_counter, events, self._collect_torch_metrics)
                    continue

                size_mb = self.rng.choice(self.alloc_sizes_mb)
                size_bytes = int(size_mb * MB)
                if session_info["bytes_allocated"] + size_bytes > max_session_bytes:
                    stats = self._finish_session(session_id, session_info)
                    session_stats.append(stats)
                    self._maybe_record_metrics(op_counter, events, self._collect_torch_metrics)
                    continue

                try:
                    t0 = time.perf_counter()
                    tensor = torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{self.device_idx}")
                    latency_us = (time.perf_counter() - t0) * 1e6
                except torch.cuda.OutOfMemoryError as exc:
                    events.append(
                        {
                            "kind": "oom",
                            "session_id": session_id,
                            "requested_bytes": size_bytes,
                            "timestamp": time.time(),
                            "message": str(exc),
                        }
                    )
                    torch.cuda.empty_cache()
                    self._maybe_record_metrics(op_counter, events, self._collect_torch_metrics)
                    continue

                session_info["allocs_done"] += 1
                session_info["bytes_allocated"] += size_bytes
                session_info["allocations"].append({"tensor": tensor, "size_bytes": size_bytes})
                events.append(
                    {
                        "kind": "alloc",
                        "session_id": session_id,
                        "size_mb": size_mb,
                        "size_bytes": size_bytes,
                        "latency_us": latency_us,
                        "timestamp": time.time(),
                    }
                )
                self._simulate_hold_work()
                self._maybe_record_metrics(op_counter, events, self._collect_torch_metrics)

        torch.cuda.synchronize(self.device_idx)
        final_metrics = torch_allocator_metrics(self.device_idx)
        summary = summarize_run(events, session_stats, final_metrics)
        return {
            "allocator": "torch",
            "device": f"cuda:{self.device_idx}",
            "duration_s": time.perf_counter() - start,
            "events": events,
            "session_metrics": session_stats,
            "summary": summary,
            "final_metrics": final_metrics,
        }

    def _free_one_allocation(self, session_info: Dict[str, Any]) -> None:
        idx = self.rng.randint(0, len(session_info["allocations"]) - 1)
        entry = session_info["allocations"].pop(idx)
        session_info["bytes_allocated"] = max(session_info["bytes_allocated"] - entry["size_bytes"], 0)
        del entry["tensor"]

    def _finish_session(self, session_id: str, session_info: Dict[str, Any]) -> Dict[str, Any]:
        for entry in session_info["allocations"]:
            del entry["tensor"]
        session_info["allocations"].clear()
        session_info["bytes_allocated"] = 0
        session_info["allocs_done"] = 0
        torch.cuda.synchronize(self.device_idx)
        metrics = torch_allocator_metrics(self.device_idx)
        return {
            "session_id": session_id,
            "torch_allocated_bytes": metrics["allocated_bytes"],
            "torch_reserved_bytes": metrics["reserved_bytes"],
            "allocator_fragmentation_ratio": metrics["fragmentation_ratio"],
            "driver_fragmentation_ratio": metrics.get("driver_fragmentation_ratio"),
            "timestamp": time.time(),
        }

    def _collect_torch_metrics(self) -> Optional[Dict[str, Any]]:
        metrics = torch_allocator_metrics(self.device_idx)
        return {
            "kind": "metrics",
            "allocator": "torch",
            "timestamp": time.time(),
            "allocated_bytes": metrics["allocated_bytes"],
            "reserved_bytes": metrics["reserved_bytes"],
            "allocator_fragmentation_ratio": metrics["fragmentation_ratio"],
            "driver_free_bytes": metrics.get("driver_free_bytes"),
            "driver_total_bytes": metrics.get("driver_total_bytes"),
            "driver_fragmentation_ratio": metrics.get("driver_fragmentation_ratio"),
        }


def torch_allocator_metrics(device_idx: int) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_idx}")
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    frag = 0.0
    if reserved > 0:
        frag = max(reserved - allocated, 0) / reserved
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    driver_frag = 1.0 - (free_bytes / total_bytes) if total_bytes else 0.0
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "fragmentation_ratio": frag,
        "driver_free_bytes": free_bytes,
        "driver_total_bytes": total_bytes,
        "driver_fragmentation_ratio": driver_frag,
    }


def summarize_run(
    events: List[Dict[str, Any]],
    session_metrics: List[Dict[str, Any]],
    final_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    latencies = [evt["latency_us"] for evt in events if evt["kind"] == "alloc"]
    summary = {
        "num_events": len(events),
        "avg_latency_us": mean(latencies) if latencies else 0.0,
        "p50_latency_us": percentile(latencies, 50.0),
        "p95_latency_us": percentile(latencies, 95.0),
        "p99_latency_us": percentile(latencies, 99.0),
    }
    if "data_capacity_bytes" in final_metrics and final_metrics.get("data_capacity_bytes"):
        reserved = final_metrics.get("data_reserved_bytes", 0)
        capacity = final_metrics["data_capacity_bytes"]
        summary["data_utilization_pct"] = round((reserved / capacity) * 100.0, 2)
    if "fragmentation_ratio" in final_metrics:
        summary["fragmentation_ratio"] = final_metrics["fragmentation_ratio"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Workload config YAML/JSON")
    parser.add_argument("--allocator", choices=["vmu", "torch"], required=True)
    parser.add_argument("--duration", type=float, default=30.0, help="Run duration (seconds)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    workload_cfg = cfg.get("session_workload", cfg)

    if args.allocator == "vmu":
        runner = VMUMemoryKernelRunner(
            workload_cfg,
            duration_s=args.duration,
            device=args.device,
            seed=args.seed,
        )
    else:
        runner = TorchMemoryKernelRunner(
            workload_cfg,
            duration_s=args.duration,
            device=args.device,
            seed=args.seed,
        )

    results = runner.run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved memory kernel results to {args.output}")


if __name__ == "__main__":
    main()

