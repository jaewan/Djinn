#!/usr/bin/env python3
"""
Experiment 4.2 QoS contention harness.

Simulates Poisson arrivals for multiple QoS classes and compares FCFS vs
priority scheduling with per-class concurrency shares. Real Djinn driver can
replace the simulator later; for now we use a fast event simulation so we can
validate the curves and produce Figure 11 inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class Request:
    qos: str
    arrival_ms: float
    service_ms: float


@dataclass
class CompletionRecord:
    qos: str
    arrival_ms: float
    completion_ms: float

    @property
    def latency_ms(self) -> float:
        return self.completion_ms - self.arrival_ms


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def expovariate(rate: float, rng: random.Random) -> float:
    return rng.expovariate(rate) if rate > 0 else float("inf")


class QoSSimulator:
    def __init__(self, cfg: Dict[str, Any], seed: int):
        self.cfg = cfg
        self.duration_s = float(cfg["simulation"]["duration_s"])
        self.rng = random.Random(seed)
        self.max_concurrency = int(cfg["system"]["max_concurrency"])
        self.base_latency_ms = float(cfg["system"]["base_latency_ms"])
        self.capacity_tokens_per_s = float(cfg["system"]["gpu_capacity_tokens_per_s"])
        self.jitter_ms = float(cfg["simulation"]["latency_jitter_ms"])
        self.qos_cfg = cfg["workload_mix"]
        self.qos_shares = cfg["system"]["qos_class_shares"]

    def _service_time_ms(self, tokens: int) -> float:
        gpu_ms = (tokens / self.capacity_tokens_per_s) * 1000.0
        jitter = self.rng.gauss(0, self.jitter_ms)
        return max(1.0, self.base_latency_ms + gpu_ms + jitter)

    def _generate_arrivals(self) -> List[Tuple[float, str, float]]:
        arrivals = []
        for qos_name, params in self.qos_cfg.items():
            rate = float(params["arrival_rate_rps"])
            t = 0.0
            while t < self.duration_s:
                delta = expovariate(rate, self.rng)
                t += delta
                if t >= self.duration_s:
                    break
                tokens = int(params["payload_tokens"])
                service_ms = self._service_time_ms(tokens)
                arrivals.append((t * 1000.0, qos_name, service_ms))
        arrivals.sort(key=lambda x: x[0])
        return arrivals

    def run(self, policy: str) -> Dict[str, Any]:
        arrivals = self._generate_arrivals()
        active: List[Tuple[float, Request]] = []  # (completion_ms, request)
        completions: List[CompletionRecord] = []
        queues: Dict[str, List[Request]] = {q: [] for q in self.qos_cfg}
        active_counts: Dict[str, int] = {q: 0 for q in self.qos_cfg}

        current_time_ms = 0.0
        idx = 0

        while idx < len(arrivals) or active:
            next_arrival_time = arrivals[idx][0] if idx < len(arrivals) else float("inf")
            next_completion_time = active[0][0] if active else float("inf")

            if next_arrival_time <= next_completion_time:
                current_time_ms = next_arrival_time
                _, qos_name, service_ms = arrivals[idx]
                idx += 1
                queues[qos_name].append(Request(qos=qos_name, arrival_ms=current_time_ms, service_ms=service_ms))
                self._dispatch(policy, queues, active, active_counts, current_time_ms)
            else:
                current_time_ms = next_completion_time
                completion_time, req = active.pop(0)
                active_counts[req.qos] -= 1
                completions.append(
                    CompletionRecord(
                        qos=req.qos,
                        arrival_ms=req.arrival_ms,
                        completion_ms=completion_time,
                    )
                )
                # After completion, try to dispatch new requests
                self._dispatch(policy, queues, active, active_counts, current_time_ms)

        metrics = self._aggregate_metrics(completions)
        metrics["policy"] = policy
        metrics["duration_s"] = self.duration_s
        return metrics

    def _dispatch(
        self,
        policy: str,
        queues: Dict[str, List[Request]],
        active: List[Tuple[float, Request]],
        active_counts: Dict[str, int],
        now_ms: float,
    ) -> None:
        while len(active) < self.max_concurrency:
            next_request = self._select_request(policy, queues, active_counts)
            if next_request is None:
                break
            completion_ms = now_ms + next_request.service_ms
            active.append((completion_ms, next_request))
            active.sort(key=lambda x: x[0])
            active_counts[next_request.qos] += 1

    def _select_request(
        self,
        policy: str,
        queues: Dict[str, List[Request]],
        active_counts: Dict[str, int],
    ) -> Optional[Request]:
        non_empty = [q for q, items in queues.items() if items]
        if not non_empty:
            return None

        if policy == "fcfs":
            earliest_qos = min(non_empty, key=lambda q: queues[q][0].arrival_ms)
            return queues[earliest_qos].pop(0)

        if policy == "qos":
            ordered = sorted(non_empty, key=lambda q: self.qos_cfg[q]["priority"], reverse=True)
            for qos in ordered:
                share = self.qos_shares.get(qos, 0.0)
                allowed = max(1, math.floor(share * self.max_concurrency))
                if active_counts[qos] < allowed:
                    return queues[qos].pop(0)
            # Fallback to FCFS if all shares saturated
            earliest_qos = min(non_empty, key=lambda q: queues[q][0].arrival_ms)
            return queues[earliest_qos].pop(0)

        raise ValueError(f"Unknown policy {policy}")

    def _aggregate_metrics(self, records: List[CompletionRecord]) -> Dict[str, Any]:
        buckets: Dict[str, List[float]] = {q: [] for q in self.qos_cfg}
        for rec in records:
            buckets[rec.qos].append(rec.latency_ms)

        per_class = {}
        for qos, samples in buckets.items():
            if not samples:
                continue
            per_class[qos] = {
                "count": len(samples),
                "latency_ms": {
                    "p50": statistics.median(samples),
                    "p90": percentile(samples, 90),
                    "p99": percentile(samples, 99),
                },
                "sla_violation_rate": self._sla_violation_rate(qos, samples),
            }

        return {
            "per_class": per_class,
            "total_requests": sum(len(v) for v in buckets.values()),
        }

    def _sla_violation_rate(self, qos: str, samples: List[float]) -> Optional[float]:
        target = self.qos_cfg[qos].get("target_p99_ms")
        if not target:
            return None
        violations = sum(1 for s in samples if s > target)
        return violations / len(samples) if samples else None


def percentile(samples: List[float], p: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    d0 = ordered[int(f)] * (c - k)
    d1 = ordered[int(c)] * (k - f)
    return d0 + d1


def run_simulation(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    simulator = QoSSimulator(cfg, seed=args.seed)
    metrics = simulator.run(args.policy)
    return {
        "metadata": {
            "driver": args.driver,
            "policy": args.policy,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "config_path": str(args.config),
            "duration_s": cfg["simulation"]["duration_s"],
        },
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--driver", choices=["synthetic", "djinn"], default="synthetic")
    parser.add_argument("--policy", choices=["fcfs", "qos"], default="qos")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.driver != "synthetic":
        raise NotImplementedError("Djinn driver not yet wired. Use --driver synthetic for now.")
    payload = run_simulation(args, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved QoS study results to {args.output}")


if __name__ == "__main__":
    main()

