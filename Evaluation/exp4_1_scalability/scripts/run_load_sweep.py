#!/usr/bin/env python3
"""
Experiment 4.1 load sweep harness.

Supports:
  * Synthetic driver – analytic model to validate harness wiring quickly.
  * Djinn driver (stub) – hooks into EnhancedModelManager for real runs.

Outputs JSON lines describing throughput, latency, and utilization per
concurrency level so Figure 10 can be rendered directly from the data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.djinn_init import ensure_initialized_before_async

from Evaluation.common.djinn_init import ensure_initialized

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class LoadPoint:
    concurrency: int
    request_rate_per_user: float
    duration_s: float


@dataclass
class LoadResult:
    concurrency: int
    requests_total: int
    tokens_per_request: int
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float
    throughput_tokens_per_s: float
    gpu_util_pct: Optional[float]
    efficiency_pct: Optional[float]
    notes: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "requests_total": self.requests_total,
            "tokens_per_request": self.tokens_per_request,
            "latency_ms": {
                "p50": self.latency_ms_p50,
                "p90": self.latency_ms_p90,
                "p99": self.latency_ms_p99,
            },
            "throughput_tokens_per_s": self.throughput_tokens_per_s,
            "gpu_util_pct": self.gpu_util_pct,
            "efficiency_pct": self.efficiency_pct,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


class SyntheticLoadDriver:
    """Analytic model that approximates the expected curves."""

    def __init__(
        self,
        base_latency_ms: float,
        saturation_concurrency: int,
        max_gpu_util_pct: float,
        tokens_per_request: int,
        rng: random.Random,
    ):
        self.base_latency_ms = base_latency_ms
        self.saturation = max(1, saturation_concurrency)
        self.max_gpu_util = max_gpu_util_pct
        self.tokens_per_request = tokens_per_request
        self.rng = rng

    async def run_point(self, point: LoadPoint) -> LoadResult:
        await asyncio.sleep(0)  # Yield control, keeps event loop honest

        load_factor = point.concurrency / self.saturation
        latency_multiplier = 1.0 if load_factor <= 1.0 else 1.0 + 0.65 * (load_factor - 1.0)
        added_queue_ms = max(0.0, (load_factor - 1.0) * self.base_latency_ms * 0.8)
        latency_samples = self._sample_latencies(point.concurrency, latency_multiplier, added_queue_ms)

        requests_total = max(
            1,
            round(point.concurrency * point.request_rate_per_user * point.duration_s),
        )
        throughput_tokens = requests_total * self.tokens_per_request
        throughput_tokens_per_s = throughput_tokens / point.duration_s

        gpu_util = min(self.max_gpu_util, self.max_gpu_util * min(load_factor, 1.0))
        efficiency = min(100.0, gpu_util / self.max_gpu_util * 100.0)
        notes = None
        if load_factor > 1.25:
            notes = "Queue saturation region"

        return LoadResult(
            concurrency=point.concurrency,
            requests_total=requests_total,
            tokens_per_request=self.tokens_per_request,
            latency_ms_p50=statistics.median(latency_samples),
            latency_ms_p90=self._percentile(latency_samples, 90),
            latency_ms_p99=self._percentile(latency_samples, 99),
            throughput_tokens_per_s=throughput_tokens_per_s * min(1.0, 1.0 / load_factor + 0.05),
            gpu_util_pct=gpu_util,
            efficiency_pct=efficiency,
            notes=notes,
        )

    def _sample_latencies(self, concurrency: int, multiplier: float, added_queue_ms: float) -> List[float]:
        samples = []
        base = self.base_latency_ms * multiplier
        sigma = base * 0.1
        for _ in range(max(50, concurrency * 5)):
            val = self.rng.gauss(base, sigma) + added_queue_ms
            samples.append(max(1.0, val))
        return samples

    @staticmethod
    def _percentile(samples: Iterable[float], p: float) -> float:
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


class DjinnLoadDriver:
    """
    Executes real Djinn requests via EnhancedModelManager so the load sweep
    reflects end-to-end scheduler behavior.
    """

    def __init__(self, cfg: Dict[str, Any], args: argparse.Namespace):
        from djinn.core.enhanced_model_manager import EnhancedModelManager  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.cfg = cfg
        self.args = args
        self._torch = torch
        self._ModelCls = AutoModelForCausalLM
        self._TokenizerCls = AutoTokenizer
        self._ManagerCls = EnhancedModelManager

        self.model_id = cfg["model"]["name"]
        self.tokenizer_id = cfg["model"].get("tokenizer", self.model_id)
        self.new_tokens = int(cfg["model"].get("new_tokens", 50))
        self.prompt_path = cfg["model"].get("prompt_path")
        self.prompt_text = cfg["model"].get("prompt_text")
        self.prompt_length = int(cfg["model"].get("prompt_length", 72))
        self.qos_defaults = cfg.get("driver_defaults", {}).get("djinn", {})
        self._manager = None
        self._model = None
        self._tokenizer = None
        self._input_template: Dict[str, Any] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        max_inflight = args.djinn_max_inflight or self.qos_defaults.get("max_inflight")
        self._semaphore = asyncio.Semaphore(max_inflight or cfg.get("system", {}).get("max_concurrency", 16))

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            # Ensure coordinator is available before creating manager
            from djinn.backend.runtime.initialization import get_coordinator as get_rt_coord
            coordinator = get_rt_coord()
            if coordinator is None:
                try:
                    from djinn.core.coordinator import get_coordinator
                    coordinator = get_coordinator()
                except RuntimeError:
                    raise RuntimeError(
                        "Djinn coordinator unavailable. Ensure djinn.init() was called "
                        "before running load sweep with --driver djinn."
                    )
            self._manager = self._ManagerCls(coordinator=coordinator, server_address=self.args.djinn_endpoint)
            if self._manager.coordinator is None:
                # Set coordinator if it wasn't set during init
                self._manager.coordinator = coordinator
            dtype_name = (self.args.djinn_dtype or "float16").lower()
            torch_dtype = {
                "float16": self._torch.float16,
                "bfloat16": self._torch.bfloat16,
                "float32": self._torch.float32,
            }[dtype_name]

            self._model = self._ModelCls.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            # Keep model on CPU; server holds real weights after registration.
            await self._manager.register_model(self._model, model_id=self.model_id)

            self._tokenizer = self._TokenizerCls.from_pretrained(self.tokenizer_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            prompt = self._load_prompt()
            encoded = self._tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.prompt_length,
                truncation=True,
                return_tensors="pt",
            )
            self._input_template = {
                "input_ids": encoded["input_ids"].to("cpu"),
                "attention_mask": encoded["attention_mask"].to("cpu"),
            }
            self._initialized = True

    def _load_prompt(self) -> str:
        if self.prompt_text:
            return self.prompt_text.strip()
        if self.prompt_path:
            return Path(self.prompt_path).read_text().strip()
        default_prompt = "Djinn load sweep prompt."
        return default_prompt

    async def run_point(self, point: LoadPoint) -> LoadResult:
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_running_loop()
        end_time = loop.time() + point.duration_s
        request_records: List[Dict[str, Any]] = []
        tasks = []
        for idx in range(point.concurrency):
            rng = random.Random(self.args.seed + idx)
            tasks.append(
                asyncio.create_task(
                    self._user_loop(idx, point.request_rate_per_user, end_time, rng)
                )
            )
        user_results = await asyncio.gather(*tasks)
        for records in user_results:
            request_records.extend(records)

        successful = [rec for rec in request_records if rec.get("latency_ms") is not None]
        latencies = sorted(rec["latency_ms"] for rec in successful)
        if not latencies:
            raise RuntimeError("No successful Djinn executions recorded; cannot compute metrics.")
        throughput_tokens = len(successful) * self.new_tokens
        elapsed = max(1e-6, point.duration_s)
        throughput_tokens_per_s = throughput_tokens / elapsed

        def percentile(values: List[float], pct: float) -> float:
            idx = (len(values) - 1) * (pct / 100.0)
            lower = math.floor(idx)
            upper = math.ceil(idx)
            if lower == upper:
                return values[int(idx)]
            return values[lower] * (upper - idx) + values[upper] * (idx - lower)

        latency_p50 = statistics.median(latencies)
        latency_p90 = percentile(latencies, 90)
        latency_p99 = percentile(latencies, 99)

        queue_waits = [rec["queue_wait_ms"] for rec in successful if rec.get("queue_wait_ms") is not None]
        notes = None
        if queue_waits:
            worst_queue = max(queue_waits)
            if worst_queue > 10.0:
                notes = f"Observed queue wait up to {worst_queue:.1f} ms"

        return LoadResult(
            concurrency=point.concurrency,
            requests_total=len(successful),
            tokens_per_request=self.new_tokens,
            latency_ms_p50=latency_p50,
            latency_ms_p90=latency_p90,
            latency_ms_p99=latency_p99,
            throughput_tokens_per_s=throughput_tokens_per_s,
            gpu_util_pct=None,
            efficiency_pct=None,
            notes=notes,
        )

    async def _user_loop(
        self,
        user_id: int,
        rate_per_user: float,
        end_time: float,
        rng: random.Random,
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        records: List[Dict[str, Any]] = []
        while loop.time() < end_time:
            delay = rng.expovariate(rate_per_user) if rate_per_user > 0 else end_time
            await asyncio.sleep(min(delay, max(0.0, end_time - loop.time())))
            if loop.time() >= end_time:
                break
            record = await self._issue_request(user_id)
            records.append(record)
        return records

    def _extract_logits_from_result(self, result: Any):
        """Extract logits tensor from remote execution result."""
        if self._torch.is_tensor(result):
            return result.cpu()
        if isinstance(result, dict):
            if "logits" in result and self._torch.is_tensor(result["logits"]):
                return result["logits"].cpu()
            for value in result.values():
                if self._torch.is_tensor(value):
                    return value.cpu()
        raise RuntimeError("Unable to extract logits tensor from remote execution result.")

    async def _issue_request(self, user_id: int) -> Dict[str, Any]:
        async with self._semaphore:
            start = time.perf_counter()
            generated = self._input_template["input_ids"].clone()
            mask = self._input_template["attention_mask"].clone()
            hints = self._build_hints()
            try:
                # Generate tokens one by one (like exp2_1)
                with self._torch.no_grad():
                    for _ in range(self.new_tokens):
                        inputs = {
                            "input_ids": generated,
                            "attention_mask": mask,
                        }
                        result = await self._manager.execute_model(
                            self._model,
                            inputs,
                            hints=hints,
                        )
                        logits = self._extract_logits_from_result(result)
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated = self._torch.cat([generated, next_token], dim=-1)
                        mask = self._torch.cat([mask, self._torch.ones_like(next_token)], dim=-1)
                
                latency_ms = (time.perf_counter() - start) * 1000.0
                metrics = self._manager.last_execution_metrics or {}
                queue_wait = metrics.get("queue_wait_ms")
                return {
                    "user_id": user_id,
                    "latency_ms": latency_ms,
                    "queue_wait_ms": queue_wait,
                }
            except Exception as exc:
                return {
                    "user_id": user_id,
                    "latency_ms": None,
                    "error": str(exc),
                }

    def _build_hints(self) -> Dict[str, Any]:
        hints = dict(self.qos_defaults)
        if self.args.djinn_qos_class:
            hints["qos_class"] = self.args.djinn_qos_class
        if self.args.djinn_deadline_ms is not None:
            hints["deadline_ms"] = self.args.djinn_deadline_ms
        # Drop None entries
        return {k: v for k, v in hints.items() if v is not None}


DRIVERS = {
    "synthetic": SyntheticLoadDriver,
    "djinn": DjinnLoadDriver,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def build_load_points(cfg: Dict[str, Any], duration_override: Optional[int]) -> List[LoadPoint]:
    duration_s = float(duration_override or cfg.get("duration_s", 60))
    rate = float(cfg["arrival"]["request_rate_per_user"])
    return [
        LoadPoint(concurrency=entry["concurrency"], request_rate_per_user=rate, duration_s=duration_s)
        for entry in cfg["load_levels"]
    ]


def init_driver(args: argparse.Namespace, cfg: Dict[str, Any]) -> Any:
    driver_cls = DRIVERS[args.driver]
    tokens_per_request = int(cfg["model"].get("new_tokens", 50))

    if args.driver == "synthetic":
        defaults = cfg.get("driver_defaults", {}).get("synthetic", {})
        base_latency = float(defaults.get("base_latency_ms", 15))
        saturation = int(defaults.get("saturation_concurrency", 16))
        max_gpu = float(defaults.get("max_gpu_util_pct", 86))
        rng = random.Random(args.seed)
        return driver_cls(
            base_latency_ms=base_latency,
            saturation_concurrency=saturation,
            max_gpu_util_pct=max_gpu,
            tokens_per_request=tokens_per_request,
            rng=rng,
        )

    if args.driver == "djinn":
        return driver_cls(cfg, args)

    raise ValueError(f"Unknown driver {args.driver}")


def _resolve_server_address(args: argparse.Namespace) -> Optional[str]:
    if args.djinn_endpoint:
        return args.djinn_endpoint
    return os.environ.get("GENIE_SERVER_ADDRESS")


async def run_load_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml_config(args.config)
    driver = init_driver(args, cfg)
    initialize_fn = getattr(driver, "initialize", None)
    if initialize_fn:
        maybe_coro = initialize_fn()
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro
    points = build_load_points(cfg, args.duration)

    results: List[LoadResult] = []
    for point in points:
        result = await driver.run_point(point)
        results.append(result)

    return {
        "metadata": {
            "model": cfg["model"]["name"],
            "driver": args.driver,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "duration_s": points[0].duration_s if points else args.duration,
            "config_path": str(args.config),
        },
        "results": [r.as_dict() for r in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config (see configs/load_sweep.yaml).",
    )
    parser.add_argument(
        "--driver",
        choices=DRIVERS.keys(),
        default="synthetic",
        help="Driver backend.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override test duration in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to JSON output file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="Random seed for synthetic driver.",
    )
    parser.add_argument(
        "--djinn-endpoint",
        type=str,
        default=None,
        help="Optional host:port override for Djinn coordinator.",
    )
    parser.add_argument(
        "--djinn-max-inflight",
        type=int,
        default=None,
        help="Limit concurrent in-flight Djinn requests (defaults to config max).",
    )
    parser.add_argument(
        "--djinn-qos-class",
        type=str,
        default=None,
        help="QoS class hint for Djinn driver (overrides config defaults).",
    )
    parser.add_argument(
        "--djinn-deadline-ms",
        type=int,
        default=None,
        help="Deadline hint passed to Djinn driver.",
    )
    parser.add_argument(
        "--djinn-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for Djinn driver registration.",
    )
    return parser.parse_args()


def save_results(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved load sweep results to {path}")


def main() -> None:
    args = parse_args()
    # Initialize Djinn before entering async context
    server_address = _resolve_server_address(args)
    ensure_initialized_before_async(server_address)
    payload = asyncio.run(run_load_sweep(args))
    save_results(args.output, payload)


if __name__ == "__main__":
    main()

