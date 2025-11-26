#!/usr/bin/env python3
"""
Phase 4 Production Load Test for OSDI Evaluation

Simulates production load with real workloads:
- 60% LLM users (GPT-2/GPT-J)
- 30% Vision users (ResNet-50/ViT-Base)
- 10% Multimodal users (CLIP)

Validates SLA targets:
- P99 latency < 100ms
- Error rate < 1%
- GPU utilization > 80%
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.djinn_init import ensure_initialized_before_async
from Evaluation.common.workloads import build_workload

# GPU monitoring
try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    _NVML_AVAILABLE = False


class GpuSampler:
    """Background GPU utilization sampler for load test."""
    
    def __init__(self, device_index: int = 0, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples: List[Dict[str, float]] = []
        self._thread: Optional[threading.Thread] = None
        self.handle = None
        
        if not _NVML_AVAILABLE:
            return
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            print(f"Warning: GPU monitoring unavailable: {e}")
            self.handle = None
    
    def start(self):
        """Start sampling."""
        if not _NVML_AVAILABLE or self.handle is None:
            return
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop sampling."""
        if hasattr(self, '_stop'):
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _run(self):
        """Background sampling loop."""
        if self.handle is None:
            return
        
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.samples.append({
                    "gpu_util_pct": float(util.gpu),
                    "mem_util_pct": float(util.memory),
                    "mem_used_mb": float(mem.used) / (1024**2),
                    "timestamp": time.time(),
                })
            except Exception as e:
                print(f"Warning: GPU sampling error: {e}")
            
            self._stop.wait(self.interval_s)
    
    def get_average_utilization(self) -> Optional[float]:
        """Get average GPU utilization."""
        if not self.samples:
            return None
        return statistics.mean(sample["gpu_util_pct"] for sample in self.samples)
    
    def get_average_memory_mb(self) -> Optional[float]:
        """Get average GPU memory usage."""
        if not self.samples:
            return None
        return statistics.mean(sample["mem_used_mb"] for sample in self.samples)


@dataclass
class LoadTestMetrics:
    """Aggregated metrics from load test."""
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p95: float
    latency_ms_p99: float
    
    throughput_per_s: float
    gpu_utilization_pct: Optional[float]
    gpu_utilization_samples: List[float]
    gpu_memory_used_mb: Optional[float]
    
    per_class_metrics: Dict[str, Dict[str, float]]
    
    sla_violations: Dict[str, bool]
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UserSimulator:
    """Simulates a single user making requests."""
    
    def __init__(
        self,
        user_id: int,
        user_class: str,
        workload_cfg: Dict[str, Any],
        request_pattern: Dict[str, Any],
        server_address: str,
    ):
        self.user_id = user_id
        self.user_class = user_class
        self.workload_cfg = workload_cfg
        self.request_pattern = request_pattern
        self.server_address = server_address
        
        self.requests_per_minute = request_pattern.get("requests_per_minute", 10)
        self.burst_probability = request_pattern.get("burst_probability", 0.1)
        self.interval_seconds = 60.0 / self.requests_per_minute
        
        self.workload = None
        self.manager = None
        self.metrics = []
        
    async def initialize(self):
        """Initialize workload and manager."""
        from djinn.core.enhanced_model_manager import EnhancedModelManager
        from djinn.backend.runtime.initialization import get_coordinator
        
        # Build workload
        impl = self.workload_cfg["implementation"]
        spec = self.workload_cfg["params"]
        self.workload = build_workload(impl, spec, "cpu", "float16")
        model = self.workload.model
        
        # Initialize manager
        coordinator = get_coordinator()
        if coordinator is None:
            raise RuntimeError(f"Coordinator unavailable for user {self.user_id}")
        
        from djinn.core.enhanced_model_manager import EnhancedModelManager
        self.manager = EnhancedModelManager(coordinator=coordinator, server_address=self.server_address)
        
        # Register model (share per HF ID to reuse cache)
        model_id = self.workload_cfg["params"].get("model_id", f"{self.user_class}_user_{self.user_id}")
        await self.manager.register_model(model, model_id=model_id)
        self.model_id = model_id
    
    async def run(self, duration_seconds: float):
        """Run user workload for specified duration."""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            # Check for burst
            if random.random() < self.burst_probability:
                # Burst: send 3-5 requests quickly
                burst_count = random.randint(3, 5)
                for _ in range(burst_count):
                    if time.time() >= end_time:
                        break
                    await self._make_request()
            else:
                # Normal: single request
                await self._make_request()
            
            # Wait for next interval
            await asyncio.sleep(self.interval_seconds)
    
    async def _make_request(self):
        """Make a single request and record metrics."""
        try:
            inputs = self.workload.prepare_inputs()
            
            import djinn
            hints = self.workload_cfg.get("semantic_hints", {})
            phase = hints.get("phase")
            if phase is None:
                phase = "vision" if self.user_class in {"vision_users", "multimodal_users"} else "decode"
            priority = hints.get("priority", "normal")
            kv_cache = hints.get("kv_cache_size_mb")
            expected_tokens = hints.get("expected_tokens")
            with djinn.session(
                phase=phase,
                priority=priority,
                kv_cache_size_mb=kv_cache,
                expected_tokens=expected_tokens,
            ):
                start = time.perf_counter()
                await self.manager.execute_model(
                    self.workload.model,
                    inputs,
                    model_id=self.model_id,
                )
            
            latency_ms = (time.perf_counter() - start) * 1000.0
            
            # INSTRUMENTATION: Extract detailed timing breakdown from manager
            exec_metrics = self.manager.last_execution_metrics or {}
            
            self.metrics.append({
                "user_id": self.user_id,
                "user_class": self.user_class,
                "latency_ms": latency_ms,
                "success": True,
                "timestamp": time.time(),
                # INSTRUMENTATION: Detailed timing breakdown
                "registration_time_ms": exec_metrics.get("registration_time_ms", 0.0),
                "execution_time_ms": exec_metrics.get("execution_time_ms", 0.0),
                "queue_latency_ms": exec_metrics.get("queue_latency_ms", 0.0),
                "executor_time_ms": exec_metrics.get("executor_time_ms", 0.0),
                "was_registered_during_request": exec_metrics.get("was_registered_during_request", False),
            })
        except Exception as e:
            self.metrics.append({
                "user_id": self.user_id,
                "user_class": self.user_class,
                "latency_ms": None,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            })


async def run_load_test(config_path: str, output_path: Optional[str] = None):
    """Run production load test."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    experiment_cfg = config["experiment"]
    user_classes_cfg = config["user_classes"]
    metrics_cfg = config.get("metrics", {})
    
    duration_seconds = experiment_cfg["duration_seconds"]
    server_address = experiment_cfg["djinn_server_address"]
    
    # Create user simulators
    users = []
    for class_name, class_cfg in user_classes_cfg.items():
        count = class_cfg["count"]
        workload_cfg = class_cfg["workload"]
        request_pattern = class_cfg["request_pattern"]
        
        for i in range(count):
            user = UserSimulator(
                user_id=len(users),
                user_class=class_name,
                workload_cfg=workload_cfg,
                request_pattern=request_pattern,
                server_address=server_address,
            )
            users.append(user)
    
    # PHASE 1.5 FIX: Pre-register all models during warmup with verification
    print(f"Warming up: Registering models for {len(users)} users...")
    warmup_start = time.time()
    
    # Initialize all users (register models)
    init_results = await asyncio.gather(*[user.initialize() for user in users], return_exceptions=True)
    
    # Verify all initializations succeeded
    failed_inits = [i for i, r in enumerate(init_results) if isinstance(r, Exception)]
    if failed_inits:
        print(f"❌ Warning: {len(failed_inits)} users failed to initialize:")
        for idx in failed_inits[:5]:
            print(f"  User {idx}: {init_results[idx]}")
    
    # PHASE 1.5 FIX: Verify all models are actually registered before starting
    print("Verifying model registration...")
    verification_failed = []
    for user in users:
        if user.manager and user.workload:
            fingerprint = user.manager.fingerprint.compute(
                user.workload.model,
                user.workload_cfg["params"].get("model_id")
            )
            if fingerprint not in user.manager.registered_models:
                verification_failed.append((user.user_id, fingerprint[:8]))
    
    if verification_failed:
        print(f"❌ Warning: {len(verification_failed)} models not registered after warmup:")
        for user_id, fp in verification_failed[:5]:
            print(f"  User {user_id}: {fp}")
    else:
        print(f"✅ All {len(users)} models verified as registered")
    
    warmup_time = time.time() - warmup_start
    print(f"✅ Warmup complete: All models registered in {warmup_time:.1f}s")
    
    # Start GPU monitoring
    gpu_sampler = GpuSampler(device_index=0, interval_s=metrics_cfg.get("collection_interval_seconds", 1.0))
    gpu_sampler.start()
    print(f"✅ GPU monitoring started")
    
    # Run load test
    print(f"Starting load test for {duration_seconds}s...")
    start_time = time.time()
    
    await asyncio.gather(*[user.run(duration_seconds) for user in users])
    
    elapsed = time.time() - start_time
    print(f"✅ Load test completed in {elapsed:.1f}s")
    
    # Stop GPU monitoring
    gpu_sampler.stop()
    gpu_util_avg = gpu_sampler.get_average_utilization()
    gpu_mem_avg = gpu_sampler.get_average_memory_mb()
    print(f"✅ GPU monitoring stopped (avg util: {gpu_util_avg:.1f}%, avg mem: {gpu_mem_avg:.1f}MB)")
    
    # Collect metrics
    all_metrics = []
    for user in users:
        all_metrics.extend(user.metrics)
    
    # Aggregate metrics
    successful = [m for m in all_metrics if m["success"]]
    failed = [m for m in all_metrics if not m["success"]]
    
    latencies = [m["latency_ms"] for m in successful if m["latency_ms"] is not None]
    
    # Aggregate per-class metrics
    per_class_metrics = {}
    for class_name in user_classes_cfg.keys():
        class_metrics = [m for m in all_metrics if m.get("user_class") == class_name]
        class_successful = [m for m in class_metrics if m["success"]]
        class_latencies = [m["latency_ms"] for m in class_successful if m["latency_ms"] is not None]
        
        per_class_metrics[class_name] = {
            "total_requests": len(class_metrics),
            "successful_requests": len(class_successful),
            "failed_requests": len(class_metrics) - len(class_successful),
            "error_rate": (len(class_metrics) - len(class_successful)) / len(class_metrics) if class_metrics else 0.0,
            "latency_ms_p50": statistics.median(class_latencies) if class_latencies else 0.0,
            "latency_ms_p99": statistics.quantiles(class_latencies, n=100)[98] if len(class_latencies) >= 100 else 0.0,
            "throughput_per_s": len(class_successful) / elapsed if elapsed > 0 else 0.0,
        }
    
    # Get GPU metrics
    gpu_util_avg = gpu_sampler.get_average_utilization()
    gpu_mem_avg = gpu_sampler.get_average_memory_mb()
    gpu_samples = [s["gpu_util_pct"] for s in gpu_sampler.samples]
    
    metrics = LoadTestMetrics(
        duration_seconds=elapsed,
        total_requests=len(all_metrics),
        successful_requests=len(successful),
        failed_requests=len(failed),
        error_rate=len(failed) / len(all_metrics) if all_metrics else 0.0,
        latency_ms_p50=statistics.median(latencies) if latencies else 0.0,
        latency_ms_p90=statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else 0.0,
        latency_ms_p95=statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0.0,
        latency_ms_p99=statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else 0.0,
        throughput_per_s=len(successful) / elapsed if elapsed > 0 else 0.0,
        gpu_utilization_pct=gpu_util_avg,
        gpu_utilization_samples=gpu_samples,
        gpu_memory_used_mb=gpu_mem_avg,
        per_class_metrics=per_class_metrics,
        sla_violations={
            "p99_latency": False,
            "error_rate": False,
            "gpu_utilization": False,
        },
    )
    
    # Save results
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = metrics_cfg.get("output_file", "").format(timestamp=timestamp)
        if not output_path:
            output_path = f"Evaluation/phase4_load_test/results/load_test_{timestamp}.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "config": config,
            "metrics": metrics.as_dict(),
            "raw_metrics": all_metrics[:1000],  # Sample of raw data
        }, f, indent=2)
    
    print(f"✅ Results saved to {output_path}")
    
    # Validate SLAs
    sla = experiment_cfg.get("sla", {})
    violations = []
    
    if metrics.latency_ms_p99 > sla.get("p99_latency_ms", 100):
        violations.append(f"P99 latency {metrics.latency_ms_p99:.1f}ms > {sla['p99_latency_ms']}ms")
        metrics.sla_violations["p99_latency"] = True
    
    if metrics.error_rate > sla.get("error_rate", 0.01):
        violations.append(f"Error rate {metrics.error_rate:.2%} > {sla['error_rate']:.2%}")
        metrics.sla_violations["error_rate"] = True
    
    gpu_util_target = sla.get("gpu_utilization", 0.80)
    if metrics.gpu_utilization_pct is not None and metrics.gpu_utilization_pct < gpu_util_target:
        violations.append(f"GPU utilization {metrics.gpu_utilization_pct:.1%} < {gpu_util_target:.1%}")
        metrics.sla_violations["gpu_utilization"] = True
    
    # Print summary
    print(f"\n{'='*60}")
    print("LOAD TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Total requests: {metrics.total_requests}")
    print(f"Successful: {metrics.successful_requests}")
    print(f"Failed: {metrics.failed_requests}")
    print(f"Error rate: {metrics.error_rate:.2%}")
    print(f"\nLatency (ms):")
    print(f"  P50: {metrics.latency_ms_p50:.1f}")
    print(f"  P90: {metrics.latency_ms_p90:.1f}")
    print(f"  P95: {metrics.latency_ms_p95:.1f}")
    print(f"  P99: {metrics.latency_ms_p99:.1f}")
    print(f"\nThroughput: {metrics.throughput_per_s:.1f} req/s")
    if metrics.gpu_utilization_pct is not None:
        print(f"GPU utilization: {metrics.gpu_utilization_pct:.1f}%")
    if metrics.gpu_memory_used_mb is not None:
        print(f"GPU memory: {metrics.gpu_memory_used_mb:.1f} MB")
    
    print(f"\nPer-class metrics:")
    for class_name, class_metrics in per_class_metrics.items():
        print(f"  {class_name}:")
        print(f"    Requests: {class_metrics['total_requests']} ({class_metrics['successful_requests']} successful)")
        print(f"    P99 latency: {class_metrics['latency_ms_p99']:.1f}ms")
        print(f"    Throughput: {class_metrics['throughput_per_s']:.1f} req/s")
    
    if violations:
        print(f"\n❌ SLA VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return False
    else:
        print(f"\n✅ All SLA targets met!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 production load test")
    parser.add_argument(
        "--config",
        type=str,
        default="Evaluation/phase4_load_test/configs/production_load_test.yaml",
        help="Path to load test config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )
    args = parser.parse_args()
    
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    experiment_cfg = config["experiment"]
    server_address = experiment_cfg["djinn_server_address"]
    ensure_initialized_before_async(server_address)

    success = asyncio.run(run_load_test(args.config, args.output))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

