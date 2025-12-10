#!/usr/bin/env python3
"""
Experiment 3: Framework Overhead Measurement (No-Op RPC Latency)

Measures the latency of a lightweight, non-GPU RPC call (get_capabilities).
This isolates the "Framework Overhead" (Python, asyncio, serialization, TCP) 
from the actual Djinn scheduling logic.

The difference between this and the resume latency shows how much is 
architecture vs. implementation.

Results:
  - Framework Overhead = Capabilities RPC latency
  - Actual Scheduling Logic = Resume latency - Framework overhead
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import List

import torch

from djinn.backend.runtime.initialization import init_async
from djinn.config import DjinnConfig
from djinn.core.coordinator import get_coordinator


async def measure_capabilities_latency(coordinator) -> float:
    """
    Measure latency of a lightweight, no-op RPC call.
    Returns client latency in milliseconds.
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Get capabilities is a lightweight RPC that does no GPU work
    capabilities = await coordinator.get_capabilities()
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    client_latency_ms = (end - start) * 1000.0
    return client_latency_ms


async def run_benchmark_async(
    num_samples: int = 50,
    server: str = "127.0.0.1:5556",
    warmup_iters: int = 5,
):
    """
    Benchmark the framework overhead by measuring get_capabilities() latency.
    """
    assert torch.cuda.is_available(), "CUDA GPU required for this benchmark."

    # Init Djinn client
    config = DjinnConfig()
    config.network.remote_server_address = server
    await init_async(config)
    coordinator = get_coordinator()

    print(f"🔬 Framework Overhead Measurement (No-Op RPC)")
    print(f"   Server: {server}")
    print(f"   Warmup iterations: {warmup_iters}")
    print(f"   Measured samples: {num_samples}")
    print()

    # Warmup
    print("Warming up...")
    for i in range(warmup_iters):
        _ = await measure_capabilities_latency(coordinator)
        print(f"  Warmup {i+1}/{warmup_iters}: OK")

    # Measure
    print(f"\nMeasuring framework overhead ({num_samples} samples)...")
    measurements = []
    for i in range(num_samples):
        latency_ms = await measure_capabilities_latency(coordinator)
        measurements.append(latency_ms)
        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{num_samples}: {latency_ms:.2f}ms")

    # Analyze
    mean_ms = statistics.mean(measurements)
    median_ms = statistics.median(measurements)
    stdev_ms = statistics.stdev(measurements)
    min_ms = min(measurements)
    max_ms = max(measurements)
    p95_ms = sorted(measurements)[int(0.95 * len(measurements))]
    p99_ms = sorted(measurements)[int(0.99 * len(measurements))]

    result = {
        "framework_overhead_ms": {
            "mean": mean_ms,
            "median": median_ms,
            "stdev": stdev_ms,
            "min": min_ms,
            "max": max_ms,
            "p95": p95_ms,
            "p99": p99_ms,
        },
        "num_samples": num_samples,
        "server": server,
        "warmup_iters": warmup_iters,
        "raw_measurements": measurements,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("📊 FRAMEWORK OVERHEAD ANALYSIS")
    print("=" * 70)
    print(f"Mean:           {mean_ms:.2f} ms")
    print(f"Median:         {median_ms:.2f} ms")
    print(f"Std Dev:        {stdev_ms:.2f} ms")
    print(f"Min:            {min_ms:.2f} ms")
    print(f"Max:            {max_ms:.2f} ms")
    print(f"P95:            {p95_ms:.2f} ms")
    print(f"P99:            {p99_ms:.2f} ms")
    print("=" * 70)
    print()
    print("📝 INTERPRETATION:")
    print(f"  • This is the latency of a NO-OP RPC (get_capabilities).")
    print(f"  • It includes: TCP round-trip, serialization, asyncio overhead.")
    print(f"  • It does NOT include: GPU work or scheduling logic.")
    print()
    print(f"  • If Resume latency is ~35ms and Framework overhead is ~{mean_ms:.1f}ms:")
    print(f"  • Then Actual Scheduling Logic = 35 - {mean_ms:.1f} = ~{35 - mean_ms:.1f}ms")
    print()

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Measure Djinn framework overhead via no-op RPC latency"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:5556",
        help="Djinn server address",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of RPC calls to measure",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/framework_overhead.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    result = await run_benchmark_async(
        num_samples=args.samples,
        server=args.server,
        warmup_iters=args.warmup,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())


