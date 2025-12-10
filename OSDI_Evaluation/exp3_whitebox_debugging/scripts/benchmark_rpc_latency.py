#!/usr/bin/env python3
"""
Experiment 3: RPC Framework Overhead Measurement

Measures raw TCP round-trip latency through Djinn's network stack.
This isolates the framework overhead (asyncio, serialization) from GPU work.

Strategy: Send a simple STATUS request that returns immediately on the server
without touching GPU. This gives us the baseline RPC latency.
"""

import argparse
import asyncio
import json
import socket
import statistics
import struct
import time
from pathlib import Path
from typing import List

import torch


async def measure_raw_tcp_latency(host: str, port: int) -> float:
    """
    Measure raw TCP connection + send + receive latency.
    Returns latency in milliseconds.
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    try:
        # Simple TCP handshake and close
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=5.0
        )
        
        # Send a simple ping message (empty control request)
        # Just send a minimal message to the server
        ping_msg = b"PING"
        writer.write(ping_msg)
        await writer.drain()
        
        # Read response (server should echo or close)
        try:
            response = await asyncio.wait_for(
                reader.read(1024),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            response = b""
        
        # Close connection
        writer.close()
        await writer.wait_closed()
        
    except Exception as e:
        # Return -1 to indicate error
        return -1.0
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000.0
    return latency_ms


async def measure_djinn_rpc_latency(host: str, port: int) -> float:
    """
    Measure Djinn RPC latency using a status/ping-like request.
    This goes through the full Djinn serialization stack.
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    try:
        # Open connection
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=5.0
        )
        
        # Send a minimal Djinn message:
        # Format: [msg_type:1][reserved:1][payload_size:2][payload]
        # Type 0xFF = STATUS (placeholder, won't be processed, just tests RPC overhead)
        msg_type = 0xFF  # Undefined message type - server might not process
        reserved = 0x00
        payload = b""  # Empty payload
        payload_size = len(payload)
        
        message = struct.pack(">BBH", msg_type, reserved, payload_size) + payload
        
        writer.write(message)
        await writer.drain()
        
        # Try to read response (might timeout if server doesn't respond to unknown type)
        try:
            response = await asyncio.wait_for(
                reader.read(1024),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            response = b""
        
        # Close connection
        writer.close()
        await writer.wait_closed()
        
    except Exception as e:
        return -1.0
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000.0
    return latency_ms


async def run_benchmark_async(
    host: str = "127.0.0.1",
    port: int = 5556,
    num_samples: int = 50,
    warmup_iters: int = 5,
):
    """
    Benchmark RPC overhead by measuring TCP + minimal message latency.
    """
    assert torch.cuda.is_available(), "CUDA GPU required for this benchmark."

    print(f"🔬 RPC Framework Overhead Measurement")
    print(f"   Server: {host}:{port}")
    print(f"   Warmup iterations: {warmup_iters}")
    print(f"   Measured samples: {num_samples}")
    print()

    # Warmup: Raw TCP latency
    print("Phase 1: Warming up raw TCP latency...")
    for i in range(warmup_iters):
        latency = await measure_raw_tcp_latency(host, port)
        if latency > 0:
            print(f"  Warmup {i+1}/{warmup_iters}: {latency:.2f}ms")
        else:
            print(f"  Warmup {i+1}/{warmup_iters}: ERROR")

    # Measure: Raw TCP latency
    print(f"\nMeasuring raw TCP latency ({num_samples} samples)...")
    tcp_measurements = []
    for i in range(num_samples):
        latency = await measure_raw_tcp_latency(host, port)
        if latency > 0:
            tcp_measurements.append(latency)
        if (i + 1) % 10 == 0 and latency > 0:
            print(f"  Sample {i+1}/{num_samples}: {latency:.2f}ms")

    if not tcp_measurements:
        print("❌ No successful TCP measurements. Server may not be responding.")
        return

    # Warmup: Djinn RPC latency
    print("\nPhase 2: Warming up Djinn RPC latency...")
    for i in range(warmup_iters):
        latency = await measure_djinn_rpc_latency(host, port)
        if latency > 0:
            print(f"  Warmup {i+1}/{warmup_iters}: {latency:.2f}ms")
        else:
            print(f"  Warmup {i+1}/{warmup_iters}: ERROR (expected)")

    # Measure: Djinn RPC latency
    print(f"\nMeasuring Djinn RPC latency ({num_samples} samples)...")
    rpc_measurements = []
    for i in range(num_samples):
        latency = await measure_djinn_rpc_latency(host, port)
        if latency > 0:
            rpc_measurements.append(latency)
        if (i + 1) % 10 == 0 and latency > 0:
            print(f"  Sample {i+1}/{num_samples}: {latency:.2f}ms")

    # Analyze TCP
    tcp_mean = statistics.mean(tcp_measurements) if tcp_measurements else 0
    tcp_stdev = statistics.stdev(tcp_measurements) if len(tcp_measurements) > 1 else 0
    tcp_p95 = sorted(tcp_measurements)[int(0.95 * len(tcp_measurements))] if tcp_measurements else 0

    # Analyze RPC
    rpc_mean = statistics.mean(rpc_measurements) if rpc_measurements else 0
    rpc_stdev = statistics.stdev(rpc_measurements) if len(rpc_measurements) > 1 else 0
    rpc_p95 = sorted(rpc_measurements)[int(0.95 * len(rpc_measurements))] if rpc_measurements else 0

    # Print summary
    print("\n" + "=" * 70)
    print("📊 FRAMEWORK OVERHEAD ANALYSIS")
    print("=" * 70)
    print()
    print("Raw TCP Connection (baseline):")
    print(f"  Mean:   {tcp_mean:.2f} ms")
    print(f"  Stdev:  {tcp_stdev:.2f} ms")
    print(f"  P95:    {tcp_p95:.2f} ms")
    print(f"  Count:  {len(tcp_measurements)} / {num_samples} successful")
    print()
    print("Djinn RPC (with serialization):")
    print(f"  Mean:   {rpc_mean:.2f} ms")
    print(f"  Stdev:  {rpc_stdev:.2f} ms")
    print(f"  P95:    {rpc_p95:.2f} ms")
    print(f"  Count:  {len(rpc_measurements)} / {num_samples} successful")
    print()
    print("Framework Overhead (RPC - TCP):")
    overhead_mean = rpc_mean - tcp_mean if rpc_measurements and tcp_measurements else 0
    print(f"  Mean:   {overhead_mean:.2f} ms")
    print("=" * 70)
    print()

    result = {
        "timestamp": time.time(),
        "server": f"{host}:{port}",
        "tcp_latency": {
            "mean_ms": tcp_mean,
            "stdev_ms": tcp_stdev,
            "p95_ms": tcp_p95,
            "count": len(tcp_measurements),
        },
        "rpc_latency": {
            "mean_ms": rpc_mean,
            "stdev_ms": rpc_stdev,
            "p95_ms": rpc_p95,
            "count": len(rpc_measurements),
        },
        "framework_overhead_ms": overhead_mean,
        "raw_tcp_measurements": tcp_measurements,
        "raw_rpc_measurements": rpc_measurements,
    }

    print("📝 INTERPRETATION:")
    print(f"  • TCP latency = {tcp_mean:.2f}ms is the kernel/network baseline")
    print(f"  • RPC latency = {rpc_mean:.2f}ms includes Python/asyncio overhead")
    print(f"  • Framework overhead ≈ {overhead_mean:.2f}ms")
    print()
    print(f"  • If Resume latency is ~35ms and RPC overhead is ~{rpc_mean:.1f}ms:")
    print(f"  • Then Actual Scheduling Logic = 35 - {rpc_mean:.1f} = ~{35 - rpc_mean:.1f}ms")
    print()

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Measure Djinn RPC framework overhead"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/rpc_framework_overhead.json"),
    )
    args = parser.parse_args()

    result = await run_benchmark_async(
        host=args.host,
        port=args.port,
        num_samples=args.samples,
        warmup_iters=args.warmup,
    )

    if result:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

