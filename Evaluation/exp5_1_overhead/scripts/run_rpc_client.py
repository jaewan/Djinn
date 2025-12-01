#!/usr/bin/env python3
"""
PyTorch RPC client for baseline evaluation.

Run this after starting the RPC server with rpc_server.py.

Usage:
    # Terminal 1: Start server
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2
    python Evaluation/exp5_1_overhead/scripts/rpc_server.py --workloads hf_tiny_gpt2

    # Terminal 2: Run client
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2
    python Evaluation/exp5_1_overhead/scripts/run_rpc_client.py --workloads hf_tiny_gpt2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import yaml
from torch.distributed import rpc

from Evaluation.common import pytorch_rpc_ops
from Evaluation.common.workloads import build_workload
from Evaluation.common.metrics import summarize_fields


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tensor_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024 * 1024)


def _tensor_dict_mb(d: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for v in d.values():
        if torch.is_tensor(v):
            total += _tensor_mb(v)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch RPC client")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml"),
    )
    parser.add_argument("--workloads", nargs="*", help="Workload name filter")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--server-name", type=str, default="rpc_server")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Evaluation/exp5_1_overhead/results/pytorch_rpc"),
    )
    parser.add_argument("--tag", type=str, default="pytorch_rpc")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check environment
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")

    if not all([master_addr, master_port, rank, world_size]):
        print("ERROR: Set environment variables:")
        print("  export MASTER_ADDR=127.0.0.1")
        print("  export MASTER_PORT=29500")
        print("  export RANK=1")
        print("  export WORLD_SIZE=2")
        sys.exit(1)

    # Force CUDA context initialization on client too
    # See: OSDI Review - Critique 1 (RPC Warmup Trap)
    torch.ones(1).cuda()
    torch.cuda.synchronize()
    print("[rpc_client] CUDA context initialized and synchronized")

    # Initialize RPC
    print(f"[rpc_client] Connecting to {master_addr}:{master_port}...")
    init_method = f"tcp://{master_addr}:{master_port}"
    options = rpc.TensorPipeRpcBackendOptions(init_method=init_method, rpc_timeout=300)
    rpc.init_rpc("rpc_client", rank=int(rank), world_size=int(world_size), rpc_backend_options=options)
    print("[rpc_client] Connected!")
    
    # Give server time to be fully ready for tensor pipe operations
    # See: OSDI Review - Code Nit #1 (RPC Sync Race Condition)
    time.sleep(1)

    # Load workload configs
    with args.config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    workload_filter = set(args.workloads) if args.workloads else None
    results = []

    for workload_cfg in payload.get("workloads", []):
        name = workload_cfg["name"]
        if workload_filter and name not in workload_filter:
            continue

        print(f"[rpc_client] Running workload: {name}")

        # Build local workload for input preparation
        impl = workload_cfg["implementation"]
        spec = copy.deepcopy(workload_cfg.get("params", {}))
        workload = build_workload(impl, spec, "cpu", args.dtype)

        def prepare_inputs() -> Dict[str, torch.Tensor]:
            prepared = workload.prepare_inputs()
            return {
                key: value.cpu() if torch.is_tensor(value) else value
                for key, value in prepared.items()
            }

        def get_units(inputs: Dict[str, torch.Tensor], outputs: torch.Tensor) -> float:
            return workload.units_from_output(inputs, outputs)

        # Warmup
        for _ in range(args.warmup_runs):
            inputs = prepare_inputs()
            _ = rpc.rpc_sync(args.server_name, pytorch_rpc_ops.rpc_forward, args=(name, inputs))

        # Timed runs
        run_records = []
        for run_id in range(1, args.runs + 1):
            inputs = prepare_inputs()
            start = time.perf_counter()
            output = rpc.rpc_sync(args.server_name, pytorch_rpc_ops.rpc_forward, args=(name, inputs))
            latency_ms = (time.perf_counter() - start) * 1000.0

            input_mb = _tensor_dict_mb(inputs)
            output_mb = _tensor_mb(output) if torch.is_tensor(output) else 0.0
            total_mb = input_mb + output_mb
            units = get_units(inputs, output)
            throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

            run_records.append({
                "run_id": run_id,
                "latency_ms": latency_ms,
                "input_mb": input_mb,
                "output_mb": output_mb,
                "total_data_mb": total_mb,
                "units_processed": units,
                "throughput_units_per_s": throughput,
            })
            print(f"  Run {run_id}: {latency_ms:.2f}ms, {throughput:.1f} units/s")

        aggregates = summarize_fields(
            run_records, ["latency_ms", "throughput_units_per_s", "total_data_mb"]
        )
        results.append({
            "workload": name,
            "baseline": "pytorch_rpc",
            "runs": run_records,
            "aggregates": aggregates,
            "metadata": {
                **workload.metadata(),
                "baseline": "pytorch_rpc",
                "runner_type": "pytorch_rpc",
            },
        })

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_timestamp()
    output_path = args.output_dir / f"{args.tag}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "tag": args.tag,
            "generated_at": timestamp,
            "results": results,
        }, handle, indent=2)
    print(f"[rpc_client] Saved results to {output_path}")

    # Shutdown
    rpc.shutdown()
    print("[rpc_client] Done")


if __name__ == "__main__":
    main()

