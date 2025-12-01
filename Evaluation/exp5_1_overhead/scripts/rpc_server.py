#!/usr/bin/env python3
"""
PyTorch RPC server for Experiment 5.1 baselines.

Launch this script on the GPU node that should host the remote models. The
server reads the same workload config used by the evaluation harness,
pre-loads the required models, and exposes a single RPC endpoint that runs a
forward/generate pass for the requested workload.

Usage (single-GPU example):

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0

python Evaluation/exp5_1_overhead/scripts/rpc_server.py \
    --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml
```
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from typing import Dict, Optional

import torch
import yaml
from torch.distributed import rpc
from torch.distributed.rendezvous import rendezvous
from torch.distributed import TCPStore

from Evaluation.common import pytorch_rpc_ops
from Evaluation.common.workloads import build_workload


class RpcWorkloadExecutor(pytorch_rpc_ops.RpcWorkloadExecutor):
    """Wraps a workload/model so it can execute arbitrary inputs via RPC."""

    def __init__(self, workload_cfg: Dict, device: str, dtype: str):
        self.workload_cfg = workload_cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.implementation = workload_cfg["implementation"]
        spec = workload_cfg.get("params", {})
        self.workload = build_workload(self.implementation, spec, device, dtype)
        self.model = getattr(self.workload, "model", None)
        if self.model is None:
            raise ValueError(f"Workload '{workload_cfg['name']}' does not expose a model")
        self.model.to(self.device)
        self.model.eval()

    # OSDI FIX: Use torch.no_grad() instead of torch.inference_mode()
    # for fair comparison with native PyTorch baseline (which uses no_grad)
    def execute(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        def _to_device(key: str, t: torch.Tensor) -> torch.Tensor:
            # Don't convert input_ids to float - they must remain int for embedding
            if key in {"input_ids", "attention_mask", "token_type_ids"}:
                if t.device == self.device:
                    return t
                return t.to(self.device, non_blocking=False)
            
            # Convert other tensors to model dtype
            if t.device == self.device and t.dtype == self.model.dtype:
                return t
            return t.to(self.device, dtype=self.model.dtype, non_blocking=False)

        tensor_inputs = {
            key: _to_device(key, value) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            if self.implementation == "hf_causal_lm":
                gen_kwargs = dict(self.workload.generation_params)
                gen_kwargs["max_new_tokens"] = self.workload.new_tokens
                output = self.model.generate(**tensor_inputs, **gen_kwargs)
            elif self.implementation.startswith("synthetic"):
                tensor = tensor_inputs.get("x")
                if tensor is None:
                    raise ValueError("Synthetic workload inputs must include key 'x'")
                output = self.model(tensor)
            elif self.implementation in {"hf_vision", "hf_image_classification"}:
                outputs = self.model(**tensor_inputs)
                output = getattr(outputs, "logits", outputs)
            else:
                raise NotImplementedError(
                    f"RPC workload implementation '{self.implementation}' not supported"
                )
        
            # OSDI FIX: Ensure GPU execution is complete before returning
            # This guarantees timing measurement includes ALL GPU work.
            # See: OSDI Review - Critique 1 (RPC Warmup Trap)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        
        return output.detach().cpu()


def _rpc_forward(workload_name: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return pytorch_rpc_ops.rpc_forward(workload_name, inputs)


def _load_workloads(
    config_path: str, device: str, dtype: str, only: Optional[set[str]]
) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    for workload in payload.get("workloads", []):
        name = workload["name"]
        if only and name not in only:
            continue
        pytorch_rpc_ops.register_executor(name, RpcWorkloadExecutor(workload, device, dtype))
        print(f"[rpc_server] Loaded workload '{name}' on {device}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch RPC baseline server")
    parser.add_argument(
        "--config",
        type=str,
        default="Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml",
        help="Workload config YAML (same as evaluation harness).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to host the models.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model dtype ('float16', 'float32', ...).",
    )
    parser.add_argument(
        "--worker-name",
        type=str,
        default="rpc_server",
        help="RPC worker name exposed to clients.",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        help="Optional whitelist of workload names to register.",
    )
    parser.add_argument(
        "--init-timeout",
        type=float,
        default=0,
        help="RPC timeout (0 disables).",
    )
    return parser.parse_args()


def _init_rpc(worker_name: str, rpc_timeout: float) -> None:
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if not (master_addr and master_port and rank and world_size):
        raise EnvironmentError(
            "MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE must be set before launching the RPC server."
        )
    
    # Force CUDA context initialization on server BEFORE RPC init
    # to avoid cold-start overhead being attributed to RPC baseline.
    # This ensures fair comparison: server is warm when client connects.
    # See: OSDI Review - Critique 1 (RPC Warmup Trap)
    torch.ones(1).cuda()
    torch.cuda.synchronize()
    print("[rpc_server] CUDA context initialized and synchronized", flush=True)
    
    rank_int = int(rank)
    world_size_int = int(world_size)
    
    # Use more lenient RPC timeout for initialization
    init_timeout = 600  # 10 minutes for initialization
    
    print(f"[rpc_server] Initializing RPC: rank={rank_int}, world_size={world_size_int}", flush=True)
    print(f"[rpc_server] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}", flush=True)
    
    # Simple approach: just call init_rpc with long timeouts
    # PyTorch handles TCPStore creation internally when both processes call init_rpc
    init_method = f"tcp://{master_addr}:{master_port}"
    options = rpc.TensorPipeRpcBackendOptions(init_method=init_method)
    options.rpc_timeout = init_timeout
    
    print(f"[rpc_server] Calling rpc.init_rpc with init_method={init_method}, timeout={init_timeout}s", flush=True)
    
    try:
        rpc.init_rpc(
            worker_name,
            rank=rank_int,
            world_size=world_size_int,
            rpc_backend_options=options,
        )
        print("[rpc_server] RPC initialization succeeded", flush=True)
    except Exception as e:
        print(f"[rpc_server] RPC initialization failed: {e}", flush=True, file=sys.stderr)
        raise


def main() -> None:
    args = _parse_args()
    workload_filter = set(args.workloads) if args.workloads else None
    pytorch_rpc_ops.clear_executors()
    _load_workloads(args.config, args.device, args.dtype, workload_filter)
    _init_rpc(args.worker_name, args.init_timeout)
    print("[rpc_server] RPC worker ready. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[rpc_server] Shutting down...", flush=True)
    finally:
        rpc.shutdown()
        print("[rpc_server] Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()


