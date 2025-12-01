#!/usr/bin/env python3
"""
Ray Actor Server for Exp 5.1 baseline comparison.

This actor-based server runs models for fair comparison with PyTorch RPC
and native baselines. Uses Ray's Plasma Object Store for tensor serialization.

Usage (single-GPU example):

```bash
# Terminal 1: Start Ray head node
ray start --head --port=6379 --num-gpus=1

# Terminal 2: Run experiments
export RAY_ADDRESS=127.0.0.1:6379
python Evaluation/exp5_1_overhead/scripts/run_overhead_sweep.py \
    --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \
    --workloads hf_tiny_gpt2

# Terminal 3: Stop Ray
ray stop
```
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import ray
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.workloads import build_workload


@ray.remote(num_gpus=1)
class ModelActor:
    """
    Ray Actor that executes models with same semantics as RPC and Native baselines.
    
    CRITICAL REQUIREMENTS FOR APPLES-TO-APPLES COMPARISON:
    1. Use torch.no_grad() (NOT @torch.inference_mode())
    2. Call torch.cuda.synchronize() before returning
    3. Use model.eval() mode
    4. Use same generation parameters as other baselines
    """

    def __init__(self, workload_cfg: Dict[str, Any], device: str, dtype: str):
        """Initialize actor with model loaded on GPU."""
        self.device = torch.device(device)
        self.dtype = dtype
        self.implementation = workload_cfg["implementation"]
        spec = workload_cfg.get("params", {})

        # Build workload (same as RPC server)
        self.workload = build_workload(self.implementation, spec, device, dtype)
        self.model = self.workload.model

        # Ensure model is on GPU and in eval mode
        self.model.to(self.device)
        self.model.eval()

        # Warmup CUDA context (same as other baselines)
        # This ensures CUDA kernels are compiled before first measurement
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            print(f"[ray_actor] CUDA context initialized on {self.device}")

    def execute(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Execute workload with IDENTICAL semantics to RPC server.
        
        This method MUST match RPC server execution exactly for fair comparison.
        """
        # Move inputs to GPU with correct dtypes (same as RPC)
        gpu_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Don't convert input_ids, attention_mask, token_type_ids to float
                if key in {"input_ids", "attention_mask", "token_type_ids"}:
                    gpu_inputs[key] = value.to(self.device)
                else:
                    # Convert other tensors to model dtype
                    gpu_inputs[key] = value.to(self.device, dtype=self.model.dtype)
            else:
                gpu_inputs[key] = value

        # Execute with SAME context as all other baselines
        # OSDI REQUIREMENT: torch.no_grad(), not @torch.inference_mode()
        with torch.no_grad():
            if self.implementation == "hf_causal_lm":
                # Generate with same parameters as other baselines
                gen_kwargs = dict(self.workload.generation_params)
                gen_kwargs["max_new_tokens"] = self.workload.new_tokens
                output = self.model.generate(**gpu_inputs, **gen_kwargs)
            elif self.implementation.startswith("synthetic"):
                # Synthetic workload (neural network simulation)
                output = self.model(gpu_inputs.get("x"))
            else:
                # Standard forward pass (vision, etc.)
                output = self.model(**gpu_inputs)
                if hasattr(output, "logits"):
                    output = output.logits

            # CRITICAL: Ensure GPU execution is complete before returning
            # This guarantees timing measurement includes ALL GPU work
            # See: OSDI Review - Critique 1 (Warmup Race Condition)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        # Return output on CPU (same as RPC)
        return output.detach().cpu()


def main():
    """
    Optional: Use this for manual Ray Actor management.
    
    In normal usage, Ray Actor is created by RayActorBaselineRunner.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ray Actor model server")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--dtype", default="float16", help="Data type")
    args = parser.parse_args()

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
        print("Ray initialized")

    print(f"Ray Actor Server ready on device {args.device}")
    print("Note: Use RayActorBaselineRunner for actual experiments")


if __name__ == "__main__":
    main()

