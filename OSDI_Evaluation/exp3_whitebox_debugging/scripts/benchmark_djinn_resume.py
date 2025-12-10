#!/usr/bin/env python3
"""
Experiment 3 Baseline: Djinn Resume Latency (IO_WAIT → Ready)

Measures resume latency from a semantic breakpoint at layer L:
  - Spawn session with breakpoint (wait_for_resume=False)
  - Measure time from sending RESUME to ready-to-run L+1
Outputs latency per layer for comparison against recompute and manual offload.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from djinn.backend.runtime.initialization import init_async
from djinn.config import DjinnConfig
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model


def prepare_inputs(tokenizer, prompt: str, max_length: int, device: torch.device):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in encoded.items()}


async def measure_resume_latency(
    coordinator,
    fingerprint: str,
    inputs: Dict[str, torch.Tensor],
    breakpoint_layer: int,
) -> tuple:
    """
    Spawn a session with breakpoint, then measure resume latency.
    Returns (client_latency_ms, server_latency_ms).
    """
    # Spawn session paused at breakpoint
    _, metrics = await coordinator.execute_remote_model_with_breakpoint(
        fingerprint=fingerprint,
        inputs={"input_ids": inputs["input_ids"]},
        breakpoint_layer_index=breakpoint_layer,
        wait_for_resume=False,
    )
    
    # Extract checkpoint activation and session ID
    session_id = metrics.get("session_id")
    checkpoint_activation = metrics.get("checkpoint_activation")
    
    # Validate required fields
    if session_id is None:
        raise RuntimeError(
            f"Breakpoint execution failed to return session_id. "
            f"Available metrics keys: {list(metrics.keys())}"
        )
    if checkpoint_activation is None:
        raise RuntimeError(
            f"Breakpoint execution failed to return checkpoint_activation. "
            f"Available metrics keys: {list(metrics.keys())}"
        )

    # Measure resume (client-side wall clock includes RPC overhead)
    torch.cuda.synchronize()
    start = time.perf_counter()

    # Resume with the unmodified checkpoint activation
    # (In a real steering scenario, you'd modify it here before passing it back)
    _, resume_metrics = await coordinator.resume_from_checkpoint(
        fingerprint=fingerprint,
        session_id=session_id,
        modified_activation=checkpoint_activation,
        layer_index=breakpoint_layer,
    )

    torch.cuda.synchronize()
    end = time.perf_counter()
    
    client_latency_ms = (end - start) * 1000.0
    server_latency_ms = resume_metrics.get("server_resume_latency_ms", None)
    
    return client_latency_ms, server_latency_ms


async def run_benchmark_async(
    model_name: str,
    layers: List[int],
    max_length: int,
    output_path: Path,
    server: str = "localhost:5556",
    warmup_iters: int = 2,
    repeat_iters: int = 5,
):
    assert torch.cuda.is_available(), "CUDA GPU required for this benchmark."
    device = torch.device("cuda")

    # Init Djinn client (outside timing loop)
    config = DjinnConfig()
    config.network.remote_server_address = server
    await init_async(config)
    coordinator = get_coordinator()
    manager = EnhancedModelManager(coordinator=coordinator)

    # Load model and register (outside timing loop)
    model = create_hf_ghost_model(model_name, task="causal-lm")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fingerprint = await manager.register_model(model, model_id="llama-2-13b")

    # Prepare inputs (outside timing loop)
    prompt = "The future of AI is" + " context token" * (max_length // 4)
    inputs = prepare_inputs(tokenizer, prompt, max_length, device)

    results = []
    for layer in layers:
        # Warmup
        for _ in range(warmup_iters):
            _ = await measure_resume_latency(
                coordinator,
                fingerprint=fingerprint,
                inputs=inputs,
                breakpoint_layer=layer,
            )

        measurements = [
            await measure_resume_latency(
                coordinator,
                fingerprint=fingerprint,
                inputs=inputs,
                breakpoint_layer=layer,
            )
            for _ in range(repeat_iters)
        ]
        
        client_latencies = [m[0] for m in measurements]
        server_latencies = [m[1] for m in measurements if m[1] is not None]
        
        client_mean_ms = statistics.mean(client_latencies)
        client_std_ms = statistics.stdev(client_latencies) if len(client_latencies) > 1 else 0.0
        
        server_mean_ms = statistics.mean(server_latencies) if server_latencies else None
        server_std_ms = statistics.stdev(server_latencies) if len(server_latencies) > 1 else 0.0

        result_dict = {
            "layer": layer,
            "resume_latency_ms": client_mean_ms,  # Keep client latency as primary metric
            "resume_latency_std_ms": client_std_ms,
            "num_samples": repeat_iters,
            "client_latency_ms": client_mean_ms,
            "client_latency_std_ms": client_std_ms,
        }
        
        # Add server latency if available
        if server_mean_ms is not None:
            result_dict["server_latency_ms"] = server_mean_ms
            result_dict["server_latency_std_ms"] = server_std_ms
            result_dict["note"] = f"Client includes RPC overhead; server latency is GPU restore time"
            print(f"[Djinn Resume] Layer {layer}: client={client_mean_ms:.1f}±{client_std_ms:.1f} ms, server={server_mean_ms:.1f}±{server_std_ms:.1f} ms over {repeat_iters} runs")
        else:
            result_dict["note"] = "Server latency not available; client includes RPC overhead"
            print(f"[Djinn Resume] Layer {layer}: client={client_mean_ms:.1f}±{client_std_ms:.1f} ms over {repeat_iters} runs")
        
        results.append(result_dict)

    # Gather environment metadata for reproducibility
    import sys
    import platform
    metadata = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "platform": platform.platform(),
        "note": "Resume latency includes RPC round-trip; server-side GPU restore time not isolated",
    }
    
    output = {
        "model": model_name,
        "layers": layers,
        "results": results,
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Djinn resume benchmark saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Djinn resume baseline: resume latency vs depth")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/exp3_resume_results/djinn_resume_latency.json"),
    )
    parser.add_argument("--server", type=str, default="localhost:5556")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per layer")
    parser.add_argument("--repeat", type=int, default=5, help="Measured iterations per layer")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark_async(
            model_name=args.model,
            layers=args.layers,
            max_length=args.max_length,
            output_path=args.output,
            server=args.server,
            warmup_iters=max(args.warmup, 0),
            repeat_iters=max(args.repeat, 1),
        )
    )


if __name__ == "__main__":
    main()
