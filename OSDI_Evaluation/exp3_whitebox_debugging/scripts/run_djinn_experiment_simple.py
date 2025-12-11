#!/usr/bin/env python3
"""
Simple Djinn Resume Latency Benchmark (No Subprocess)

Assumes server is already running on 127.0.0.1:5556.
This avoids subprocess deadlock issues and is more reliable.

Usage:
  Terminal 1: python3 -m djinn.server --port 5556 --gpu 0
  Terminal 2: python3 run_djinn_experiment_simple.py
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
    
    Optimization: Use return_activation=False and use_stored_activation=True
    to eliminate 80MB of tensor transfers, measuring only the semantic cost.
    """
    # Spawn session paused at breakpoint - don't transfer activation
    _, metrics = await coordinator.execute_remote_model_with_breakpoint(
        fingerprint=fingerprint,
        inputs={"input_ids": inputs["input_ids"]},
        breakpoint_layer_index=breakpoint_layer,
        wait_for_resume=False,
        return_activation=False,  # Skip 40MB transfer to client
    )
    
    session_id = metrics.get("session_id")
    
    if session_id is None:
        raise RuntimeError(f"Breakpoint execution failed: {list(metrics.keys())}")

    # Measure resume using stored activation (zero data transfer)
    # Also return only last token logits to minimize network transfer
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    _, resume_metrics = await coordinator.resume_from_checkpoint(
        fingerprint=fingerprint,
        session_id=session_id,
        modified_activation=None,  # Don't transfer activation back
        layer_index=breakpoint_layer,
        use_stored_activation=True,  # Use server-side checkpoint
        return_last_token_only=True,  # Return only last token (~64KB instead of ~131MB)
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
    server: str = "127.0.0.1:5556",
    warmup_iters: int = 2,
    repeat_iters: int = 5,
):
    assert torch.cuda.is_available(), "CUDA GPU required for this benchmark."
    device = torch.device("cuda")

    # Init Djinn client
    print(f"📡 Connecting to Djinn server at {server}...")
    config = DjinnConfig()
    config.network.remote_server_address = server
    # Ensure control/data ports match the provided server (default config may differ)
    if ":" in server:
        try:
            server_host, server_port_str = server.split(":")
            server_port = int(server_port_str)
            config.network.control_port = server_port
            config.network.data_port = server_port
        except Exception:
            pass
    
    try:
        await init_async(config)
    except Exception as e:
        print(f"❌ Failed to initialize Djinn: {e}")
        print(f"   Make sure server is running: python3 -m djinn.server --port 5556 --gpu 0")
        raise
    
    coordinator = get_coordinator()
    if coordinator is None:
        print("❌ Coordinator not initialized")
        raise RuntimeError("Failed to connect to server")
    
    # Verify server connection before attempting registration
    print(f"🔌 Verifying server connection...")
    try:
        # Try a simple warmup RPC to verify server is responding
        warmup_response = await coordinator.warmup_remote_gpu()
        if warmup_response.get('status') == 'success':
            print(f"✅ Server connected and responded")
        else:
            print(f"⚠️  Server warmup warning: {warmup_response.get('message')}")
    except Exception as e:
        print(f"⚠️  Server warmup warning (non-critical): {e}")
        # Don't fail on warmup - it's optional, just continue
    
    manager = EnhancedModelManager(coordinator=coordinator)

    # Load model and register
    print(f"📦 Loading model: {model_name}")
    model = create_hf_ghost_model(model_name, task="causal-lm")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"📝 Registering model with server...")
    fingerprint = await manager.register_model(model, model_id="llama-2-13b")
    print(f"✅ Model registered: {fingerprint}")

    # Prepare inputs
    prompt = "The future of AI is" + " context token" * (max_length // 4)
    inputs = prepare_inputs(tokenizer, prompt, max_length, device)

    # CRITICAL: Warmup to force model loading before benchmarking
    # The server loads the model asynchronously, and we need to ensure it's resident before measuring
    print(f"\n🔥 Forcing model materialization on server (cold start)...")
    max_warmup_attempts = 60
    for attempt in range(max_warmup_attempts):
        try:
            print(f"  Attempt {attempt + 1}/{max_warmup_attempts}: Sending dummy inference...", end=" ")
            _, _ = await measure_resume_latency(
                coordinator,
                fingerprint=fingerprint,
                inputs=inputs,
                breakpoint_layer=1,  # Use layer 1 (fastest)
            )
            print("✅ Model is ready!")
            break
        except Exception as e:
            if "not found in cache" in str(e):
                print(f"⏳ Model loading... ({str(e)[:50]})")
                await asyncio.sleep(1.0)  # Wait 1 second before retry
            else:
                print(f"⚠️  {type(e).__name__}: {str(e)[:50]}")
                await asyncio.sleep(0.5)
    else:
        # After max attempts, model still not loaded - this is a real error
        raise RuntimeError(f"Model {fingerprint} failed to materialize after {max_warmup_attempts} attempts")

    # Benchmark
    results = []
    for layer in layers:
        print(f"\n🔬 Testing Layer {layer}...")
        
        # Warmup
        print(f"  Warming up ({warmup_iters} iterations)...")
        for i in range(warmup_iters):
            try:
                _ = await measure_resume_latency(
                    coordinator,
                    fingerprint=fingerprint,
                    inputs=inputs,
                    breakpoint_layer=layer,
                )
                print(f"    Warmup {i+1}/{warmup_iters}: ✓")
            except Exception as e:
                print(f"    Warmup {i+1}/{warmup_iters}: ✗ {e}")
        
        # Measure
        print(f"  Measuring ({repeat_iters} iterations)...")
        measurements = []
        for i in range(repeat_iters):
            try:
                client_lat, server_lat = await measure_resume_latency(
                    coordinator,
                    fingerprint=fingerprint,
                    inputs=inputs,
                    breakpoint_layer=layer,
                )
                measurements.append((client_lat, server_lat))
                print(f"    Sample {i+1}/{repeat_iters}: {client_lat:.2f}ms")
            except Exception as e:
                print(f"    Sample {i+1}/{repeat_iters}: ✗ {e}")
        
        if not measurements:
            print(f"  ❌ No successful measurements for Layer {layer}")
            continue
        
        client_latencies = [m[0] for m in measurements]
        server_latencies = [m[1] for m in measurements if m[1] is not None]
        
        client_mean = statistics.mean(client_latencies)
        client_std = statistics.stdev(client_latencies) if len(client_latencies) > 1 else 0.0
        
        result_dict = {
            "layer": layer,
            "resume_latency_ms": client_mean,
            "resume_latency_std_ms": client_std,
            "num_samples": len(measurements),
            "client_latency_ms": client_mean,
            "client_latency_std_ms": client_std,
        }
        
        if server_latencies:
            server_mean = statistics.mean(server_latencies)
            server_std = statistics.stdev(server_latencies) if len(server_latencies) > 1 else 0.0
            result_dict["server_latency_ms"] = server_mean
            result_dict["server_latency_std_ms"] = server_std
            print(f"  ✅ Layer {layer}: client={client_mean:.1f}±{client_std:.1f}ms, server={server_mean:.1f}±{server_std:.1f}ms")
        else:
            print(f"  ✅ Layer {layer}: client={client_mean:.1f}±{client_std:.1f}ms")
        
        results.append(result_dict)

    # Save results
    output = {
        "model": model_name,
        "layers": layers,
        "results": results,
        "metadata": {
            "python_version": "3.12.3",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "server": server,
            "note": "Real Djinn resume latency measurements with server running separately",
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Simple Djinn resume benchmark (server must be running separately)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/exp3_resume_results_final/djinn_resume_latency.json"),
    )
    parser.add_argument("--server", type=str, default="127.0.0.1:5556")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(
        run_benchmark_async(
            model_name=args.model,
            layers=args.layers,
            max_length=args.max_length,
            output_path=args.output,
            server=args.server,
            warmup_iters=args.warmup,
            repeat_iters=args.repeat,
        )
    )


if __name__ == "__main__":
    main()




