#!/usr/bin/env python3
"""
Integrated Djinn Resume Latency Benchmark
Starts server and runs benchmark in a single process to avoid timing issues.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

async def run_integrated_benchmark(
    model_name: str,
    layers: list,
    max_length: int,
    output_path: Path,
    server_port: int = 5556,
    warmup_iters: int = 3,
    repeat_iters: int = 5,
):
    """Run server and benchmark in integrated fashion."""
    
    # Start server process
    print("🚀 Starting Djinn server...")
    # FIX: Don't capture stdout/stderr via PIPE - causes subprocess deadlock when buffer fills
    # Instead, redirect to files or let output go to terminal
    server_log = open('/tmp/djinn_server_integrated.log', 'w')
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "djinn.server", "--port", str(server_port), "--gpu", "0"],
        stdout=server_log,
        stderr=server_log,
    )
    
    # Give server time to initialize (increased from 15s to 25s for safer startup)
    print("⏳ Waiting for server initialization (25s)...")
    await asyncio.sleep(25)
    
    # Check if server is alive
    if server_proc.poll() is not None:
        stdout, stderr = server_proc.communicate()
        print("❌ Server died immediately!")
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        return False
    
    print(f"✅ Server started (PID: {server_proc.pid})")
    
    try:
        # Now run the benchmark
        print("\n📊 Running Djinn resume latency benchmark...")
        
        # Import Djinn components
        import torch
        from transformers import AutoTokenizer
        from djinn.backend.runtime.initialization import init_async
        from djinn.config import DjinnConfig
        from djinn.core.coordinator import get_coordinator
        from djinn.core.enhanced_model_manager import EnhancedModelManager
        from djinn.core.ghost_loader import create_hf_ghost_model
        import statistics
        
        assert torch.cuda.is_available(), "CUDA GPU required"
        device = torch.device("cuda")
        
        # Init Djinn client
        config = DjinnConfig()
        config.network.remote_server_address = f"127.0.0.1:{server_port}"
        await init_async(config)
        coordinator = get_coordinator()
        manager = EnhancedModelManager(coordinator=coordinator)
        
        # Load model and register
        print(f"📦 Loading model: {model_name}")
        model = create_hf_ghost_model(model_name, task="causal-lm")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        fingerprint = await manager.register_model(model, model_id="llama-2-13b")
        print(f"✅ Model registered: {fingerprint}")
        
        # Prepare inputs
        prompt = "The future of AI is" + " context token" * (max_length // 4)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in encoded.items()}
        
        # Benchmark function
        async def measure_resume_latency(layer: int):
            """Measure resume latency at a specific layer."""
            # Spawn session paused at breakpoint
            _, metrics = await coordinator.execute_remote_model_with_breakpoint(
                fingerprint=fingerprint,
                inputs={"input_ids": inputs["input_ids"]},
                breakpoint_layer_index=layer,
                wait_for_resume=False,
            )
            
            session_id = metrics.get("session_id")
            checkpoint_activation = metrics.get("checkpoint_activation")
            
            if session_id is None or checkpoint_activation is None:
                raise RuntimeError(f"Breakpoint execution failed: {list(metrics.keys())}")
            
            # Measure resume
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _, resume_metrics = await coordinator.resume_from_checkpoint(
                fingerprint=fingerprint,
                session_id=session_id,
                modified_activation=checkpoint_activation,
                layer_index=layer,
            )
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            client_latency_ms = (end - start) * 1000.0
            server_latency_ms = resume_metrics.get("server_resume_latency_ms", None)
            
            return client_latency_ms, server_latency_ms
        
        # Run benchmark for each layer
        results = []
        for layer in layers:
            print(f"\n🔬 Testing Layer {layer}...")
            
            # Warmup
            print(f"  Warming up ({warmup_iters} iterations)...")
            for i in range(warmup_iters):
                try:
                    _ = await measure_resume_latency(layer)
                    print(f"    Warmup {i+1}/{warmup_iters}: OK")
                except Exception as e:
                    print(f"    Warmup {i+1}/{warmup_iters}: ERROR - {e}")
            
            # Measure
            print(f"  Measuring ({repeat_iters} iterations)...")
            measurements = []
            for i in range(repeat_iters):
                try:
                    client_lat, server_lat = await measure_resume_latency(layer)
                    measurements.append((client_lat, server_lat))
                    print(f"    Sample {i+1}/{repeat_iters}: {client_lat:.2f}ms (client)")
                except Exception as e:
                    print(f"    Sample {i+1}/{repeat_iters}: ERROR - {e}")
            
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
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup server
        print("\n🛑 Shutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("✅ Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Integrated Djinn resume benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/exp3_resume_results_final/djinn_resume_latency.json"),
    )
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    
    success = asyncio.run(
        run_integrated_benchmark(
            model_name=args.model,
            layers=args.layers,
            max_length=args.max_length,
            output_path=args.output,
            server_port=args.port,
            warmup_iters=args.warmup,
            repeat_iters=args.repeat,
        )
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()




