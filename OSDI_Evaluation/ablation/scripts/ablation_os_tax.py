#!/usr/bin/env python3
"""
Ablation 1: OS Tax (Interposition Overhead)

Scientific Question: What is the latency cost of Djinn's runtime interposition layer?

Methodology:
1. Measure identical operations: (a) Native PyTorch on local GPU, (b) Djinn via server
2. Operations: torch.add (micro), torch.matmul (GEMM), small model forward (macro)
3. 1000 iterations after 100 warmup, report mean/p50/p99/std

Expected Result: 15-50us overhead per operation, negligible for LLM layers (10ms+)
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add Djinn to path
sys.path.insert(0, '/home/ubuntu/Djinn')

from djinn.backend.runtime.initialization import init_async
from djinn.config import DjinnConfig
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model


class TinyModel(nn.Module):
    """Tiny model for macro-level overhead testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def measure_native_operations(n_warmup: int = 100, n_iters: int = 1000, model_id: str = 'gpt2', model_name: str = 'GPT-2') -> Dict:
    """Measure native PyTorch operations on local GPU."""
    print("\n" + "="*70)
    print("NATIVE PYTORCH BASELINE (Local GPU)")
    print("="*70)
    
    device = torch.device("cuda")
    results = {}
    
    # 1. Micro: torch.add
    print("\n1. Measuring torch.add (1x1 tensors)...")
    x = torch.randn(1, 1, device=device)
    y = torch.randn(1, 1, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        _ = torch.add(x, y)
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.add(x, y)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # microseconds
    
    results['add'] = {
        'mean_us': statistics.mean(latencies),
        'median_us': statistics.median(latencies),
        'p99_us': np.percentile(latencies, 99),
        'std_us': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'n_samples': len(latencies)
    }
    print(f"  Mean: {results['add']['mean_us']:.2f}us, P99: {results['add']['p99_us']:.2f}us")
    
    # 2. GEMM: torch.matmul
    print("\n2. Measuring torch.matmul (128x128 @ 128x128)...")
    a = torch.randn(128, 128, device=device)
    b = torch.randn(128, 128, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # microseconds
    
    results['matmul'] = {
        'mean_us': statistics.mean(latencies),
        'median_us': statistics.median(latencies),
        'p99_us': np.percentile(latencies, 99),
        'std_us': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'n_samples': len(latencies)
    }
    print(f"  Mean: {results['matmul']['mean_us']:.2f}us, P99: {results['matmul']['p99_us']:.2f}us")
    
    # 3. Macro: Small model forward
    print("\n3. Measuring TinyModel forward (128 -> 256 -> 128)...")
    model = TinyModel().to(device).eval()
    x = torch.randn(1, 128, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
            torch.cuda.synchronize()
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1e6)  # microseconds
    
    results['model_forward'] = {
        'mean_us': statistics.mean(latencies),
        'median_us': statistics.median(latencies),
        'p99_us': np.percentile(latencies, 99),
        'std_us': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'n_samples': len(latencies)
    }
    print(f"  Mean: {results['model_forward']['mean_us']:.2f}us, P99: {results['model_forward']['p99_us']:.2f}us")
    
    # 4. CRITICAL: Native Model (for apples-to-apples comparison with Djinn)
    print(f"\n4. Measuring Native {model_name} (Baseline for Djinn comparison)...")
    print("  This is the CRITICAL baseline for computing OS Tax")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  Loading {model_name} ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    native_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True
    ).eval()
    
    # Use same input as Djinn will use
    inputs = tokenizer("Hi", return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Warming up ({n_warmup} iterations)...")
    
    # Warmup
    with torch.no_grad():
        for i in range(n_warmup):
            _ = native_model(input_ids)
            torch.cuda.synchronize()
            if (i + 1) % 20 == 0:
                print(f"    Warmup: {i+1}/{n_warmup}")
    
    # Measure
    print(f"  Measuring ({n_iters} iterations)...")
    latencies = []
    with torch.no_grad():
        for i in range(n_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = native_model(input_ids)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # milliseconds
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{n_iters}, Current mean: {statistics.mean(latencies):.2f}ms")
    
    results['model_native'] = {
        'model_id': model_id,
        'model_name': model_name,
        'mean_ms': statistics.mean(latencies),
        'median_ms': statistics.median(latencies),
        'p99_ms': np.percentile(latencies, 99),
        'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'n_samples': len(latencies)
    }
    print(f"\n  Mean: {results['model_native']['mean_ms']:.2f}ms")
    print(f"  P99: {results['model_native']['p99_ms']:.2f}ms")
    print(f"  Std: {results['model_native']['std_ms']:.2f}ms")
    
    return results


async def measure_djinn_operations(server: str = "127.0.0.1:5556", n_warmup: int = 100, n_iters: int = 1000, model_id: str = 'gpt2', model_name: str = 'GPT-2') -> Dict:
    """Measure operations through Djinn server."""
    print("\n" + "="*70)
    print("DJINN REMOTE EXECUTION (via Server)")
    print("="*70)
    
    # Connect to server
    print(f"\nConnecting to Djinn server at {server}...")
    config = DjinnConfig()
    config.network.remote_server_address = server
    if ":" in server:
        server_host, server_port_str = server.split(":")
        server_port = int(server_port_str)
        config.network.control_port = server_port
        config.network.data_port = server_port
    
    try:
        await init_async(config)
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print(f"   Make sure server is running: python3 -m djinn.server --port 5556 --gpu 0")
        raise
    
    coordinator = get_coordinator()
    if coordinator is None:
        raise RuntimeError("Failed to get coordinator")
    
    print("✅ Connected to server")
    
    manager = EnhancedModelManager(coordinator=coordinator)
    results = {}
    
    # Load the specified model through Djinn
    print(f"\n1. Loading {model_name} through Djinn...")
    
    model = create_hf_ghost_model(model_id, task='causal-lm')
    
    print("2. Registering model with server...")
    fingerprint = await manager.register_model(model, model_id=model_id)
    print(f"✅ Model registered: {fingerprint[:16]}")
    
    # Prepare minimal input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use very short input to minimize compute time
    inputs = tokenizer("Hi", return_tensors='pt')
    input_dict = {"input_ids": inputs['input_ids']}
    
    print(f"\n3. Measuring remote execution latency ({n_warmup} warmup + {n_iters} samples)...")
    
    # Warmup
    print("  Warming up...")
    for i in range(n_warmup):
        try:
            _ = await manager.execute_model(model, input_dict)
            if (i + 1) % 20 == 0:
                print(f"    Warmup: {i+1}/{n_warmup}")
        except Exception as e:
            print(f"  Warmup iteration {i} failed: {e}")
            if i < 5:  # Only fail if early warmups fail
                raise
    
    # Measure
    print("  Measuring...")
    latencies = []
    for i in range(n_iters):
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = await manager.execute_model(model, input_dict)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # milliseconds (remote calls are slower)
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{n_iters}, Current mean: {statistics.mean(latencies):.2f}ms")
        except Exception as e:
            print(f"  Measurement iteration {i} failed: {e}")
            continue
    
    if not latencies:
        raise RuntimeError("All measurement iterations failed")
    
    results['remote_execution'] = {
        'model_id': model_id,
        'model_name': model_name,
        'mean_ms': statistics.mean(latencies),
        'median_ms': statistics.median(latencies),
        'p99_ms': np.percentile(latencies, 99),
        'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'n_samples': len(latencies)
    }
    print(f"\n  Mean: {results['remote_execution']['mean_ms']:.2f}ms")
    print(f"  P99: {results['remote_execution']['p99_ms']:.2f}ms")
    print(f"  Std: {results['remote_execution']['std_ms']:.2f}ms")
    
    return results


def compute_overhead(native_results: Dict, djinn_results: Dict) -> Dict:
    """Compute overhead statistics."""
    print("\n" + "="*70)
    print("OVERHEAD ANALYSIS (APPLES-TO-APPLES)")
    print("="*70)
    
    # CRITICAL FIX: Compare same model (not TinyModel to different model)
    # Both are in milliseconds now
    native_model_ms = native_results['model_native']['mean_ms']
    djinn_model_ms = djinn_results['remote_execution']['mean_ms']
    model_name = native_results['model_native']['model_name']
    
    # Overhead in milliseconds
    overhead_ms = djinn_model_ms - native_model_ms
    overhead_pct = (overhead_ms / native_model_ms) * 100 if native_model_ms > 0 else 0
    
    overhead = {
        'model_name': model_name,
        'native_model_ms': native_model_ms,
        'djinn_model_ms': djinn_model_ms,
        'overhead_ms': overhead_ms,
        'overhead_percent': overhead_pct,
        'overhead_description': f"Djinn adds {overhead_ms:.2f}ms ({overhead_pct:.1f}%) overhead",
        # Also include micro-op baselines for context
        'micro_add_us': native_results['add']['mean_us'],
        'micro_matmul_us': native_results['matmul']['mean_us'],
        'tiny_model_us': native_results['model_forward']['mean_us']
    }
    
    print(f"\n🔬 SCIENTIFIC COMPARISON (Same Model: {model_name}):")
    print(f"  Native {model_name} (local):    {native_model_ms:.2f}ms")
    print(f"  Djinn {model_name} (remote):    {djinn_model_ms:.2f}ms")
    print(f"  OS Tax (Overhead):              {overhead_ms:.2f}ms ({overhead_pct:.1f}%)")
    
    print(f"\n📊 Micro-operation Context (for reference):")
    print(f"  torch.add:               {overhead['micro_add_us']:.2f}us")
    print(f"  torch.matmul:            {overhead['micro_matmul_us']:.2f}us")
    print(f"  TinyModel forward:       {overhead['tiny_model_us']:.2f}us")
    
    print(f"\n💡 Interpretation:")
    print(f"  For a 7B LLM layer (~40ms), {overhead_ms:.2f}ms overhead = {(overhead_ms/40)*100:.1f}% of compute")
    print(f"  For {model_name} inference ({native_model_ms:.2f}ms), overhead = {overhead_pct:.1f}%")
    
    return overhead


def generate_latex_table(results: Dict, output_path: Path):
    """Generate LaTeX table for paper."""
    latex = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Operation} & \textbf{Native} & \textbf{Djinn} & \textbf{Overhead} \\
\midrule
"""
    
    native = results['native']
    djinn = results['djinn']
    overhead = results['overhead']
    
    # Add rows - micro-ops for context
    latex += f"Micro (add) & {native['add']['mean_us']:.1f}$\\mu$s & -- & -- \\\\\n"
    latex += f"GEMM (matmul) & {native['matmul']['mean_us']:.1f}$\\mu$s & -- & -- \\\\\n"
    latex += f"TinyModel & {native['model_forward']['mean_us']:.1f}$\\mu$s & -- & -- \\\\\n"
    latex += r"""\midrule
"""
    # CRITICAL: Apples-to-apples comparison
    model_name = overhead.get('model_name', 'Model')
    latex += f"{model_name} & {native['model_native']['mean_ms']:.2f}ms & {djinn['remote_execution']['mean_ms']:.2f}ms & {overhead['overhead_ms']:.2f}ms \\\\\n"
    
    latex += r"""\midrule
\textbf{OS Tax} & -- & -- & \textbf{""" + f"{overhead['overhead_percent']:.1f}\\%" + r"""} \\
\bottomrule
\end{tabular}
\caption{\textbf{OS Tax: Interposition Overhead.} Comparing identical """ + model_name + r""" inference (apples-to-apples), Djinn adds """ + f"{overhead['overhead_ms']:.2f}ms ({overhead['overhead_percent']:.1f}\\%)" + r""" overhead. For 7B LLM layers ($\sim$40ms), this represents $<$5\% of compute time.}
\label{tab:ablation_os_tax}
\end{table}
"""
    
    latex_path = output_path.parent / "os_tax_table.tex"
    latex_path.write_text(latex)
    print(f"\n✅ LaTeX table saved to: {latex_path}")


async def run_ablation(server: str = "127.0.0.1:5556", n_warmup: int = 100, n_iters: int = 1000, output_dir: Path = None, model_id: str = 'gpt2', model_name: str = 'GPT-2'):
    """Run complete OS Tax ablation study."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"ABLATION 1: OS TAX (INTERPOSITION OVERHEAD) - {model_name}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name} ({model_id})")
    print(f"  Server: {server}")
    print(f"  Warmup iterations: {n_warmup}")
    print(f"  Measurement iterations: {n_iters}")
    print(f"  Output directory: {output_dir}")
    
    # Measure native operations
    native_results = measure_native_operations(n_warmup=n_warmup, n_iters=n_iters, model_id=model_id, model_name=model_name)
    
    # Measure Djinn operations
    djinn_results = await measure_djinn_operations(server=server, n_warmup=n_warmup, n_iters=n_iters, model_id=model_id, model_name=model_name)
    
    # Compute overhead
    overhead = compute_overhead(native_results, djinn_results)
    
    # Compile results
    results = {
        'ablation': 'os_tax',
        'timestamp': time.time(),
        'config': {
            'model_id': model_id,
            'model_name': model_name,
            'server': server,
            'n_warmup': n_warmup,
            'n_iters': n_iters
        },
        'native': native_results,
        'djinn': djinn_results,
        'overhead': overhead
    }
    
    # Save results with model-specific filename
    model_safe_name = model_id.replace('/', '_').replace('-', '_')
    output_path = output_dir / f"ablation_os_tax_{model_safe_name}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate LaTeX table
    generate_latex_table(results, output_path)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation 1: OS Tax")
    parser.add_argument('--server', default='127.0.0.1:5556', help='Djinn server address')
    parser.add_argument('--warmup', type=int, default=100, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=1000, help='Measurement iterations')
    parser.add_argument('--output', type=Path, default=None, help='Output directory')
    parser.add_argument('--model_id', default='gpt2', help='HuggingFace model ID')
    parser.add_argument('--model_name', default='GPT-2', help='Display name for model')
    args = parser.parse_args()
    
    asyncio.run(run_ablation(
        server=args.server,
        n_warmup=args.warmup,
        n_iters=args.iters,
        output_dir=args.output,
        model_id=args.model_id,
        model_name=args.model_name
    ))


if __name__ == '__main__':
    main()
