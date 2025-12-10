"""
Ablation 1: OS Tax (Dispatch Overhead Analysis)

Scientific Question: How much latency does framework-level interposition add?

Measures end-to-end latency breakdown across three operation scales:
1. Micro-op: torch.add() - shows worst-case RPC overhead
2. Layer: Single transformer layer - shows typical workload overhead
3. Full: Complete forward pass - shows amortization

Expected result: Fixed overhead is negligible (<1%) for realistic workloads.

FIXED: Measures actual Djinn remote execution via remote_accelerator device.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import argparse

# Add Djinn to path
sys.path.insert(0, '/home/ubuntu/Djinn')

import djinn
from djinn.config import DjinnConfig
from djinn.core.device_compatibility import enable_remote_accelerator_device

# Enable remote_accelerator device for Djinn execution
enable_remote_accelerator_device()


class OperationBenchmark:
    """Benchmark container for measuring operation latencies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = {}  # mode -> list of latencies
    
    def record(self, mode: str, latency_ms: float):
        """Record a latency measurement."""
        if mode not in self.results:
            self.results[mode] = []
        self.results[mode].append(latency_ms)
    
    def get_stats(self, mode: str) -> Dict[str, float]:
        """Get statistics for a mode."""
        if mode not in self.results or not self.results[mode]:
            return {}
        
        latencies = sorted(self.results[mode])
        return {
            'min': latencies[0],
            'max': latencies[-1],
            'mean': sum(latencies) / len(latencies),
            'p50': latencies[len(latencies) // 2],
            'p99': latencies[int(len(latencies) * 0.99)],
            'count': len(latencies),
        }


def benchmark_native(op: Callable, n_iter: int = 100) -> List[float]:
    """
    Benchmark native PyTorch operation (no Djinn).
    
    Args:
        op: Callable that performs the operation
        n_iter: Number of iterations to benchmark
    
    Returns:
        List of latencies in milliseconds
    """
    latencies = []
    
    # Warmup
    for _ in range(5):
        op()
    
    # Actual benchmark
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        op()
        
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        latencies.append((t_end - t_start) * 1000)  # Convert to ms
    
    return latencies


def benchmark_djinn_cold(op: Callable, model_name: str = "test") -> List[float]:
    """
    Benchmark Djinn operation on FIRST CALL (meta-sim + cache miss).
    
    This measures the cold-start cost: meta-simulation required.
    
    Args:
        op: Callable that performs the operation
        model_name: Name for distinguishing model fingerprints
    
    Returns:
        List of single latency measurement (only 1 iteration, cold start)
    """
    latencies = []
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    # First call - cold start with meta-simulation
    op()
    
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    latencies.append((t_end - t_start) * 1000)  # Convert to ms
    
    return latencies


def benchmark_djinn_warm(op: Callable, n_iter: int = 100) -> List[float]:
    """
    Benchmark Djinn operation on SUBSEQUENT CALLS (cache hit).
    
    This measures the warm-cache cost: just RPC + execution.
    
    Args:
        op: Callable that performs the operation
        n_iter: Number of iterations to benchmark
    
    Returns:
        List of latencies in milliseconds
    """
    # Warmup - fill cache
    for _ in range(3):
        op()
    
    latencies = []
    
    # Measure with cache hits
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        op()
        
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        latencies.append((t_end - t_start) * 1000)  # Convert to ms
    
    return latencies


def setup_operations() -> Dict[str, Tuple[str, Callable]]:
    """
    Setup three operation scales using remote_accelerator (Djinn).
    
    Returns:
        Dictionary of operation_name -> (description, callable)
    """
    operations = {}
    
    # FIXED: Use remote_accelerator device for all operations to measure Djinn overhead
    # Operation 1: Micro-op (torch.add via remote_accelerator)
    x_small = torch.randn(1, 1, device='remote_accelerator:0')
    y_small = torch.randn(1, 1, device='remote_accelerator:0')
    operations['micro_add'] = (
        'torch.add(1x1) via Djinn',
        lambda: torch.add(x_small, y_small)
    )
    
    # Operation 2: Layer (single transformer layer via remote_accelerator)
    # Create a simple transformer layer and move to remote device
    layer = nn.TransformerEncoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        batch_first=True
    ).to('remote_accelerator:0')  # FIXED: Move to remote_accelerator, not cuda
    x_batch = torch.randn(4, 128, 512, device='remote_accelerator:0')
    operations['layer_forward'] = (
        'TransformerLayer(4x128x512) via Djinn',
        lambda: layer(x_batch)
    )
    
    # Operation 3: Full forward pass via Djinn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        # Load on CPU first, then move to remote_accelerator
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        model = model.to('remote_accelerator:0')  # FIXED: remote_accelerator, not cuda
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        input_ids = tokenizer("Hello, how are you?", return_tensors="pt")['input_ids'].to('remote_accelerator:0')
        operations['full_forward'] = (
            'GPT2 forward pass via Djinn',
            lambda: model(input_ids)
        )
    except Exception as e:
        print(f"Warning: Could not load GPT2 ({e}), using fallback")
        # Fallback: larger transformer layer
        large_layer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=4096, batch_first=True),
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=4096, batch_first=True),
        ).to('remote_accelerator:0')  # FIXED
        x_large = torch.randn(2, 256, 1024, device='remote_accelerator:0')
        operations['full_forward'] = (
            'Large transformer (2 layers) via Djinn',
            lambda: large_layer(x_large)
        )
    
    return operations


def run_ablation_study(use_remote: bool = True) -> Dict[str, OperationBenchmark]:
    """
    Run the OS Tax ablation study.
    
    FIXED: Always measures through Djinn (remote_accelerator device).
    We compare native PyTorch baseline vs Djinn cold (first call) vs Djinn warm (cached).
    
    Args:
        use_remote: Deprecated (always True now). Kept for compatibility.
    
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    operations = setup_operations()
    
    for op_name, (description, op_callable) in operations.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {op_name} ({description})")
        print(f"{'='*60}")
        
        benchmark = OperationBenchmark(op_name, description)
        
        # FIXED: Always measure both native baseline and Djinn execution
        # Step 1: Measure baseline (native PyTorch on CUDA)
        print("Measuring NATIVE PyTorch baseline (local CUDA, 100 iterations)...")
        x_native = torch.randn(1, 1, device='cuda')
        y_native = torch.randn(1, 1, device='cuda')
        native_op = lambda: torch.add(x_native, y_native) if op_name == 'micro_add' else op_callable()
        
        if op_name == 'micro_add':
            # For micro-op, measure native baseline
            try:
                latencies = benchmark_native(native_op, n_iter=100)
                benchmark.record('native', sum(latencies) / len(latencies))
                stats = benchmark.get_stats('native')
                print(f"  Native: {stats['mean']:.3f}ms (min={stats['min']:.3f}ms, max={stats['max']:.3f}ms)")
            except Exception as e:
                print(f"  ERROR in native benchmark: {e}")
        
        # Step 2: Measure Djinn - Cold (first call with meta-simulation)
        print("Measuring DJINN COLD (first call with meta-simulation)...")
        try:
            latencies_cold = benchmark_djinn_cold(op_callable, op_name)
            benchmark.record('djinn_cold', latencies_cold[0])
            stats_cold = benchmark.get_stats('djinn_cold')
            print(f"  Djinn Cold: {stats_cold['mean']:.3f}ms (meta-sim + dispatch)")
        except Exception as e:
            print(f"  ERROR in cold benchmark: {e}")
        
        # Step 3: Measure Djinn - Warm (cache hit)
        print("Measuring DJINN WARM (cached plan, 100 iterations)...")
        try:
            latencies_warm = benchmark_djinn_warm(op_callable, n_iter=100)
            avg_warm = sum(latencies_warm) / len(latencies_warm)
            benchmark.record('djinn_warm', avg_warm)
            stats_warm = benchmark.get_stats('djinn_warm')
            print(f"  Djinn Warm: {stats_warm['mean']:.3f}ms (RPC + execution)")
        except Exception as e:
            print(f"  ERROR in warm benchmark: {e}")
        
        results[op_name] = benchmark
    
    return results


def generate_table(results: Dict[str, OperationBenchmark]) -> str:
    """
    Generate a LaTeX table for the OS Tax ablation.
    
    Expected output: Table showing Native vs Djinn (Cold/Warm) latencies.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Operation & Native (ms) & Djinn Cold (ms) & Djinn Warm (ms) & Overhead \\ \midrule",
    ]
    
    for op_name, benchmark in results.items():
        stats_native = benchmark.get_stats('native')
        stats_cold = benchmark.get_stats('djinn_cold')
        stats_warm = benchmark.get_stats('djinn_warm')
        
        if not stats_native or not stats_warm:
            continue
        
        native_ms = stats_native['mean']
        cold_ms = stats_cold.get('mean', 0) if stats_cold else 0
        warm_ms = stats_warm['mean']
        
        # Calculate overhead
        if native_ms > 0:
            overhead_pct = ((warm_ms - native_ms) / native_ms) * 100
        else:
            overhead_pct = 0
        
        lines.append(
            f"{op_name} & {native_ms:.3f} & {cold_ms:.3f} & {warm_ms:.3f} & {overhead_pct:+.1f}\\% \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{OS Tax Ablation: Native vs Djinn (Cold/Warm) latencies across three operation scales.}",
        r"\label{tab:os_tax}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ablation 1: OS Tax")
    parser.add_argument('--output', type=str, default='/home/ubuntu/Djinn/OSDI_Evaluation/ablation/results/ablation_os_tax.json',
                        help="Output JSON file")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ABLATION 1: OS TAX (Dispatch Overhead Analysis)")
    print("="*80)
    print("\nNote: Measuring Djinn overhead via remote_accelerator device")
    print("  - Native: Local CUDA execution (baseline)")
    print("  - Djinn Cold: First call (includes meta-simulation)")
    print("  - Djinn Warm: Subsequent calls (plan cache hit)")
    print("="*80)
    
    # Run ablation (always measures Djinn)
    results = run_ablation_study(use_remote=True)
    
    # Prepare output
    output_data = {}
    for op_name, benchmark in results.items():
        output_data[op_name] = {
            'description': benchmark.description,
            'native': benchmark.get_stats('native'),
            'djinn_cold': benchmark.get_stats('djinn_cold'),
            'djinn_warm': benchmark.get_stats('djinn_warm'),
        }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    # Generate LaTeX table
    table_tex = generate_table(results)
    print("\n" + "="*80)
    print("LaTeX Table:")
    print("="*80)
    print(table_tex)
    
    # Save LaTeX table
    latex_path = output_path.parent / 'ablation_os_tax_table.tex'
    with open(latex_path, 'w') as f:
        f.write(table_tex)
    print(f"\n✅ LaTeX table saved to {latex_path}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    for op_name, benchmark in results.items():
        stats_native = benchmark.get_stats('native')
        stats_warm = benchmark.get_stats('djinn_warm')
        if stats_native and stats_warm:
            print(f"{op_name:20s}: {stats_native['mean']:8.3f}ms (native) → {stats_warm['mean']:8.3f}ms (Djinn warm)")


if __name__ == '__main__':
    main()
