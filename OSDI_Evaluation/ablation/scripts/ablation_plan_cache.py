#!/usr/bin/env python3
"""
Ablation 2: Plan Cache Effectiveness

Scientific Question: Does meta-simulation caching eliminate planning overhead?

Methodology:
1. Connect to Djinn server with working pattern
2. Register a model (GPT-2 for speed)
3. Execute 10 requests with SAME input shape (all cache hits after first)
4. Execute 10 requests with DIFFERENT input shapes (cache misses)
5. Query /metrics/vmu endpoint to get plan_cache.hits, plan_cache.misses
6. Measure latency difference: cold (first) vs warm (cached)

Expected Result: Cold ~20-50ms, Warm ~0.5-1ms, Cache hit rate >95% for uniform workloads
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

# Add Djinn to path
sys.path.insert(0, '/home/ubuntu/Djinn')

from djinn.backend.runtime.initialization import init_async
from djinn.config import DjinnConfig
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model


async def fetch_cache_stats(server: str) -> Dict:
    """Fetch plan cache statistics from server metrics endpoint."""
    import aiohttp
    
    # Parse server address
    if ":" in server:
        host, port = server.split(":")
    else:
        host = server
        port = "5556"
    
    # Metrics are on port 9095 by default
    metrics_url = f"http://{host}:9095/metrics/vmu"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('plan_cache', {})
                else:
                    print(f"⚠️  Failed to fetch metrics: HTTP {response.status}")
                    return {}
    except Exception as e:
        print(f"⚠️  Could not fetch cache stats: {e}")
        return {}


async def measure_uniform_workload(
    manager: EnhancedModelManager,
    model,
    fingerprint: str,
    tokenizer,
    n_requests: int = 10,
    n_trials: int = 3
) -> Dict:
    """
    Measure latency for uniform workload (same input shape).
    First request is cold (cache miss), subsequent are warm (cache hits).
    """
    print("\n" + "="*70)
    print("UNIFORM WORKLOAD (Same Input Shape)")
    print("="*70)
    print(f"  {n_requests} requests × {n_trials} trials")
    print(f"  Expected: First request cold, rest warm (cached)")
    
    # Use fixed input shape
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors='pt')
    input_dict = {"input_ids": inputs['input_ids']}
    
    all_latencies = []
    cold_latencies = []  # First request per trial
    warm_latencies = []  # Subsequent requests
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}:")
        trial_latencies = []
        
        for i in range(n_requests):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            try:
                _ = await manager.execute_model(model, input_dict)
            except Exception as e:
                print(f"  Request {i+1} failed: {e}")
                continue
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            trial_latencies.append(latency_ms)
            all_latencies.append(latency_ms)
            
            if i == 0:
                cold_latencies.append(latency_ms)
                print(f"  Request {i+1} (COLD): {latency_ms:.2f}ms")
            else:
                warm_latencies.append(latency_ms)
                if i % 3 == 0:
                    print(f"  Request {i+1} (warm): {latency_ms:.2f}ms")
        
        print(f"  Trial {trial+1} mean: {statistics.mean(trial_latencies):.2f}ms")
    
    # Compute statistics
    results = {
        'n_requests': n_requests * n_trials,
        'n_trials': n_trials,
        'cold': {
            'mean_ms': statistics.mean(cold_latencies) if cold_latencies else 0,
            'median_ms': statistics.median(cold_latencies) if cold_latencies else 0,
            'std_ms': statistics.stdev(cold_latencies) if len(cold_latencies) > 1 else 0,
            'n_samples': len(cold_latencies)
        },
        'warm': {
            'mean_ms': statistics.mean(warm_latencies) if warm_latencies else 0,
            'median_ms': statistics.median(warm_latencies) if warm_latencies else 0,
            'p99_ms': np.percentile(warm_latencies, 99) if warm_latencies else 0,
            'std_ms': statistics.stdev(warm_latencies) if len(warm_latencies) > 1 else 0,
            'n_samples': len(warm_latencies)
        },
        'all': {
            'mean_ms': statistics.mean(all_latencies) if all_latencies else 0,
            'median_ms': statistics.median(all_latencies) if all_latencies else 0,
            'n_samples': len(all_latencies)
        }
    }
    
    if cold_latencies and warm_latencies:
        speedup = statistics.mean(cold_latencies) / statistics.mean(warm_latencies)
        results['speedup'] = speedup
        print(f"\n  Cold (first): {results['cold']['mean_ms']:.2f}ms")
        print(f"  Warm (cached): {results['warm']['mean_ms']:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")
    
    return results


async def measure_varied_workload(
    manager: EnhancedModelManager,
    model,
    fingerprint: str,
    tokenizer,
    n_requests: int = 10
) -> Dict:
    """
    Measure latency for varied workload (different input shapes).
    Each request has different length, causing cache misses.
    """
    print("\n" + "="*70)
    print("VARIED WORKLOAD (Different Input Shapes)")
    print("="*70)
    print(f"  {n_requests} requests with varying input lengths")
    print(f"  Expected: All requests cold (cache misses)")
    
    # Generate prompts of different lengths
    base_prompt = "Hello"
    prompts = [base_prompt + " world" * i for i in range(n_requests)]
    
    latencies = []
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_dict = {"input_ids": inputs['input_ids']}
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        try:
            _ = await manager.execute_model(model, input_dict)
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")
            continue
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 3 == 0:
            print(f"  Request {i+1} (shape {inputs['input_ids'].shape}): {latency_ms:.2f}ms")
    
    results = {
        'n_requests': len(latencies),
        'mean_ms': statistics.mean(latencies) if latencies else 0,
        'median_ms': statistics.median(latencies) if latencies else 0,
        'p99_ms': np.percentile(latencies, 99) if latencies else 0,
        'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'n_samples': len(latencies)
    }
    
    print(f"\n  Mean: {results['mean_ms']:.2f}ms")
    print(f"  Median: {results['median_ms']:.2f}ms")
    
    return results


async def run_ablation(
    server: str = "127.0.0.1:5556",
    n_uniform: int = 10,
    n_varied: int = 10,
    n_trials: int = 3,
    output_dir: Path = None
):
    """Run complete Plan Cache ablation study."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ABLATION 2: PLAN CACHE EFFECTIVENESS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Server: {server}")
    print(f"  Uniform workload: {n_uniform} requests × {n_trials} trials")
    print(f"  Varied workload: {n_varied} requests")
    print(f"  Output directory: {output_dir}")
    
    # Connect to server
    print(f"\n📡 Connecting to Djinn server at {server}...")
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
    
    # Initialize manager
    manager = EnhancedModelManager(coordinator=coordinator)
    
    # Load model
    print("\n📦 Loading GPT-2 model...")
    model = create_hf_ghost_model('gpt2', task='causal-lm')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Register model
    print("📝 Registering model with server...")
    fingerprint = await manager.register_model(model, model_id="gpt2")
    print(f"✅ Model registered: {fingerprint[:16]}")
    
    # Get initial cache stats
    print("\n📊 Fetching initial cache stats...")
    initial_stats = await fetch_cache_stats(server)
    print(f"  Initial cache stats: {initial_stats}")
    
    # Warmup
    print("\n🔥 Warming up (5 requests)...")
    inputs = tokenizer("Warmup", return_tensors='pt')
    for i in range(5):
        try:
            _ = await manager.execute_model(model, {"input_ids": inputs['input_ids']})
            print(f"  Warmup {i+1}/5 ✓")
        except Exception as e:
            print(f"  Warmup {i+1}/5 failed: {e}")
    
    # Get post-warmup cache stats
    warmup_stats = await fetch_cache_stats(server)
    print(f"  Post-warmup cache stats: {warmup_stats}")
    
    # Measure uniform workload (cache hits)
    uniform_results = await measure_uniform_workload(
        manager, model, fingerprint, tokenizer,
        n_requests=n_uniform,
        n_trials=n_trials
    )
    
    # Get cache stats after uniform workload
    uniform_stats = await fetch_cache_stats(server)
    print(f"\n📊 Cache stats after uniform workload: {uniform_stats}")
    
    # Measure varied workload (cache misses)
    varied_results = await measure_varied_workload(
        manager, model, fingerprint, tokenizer,
        n_requests=n_varied
    )
    
    # Get final cache stats
    final_stats = await fetch_cache_stats(server)
    print(f"\n📊 Final cache stats: {final_stats}")
    
    # Compute cache effectiveness
    cache_analysis = {
        'initial': initial_stats,
        'after_warmup': warmup_stats,
        'after_uniform': uniform_stats,
        'final': final_stats,
    }
    
    if final_stats and initial_stats:
        total_hits = final_stats.get('hits', 0) - initial_stats.get('hits', 0)
        total_misses = final_stats.get('misses', 0) - initial_stats.get('misses', 0)
        total_requests = total_hits + total_misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        cache_analysis['effectiveness'] = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate
        }
        
        print(f"\n📈 Cache Effectiveness:")
        print(f"  Total requests: {total_requests}")
        print(f"  Cache hits: {total_hits}")
        print(f"  Cache misses: {total_misses}")
        print(f"  Hit rate: {hit_rate:.1f}%")
    
    # Compile results
    results = {
        'ablation': 'plan_cache',
        'timestamp': time.time(),
        'config': {
            'server': server,
            'n_uniform': n_uniform,
            'n_varied': n_varied,
            'n_trials': n_trials
        },
        'uniform_workload': uniform_results,
        'varied_workload': varied_results,
        'cache_stats': cache_analysis
    }
    
    # Save results
    output_path = output_dir / "ablation_plan_cache.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate LaTeX table
    generate_latex_table(results, output_path)
    
    return results


def generate_latex_table(results: Dict, output_path: Path):
    """Generate LaTeX table for paper."""
    uniform = results['uniform_workload']
    varied = results['varied_workload']
    cache_eff = results['cache_stats'].get('effectiveness', {})
    
    latex = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lrr}
\toprule
\textbf{Workload} & \textbf{Latency (ms)} & \textbf{Cache Hit Rate} \\
\midrule
"""
    
    # Add rows
    if 'cold' in uniform and 'warm' in uniform:
        latex += f"Uniform (cold) & {uniform['cold']['mean_ms']:.1f} & 0\\% \\\\\n"
        latex += f"Uniform (warm) & {uniform['warm']['mean_ms']:.1f} & {cache_eff.get('hit_rate_percent', 0):.0f}\\% \\\\\n"
        speedup = uniform.get('speedup', 1.0)
        latex += f"\\textit{{Speedup}} & \\textit{{{speedup:.1f}$\\times$}} & -- \\\\\n"
    
    latex += r"""\midrule
"""
    latex += f"Varied shapes & {varied['mean_ms']:.1f} & Low \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\caption{\textbf{Plan Cache Effectiveness.} Uniform workloads (same input shape) benefit from plan caching with """ + f"{uniform.get('speedup', 1.0):.1f}$\\times$" + r""" speedup. Varied workloads incur cache misses but still complete successfully.}
\label{tab:ablation_plan_cache}
\end{table}
"""
    
    latex_path = output_path.parent / "plan_cache_table.tex"
    latex_path.write_text(latex)
    print(f"✅ LaTeX table saved to: {latex_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation 2: Plan Cache")
    parser.add_argument('--server', default='127.0.0.1:5556', help='Djinn server address')
    parser.add_argument('--uniform', type=int, default=10, help='Uniform workload requests')
    parser.add_argument('--varied', type=int, default=10, help='Varied workload requests')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--output', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    asyncio.run(run_ablation(
        server=args.server,
        n_uniform=args.uniform,
        n_varied=args.varied,
        n_trials=args.trials,
        output_dir=args.output
    ))


if __name__ == '__main__':
    main()
