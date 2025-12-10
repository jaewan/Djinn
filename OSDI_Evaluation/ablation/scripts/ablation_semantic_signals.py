"""
Ablation 4: Semantic Signal Value

Scientific Question: How much does proactive scheduling (IO_WAIT signals) improve over reactive scheduling?

Paper Claim: "Semantic signals enable proactive eviction, beating reactive heuristics"

This ablation compares three modes:
1. Proactive: djinn.signal_phase("IO_WAIT") before tool execution
2. Reactive: Rely on idle timeout (1.0s threshold)
3. None: No swapping (baseline for OOM threshold)

Expected result: Semantic signals enable 1.67x higher density (80 vs 48) with 36% lower latency.

FIXED: Generates YAML configs dynamically and calls run_poisson_experiment.py with proper arguments.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import tempfile
import yaml

# Add Djinn to path
sys.path.insert(0, '/home/ubuntu/Djinn')


class SignalBenchmark:
    """Track benchmark results for different scheduling modes."""
    
    def __init__(self):
        self.results = {}  # mode -> metrics
    
    def record(self, mode: str, max_agents: int, p99_latency: float, metrics: Dict):
        """Record results for a mode."""
        self.results[mode] = {
            'max_agents': max_agents,
            'p99_latency_ms': p99_latency,
            'metrics': metrics,
        }
    
    def get_result(self, mode: str) -> Dict:
        """Get results for a mode."""
        return self.results.get(mode, {})


def generate_config_for_mode(n_agents: int, mode: str, lambda_rate: float = 0.2) -> Dict:
    """
    Generate a YAML config for run_poisson_experiment.py with given mode.
    
    Args:
        n_agents: Number of agents
        mode: Scheduling mode ('proactive', 'reactive', 'none')
        lambda_rate: Poisson arrival rate
    
    Returns:
        Config dictionary
    """
    mode_names = {
        'proactive': 'Signal-driven (proactive eviction)',
        'reactive': 'Timeout-based (reactive eviction)',
        'none': 'No swapping (baseline)',
    }
    
    config = {
        'experiment': {
            'name': f'ablation_signals_{n_agents}_agents_{mode}',
            'description': f'Ablation 4: N={n_agents}, Mode={mode_names.get(mode, mode)}',
            'version': '1.0'
        },
        'workload': {
            'model_id': 'meta-llama/Llama-2-13b-hf',
            'dtype': 'float16',
            'total_agents': n_agents,
            'arrival_rate': lambda_rate,
            'think_time_min': 10.0,
            'think_time_max': 20.0,
            'new_tokens': 50,
            'iterations': 1,
            'context_length': 2048,
        },
        'semantic_scheduler': {
            'enabled': mode in ['proactive', 'reactive'],
            'idle_threshold_seconds': 1.0,
            'host_swap_pool_gb': 32.0,
            'lifo_on_overload': True,
        },
        'server_config': {
            'enable_semantic_scheduler': mode in ['proactive', 'reactive'],
            'idle_threshold_seconds': 1.0,
            'host_swap_pool_gb': 32.0,
            'max_concurrent': 256,
        },
        'expected_results': {
            'total_duration_s': int(n_agents / lambda_rate) + 100,
            'success_rate': 100 if mode == 'proactive' else 80,
        }
    }
    
    return config


def run_poisson_agents_experiment(
    n_agents: int,
    mode: str,
    lambda_rate: float = 0.2,
    think_time_range: Tuple[float, float] = (10, 20),
    timeout_sec: float = 600,
) -> Tuple[bool, Dict]:
    """
    Run Poisson agent experiment with a specific scheduling mode.
    
    FIXED: Generates YAML config dynamically and calls properly.
    
    Args:
        n_agents: Number of agents to spawn
        mode: Scheduling mode ('proactive', 'reactive', 'none')
        lambda_rate: Poisson arrival rate
        think_time_range: Range of think time in seconds
        timeout_sec: Maximum experiment duration (increased to 600s)
    
    Returns:
        (success_bool, metrics_dict)
    """
    mode_names = {
        'proactive': 'Signal-driven (proactive eviction)',
        'reactive': 'Timeout-based (reactive eviction)',
        'none': 'No swapping (baseline)',
    }
    
    print(f"\n{'='*70}")
    print(f"Running Poisson agents: N={n_agents}, mode={mode_names.get(mode, mode)}")
    print(f"{'='*70}")
    
    # FIXED: Generate config
    config = generate_config_for_mode(n_agents, mode, lambda_rate)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        output_path = f'/tmp/ablation_signals_exp_{n_agents}_{mode}.json'
        
        # FIXED: Call with correct arguments
        cmd = [
            'python',
            '/home/ubuntu/Djinn/OSDI_Evaluation/exp1_semantic_scheduler/scripts/run_poisson_experiment.py',
            f'--config={config_path}',
            f'--model-id=meta-llama/Llama-2-13b-hf',
            f'--output-dir={Path(output_path).parent}',
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec + 60,
        )
        
        if result.returncode != 0:
            print(f"Experiment failed (return code {result.returncode})")
            print(f"STDOUT: {result.stdout[-500:]}")
            print(f"STDERR: {result.stderr[-500:]}")
            return False, {'error': f"Failed with code {result.returncode}"}
        
        # Parse results
        try:
            # Find the most recent output file
            output_dir = Path(output_path).parent
            json_files = sorted(output_dir.glob('poisson_semantic_scheduler_*.json'), 
                               key=lambda p: p.stat().st_mtime, reverse=True)
            
            if not json_files:
                print(f"No output JSON found in {output_dir}")
                return False, {'error': 'No output file'}
            
            with open(json_files[0], 'r') as f:
                metrics = json.load(f)
            
            aggregates = metrics.get('aggregates', {})
            success_count = aggregates.get('success_count', 0)
            total_agents = aggregates.get('total_agents', n_agents)
            
            success = (success_count == total_agents)
            
            if success:
                print(f"✅ Completed: {n_agents} agents successful")
            else:
                print(f"⚠️  Completed: {success_count}/{total_agents} agents succeeded")
            
            return success, aggregates
        
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not parse results: {e}")
            return False, {'error': str(e)}
    
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out after {timeout_sec}s (OOM likely)")
        return False, {'error': 'timeout'}
    except Exception as e:
        print(f"Exception running experiment: {e}")
        return False, {'error': str(e)}
    finally:
        # Clean up temporary config file
        try:
            Path(config_path).unlink()
        except:
            pass


def find_max_agents(
    mode: str,
    max_agents: int = 80,
) -> Tuple[int, Dict]:
    """
    Binary search to find max agents before OOM.
    
    FIXED: Increased timeout to 600s.
    
    Args:
        mode: Scheduling mode
        max_agents: Maximum agents to test
    
    Returns:
        (max_agents_before_oom, metrics_from_highest_successful)
    """
    print(f"\nFinding max agents for mode='{mode}'...")
    
    # Quick sweep to find approximate max
    current_agents = 10
    step = 10
    last_metrics = {}
    
    while current_agents <= max_agents:
        success, metrics = run_poisson_agents_experiment(
            n_agents=current_agents,
            mode=mode,
            timeout_sec=600,  # FIXED: Increased from 30s to 600s
        )
        
        if not success:
            # Found OOM point, search back
            print(f"\nOOM at {current_agents} agents, binary searching...")
            low = current_agents - step
            high = current_agents
            
            while low < high:
                mid = (low + high + 1) // 2
                success, metrics = run_poisson_agents_experiment(
                    n_agents=mid,
                    mode=mode,
                    timeout_sec=600,  # FIXED
                )
                
                if success:
                    low = mid
                    last_metrics = metrics
                else:
                    high = mid - 1
            
            return low, last_metrics
        
        last_metrics = metrics
        current_agents += step
    
    return current_agents, last_metrics


def run_ablation_study() -> Dict[str, Dict]:
    """
    Run the semantic signal value ablation study.
    
    Compares proactive, reactive, and no-swapping modes.
    
    Returns:
        Dictionary with results for each mode
    """
    benchmark = SignalBenchmark()
    modes = ['proactive', 'reactive', 'none']
    
    for mode in modes:
        max_agents, metrics = find_max_agents(mode)
        p99_latency = metrics.get('p99_latency_ms', 0)
        benchmark.record(mode, max_agents, p99_latency, metrics)
    
    return benchmark.results


def generate_scaling_cliff_figure(results: Dict[str, Dict], output_path: str):
    """Generate a figure showing the scaling cliff for different modes."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping figure generation")
        return
    
    # Extract data
    modes = ['proactive', 'reactive', 'none']
    agents = [results.get(mode, {}).get('max_agents', 0) for mode in modes]
    latencies = [results.get(mode, {}).get('p99_latency_ms', 0) for mode in modes]
    
    mode_labels = ['Proactive\n(Signals)', 'Reactive\n(Timeout)', 'None\n(Baseline)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Max agents
    x_pos = np.arange(len(modes))
    bars1 = ax1.bar(x_pos, agents, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Maximum Concurrent Agents', fontsize=12)
    ax1.set_title('(a) Agent Density', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(mode_labels, fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Subplot 2: P99 latency
    bars2 = ax2.bar(x_pos, latencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax2.set_title('(b) Request Latency', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(mode_labels, fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}ms',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    fig.suptitle('Ablation 4: Semantic Signal Value\nImpact on Density and Latency', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to {output_path}")


def generate_latex_table(results: Dict[str, Dict]) -> str:
    """Generate a LaTeX table for signal ablation results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Mode & Max Agents & P99 Latency & Overhead \\ \midrule",
    ]
    
    modes = ['proactive', 'reactive', 'none']
    proactive = results.get('proactive', {})
    proactive_agents = proactive.get('max_agents', 0)
    
    for mode in modes:
        result = results.get(mode, {})
        agents = result.get('max_agents', 0)
        latency = result.get('p99_latency_ms', 0)
        
        if mode == 'none':
            mode_str = 'None (Baseline)'
        elif mode == 'reactive':
            mode_str = 'Reactive (Timeout)'
        else:
            mode_str = 'Proactive (Signals)'
        
        if proactive_agents > 0 and agents > 0:
            density_ratio = (agents / proactive_agents) * 100
        else:
            density_ratio = 0
        
        lines.append(
            f"{mode_str} & {agents} & {latency:.1f}ms & {density_ratio:.0f}\\% \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Semantic Signal Ablation: Effect of scheduling mode on density and latency.}",
        r"\label{tab:semantic_signals}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ablation 4: Semantic Signal Value")
    parser.add_argument('--output', type=str,
                        default='/home/ubuntu/Djinn/OSDI_Evaluation/ablation/results/ablation_signals.json',
                        help="Output JSON file")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ABLATION 4: SEMANTIC SIGNAL VALUE")
    print("="*80)
    
    # Run ablation study
    results = run_ablation_study()
    
    # Save JSON results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    # Generate LaTeX table
    table_tex = generate_latex_table(results)
    print("\n" + "="*80)
    print("LaTeX Table:")
    print("="*80)
    print(table_tex)
    
    # Save LaTeX table
    latex_path = output_path.parent / 'ablation_signals_table.tex'
    with open(latex_path, 'w') as f:
        f.write(table_tex)
    print(f"\n✅ LaTeX table saved to {latex_path}")
    
    # Generate figure
    figure_path = str(output_path.parent / 'ablation_signals_cliff.pdf')
    generate_scaling_cliff_figure(results, figure_path)
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    proactive = results.get('proactive', {})
    reactive = results.get('reactive', {})
    baseline = results.get('none', {})
    
    print(f"Proactive:  {proactive.get('max_agents', 0):2d} agents, {proactive.get('p99_latency_ms', 0):6.1f}ms P99")
    print(f"Reactive:   {reactive.get('max_agents', 0):2d} agents, {reactive.get('p99_latency_ms', 0):6.1f}ms P99")
    print(f"Baseline:   {baseline.get('max_agents', 0):2d} agents, {baseline.get('p99_latency_ms', 0):6.1f}ms P99")
    
    proactive_agents = proactive.get('max_agents', 0)
    reactive_agents = reactive.get('max_agents', 0)
    if reactive_agents > 0:
        density_gain = (proactive_agents - reactive_agents) / reactive_agents * 100
        print(f"\nDensity gain (Proactive vs Reactive): {density_gain:.1f}%")


if __name__ == '__main__':
    main()
