"""
Ablation 2: Session Arena Decomposition (Memory Architecture)

Scientific Question: How much of the 80-agent density comes from Session Arenas vs Semantic Scheduling?

Paper Claim: "Session Arenas reduce static overhead from 300MB to 64MB, the primary enabler for density"

This ablation sweeps arena sizes (64, 128, 256, 300 MB) with both semantic (proactive signals)
and reactive (timeout-based) scheduling to decompose their contributions.

Expected result: Session Arenas enable ~60% of density gain; Semantic Scheduling ~40%.

FIXED: Generates YAML config files dynamically and calls run_poisson_experiment.py with proper arguments.
Also sets GENIE_VMU_SESSION_ARENA_MB environment variable before executing.
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


class ArenaBenchmark:
    """Track benchmark results for different arena sizes and scheduling modes."""
    
    def __init__(self):
        self.results = {}  # (arena_mb, mode) -> max_agents
    
    def record(self, arena_mb: int, mode: str, max_agents: int):
        """Record max agents for a configuration."""
        self.results[(arena_mb, mode)] = max_agents
    
    def get_result(self, arena_mb: int, mode: str) -> int:
        """Get max agents for a configuration."""
        return self.results.get((arena_mb, mode), 0)


def generate_config_for_arena(arena_mb: int, use_signals: bool) -> Dict:
    """
    Generate a YAML config for run_poisson_experiment.py with given arena size and mode.
    
    Args:
        arena_mb: Session arena size in MB
        use_signals: If True, use semantic signals; False = reactive timeout
    
    Returns:
        Config dictionary
    """
    mode_name = "semantic" if use_signals else "reactive"
    
    config = {
        'experiment': {
            'name': f'ablation_arena_{arena_mb}mb_{mode_name}',
            'description': f'Ablation 2: Arena={arena_mb}MB, Mode={mode_name}',
            'version': '1.0'
        },
        'workload': {
            'model_id': 'meta-llama/Llama-2-13b-hf',
            'dtype': 'float16',
            'total_agents': 80,  # Start with hero result target
            'arrival_rate': 0.2,  # 1 agent per 5 seconds
            'think_time_min': 10.0,
            'think_time_max': 20.0,
            'new_tokens': 50,
            'iterations': 1,
            'context_length': 2048,
        },
        'semantic_scheduler': {
            'enabled': use_signals,
            'idle_threshold_seconds': 1.0,
            'host_swap_pool_gb': 32.0,
            'lifo_on_overload': True,
        },
        'server_config': {
            'enable_semantic_scheduler': use_signals,
            'idle_threshold_seconds': 1.0,
            'host_swap_pool_gb': 32.0,
            'max_concurrent': 256,
        },
        'expected_results': {
            'total_duration_s': 500,
            'p99_wake_latency_ms': 500,
            'p99_request_latency_ms': 3000,
            'success_rate': 100,
            'swaps_gt': 100 if use_signals else 0,
        }
    }
    
    return config


def run_density_experiment(arena_mb: int, use_signals: bool, timeout_sec: float = 600) -> Tuple[int, Dict]:
    """
    Run density experiment with given arena size and scheduling mode.
    
    FIXED: Generates YAML config dynamically and sets environment variable.
    
    Args:
        arena_mb: Session arena size in MB (64, 128, 256, 300)
        use_signals: If True, use semantic signals (IO_WAIT). If False, use reactive timeout.
        timeout_sec: Maximum time to wait (default 600s = 10 minutes)
    
    Returns:
        (max_agents_before_oom, metrics_dict)
    """
    mode_name = "semantic" if use_signals else "reactive"
    print(f"\n{'='*70}")
    print(f"Running density experiment: arena={arena_mb}MB, mode={mode_name}")
    print(f"{'='*70}")
    
    # FIXED: Generate config and write to temporary file
    config = generate_config_for_arena(arena_mb, use_signals)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # FIXED: Set environment variable before running
        env = os.environ.copy()
        env['GENIE_VMU_SESSION_ARENA_MB'] = str(arena_mb)
        
        output_path = f'/tmp/ablation_arena_exp_{arena_mb}_{mode_name}.json'
        
        # FIXED: Call with correct arguments
        cmd = [
            'python',
            '/home/ubuntu/Djinn/OSDI_Evaluation/exp1_semantic_scheduler/scripts/run_poisson_experiment.py',
            f'--config={config_path}',
            f'--model-id=meta-llama/Llama-2-13b-hf',
            f'--output-dir={Path(output_path).parent}',
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Environment: GENIE_VMU_SESSION_ARENA_MB={arena_mb}")
        
        # Run experiment with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec + 60,  # Add buffer for process cleanup
            env=env,
        )
        
        if result.returncode != 0:
            print(f"Experiment failed (return code {result.returncode})")
            print(f"STDOUT: {result.stdout[-500:]}")
            print(f"STDERR: {result.stderr[-500:]}")
            return 0, {'error': f"Failed with code {result.returncode}"}
        
        # Parse results - note: output filename includes timestamp
        try:
            # Find the most recent output file
            output_dir = Path(output_path).parent
            json_files = sorted(output_dir.glob('poisson_semantic_scheduler_*.json'), 
                               key=lambda p: p.stat().st_mtime, reverse=True)
            
            if not json_files:
                print(f"No output JSON found in {output_dir}")
                return 0, {'error': 'No output file'}
            
            with open(json_files[0], 'r') as f:
                metrics = json.load(f)
            
            # Extract aggregates
            aggregates = metrics.get('aggregates', {})
            success_count = aggregates.get('success_count', 0)
            total_agents = aggregates.get('total_agents', 80)
            
            # Interpret success: if all agents succeeded, that's our max
            if success_count == total_agents:
                max_agents = total_agents
                print(f"✅ Completed: {max_agents} agents (all succeeded)")
            else:
                max_agents = success_count
                print(f"⚠️  Completed: {max_agents}/{total_agents} agents succeeded")
            
            return max_agents, aggregates
        
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not parse results: {e}")
            return 0, {'error': str(e)}
    
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out after {timeout_sec}s (OOM or network issue likely)")
        return 0, {'error': 'timeout'}
    except Exception as e:
        print(f"Exception running experiment: {e}")
        return 0, {'error': str(e)}
    finally:
        # Clean up temporary config file
        try:
            Path(config_path).unlink()
        except:
            pass


def run_ablation_study(arena_sizes: List[int], modes: List[str]) -> Dict[Tuple[int, str], int]:
    """
    Run the session arena decomposition ablation.
    
    FIXED: Increased timeout to 600s to allow full experiment to run.
    
    Args:
        arena_sizes: List of arena sizes to test (MB)
        modes: List of scheduling modes ('semantic', 'reactive')
    
    Returns:
        Dictionary of (arena_mb, mode) -> max_agents
    """
    benchmark = ArenaBenchmark()
    
    for arena_mb in arena_sizes:
        for mode in modes:
            use_signals = (mode == 'semantic')
            # FIXED: Increased timeout from 30s to 600s (10 minutes)
            max_agents, metrics = run_density_experiment(
                arena_mb=arena_mb,
                use_signals=use_signals,
                timeout_sec=600  # Increased from 30s
            )
            benchmark.record(arena_mb, mode, max_agents)
    
    return benchmark.results


def generate_decomposition_figure(results: Dict[Tuple[int, str], int], output_path: str):
    """
    Generate a figure showing arena size vs max agents for both modes.
    
    This shows the decomposition of density gains.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping figure generation")
        return
    
    # Extract data
    arena_sizes = sorted(set(arena for arena, _ in results.keys()))
    semantic_agents = [results.get((a, 'semantic'), 0) for a in arena_sizes]
    reactive_agents = [results.get((a, 'reactive'), 0) for a in arena_sizes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(arena_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, semantic_agents, width, label='Semantic (Proactive Signals)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, reactive_agents, width, label='Reactive (Timeout)', color='#e74c3c')
    
    # Labels and formatting
    ax.set_xlabel('Session Arena Size (MB)', fontsize=12)
    ax.set_ylabel('Maximum Concurrent Agents', fontsize=12)
    ax.set_title('Ablation 2: Session Arena Decomposition\nEffect of Arena Size on Agent Density', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}MB' for a in arena_sizes])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to {output_path}")


def generate_latex_table(results: Dict[Tuple[int, str], int]) -> str:
    """Generate a LaTeX table for arena decomposition results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Arena Size & Semantic & Reactive & Gain \\ \midrule",
    ]
    
    arena_sizes = sorted(set(arena for arena, _ in results.keys()))
    
    for arena_mb in arena_sizes:
        semantic = results.get((arena_mb, 'semantic'), 0)
        reactive = results.get((arena_mb, 'reactive'), 0)
        
        if reactive > 0:
            gain_pct = ((semantic - reactive) / reactive) * 100
        else:
            gain_pct = 0
        
        lines.append(
            f"{arena_mb} MB & {semantic} & {reactive} & {gain_pct:+.0f}\\% \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Session Arena Decomposition: Effect of arena size on max concurrent agents.}",
        r"\label{tab:arena_decomposition}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ablation 2: Session Arena Decomposition")
    parser.add_argument('--arena-sizes', type=int, nargs='+', default=[64, 128, 256, 300],
                        help="Arena sizes to test (MB)")
    parser.add_argument('--modes', type=str, nargs='+', default=['semantic', 'reactive'],
                        help="Scheduling modes to test")
    parser.add_argument('--output', type=str, 
                        default='/home/ubuntu/Djinn/OSDI_Evaluation/ablation/results/ablation_arena.json',
                        help="Output JSON file")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ABLATION 2: SESSION ARENA DECOMPOSITION")
    print("="*80)
    
    # Run ablation study
    results = run_ablation_study(args.arena_sizes, args.modes)
    
    # Save JSON results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert tuple keys to strings for JSON serialization
        json_results = {f"{arena}_{mode}": agents for (arena, mode), agents in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    # Generate LaTeX table
    table_tex = generate_latex_table(results)
    print("\n" + "="*80)
    print("LaTeX Table:")
    print("="*80)
    print(table_tex)
    
    # Save LaTeX table
    latex_path = output_path.parent / 'ablation_arena_table.tex'
    with open(latex_path, 'w') as f:
        f.write(table_tex)
    print(f"\n✅ LaTeX table saved to {latex_path}")
    
    # Generate figure
    figure_path = str(output_path.parent / 'ablation_arena_decomposition.pdf')
    generate_decomposition_figure(results, figure_path)
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    for arena_mb in sorted(set(a for a, _ in results.keys())):
        semantic = results.get((arena_mb, 'semantic'), 0)
        reactive = results.get((arena_mb, 'reactive'), 0)
        print(f"Arena {arena_mb}MB: Semantic={semantic}, Reactive={reactive}, Gain={(semantic-reactive)/reactive*100:.0f}%")


if __name__ == '__main__':
    main()
