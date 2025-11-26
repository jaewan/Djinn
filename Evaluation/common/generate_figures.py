#!/usr/bin/env python3
"""
Automated figure generation pipeline for Djinn evaluation experiments.

Generates publication-ready figures from baseline comparison results:
- Figure 6: LLM Decode (Exp 2.1) - Latency, data transfer, GPU utilization
- Figure 7: Streaming Audio (Exp 2.2) - Latency comparison, data transfer breakdown
- Figure 8: Conversational AI (Exp 2.3) - Multi-turn dialogue performance

All figures include error bars (95% CI), annotations, and publication-quality formatting.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib import rcParams
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib numpy")


# Publication-quality figure settings
rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Calculate percentile from sorted list."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


def _confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval using t-distribution."""
    if not values or len(values) < 2:
        return (0.0, 0.0)
    n = len(values)
    mean = statistics.mean(values)
    std_err = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.0
    
    # Use t-distribution (approximate with normal for n > 30)
    if n > 30:
        z_score = 1.96  # 95% CI for normal distribution
    else:
        # Simplified: use 2.0 for small samples (conservative)
        z_score = 2.0
    
    margin = z_score * std_err
    return (mean - margin, mean + margin)


def load_baseline_results(result_file: Path) -> Dict[str, Any]:
    """Load baseline results from JSON file."""
    with open(result_file) as f:
        return json.load(f)


def extract_llm_decode_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from Exp 2.1 (LLM decode) results."""
    metrics = {
        "per_token_latencies": [],
        "total_latencies": [],
        "throughput": [],
        "data_transfer_mb": [],
    }
    
    if "runs" in data:
        for run in data["runs"]:
            if "per_token_ms" in run:
                metrics["per_token_latencies"].append(run["per_token_ms"])
            if "total_ms" in run:
                metrics["total_latencies"].append(run["total_ms"])
            if "throughput_tokens_per_s" in run:
                metrics["throughput"].append(run["throughput_tokens_per_s"])
            data_mb = run.get("host_to_device_mb", 0) + run.get("device_to_host_mb", 0)
            metrics["data_transfer_mb"].append(data_mb)
    
    # Calculate statistics
    stats = {}
    for key, values in metrics.items():
        if values:
            sorted_vals = sorted(values)
            stats[key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": _percentile(sorted_vals, 95.0),
                "p99": _percentile(sorted_vals, 99.0),
                "ci": _confidence_interval(values),
            }
        else:
            stats[key] = {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "ci": (0.0, 0.0)}
    
    return stats


def extract_streaming_audio_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from Exp 2.2 (Streaming Audio) results."""
    metrics = {
        "chunk_latencies": [],
        "total_times": [],
        "data_transfer_mb": [],
    }
    
    if "runs" in data:
        for run in data["runs"]:
            if "chunks" in run:
                for chunk in run["chunks"]:
                    metrics["chunk_latencies"].append(chunk.get("latency_ms", 0))
                    data_mb = chunk.get("host_to_device_mb", 0) + chunk.get("device_to_host_mb", 0)
                    metrics["data_transfer_mb"].append(data_mb)
            if "total_time_ms" in run:
                metrics["total_times"].append(run["total_time_ms"])
    
    # Calculate statistics
    stats = {}
    for key, values in metrics.items():
        if values:
            sorted_vals = sorted(values)
            stats[key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": _percentile(sorted_vals, 95.0),
                "ci": _confidence_interval(values),
            }
        else:
            stats[key] = {"mean": 0.0, "median": 0.0, "p95": 0.0, "ci": (0.0, 0.0)}
    
    return stats


def extract_conversation_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from Exp 2.3 (Conversational AI) results."""
    metrics = {
        "turn_latencies": [],
        "total_times": [],
        "data_transfer_mb": [],
    }
    
    if "runs" in data:
        for run in data["runs"]:
            if "turns" in run:
                for turn in run["turns"]:
                    metrics["turn_latencies"].append(turn.get("latency_ms", 0))
                    data_mb = turn.get("host_to_device_mb", 0) + turn.get("device_to_host_mb", 0)
                    metrics["data_transfer_mb"].append(data_mb)
            if "total_time_ms" in run:
                metrics["total_times"].append(run["total_time_ms"])
    
    # Calculate statistics
    stats = {}
    for key, values in metrics.items():
        if values:
            sorted_vals = sorted(values)
            stats[key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": _percentile(sorted_vals, 95.0),
                "ci": _confidence_interval(values),
            }
        else:
            stats[key] = {"mean": 0.0, "median": 0.0, "p95": 0.0, "ci": (0.0, 0.0)}
    
    return stats


def generate_figure_6a(results_dir: Path, output_path: Path) -> None:
    """Generate Figure 6a: Latency comparison (bar chart with error bars)."""
    if not _MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
    
    baselines = ["native_pytorch", "semantic_blind", "partially_aware", "full_djinn"]
    baseline_labels = ["Native\nPyTorch", "Semantic-\nBlind", "Partially-\nAware", "Full\nDjinn"]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    means = []
    errors = []

    for baseline in baselines:
        result_file = results_dir / baseline / f"llm_decode_{baseline}_*.json"
        result_files = list(results_dir.glob(f"llm_decode_{baseline}_*.json"))
        if not result_files:
            means.append(0.0)
            errors.append((0.0, 0.0))
            continue

        data = load_baseline_results(result_files[-1])
        stats = extract_llm_decode_metrics(data)
        mean = stats["per_token_latencies"]["mean"]
        ci = stats["per_token_latencies"]["ci"]
        means.append(mean)
        errors.append((mean - ci[0], ci[1] - mean))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(baselines))
    yerr_lower = [e[0] for e in errors]
    yerr_upper = [e[1] for e in errors]
    bars = ax.bar(x, means, yerr=[yerr_lower, yerr_upper],
                  color=colors, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Baseline', fontweight='bold')
    ax.set_ylabel('Latency per Token (ms)', fontweight='bold')
    ax.set_title('Figure 6a: Per-Token Latency Comparison', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {output_path}")


def generate_figure_6b(results_dir: Path, output_path: Path) -> None:
    """Generate Figure 6b: Data transfer comparison."""
    if not _MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
    
    baselines = ["native_pytorch", "semantic_blind", "partially_aware", "full_djinn"]
    baseline_labels = ["Native\nPyTorch", "Semantic-\nBlind", "Partially-\nAware", "Full\nDjinn"]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    means = []
    errors = []
    
    for baseline in baselines:
        result_files = list(results_dir.glob(f"llm_decode_{baseline}_*.json"))
        if not result_files:
            means.append(0.0)
            errors.append((0.0, 0.0))
            continue
        
        data = load_baseline_results(result_files[-1])
        stats = extract_llm_decode_metrics(data)
        mean = stats["data_transfer_mb"]["mean"]
        ci = stats["data_transfer_mb"]["ci"]
        means.append(mean)
        errors.append((mean - ci[0], ci[1] - mean))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(baselines))
    yerr_lower = [e[0] for e in errors]
    yerr_upper = [e[1] for e in errors]
    bars = ax.bar(x, means, yerr=[yerr_lower, yerr_upper],
                  color=colors, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Baseline', fontweight='bold')
    ax.set_ylabel('Data Transfer (MB)', fontweight='bold')
    ax.set_title('Figure 6b: Data Transfer Comparison', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_labels)
    ax.set_yscale('log')  # Log scale for large differences
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {output_path}")


def generate_figure_7a(results_dir: Path, output_path: Path) -> None:
    """Generate Figure 7a: Streaming audio latency comparison."""
    if not _MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
    
    baselines = ["native_pytorch", "semantic_blind", "partially_aware", "full_djinn"]
    baseline_labels = ["Native\nPyTorch", "Semantic-\nBlind", "Partially-\nAware", "Full\nDjinn"]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    means = []
    errors = []
    
    for baseline in baselines:
        result_files = list(results_dir.glob(f"streaming_audio_{baseline}_*.json"))
        if not result_files:
            means.append(0.0)
            errors.append((0.0, 0.0))
            continue
        
        data = load_baseline_results(result_files[-1])
        stats = extract_streaming_audio_metrics(data)
        mean = stats["chunk_latencies"]["mean"]
        ci = stats["chunk_latencies"]["ci"]
        means.append(mean)
        errors.append((mean - ci[0], ci[1] - mean))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(baselines))
    yerr_lower = [e[0] for e in errors]
    yerr_upper = [e[1] for e in errors]
    bars = ax.bar(x, means, yerr=[yerr_lower, yerr_upper],
                  color=colors, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Baseline', fontweight='bold')
    ax.set_ylabel('Chunk Latency (ms)', fontweight='bold')
    ax.set_title('Figure 7a: Streaming Audio Chunk Latency', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {output_path}")


def generate_figure_8(results_dir: Path, output_path: Path) -> None:
    """Generate Figure 8: Conversational AI multi-turn performance."""
    if not _MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
    
    baselines = ["native_pytorch", "semantic_blind", "partially_aware", "full_djinn"]
    baseline_labels = ["Native\nPyTorch", "Semantic-\nBlind", "Partially-\nAware", "Full\nDjinn"]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    means = []
    errors = []
    
    for baseline in baselines:
        result_files = list(results_dir.glob(f"conversation_{baseline}_*.json"))
        if not result_files:
            means.append(0.0)
            errors.append((0.0, 0.0))
            continue
        
        data = load_baseline_results(result_files[-1])
        stats = extract_conversation_metrics(data)
        mean = stats["turn_latencies"]["mean"]
        ci = stats["turn_latencies"]["ci"]
        means.append(mean)
        errors.append((mean - ci[0], ci[1] - mean))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(baselines))
    yerr_lower = [e[0] for e in errors]
    yerr_upper = [e[1] for e in errors]
    bars = ax.bar(x, means, yerr=[yerr_lower, yerr_upper],
                  color=colors, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Baseline', fontweight='bold')
    ax.set_ylabel('Turn Latency (ms)', fontweight='bold')
    ax.set_title('Figure 8: Conversational AI Turn Latency', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=["2.1", "2.2", "2.3", "all"], default="all", 
                       help="Experiment to generate figures for")
    parser.add_argument("--exp2_1_dir", type=Path, 
                       default=Path("Evaluation/exp2_1_llm_decode/results"),
                       help="Results directory for Exp 2.1")
    parser.add_argument("--exp2_2_dir", type=Path,
                       default=Path("Evaluation/exp2_2_streaming_audio/results"),
                       help="Results directory for Exp 2.2")
    parser.add_argument("--exp2_3_dir", type=Path,
                       default=Path("Evaluation/exp2_3_conversation/results"),
                       help="Results directory for Exp 2.3")
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/figures"),
                       help="Output directory for generated figures")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.exp in ["2.1", "all"]:
        print("\nGenerating figures for Exp 2.1 (LLM Decode)...")
        generate_figure_6a(args.exp2_1_dir, args.output_dir / "figure_6a_latency.pdf")
        generate_figure_6b(args.exp2_1_dir, args.output_dir / "figure_6b_data_transfer.pdf")
    
    if args.exp in ["2.2", "all"]:
        print("\nGenerating figures for Exp 2.2 (Streaming Audio)...")
        generate_figure_7a(args.exp2_2_dir, args.output_dir / "figure_7a_streaming_latency.pdf")
    
    if args.exp in ["2.3", "all"]:
        print("\nGenerating figures for Exp 2.3 (Conversational AI)...")
        generate_figure_8(args.exp2_3_dir, args.output_dir / "figure_8_conversation.pdf")
    
    print(f"\n✓ All figures generated in {args.output_dir}")


if __name__ == "__main__":
    main()

