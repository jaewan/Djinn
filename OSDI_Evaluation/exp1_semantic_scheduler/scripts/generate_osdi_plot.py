#!/usr/bin/env python3
"""
Generate OSDI-Quality Latency vs Load Plot.

Compares all baselines on same axes:
- X-axis: Number of concurrent agents (N)
- Y-axis: P99 latency (ms) - log scale

Expected plot:
- Ray: Crashes at N=2-3 (marked with X)
- Serverless: Flat ~30s (cold start dominated)
- vLLM: Flat then crashes at N~48
- Djinn: Gradual increase to N=80+

Key insight: Djinn's "sweet spot" is where P99 < 2s (interactive zone).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not installed, plot generation will be skipped")
    plt = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict[str, Dict[int, Any]]:
    """Load all baseline results."""
    results = {
        "ray": {},
        "serverless": {},
        "vllm": {},
        "djinn": {},
    }
    
    # Find latest result files for each baseline
    for baseline in ["ray", "serverless", "vllm", "djinn"]:
        pattern = f"{baseline}_*_sweep_*.json" if baseline != "djinn" else f"djinn_sweep_*.json"
        files = sorted(results_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No results found for {baseline}")
            continue
        
        latest = files[-1]
        logger.info(f"Loading {baseline}: {latest.name}")
        
        with open(latest) as f:
            data = json.load(f)
            
            if isinstance(data, dict) and all(isinstance(k, str) and k.isdigit() for k in data.keys()):
                # Results are keyed by N as strings
                results[baseline] = {int(k): v for k, v in data.items()}
            else:
                results[baseline] = data
    
    return results


def extract_latencies(results: Dict[str, Dict[int, Any]]) -> Dict[str, tuple]:
    """
    Extract N values and P99 latencies for each baseline.
    
    Returns:
        {baseline: (ns, latencies)}
    """
    extracted = {}
    
    for baseline, data in results.items():
        ns = []
        latencies = []
        
        for n in sorted(data.keys()):
            result = data[n] if isinstance(data[n], dict) else {}
            
            # Skip errors
            status = result.get("status", "unknown")
            if status not in ["success", "cuda_oom", "oom"]:
                continue
            
            # Extract P99 latency
            if "p99_latency_ms" in result:
                p99 = result["p99_latency_ms"]
            elif "latency_stats" in result and "p99_ms" in result["latency_stats"]:
                p99 = result["latency_stats"]["p99_ms"]
            elif baseline == "serverless" and "latency_stats" in result:
                # Serverless might have flat latency
                p99 = result.get("latency_stats", {}).get("p99_ms", 0)
            else:
                logger.warning(f"No P99 latency for {baseline} N={n}")
                continue
            
            if p99 > 0:
                ns.append(n)
                latencies.append(p99)
            
            # Stop at OOM for vLLM
            if baseline == "vllm" and status == "cuda_oom":
                break
        
        if ns:
            extracted[baseline] = (ns, latencies)
    
    return extracted


def generate_plot(
    results: Dict[str, Dict[int, Any]],
    output_path: Path = Path("osdi_latency_vs_load.pdf"),
):
    """Generate OSDI comparison plot."""
    if plt is None:
        logger.warning("matplotlib not available, skipping plot generation")
        return
    
    logger.info(f"Generating plot: {output_path}")
    
    # Extract data
    data = extract_latencies(results)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each baseline
    colors = {
        "ray": "#d62728",       # Red
        "serverless": "#2ca02c",  # Green
        "vllm": "#1f77b4",      # Blue
        "djinn": "#9467bd",     # Purple
    }
    
    markers = {
        "ray": "X",
        "serverless": "s",
        "vllm": "o",
        "djinn": "D",
    }
    
    for baseline in ["ray", "vllm", "serverless", "djinn"]:
        if baseline not in data:
            logger.warning(f"No data for {baseline}")
            continue
        
        ns, latencies = data[baseline]
        
        if not ns:
            logger.warning(f"No valid latencies for {baseline}")
            continue
        
        ax.plot(
            ns, latencies,
            color=colors[baseline],
            marker=markers[baseline],
            markersize=8,
            linewidth=2,
            label=baseline.upper(),
            zorder=3
        )
    
    # Add interactive threshold line
    ax.axhline(
        y=2000,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Interactive (<2s)",
        zorder=2
    )
    
    # Shade interactive zone
    ax.fill_between(
        [0, 100], 0, 2000,
        alpha=0.1,
        color="green",
        zorder=1
    )
    
    # Formatting
    ax.set_xlabel("Concurrent Agents (N)", fontsize=14, fontweight="bold")
    ax.set_ylabel("P99 Latency (ms) - Log Scale", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0, 85)
    ax.set_ylim(100, 100000)
    ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=12, framealpha=0.95)
    ax.set_title("Djinn vs Baselines: Latency Under Load", fontsize=16, fontweight="bold")
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✅ Saved plot to {output_path}")
    
    # Also save PNG
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✅ Saved plot to {png_path}")
    
    plt.close()


def print_summary(results: Dict[str, Dict[int, Any]]):
    """Print summary table."""
    logger.info("\n" + "="*100)
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info("="*100)
    
    data = extract_latencies(results)
    
    all_ns = set()
    for baseline_data in data.values():
        all_ns.update(baseline_data[0])
    
    all_ns = sorted(all_ns)
    
    # Print header
    header = f"{'N':>4} | "
    for baseline in ["ray", "vllm", "serverless", "djinn"]:
        header += f"{baseline.upper():>12} | "
    logger.info(header)
    logger.info("-" * 70)
    
    # Print rows
    for n in all_ns:
        row = f"{n:>4} | "
        for baseline in ["ray", "vllm", "serverless", "djinn"]:
            if baseline in data:
                ns, lats = data[baseline]
                if n in ns:
                    idx = ns.index(n)
                    lat = lats[idx]
                    is_interactive = lat < 2000
                    marker = "✅" if is_interactive else "⚠️"
                    row += f"{lat:>10.0f}ms{marker} | "
                else:
                    row += f"{'N/A':>12} | "
            else:
                row += f"{'N/A':>12} | "
        logger.info(row)
    
    logger.info("="*100)
    logger.info("Legend: ✅ = Interactive (<2s), ⚠️  = Queueing (>2s)")
    logger.info("="*100 + "\n")


def main():
    """Generate plot and summary."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("OSDI_Evaluation/exp1_semantic_scheduler/results"))
    parser.add_argument("--output", type=Path, default=Path("OSDI_Evaluation/exp1_semantic_scheduler/osdi_latency_vs_load.pdf"))
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Load results
    results = load_results(args.results_dir)
    
    # Print summary
    print_summary(results)
    
    # Generate plot
    generate_plot(results, args.output)


if __name__ == "__main__":
    main()
