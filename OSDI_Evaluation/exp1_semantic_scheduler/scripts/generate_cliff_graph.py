#!/usr/bin/env python3
"""
Generate the "Money Shot" Cliff Visualization.

Creates a graph showing:
- X-axis: Number of Concurrent Agents (N)
- Y-axis: P99 Latency (ms) - LOG SCALE
- vLLM line: Flat until N~48, then vertical (OOM/crash)
- Djinn line: Linear increase but continues to N=80+

This is the primary visualization for OSDI paper submission.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cliff_results(results_file: Path) -> Dict[str, Any]:
    """Load cliff experiment results from JSON."""
    with open(results_file) as f:
        return json.load(f)


def extract_data_for_plotting(results: Dict[str, Any]) -> Tuple[
    List[int], List[float], List[int], List[float], Optional[int]
]:
    """
    Extract vLLM and Djinn latency data from results.
    
    Returns:
        (vllm_ns, vllm_latencies, djinn_ns, djinn_latencies, cliff_point)
    """
    vllm_data = results.get("results_by_system", {}).get("vllm", {})
    djinn_data = results.get("results_by_system", {}).get("djinn", {})
    cliff_point = results.get("cliff_analysis", {}).get("vllm_crash_point")
    
    # Extract vLLM data
    vllm_ns = []
    vllm_latencies = []
    for n in sorted(vllm_data.keys()):
        if isinstance(n, str):
            n = int(n)
        result = vllm_data[str(n)] if str(n) in vllm_data else vllm_data.get(n, {})
        status = result.get("status", "unknown")
        
        if status == "success":
            # vLLM results have latency_stats
            latency_ms = result.get("latency_stats", {}).get("total_batch_ms", 0)
            if latency_ms > 0:
                vllm_ns.append(n)
                vllm_latencies.append(latency_ms / n)  # Per-request latency
    
    # Extract Djinn data
    djinn_ns = []
    djinn_latencies = []
    for n in sorted(djinn_data.keys()):
        if isinstance(n, str):
            n = int(n)
        result = djinn_data[str(n)] if str(n) in djinn_data else djinn_data.get(n, {})
        status = result.get("status", "unknown")
        
        if status == "success":
            # Djinn results have aggregates
            latency_ms = result.get("aggregates", {}).get("latency_stats", {}).get("p99_ms", 0)
            if latency_ms > 0:
                djinn_ns.append(n)
                djinn_latencies.append(latency_ms)
    
    return vllm_ns, vllm_latencies, djinn_ns, djinn_latencies, cliff_point


def create_cliff_graph(
    vllm_ns: List[int],
    vllm_latencies: List[float],
    djinn_ns: List[int],
    djinn_latencies: List[float],
    cliff_point: Optional[int],
    output_path: Path,
):
    """Create and save the cliff visualization."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot vLLM line
    if vllm_ns and vllm_latencies:
        ax.plot(
            vllm_ns, vllm_latencies,
            'o-', color='#d62728', linewidth=2.5, markersize=8,
            label='vLLM (Reactive Paging)', zorder=3
        )
    
    # Plot Djinn line
    if djinn_ns and djinn_latencies:
        ax.plot(
            djinn_ns, djinn_latencies,
            's-', color='#2ca02c', linewidth=2.5, markersize=8,
            label='Djinn (Semantic Scheduler)', zorder=3
        )
    
    # Add vertical dashed line at cliff point
    if cliff_point:
        ax.axvline(
            x=cliff_point, color='#d62728', linestyle='--', linewidth=2, alpha=0.6,
            label=f'vLLM OOM Cliff (N={cliff_point})', zorder=2
        )
        
        # Add annotation
        ax.text(
            cliff_point, ax.get_ylim()[1] * 0.85,
            f'vLLM Crashes\nat N={cliff_point}',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7)
        )
    
    # Formatting
    ax.set_xlabel('Number of Concurrent Agents (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('P99 Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Djinn vs vLLM: The Scalability Cliff\n(Semantic Scheduling Enables 1.67x Density)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Use log scale for Y-axis to show dynamics better
    # ax.set_yscale('log')
    
    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    # X-axis: show all N values tested
    if vllm_ns and djinn_ns:
        all_ns = sorted(set(vllm_ns + djinn_ns))
        ax.set_xticks(all_ns)
    
    # Tight layout
    fig.tight_layout()
    
    # Save as PDF and PNG
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    logger.info(f"✅ Saved cliff graph to:")
    logger.info(f"   {pdf_path}")
    logger.info(f"   {png_path}")
    
    plt.close(fig)


def create_comparison_table(
    vllm_ns: List[int],
    vllm_latencies: List[float],
    djinn_ns: List[int],
    djinn_latencies: List[float],
    cliff_point: Optional[int],
    output_path: Path,
):
    """Create a text table comparing vLLM and Djinn at each N."""
    
    lines = []
    lines.append("=" * 90)
    lines.append("DJINN vs vLLM: SCALABILITY COMPARISON")
    lines.append("=" * 90)
    lines.append("")
    
    # Create mapping
    vllm_map = {n: lat for n, lat in zip(vllm_ns, vllm_latencies)}
    djinn_map = {n: lat for n, lat in zip(djinn_ns, djinn_latencies)}
    
    all_ns = sorted(set(vllm_ns + djinn_ns))
    
    lines.append(f"{'N':>4} | {'vLLM P99 (ms)':>15} | {'vLLM Status':>20} | {'Djinn P99 (ms)':>15} | {'Djinn Status':>20}")
    lines.append("-" * 90)
    
    for n in all_ns:
        vllm_lat = vllm_map.get(n)
        djinn_lat = djinn_map.get(n)
        
        vllm_str = f"{vllm_lat:.0f}" if vllm_lat else "OOM"
        vllm_status = "✅ OK" if vllm_lat else "❌ CRASH"
        djinn_str = f"{djinn_lat:.0f}" if djinn_lat else "FAIL"
        djinn_status = "✅ OK" if djinn_lat else "❌ FAIL"
        
        if cliff_point and n == cliff_point:
            vllm_status = "🔴 CLIFF"
        
        lines.append(
            f"{n:>4} | {vllm_str:>15} | {vllm_status:>20} | {djinn_str:>15} | {djinn_status:>20}"
        )
    
    lines.append("-" * 90)
    lines.append("")
    
    if cliff_point and djinn_ns:
        max_djinn = max(djinn_ns)
        scaling_advantage = max_djinn / cliff_point if cliff_point > 0 else 0
        
        lines.append(f"vLLM Crash Point:        N={cliff_point}")
        lines.append(f"Djinn Max Agents:        N={max_djinn}")
        lines.append(f"Scaling Advantage:       {scaling_advantage:.2f}x")
        lines.append("")
    
    lines.append("=" * 90)
    
    # Write to file
    table_path = output_path.parent / f"{output_path.stem}_table.txt"
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"✅ Saved comparison table to: {table_path}")
    
    # Print to console
    for line in lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cliff visualization from cliff experiment results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to cliff experiment results JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("OSDI_Evaluation/exp1_semantic_scheduler/results/cliff_graph"),
        help="Output path for graph (without extension)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("OSDI_Evaluation/exp1_semantic_scheduler/results"),
        help="Directory containing cliff experiment results (auto-finds latest)"
    )
    
    args = parser.parse_args()
    
    # Find input file
    if args.input:
        results_file = args.input
    else:
        # Find the latest cliff_experiment JSON file
        cliff_files = list(args.results_dir.glob("cliff_experiment_*.json"))
        if not cliff_files:
            logger.error("No cliff experiment results found. Run cliff_experiment.py first.")
            return 1
        results_file = sorted(cliff_files)[-1]
        logger.info(f"Using latest results file: {results_file}")
    
    # Load results
    logger.info(f"Loading results from: {results_file}")
    results = load_cliff_results(results_file)
    
    # Extract data
    vllm_ns, vllm_latencies, djinn_ns, djinn_latencies, cliff_point = extract_data_for_plotting(results)
    
    logger.info(f"vLLM data points: {len(vllm_ns)}")
    logger.info(f"Djinn data points: {len(djinn_ns)}")
    logger.info(f"Cliff point: N={cliff_point}")
    
    # Create visualizations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nGenerating cliff graph...")
    create_cliff_graph(vllm_ns, vllm_latencies, djinn_ns, djinn_latencies, cliff_point, args.output)
    
    logger.info("\nGenerating comparison table...")
    create_comparison_table(vllm_ns, vllm_latencies, djinn_ns, djinn_latencies, cliff_point, args.output)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Visualization complete!")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
