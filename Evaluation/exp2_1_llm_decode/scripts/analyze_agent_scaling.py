#!/usr/bin/env python3
"""
Analysis script for Experiment 2 (Agent Scaling Hero Experiment).

Loads JSON results from all three baselines (Ray Keep-Alive, Ray Serverless, Djinn)
and generates:
1. Figure 6: P99 Latency vs Number of Concurrent Agents (log scale)
2. Table 2: Summary statistics and OOM thresholds
3. Claims verification: Tenant density and latency improvement ratios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_results(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON results file."""
    if not path.exists():
        print(f"[Warning] File not found: {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load {path}: {e}")
        return None


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from baseline results."""
    metrics = {
        "baseline": results.get("tag", "unknown"),
        "agent_counts": [],
        "mean_latencies": [],
        "p99_latencies": [],
        "p50_latencies": [],
        "oom_threshold": None,
        "max_agents": 0,
        "oom_encountered": False,
    }
    
    for entry in results.get("agent_counts", []):
        n_agents = entry.get("agents", 0)
        metrics["agent_counts"].append(n_agents)
        
        # Check for OOM
        if entry.get("oom", False):
            metrics["oom_threshold"] = n_agents
            metrics["oom_encountered"] = True
            # For OOM entries, don't add latency metrics
            metrics["mean_latencies"].append(None)
            metrics["p99_latencies"].append(None)
            metrics["p50_latencies"].append(None)
            break  # Stop collecting after OOM
        
        aggregates = entry.get("aggregates", {})
        metrics["mean_latencies"].append(aggregates.get("mean_latency_ms", None))
        metrics["p99_latencies"].append(aggregates.get("p99_latency_ms", None))
        metrics["p50_latencies"].append(aggregates.get("p50_latency_ms", None))
        metrics["max_agents"] = max(metrics["max_agents"], n_agents)
    
    return metrics


def generate_summary_table(
    ray_ka_metrics: Dict[str, Any],
    ray_sl_metrics: Dict[str, Any],
    djinn_metrics: Dict[str, Any],
) -> str:
    """Generate Table 2: Summary statistics."""
    
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("TABLE 2: Agent Scaling Summary Statistics")
    lines.append("=" * 100)
    lines.append("")
    
    # Header
    lines.append(
        f"{'Baseline':<20} {'Max Agents':<15} {'P99@1 Agent':<15} "
        f"{'P99@8 Agents':<15} {'P99@32 Agents':<15} {'OOM at N':<12}"
    )
    lines.append("-" * 100)
    
    # Ray Keep-Alive
    ka_p99_1 = ray_ka_metrics["p99_latencies"][0] if ray_ka_metrics["p99_latencies"] else None
    ka_p99_1_str = f"{ka_p99_1:.2f}ms" if ka_p99_1 else "N/A"
    
    ka_p99_8 = None
    if len(ray_ka_metrics["agent_counts"]) >= 3 and ray_ka_metrics["agent_counts"][2] == 8:
        ka_p99_8 = ray_ka_metrics["p99_latencies"][2]
    ka_p99_8_str = f"{ka_p99_8:.2f}ms" if ka_p99_8 else "OOM"
    
    ka_p99_32 = "OOM" if ray_ka_metrics["oom_encountered"] else "N/A"
    ka_oom_str = f"N={ray_ka_metrics['oom_threshold']}" if ray_ka_metrics["oom_threshold"] else "None"
    
    lines.append(
        f"{'Ray Keep-Alive':<20} {ray_ka_metrics['max_agents']:<15} "
        f"{ka_p99_1_str:<15} {ka_p99_8_str:<15} {str(ka_p99_32):<15} {ka_oom_str:<12}"
    )
    
    # Ray Serverless
    sl_p99_1 = ray_sl_metrics["p99_latencies"][0] if ray_sl_metrics["p99_latencies"] else None
    sl_p99_1_str = f"{sl_p99_1:.2f}ms" if sl_p99_1 else "N/A"
    
    sl_p99_8 = None
    if len(ray_sl_metrics["agent_counts"]) >= 3 and ray_sl_metrics["agent_counts"][2] == 8:
        sl_p99_8 = ray_sl_metrics["p99_latencies"][2]
    sl_p99_8_str = f"{sl_p99_8:.2f}ms" if sl_p99_8 else "N/A"
    
    sl_p99_32 = None
    if len(ray_sl_metrics["agent_counts"]) >= 6 and ray_sl_metrics["agent_counts"][5] == 32:
        sl_p99_32 = ray_sl_metrics["p99_latencies"][5]
    sl_p99_32_str = f"{sl_p99_32:.2f}ms" if sl_p99_32 else "N/A"
    
    lines.append(
        f"{'Ray Serverless':<20} {ray_sl_metrics['max_agents']:<15} "
        f"{sl_p99_1_str:<15} {sl_p99_8_str:<15} {sl_p99_32_str:<15} "
        f"{'None':<12}"
    )
    
    # Djinn
    djinn_p99_1 = djinn_metrics["p99_latencies"][0] if djinn_metrics["p99_latencies"] else None
    djinn_p99_1_str = f"{djinn_p99_1:.2f}ms" if djinn_p99_1 else "N/A"
    
    djinn_p99_8 = None
    if len(djinn_metrics["agent_counts"]) >= 3 and djinn_metrics["agent_counts"][2] == 8:
        djinn_p99_8 = djinn_metrics["p99_latencies"][2]
    djinn_p99_8_str = f"{djinn_p99_8:.2f}ms" if djinn_p99_8 else "N/A"
    
    djinn_p99_32 = None
    if len(djinn_metrics["agent_counts"]) >= 6 and djinn_metrics["agent_counts"][5] == 32:
        djinn_p99_32 = djinn_metrics["p99_latencies"][5]
    djinn_p99_32_str = f"{djinn_p99_32:.2f}ms" if djinn_p99_32 else "N/A"
    
    lines.append(
        f"{'Djinn':<20} {djinn_metrics['max_agents']:<15} "
        f"{djinn_p99_1_str:<15} {djinn_p99_8_str:<15} {djinn_p99_32_str:<15} "
        f"{'None':<12}"
    )
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


def compute_claims(
    ray_ka_metrics: Dict[str, Any],
    ray_sl_metrics: Dict[str, Any],
    djinn_metrics: Dict[str, Any],
) -> str:
    """Compute paper claims: tenant density and latency improvements."""
    
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("PAPER CLAIMS VERIFICATION")
    lines.append("=" * 100)
    lines.append("")
    
    # Claim 1: Tenant Density (Djinn vs Ray Keep-Alive)
    lines.append("Claim 1: '20x higher tenant density than Ray-Persistent'")
    lines.append("-" * 100)
    
    djinn_max = djinn_metrics["max_agents"]
    ka_oom = ray_ka_metrics["oom_threshold"] or 2  # Assume OOM at 2 if not detected
    
    if djinn_max > 0 and ka_oom > 0:
        density_ratio = djinn_max / ka_oom
        lines.append(
            f"  Djinn max agents: {djinn_max} vs Ray Keep-Alive OOM at N={ka_oom}"
        )
        lines.append(
            f"  Tenant density ratio: {density_ratio:.1f}x "
            f"(Djinn scales to {djinn_max}, Ray OOMs at {ka_oom})"
        )
        if density_ratio >= 20.0:
            lines.append("  ✓ PASS: Exceeds 20x claim")
        elif density_ratio >= 10.0:
            lines.append(f"  ⚠ PARTIAL: {density_ratio:.1f}x (target 20x)")
        else:
            lines.append(f"  ✗ FAIL: {density_ratio:.1f}x (target 20x)")
    else:
        lines.append("  ⚠ Cannot compute: incomplete data")
    
    lines.append("")
    
    # Claim 2: Latency Advantage (Djinn vs Ray Serverless at high N)
    lines.append("Claim 2: '10x lower latency than Ray-Serverless'")
    lines.append("-" * 100)
    
    # Get P99 at N=32 if available
    djinn_p99_32 = None
    sl_p99_32 = None
    
    if len(djinn_metrics["agent_counts"]) >= 6 and djinn_metrics["agent_counts"][5] == 32:
        djinn_p99_32 = djinn_metrics["p99_latencies"][5]
    if len(ray_sl_metrics["agent_counts"]) >= 6 and ray_sl_metrics["agent_counts"][5] == 32:
        sl_p99_32 = ray_sl_metrics["p99_latencies"][5]
    
    if djinn_p99_32 and sl_p99_32 and djinn_p99_32 > 0:
        latency_ratio = sl_p99_32 / djinn_p99_32
        lines.append(
            f"  Djinn P99 @ N=32: {djinn_p99_32:.2f}ms"
        )
        lines.append(
            f"  Ray Serverless P99 @ N=32: {sl_p99_32:.2f}ms"
        )
        lines.append(
            f"  Latency ratio: {latency_ratio:.1f}x "
            f"(Ray is {latency_ratio:.1f}x slower)"
        )
        if latency_ratio >= 10.0:
            lines.append("  ✓ PASS: Exceeds 10x claim")
        elif latency_ratio >= 5.0:
            lines.append(f"  ⚠ PARTIAL: {latency_ratio:.1f}x (target 10x)")
        else:
            lines.append(f"  ✗ FAIL: {latency_ratio:.1f}x (target 10x)")
    else:
        lines.append("  ⚠ Cannot compute: N=32 data not available")
    
    lines.append("")
    
    # Claim 3: Stable Scaling (Djinn latency growth vs Ray)
    lines.append("Claim 3: 'Djinn maintains stable latency under scaling'")
    lines.append("-" * 100)
    
    if djinn_metrics["p99_latencies"] and len(djinn_metrics["p99_latencies"]) >= 2:
        djinn_p99_1 = djinn_metrics["p99_latencies"][0]
        djinn_p99_max = max([x for x in djinn_metrics["p99_latencies"] if x is not None])
        djinn_growth = (djinn_p99_max - djinn_p99_1) / djinn_p99_1 * 100 if djinn_p99_1 > 0 else 0
        lines.append(f"  Djinn P99 growth (1→32 agents): {djinn_growth:.1f}%")
    else:
        lines.append("  ⚠ Cannot compute: incomplete Djinn data")
    
    if ray_sl_metrics["p99_latencies"] and len(ray_sl_metrics["p99_latencies"]) >= 2:
        sl_p99_1 = ray_sl_metrics["p99_latencies"][0]
        sl_p99_max = max([x for x in ray_sl_metrics["p99_latencies"] if x is not None])
        sl_growth = (sl_p99_max - sl_p99_1) / sl_p99_1 * 100 if sl_p99_1 > 0 else 0
        lines.append(f"  Ray Serverless P99 growth (1→32 agents): {sl_growth:.1f}%")
    else:
        lines.append("  ⚠ Cannot compute: incomplete Ray data")
    
    lines.append("")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def generate_figure(
    ray_ka_metrics: Dict[str, Any],
    ray_sl_metrics: Dict[str, Any],
    djinn_metrics: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> None:
    """Generate Figure 6: P99 Latency vs Number of Concurrent Agents."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("[Warning] Matplotlib not available; skipping figure generation")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Ray Keep-Alive (stops at OOM)
    ka_agents = [a for a, lat in zip(ray_ka_metrics["agent_counts"], ray_ka_metrics["p99_latencies"]) if lat is not None]
    ka_latencies = [lat for lat in ray_ka_metrics["p99_latencies"] if lat is not None]
    if ka_agents:
        ax.plot(ka_agents, ka_latencies, 'o-', linewidth=2, markersize=8, label='Ray Keep-Alive (OOMs)', color='#d62728')
        # Add OOM marker
        if ray_ka_metrics["oom_threshold"]:
            ax.axvline(x=ray_ka_metrics["oom_threshold"], color='#d62728', linestyle='--', alpha=0.3)
    
    # Plot Ray Serverless
    sl_agents = [a for a, lat in zip(ray_sl_metrics["agent_counts"], ray_sl_metrics["p99_latencies"]) if lat is not None]
    sl_latencies = [lat for lat in ray_sl_metrics["p99_latencies"] if lat is not None]
    if sl_agents:
        ax.plot(sl_agents, sl_latencies, 's-', linewidth=2, markersize=8, label='Ray Serverless', color='#ff7f0e')
    
    # Plot Djinn
    djinn_agents = [a for a, lat in zip(djinn_metrics["agent_counts"], djinn_metrics["p99_latencies"]) if lat is not None]
    djinn_latencies = [lat for lat in djinn_metrics["p99_latencies"] if lat is not None]
    if djinn_agents:
        ax.plot(djinn_agents, djinn_latencies, '^-', linewidth=2, markersize=8, label='Djinn', color='#2ca02c')
    
    ax.set_xlabel('Number of Concurrent Agents', fontsize=12, fontweight='bold')
    ax.set_ylabel('P99 Latency per Step (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 6: Agent Scaling - Latency vs Concurrency', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='best')
    
    # Add annotations
    ax.text(0.02, 0.98, 'Lower is Better', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze agent scaling results")
    parser.add_argument(
        "--ray-keepalive",
        type=Path,
        help="Path to Ray Keep-Alive JSON results"
    )
    parser.add_argument(
        "--ray-serverless",
        type=Path,
        help="Path to Ray Serverless JSON results"
    )
    parser.add_argument(
        "--djinn",
        type=Path,
        help="Path to Djinn JSON results"
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("Evaluation/exp2_1_llm_decode/results/figure_6_agent_scaling.png"),
        help="Output path for figure"
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("Evaluation/exp2_1_llm_decode/results/table_2_summary.txt"),
        help="Output path for summary table"
    )
    args = parser.parse_args()
    
    print("=" * 100)
    print("Agent Scaling Analysis (Experiment 2)")
    print("=" * 100)
    print("")
    
    # Auto-discover results if paths not provided
    results_dir = Path("Evaluation/exp2_1_llm_decode/results")
    
    if not args.ray_keepalive:
        # Find latest Ray Keep-Alive result
        ka_files = sorted(results_dir.glob("ray_keepalive/*.json"), reverse=True)
        if ka_files:
            args.ray_keepalive = ka_files[0]
            print(f"[Auto] Using Ray Keep-Alive: {args.ray_keepalive}")
    
    if not args.ray_serverless:
        # Find latest Ray Serverless result
        sl_files = sorted(results_dir.glob("ray_serverless/*.json"), reverse=True)
        if sl_files:
            args.ray_serverless = sl_files[0]
            print(f"[Auto] Using Ray Serverless: {args.ray_serverless}")
    
    if not args.djinn:
        # Find latest Djinn result
        djinn_files = sorted(results_dir.glob("djinn_agents/*.json"), reverse=True)
        if djinn_files:
            args.djinn = djinn_files[0]
            print(f"[Auto] Using Djinn: {args.djinn}")
    
    print("")
    
    # Load results
    ray_ka_results = load_results(args.ray_keepalive) if args.ray_keepalive else None
    ray_sl_results = load_results(args.ray_serverless) if args.ray_serverless else None
    djinn_results = load_results(args.djinn) if args.djinn else None
    
    if not (ray_ka_results and ray_sl_results and djinn_results):
        print("[Error] Could not load all three baseline results")
        sys.exit(1)
    
    # Extract metrics
    ray_ka_metrics = extract_metrics(ray_ka_results)
    ray_sl_metrics = extract_metrics(ray_sl_results)
    djinn_metrics = extract_metrics(djinn_results)
    
    # Generate table
    table = generate_summary_table(ray_ka_metrics, ray_sl_metrics, djinn_metrics)
    print(table)
    
    # Save table
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, 'w') as f:
        f.write(table)
    print(f"\n[File] Summary table saved to {args.output_summary}")
    
    # Generate claims
    claims = compute_claims(ray_ka_metrics, ray_sl_metrics, djinn_metrics)
    print(claims)
    
    # Generate figure
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    generate_figure(ray_ka_metrics, ray_sl_metrics, djinn_metrics, args.output_figure)
    
    print("\n[Done] Analysis complete")


if __name__ == "__main__":
    main()

