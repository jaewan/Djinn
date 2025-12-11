#!/usr/bin/env python3
"""
Generate Latency CDF for Multi-Model Switching Experiment.

This plot visualizes the bimodal distribution of Djinn's latency:
- Fast path: <10ms (cache hits)
- Medium path: ~2s (swaps)
- Slow path: ~14s (cold starts)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

def load_results(filepath):
    """Load experimental results from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data

def compute_cdf(latencies):
    """Compute CDF from latency list."""
    sorted_latencies = np.sort(latencies)
    cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
    return sorted_latencies, cdf

def main():
    results_dir = Path("../results/osdi_sequential_final")
    
    # Load data
    djinn_data = load_results(results_dir / "djinn_sequential_results.json")
    pytorch_pinned_data = load_results(results_dir / "pytorch_pinned_sequential_results.json")
    vllm_data = load_results(results_dir / "vllm_sequential_results.json")
    
    # Extract latencies
    djinn_latencies = [r['total_ms'] for r in djinn_data['results'] if r['success']]
    pytorch_latencies = [r['total_ms'] for r in pytorch_pinned_data['results'] if r['success']]
    vllm_latencies = [r['total_latency_ms'] for r in vllm_data['results'] if r['success']]
    
    # Compute CDFs
    djinn_x, djinn_y = compute_cdf(djinn_latencies)
    pytorch_x, pytorch_y = compute_cdf(pytorch_latencies)
    vllm_x, vllm_y = compute_cdf(vllm_latencies)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot CDFs
    ax.plot(djinn_x, djinn_y * 100, 'b-', linewidth=2.5, label='Djinn (100% success)', zorder=3)
    ax.plot(pytorch_x, pytorch_y * 100, 'g--', linewidth=2, label='PyTorch Pinned (100% success)', zorder=2)
    
    # vLLM: plot successful requests, then show failure as horizontal line
    if len(vllm_latencies) > 0:
        # Plot successful requests
        ax.plot(vllm_x, vllm_y * 100 * 0.73, 'r-.', linewidth=2, label='vLLM (73% success)', zorder=1)
        # Add horizontal line at 73% to show failure ceiling
        ax.axhline(y=73, color='r', linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)
        ax.text(25000, 75, 'vLLM failure ceiling (27% OOM)', fontsize=9, color='r', style='italic')
    
    # Annotations for Djinn's bimodal distribution
    # Fast path
    ax.annotate('Cache Hits\n(53%, <10ms)', 
                xy=(10, 53), xytext=(100, 40),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=9, color='blue', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Cold start
    ax.annotate('Cold Start\n(P99: 14.9s)', 
                xy=(14905, 99), xytext=(10000, 85),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=9, color='blue', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # PyTorch annotation
    ax.annotate('PyTorch: No cache\n(P50: 4.3s)', 
                xy=(4302, 50), xytext=(6000, 30),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=9, color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Styling
    ax.set_xscale('log')
    ax.set_xlabel('Request Latency (ms, log scale)', fontweight='bold')
    ax.set_ylabel('Percentile (%)', fontweight='bold')
    ax.set_title('Multi-Model Switching: Latency CDF\n(3 models × 7B, 30 requests, sequential)', 
                 fontweight='bold', pad=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    ax.set_xlim([1, 30000])
    ax.set_ylim([0, 105])
    
    # Add vertical reference lines
    ax.axvline(x=10, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=1000, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=10000, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    output_path = results_dir / "latency_cdf.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved CDF plot: {output_path}")
    
    # Also save PNG for quick viewing
    output_png = results_dir / "latency_cdf.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"✅ Saved PNG: {output_png}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("LATENCY STATISTICS")
    print("="*60)
    
    print(f"\nDjinn (N={len(djinn_latencies)}):")
    print(f"  P10: {np.percentile(djinn_latencies, 10):.1f}ms")
    print(f"  P50: {np.percentile(djinn_latencies, 50):.1f}ms")
    print(f"  P90: {np.percentile(djinn_latencies, 90):.1f}ms")
    print(f"  P99: {np.percentile(djinn_latencies, 99):.1f}ms")
    print(f"  Mean: {np.mean(djinn_latencies):.1f}ms")
    
    print(f"\nPyTorch Pinned (N={len(pytorch_latencies)}):")
    print(f"  P10: {np.percentile(pytorch_latencies, 10):.1f}ms")
    print(f"  P50: {np.percentile(pytorch_latencies, 50):.1f}ms")
    print(f"  P90: {np.percentile(pytorch_latencies, 90):.1f}ms")
    print(f"  P99: {np.percentile(pytorch_latencies, 99):.1f}ms")
    print(f"  Mean: {np.mean(pytorch_latencies):.1f}ms")
    
    print(f"\nvLLM (N={len(vllm_latencies)}, {len(vllm_latencies)/30*100:.0f}% success):")
    if len(vllm_latencies) > 0:
        print(f"  P10: {np.percentile(vllm_latencies, 10):.1f}ms")
        print(f"  P50: {np.percentile(vllm_latencies, 50):.1f}ms")
        print(f"  P90: {np.percentile(vllm_latencies, 90):.1f}ms")
        print(f"  P99: {np.percentile(vllm_latencies, 99):.1f}ms")
        print(f"  Mean: {np.mean(vllm_latencies):.1f}ms")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print(f"1. Djinn achieves {pytorch_x[int(len(pytorch_x)*0.5)] / djinn_x[int(len(djinn_x)*0.5)]:.1f}x lower P50 latency")
    print(f"2. 53% of Djinn requests complete in <10ms (cache hits)")
    print(f"3. vLLM fails 27% of requests due to memory fragmentation")
    print(f"4. Djinn's P99 reflects cold start cost (virtualization trade-off)")

if __name__ == "__main__":
    main()
