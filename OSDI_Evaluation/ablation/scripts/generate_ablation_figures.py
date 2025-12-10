"""
Generate publication-quality figures for OSDI ablation studies.

This script loads JSON results from ablation experiments and generates
high-quality PDF figures suitable for inclusion in the OSDI paper.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping figure generation")


# Configure matplotlib for publication quality
if MATPLOTLIB_AVAILABLE:
    mpl.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
    })


def load_results(results_dir: Path) -> Dict[int, Dict]:
    """Load all ablation results from JSON files."""
    results = {}
    
    for i in range(1, 5):
        json_file = results_dir / f"ablation_{i}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                results[i] = json.load(f)
        else:
            print(f"Warning: ablation_{i}.json not found")
    
    return results


def generate_os_tax_figure(results: Dict, output_path: str):
    """Generate figure for OS Tax ablation (Ablation 1)."""
    if not MATPLOTLIB_AVAILABLE or 1 not in results:
        return
    
    print("Generating OS Tax figure...")
    
    ablation_1 = results[1]
    operations = []
    native_latencies = []
    warm_latencies = []
    
    for op_name, op_data in ablation_1.items():
        if isinstance(op_data, dict) and 'native' in op_data and 'djinn_warm' in op_data:
            operations.append(op_name)
            native_latencies.append(op_data['native'].get('mean', 0))
            warm_latencies.append(op_data['djinn_warm'].get('mean', 0))
    
    if not operations:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(operations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, native_latencies, width, label='Native PyTorch', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, warm_latencies, width, label='Djinn (Warm Cache)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Operation Type', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Ablation 1: OS Tax - Framework Overhead', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_comparison_figure(results_dir: Path, output_path: str):
    """Generate comparison figure showing all four ablations together."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print("Generating comparison figure...")
    
    # This would load and visualize key metrics from all ablations
    # For now, create a placeholder summary
    fig = plt.figure(figsize=(14, 10))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('OSDI Ablation Study: System Microbenchmarks (Section 5.1)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # For each subplot, add placeholder text
    labels = [
        'Ablation 1: OS Tax\n(Dispatch Overhead)',
        'Ablation 2: Session Arena\n(Memory Decomposition)',
        'Ablation 3: Plan Cache\n(Caching Effectiveness)',
        'Ablation 4: Semantic Signals\n(Scheduling Value)',
    ]
    
    for idx, (ax, label) in enumerate(zip(gs, labels)):
        ax_obj = fig.add_subplot(ax)
        ax_obj.text(0.5, 0.5, label, ha='center', va='center', fontsize=12, fontweight='bold')
        ax_obj.set_xlim(0, 1)
        ax_obj.set_ylim(0, 1)
        ax_obj.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_summary_latex_document(results_dir: Path, output_path: str):
    """Generate a LaTeX document with all ablation tables and figures."""
    
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        r"\usepackage[margin=1in]{geometry}",
        "",
        r"\title{OSDI Ablation Study: System Microbenchmarks (Section 5.1)}",
        r"\author{Djinn Project}",
        r"\date{\today}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section*{Executive Summary}",
        "",
        r"This document presents four ablation studies validating the engineering contributions",
        r"of the Djinn Tensor Operating System. Each ablation isolates one architectural component",
        r"and quantifies its impact on system performance.",
        "",
        r"\section*{Ablation 1: OS Tax (Dispatch Overhead Analysis)}",
        "",
        r"See \texttt{ablation\_os\_tax\_table.tex} for results.",
        "",
        r"\section*{Ablation 2: Session Arena Decomposition}",
        "",
        r"See \texttt{ablation\_arena\_table.tex} for results.",
        "",
        r"\section*{Ablation 3: Plan Cache Effectiveness}",
        "",
        r"See \texttt{ablation\_cache\_table.tex} for results.",
        "",
        r"\section*{Ablation 4: Semantic Signal Value}",
        "",
        r"See \texttt{ablation\_signals\_table.tex} for results.",
        "",
        r"\end{document}",
    ]
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"✅ Saved LaTeX document: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality figures for ablations")
    parser.add_argument('--results-dir', type=str,
                        default='/home/ubuntu/Djinn/OSDI_Evaluation/ablation/results',
                        help='Directory containing ablation results JSON files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures (defaults to results-dir/figures)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating ablation figures...")
    print(f"  Input: {results_dir}")
    print(f"  Output: {output_dir}")
    
    # Load results
    results = load_results(results_dir)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib not available. Skipping figure generation.")
        print("Install with: pip install matplotlib")
        return 1
    
    # Generate individual figures
    if 1 in results:
        generate_os_tax_figure(results, str(output_dir / 'ablation_1_os_tax.pdf'))
    
    # Generate comparison figure
    generate_comparison_figure(results_dir, str(output_dir / 'ablation_summary.pdf'))
    
    # Generate LaTeX document
    generate_summary_latex_document(results_dir, str(output_dir / 'ablation_summary.tex'))
    
    print("\n✅ Figure generation complete!")
    print(f"Check {output_dir} for PDF files")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
