#!/usr/bin/env python3
"""
Generate Figure 7: Resume Latency vs. Layer Depth (Crossover Plot)

Reads combined JSON from run_experiment3_resume_latency.py and produces:
  - figure7_resume_latency.pdf (vector, publication-ready)
  - figure7_resume_latency.png (preview)
  - capabilities_table.json (layer 40 latency snapshot)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def extract_series(results: Dict, key: str) -> Dict[str, List[float]]:
    data = results.get("results", {}).get(key, {})
    layers = data.get("layers", [])
    series = data.get("results", [])
    x = [entry["layer"] for entry in series] if series else layers
    y = [entry["resume_latency_ms"] for entry in series] if series else []
    # Extract error bars (std dev)
    yerr = [entry.get("resume_latency_std_ms", 0) for entry in series] if series else []
    return {"x": x, "y": y, "yerr": yerr}


def build_table(results: Dict, target_layer: int = 40) -> Dict[str, any]:
    """Build comprehensive capabilities table with quantitative and qualitative metrics."""
    table = {}
    
    # Extract latencies for target layer
    for key in ["recompute", "manual_offload", "djinn"]:
        data = results.get("results", {}).get(key, {})
        series = data.get("results", [])
        latency_val = None
        data_mb = None
        
        for entry in series:
            if entry.get("layer") == target_layer:
                latency_val = entry.get("resume_latency_ms")
                # Try to extract data size if available
                if key == "manual_offload":
                    data_mb = entry.get("data_transferred_mb")
                break
        
        # Build entry for each method
        if key == "recompute":
            table[key] = {
                "resume_latency_ms": latency_val,
                "data_transferred_mb": 0,
                "supports_state_editing": "Yes",
                "max_concurrent_sessions": "OOM (unbounded compute)",
                "inference": "Recomputes activations from scratch; no memory overhead but linear compute cost with depth",
            }
        elif key == "manual_offload":
            table[key] = {
                "resume_latency_ms": latency_val,
                "data_transferred_mb": data_mb,
                "supports_state_editing": "Hard (requires manual .to() calls)",
                "max_concurrent_sessions": "50+ (with explicit CPU offload)",
                "inference": "Uses pinned CPU memory; speed-of-light PCIe baseline; user-managed code",
            }
        elif key == "djinn":
            table[key] = {
                "resume_latency_ms": latency_val,
                "data_transferred_mb": "Variable (managed internally)",
                "supports_state_editing": "Yes (transparent)",
                "max_concurrent_sessions": "50+ (automatic semantic management)",
                "inference": "Semantic OS; RPC overhead but transparent to user",
            }
    
    return table


def plot_crossover(series: Dict[str, Dict[str, List[float]]], output_dir: Path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.figsize": (10, 6),
            "lines.linewidth": 2.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot recompute (red dashed line)
    if series.get("recompute", {}).get("x"):
        ax.errorbar(
            series["recompute"]["x"],
            series["recompute"]["y"],
            yerr=series["recompute"].get("yerr", []),
            label="Stateless Recompute",
            color="#D9534F",
            linestyle="--",
            marker="o",
            capsize=5,
            capthick=2,
            elinewidth=1.5,
        )
    
    # Plot manual offload (green dash-dot line)
    if series.get("manual_offload", {}).get("x"):
        ax.errorbar(
            series["manual_offload"]["x"],
            series["manual_offload"]["y"],
            yerr=series["manual_offload"].get("yerr", []),
            label="Manual CPU Offload (Pinned)",
            color="#5CB85C",
            linestyle="-.",
            marker="s",
            capsize=5,
            capthick=2,
            elinewidth=1.5,
        )
    
    # Plot Djinn (blue solid line)
    if series.get("djinn", {}).get("x"):
        ax.errorbar(
            series["djinn"]["x"],
            series["djinn"]["y"],
            yerr=series["djinn"].get("yerr", []),
            label="Djinn (IO_WAIT → Resume)",
            color="#0275D8",
            linestyle="-",
            marker="^",
            capsize=5,
            capthick=2,
            elinewidth=1.5,
        )

    # Find and annotate crossover point (where Recompute exceeds Offload)
    recompute_x = series.get("recompute", {}).get("x", [])
    recompute_y = series.get("recompute", {}).get("y", [])
    offload_x = series.get("manual_offload", {}).get("x", [])
    offload_y = series.get("manual_offload", {}).get("y", [])
    
    if recompute_x and offload_x and recompute_y and offload_y:
        # Find the first point where recompute > offload
        for i, rx in enumerate(recompute_x):
            if i < len(recompute_y):
                for j, ox in enumerate(offload_x):
                    if j < len(offload_y) and rx == ox and recompute_y[i] > offload_y[j]:
                        # Crossover found
                        ax.annotate(
                            'Crossover',
                            xy=(rx, recompute_y[i]),
                            xytext=(rx + 3, recompute_y[i] + 20),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1),
                            fontsize=11,
                            fontweight='bold',
                        )
                        break

    ax.set_xlabel("Breakpoint Depth (Layer Index)")
    ax.set_ylabel("Resume Latency (ms)")
    ax.set_xlim(0, 45)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure7_resume_latency.pdf"
    png_path = output_dir / "figure7_resume_latency.png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"✅ Saved: {pdf_path}")
    print(f"✅ Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate resume latency crossover plot")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/exp3_resume_results/resume_latency_combined.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/Djinn/OSDI_Evaluation/exp3_whitebox_debugging"),
    )
    parser.add_argument("--target-layer", type=int, default=40)
    args = parser.parse_args()

    results = load_results(args.input)
    series = {
        "recompute": extract_series(results, "recompute"),
        "manual_offload": extract_series(results, "manual_offload"),
        "djinn": extract_series(results, "djinn"),
    }

    plot_crossover(series, args.output_dir)

    table = build_table(results, target_layer=args.target_layer)
    table_path = args.output_dir / "capabilities_table.json"
    with open(table_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"✅ Capabilities table snapshot saved to {table_path}")
    
    # Also print human-readable table
    print("\n" + "="*100)
    print("CAPABILITIES COMPARISON TABLE (Layer 40)")
    print("="*100)
    headers = ["Method", "Resume Latency (ms)", "Data Xfer (MB)", "State Editing", "Max Sessions"]
    print(f"{headers[0]:25} {headers[1]:20} {headers[2]:20} {headers[3]:25} {headers[4]:<30}")
    print("-"*100)
    for method, metrics in table.items():
        latency = f"{metrics['resume_latency_ms']:.1f}" if metrics['resume_latency_ms'] else "N/A"
        data = f"{metrics['data_transferred_mb']:.1f}" if isinstance(metrics['data_transferred_mb'], (int, float)) else str(metrics['data_transferred_mb'])
        editing = metrics['supports_state_editing']
        sessions = metrics['max_concurrent_sessions']
        print(f"{method:25} {latency:20} {data:20} {editing:25} {sessions:<30}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
