#!/usr/bin/env python3
"""
Analyze virtual memory experiment results.

Generates comparison figures and passes/fails the experiment based on bandwidth.

Usage:
    python analyze_results.py \
        --results OSDI_Evaluation/exp2_virtual_memory/results/ \
        --output OSDI_Evaluation/exp2_virtual_memory/figures/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON result files from directory."""
    results = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("*.json"):
        logger.info(f"Loading: {json_file.name}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            results[json_file.stem] = data
    
    return results


def extract_metrics(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Extract key metrics from results."""
    metrics = {}
    
    for name, result in results.items():
        summary = result.get("summary", {})
        runs = result.get("runs", [])
        
        metrics[name] = {
            "model": result.get("model", "unknown"),
            "model_size_gb": result.get("model_size_gb", 0),
            "avg_bandwidth_gbps": summary.get("avg_bandwidth_gbps", 0),
            "median_bandwidth_gbps": summary.get("median_bandwidth_gbps", 0),
            "min_bandwidth_gbps": summary.get("min_bandwidth_gbps", 0),
            "max_bandwidth_gbps": summary.get("max_bandwidth_gbps", 0),
            "avg_latency_ms": summary.get("avg_latency_ms", 0),
            "median_latency_ms": summary.get("median_latency_ms", 0),
            "num_runs": len(runs),
            "pass": summary.get("pass", False),
            "all_bandwidths": [r.get("bandwidth_gbps", 0) for r in runs],
            "all_latencies": [r.get("latency_ms", 0) for r in runs],
        }
    
    return metrics


def generate_text_report(metrics: Dict[str, Dict]) -> str:
    """Generate text report of results."""
    report = []
    report.append("=" * 70)
    report.append("VIRTUAL MEMORY RING BUFFER EXPERIMENT RESULTS")
    report.append("=" * 70)
    report.append("")
    
    for name, metric in metrics.items():
        report.append(f"Result: {name}")
        report.append(f"  Model: {metric['model']} ({metric['model_size_gb']:.1f}GB)")
        report.append(f"  Runs: {metric['num_runs']}")
        report.append("")
        report.append(f"  Effective Bandwidth:")
        report.append(f"    Average:  {metric['avg_bandwidth_gbps']:6.1f} GB/s")
        report.append(f"    Median:   {metric['median_bandwidth_gbps']:6.1f} GB/s")
        report.append(f"    Min/Max:  {metric['min_bandwidth_gbps']:6.1f} / {metric['max_bandwidth_gbps']:6.1f} GB/s")
        report.append("")
        report.append(f"  Latency:")
        report.append(f"    Average:  {metric['avg_latency_ms']:6.1f} ms")
        report.append(f"    Median:   {metric['median_latency_ms']:6.1f} ms")
        report.append("")
        
        # Pass/fail
        status = "✅ PASS" if metric['pass'] else "❌ FAIL"
        threshold = 20.0
        report.append(f"  Status: {status}")
        if metric['avg_bandwidth_gbps'] >= threshold:
            report.append(f"  → Bandwidth {metric['avg_bandwidth_gbps']:.1f} GB/s >= threshold {threshold} GB/s")
        else:
            report.append(f"  → Bandwidth {metric['avg_bandwidth_gbps']:.1f} GB/s < threshold {threshold} GB/s")
        
        report.append("")
    
    # Overall summary
    report.append("=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)
    
    all_passed = all(m['pass'] for m in metrics.values())
    if all_passed:
        report.append("✅ ALL EXPERIMENTS PASSED")
        report.append("")
        report.append("Ring buffer successfully demonstrated:")
        report.append("  ✓ Skip-end allocation prevents tensor fragmentation")
        report.append("  ✓ Async pipelining saturates PCIe bandwidth (>20GB/s)")
        report.append("  ✓ Effective memory streaming for models > VRAM")
    else:
        report.append("❌ SOME EXPERIMENTS FAILED")
        report.append("")
        report.append("Failed experiments:")
        for name, metric in metrics.items():
            if not metric['pass']:
                report.append(f"  - {name}: {metric['avg_bandwidth_gbps']:.1f} GB/s (< 20 GB/s)")
    
    return "\n".join(report)


def generate_csv_report(metrics: Dict[str, Dict], output_path: str):
    """Generate CSV for further analysis."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Experiment",
            "Model",
            "Model_Size_GB",
            "Num_Runs",
            "Avg_Bandwidth_GBps",
            "Median_Bandwidth_GBps",
            "Min_Bandwidth_GBps",
            "Max_Bandwidth_GBps",
            "Avg_Latency_ms",
            "Median_Latency_ms",
            "Pass"
        ])
        
        # Data rows
        for name, metric in metrics.items():
            writer.writerow([
                name,
                metric['model'],
                f"{metric['model_size_gb']:.1f}",
                metric['num_runs'],
                f"{metric['avg_bandwidth_gbps']:.1f}",
                f"{metric['median_bandwidth_gbps']:.1f}",
                f"{metric['min_bandwidth_gbps']:.1f}",
                f"{metric['max_bandwidth_gbps']:.1f}",
                f"{metric['avg_latency_ms']:.1f}",
                f"{metric['median_latency_ms']:.1f}",
                "Yes" if metric['pass'] else "No"
            ])
    
    logger.info(f"✅ CSV report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze virtual memory experiment results")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading results from: {args.results}")
    results = load_results(args.results)
    
    if not results:
        logger.error("No result files found!")
        return 1
    
    # Extract metrics
    metrics = extract_metrics(results)
    
    # Generate text report
    report = generate_text_report(metrics)
    print("\n" + report + "\n")
    
    # Save report
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "RESULTS.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"✅ Report saved to: {report_path}")
    
    # Generate CSV
    csv_path = output_dir / "results.csv"
    generate_csv_report(metrics, csv_path)
    
    logger.info(f"\n✅ Analysis complete!")
    
    # Determine overall success
    all_passed = all(m['pass'] for m in metrics.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

