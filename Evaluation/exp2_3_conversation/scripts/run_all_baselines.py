#!/usr/bin/env python3
"""
Run Experiment 2.3 (Conversational AI) with all 4 baselines:
1. native_pytorch - Local execution (upper bound)
2. semantic_blind - Djinn remote with all semantic features disabled
3. partially_aware - Djinn remote with model caching but no KV awareness
4. full_djinn - Djinn remote with all semantic features enabled

This script validates the semantic-blind baseline and generates comparison metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.djinn_init import ensure_initialized_before_async


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="EleutherAI/gpt-j-6b", help="HF model identifier")
    parser.add_argument("--conversation-file", type=Path, default=Path("Evaluation/exp2_3_conversation/prompts/conversation.json"), help="Path to conversation JSON")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens per turn")
    parser.add_argument("--runs", type=int, default=30, help="Number of measured runs per baseline")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Unmeasured warmup runs")
    parser.add_argument("--device", default="cuda:0", help="torch device string (for native baseline)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--djinn-server", default="localhost:5556", help="Djinn server address")
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_3_conversation/results"))
    parser.add_argument("--tag", default="all_baselines", help="Tag for output files")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-gpu", action="store_true", help="Collect GPU utilization")
    parser.add_argument("--skip-native", action="store_true", help="Skip native PyTorch baseline")
    parser.add_argument("--skip-semantic-blind", action="store_true", help="Skip semantic-blind baseline")
    parser.add_argument("--skip-partially-aware", action="store_true", help="Skip partially-aware baseline")
    parser.add_argument("--skip-full-djinn", action="store_true", help="Skip full Djinn baseline")
    return parser.parse_args()


def run_native_baseline(args: argparse.Namespace) -> Path:
    """Run native PyTorch baseline (local execution)."""
    print("\n" + "="*80)
    print("Running NATIVE PYTORCH baseline (local execution)")
    print("="*80)
    
    script_path = REPO_ROOT / "Evaluation/exp2_3_conversation/scripts/run_local_conversation_baseline.py"
    output_dir = args.output_dir / "native_pytorch"
    
    cmd = [
        sys.executable, str(script_path),
        "--model-id", args.model_id,
        "--conversation-file", str(args.conversation_file),
        "--max-new-tokens", str(args.max_new_tokens),
        "--runs", str(args.runs),
        "--warmup-runs", str(args.warmup_runs),
        "--device", args.device,
        "--dtype", args.dtype,
        "--output-dir", str(output_dir),
        "--tag", "native_pytorch",
        "--seed", str(args.seed),
    ]
    if args.sample_gpu:
        cmd.append("--sample-gpu")
    
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Native baseline failed:\n{result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    # Find the output file
    output_files = sorted(
        output_dir.glob("conversation_native_pytorch_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if output_files:
        return output_files[-1]
    raise RuntimeError(f"No output file found in {output_dir}")


def run_remote_baseline(args: argparse.Namespace, baseline_name: str, semantic_aware: bool, use_session: bool = True) -> Path:
    """Run a remote Djinn baseline."""
    print("\n" + "="*80)
    print(f"Running {baseline_name.upper()} baseline (Djinn remote)")
    print(f"  Semantic-aware: {semantic_aware}, Use session: {use_session}")
    print("="*80)
    
    script_path = REPO_ROOT / "Evaluation/exp2_3_conversation/scripts/run_local_conversation_baseline.py"
    output_dir = args.output_dir / baseline_name
    
    cmd = [
        sys.executable, str(script_path),
        "--model-id", args.model_id,
        "--conversation-file", str(args.conversation_file),
        "--max-new-tokens", str(args.max_new_tokens),
        "--runs", str(args.runs),
        "--warmup-runs", str(args.warmup_runs),
        "--backend", "djinn",
        "--djinn-server", args.djinn_server,
        "--dtype", args.dtype,
        "--output-dir", str(output_dir),
        "--tag", baseline_name,
        "--seed", str(args.seed),
    ]
    
    if not semantic_aware:
        cmd.append("--no-semantic-aware")
    if not use_session:
        cmd.append("--no-semantic-session")
    
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {baseline_name} baseline failed:\n{result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    # Find the output file
    output_files = sorted(
        output_dir.glob(f"conversation_{baseline_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if output_files:
        return output_files[-1]
    raise RuntimeError(f"No output file found in {output_dir}")


def generate_comparison_report(results: Dict[str, Path], args: argparse.Namespace) -> Dict[str, Any]:
    """Generate comparison report from all baseline results."""
    report = {
        "experiment": "exp2_3_conversation",
        "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens,
        "runs_per_baseline": args.runs,
        "baselines": {},
        "comparisons": {},
    }
    
    baseline_data = {}
    for baseline_name, result_file in results.items():
        with open(result_file) as f:
            data = json.load(f)
        baseline_data[baseline_name] = data
        
        # Extract aggregates from runs
        if data.get("runs") and len(data["runs"]) > 0:
            all_turn_latencies = []
            all_total_times = []
            total_data_transfer = 0.0
            
            for run in data["runs"]:
                if "turns" in run:
                    for turn in run["turns"]:
                        all_turn_latencies.append(turn.get("latency_ms", 0))
                        total_data_transfer += turn.get("host_to_device_mb", 0) + turn.get("device_to_host_mb", 0)
                all_total_times.append(run.get("total_time_ms", 0))
            
            import statistics
            report["baselines"][baseline_name] = {
                "mean_turn_latency_ms": statistics.mean(all_turn_latencies) if all_turn_latencies else None,
                "mean_total_time_ms": statistics.mean(all_total_times) if all_total_times else None,
                "total_data_transfer_mb": total_data_transfer / len(data["runs"]) if data["runs"] else None,
                "result_file": str(result_file),
            }
    
    # Generate comparisons
    if "native_pytorch" in baseline_data and "full_djinn" in baseline_data:
        native = report["baselines"]["native_pytorch"]
        full = report["baselines"]["full_djinn"]
        report["comparisons"]["full_djinn_vs_native"] = {
            "latency_overhead_pct": ((full["mean_total_time_ms"] / native["mean_total_time_ms"]) - 1) * 100 if native["mean_total_time_ms"] else None,
        }
    
    if "semantic_blind" in baseline_data and "full_djinn" in baseline_data:
        blind = report["baselines"]["semantic_blind"]
        full = report["baselines"]["full_djinn"]
        report["comparisons"]["full_djinn_vs_semantic_blind"] = {
            "speedup": blind["mean_total_time_ms"] / full["mean_total_time_ms"] if full["mean_total_time_ms"] else None,
            "data_savings_pct": ((blind["total_data_transfer_mb"] - full["total_data_transfer_mb"]) / blind["total_data_transfer_mb"] * 100) if blind.get("total_data_transfer_mb") else None,
        }
    
    if "partially_aware" in baseline_data and "full_djinn" in baseline_data:
        partial = report["baselines"]["partially_aware"]
        full = report["baselines"]["full_djinn"]
        report["comparisons"]["full_djinn_vs_partially_aware"] = {
            "speedup": partial["mean_total_time_ms"] / full["mean_total_time_ms"] if full["mean_total_time_ms"] else None,
            "data_savings_pct": ((partial["total_data_transfer_mb"] - full["total_data_transfer_mb"]) / partial["total_data_transfer_mb"] * 100) if partial.get("total_data_transfer_mb") else None,
        }
    
    return report


def main():
    args = parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Djinn once for all remote baselines
    if not (args.skip_semantic_blind and args.skip_partially_aware and args.skip_full_djinn):
        print("Initializing Djinn runtime...")
        ensure_initialized_before_async(args.djinn_server)
        print("✓ Djinn runtime initialized\n")
    
    results: Dict[str, Path] = {}
    
    # Run native PyTorch baseline
    if not args.skip_native:
        try:
            results["native_pytorch"] = run_native_baseline(args)
        except Exception as e:
            print(f"ERROR: Failed to run native baseline: {e}")
            if not (args.skip_semantic_blind and args.skip_partially_aware and args.skip_full_djinn):
                print("Continuing with remote baselines...")
            else:
                sys.exit(1)
    
    # Run semantic-blind baseline
    if not args.skip_semantic_blind:
        try:
            results["semantic_blind"] = run_remote_baseline(args, "semantic_blind", semantic_aware=False, use_session=False)
        except Exception as e:
            print(f"ERROR: Failed to run semantic-blind baseline: {e}")
    
    # Run partially-aware baseline
    if not args.skip_partially_aware:
        try:
            results["partially_aware"] = run_remote_baseline(args, "partially_aware", semantic_aware=True, use_session=False)
        except Exception as e:
            print(f"ERROR: Failed to run partially-aware baseline: {e}")
    
    # Run full Djinn baseline
    if not args.skip_full_djinn:
        try:
            results["full_djinn"] = run_remote_baseline(args, "full_djinn", semantic_aware=True, use_session=True)
        except Exception as e:
            print(f"ERROR: Failed to run full Djinn baseline: {e}")
    
    # Generate comparison report
    if len(results) >= 2:
        print("\n" + "="*80)
        print("Generating comparison report...")
        print("="*80)
        
        report = generate_comparison_report(results, args)
        
        timestamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        report_file = args.output_dir / f"comparison_report_{args.tag}_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Comparison report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        for baseline_name, baseline_data in report["baselines"].items():
            print(f"\n{baseline_name}:")
            print(f"  Mean turn latency: {baseline_data['mean_turn_latency_ms']:.2f} ms")
            print(f"  Mean total time: {baseline_data['mean_total_time_ms']:.2f} ms")
            if baseline_data.get("total_data_transfer_mb"):
                print(f"  Data transfer: {baseline_data['total_data_transfer_mb']:.2f} MB")
        
        if "comparisons" in report:
            print("\nComparisons:")
            for comp_name, comp_data in report["comparisons"].items():
                print(f"\n{comp_name}:")
                for key, value in comp_data.items():
                    if value is not None:
                        if "speedup" in key:
                            print(f"  {key}: {value:.2f}×")
                        elif "pct" in key or "overhead" in key:
                            print(f"  {key}: {value:.2f}%")
                        else:
                            print(f"  {key}: {value:.2f}")
    else:
        print("\nWARNING: Need at least 2 baselines to generate comparison report")
    
    print("\n✓ All baselines completed")


if __name__ == "__main__":
    main()

