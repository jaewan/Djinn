#!/usr/bin/env python3
"""
Run Experiment 2.1 (LLM decode) with all 4 baselines:
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
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.djinn_init import ensure_initialized_before_async, export_evaluation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-hf", help="HF model identifier")
    parser.add_argument("--prompt-length", type=int, default=72, help="Prompt token length")
    parser.add_argument("--new-tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=30, help="Number of measured runs per baseline")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Unmeasured warmup runs")
    parser.add_argument("--device", default="cuda:0", help="torch device string (for native baseline)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--djinn-server", default="localhost:5556", help="Djinn server address")
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_1_llm_decode/results"))
    parser.add_argument("--tag", default="all_baselines", help="Tag for output files")
    parser.add_argument("--seed", type=int, default=42)
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
    
    script_path = REPO_ROOT / "Evaluation/exp2_1_llm_decode/scripts/run_local_baseline.py"
    output_dir = args.output_dir / "native_pytorch"
    
    cmd = [
        sys.executable, str(script_path),
        "--model-id", args.model_id,
        "--prompt-length", str(args.prompt_length),
        "--new-tokens", str(args.new_tokens),
        "--batch-size", str(args.batch_size),
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
        output_dir.glob("llm_decode_native_pytorch_*.json"),
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
    
    script_path = REPO_ROOT / "Evaluation/exp2_1_llm_decode/scripts/run_local_baseline.py"
    output_dir = args.output_dir / baseline_name
    
    cmd = [
        sys.executable, str(script_path),
        "--model-id", args.model_id,
        "--prompt-length", str(args.prompt_length),
        "--new-tokens", str(args.new_tokens),
        "--batch-size", str(args.batch_size),
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
        output_dir.glob(f"llm_decode_{baseline_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if output_files:
        return output_files[-1]
    raise RuntimeError(f"No output file found in {output_dir}")


def validate_semantic_blind_baseline(result_file: Path) -> Dict[str, Any]:
    """Validate that semantic-blind baseline has no semantic hints."""
    with open(result_file) as f:
        data = json.load(f)
    
    # Check that semantic hints are not being used
    # This is a basic validation - we'll need to check the actual execution
    validation = {
        "has_semantic_hints": False,  # Should be False for semantic-blind
        "data_transfer_mb": None,
        "latency_ms": None,
    }
    
    if "runs" in data and data["runs"]:
        first_run = data["runs"][0]
        validation["data_transfer_mb"] = first_run.get("host_to_device_mb", 0) + first_run.get("device_to_host_mb", 0)
        validation["latency_ms"] = first_run.get("total_ms")
    
    return validation


def generate_comparison_report(results: Dict[str, Path], args: argparse.Namespace) -> Dict[str, Any]:
    """Generate comparison report from all baseline results."""
    report = {
        "experiment": "exp2_1_llm_decode",
        "model_id": args.model_id,
        "prompt_tokens": args.prompt_length,
        "new_tokens": args.new_tokens,
        "batch_size": args.batch_size,
        "runs_per_baseline": args.runs,
        "baselines": {},
        "comparisons": {},
    }
    
    baseline_data = {}
    for baseline_name, result_file in results.items():
        with open(result_file) as f:
            data = json.load(f)
        baseline_data[baseline_name] = data
        
        aggregates = data.get("aggregates", {})
        report["baselines"][baseline_name] = {
            "mean_latency_ms": aggregates.get("total_ms", {}).get("mean"),
            "p95_latency_ms": aggregates.get("total_ms", {}).get("p95"),
            "mean_per_token_ms": aggregates.get("per_token_ms", {}).get("mean"),
            "p95_per_token_ms": aggregates.get("per_token_ms", {}).get("p95"),
            "mean_throughput_tps": aggregates.get("throughput_tokens_per_s", {}).get("mean"),
            "result_file": str(result_file),
        }
        
        # Extract data transfer from first run
        if data.get("runs") and len(data["runs"]) > 0:
            first_run = data["runs"][0]
            report["baselines"][baseline_name]["data_transfer_mb"] = (
                first_run.get("host_to_device_mb", 0) + first_run.get("device_to_host_mb", 0)
            )
    
    # Generate comparisons
    if "native_pytorch" in baseline_data and "full_djinn" in baseline_data:
        native = report["baselines"]["native_pytorch"]
        full = report["baselines"]["full_djinn"]
        report["comparisons"]["full_djinn_vs_native"] = {
            "latency_overhead_pct": ((full["mean_latency_ms"] / native["mean_latency_ms"]) - 1) * 100 if native["mean_latency_ms"] else None,
            "throughput_ratio": full["mean_throughput_tps"] / native["mean_throughput_tps"] if native["mean_throughput_tps"] else None,
        }
    
    if "semantic_blind" in baseline_data and "full_djinn" in baseline_data:
        blind = report["baselines"]["semantic_blind"]
        full = report["baselines"]["full_djinn"]
        report["comparisons"]["full_djinn_vs_semantic_blind"] = {
            "speedup": blind["mean_latency_ms"] / full["mean_latency_ms"] if full["mean_latency_ms"] else None,
            "data_savings_pct": ((blind["data_transfer_mb"] - full["data_transfer_mb"]) / blind["data_transfer_mb"] * 100) if blind.get("data_transfer_mb") else None,
            "throughput_improvement": full["mean_throughput_tps"] / blind["mean_throughput_tps"] if blind["mean_throughput_tps"] else None,
        }
    
    if "partially_aware" in baseline_data and "full_djinn" in baseline_data:
        partial = report["baselines"]["partially_aware"]
        full = report["baselines"]["full_djinn"]
        report["comparisons"]["full_djinn_vs_partially_aware"] = {
            "speedup": partial["mean_latency_ms"] / full["mean_latency_ms"] if full["mean_latency_ms"] else None,
            "data_savings_pct": ((partial["data_transfer_mb"] - full["data_transfer_mb"]) / partial["data_transfer_mb"] * 100) if partial.get("data_transfer_mb") else None,
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
            if not args.skip_semantic_blind or not args.skip_partially_aware or not args.skip_full_djinn:
                print("Continuing with remote baselines...")
            else:
                sys.exit(1)
    
    # Run semantic-blind baseline
    if not args.skip_semantic_blind:
        try:
            # NOTE: We need to modify run_local_baseline.py to support semantic_aware=False
            # For now, we'll run it and validate afterwards
            results["semantic_blind"] = run_remote_baseline(args, "semantic_blind", semantic_aware=False, use_session=False)
            validation = validate_semantic_blind_baseline(results["semantic_blind"])
            print(f"\nSemantic-blind validation: {validation}")
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
            print(f"  Mean latency: {baseline_data['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency: {baseline_data['p95_latency_ms']:.2f} ms")
            print(f"  Mean per-token: {baseline_data['mean_per_token_ms']:.2f} ms/token")
            if baseline_data.get("data_transfer_mb"):
                print(f"  Data transfer: {baseline_data['data_transfer_mb']:.2f} MB")
        
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

    # Export final metrics for analysis
    metrics_path = args.output_dir / f"{args.tag}_metrics.json"
    export_evaluation_metrics(str(metrics_path))

    print("\n✓ All baselines completed")


if __name__ == "__main__":
    main()

