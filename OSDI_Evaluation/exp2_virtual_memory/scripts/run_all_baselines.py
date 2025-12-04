#!/usr/bin/env python3
"""
Orchestration Script: Run All Baselines for Experiment 2

Executes HuggingFace Accelerate, DeepSpeed, and Djinn ring buffer experiments
in sequence, then generates comparison reports with bandwidth and TTFT metrics.

Usage:
    python run_all_baselines.py \
        --model meta-llama/Llama-2-70b-hf \
        --runs 5 \
        --ttft-enabled \
        --output-dir results/experiment2
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], name: str, timeout: int = 3600) -> bool:
    """Run a command and return success status."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {name}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, timeout=timeout, check=True)
        logger.info(f"✅ {name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {name} failed with exit code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {name} timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"❌ {name} error: {e}")
        return False


def load_results(json_file: Path) -> Optional[Dict[str, Any]]:
    """Load results from JSON file."""
    try:
        with open(json_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {json_file}: {e}")
        return None


def generate_comparison_report(
    results_dir: Path,
    model_id: str,
) -> Dict[str, Any]:
    """Generate comparison report from all baseline results."""
    
    baselines = {
        "hf_accelerate": results_dir / "hf_accelerate.json",
        "deepspeed": results_dir / "deepspeed.json",
        "djinn_ring_buffer": results_dir / "djinn_ring_buffer.json",
    }
    
    comparison = {}
    
    for name, json_file in baselines.items():
        data = load_results(json_file)
        if data:
            summary = data.get("summary", {})
            comparison[name] = {
                "bandwidth_gbps": summary.get("avg_bandwidth_gbps", 0.0),
                "bandwidth_std": summary.get("stdev_bandwidth_gbps", 0.0),
                "latency_ms": summary.get("avg_latency_ms", 0.0),
                "ttft_ms": summary.get("avg_ttft_ms"),
                "success": True,
            }
        else:
            comparison[name] = {"success": False}
    
    return comparison


def print_comparison_table(comparison: Dict[str, Any]):
    """Print comparison table."""
    
    logger.info("\n" + "=" * 90)
    logger.info("EXPERIMENT 2 BASELINE COMPARISON")
    logger.info("=" * 90)
    logger.info(f"\n{'Baseline':<20} {'Bandwidth (GB/s)':<20} {'Latency (ms)':<20} {'TTFT (ms)':<20}")
    logger.info("-" * 90)
    
    for baseline, metrics in comparison.items():
        if metrics.get("success", False):
            bw = metrics.get("bandwidth_gbps", 0.0)
            bw_std = metrics.get("bandwidth_std", 0.0)
            latency = metrics.get("latency_ms", 0.0)
            ttft = metrics.get("ttft_ms")
            
            bw_str = f"{bw:.1f} ± {bw_std:.1f}"
            ttft_str = f"{ttft:.1f}" if ttft else "N/A"
            
            logger.info(f"{baseline:<20} {bw_str:<20} {latency:<20.1f} {ttft_str:<20}")
        else:
            logger.info(f"{baseline:<20} {'FAILED':<20} {'':<20} {'':<20}")
    
    logger.info("=" * 90)
    
    # Print evaluation against targets
    logger.info("\nEVALUATION AGAINST TARGETS:")
    logger.info("-" * 90)
    
    target_bandwidth = 20.0
    target_ttft = 7000.0  # 7 seconds
    
    for baseline, metrics in comparison.items():
        if metrics.get("success", False):
            bw = metrics.get("bandwidth_gbps", 0.0)
            ttft = metrics.get("ttft_ms")
            
            bw_status = "✅" if bw >= target_bandwidth else "❌"
            ttft_status = "✅" if (ttft and ttft <= target_ttft) else "❌" if ttft else "N/A"
            
            logger.info(f"{baseline:<20} BW: {bw_status} {bw:.1f}GB/s (target {target_bandwidth}GB/s), "
                       f"TTFT: {ttft_status} {ttft_str if ttft else 'N/A'} (target {target_ttft}ms)")


def main():
    parser = argparse.ArgumentParser(description="Run all baselines for Experiment 2")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-hf",
                       help="Model ID")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of runs per baseline")
    parser.add_argument("--output-dir", type=str, default="results/experiment2",
                       help="Output directory for results")
    parser.add_argument("--skip-hf", action="store_true",
                       help="Skip HuggingFace Accelerate baseline")
    parser.add_argument("--skip-deepspeed", action="store_true",
                       help="Skip DeepSpeed baseline")
    parser.add_argument("--skip-djinn", action="store_true",
                       help="Skip Djinn ring buffer baseline")
    parser.add_argument("--ttft-enabled", action="store_true",
                       help="Enable TTFT measurement using generate()")
    parser.add_argument("--generation-length", type=int, default=50,
                       help="Tokens to generate for TTFT measurement")
    parser.add_argument("--timeout", type=int, default=7200,
                       help="Timeout per baseline in seconds (default: 7200 = 2 hours)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 90)
    logger.info("EXPERIMENT 2: COMPREHENSIVE BASELINE COMPARISON")
    logger.info("=" * 90)
    logger.info("")
    logger.info(f"Model: {args.model}")
    logger.info(f"Runs per baseline: {args.runs}")
    logger.info(f"TTFT measurement: {'enabled' if args.ttft_enabled else 'disabled'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    results = {}
    
    # Run HuggingFace Accelerate
    if not args.skip_hf:
        logger.info("[1/3] Running HuggingFace Accelerate baseline...")
        cmd = [
            "python3", str(script_dir / "baseline_hf_accelerate.py"),
            "--model", args.model,
            "--runs", str(args.runs),
            "--output", str(output_dir / "hf_accelerate.json"),
        ]
        if args.ttft_enabled:
            cmd.append("--ttft-enabled")
            cmd.extend(["--generation-length", str(args.generation_length)])
        
        success = run_command(cmd, "HuggingFace Accelerate Baseline", timeout=args.timeout)
        results["hf_accelerate"] = success
    
    # Run DeepSpeed
    if not args.skip_deepspeed:
        logger.info("[2/3] Running DeepSpeed baseline...")
        cmd = [
            "python3", str(script_dir / "baseline_deepspeed.py"),
            "--model", args.model,
            "--runs", str(args.runs),
            "--output", str(output_dir / "deepspeed.json"),
            "--skip-if-unavailable",  # Don't fail if DeepSpeed not installed
        ]
        if args.ttft_enabled:
            cmd.append("--ttft-enabled")
            cmd.extend(["--generation-length", str(args.generation_length)])
        
        success = run_command(cmd, "DeepSpeed Baseline", timeout=args.timeout)
        results["deepspeed"] = success
    
    # Run Djinn Ring Buffer
    if not args.skip_djinn:
        logger.info("[3/3] Running Djinn ring buffer baseline...")
        
        # Load the config for L4
        config_file = script_dir.parent / "configs" / "virt_mem_l4.yaml"
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            results["djinn_ring_buffer"] = False
        else:
            cmd = [
                "python3", str(script_dir / "run_virtual_memory_experiment.py"),
                "--config", str(config_file),
                "--output", str(output_dir / "djinn_ring_buffer.json"),
                "--runs", str(args.runs),
                "--model", args.model,
            ]
            if args.ttft_enabled:
                cmd.append("--ttft-enabled")
            
            success = run_command(cmd, "Djinn Ring Buffer Baseline", timeout=args.timeout)
            results["djinn_ring_buffer"] = success
    
    # Generate comparison report
    logger.info("\n" + "=" * 90)
    logger.info("GENERATING COMPARISON REPORT...")
    logger.info("=" * 90)
    
    comparison = generate_comparison_report(output_dir, args.model)
    print_comparison_table(comparison)
    
    # Save comparison report
    report_file = output_dir / "comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "model": args.model,
            "comparison": comparison,
            "timestamp": time.time(),
        }, f, indent=2)
    
    logger.info(f"\n✅ Comparison report saved to: {report_file}")
    
    # Print summary
    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY")
    logger.info("=" * 90)
    
    total_baselines = len([r for r in results.values() if r])
    logger.info(f"Completed baselines: {total_baselines}/{len(results)}")
    
    for baseline, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"{status} {baseline}")
    
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review comparison_report.json for detailed metrics")
    logger.info("2. Check if Djinn reaches target: >20GB/s bandwidth, <7s TTFT")
    logger.info("3. Analyze PCIe traces if available (nvidia-smi dmon output)")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
