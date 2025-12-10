#!/usr/bin/env python3
"""
Unified Baseline Runner for OSDI Evaluation.

Orchestrates ALL baselines (Ray, Serverless, vLLM, Djinn) with same traces.

Execution order:
1. Generate traces for all N values
2. Run Ray baseline (quick, crashes early)
3. Run Serverless emulator (measures cold start)
4. Kill Djinn server, run vLLM sweep (find crash point)
5. Kill vLLM, start Djinn server, run Djinn sweep
6. Generate comparison plot
7. Save unified results
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_trace_generation():
    """Generate traces for all N values."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: GENERATE TRACES")
    logger.info("="*80)
    
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "trace_generator.py")]
    result = subprocess.run(cmd, cwd=str(script_dir.parent), capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Trace generation failed: {result.stderr}")
        return False
    
    logger.info(result.stdout)
    return True


def run_ray_baseline():
    """Run Ray baseline sweep."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: RUN RAY BASELINE")
    logger.info("="*80)
    
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "baseline_ray_actors.py")]
    
    try:
        result = subprocess.run(cmd, cwd=str(script_dir.parent), capture_output=True, text=True, timeout=300)
        logger.info(result.stdout)
        if result.returncode != 0 and result.stderr:
            logger.warning(f"Ray baseline stderr: {result.stderr[:500]}")
        return True
    
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Ray baseline timed out after 300s - likely hanging during model load")
        logger.info("This is expected behavior: Ray cannot load 26GB model on shared GPU")
        # Kill lingering Ray processes
        subprocess.run(["pkill", "-9", "-f", "ray"], stderr=subprocess.DEVNULL)
        time.sleep(1)
        return True  # Continue to next baseline


def run_serverless_baseline():
    """Run serverless emulator baseline."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: RUN SERVERLESS EMULATOR")
    logger.info("="*80)
    
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "baseline_serverless_emulator.py"), "--use-cache"]
    result = subprocess.run(cmd, cwd=str(script_dir.parent), capture_output=True, text=True, timeout=120)
    
    logger.info(result.stdout)
    if result.returncode != 0 and result.stderr:
        logger.warning(f"Serverless baseline warnings: {result.stderr[:500]}")
    
    return True


def run_vllm_baseline():
    """Run vLLM baseline sweep."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: RUN vLLM BASELINE")
    logger.info("="*80)
    
    # Kill any existing Djinn server
    subprocess.run(["pkill", "-9", "-f", "run_server.py"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "baseline_vllm_fixed.py")]
    result = subprocess.run(cmd, cwd=str(script_dir.parent), capture_output=True, text=True, timeout=900)
    
    logger.info(result.stdout)
    if result.returncode != 0 and result.stderr:
        logger.warning(f"vLLM baseline warnings: {result.stderr[:500]}")
    
    return True


def run_djinn_baseline():
    """Run Djinn baseline sweep."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: RUN DJINN BASELINE")
    logger.info("="*80)
    
    # Start Djinn server (enable semantic scheduler + swap pool)
    logger.info("Starting Djinn server...")
    server_cmd = [
        sys.executable,
        "-m",
        "djinn.server.server_main",
        "--enable-semantic-scheduler",
        "--idle-threshold-seconds", "1.0",
        "--host-swap-pool-gb", "32",
    ]
    server_log = open("/tmp/djinn_server_live.log", "w")
    server_proc = subprocess.Popen(
        server_cmd,
        cwd="/home/ubuntu/Djinn",
        stdout=server_log,
        stderr=server_log
    )
    time.sleep(10)  # Wait for server to start
    
    try:
        script_dir = Path(__file__).parent
        cmd = [sys.executable, str(script_dir / "run_sweep_experiment.py")]
        result = subprocess.run(cmd, cwd=str(script_dir.parent), capture_output=True, text=True, timeout=1800)
        
        logger.info(result.stdout)
        if result.returncode != 0 and result.stderr:
            logger.warning(f"Djinn baseline warnings: {result.stderr[:500]}")
        
        return True
    
    finally:
        # Kill server
        server_proc.terminate()
        time.sleep(2)
        server_proc.kill()
        try:
            server_log.close()
        except Exception:
            pass


def generate_osdi_plot(results_dir: Path):
    """Generate OSDI comparison plot."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: GENERATE PLOT")
    logger.info("="*80)
    
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "generate_osdi_plot.py"), "--results-dir", str(results_dir)]
    result = subprocess.run(cmd, cwd=str(script_dir.parent), capture_output=True, text=True, timeout=60)
    
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(f"Plot generation failed: {result.stderr}")
        return False
    
    return True


def main():
    """Run all baselines in sequence."""
    results_dir = Path("OSDI_Evaluation/exp1_semantic_scheduler/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("OSDI BASELINE EVALUATION - UNIFIED RUNNER")
    logger.info("="*80)
    logger.info("Running all baselines with identical traces...")
    
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # Phase 1: Generate traces
    if not run_trace_generation():
        logger.error("Trace generation failed")
        return False
    
    # Phase 2: Ray baseline
    if not run_ray_baseline():
        logger.warning("Ray baseline had issues, continuing...")
    
    # Phase 3: Serverless emulator
    if not run_serverless_baseline():
        logger.warning("Serverless baseline had issues, continuing...")
    
    # Phase 4: vLLM baseline
    if not run_vllm_baseline():
        logger.warning("vLLM baseline had issues, continuing...")
    
    # Phase 5: Djinn baseline
    if not run_djinn_baseline():
        logger.warning("Djinn baseline had issues, continuing...")
    
    # Phase 6: Generate plot
    if not generate_osdi_plot(results_dir):
        logger.warning("Plot generation had issues")
    
    # Create summary
    logger.info("\n" + "="*80)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("\nNext steps:")
    logger.info("1. Review results/baseline_comparison_*.json")
    logger.info("2. Check osdi_latency_vs_load.pdf for comparison graph")
    logger.info("3. Find 'sweet spot' where Djinn maintains P99 < 2000ms")
    logger.info("4. Update evaluation report with findings")
    logger.info("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-traces", action="store_true", help="Skip trace generation")
    parser.add_argument("--skip-ray", action="store_true", help="Skip Ray baseline")
    parser.add_argument("--skip-serverless", action="store_true", help="Skip serverless baseline")
    parser.add_argument("--skip-vllm", action="store_true", help="Skip vLLM baseline")
    parser.add_argument("--skip-djinn", action="store_true", help="Skip Djinn baseline")
    
    args = parser.parse_args()
    
    if not args.skip_traces and not run_trace_generation():
        sys.exit(1)
    
    if not args.skip_ray and not run_ray_baseline():
        logger.warning("Ray baseline failed")
    
    if not args.skip_serverless and not run_serverless_baseline():
        logger.warning("Serverless baseline failed")
    
    if not args.skip_vllm and not run_vllm_baseline():
        logger.warning("vLLM baseline failed")
    
    if not args.skip_djinn and not run_djinn_baseline():
        logger.warning("Djinn baseline failed")
    
    if generate_osdi_plot(Path("OSDI_Evaluation/exp1_semantic_scheduler/results")):
        logger.info("✅ Complete!")
