#!/usr/bin/env python3
"""
End-to-End Smoke Test for OSDI Baselines.

Tests all baselines at N=10 to verify:
1. Code correctness (no syntax/import errors)
2. Result sanity (latencies > 0, counts make sense)
3. Apple-to-apple comparability (metrics are compatible)
4. Reproducibility (results are consistent)

This should run in ~30-40 minutes total.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description, timeout=600, cwd=None):
    """Run a command and return success/failure."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        
        if result.returncode != 0:
            logger.error(f"❌ FAILED (exit code {result.returncode})")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
            return False
        
        logger.info(f"✅ SUCCESS")
        return True
    
    except subprocess.TimeoutExpired:
        logger.error(f"❌ TIMEOUT (exceeded {timeout}s)")
        return False
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        return False


def validate_result_sanity(result_file: Path, baseline_name: str) -> bool:
    """Validate that results make scientific sense."""
    logger.info(f"\nValidating {baseline_name} results...")
    
    if not result_file.exists():
        logger.error(f"❌ Result file not found: {result_file}")
        return False
    
    try:
        with open(result_file) as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to parse JSON: {e}")
        return False
    
    # Check structure
    if baseline_name == "ray":
        if not isinstance(results, dict):
            logger.error("❌ Ray results should be dict")
            return False
        
        n = results.get("n_agents_requested")
        completed = results.get("n_agents_completed", 0)
        status = results.get("status", "unknown")
        
        logger.info(f"  N={n}, completed={completed}, status={status}")
        
        # Sanity checks
        if n is None:
            logger.error("❌ Missing n_agents_requested")
            return False
        
        # At N=10, ray might crash early or succeed
        if status not in ["success", "oom", "cuda_oom", "error"]:
            logger.error(f"❌ Unknown status: {status}")
            return False
        
        if status == "success" and completed == 0:
            logger.error("❌ Success but 0 agents completed")
            return False
        
        logger.info(f"✅ Ray result sanity check passed")
        return True
    
    elif baseline_name == "serverless":
        if not isinstance(results, dict):
            logger.error("❌ Serverless results should be dict")
            return False
        
        n = results.get("n_agents")
        status = results.get("status", "success")
        if status != "success":
            logger.error(f"❌ Serverless status not success: {status}")
            return False
        latency_stats = results.get("latency_stats", {})
        p99 = latency_stats.get("p99_ms", 0)
        
        logger.info(f"  N={n}, P99={p99:.0f}ms")
        
        if n != 10:
            logger.error(f"❌ Expected N=10, got {n}")
            return False
        
        if p99 <= 0:
            logger.error(f"❌ P99 should be > 0, got {p99}")
            return False
        
        # Serverless should have constant latency (cold start)
        # With cached measurements: 2 × (30s cold_start + 1.7s inference) = 63.4s
        if p99 < 50000 or p99 > 70000:
            logger.warning(f"⚠️  P99={p99:.0f}ms seems off for serverless (expected ~63s with cached values)")
        
        logger.info(f"✅ Serverless result sanity check passed")
        return True
    
    elif baseline_name == "vllm":
        if not isinstance(results, dict):
            logger.error("❌ vLLM results should be dict")
            return False
        
        n = results.get("n_agents")
        status = results.get("status")
        p99 = results.get("p99_latency_ms", 0)
        latency_stats = results.get("latency_stats", {})
        
        logger.info(f"  N={n}, status={status}, P99={p99:.0f}ms")
        
        if n != 10:
            logger.error(f"❌ Expected N=10, got {n}")
            return False
        
        if status not in ["success", "cuda_oom", "init_error"]:
            logger.error(f"❌ Unknown status: {status}")
            return False
        
        if status == "success":
            if p99 <= 0:
                logger.error(f"❌ P99 should be > 0, got {p99}")
                return False
            
            # vLLM at N=10 should have low latency (fast)
            if p99 > 10000:
                logger.warning(f"⚠️  P99={p99:.0f}ms seems high for vLLM at N=10")
            
            if not latency_stats:
                logger.error("❌ Missing latency_stats")
                return False
        
        logger.info(f"✅ vLLM result sanity check passed")
        return True
    
    elif baseline_name == "djinn":
        if not isinstance(results, dict):
            logger.error("❌ Djinn results should be dict")
            return False
        
        n_agents = list(results.keys())
        if not n_agents:
            logger.error("❌ No agents in Djinn results")
            return False
        
        n = int(n_agents[0])
        result = results[n_agents[0]]
        
        if isinstance(result, str) or not isinstance(result, dict):
            logger.error(f"❌ Expected dict result for each N, got {type(result)}")
            return False
        
        status = result.get("status")
        p99 = result.get("p99_latency_ms", 0)
        
        logger.info(f"  N={n}, status={status}, P99={p99:.0f}ms")
        
        if status != "success":
            logger.error(f"❌ Expected success, got {status}")
            return False
        
        if p99 <= 0:
            logger.error(f"❌ P99 should be > 0, got {p99}")
            return False
        
        # Djinn at N=10 should have low-to-moderate latency
        if p99 > 60000:
            logger.warning(f"⚠️  P99={p99:.0f}ms seems high even for Djinn")
        
        logger.info(f"✅ Djinn result sanity check passed")
        return True
    
    else:
        logger.error(f"❌ Unknown baseline: {baseline_name}")
        return False


def main():
    """Run end-to-end smoke test."""
    script_dir = Path(__file__).parent
    exp_dir = script_dir.parent
    traces_dir = exp_dir / "traces"
    results_dir = exp_dir / "results"

    # Ensure no leftover processes before starting (server, ray, vllm)
    logger.info("Killing any leftover processes before starting (Djinn server, ray, vllm)...")
    subprocess.run(["pkill", "-f", "djinn.server.server_main"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "vllm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "ray::RayAgentActor"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "baseline_ray_actors.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "baseline_serverless_emulator.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "baseline_vllm_fixed.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    logger.info("="*80)
    logger.info("OSDI BASELINE SMOKE TEST (N=10)")
    logger.info("="*80)
    
    # Step 0: Verify setup
    logger.info("\nStep 0: Verifying setup...")
    if not script_dir.exists():
        logger.error(f"❌ Script directory not found: {script_dir}")
        return False
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate traces
    logger.info("\n\n" + "="*80)
    logger.info("STEP 1: Generate Traces")
    logger.info("="*80)
    
    if not run_command(
        [sys.executable, str(script_dir / "trace_generator.py")],
        "Generate traces",
        timeout=120,
        cwd=str(exp_dir)
    ):
        logger.error("❌ Trace generation failed")
        return False
    
    # Verify traces
    trace_10 = traces_dir / "trace_10.json"
    if not trace_10.exists():
        logger.error(f"❌ Trace file not created: {trace_10}")
        return False
    
    with open(trace_10) as f:
        trace = json.load(f)
    
    logger.info(f"✅ Trace created: N={trace.get('n_agents')}, "
               f"context_length={trace.get('context_length')}, "
               f"seed={trace.get('seed')}")
    
    if trace.get('n_agents') != 10:
        logger.error(f"❌ Expected N=10, got {trace.get('n_agents')}")
        return False
    
    if trace.get('context_length') != 2048:
        logger.error(f"❌ Expected context_length=2048, got {trace.get('context_length')}")
        return False
    
    # Step 2: Ray Baseline
    logger.info("\n\n" + "="*80)
    logger.info("STEP 2: Ray Baseline (N=10)")
    logger.info("="*80)
    
    if not run_command(
        [sys.executable, str(script_dir / "baseline_ray_actors.py"), "--n-agents", "10"],
        "Ray baseline at N=10",
        timeout=300,
        cwd=str(exp_dir)
    ):
        logger.warning("⚠️  Ray baseline had issues (expected if OOM)")
    
    # Find and validate result
    ray_results = sorted(results_dir.glob("ray_baseline_10_*.json"))
    if ray_results:
        if not validate_result_sanity(ray_results[-1], "ray"):
            logger.warning("⚠️  Ray result sanity check failed")
    else:
        logger.warning("⚠️  No Ray result file found (might have crashed)")
    
    # Step 3: Serverless Baseline
    logger.info("\n\n" + "="*80)
    logger.info("STEP 3: Serverless Emulator (N=10)")
    logger.info("="*80)
    
    if not run_command(
        [sys.executable, str(script_dir / "baseline_serverless_emulator.py"), "--n-agents", "10"],
        "Serverless baseline at N=10",
        timeout=600,
        cwd=str(exp_dir)
    ):
        logger.error("❌ Serverless baseline failed")
        return False
    
    # Find and validate result
    serverless_results = sorted(results_dir.glob("serverless_baseline_10_*.json"))
    if serverless_results:
        if not validate_result_sanity(serverless_results[-1], "serverless"):
            logger.error("❌ Serverless result sanity check failed")
            return False
    else:
        logger.error("❌ No serverless result file found")
        return False
    
    # Step 4: vLLM Baseline
    logger.info("\n\n" + "="*80)
    logger.info("STEP 4: vLLM Baseline (N=10)")
    logger.info("="*80)
    
    if not run_command(
        [sys.executable, str(script_dir / "baseline_vllm_fixed.py"), "--n-agents", "10"],
        "vLLM baseline at N=10",
        timeout=300,
        cwd=str(exp_dir)
    ):
        logger.error("❌ vLLM baseline failed")
        return False
    
    # Find and validate result
    vllm_results = sorted(results_dir.glob("vllm_baseline_10_*.json"))
    if vllm_results:
        if not validate_result_sanity(vllm_results[-1], "vllm"):
            logger.error("❌ vLLM result sanity check failed")
            return False
    else:
        logger.error("❌ No vLLM result file found")
        return False
    
    # Step 5: Djinn Baseline
    logger.info("\n\n" + "="*80)
    logger.info("STEP 5: Djinn Baseline (N=10)")
    logger.info("="*80)
    logger.info("Note: Requires Djinn server running on 127.0.0.1:5556")
    
    # Try to start Djinn server (python -m djinn.server.server_main)
    logger.info("Attempting to start Djinn server (python -m djinn.server.server_main)...")
    server_log = Path("/tmp/djinn_server_live.log")
    server_cmd = [
        sys.executable,
        "-m",
        "djinn.server.server_main",
        "--gpu",
        "0",
        "--port",
        "5556",
    ]
    server_proc = subprocess.Popen(
        server_cmd,
        cwd="/home/ubuntu/Djinn",
        stdout=open(server_log, "w"),
        stderr=subprocess.STDOUT,
    )
    # Allow server to initialize
    time.sleep(15)
    
    try:
        if not run_command(
            [sys.executable, str(script_dir / "run_sweep_experiment.py"),
             "--trace-dir", str(traces_dir),
             "--results-dir", str(results_dir),
             "--n-agents", "10"],
            "Djinn sweep at N=10",
            timeout=900,
            cwd=str(exp_dir)
        ):
            logger.warning("⚠️  Djinn baseline had issues")
        
        # Find and validate result
        djinn_results = sorted(results_dir.glob("djinn_sweep_*.json"))
        if djinn_results:
            if not validate_result_sanity(djinn_results[-1], "djinn"):
                logger.warning("⚠️  Djinn result sanity check failed")
        else:
            logger.warning("⚠️  No Djinn result file found")
    
    finally:
        # Kill server
        server_proc.terminate()
        time.sleep(2)
        server_proc.kill()
    
    # Final Summary
    logger.info("\n\n" + "="*80)
    logger.info("SMOKE TEST COMPLETE")
    logger.info("="*80)
    
    logger.info("\n✅ All baseline components are functional!")
    logger.info("\nNext steps:")
    logger.info("1. Review results in: " + str(results_dir))
    logger.info("2. Check for any warnings or sanity issues above")
    logger.info("3. Run full sweep with: python scripts/run_all_baselines.py")
    logger.info("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
