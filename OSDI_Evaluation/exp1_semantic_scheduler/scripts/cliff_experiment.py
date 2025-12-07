#!/usr/bin/env python3
"""
Cliff Experiment: Find where vLLM crashes and show Djinn continues.

This script runs the same workload at increasing N values (10, 20, ..., 80) on both:
1. vLLM (batched concurrent) - expects crash at N~45-50
2. Djinn (semantic scheduler) - should continue to N=80+

Goal: Generate the "money shot" graph showing vLLM's OOM cliff vs Djinn's graceful scaling.
"""

import argparse
import json
import logging
import subprocess
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_vllm_baseline_at_n(n: int, vllm_script: Path) -> Dict[str, Any]:
    """Run vLLM baseline at specific N using the fixed baseline script."""
    logger.info(f"\n{'='*80}")
    logger.info(f"vLLM BASELINE: N={n}")
    logger.info(f"{'='*80}")
    
    try:
        # Run the vLLM baseline script which tests a single N value
        cmd = [
            sys.executable,
            str(vllm_script),
            "--n-agents", str(n),
        ]
        
        # Capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode == 0:
            logger.info(f"✅ vLLM N={n} completed")
            # Parse JSON output if available
            try:
                lines = result.stdout.split('\n')
                for line in reversed(lines):
                    if line.startswith('{'):
                        return json.loads(line)
            except:
                pass
            return {"status": "success", "n_agents": n}
        else:
            if "OutOfMemoryError" in result.stderr or "OOM" in result.stderr:
                logger.error(f"❌ vLLM N={n} OOM")
                return {"status": "oom", "n_agents": n}
            else:
                logger.error(f"❌ vLLM N={n} error: {result.stderr[:200]}")
                return {"status": "error", "n_agents": n, "error": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        logger.error(f"❌ vLLM N={n} timeout")
        return {"status": "timeout", "n_agents": n}
    except Exception as e:
        logger.error(f"❌ vLLM N={n} exception: {e}")
        return {"status": "error", "n_agents": n, "error": str(e)}


def run_djinn_experiment_at_n(n: int, djinn_script: Path, djinn_config: Path) -> Dict[str, Any]:
    """Run Djinn experiment at specific N."""
    logger.info(f"\n{'='*80}")
    logger.info(f"DJINN SEMANTIC SCHEDULER: N={n}")
    logger.info(f"{'='*80}")
    
    try:
        # Create a config for this specific N
        with open(djinn_config) as f:
            base_config = yaml.safe_load(f)
        
        base_config["workload"]["total_agents"] = n
        
        # Write to temp config
        temp_config = Path(f"/tmp/cliff_djinn_n{n}.yaml")
        with open(temp_config, 'w') as f:
            yaml.dump(base_config, f)
        
        # Run Djinn
        cmd = [
            sys.executable,
            str(djinn_script),
            "--config", str(temp_config),
            "--model-id", "meta-llama/Llama-2-7b-hf",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minutes max
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Djinn N={n} completed")
            try:
                lines = result.stdout.split('\n')
                for line in reversed(lines):
                    if line.startswith('{'):
                        return json.loads(line)
            except:
                pass
            return {"status": "success", "n_agents": n}
        else:
            logger.error(f"❌ Djinn N={n} error: {result.stderr[:200]}")
            return {"status": "error", "n_agents": n, "error": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Djinn N={n} timeout")
        return {"status": "timeout", "n_agents": n}
    except Exception as e:
        logger.error(f"❌ Djinn N={n} exception: {e}")
        return {"status": "error", "n_agents": n, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Cliff Experiment: Find vLLM OOM point and compare with Djinn"
    )
    parser.add_argument(
        "--vllm-script",
        type=Path,
        default=Path("OSDI_Evaluation/exp1_semantic_scheduler/scripts/baseline_vllm_actual.py"),
        help="Path to vLLM baseline script"
    )
    parser.add_argument(
        "--djinn-script",
        type=Path,
        default=Path("OSDI_Evaluation/exp1_semantic_scheduler/scripts/run_poisson_experiment.py"),
        help="Path to Djinn experiment script"
    )
    parser.add_argument(
        "--djinn-config",
        type=Path,
        default=Path("OSDI_Evaluation/exp1_semantic_scheduler/configs/agent_scaling_hero.yaml"),
        help="Path to Djinn config file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("OSDI_Evaluation/exp1_semantic_scheduler/results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--vllm-only",
        action="store_true",
        help="Only run vLLM (skip Djinn)"
    )
    parser.add_argument(
        "--djinn-only",
        action="store_true",
        help="Only run Djinn (skip vLLM)"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("CLIFF EXPERIMENT: vLLM vs Djinn Scaling")
    logger.info("=" * 80)
    
    # Agent counts to test
    # Start at 10, increase to find the cliff
    agent_counts = [10, 20, 30, 40, 45, 48, 50, 55, 60, 70, 80]
    
    results_by_system = {
        "vllm": {},
        "djinn": {},
    }
    
    cliff_point_vllm = None
    max_successful_djinn = 0
    
    # === PHASE 1: Run vLLM to find cliff ===
    if not args.djinn_only:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: vLLM CLIFF DETECTION")
        logger.info("=" * 80)
        
        for n in agent_counts:
            result = run_vllm_baseline_at_n(n, args.vllm_script)
            results_by_system["vllm"][n] = result
            
            if result["status"] in ["oom", "oom_init"]:
                cliff_point_vllm = n
                logger.error(f"\n🔴 vLLM CLIFF FOUND at N={n}")
                # Skip remaining vLLM tests
                for remaining_n in agent_counts[agent_counts.index(n) + 1:]:
                    results_by_system["vllm"][remaining_n] = {
                        "status": "skipped",
                        "n_agents": remaining_n,
                        "reason": f"OOM at N={n}"
                    }
                break
            
            # Cool down
            time.sleep(2)
    
    # === PHASE 2: Run Djinn ===
    if not args.vllm_only:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: DJINN SCALING TEST")
        logger.info("=" * 80)
        
        for n in agent_counts:
            result = run_djinn_experiment_at_n(n, args.djinn_script, args.djinn_config)
            results_by_system["djinn"][n] = result
            
            if result["status"] == "success":
                max_successful_djinn = n
                
                # Extract latency stats if available
                if "aggregates" in result:
                    p99 = result["aggregates"].get("latency_stats", {}).get("p99_ms", 0)
                    logger.info(f"✅ Djinn N={n}: P99={p99:.0f}ms")
            
            # Cool down
            time.sleep(2)
    
    # === ANALYSIS ===
    logger.info("\n" + "=" * 80)
    logger.info("CLIFF ANALYSIS")
    logger.info("=" * 80)
    
    logger.info("\nvLLM Results:")
    for n in sorted(results_by_system["vllm"].keys()):
        result = results_by_system["vllm"][n]
        status = result["status"]
        if status == "success":
            logger.info(f"  ✅ N={n:2d}: Success")
        elif status == "oom":
            logger.info(f"  ❌ N={n:2d}: OOM CRASH (CLIFF POINT)")
        elif status == "skipped":
            logger.info(f"  ⊘  N={n:2d}: Skipped")
        else:
            logger.info(f"  ❌ N={n:2d}: {status.upper()}")
    
    logger.info("\nDjinn Results:")
    for n in sorted(results_by_system["djinn"].keys()):
        result = results_by_system["djinn"][n]
        status = result["status"]
        if status == "success":
            p99 = result.get("aggregates", {}).get("latency_stats", {}).get("p99_ms", "?")
            logger.info(f"  ✅ N={n:2d}: Success (P99={p99}ms)")
        elif status == "skipped":
            logger.info(f"  ⊘  N={n:2d}: Skipped")
        else:
            logger.info(f"  ❌ N={n:2d}: {status.upper()}")
    
    logger.info("\n" + "-" * 80)
    if cliff_point_vllm:
        logger.info(f"🔴 vLLM CRASH POINT: N={cliff_point_vllm}")
        logger.info(f"✅ Djinn SUCCESS UP TO: N={max_successful_djinn}")
        if max_successful_djinn > cliff_point_vllm:
            scaling_advantage = max_successful_djinn / cliff_point_vllm
            logger.info(f"📊 SCALING ADVANTAGE: {scaling_advantage:.2f}x")
    else:
        logger.info(f"✅ vLLM: No crash detected (tested up to N=80)")
        logger.info(f"✅ Djinn: Success up to N={max_successful_djinn}")
    logger.info("-" * 80)
    
    # === SAVE RESULTS ===
    payload = {
        "tag": "cliff_experiment",
        "generated_at": _utc_timestamp(),
        "cliff_analysis": {
            "vllm_crash_point": cliff_point_vllm,
            "djinn_max_successful": max_successful_djinn,
            "scaling_advantage": (max_successful_djinn / cliff_point_vllm) if cliff_point_vllm else None,
        },
        "results_by_system": results_by_system,
    }
    
    output_file = args.output_dir / f"cliff_experiment_{_utc_timestamp()}.json"
    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
