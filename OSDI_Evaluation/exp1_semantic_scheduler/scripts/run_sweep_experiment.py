#!/usr/bin/env python3
"""
Djinn Sweep Experiment for OSDI Evaluation.

Runs Djinn at multiple N values to find the "sweet spot" where
Djinn maintains interactive latency (<2s P99) while baselines fail.

Key metric: What is the maximum N where Djinn maintains P99 < 2000ms?

This script wraps run_poisson_experiment.py with sweep logic.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import asyncio

from trace_generator import load_trace

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_djinn_at_n(
    n_agents: int,
    trace_dir: Path,
    results_dir: Path,
    model_id: str = "meta-llama/Llama-2-13b-hf",
) -> Dict[str, Any]:
    """
    Run Djinn experiment at specific N using run_poisson_experiment.py
    
    Args:
        n_agents: Number of agents
        trace_dir: Directory containing trace files
        model_id: Model ID
    
    Returns:
        Result dictionary with key metrics
    """
    import subprocess
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DJINN SWEEP: N={n_agents}")
    logger.info(f"{'='*80}\n")
    
    # Load trace to create config
    trace_file = trace_dir / f"trace_{n_agents}.json"
    if not trace_file.exists():
        logger.error(f"Trace not found: {trace_file}")
        return {"n_agents": n_agents, "status": "trace_missing"}
    
    trace = load_trace(trace_file)
    
    # Create temporary config for this N
    config = {
        "experiment": {
            "name": f"semantic_scheduler_sweep_n{n_agents}",
            "description": f"SWEEP: N={n_agents} with controlled load",
            "version": "1.0"
        },
        "workload": {
            "model_id": model_id,
            "dtype": "float16",
            "total_agents": n_agents,
            "arrival_rate": 0.2,
            "think_time_min": 10.0,
            "think_time_max": 20.0,
            "new_tokens": 50,
            "iterations": 1,
            "context_length": 2048
        },
        "semantic_scheduler": {
            "enabled": True,
            "idle_threshold_seconds": 1.0,
            "host_swap_pool_gb": 32.0,
            "lifo_on_overload": True
        },
        "server_config": {
            "enable_semantic_scheduler": True,
            "idle_threshold_seconds": 1.0,
            "host_swap_pool_gb": 32.0,
            "max_concurrent": 256
        }
    }
    
    config_file = Path(f"/tmp/djinn_sweep_n{n_agents}.yaml")
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run experiment
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_poisson_experiment.py"),
        "--config", str(config_file),
        "--model-id", model_id,
    ]
    
    result_file = Path(f"/tmp/djinn_result_n{n_agents}.json")
    
    try:
        # Run in subprocess to isolate
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent.parent  # /home/ubuntu/Djinn
        )
        
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)  # 15min timeout
        
        if proc.returncode != 0:
            logger.error(f"Experiment failed: {stderr.decode()}")
            return {"n_agents": n_agents, "status": "error", "error": stderr.decode()[:500]}
        
        # The experiment should have written results to results/poisson_*
        # Find the latest one in the provided results_dir
        result_files = sorted(results_dir.glob("poisson_semantic_scheduler_*.json"))
        if not result_files:
            logger.error("No result file found")
            return {"n_agents": n_agents, "status": "no_result"}
        
        latest_result = result_files[-1]
        with open(latest_result) as f:
            full_result = json.load(f)
        
        # Extract key metrics
        agg = full_result.get("aggregates", {})
        return {
            "n_agents": n_agents,
            "status": "success",
            "p99_latency_ms": agg.get("latency_stats", {}).get("p99_ms", 0),
            "p50_latency_ms": agg.get("latency_stats", {}).get("p50_ms", 0),
            "success_count": agg.get("success_count", 0),
            "swaps": agg.get("kv_metrics", {}).get("swaps", 0),
            "duration_s": agg.get("total_duration_s", 0),
            "result_file": str(latest_result),
        }
    
    except asyncio.TimeoutError:
        logger.error(f"Experiment timeout at N={n_agents}")
        return {"n_agents": n_agents, "status": "timeout"}
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return {"n_agents": n_agents, "status": "error", "error": str(e)}
    finally:
        config_file.unlink(missing_ok=True)


async def run_djinn_sweep(
    trace_dir: Path,
    results_dir: Path,
    agent_counts: List[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run Djinn sweep across multiple N values.
    
    Args:
        trace_dir: Directory with traces
        results_dir: Output directory
        agent_counts: List of N values to test
    
    Returns:
        Dictionary mapping N -> results
    """
    if agent_counts is None:
        agent_counts = [10, 20, 30, 40, 50, 60, 70, 80]
    
    results = {}
    
    for n in agent_counts:
        result = await run_djinn_at_n(n, trace_dir, results_dir)
        results[n] = result
        
        # Log key finding
        if result["status"] == "success":
            p99 = result["p99_latency_ms"]
            is_interactive = p99 < 2000
            logger.info(f"N={n}: P99={p99:.0f}ms {'✅ INTERACTIVE' if is_interactive else '⚠️  QUEUING'}")
        
        # Cool down between experiments
        await asyncio.sleep(2)
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = results_dir / f"djinn_sweep_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Saved sweep results to {output_file}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=Path("OSDI_Evaluation/exp1_semantic_scheduler/traces"))
    parser.add_argument("--results-dir", type=Path, default=Path("OSDI_Evaluation/exp1_semantic_scheduler/results"))
    parser.add_argument("--n-agents", type=int, default=None, help="Optional: run only a single N value")
    
    args = parser.parse_args()
    
    # Run sweep
    agent_counts = [args.n_agents] if args.n_agents is not None else None
    asyncio.run(run_djinn_sweep(args.trace_dir, args.results_dir, agent_counts=agent_counts))
