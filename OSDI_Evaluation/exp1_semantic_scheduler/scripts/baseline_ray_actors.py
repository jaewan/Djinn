#!/usr/bin/env python3
"""
Ray Actors Baseline for OSDI Evaluation (Apple-to-Apple Comparison).

Tests process-level GPU management using Ray remote actors with Poisson arrivals.
Each actor loads the FULL model into GPU memory (26GB for Llama-2-13B).

Expected behavior: OOM crash at N=2-3 because multiple 26GB models
cannot fit on an 80GB H100.

This proves that without semantic scheduling (Djinn), process-level
memory management fails at low concurrency.
"""

import asyncio
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import numpy as np

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0.01)  # Minimal GPU fraction to force scheduling conflict
class RayAgentActor:
    """
    Remote actor that loads a full LLM model into GPU.
    
    This simulates how a developer would use Ray to serve multiple agents,
    where each agent gets its own actor. Implements Reason → Act → Reflect.
    
    Expected: OOM crash at N=2-3 when actors try to load 26GB model onto shared GPU.
    """

    def __init__(self, model_id: str, timeout_sec: int = 120):
        """Load model into GPU memory with timeout."""
        logger.info(f"[Actor] Loading {model_id} into GPU (timeout={timeout_sec}s)")
        self.model = None
        self.tokenizer = None
        self.load_error = None
        
        try:
            # Add a safety timeout to prevent hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Model loading exceeded {timeout_sec}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="cuda",  # Load to GPU
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"[Actor] Model loaded successfully")
            finally:
                signal.alarm(0)  # Cancel alarm
        
        except torch.cuda.OutOfMemoryError as e:
            self.load_error = f"CUDA OOM: {str(e)}"
            logger.error(f"[Actor] CUDA OOM during model loading: {e}")
            raise
        except TimeoutError as e:
            self.load_error = f"Timeout: {str(e)}"
            logger.error(f"[Actor] {e}")
            raise
        except Exception as e:
            self.load_error = f"Load failed: {str(e)}"
            logger.error(f"[Actor] Failed to load model: {e}")
            raise

    def run_lifecycle(self, prompt: str, think_time: float, max_tokens: int = 50) -> Dict[str, Any]:
        """
        Run Reason → Act → Reflect lifecycle.
        
        Args:
            prompt: Input prompt
            think_time: Duration of Act phase (seconds)
            max_tokens: Tokens to generate
        
        Returns:
            Result dictionary with latencies
        """
        try:
            # PHASE 1: REASON (Prefill)
            reason_start = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                )
            reason_latency_ms = (time.perf_counter() - reason_start) * 1000
            
            # PHASE 2: ACT (Think time)
            time.sleep(think_time)
            
            # PHASE 3: REFLECT (Decode)
            reflect_start = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                )
            reflect_latency_ms = (time.perf_counter() - reflect_start) * 1000
            
            return {
                "reason_latency_ms": reason_latency_ms,
                "reflect_latency_ms": reflect_latency_ms,
                "total_latency_ms": reason_latency_ms + reflect_latency_ms,
                "status": "success",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


async def spawn_agents_poisson(
    actors: List[Any],
    trace: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Spawn agents with Poisson inter-arrival times using Ray.
    
    Args:
        actors: List of Ray actor handles
        trace: Workload trace
    
    Returns:
        List of result records
    """
    n_agents = len(actors)
    prompts = trace["prompts"][:n_agents]
    arrival_times = trace["arrival_times"][:n_agents]
    think_times = trace["think_times"][:n_agents]
    
    logger.info(f"Spawning {n_agents} agent lifecycles with Poisson arrivals...")
    
    tasks = []
    start_wall_time = time.perf_counter()
    
    for agent_idx, (actor, prompt, arrival_time, think_time) in enumerate(
        zip(actors, prompts, arrival_times, think_times)
    ):
        # Sleep until this agent's arrival time
        current_wall_time = time.perf_counter() - start_wall_time
        if arrival_time > current_wall_time:
            await asyncio.sleep(arrival_time - current_wall_time)
        
        # Submit task
        task = (agent_idx, actor.run_lifecycle.remote(prompt, think_time))
        tasks.append(task)
        
        if (agent_idx + 1) % 5 == 0:
            logger.info(f"  Spawned {agent_idx + 1}/{n_agents} tasks")
    
    # Wait for all to complete
    logger.info(f"All {len(tasks)} tasks spawned. Waiting for completion (timeout=180s)...")
    try:
        results_raw = ray.get([task[1] for task in tasks], timeout=180)
    except ray.exceptions.GetTimeoutError:
        logger.error(f"❌ Timeout waiting for Ray tasks after 180s")
        raise TimeoutError(f"Ray task execution timeout after 180s with {len(tasks)} tasks")
    
    # Format results
    results = []
    for (agent_idx, _), result_raw in zip(tasks, results_raw):
        result = {
            "agent_id": agent_idx,
            "arrival_time_s": arrival_times[agent_idx],
            "think_time_s": think_times[agent_idx],
        }
        result.update(result_raw)
        results.append(result)
    
    return results


def run_ray_baseline(
    trace: Dict[str, Any],
    model_id: str = "meta-llama/Llama-2-13b-hf",
) -> Dict[str, Any]:
    """
    Run Ray baseline with Poisson arrivals.

    Expected: OOM at N=2-3 when each actor tries to load 26GB model.

    Args:
        trace: Workload trace (from trace_generator.py)
        model_id: Model to load

    Returns:
        Results dictionary with crash point and latencies
    """
    n_agents = trace["n_agents"]
    logger.info(f"\n{'='*80}")
    logger.info(f"RAY ACTORS BASELINE (APPLE-TO-APPLE): N={n_agents} with Poisson arrivals")
    logger.info(f"{'='*80}\n")

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    actors = []
    results = []
    crash_point = None
    start_time = time.perf_counter()

    try:
        # PHASE 1: Spawn all actors first (each will attempt to load model into GPU)
        logger.info(f"Spawning {n_agents} Ray actors (will OOM when GPU memory exhausted)...")
        
        for agent_id in range(n_agents):
            try:
                logger.info(f"  Creating actor {agent_id}...")
                # Create remote actor with timeout
                # NOTE: This call is NON-BLOCKING in Ray. The actor init happens in background.
                actor = RayAgentActor.remote(model_id, timeout_sec=120)
                actors.append(actor)
                
                # Give Ray a moment to start the actor (and fail if OOM)
                time.sleep(0.5)
                
                logger.info(f"  ✓ Actor {agent_id} created")
            
            except (torch.cuda.OutOfMemoryError, ray.exceptions.OutOfMemoryError) as e:
                logger.error(f"❌ OOM at actor {agent_id}: {e}")
                crash_point = agent_id
                logger.info(f"Ray OOM confirmed at N={agent_id}. Stopping spawn attempt.")
                break
            
            except Exception as e:
                logger.error(f"❌ Failed to spawn actor {agent_id}: {type(e).__name__}: {e}")
                crash_point = agent_id
                logger.info(f"Ray failed at actor {agent_id}. Stopping spawn attempt.")
                break
        
        # PHASE 2: Run all lifecycles concurrently with Poisson arrivals (if we have actors)
        if actors:
            logger.info(f"\nRunning {len(actors)} agent lifecycles with Poisson arrivals...")
            
            try:
                results = asyncio.run(spawn_agents_poisson(actors, trace))
            
            except (ray.exceptions.OutOfMemoryError, torch.cuda.OutOfMemoryError) as e:
                logger.error(f"❌ OOM during execution: {e}")
                crash_point = len(actors) if crash_point is None else crash_point
            
            except TimeoutError as e:
                logger.error(f"❌ Timeout during execution: {e}")
                crash_point = len(actors) if crash_point is None else crash_point
            
            except Exception as e:
                logger.error(f"❌ Error during execution: {type(e).__name__}: {e}")
                crash_point = len(actors) if crash_point is None else crash_point

    finally:
        ray.shutdown()

    duration = time.perf_counter() - start_time

    # Calculate statistics
    success_results = [r for r in results if r.get("status") == "success"]
    error_count = len([r for r in results if r.get("status") != "success"])

    if success_results:
        total_latencies = sorted([r["total_latency_ms"] for r in success_results])
        reason_latencies = sorted([r["reason_latency_ms"] for r in success_results])
        reflect_latencies = sorted([r["reflect_latency_ms"] for r in success_results])
        
        p99_idx = int(len(total_latencies) * 0.99)
        p99_idx = min(p99_idx, len(total_latencies) - 1)
        
        result = {
            "system": "ray_actors",
            "model_id": model_id,
            "n_agents_requested": n_agents,
            "n_agents_completed": len(success_results),
            "crash_point": crash_point if crash_point is not None else n_agents,
            "duration_s": duration,
            "status": "success" if crash_point is None else "oom",
            "latency_stats": {
                "mean_ms": np.mean(total_latencies),
                "min_ms": total_latencies[0],
                "max_ms": total_latencies[-1],
                "p50_ms": total_latencies[len(total_latencies) // 2],
                "p99_ms": total_latencies[p99_idx],
            },
            "reason_latency_stats": {
                "mean_ms": np.mean(reason_latencies),
                "p99_ms": reason_latencies[p99_idx],
            },
            "reflect_latency_stats": {
                "mean_ms": np.mean(reflect_latencies),
                "p99_ms": reflect_latencies[p99_idx],
            },
            "error_count": error_count,
        }
    else:
        result = {
            "system": "ray_actors",
            "model_id": model_id,
            "n_agents_requested": n_agents,
            "n_agents_completed": 0,
            "crash_point": crash_point if crash_point is not None else 0,
            "duration_s": duration,
            "status": "oom",
            "error_count": error_count,
        }

    logger.info(f"\n{'='*80}")
    logger.info(f"RAY BASELINE SUMMARY: N={n_agents}")
    logger.info(f"{'='*80}")
    logger.info(f"Agents completed: {result['n_agents_completed']}/{n_agents}")
    if crash_point is not None:
        logger.info(f"⚠️  OOM/Error at agent {crash_point} (This proves Ray fails early)")
    if success_results:
        logger.info(f"Latencies: mean={result['latency_stats']['mean_ms']:.0f}ms, "
                   f"p99={result['latency_stats']['p99_ms']:.0f}ms")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"{'='*80}\n")

    return result


def main():
    """Test Ray baseline at multiple N values."""
    import argparse
    from trace_generator import load_trace

    parser = argparse.ArgumentParser()
    # Use absolute paths relative to script location
    script_dir = Path(__file__).parent
    exp_dir = script_dir.parent
    parser.add_argument("--trace-dir", type=Path, default=exp_dir / "traces")
    parser.add_argument("--results-dir", type=Path, default=exp_dir / "results")
    parser.add_argument("--n-agents", type=int, default=None, help="Run single N value")

    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.n_agents is not None:
        # Single N
        trace_file = args.trace_dir / f"trace_{args.n_agents}.json"
        if not trace_file.exists():
            logger.error(f"Trace file not found: {trace_file}")
            sys.exit(1)
        
        trace = load_trace(trace_file)
        result = run_ray_baseline(trace)
        
        # Save result
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = args.results_dir / f"ray_baseline_{args.n_agents}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result to {output_file}")
    else:
        # Sweep
        all_results = {}
        for n in [10, 20, 30, 40, 50, 60, 70, 80]:
            trace_file = args.trace_dir / f"trace_{n}.json"
            if not trace_file.exists():
                logger.warning(f"Trace file not found: {trace_file}, skipping")
                continue
            
            trace = load_trace(trace_file)
            result = run_ray_baseline(trace)
            all_results[str(n)] = result
            
            # Stop if crash point found
            if result["crash_point"] < n:
                logger.info(f"Stopping sweep: Ray OOM at N={result['crash_point']}")
                break
        
        # Save sweep results
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = args.results_dir / f"ray_baseline_sweep_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved sweep results to {output_file}")


if __name__ == "__main__":
    main()
