#!/usr/bin/env python3
"""
Ray Actors Baseline for OSDI Evaluation.

Tests process-level GPU management using Ray remote actors.
Each actor loads the FULL model into GPU memory (26GB for Llama-2-13B).

Expected behavior: OOM crash at N=2-3 because multiple 26GB models
cannot fit on an 80GB H100.

This proves that without semantic scheduling (Djinn), process-level
memory management fails at low concurrency.
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0.1)  # Fractional GPU allows Ray to schedule multiple actors
class RayAgentActor:
    """
    Remote actor that loads a full LLM model into GPU.
    
    This simulates how a developer would use Ray to serve multiple agents,
    where each agent gets its own actor.
    """

    def __init__(self, model_id: str):
        """Load model into GPU memory."""
        logger.info(f"[Actor] Loading {model_id} into GPU")
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
        except Exception as e:
            logger.error(f"[Actor] Failed to load model: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text from prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"[Actor] Generation failed: {e}")
            raise


def run_ray_baseline(
    trace: Dict[str, Any],
    model_id: str = "meta-llama/Llama-2-13b-hf",
) -> Dict[str, Any]:
    """
    Run Ray baseline at specified N.

    Expected: OOM at N=2-3 when each actor tries to load 26GB model.

    Args:
        trace: Workload trace (from trace_generator.py)
        model_id: Model to load

    Returns:
        Results dictionary with crash point and latencies
    """
    n_agents = trace["n_agents"]
    logger.info(f"\n{'='*80}")
    logger.info(f"RAY ACTORS BASELINE: N={n_agents}")
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
        # PHASE 1: Spawn all actors first (without blocking on inference)
        logger.info(f"Spawning {n_agents} Ray actors...")
        
        for agent_id, (arrival_time, prompt, think_time) in enumerate(
            zip(trace["arrival_times"], trace["prompts"], trace["think_times"])
        ):
            try:
                # Create remote actor
                actor = RayAgentActor.remote(model_id)
                actors.append((agent_id, actor, prompt, arrival_time))
                
                if (agent_id + 1) % 5 == 0:
                    logger.info(f"  Spawned {agent_id + 1}/{n_agents} actors")
            
            except Exception as e:
                logger.error(f"❌ Failed to spawn actor {agent_id}: {e}")
                crash_point = agent_id
                break
        
        # PHASE 2: Run all inference concurrently using ray.get on all at once
        if actors:
            logger.info(f"Running inference on {len(actors)} actors concurrently...")
            
            # Prepare all generation tasks
            generation_tasks = [
                (agent_id, actor.generate.remote(prompt, max_tokens=50), arrival_time)
                for agent_id, actor, prompt, arrival_time in actors
            ]
            
            # Wait for all to complete and measure latency
            try:
                req_start = time.perf_counter()
                outputs = ray.get(
                    [task[1] for task in generation_tasks],
                    timeout=300,  # 5 min total timeout
                )
                total_concurrent_time_ms = (time.perf_counter() - req_start) * 1000
                
                # Record results
                for (agent_id, _, arrival_time), output in zip(generation_tasks, outputs):
                    results.append({
                        "agent_id": agent_id,
                        "status": "success",
                        "latency_ms": total_concurrent_time_ms,  # All agents see same latency
                        "arrival_time_s": arrival_time,
                    })
                
                logger.info(f"  ✅ All {len(actors)} concurrent inference completed in {total_concurrent_time_ms:.0f}ms")
            
            except (ray.exceptions.OutOfMemoryError, torch.cuda.OutOfMemoryError) as e:
                logger.error(f"❌ OOM during concurrent execution: {e}")
                crash_point = len(actors)
                results.append({
                    "agent_id": len(actors),
                    "status": "oom",
                    "error": str(e),
                })
            
            except Exception as e:
                logger.error(f"❌ Error during concurrent execution: {e}")
                crash_point = len(actors)
                results.append({
                    "agent_id": len(actors),
                    "status": "error",
                    "error": str(e),
                })

    finally:
        ray.shutdown()

    duration = time.perf_counter() - start_time

    # Calculate statistics
    success_results = [r for r in results if r["status"] == "success"]
    latencies = [r["latency_ms"] for r in success_results]

    result = {
        "system": "ray_actors",
        "model_id": model_id,
        "n_agents_requested": n_agents,
        "n_agents_completed": len(success_results),
        "crash_point": crash_point if crash_point is not None else n_agents,
        "duration_s": duration,
        "results": results,
        "status": "success" if crash_point is None else "oom",
    }

    if latencies:
        latencies_sorted = sorted(latencies)
        result.update(
            {
                "latency_stats": {
                    "mean_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "p50_ms": latencies_sorted[len(latencies_sorted) // 2],
                    "p99_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)],
                }
            }
        )

    logger.info(f"\n{'='*80}")
    logger.info(f"RAY BASELINE SUMMARY: N={n_agents}")
    logger.info(f"{'='*80}")
    logger.info(f"Agents completed: {len(success_results)}/{n_agents}")
    if crash_point is not None:
        logger.info(f"⚠️  OOM/Error at agent {crash_point} (This proves Ray fails early)")
    if latencies:
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
            all_results[n] = result
            
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
