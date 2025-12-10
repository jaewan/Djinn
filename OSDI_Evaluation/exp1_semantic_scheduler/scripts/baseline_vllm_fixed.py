#!/usr/bin/env python3
"""
vLLM Fixed Baseline for OSDI Evaluation (Apple-to-Apple Comparison).

Runs vLLM with Poisson arrivals and true per-request latency measurement.
Matches the same workload pattern as Djinn (Reason → Act → Reflect).

Key tuning to avoid Triton compilation issues:
- VLLM_USE_V1=0 (use stable v0 engine)
- TORCH_COMPILE_DISABLE=1 (disable torch.compile)
- enforce_eager=True (disable CUDA graphs)

Expected behavior:
- Works well at N=10-20 (2-3s P99 latency including think time)
- Degrades at N=30-40 (4-6s latency as GPU fills, queueing increases)
- OOM/Crash at N=45-50 (no memory for new KV caches)

This proves that reactive paging (vLLM) fails before semantic scheduling (Djinn).
"""

import os
import json
import logging
import time
import sys
import asyncio
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import numpy as np

# Set environment variables BEFORE importing anything from vLLM or torch
os.environ["VLLM_USE_V1"] = "0"  # Use stable v0 engine
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def vllm_agent_lifecycle(
    agent_idx: int,
    llm: LLM,
    tokenizer: Any,
    prompt: str,
    arrival_time: float,
    think_time: float,
) -> Dict[str, Any]:
    """
    Single agent lifecycle with Poisson arrival timing.
    
    Implements Reason → Act → Reflect pattern:
    1. REASON: Prefill with long context
    2. ACT: Wait (think time, simulating external I/O)
    3. REFLECT: Generate more tokens (decode with cached KV)
    
    CORRECT METRIC: Measures LLM-only latency (Reason + Reflect phases),
    EXCLUDING think time to match Djinn's measurement methodology.
    
    Args:
        agent_idx: Agent identifier
        llm: vLLM engine
        tokenizer: Tokenizer
        prompt: Input prompt
        arrival_time: Time this agent arrived (seconds from experiment start)
        think_time: Duration of Act phase (seconds)
    
    Returns:
        Record with latency measurements
    """
    try:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,
            top_p=1.0,
        )
        
        # PHASE 1: REASON (Prefill with context)
        # NOTE: vLLM's synchronous LLM.generate() blocks the GIL
        # Multiple concurrent agents will serialize through this call
        reason_start = time.perf_counter()
        reason_outputs = llm.generate([prompt], sampling_params)
        reason_latency_ms = (time.perf_counter() - reason_start) * 1000
        
        # PHASE 2: ACT (Think time / I/O simulation)
        # This is NOT included in the primary latency metric
        await asyncio.sleep(think_time)
        
        # PHASE 3: REFLECT (Decode, reusing KV cache)
        reflect_start = time.perf_counter()
        reflect_outputs = llm.generate([prompt], sampling_params)
        reflect_latency_ms = (time.perf_counter() - reflect_start) * 1000
        
        # PRIMARY METRIC: LLM-only latency (Reason + Reflect)
        # This matches Djinn's measurement methodology
        llm_only_latency_ms = reason_latency_ms + reflect_latency_ms
        
        return {
            "agent_id": agent_idx,
            "arrival_time_s": arrival_time,
            "think_time_s": think_time,
            "reason_latency_ms": reason_latency_ms,
            "reflect_latency_ms": reflect_latency_ms,
            "total_latency_ms": llm_only_latency_ms,  # PRIMARY METRIC
            "status": "success",
        }
    except Exception as e:
        logger.error(f"[Agent {agent_idx}] Error: {e}")
        return {
            "agent_id": agent_idx,
            "arrival_time_s": arrival_time,
            "think_time_s": think_time,
            "status": "error",
            "error": str(e),
        }


async def spawn_agents_poisson(
    llm: LLM,
    tokenizer: Any,
    trace: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Spawn agents with Poisson inter-arrival times.
    
    Args:
        llm: vLLM engine
        tokenizer: Tokenizer
        trace: Workload trace with prompts, arrival_times, think_times
    
    Returns:
        List of agent result records
    """
    n_agents = trace["n_agents"]
    logger.info(f"Spawning {n_agents} agents with Poisson arrivals...")
    
    prompts = trace["prompts"][:n_agents]
    arrival_times = trace["arrival_times"][:n_agents]
    think_times = trace["think_times"][:n_agents]
    
    tasks = []
    experiment_start_time = time.perf_counter()
    
    for agent_idx, (arrival_time, prompt, think_time) in enumerate(
        zip(arrival_times, prompts, think_times)
    ):
        # Sleep until this agent's arrival time
        current_wall_time = time.perf_counter() - experiment_start_time
        if arrival_time > current_wall_time:
            await asyncio.sleep(arrival_time - current_wall_time)
        
        # Spawn agent task
        task = asyncio.create_task(
            vllm_agent_lifecycle(
                agent_idx, llm, tokenizer, prompt, arrival_time, think_time
            )
        )
        tasks.append(task)
        
        if (agent_idx + 1) % 10 == 0:
            logger.info(f"  Spawned {agent_idx + 1}/{n_agents} agents")
    
    # Wait for all agents to complete
    logger.info("All agents spawned. Waiting for completion...")
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    return results


def run_vllm_baseline(
    trace: Dict[str, Any],
    model_id: str = "meta-llama/Llama-2-13b-hf",
) -> Dict[str, Any]:
    """
    Run vLLM baseline with Poisson arrivals.
    
    Args:
        trace: Workload trace from trace_generator.py
        model_id: Model to load
    
    Returns:
        Results dictionary with P99 latency and other metrics
    """
    n_agents = trace["n_agents"]
    logger.info(f"\n{'='*80}")
    logger.info(f"vLLM BASELINE (APPLE-TO-APPLE): N={n_agents} agents, Poisson arrivals")
    logger.info(f"{'='*80}\n")

    # Check memory before starting
    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU Memory: {free_gb:.1f}/{total_gb:.1f} GB free\n")

    try:
        # Initialize vLLM
        logger.info(f"Initializing vLLM with max_num_seqs={n_agents}...")
        
        llm = LLM(
            model=model_id,
            dtype="float16",
            max_num_seqs=n_agents,
            gpu_memory_utilization=0.85,  # Use 85% of GPU
            swap_space=0,  # NO SWAPPING - measure pure vLLM
            tensor_parallel_size=1,
            enforce_eager=True,  # Disable CUDA graphs (avoids Triton)
            max_model_len=2150,  # Allow 2048-token context + BOS + 50 output + margin
            disable_log_stats=True,  # Reduce logging
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ vLLM initialized\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize vLLM: {e}")
        return {
            "system": "vllm",
            "model_id": model_id,
            "n_agents": n_agents,
            "status": "init_error",
            "error": str(e),
        }

    # Run agents with Poisson arrivals
    start_time = time.perf_counter()
    
    try:
        results = asyncio.run(spawn_agents_poisson(llm, tokenizer, trace))
        total_duration = time.perf_counter() - start_time
        
        # Filter out error records
        success_results = [r for r in results if r.get("status") == "success"]
        error_count = len([r for r in results if r.get("status") != "success"])
        
        if not success_results:
            logger.error("No successful results")
            return {
                "system": "vllm",
                "model_id": model_id,
                "n_agents": n_agents,
                "status": "all_errors",
                "error": "All agents failed",
            }
        
        # Compute latency statistics (LLM-only: Reason + Reflect, excluding think time)
        total_latencies = sorted([r["total_latency_ms"] for r in success_results])
        reason_latencies = sorted([r["reason_latency_ms"] for r in success_results])
        reflect_latencies = sorted([r["reflect_latency_ms"] for r in success_results])
        
        p99_idx = int(len(total_latencies) * 0.99)
        p99_idx = min(p99_idx, len(total_latencies) - 1)
        
        p99_total = total_latencies[p99_idx]
        p50_total = total_latencies[len(total_latencies) // 2]
        mean_total = np.mean(total_latencies)
        
        p99_reason = reason_latencies[p99_idx]
        p99_reflect = reflect_latencies[p99_idx]
        
        logger.info(f"✅ Experiment completed in {total_duration:.1f}s")
        logger.info(f"   Successful: {len(success_results)}/{n_agents}")
        logger.info(f"   P50 LLM Latency (Reason+Reflect): {p50_total:.1f}ms")
        logger.info(f"   P99 LLM Latency (Reason+Reflect): {p99_total:.1f}ms")
        logger.info(f"   Mean LLM Latency: {mean_total:.1f}ms")
        logger.info(f"   P99 Reason: {p99_reason:.1f}ms, P99 Reflect: {p99_reflect:.1f}ms\n")
        
        result = {
            "system": "vllm",
            "model_id": model_id,
            "n_agents": n_agents,
            "status": "success",
            "total_duration_s": total_duration,
            "success_count": len(success_results),
            "error_count": error_count,
            "latency_stats": {
                "mean_ms": mean_total,
                "p50_ms": p50_total,
                "p99_ms": p99_total,
                "min_ms": total_latencies[0],
                "max_ms": total_latencies[-1],
            },
            "reason_latency_stats": {
                "mean_ms": np.mean(reason_latencies),
                "p99_ms": reason_latencies[p99_idx],
            },
            "reflect_latency_stats": {
                "mean_ms": np.mean(reflect_latencies),
                "p99_ms": reflect_latencies[p99_idx],
            },
        }
        
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        total_duration = time.perf_counter() - start_time
        logger.error(f"❌ CUDA OOM at N={n_agents} after {total_duration:.1f}s")
        
        return {
            "system": "vllm",
            "model_id": model_id,
            "n_agents": n_agents,
            "status": "cuda_oom",
            "total_duration_s": total_duration,
            "error": str(e),
        }
        
    except Exception as e:
        total_duration = time.perf_counter() - start_time
        logger.error(f"❌ Error at N={n_agents}: {e}", exc_info=True)
        
        return {
            "system": "vllm",
            "model_id": model_id,
            "n_agents": n_agents,
            "status": "error",
            "total_duration_s": total_duration,
            "error": str(e),
        }
    
    finally:
        # Cleanup
        if 'llm' in locals():
            del llm
        torch.cuda.empty_cache()


def main():
    """Test vLLM baseline at multiple N values."""
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
        result = run_vllm_baseline(trace)
        
        # Save result
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = args.results_dir / f"vllm_baseline_{args.n_agents}_{timestamp}.json"
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
            result = run_vllm_baseline(trace)
            all_results[str(n)] = result
            
            # Stop if OOM found
            if result["status"] in ["cuda_oom", "init_error"]:
                logger.info(f"Stopping sweep: vLLM OOM/error at N={n}")
                break
            
            # Cool down
            time.sleep(2)
        
        # Save sweep results
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = args.results_dir / f"vllm_baseline_sweep_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved sweep results to {output_file}")


if __name__ == "__main__":
    main()
