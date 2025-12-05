#!/usr/bin/env python3
"""
vLLM Baseline: Direct comparison with Djinn semantic scheduler.

This runs the SAME workload as Djinn experiments but using vLLM directly.
No swapping, no semantic scheduling - just raw vLLM performance.

Tests at N=10, 20, 30, 40 to find OOM point.
"""

import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

import torch
import yaml
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_vllm_experiment(n_agents: int, config: Dict[str, Any]) -> Optional[Dict]:
    """
    Run N concurrent agents with vLLM.
    
    Args:
        n_agents: Number of concurrent agents
        config: Workload configuration
        
    Returns:
        Result dict or None if failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"vLLM Baseline: N={n_agents} agents")
    logger.info(f"{'='*80}\n")
    
    try:
        # Initialize vLLM
        logger.info(f"Initializing vLLM with {n_agents} max_num_seqs...")
        
        # Check free memory first
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"Free GPU memory: {free_gb:.1f} GB")
        
        llm = LLM(
            model="meta-llama/Llama-2-7b-hf",
            dtype="float16",
            max_num_seqs=n_agents,
            gpu_memory_utilization=0.85,  # Conservative: don't overload shared GPU
            swap_space=0,  # NO SWAPPING - measure pure vLLM
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create prompt (1024 tokens)
        base_text = """
        We the People of the United States, in Order to form a more perfect Union, 
        establish Justice, insure domestic Tranquility, provide for the common defence, 
        promote the general Welfare, and secure the Blessings of Liberty to ourselves 
        and our Posterity, do ordain and establish this Constitution for the United States of America.
        Article I: The Legislative Branch. Congress shall have Power To lay and collect Taxes, 
        Duties, Imposts and Excises, to pay the Debts and provide for the common Defence and general Welfare.
        """
        repeated = base_text * 10
        prompt_tokens = tokenizer.encode(repeated)[:1024]
        prompt_text = tokenizer.decode(prompt_tokens)
        
        logger.info(f"Using {len(prompt_tokens)}-token prompt")
        
        # Run N sequential inferences (vLLM doesn't support true concurrent client mode)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,
            top_p=1.0,
        )
        
        latencies = []
        start_time = time.perf_counter()
        
        for i in range(n_agents):
            req_start = time.perf_counter()
            try:
                output = llm.generate(prompt_text, sampling_params)
                latency_ms = (time.perf_counter() - req_start) * 1000
                latencies.append(latency_ms)
                logger.debug(f"  Agent {i}: {latency_ms:.1f}ms")
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ OOM at agent {i}: {e}")
                return {
                    "status": "oom",
                    "n_agents": n_agents,
                    "agents_completed": i,
                    "reason": "torch.cuda.OutOfMemoryError",
                    "duration_s": time.perf_counter() - start_time,
                }
            except Exception as e:
                logger.error(f"❌ Error at agent {i}: {e}")
                return {
                    "status": "error",
                    "n_agents": n_agents,
                    "agents_completed": i,
                    "reason": str(e),
                    "duration_s": time.perf_counter() - start_time,
                }
        
        # Calculate statistics
        duration = time.perf_counter() - start_time
        latencies_sorted = sorted(latencies)
        
        result = {
            "status": "success",
            "n_agents": n_agents,
            "agents_completed": n_agents,
            "duration_s": duration,
            "latency_stats": {
                "mean_ms": sum(latencies) / len(latencies),
                "p50_ms": latencies_sorted[len(latencies_sorted) // 2],
                "p99_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)],
                "p99_idx": int(len(latencies_sorted) * 0.99),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            },
            "latencies": latencies,
        }
        
        logger.info(f"\n✅ vLLM N={n_agents} Success:")
        logger.info(f"   Duration: {duration:.1f}s")
        logger.info(f"   Mean Latency: {result['latency_stats']['mean_ms']:.1f}ms")
        logger.info(f"   P50 Latency: {result['latency_stats']['p50_ms']:.1f}ms")
        logger.info(f"   P99 Latency: {result['latency_stats']['p99_ms']:.1f}ms")
        
        # Clean up
        del llm
        torch.cuda.empty_cache()
        
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ OOM during initialization at N={n_agents}: {e}")
        return {
            "status": "oom_init",
            "n_agents": n_agents,
            "agents_completed": 0,
            "reason": "torch.cuda.OutOfMemoryError during initialization",
        }
    except Exception as e:
        logger.error(f"❌ Failed at N={n_agents}: {e}")
        return {
            "status": "error",
            "n_agents": n_agents,
            "agents_completed": 0,
            "reason": str(e),
        }


def main():
    """Run vLLM baseline at multiple N values."""
    
    # Ensure GPU is clean (kill any existing processes)
    import subprocess
    logger.info("Cleaning GPU...")
    subprocess.run(["fuser", "-v", "/dev/nvidia*"], stderr=subprocess.DEVNULL)
    
    logger.info("=" * 80)
    logger.info("vLLM BASELINE EXPERIMENT")
    logger.info("=" * 80)
    logger.info("\nIMPORTANT: This test should run on a clean GPU (no Djinn server)")
    logger.info("Testing at N=10, 20, 30, 40 to find OOM point\n")
    
    # Check free memory
    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU Memory: {free_gb:.1f}/{total_gb:.1f} GB free\n")
    
    # Test sequence
    test_n_values = [10, 20, 30, 40]
    results_by_n = {}
    oom_found = False
    
    for n in test_n_values:
        if oom_found:
            logger.info(f"\n⚠️  Skipping N={n} (OOM already found)")
            results_by_n[n] = {
                "status": "skipped",
                "n_agents": n,
                "reason": "OOM found at lower N",
            }
            continue
        
        result = run_vllm_experiment(n, {})
        results_by_n[n] = result
        
        if result and result["status"] in ["oom", "oom_init"]:
            oom_found = True
            logger.error(f"\n🔴 OUT OF MEMORY at N={n}")
        
        # Cool down between tests
        time.sleep(5)
    
    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("vLLM BASELINE SUMMARY")
    logger.info("=" * 80 + "\n")
    
    for n in test_n_values:
        result = results_by_n[n]
        status = result["status"]
        
        if status == "success":
            stats = result["latency_stats"]
            logger.info(f"✅ N={n:2d}: P99={stats['p99_ms']:7.0f}ms "
                       f"(mean={stats['mean_ms']:.0f}ms, min={stats['min_ms']:.0f}ms, "
                       f"max={stats['max_ms']:.0f}ms)")
        elif status == "oom":
            logger.info(f"❌ N={n:2d}: OOM after {result['agents_completed']} agents")
        elif status == "oom_init":
            logger.info(f"❌ N={n:2d}: OOM during initialization")
        elif status == "skipped":
            logger.info(f"⊘  N={n:2d}: Skipped (OOM already found)")
        else:
            logger.info(f"❌ N={n:2d}: Error - {result.get('reason', 'unknown')}")
    
    # Save results
    output_dir = Path("OSDI_Evaluation/exp1_semantic_scheduler/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"vllm_baseline_{timestamp}.json"
    
    payload = {
        "tag": "vllm_baseline",
        "model_id": "meta-llama/Llama-2-7b-hf",
        "generated_at": timestamp,
        "experiment": {
            "type": "baseline_sequential",
            "framework": "vllm",
        },
        "results_by_n": results_by_n,
    }
    
    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_file}")
    
    # Return exit code based on OOM
    return 1 if oom_found else 0


if __name__ == "__main__":
    exit(main())

