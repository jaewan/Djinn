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
    Run N concurrent agents with vLLM using BATCHED GENERATION (true concurrency).
    
    Args:
        n_agents: Number of concurrent agents
        config: Workload configuration
        
    Returns:
        Result dict or None if failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"vLLM Baseline: N={n_agents} agents (BATCHED CONCURRENT)")
    logger.info(f"{'='*80}\n")
    
    try:
        # Initialize vLLM
        logger.info(f"Initializing vLLM with max_num_seqs={n_agents}...")
        
        # Check free memory first
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"Free GPU memory: {free_gb:.1f} GB")
        
        llm = LLM(
            model="meta-llama/Llama-2-7b-hf",
            dtype="float16",
            max_num_seqs=n_agents,
            gpu_memory_utilization=0.70,  # Conservative to avoid memory issues
            swap_space=0,  # NO SWAPPING - measure pure vLLM
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            max_model_len=2048,  # Allow 1024 token context + 50 output
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create prompt (1024 tokens - same as Djinn)
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
        
        # CRITICAL: Send N copies as a BATCH in a single call
        # This tests true concurrent scheduling (like real multi-user scenario)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,
            top_p=1.0,
        )
        
        # Create N identical prompts
        prompts = [prompt_text] * n_agents
        
        logger.info(f"Submitting {n_agents} prompts in single batched call (concurrent mode)...")
        
        start_time = time.perf_counter()
        
        try:
            # This is the key test: vLLM batches all N requests together
            # If it OOMs, it will fail here
            outputs = llm.generate(prompts, sampling_params)
            
            duration = time.perf_counter() - start_time
            
            # Calculate per-request latency (assumes even distribution)
            # In reality vLLM processes these in parallel/interleaved
            per_request_latency_ms = (duration / n_agents) * 1000
            
            result = {
                "status": "success",
                "n_agents": n_agents,
                "agents_completed": n_agents,
                "duration_s": duration,
                "total_requests": len(outputs),
                "latency_stats": {
                    "per_request_ms": per_request_latency_ms,
                    "total_batch_ms": duration * 1000,
                    "throughput_reqs_per_sec": n_agents / duration,
                },
            }
            
            logger.info(f"\n✅ vLLM N={n_agents} Success (BATCHED):")
            logger.info(f"   Total Duration: {duration:.1f}s")
            logger.info(f"   Per-request Latency: {per_request_latency_ms:.1f}ms")
            logger.info(f"   Throughput: {n_agents / duration:.2f} requests/s")
            logger.info(f"   Successfully generated {len(outputs)} outputs")
            
            # Clean up
            del llm
            torch.cuda.empty_cache()
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            duration = time.perf_counter() - start_time
            logger.error(f"❌ OOM at N={n_agents} after {duration:.1f}s: {e}")
            return {
                "status": "oom",
                "n_agents": n_agents,
                "agents_completed": 0,
                "reason": "torch.cuda.OutOfMemoryError during batched generation",
                "duration_s": duration,
            }
        
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
    """Run vLLM baseline at multiple N values to find OOM cliff."""
    
    # Ensure GPU is clean (kill any existing processes)
    import subprocess
    logger.info("Cleaning GPU...")
    subprocess.run(["fuser", "-v", "/dev/nvidia*"], stderr=subprocess.DEVNULL)
    
    logger.info("=" * 80)
    logger.info("vLLM BASELINE EXPERIMENT - CLIFF DETECTION")
    logger.info("=" * 80)
    logger.info("\nIMPORTANT: This test should run on a clean GPU (no Djinn server)")
    logger.info("Testing batched concurrent requests to find OOM cliff point")
    logger.info("Target: Find N where vLLM fails (should be ~45-50)\n")
    
    # Check free memory
    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU Memory: {free_gb:.1f}/{total_gb:.1f} GB free\n")
    
    # Test sequence: sweep to find cliff
    # Start at 10, increase in steps of 10 until we hit the cliff
    test_n_values = [10, 20, 30, 40, 45, 48, 50, 55, 60]
    results_by_n = {}
    oom_found = False
    cliff_point = None
    
    for n in test_n_values:
        if oom_found:
            logger.info(f"\n⚠️  Skipping N={n} (OOM already found at N={cliff_point})")
            results_by_n[n] = {
                "status": "skipped",
                "n_agents": n,
                "reason": f"OOM found at N={cliff_point}",
            }
            continue
        
        result = run_vllm_experiment(n, {})
        results_by_n[n] = result
        
        if result and result["status"] in ["oom", "oom_init"]:
            oom_found = True
            cliff_point = n
            logger.error(f"\n🔴 OUT OF MEMORY at N={n} - CLIFF FOUND!")
        
        # Cool down between tests
        time.sleep(3)
    
    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("vLLM BASELINE SUMMARY - CLIFF ANALYSIS")
    logger.info("=" * 80 + "\n")
    
    successful_ns = []
    
    for n in test_n_values:
        result = results_by_n[n]
        status = result["status"]
        
        if status == "success":
            stats = result["latency_stats"]
            logger.info(f"✅ N={n:2d}: {stats['total_batch_ms']:7.1f}ms total "
                       f"({stats['per_request_ms']:.1f}ms/req, "
                       f"{stats['throughput_reqs_per_sec']:.2f} req/s)")
            successful_ns.append(n)
        elif status == "oom":
            logger.info(f"❌ N={n:2d}: OUT OF MEMORY - vLLM CRASH")
            if cliff_point is None:
                cliff_point = n
        elif status == "oom_init":
            logger.info(f"❌ N={n:2d}: OOM during initialization")
            if cliff_point is None:
                cliff_point = n
        elif status == "skipped":
            logger.info(f"⊘  N={n:2d}: Skipped (OOM at N={cliff_point})")
        else:
            logger.info(f"❌ N={n:2d}: Error - {result.get('reason', 'unknown')}")
    
    logger.info("\n" + "-" * 80)
    if cliff_point:
        logger.info(f"🔴 vLLM CRASH POINT: N={cliff_point}")
        logger.info(f"✅ Successful up to: N={max(successful_ns) if successful_ns else 0}")
    else:
        logger.info(f"✅ No crash detected. Tested up to N={max(successful_ns) if successful_ns else 0}")
    logger.info("-" * 80)
    
    # Save results
    output_dir = Path("OSDI_Evaluation/exp1_semantic_scheduler/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"vllm_baseline_{timestamp}.json"
    
    payload = {
        "tag": "vllm_baseline_cliff",
        "model_id": "meta-llama/Llama-2-7b-hf",
        "generated_at": timestamp,
        "experiment": {
            "type": "baseline_batched_concurrent",
            "framework": "vllm",
            "description": "True concurrent batched requests to find OOM cliff",
        },
        "cliff_analysis": {
            "cliff_point": cliff_point,
            "max_successful_n": max(successful_ns) if successful_ns else 0,
            "oom_found": oom_found,
        },
        "results_by_n": results_by_n,
    }
    
    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_file}")
    logger.info(f"\n📊 KEY FINDING: vLLM crash point is N={cliff_point}")
    logger.info(f"📊 Djinn target: Scale beyond N={cliff_point} with 100% success rate")
    
    # Return exit code based on OOM
    return 1 if oom_found else 0


if __name__ == "__main__":
    exit(main())

