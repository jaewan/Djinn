#!/usr/bin/env python3
"""
Simple vLLM Baseline: Direct sequential inference comparison.

Avoids multiprocessing complications by using vLLM's synchronous API.
"""

import json
import time
import torch
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def measure_vllm_latency(n_inferences: int, gpu_memory_util: float = 0.7) -> Optional[Dict]:
    """
    Measure vLLM latency for N sequential inferences.
    
    Args:
        n_inferences: Number of sequential inferences to measure
        gpu_memory_util: GPU memory utilization fraction
        
    Returns:
        Dictionary with latency statistics or None if failed
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"vLLM Baseline: {n_inferences} sequential inferences")
        logger.info(f"{'='*80}\n")
        
        # Check GPU memory
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Memory: {free_gb:.1f}/{total_gb:.1f} GB free")
        
        # Initialize vLLM with conservative settings
        logger.info(f"Initializing vLLM (batch_size=1)...")
        llm = LLM(
            model="meta-llama/Llama-2-7b-hf",
            dtype="float16",
            gpu_memory_utilization=gpu_memory_util,
            swap_space=0,
            max_num_seqs=1,  # Sequential only
            disable_log_stats=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create 1024-token prompt
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
        
        logger.info(f"Using {len(prompt_tokens)}-token prompt\n")
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,
            top_p=1.0,
        )
        
        # Warm up
        logger.info("Warming up...")
        llm.generate(prompt_text, sampling_params)
        
        # Measure latencies
        logger.info(f"Running {n_inferences} inferences...")
        latencies = []
        start_total = time.perf_counter()
        
        for i in range(n_inferences):
            req_start = time.perf_counter()
            try:
                output = llm.generate(prompt_text, sampling_params)
                latency_ms = (time.perf_counter() - req_start) * 1000
                latencies.append(latency_ms)
                if (i + 1) % 5 == 0:
                    logger.debug(f"  Inference {i+1}/{n_inferences}: {latency_ms:.1f}ms")
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ OOM at inference {i}")
                return {
                    "status": "oom",
                    "n_inferences": n_inferences,
                    "completed": i,
                    "reason": "torch.cuda.OutOfMemoryError",
                }
        
        total_time = time.perf_counter() - start_total
        
        # Statistics
        latencies_sorted = sorted(latencies)
        result = {
            "status": "success",
            "n_inferences": n_inferences,
            "total_time_s": total_time,
            "throughput_inf_per_s": n_inferences / total_time,
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "p50": latencies_sorted[len(latencies_sorted) // 2],
                "p99": latencies_sorted[int(len(latencies_sorted) * 0.99)],
                "max": max(latencies),
            }
        }
        
        logger.info(f"\n✅ Success:")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Throughput: {result['throughput_inf_per_s']:.2f} inf/s")
        logger.info(f"   Mean latency: {result['latency_ms']['mean']:.1f}ms")
        logger.info(f"   P99 latency: {result['latency_ms']['p99']:.1f}ms")
        
        del llm
        torch.cuda.empty_cache()
        
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ OOM during init: {e}")
        return {
            "status": "oom_init",
            "n_inferences": n_inferences,
            "completed": 0,
            "reason": "torch.cuda.OutOfMemoryError",
        }
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {
            "status": "error",
            "n_inferences": n_inferences,
            "completed": 0,
            "reason": str(e),
        }


def main():
    """Run vLLM baseline measurements."""
    
    logger.info("=" * 80)
    logger.info("vLLM SIMPLE BASELINE")
    logger.info("=" * 80)
    logger.info("\nMeasuring latency for sequential inferences\n")
    
    test_configs = [
        {"n_inferences": 10, "util": 0.6},
        {"n_inferences": 20, "util": 0.6},
        {"n_inferences": 40, "util": 0.6},
    ]
    
    results = {}
    
    for config in test_configs:
        n = config["n_inferences"]
        util = config["util"]
        
        result = measure_vllm_latency(n, util)
        results[n] = result
        
        if result["status"] == "oom":
            logger.error(f"⚠️  OOM - stopping tests")
            break
        
        time.sleep(5)  # Cool down
    
    # Report
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80 + "\n")
    
    for n in [10, 20, 40]:
        if n not in results:
            continue
        r = results[n]
        if r["status"] == "success":
            logger.info(f"✅ N={n:2d}: P99={r['latency_ms']['p99']:6.0f}ms "
                       f"(mean={r['latency_ms']['mean']:.0f}ms, "
                       f"throughput={r['throughput_inf_per_s']:.2f} inf/s)")
        else:
            logger.info(f"❌ N={n:2d}: {r['status']} - {r.get('reason', 'unknown')}")
    
    # Save
    output_dir = Path("OSDI_Evaluation/exp1_semantic_scheduler/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"vllm_baseline_simple_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "tag": "vllm_baseline_simple",
            "model": "meta-llama/Llama-2-7b-hf",
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)
    
    logger.info(f"\n✅ Saved to {output_file}")


if __name__ == "__main__":
    main()



