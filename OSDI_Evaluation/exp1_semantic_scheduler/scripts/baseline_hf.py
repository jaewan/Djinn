#!/usr/bin/env python3
"""
HuggingFace Baseline: Direct transformer inference (no vLLM complications).

Measures latency for sequential inference using HF transformers directly.
"""

import json
import time
import torch
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def measure_hf_latency(n_inferences: int) -> Optional[Dict]:
    """
    Measure HF transformers latency for N sequential inferences.
    
    Args:
        n_inferences: Number of sequential inferences to measure
        
    Returns:
        Dictionary with latency statistics or None if failed
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"HuggingFace Baseline: {n_inferences} sequential inferences")
        logger.info(f"{'='*80}\n")
        
        # Check GPU memory
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Memory: {free_gb:.1f}/{total_gb:.1f} GB free")
        
        # Load model once
        logger.info("Loading Llama-2-7B...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ Model loaded")
        
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
        
        # Warm up
        logger.info("Warming up...")
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=50, do_sample=False)
        del inputs
        torch.cuda.empty_cache()
        
        # Measure latencies
        logger.info(f"Running {n_inferences} inferences...")
        latencies = []
        start_total = time.perf_counter()
        
        for i in range(n_inferences):
            req_start = time.perf_counter()
            try:
                inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                latency_ms = (time.perf_counter() - req_start) * 1000
                latencies.append(latency_ms)
                if (i + 1) % 5 == 0:
                    logger.debug(f"  Inference {i+1}/{n_inferences}: {latency_ms:.1f}ms")
                del inputs, output
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ OOM at inference {i}")
                return {
                    "status": "oom",
                    "n_inferences": n_inferences,
                    "completed": i,
                    "reason": "torch.cuda.OutOfMemoryError",
                }
            except Exception as e:
                logger.error(f"❌ Error at inference {i}: {e}")
                return {
                    "status": "error",
                    "n_inferences": n_inferences,
                    "completed": i,
                    "reason": str(e),
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
        
        del model
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
    """Run HF baseline measurements."""
    
    logger.info("=" * 80)
    logger.info("HUGGINGFACE BASELINE")
    logger.info("=" * 80)
    logger.info("\nMeasuring latency for sequential inferences\n")
    
    test_configs = [
        {"n_inferences": 5},
        {"n_inferences": 10},
        {"n_inferences": 20},
    ]
    
    results = {}
    
    for config in test_configs:
        n = config["n_inferences"]
        result = measure_hf_latency(n)
        results[n] = result
        
        if result and result["status"] == "oom":
            logger.error(f"⚠️  OOM - stopping tests")
            break
        
        time.sleep(5)  # Cool down
    
    # Report
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80 + "\n")
    
    for n in [5, 10, 20]:
        if n not in results:
            continue
        r = results[n]
        if r and r["status"] == "success":
            logger.info(f"✅ N={n:2d}: P99={r['latency_ms']['p99']:6.0f}ms "
                       f"(mean={r['latency_ms']['mean']:.0f}ms, "
                       f"throughput={r['throughput_inf_per_s']:.2f} inf/s)")
        elif r:
            logger.info(f"❌ N={n:2d}: {r['status']} - {r.get('reason', 'unknown')}")
        else:
            logger.info(f"⊘  N={n:2d}: Skipped")
    
    # Save
    output_dir = Path("OSDI_Evaluation/exp1_semantic_scheduler/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"hf_baseline_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "tag": "hf_baseline",
            "model": "meta-llama/Llama-2-7b-hf",
            "framework": "huggingface_transformers",
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)
    
    logger.info(f"\n✅ Saved to {output_file}")


if __name__ == "__main__":
    main()



