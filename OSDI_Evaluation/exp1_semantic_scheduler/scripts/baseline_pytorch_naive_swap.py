#!/usr/bin/env python3
"""
PyTorch Naive Swap Baseline (OSDI Comparison).

Standard approach: model.to('cuda') -> generate() -> model.to('cpu')

This represents what a developer would do today without Djinn:
- Load all models to CPU RAM
- For each request: move model to GPU, generate, move back to CPU
- No ring buffer, no smart swapping, just naive .to() calls

Expected behavior:
- High latency due to full model transfers (14GB * 2 = 28GB per request)
- Memory safe (only one model on GPU at a time)
- Simple but slow
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models_to_cpu(model_ids: dict) -> dict:
    """Load all models to CPU RAM."""
    logger.info("=" * 80)
    logger.info("LOADING MODELS TO CPU")
    logger.info("=" * 80)
    
    models = {}
    
    for model_name, model_id in model_ids.items():
        logger.info(f"\nLoading {model_name} ({model_id})...")
        load_start = time.perf_counter()
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map='cpu',
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            load_time = time.perf_counter() - load_start
            
            # Calculate model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            
            logger.info(f"  ✅ {model_name}: {model_size / 1024**3:.2f}GB, loaded in {load_time:.1f}s")
            
            models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'size_gb': model_size / 1024**3,
            }
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ All {len(models)} models loaded to CPU")
    logger.info("=" * 80)
    
    return models


async def naive_swap_agent(
    agent_id: int,
    model_name: str,
    models: dict,
    prompt: str,
    arrival_time: float,
    start_time: float,
    max_tokens: int = 50,
) -> dict:
    """
    Run agent with naive PyTorch swapping.
    
    Standard approach: model.to('cuda') -> generate -> model.to('cpu')
    """
    try:
        # Wait for arrival
        wait_time = arrival_time - (time.perf_counter() - start_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        actual_arrival = time.perf_counter()
        
        model_info = models[model_name]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"]
        
        # PHASE 1: Load model from CPU to GPU
        load_start = time.perf_counter()
        model_gpu = model.to('cuda')
        torch.cuda.synchronize()
        load_time_ms = (time.perf_counter() - load_start) * 1000
        
        # PHASE 2: Run inference
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model_gpu.generate(
                input_ids.to('cuda'),
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        torch.cuda.synchronize()
        gen_time_ms = (time.perf_counter() - gen_start) * 1000
        
        # PHASE 3: Offload model back to CPU
        offload_start = time.perf_counter()
        model_gpu.to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        offload_time_ms = (time.perf_counter() - offload_start) * 1000
        
        total_latency_ms = (time.perf_counter() - actual_arrival) * 1000
        
        return {
            "agent_id": agent_id,
            "model": model_name,
            "success": True,
            "load_time_ms": load_time_ms,
            "gen_time_ms": gen_time_ms,
            "offload_time_ms": offload_time_ms,
            "total_latency_ms": total_latency_ms,
            "output_tokens": outputs.shape[1] - input_ids.shape[1],
        }
        
    except Exception as e:
        logger.error(f"Agent {agent_id} ({model_name}) failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "agent_id": agent_id,
            "model": model_name,
            "success": False,
            "error": str(e),
        }


async def run_baseline(trace_file: Path, max_tokens: int = 50):
    """Run PyTorch naive swap baseline."""
    logger.info("=" * 80)
    logger.info("PYTORCH NAIVE SWAP BASELINE")
    logger.info("=" * 80)
    
    # Load trace
    with open(trace_file) as f:
        trace_data = json.load(f)
    
    agents = trace_data["agents"]
    model_configs = trace_data.get("model_configs", {})
    
    logger.info(f"\nExperiment: {len(agents)} agents")
    
    # Extract model IDs from trace
    # If model_configs not in trace, use default mapping
    if model_configs:
        model_ids = {}
        for model_name, config in model_configs.items():
            model_ids[model_name] = config['model_id']
    else:
        # Default mapping for GPT-2 experiments
        model_ids = {
            'llama-7b': 'gpt2',
            'llama-13b': 'gpt2-medium',
            'mistral-7b': 'gpt2-medium',  # Use gpt2-medium as proxy
        }
    
    # Load all models to CPU
    models = load_models_to_cpu(model_ids)
    
    # Run agents sequentially (one at a time to avoid OOM)
    logger.info("\nRunning agents (sequential to avoid OOM)...")
    start_time = time.perf_counter()
    
    results = []
    for i, agent in enumerate(agents):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(agents)}")
        
        result = await naive_swap_agent(
            agent_id=agent["agent_id"],
            model_name=agent["model"],
            models=models,
            prompt=agent["prompt_text"],
            arrival_time=agent["arrival_time"],
            start_time=start_time,
            max_tokens=max_tokens,
        )
        results.append(result)
    
    experiment_time = time.perf_counter() - start_time
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    if successful:
        load_times = [r["load_time_ms"] for r in successful]
        gen_times = [r["gen_time_ms"] for r in successful]
        offload_times = [r["offload_time_ms"] for r in successful]
        total_latencies = [r["total_latency_ms"] for r in successful]
        
        by_model = {}
        for r in successful:
            model = r["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)
        
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)
        logger.info(f"Completed: {len(successful)}/{len(agents)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"\nLatency Breakdown:")
        logger.info(f"  P50 Load Time: {sorted(load_times)[int(len(load_times)*0.50)]:.1f}ms")
        logger.info(f"  P99 Load Time: {sorted(load_times)[int(len(load_times)*0.99)]:.1f}ms")
        logger.info(f"  P50 Gen Time: {sorted(gen_times)[int(len(gen_times)*0.50)]:.1f}ms")
        logger.info(f"  P99 Gen Time: {sorted(gen_times)[int(len(gen_times)*0.99)]:.1f}ms")
        logger.info(f"  P50 Offload Time: {sorted(offload_times)[int(len(offload_times)*0.50)]:.1f}ms")
        logger.info(f"  P99 Offload Time: {sorted(offload_times)[int(len(offload_times)*0.99)]:.1f}ms")
        logger.info(f"\nTotal Latency:")
        logger.info(f"  P50: {sorted(total_latencies)[int(len(total_latencies)*0.50)]:.1f}ms")
        logger.info(f"  P99: {sorted(total_latencies)[int(len(total_latencies)*0.99)]:.1f}ms")
        logger.info(f"\nExperiment duration: {experiment_time:.1f}s")
        
        logger.info(f"\nBy Model:")
        for model, model_results in by_model.items():
            avg_load = sum(r["load_time_ms"] for r in model_results) / len(model_results)
            avg_gen = sum(r["gen_time_ms"] for r in model_results) / len(model_results)
            avg_total = sum(r["total_latency_ms"] for r in model_results) / len(model_results)
            logger.info(f"  {model}: {len(model_results)} agents")
            logger.info(f"    Avg load: {avg_load:.1f}ms, gen: {avg_gen:.1f}ms, total: {avg_total:.1f}ms")
        
        return {
            "baseline": "pytorch_naive_swap",
            "stats": {
                "completed": len(successful),
                "failed": len(failed),
                "p50_load_time_ms": sorted(load_times)[int(len(load_times)*0.50)],
                "p99_load_time_ms": sorted(load_times)[int(len(load_times)*0.99)],
                "p50_gen_time_ms": sorted(gen_times)[int(len(gen_times)*0.50)],
                "p99_gen_time_ms": sorted(gen_times)[int(len(gen_times)*0.99)],
                "p50_offload_time_ms": sorted(offload_times)[int(len(offload_times)*0.50)],
                "p99_offload_time_ms": sorted(offload_times)[int(len(offload_times)*0.99)],
                "p50_total_latency_ms": sorted(total_latencies)[int(len(total_latencies)*0.50)],
                "p99_total_latency_ms": sorted(total_latencies)[int(len(total_latencies)*0.99)],
                "experiment_duration_s": experiment_time,
            },
            "results": results,
        }
    else:
        logger.error("All agents failed!")
        return {"baseline": "pytorch_naive_swap", "stats": {"completed": 0, "failed": len(results)}, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True, help="Trace file")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--output", type=str, default="../results/baseline_pytorch_naive.json")
    args = parser.parse_args()
    
    trace_file = Path(args.trace)
    results = asyncio.run(run_baseline(trace_file, args.max_tokens))
    
    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved: {output_file}")


if __name__ == "__main__":
    main()
