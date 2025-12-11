#!/usr/bin/env python3
"""
vLLM Sequential Load Baseline (OSDI Comparison).

Measures actual vLLM model loading/unloading for multi-model scenarios.

Approach:
- For each model switch: unload current engine, load new engine
- Measure cold start time (engine initialization)
- Run inference
- Repeat for each model switch

Expected behavior:
- High latency on model switches (~5-30s per load)
- Fast inference once loaded (vLLM's strength)
- Shows the cost of not having persistent model state
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import argparse
import os

# Set environment variables before importing vLLM
os.environ["VLLM_USE_V1"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_vllm_engine(model_id: str, gpu_memory_utilization: float = 0.5):
    """Load vLLM engine (cold start)."""
    logger.info(f"Loading vLLM engine for {model_id}...")
    load_start = time.perf_counter()
    
    try:
        # Use lower memory utilization to handle model switching
        # vLLM is not designed for frequent model changes
        engine_args = AsyncEngineArgs(
            model=model_id,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=512,  # Shorter context for stability
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs for stability
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        load_time = time.perf_counter() - load_start
        logger.info(f"  ✅ Engine loaded in {load_time:.1f}s")
        
        return engine, load_time
        
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        raise


async def unload_vllm_engine(engine):
    """Unload vLLM engine - aggressive cleanup for model switching."""
    if engine:
        try:
            # Shutdown engine properly
            if hasattr(engine, 'shutdown'):
                await engine.shutdown()
        except Exception:
            pass
        
        del engine
        
        # Aggressive GPU memory cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        await asyncio.sleep(1.0)  # Give more time for cleanup


async def vllm_sequential_agent(
    agent_id: int,
    model_name: str,
    model_id: str,
    prompt: str,
    current_engine: dict,
    max_tokens: int = 50,
) -> dict:
    """
    Run agent with vLLM sequential loading.
    
    If model changes, unload current engine and load new one.
    """
    try:
        load_time_ms = 0
        
        # Check if we need to switch models
        if current_engine['model'] != model_name:
            logger.info(f"  Agent {agent_id}: Switching from {current_engine['model']} to {model_name}")
            
            # Unload current engine
            await unload_vllm_engine(current_engine['engine'])
            
            # Load new engine (measure cold start)
            load_start = time.perf_counter()
            engine, _ = await load_vllm_engine(model_id)
            load_time_ms = (time.perf_counter() - load_start) * 1000
            
            # Update current engine
            current_engine['model'] = model_name
            current_engine['engine'] = engine
        
        # Run inference
        gen_start = time.perf_counter()
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        
        request_id = f"agent_{agent_id}"
        
        outputs = []
        async for output in current_engine['engine'].generate(prompt, sampling_params, request_id):
            outputs.append(output)
        
        gen_time_ms = (time.perf_counter() - gen_start) * 1000
        
        total_latency_ms = load_time_ms + gen_time_ms
        
        return {
            "agent_id": agent_id,
            "model": model_name,
            "success": True,
            "load_time_ms": load_time_ms,
            "gen_time_ms": gen_time_ms,
            "total_latency_ms": total_latency_ms,
            "switched": load_time_ms > 0,
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
    """Run vLLM sequential load baseline."""
    logger.info("=" * 80)
    logger.info("vLLM SEQUENTIAL LOAD BASELINE")
    logger.info("=" * 80)
    
    # Load trace
    with open(trace_file) as f:
        trace_data = json.load(f)
    
    agents = trace_data["agents"]
    model_configs = trace_data.get("model_configs", {})
    
    logger.info(f"\nExperiment: {len(agents)} agents")
    
    # Extract model IDs
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
    
    # Track current engine
    current_engine = {
        'model': None,
        'engine': None,
    }
    
    # Run agents sequentially
    logger.info("\nRunning agents (sequential)...")
    start_time = time.perf_counter()
    
    results = []
    for i, agent in enumerate(agents):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(agents)}")
        
        model_id = model_ids.get(agent["model"])
        if not model_id:
            logger.error(f"Unknown model: {agent['model']}")
            continue
        
        # Truncate prompt to fit model's max length
        prompt = agent["prompt_text"][:3000]  # Rough truncation, tokenizer will handle exact
        
        result = await vllm_sequential_agent(
            agent_id=agent["agent_id"],
            model_name=agent["model"],
            model_id=model_id,
            prompt=prompt,
            current_engine=current_engine,
            max_tokens=max_tokens,
        )
        results.append(result)
    
    # Cleanup
    await unload_vllm_engine(current_engine['engine'])
    
    experiment_time = time.perf_counter() - start_time
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    switched = [r for r in successful if r.get("switched")]
    
    if successful:
        load_times = [r["load_time_ms"] for r in successful if r["load_time_ms"] > 0]
        gen_times = [r["gen_time_ms"] for r in successful]
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
        logger.info(f"Model switches: {len(switched)}")
        
        if load_times:
            logger.info(f"\nCold Start (Model Loading):")
            logger.info(f"  Count: {len(load_times)}")
            logger.info(f"  P50: {sorted(load_times)[int(len(load_times)*0.50)]:.1f}ms")
            logger.info(f"  P99: {sorted(load_times)[int(len(load_times)*0.99)]:.1f}ms")
            logger.info(f"  Mean: {sum(load_times)/len(load_times):.1f}ms")
        
        logger.info(f"\nGeneration Time:")
        logger.info(f"  P50: {sorted(gen_times)[int(len(gen_times)*0.50)]:.1f}ms")
        logger.info(f"  P99: {sorted(gen_times)[int(len(gen_times)*0.99)]:.1f}ms")
        
        logger.info(f"\nTotal Latency:")
        logger.info(f"  P50: {sorted(total_latencies)[int(len(total_latencies)*0.50)]:.1f}ms")
        logger.info(f"  P99: {sorted(total_latencies)[int(len(total_latencies)*0.99)]:.1f}ms")
        
        logger.info(f"\nExperiment duration: {experiment_time:.1f}s")
        
        logger.info(f"\nBy Model:")
        for model, model_results in by_model.items():
            switches = sum(1 for r in model_results if r.get("switched"))
            avg_total = sum(r["total_latency_ms"] for r in model_results) / len(model_results)
            logger.info(f"  {model}: {len(model_results)} agents, {switches} switches, avg={avg_total:.1f}ms")
        
        return {
            "baseline": "vllm_sequential",
            "stats": {
                "completed": len(successful),
                "failed": len(failed),
                "model_switches": len(switched),
                "p50_load_time_ms": sorted(load_times)[int(len(load_times)*0.50)] if load_times else 0,
                "p99_load_time_ms": sorted(load_times)[int(len(load_times)*0.99)] if load_times else 0,
                "mean_load_time_ms": sum(load_times)/len(load_times) if load_times else 0,
                "p50_gen_time_ms": sorted(gen_times)[int(len(gen_times)*0.50)],
                "p99_gen_time_ms": sorted(gen_times)[int(len(gen_times)*0.99)],
                "p50_total_latency_ms": sorted(total_latencies)[int(len(total_latencies)*0.50)],
                "p99_total_latency_ms": sorted(total_latencies)[int(len(total_latencies)*0.99)],
                "experiment_duration_s": experiment_time,
            },
            "results": results,
        }
    else:
        logger.error("All agents failed!")
        return {"baseline": "vllm_sequential", "stats": {"completed": 0, "failed": len(results)}, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True, help="Trace file")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--output", type=str, default="../results/baseline_vllm_sequential.json")
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
