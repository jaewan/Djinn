#!/usr/bin/env python3
"""
Multi-Model Experiment with 7B Models (OSDI Version - SEQUENTIAL).

Runs real inference with Llama-7B, Mistral-7B, and Phi-2 models.
Uses actual model.generate() with ring buffer weight management.

IMPORTANT: Runs agents SEQUENTIALLY (one at a time) for fair comparison
with PyTorch naive baseline. This eliminates lock contention and provides
apples-to-apples latency comparison.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import argparse
import yaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from djinn.backend.runtime.ring_buffer import WeightRingBuffer
from djinn.backend.runtime.weight_hooks import RingBufferHookManager
from djinn.server.model_weight_swap_pool import get_model_swap_pool
from djinn.server.model_switch_coordinator import ModelSwitchCoordinator
from djinn.policies.model_lru import ModelLRUPolicy

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_models_7b(ring_buffer, coordinator, swap_pool, model_configs):
    """Load 7B-class models with ring buffer management and weight hooks."""
    logger.info("=" * 80)
    logger.info("LOADING 7B MODELS WITH ZERO-COPY RING BUFFER")
    logger.info("=" * 80)
    
    models = {}
    
    for model_name, config in model_configs.items():
        logger.info(f"\nLoading {model_name} ({config['model_id']})...")
        load_start = time.perf_counter()
        
        try:
            # Load model to CPU first
            model = AutoModelForCausalLM.from_pretrained(
                config['model_id'],
                torch_dtype=torch.float16,
                device_map='cpu',
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Get state dict
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            model_size = sum(t.numel() * t.element_size() for t in state_dict.values())
            
            logger.info(f"  Model size: {model_size / 1024**3:.2f}GB")
            
            # Check if we need to evict
            if not ring_buffer.can_fit_model(model_size):
                logger.info(f"  Buffer full, evicting to make room...")
                resident = ring_buffer.get_resident_models()
                if resident:
                    victim = resident[0]
                    logger.info(f"  Evicting {victim}...")
                    weights = ring_buffer.get_model_weights_from_buffer(victim)
                    ring_buffer.evict_model(victim)
                    swap_pool.evict_model_to_host(victim, weights)
            
            # Register in ring buffer
            registration = ring_buffer.register_model(model_name, state_dict)
            
            # Install ring buffer hooks for zero-copy weight access
            hook_manager = RingBufferHookManager(
                model=model,
                ring_buffer=ring_buffer,
                model_id=model_name,
                streamer=None,  # No streaming for fully resident models
            )
            hook_manager.install_hooks()
            
            load_time = time.perf_counter() - load_start
            logger.info(f"  ✅ {model_name}: {registration.total_bytes / 1024**3:.2f}GB, loaded in {load_time:.1f}s")
            logger.info(f"     Ring buffer hooks installed (zero-copy mode)")
            logger.info(f"     Resident: {ring_buffer.get_resident_models()}")
            logger.info(f"     Resident bytes: {ring_buffer.get_resident_bytes() / 1024**3:.2f}GB")
            
            models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'hook_manager': hook_manager,
                'size_gb': registration.total_bytes / 1024**3,
                'config': config,
            }
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ All {len(models)} models loaded with ring buffer hooks")
    logger.info(f"   Resident: {ring_buffer.get_resident_models()}")
    logger.info(f"   Swapped: {list(swap_pool.swapped_models.keys())}")
    logger.info(f"   Resident bytes: {ring_buffer.get_resident_bytes() / 1024**3:.2f}GB")
    logger.info("=" * 80)
    
    return models


async def run_agent_with_model_7b(
    agent_id: int,
    model_name: str,
    models: dict,
    ring_buffer: WeightRingBuffer,
    coordinator: ModelSwitchCoordinator,
    prompt: str,
    arrival_time: float,
    start_time: float,
    max_tokens: int = 50,
) -> dict:
    """Run single agent with model switching and REAL inference (ZERO-COPY)."""
    try:
        # Wait for arrival
        wait_time = arrival_time - (time.perf_counter() - start_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        actual_arrival = time.perf_counter()
        
        # Ensure model is resident (triggers switching)
        switch_start = time.perf_counter()
        success = await coordinator.ensure_model_resident(model_name)
        switch_latency_ms = (time.perf_counter() - switch_start) * 1000
        
        if not success:
            return {
                "agent_id": agent_id,
                "model": model_name,
                "success": False,
                "error": "Failed to ensure model resident",
            }
        
        # Run REAL inference with ZERO-COPY ring buffer weights
        model_info = models[model_name]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to('cuda')
        
        # ZERO-COPY: Bind model parameters to ring buffer views
        inference_start = time.perf_counter()
        
        # Get weights from ring buffer (these are GPU tensors, views into ring buffer)
        rb_weights = ring_buffer.get_model_weights_from_buffer(model_name)
        
        # Temporarily bind model parameters to ring buffer views
        # This is O(N) pointer updates, but no data copying
        original_params = {}
        for name, param in model.named_parameters():
            if name in rb_weights:
                original_params[name] = param.data
                param.data = rb_weights[name]  # Zero-copy: point to ring buffer
        
        # Also move buffers (like biases, layernorm params) to GPU
        for name, buffer in model.named_buffers():
            if buffer.device.type == 'cpu':
                buffer.data = buffer.data.to('cuda')
        
        # Run actual inference
        # Parameters now point directly to ring buffer (zero-copy)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        torch.cuda.synchronize()
        inference_latency_ms = (time.perf_counter() - inference_start) * 1000
        
        # Restore original parameters
        for name, original_data in original_params.items():
            model.get_parameter(name).data = original_data
        
        # Move buffers back to CPU
        for name, buffer in model.named_buffers():
            if buffer.device.type == 'cuda':
                buffer.data = buffer.data.to('cpu')
        
        total_latency_ms = (time.perf_counter() - actual_arrival) * 1000
        
        return {
            "agent_id": agent_id,
            "model": model_name,
            "success": True,
            "switch_latency_ms": switch_latency_ms,
            "inference_latency_ms": inference_latency_ms,
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


async def run_experiment(trace_file: Path, config_file: Path, buffer_gb: float):
    """Run multi-model experiment with 7B models."""
    logger.info("=" * 80)
    logger.info("MULTI-MODEL 7B EXPERIMENT (OSDI)")
    logger.info("=" * 80)
    
    # Load config
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Load trace
    with open(trace_file) as f:
        trace_data = json.load(f)
    
    agents = trace_data["agents"]
    logger.info(f"\nExperiment: {len(agents)} agents")
    logger.info(f"Buffer: {buffer_gb}GB")
    
    # Create infrastructure
    ring_buffer = WeightRingBuffer(capacity_bytes=int(buffer_gb * 1024**3), device='cuda:0')
    swap_pool = get_model_swap_pool(pool_size_gb=config['buffer']['swap_pool_gb'])
    swap_pool.clear()
    
    policy = ModelLRUPolicy()
    coordinator = ModelSwitchCoordinator(ring_buffer, swap_pool, policy)
    
    # Load models
    models = await load_models_7b(ring_buffer, coordinator, swap_pool, config['models'])
    
    # Run agents SEQUENTIALLY (for fair comparison with PyTorch baseline)
    logger.info("\nRunning agents SEQUENTIALLY (apples-to-apples comparison)...")
    start_time = time.perf_counter()
    
    results = []
    for i, agent in enumerate(agents):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(agents)}")
        
        result = await run_agent_with_model_7b(
            agent_id=agent["agent_id"],
            model_name=agent["model"],
            models=models,
            ring_buffer=ring_buffer,
            coordinator=coordinator,
            prompt=agent["prompt_text"],
            arrival_time=agent["arrival_time"],
            start_time=start_time,
            max_tokens=config['experiment']['max_tokens'],
        )
        results.append(result)
    
    experiment_time = time.perf_counter() - start_time
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    if successful:
        switch_latencies = [r["switch_latency_ms"] for r in successful]
        inference_latencies = [r["inference_latency_ms"] for r in successful]
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
        logger.info(f"P50 Total Latency: {sorted(total_latencies)[int(len(total_latencies)*0.50)]:.1f}ms")
        logger.info(f"P99 Total Latency: {sorted(total_latencies)[int(len(total_latencies)*0.99)]:.1f}ms")
        logger.info(f"P50 Switch Latency: {sorted(switch_latencies)[int(len(switch_latencies)*0.50)]:.1f}ms")
        logger.info(f"P99 Switch Latency: {sorted(switch_latencies)[int(len(switch_latencies)*0.99)]:.1f}ms")
        logger.info(f"P50 Inference Latency: {sorted(inference_latencies)[int(len(inference_latencies)*0.50)]:.1f}ms")
        logger.info(f"P99 Inference Latency: {sorted(inference_latencies)[int(len(inference_latencies)*0.99)]:.1f}ms")
        logger.info(f"Experiment duration: {experiment_time:.1f}s")
        
        logger.info(f"\nBy Model:")
        for model, model_results in by_model.items():
            avg_latency = sum(r["total_latency_ms"] for r in model_results) / len(model_results)
            logger.info(f"  {model}: {len(model_results)} agents, avg={avg_latency:.1f}ms")
        
        # Coordinator stats
        stats = coordinator.get_stats()
        logger.info(f"\nCoordinator Statistics:")
        logger.info(f"  Switches: {stats['switches_performed']}")
        logger.info(f"  Evictions: {stats['evictions_triggered']}")
        logger.info(f"  Restorations: {stats['restorations_performed']}")
        logger.info(f"  Avg switch latency: {stats['avg_switch_latency_ms']:.1f}ms")
        
        return {
            "stats": {
                "completed": len(successful),
                "failed": len(failed),
                "p50_total_latency_ms": sorted(total_latencies)[int(len(total_latencies)*0.50)],
                "p99_total_latency_ms": sorted(total_latencies)[int(len(total_latencies)*0.99)],
                "p50_switch_latency_ms": sorted(switch_latencies)[int(len(switch_latencies)*0.50)],
                "p99_switch_latency_ms": sorted(switch_latencies)[int(len(switch_latencies)*0.99)],
                "p50_inference_latency_ms": sorted(inference_latencies)[int(len(inference_latencies)*0.50)],
                "p99_inference_latency_ms": sorted(inference_latencies)[int(len(inference_latencies)*0.99)],
                "experiment_duration_s": experiment_time,
                "coordinator_stats": stats,
            },
            "results": results,
        }
    else:
        logger.error("All agents failed!")
        return {"stats": {"completed": 0, "failed": len(results)}, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, default="../configs/multimodel_7b.yaml")
    parser.add_argument("--buffer-gb", type=float, default=25.0)
    parser.add_argument("--output-file", type=str, default="../results/djinn_7b_real.json")
    args = parser.parse_args()
    
    trace_file = Path(args.trace_file)
    config_file = Path(args.config_file)
    results = asyncio.run(run_experiment(trace_file, config_file, args.buffer_gb))
    
    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved: {output_file}")


if __name__ == "__main__":
    main()
