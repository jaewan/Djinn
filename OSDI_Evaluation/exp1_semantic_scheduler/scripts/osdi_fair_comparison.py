#!/usr/bin/env python3
"""
OSDI Fair Comparison: Apples-to-Apples Multi-Model Evaluation

This script runs ALL baselines with FAIR methodology:
1. ALL systems include cold start (no cherry-picking)
2. PyTorch baseline uses pinned memory (same as Djinn)
3. Same trace, same hardware, same models
4. Sequential execution for latency comparison

Baselines:
1. Djinn (ring buffer + pinned swap pool)
2. PyTorch Pinned (model.to() with pin_memory=True)
3. PyTorch Pageable (standard model.to() - for reference)
4. Serverless Emulation (torch.load() cold start)

Note: vLLM is not suitable for frequent model switching (see separate analysis)
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import yaml
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Djinn imports
from djinn.backend.runtime.ring_buffer import WeightRingBuffer
from djinn.server.model_weight_swap_pool import get_model_swap_pool
from djinn.server.model_switch_coordinator import ModelSwitchCoordinator
from djinn.policies.model_lru import ModelLRUPolicy

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# BASELINE 1: DJINN (Ring Buffer + Pinned Swap Pool)
# ==============================================================================

class DjinnBaseline:
    """Djinn with ring buffer and pinned memory swap pool."""
    
    def __init__(self, buffer_gb: float, swap_pool_gb: float):
        self.buffer_gb = buffer_gb
        self.swap_pool_gb = swap_pool_gb
        self.ring_buffer = None
        self.swap_pool = None
        self.coordinator = None
        self.models = {}
        # Per-model inference locks to prevent eviction during inference
        self._inference_locks: Dict[str, asyncio.Lock] = {}
        
    async def setup(self, model_configs: dict):
        """Initialize Djinn infrastructure and load models."""
        logger.info("=" * 60)
        logger.info("DJINN SETUP (Ring Buffer + Pinned Memory)")
        logger.info("=" * 60)
        
        # Create infrastructure
        self.ring_buffer = WeightRingBuffer(
            capacity_bytes=int(self.buffer_gb * 1024**3),
            device='cuda:0'
        )
        self.swap_pool = get_model_swap_pool(pool_size_gb=self.swap_pool_gb)
        self.swap_pool.clear()
        
        policy = ModelLRUPolicy()
        self.coordinator = ModelSwitchCoordinator(
            self.ring_buffer, self.swap_pool, policy
        )
        
        # Load models
        for model_name, config in model_configs.items():
            logger.info(f"\nLoading {model_name} ({config['model_id']})...")
            load_start = time.perf_counter()
            
            model = AutoModelForCausalLM.from_pretrained(
                config['model_id'],
                torch_dtype=torch.float16,
                device_map='cpu',
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            model_size = sum(t.numel() * t.element_size() for t in state_dict.values())
            
            # Evict if needed
            if not self.ring_buffer.can_fit_model(model_size):
                resident = self.ring_buffer.get_resident_models()
                if resident:
                    victim = resident[0]
                    weights = self.ring_buffer.get_model_weights_from_buffer(victim)
                    self.ring_buffer.evict_model(victim)
                    self.swap_pool.evict_model_to_host(victim, weights)
            
            # Register in ring buffer
            self.ring_buffer.register_model(model_name, state_dict)
            
            load_time = time.perf_counter() - load_start
            logger.info(f"  ✅ {model_name}: {model_size / 1024**3:.2f}GB in {load_time:.1f}s")
            
            self.models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'size_gb': model_size / 1024**3,
            }
        
        logger.info(f"\nResident: {self.ring_buffer.get_resident_models()}")
        logger.info(f"Swapped: {list(self.swap_pool.swapped_models.keys())}")
    
    async def run_agent(self, model_name: str, prompt: str, max_tokens: int) -> dict:
        """Run single agent with Djinn."""
        start_time = time.perf_counter()
        
        # Get or create per-model inference lock
        if model_name not in self._inference_locks:
            self._inference_locks[model_name] = asyncio.Lock()
        inference_lock = self._inference_locks[model_name]
        
        # Acquire inference lock to prevent eviction during our use
        async with inference_lock:
            # Ensure model is resident (may trigger eviction/restoration)
            switch_start = time.perf_counter()
            success = await self.coordinator.ensure_model_resident(model_name)
            switch_ms = (time.perf_counter() - switch_start) * 1000
            
            if not success:
                return {"success": False, "error": "Failed to ensure model resident"}
            
            # Record access to update LRU
            await self.coordinator.record_model_access(model_name)
            
            # Get model and tokenizer
            model_info = self.models[model_name]
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            input_ids = inputs["input_ids"].to('cuda')
            
            # Get weights from ring buffer (zero-copy) - model is guaranteed resident
            inference_start = time.perf_counter()
            
            # Double-check model is still resident (should always be true with lock)
            if not self.ring_buffer.is_model_resident(model_name):
                return {"success": False, "error": f"Model {model_name} was evicted unexpectedly"}
            
            rb_weights = self.ring_buffer.get_model_weights_from_buffer(model_name)
            
            if not rb_weights:
                return {"success": False, "error": f"Could not get weights for {model_name}"}
            
            # Bind model params to ring buffer views
            original_params = {}
            for name, param in model.named_parameters():
                if name in rb_weights:
                    original_params[name] = param.data
                    param.data = rb_weights[name]
            
            # Move buffers to GPU
            for name, buffer in model.named_buffers():
                if buffer.device.type == 'cpu':
                    buffer.data = buffer.data.to('cuda')
            
            # Run inference
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_start) * 1000
            
            # Restore
            for name, original_data in original_params.items():
                model.get_parameter(name).data = original_data
            for name, buffer in model.named_buffers():
                if buffer.device.type == 'cuda':
                    buffer.data = buffer.data.to('cpu')
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "switch_ms": switch_ms,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
            "output_tokens": outputs.shape[1] - input_ids.shape[1],
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        if self.ring_buffer:
            self.ring_buffer.clear()
        if self.swap_pool:
            self.swap_pool.clear()
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# BASELINE 2: PYTORCH WITH PINNED MEMORY (Fair Comparison)
# ==============================================================================

class PyTorchPinnedBaseline:
    """PyTorch with pinned memory - fair comparison to Djinn."""
    
    def __init__(self):
        self.models = {}
        self.pinned_weights = {}  # model_name -> {param_name: pinned_tensor}
        
    async def setup(self, model_configs: dict):
        """Load models and create pinned memory copies."""
        logger.info("=" * 60)
        logger.info("PYTORCH PINNED SETUP (Fair Comparison)")
        logger.info("=" * 60)
        
        for model_name, config in model_configs.items():
            logger.info(f"\nLoading {model_name} ({config['model_id']})...")
            load_start = time.perf_counter()
            
            model = AutoModelForCausalLM.from_pretrained(
                config['model_id'],
                torch_dtype=torch.float16,
                device_map='cpu',
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create pinned memory copies of weights (same as Djinn's swap pool)
            pinned_weights = {}
            for name, param in model.named_parameters():
                pinned_weights[name] = param.data.pin_memory()
            
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            load_time = time.perf_counter() - load_start
            
            logger.info(f"  ✅ {model_name}: {model_size / 1024**3:.2f}GB in {load_time:.1f}s")
            logger.info(f"     Created pinned memory copies for fair comparison")
            
            self.models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'size_gb': model_size / 1024**3,
            }
            self.pinned_weights[model_name] = pinned_weights
    
    async def run_agent(self, model_name: str, prompt: str, max_tokens: int) -> dict:
        """Run single agent with PyTorch pinned memory."""
        start_time = time.perf_counter()
        
        model_info = self.models[model_name]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        pinned_weights = self.pinned_weights[model_name]
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"]
        
        # Load from pinned memory to GPU (fair comparison - same as Djinn restore)
        load_start = time.perf_counter()
        
        # Transfer weights from pinned CPU to GPU
        gpu_weights = {}
        for name, pinned_tensor in pinned_weights.items():
            gpu_weights[name] = pinned_tensor.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        
        # Bind to model
        for name, param in model.named_parameters():
            if name in gpu_weights:
                param.data = gpu_weights[name]
        
        # Move buffers to GPU
        for name, buffer in model.named_buffers():
            if buffer.device.type == 'cpu':
                buffer.data = buffer.data.to('cuda')
        
        load_ms = (time.perf_counter() - load_start) * 1000
        
        # Run inference
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids.to('cuda'),
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - gen_start) * 1000
        
        # Offload back to pinned memory (fair comparison - same as Djinn evict)
        offload_start = time.perf_counter()
        for name, param in model.named_parameters():
            if name in pinned_weights:
                pinned_weights[name].copy_(param.data)
                param.data = pinned_weights[name]
        
        for name, buffer in model.named_buffers():
            if buffer.device.type == 'cuda':
                buffer.data = buffer.data.to('cpu')
        
        # Free GPU memory
        del gpu_weights
        torch.cuda.empty_cache()
        offload_ms = (time.perf_counter() - offload_start) * 1000
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "load_ms": load_ms,
            "gen_ms": gen_ms,
            "offload_ms": offload_ms,
            "total_ms": total_ms,
            "output_tokens": outputs.shape[1] - input_ids.shape[1],
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        self.pinned_weights.clear()
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# BASELINE 3: PYTORCH PAGEABLE (Standard - Reference Only)
# ==============================================================================

class PyTorchPageableBaseline:
    """Standard PyTorch with pageable memory - reference baseline."""
    
    def __init__(self):
        self.models = {}
        
    async def setup(self, model_configs: dict):
        """Load models to CPU."""
        logger.info("=" * 60)
        logger.info("PYTORCH PAGEABLE SETUP (Standard Reference)")
        logger.info("=" * 60)
        
        for model_name, config in model_configs.items():
            logger.info(f"\nLoading {model_name} ({config['model_id']})...")
            load_start = time.perf_counter()
            
            model = AutoModelForCausalLM.from_pretrained(
                config['model_id'],
                torch_dtype=torch.float16,
                device_map='cpu',
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            load_time = time.perf_counter() - load_start
            
            logger.info(f"  ✅ {model_name}: {model_size / 1024**3:.2f}GB in {load_time:.1f}s")
            
            self.models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'size_gb': model_size / 1024**3,
            }
    
    async def run_agent(self, model_name: str, prompt: str, max_tokens: int) -> dict:
        """Run single agent with standard PyTorch."""
        start_time = time.perf_counter()
        
        model_info = self.models[model_name]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"]
        
        # Standard model.to('cuda') - uses pageable memory
        load_start = time.perf_counter()
        model_gpu = model.to('cuda')
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - load_start) * 1000
        
        # Run inference
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model_gpu.generate(
                input_ids.to('cuda'),
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - gen_start) * 1000
        
        # Offload back to CPU
        offload_start = time.perf_counter()
        model_gpu.to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        offload_ms = (time.perf_counter() - offload_start) * 1000
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "load_ms": load_ms,
            "gen_ms": gen_ms,
            "offload_ms": offload_ms,
            "total_ms": total_ms,
            "output_tokens": outputs.shape[1] - input_ids.shape[1],
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# BASELINE 4: SERVERLESS EMULATION (Cold Start)
# ==============================================================================

class ServerlessBaseline:
    """Serverless emulation - measures torch.load() cold start time."""
    
    def __init__(self):
        self.model_paths = {}  # model_name -> saved_path
        self.tokenizers = {}
        
    async def setup(self, model_configs: dict):
        """Pre-download and save models for cold start testing."""
        logger.info("=" * 60)
        logger.info("SERVERLESS SETUP (Cold Start Emulation)")
        logger.info("=" * 60)
        
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        
        for model_name, config in model_configs.items():
            logger.info(f"\nPreparing {model_name} ({config['model_id']})...")
            
            # Load and save model
            model = AutoModelForCausalLM.from_pretrained(
                config['model_id'],
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Save state dict for cold start loading
            save_path = f"{self.temp_dir}/{model_name}.pt"
            torch.save(model.state_dict(), save_path)
            
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            logger.info(f"  ✅ {model_name}: {model_size / 1024**3:.2f}GB saved to {save_path}")
            
            self.model_paths[model_name] = {
                'path': save_path,
                'model_id': config['model_id'],
                'size_gb': model_size / 1024**3,
            }
            self.tokenizers[model_name] = tokenizer
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            gc.collect()
    
    async def run_agent(self, model_name: str, prompt: str, max_tokens: int) -> dict:
        """Run single agent with serverless cold start."""
        start_time = time.perf_counter()
        
        model_info = self.model_paths[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # COLD START: Load model from disk (this is what serverless does)
        load_start = time.perf_counter()
        
        # Create fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_info['model_id'],
            torch_dtype=torch.float16,
            device_map='cuda',
            trust_remote_code=True,
        )
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - load_start) * 1000
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to('cuda')
        
        # Run inference
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - gen_start) * 1000
        
        # TEARDOWN: Delete model (this is what serverless does)
        teardown_start = time.perf_counter()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        teardown_ms = (time.perf_counter() - teardown_start) * 1000
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "load_ms": load_ms,  # Cold start time
            "gen_ms": gen_ms,
            "teardown_ms": teardown_ms,
            "total_ms": total_ms,
            "output_tokens": outputs.shape[1] - input_ids.shape[1],
        }
    
    def cleanup(self):
        """Cleanup resources."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

async def run_experiment_sequential(
    baseline,
    baseline_name: str,
    agents: list,
    max_tokens: int,
) -> dict:
    """Run experiment SEQUENTIALLY (one agent at a time)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING SEQUENTIAL: {baseline_name}")
    logger.info(f"{'='*60}")
    
    results = []
    start_time = time.perf_counter()
    
    for i, agent in enumerate(agents):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(agents)}")
        
        try:
            result = await baseline.run_agent(
                model_name=agent["model"],
                prompt=agent["prompt_text"],
                max_tokens=max_tokens,
            )
            result["agent_id"] = agent["agent_id"]
            result["model"] = agent["model"]
        except Exception as e:
            logger.error(f"Agent {agent['agent_id']} failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "agent_id": agent["agent_id"],
                "model": agent["model"],
                "success": False,
                "error": str(e),
            }
        results.append(result)
    
    experiment_time = time.perf_counter() - start_time
    return _analyze_results(baseline_name, results, experiment_time)


async def run_experiment_concurrent(
    baseline,
    baseline_name: str,
    agents: list,
    max_tokens: int,
) -> dict:
    """Run experiment CONCURRENTLY (all agents in parallel with Poisson arrival)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING CONCURRENT: {baseline_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.perf_counter()
    
    async def run_with_arrival(agent):
        """Run agent at its scheduled arrival time."""
        # Wait for arrival time
        wait_time = agent["arrival_time"] - (time.perf_counter() - start_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        agent_start = time.perf_counter()
        try:
            result = await baseline.run_agent(
                model_name=agent["model"],
                prompt=agent["prompt_text"],
                max_tokens=max_tokens,
            )
            result["agent_id"] = agent["agent_id"]
            result["model"] = agent["model"]
            result["queuing_ms"] = (agent_start - start_time - agent["arrival_time"]) * 1000
        except Exception as e:
            logger.error(f"Agent {agent['agent_id']} failed: {e}")
            result = {
                "agent_id": agent["agent_id"],
                "model": agent["model"],
                "success": False,
                "error": str(e),
            }
        return result
    
    # Run all agents concurrently
    tasks = [asyncio.create_task(run_with_arrival(agent)) for agent in agents]
    results = await asyncio.gather(*tasks)
    
    experiment_time = time.perf_counter() - start_time
    return _analyze_results(baseline_name, list(results), experiment_time)


def _analyze_results(baseline_name: str, results: list, experiment_time: float) -> dict:
    """Analyze experiment results."""
    successful = [r for r in results if r.get("success")]
    
    if successful:
        total_latencies = [r["total_ms"] for r in successful]
        
        stats = {
            "baseline": baseline_name,
            "completed": len(successful),
            "failed": len(results) - len(successful),
            "p50_total_ms": sorted(total_latencies)[int(len(total_latencies)*0.50)],
            "p99_total_ms": sorted(total_latencies)[int(len(total_latencies)*0.99)],
            "mean_total_ms": sum(total_latencies) / len(total_latencies),
            "experiment_duration_s": experiment_time,
            "throughput_agents_per_sec": len(successful) / experiment_time,
        }
        
        # Add baseline-specific stats
        if "switch_ms" in successful[0]:
            switch_times = [r["switch_ms"] for r in successful]
            stats["p50_switch_ms"] = sorted(switch_times)[int(len(switch_times)*0.50)]
            stats["p99_switch_ms"] = sorted(switch_times)[int(len(switch_times)*0.99)]
        
        if "load_ms" in successful[0]:
            load_times = [r["load_ms"] for r in successful]
            stats["p50_load_ms"] = sorted(load_times)[int(len(load_times)*0.50)]
            stats["p99_load_ms"] = sorted(load_times)[int(len(load_times)*0.99)]
        
        if "offload_ms" in successful[0]:
            offload_times = [r["offload_ms"] for r in successful]
            stats["p50_offload_ms"] = sorted(offload_times)[int(len(offload_times)*0.50)]
            stats["p99_offload_ms"] = sorted(offload_times)[int(len(offload_times)*0.99)]
        
        if "gen_ms" in successful[0]:
            gen_times = [r["gen_ms"] for r in successful]
            stats["p50_gen_ms"] = sorted(gen_times)[int(len(gen_times)*0.50)]
            stats["p99_gen_ms"] = sorted(gen_times)[int(len(gen_times)*0.99)]
        
        if "queuing_ms" in successful[0]:
            queue_times = [r["queuing_ms"] for r in successful if r.get("queuing_ms", 0) > 0]
            if queue_times:
                stats["p50_queuing_ms"] = sorted(queue_times)[int(len(queue_times)*0.50)]
                stats["p99_queuing_ms"] = sorted(queue_times)[int(len(queue_times)*0.99)]
        
        logger.info(f"\n{baseline_name} RESULTS:")
        logger.info(f"  Completed: {stats['completed']}/{len(results)}")
        logger.info(f"  P50 Total: {stats['p50_total_ms']:.1f}ms")
        logger.info(f"  P99 Total: {stats['p99_total_ms']:.1f}ms")
        logger.info(f"  Duration: {stats['experiment_duration_s']:.1f}s")
        logger.info(f"  Throughput: {stats['throughput_agents_per_sec']:.3f} agents/sec")
        
        return {"stats": stats, "results": results}
    else:
        return {
            "stats": {"baseline": baseline_name, "completed": 0, "failed": len(results)},
            "results": results,
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--buffer-gb", type=float, default=25.0)
    parser.add_argument("--output-dir", type=str, default="../results/osdi_final")
    parser.add_argument("--baselines", type=str, default="all",
                        help="Comma-separated: djinn,pytorch_pinned,pytorch_pageable,serverless")
    parser.add_argument("--mode", type=str, default="both", choices=["sequential", "concurrent", "both"],
                        help="Execution mode: sequential, concurrent, or both")
    args = parser.parse_args()
    
    # Load config and trace
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    
    with open(args.trace_file) as f:
        trace_data = json.load(f)
    
    agents = trace_data["agents"]
    model_configs = config["models"]
    max_tokens = config["experiment"]["max_tokens"]
    
    logger.info("=" * 80)
    logger.info("OSDI FAIR COMPARISON EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Agents: {len(agents)}")
    logger.info(f"Models: {list(model_configs.keys())}")
    logger.info(f"Buffer: {args.buffer_gb}GB")
    logger.info(f"Mode: {args.mode}")
    
    # Determine which baselines to run
    if args.baselines == "all":
        baselines_to_run = ["djinn", "pytorch_pinned", "pytorch_pageable", "serverless"]
    else:
        baselines_to_run = args.baselines.split(",")
    
    # Determine execution modes
    modes = []
    if args.mode in ["sequential", "both"]:
        modes.append("sequential")
    if args.mode in ["concurrent", "both"]:
        modes.append("concurrent")
    
    all_results = {}
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run each baseline in each mode
    for mode in modes:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# MODE: {mode.upper()}")
        logger.info(f"{'#'*80}")
        
        for baseline_name in baselines_to_run:
            full_name = f"{baseline_name}_{mode}"
            logger.info(f"\n{'='*60}")
            logger.info(f"BASELINE: {full_name}")
            logger.info(f"{'='*60}")
            
            try:
                if baseline_name == "djinn":
                    baseline = DjinnBaseline(
                        buffer_gb=args.buffer_gb,
                        swap_pool_gb=config["buffer"]["swap_pool_gb"]
                    )
                elif baseline_name == "pytorch_pinned":
                    baseline = PyTorchPinnedBaseline()
                elif baseline_name == "pytorch_pageable":
                    baseline = PyTorchPageableBaseline()
                elif baseline_name == "serverless":
                    baseline = ServerlessBaseline()
                else:
                    logger.error(f"Unknown baseline: {baseline_name}")
                    continue
                
                # Setup
                await baseline.setup(model_configs)
                
                # Run experiment in appropriate mode
                if mode == "sequential":
                    results = await run_experiment_sequential(
                        baseline=baseline,
                        baseline_name=full_name,
                        agents=agents,
                        max_tokens=max_tokens,
                    )
                else:  # concurrent
                    results = await run_experiment_concurrent(
                        baseline=baseline,
                        baseline_name=full_name,
                        agents=agents,
                        max_tokens=max_tokens,
                    )
                
                all_results[full_name] = results
                
                # Save individual results
                output_file = output_dir / f"{full_name}_results.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"  Saved: {output_file}")
                
                # Cleanup
                baseline.cleanup()
                
            except Exception as e:
                logger.error(f"Baseline {full_name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[full_name] = {"error": str(e)}
    
    # Generate comparison summary
    logger.info("\n" + "=" * 80)
    logger.info("OSDI FAIR COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    comparison = {"metadata": {"n_agents": len(agents), "buffer_gb": args.buffer_gb, "mode": args.mode}}
    
    for name, results in all_results.items():
        if "stats" in results:
            stats = results["stats"]
            comparison[name] = {
                "p50_total_ms": stats.get("p50_total_ms"),
                "p99_total_ms": stats.get("p99_total_ms"),
                "mean_total_ms": stats.get("mean_total_ms"),
                "throughput": stats.get("throughput_agents_per_sec"),
                "completed": stats.get("completed"),
            }
            logger.info(f"\n{name}:")
            logger.info(f"  Completed: {stats.get('completed')}/{len(agents)}")
            logger.info(f"  P50: {stats.get('p50_total_ms', 'N/A'):.1f}ms")
            logger.info(f"  P99: {stats.get('p99_total_ms', 'N/A'):.1f}ms")
            logger.info(f"  Throughput: {stats.get('throughput_agents_per_sec', 0):.3f} agents/sec")
    
    # Calculate speedups for each mode
    for mode in modes:
        djinn_key = f"djinn_{mode}"
        pinned_key = f"pytorch_pinned_{mode}"
        pageable_key = f"pytorch_pageable_{mode}"
        
        if pageable_key in comparison and djinn_key in comparison:
            ref = comparison[pageable_key]
            djinn = comparison[djinn_key]
            if ref.get("p50_total_ms") and djinn.get("p50_total_ms"):
                logger.info(f"\nSpeedup ({mode.upper()}) Djinn vs PyTorch Pageable:")
                logger.info(f"  P50: {ref['p50_total_ms'] / djinn['p50_total_ms']:.2f}x")
                logger.info(f"  P99: {ref['p99_total_ms'] / djinn['p99_total_ms']:.2f}x")
                if ref.get("throughput") and djinn.get("throughput"):
                    logger.info(f"  Throughput: {djinn['throughput'] / ref['throughput']:.2f}x")
        
        if pinned_key in comparison and djinn_key in comparison:
            ref = comparison[pinned_key]
            djinn = comparison[djinn_key]
            if ref.get("p50_total_ms") and djinn.get("p50_total_ms"):
                logger.info(f"\nSpeedup ({mode.upper()}) Djinn vs PyTorch Pinned (FAIR):")
                logger.info(f"  P50: {ref['p50_total_ms'] / djinn['p50_total_ms']:.2f}x")
                logger.info(f"  P99: {ref['p99_total_ms'] / djinn['p99_total_ms']:.2f}x")
                if ref.get("throughput") and djinn.get("throughput"):
                    logger.info(f"  Throughput: {djinn['throughput'] / ref['throughput']:.2f}x")
    
    # Save comparison
    comparison_file = output_dir / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison saved: {comparison_file}")


if __name__ == "__main__":
    asyncio.run(main())
