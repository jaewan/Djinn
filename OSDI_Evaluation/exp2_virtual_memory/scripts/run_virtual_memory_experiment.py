#!/usr/bin/env python3
"""
Run virtual memory (ring buffer) experiment.

Measures effective bandwidth when streaming weights through a circular buffer
smaller than model size. Validates skip-end allocation and async pipelining.

Usage:
    python run_virtual_memory_experiment.py \
        --config configs/virt_mem_config.yaml \
        --model meta-llama/Llama-2-70b-hf \
        --runs 5 \
        --output results/ring_buffer_full.json
"""

import argparse
import json
import logging
import time
import gc
import os
from pathlib import Path
from typing import Dict, Any
import yaml

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add Djinn to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from djinn.backend.runtime.ring_buffer import WeightRingBuffer
from djinn.backend.runtime.weight_streamer import WeightStreamer
from djinn.backend.runtime.weight_hooks import install_ring_buffer_hooks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_id: str, dtype: str = "float16"):
    """Load model from HuggingFace."""
    logger.info(f"Loading model: {model_id}")
    
    # Determine torch dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model to CPU first
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cpu"
    )
    
    logger.info(f"✅ Loaded model: {model.config.model_type}, params: {model.num_parameters() / 1e9:.1f}B")
    
    return model, tokenizer


def get_model_size_bytes(model: nn.Module, dtype: str) -> int:
    """Calculate model size in bytes."""
    if dtype == "float16":
        bytes_per_param = 2
    else:
        bytes_per_param = 4
    
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * bytes_per_param


def setup_ring_buffer(model: nn.Module, config: Dict[str, Any], device: torch.device):
    """Setup ring buffer and hooks."""
    rb_config = config["experiment"]["ring_buffer"]
    capacity_bytes = int(rb_config["capacity_gb"] * 1024**3)
    
    logger.info(f"Setting up ring buffer: {rb_config['capacity_gb']}GB")
    
    # Create ring buffer
    ring_buffer = WeightRingBuffer(capacity_bytes, device=device)
    
    # Register model
    model_state_dict = model.state_dict()
    ring_buffer.register_model("default", model_state_dict)
    
    # Create weight streamer
    streamer = WeightStreamer(ring_buffer, device=device)
    streamer.start()
    
    # Install hooks
    hook_mgr = install_ring_buffer_hooks(
        model,
        ring_buffer=ring_buffer,
        model_id="default",
        streamer=streamer
    )
    
    return ring_buffer, streamer, hook_mgr


def run_inference(
    model: nn.Module,
    tokenizer,
    config: Dict[str, Any],
    device: torch.device,
    reference_output=None
) -> Dict[str, Any]:
    """
    Run single inference pass and measure metrics.
    
    Returns:
        Dict with metrics: latency, bandwidth, logit_diff, memory_used
    """
    inf_config = config["experiment"]["inference"]
    
    # Create prompt
    prompt = "The future of AI is" + " " * (inf_config["prompt_length"] - 5)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Measure memory before
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Measure inference time
    start_time = time.perf_counter()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        logits = outputs.logits
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    # Get memory stats
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Calculate logit difference from reference
    logit_diff = 0.0
    if reference_output is not None:
        logit_diff = torch.norm(logits - reference_output).item()
    
    # Calculate effective bandwidth
    # Model size / time = bandwidth
    model_size_bytes = get_model_size_bytes(model, config["experiment"]["model"]["dtype"])
    bandwidth_gbps = (model_size_bytes / elapsed) / 1024**3
    
    return {
        "latency_ms": elapsed * 1000,
        "bandwidth_gbps": bandwidth_gbps,
        "peak_vram_mb": peak_vram,
        "logit_diff": logit_diff,
        "logits": logits.cpu().numpy() if reference_output is None else None,
    }


def run_experiment(config_path: str, model_id: str, num_runs: int, output_path: str):
    """Run full virtual memory experiment."""
    
    # Load config
    config = load_config(config_path)
    
    # Device setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Virtual Memory Ring Buffer Experiment")
    logger.info(f"{'='*70}\n")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, config["experiment"]["model"]["dtype"])
    model_size_gb = get_model_size_bytes(model, config["experiment"]["model"]["dtype"]) / 1024**3
    logger.info(f"Model size: {model_size_gb:.1f}GB\n")
    
    # Setup ring buffer
    if config["experiment"]["ring_buffer"]["enabled"]:
        ring_buffer, streamer, hook_mgr = setup_ring_buffer(model, config, device)
        logger.info(f"✅ Ring buffer and hooks installed\n")
    
    # Run warmup
    logger.info(f"Running {config['experiment']['measurement']['warmup_runs']} warmup iterations...")
    for i in range(config["experiment"]["measurement"]["warmup_runs"]):
        _ = run_inference(model, tokenizer, config, device)
        if config["experiment"]["measurement"]["gc_between_runs"]:
            gc.collect()
            torch.cuda.empty_cache()
    
    logger.info(f"✅ Warmup complete\n")
    
    # Run measurements
    logger.info(f"Running {num_runs} measurement iterations...\n")
    results = []
    reference_output = None
    
    for run_id in range(num_runs):
        # Disable GC during measurement
        if config["experiment"]["measurement"]["disable_gc_during_run"]:
            gc.disable()
        
        try:
            metrics = run_inference(
                model, tokenizer, config, device,
                reference_output=reference_output
            )
            
            # Store reference for logit checking (keep as torch tensor on same device)
            if reference_output is None and metrics["logits"] is not None:
                reference_output = torch.from_numpy(metrics["logits"]).to(device)
            
            results.append({
                "run_id": run_id,
                **metrics
            })
            
            logger.info(
                f"Run {run_id}: "
                f"latency {metrics['latency_ms']:.1f}ms, "
                f"bandwidth {metrics['bandwidth_gbps']:.1f}GB/s, "
                f"memory {metrics['peak_vram_mb']:.0f}MB"
            )
        
        finally:
            if config["experiment"]["measurement"]["disable_gc_during_run"]:
                gc.enable()
        
        # GC between runs
        if config["experiment"]["measurement"]["gc_between_runs"]:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Compute summary statistics
    latencies = [r["latency_ms"] for r in results]
    bandwidths = [r["bandwidth_gbps"] for r in results]
    
    summary = {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": sorted(latencies)[len(latencies)//2],
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "avg_bandwidth_gbps": sum(bandwidths) / len(bandwidths),
        "median_bandwidth_gbps": sorted(bandwidths)[len(bandwidths)//2],
        "min_bandwidth_gbps": min(bandwidths),
        "max_bandwidth_gbps": max(bandwidths),
        "pass": all(bw > 20.0 for bw in bandwidths) if num_runs > 0 else False,
    }
    
    # Create output
    output = {
        "experiment": config["experiment"]["name"],
        "model": model_id,
        "model_size_gb": model_size_gb,
        "config": config,
        "runs": results,
        "summary": summary,
    }
    
    # Save results
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for run in output["runs"]:
            run.pop("logits", None)  # Don't save full logits
        
        json.dump(output, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*70}\n")
    logger.info(f"Average latency: {summary['avg_latency_ms']:.1f}ms")
    logger.info(f"Average bandwidth: {summary['avg_bandwidth_gbps']:.1f}GB/s")
    logger.info(f"Median latency: {summary['median_latency_ms']:.1f}ms")
    logger.info(f"Median bandwidth: {summary['median_bandwidth_gbps']:.1f}GB/s")
    
    if summary["pass"]:
        logger.info(f"\n✅ PASS: Bandwidth sustained > 20 GB/s")
    else:
        logger.info(f"\n❌ FAIL: Bandwidth did not sustain > 20 GB/s")
    
    logger.info(f"\nResults saved to: {output_path}\n")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Run virtual memory experiment")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--model", default="EleutherAI/gpt-j-6B", help="Model ID")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    run_experiment(args.config, args.model, args.runs, args.output)


if __name__ == "__main__":
    main()

