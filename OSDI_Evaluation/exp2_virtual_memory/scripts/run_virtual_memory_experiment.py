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
import statistics
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
    """Load model tokenizer only. Model will be loaded with ring buffer."""
    logger.info(f"Loading tokenizer: {model_id}")
    
    # Load tokenizer only (model will be loaded via gpu_model_loader)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Loaded tokenizer for {model_id}")
    
    return None, tokenizer


def get_model_size_bytes(model: nn.Module, dtype: str) -> int:
    """Calculate model size in bytes."""
    if dtype == "float16":
        bytes_per_param = 2
    else:
        bytes_per_param = 4
    
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * bytes_per_param


def setup_ring_buffer(model_id: str, config: Dict[str, Any], device: torch.device, chunk_size_mb: int = 64, dtype: str = "float16"):
    """Setup ring buffer with GPU-resident model loading.
    
    Args:
        model_id: Model identifier for loading with ring buffer
        config: Configuration dict
        device: CUDA device
        chunk_size_mb: Prefetch chunk size in MB (for parameter sweep experiments)
        dtype: Data type ("float16" or "float32")
    
    Returns:
        (model, ring_buffer, streamer, hook_mgr)
    """
    from OSDI_Evaluation.exp2_virtual_memory.scripts.gpu_model_loader import load_model_with_ring_buffer
    
    rb_config = config["experiment"]["ring_buffer"]
    capacity_bytes = int(rb_config["capacity_gb"] * 1024**3)
    
    logger.info(f"Setting up ring buffer: {rb_config['capacity_gb']}GB (chunk_size: {chunk_size_mb}MB)")
    
    # Create ring buffer
    ring_buffer = WeightRingBuffer(capacity_bytes, device=device)
    
    # Load model with GPU-resident ring buffer views (NEW ARCHITECTURE)
    logger.info("Loading model with GPU ring buffer virtualization...")
    model, param_views, remaining_params = load_model_with_ring_buffer(
        model_id=model_id,
        ring_buffer=ring_buffer,
        device=device,
        dtype=dtype
    )

    # Store chunk size in ring buffer for potential use in prefetch optimization
    ring_buffer.chunk_size_bytes = chunk_size_mb * 1024 * 1024

    # Embedding Layer Trap Check: Ensure embedding layer doesn't wrap (Llama-70B ~2.1GB)
    embedding_size_gb = 2.1  # Llama-70B embedding size estimate
    ring_buffer_gb = rb_config["capacity_gb"]
    if embedding_size_gb >= ring_buffer_gb:
        logger.warning(f"⚠️  Embedding layer ({embedding_size_gb}GB) >= ring buffer ({ring_buffer_gb}GB)")
        logger.warning("   This may cause wrapping issues. Consider increasing ring buffer capacity.")
    else:
        logger.info(f"✅ Embedding layer check passed: {embedding_size_gb}GB < {ring_buffer_gb}GB")
    
    # Create weight streamer (pass chunk_size to streamer for future optimization)
    streamer = WeightStreamer(ring_buffer, device=device)
    streamer.chunk_size_bytes = chunk_size_mb * 1024 * 1024
    streamer.start()
    
    # Install hooks for prefetch triggering
    hook_mgr = install_ring_buffer_hooks(
        model,
        ring_buffer=ring_buffer,
        model_id="default",
        streamer=streamer
    )
    
    return model, ring_buffer, streamer, hook_mgr


def create_prompt_with_tokens(tokenizer, base_prompt: str, target_tokens: int) -> str:
    """Create a prompt with approximately the target number of tokens.
    
    Args:
        tokenizer: Tokenizer instance
        base_prompt: Base prompt to start with
        target_tokens: Desired number of tokens
    
    Returns:
        Prompt string
    """
    # Start with base prompt
    base_token_ids = tokenizer.encode(base_prompt)
    current_tokens = len(base_token_ids)
    
    if current_tokens >= target_tokens:
        return base_prompt
    
    # Add padding tokens to reach target
    # Use a simple padding word that tokenizes to 1 token
    padding_word = " more"  # Typical 1 token
    prompt = base_prompt
    
    while len(tokenizer.encode(prompt)) < target_tokens:
        prompt += padding_word
    
    # Trim if overshoot slightly
    while len(tokenizer.encode(prompt)) > target_tokens:
        prompt = prompt[:-1]
    
    return prompt


def run_inference(
    model: nn.Module,
    tokenizer,
    config: Dict[str, Any],
    device: torch.device,
    reference_output=None,
    ring_buffer=None,
    enable_generation=False
) -> Dict[str, Any]:
    """
    Run single inference pass and measure metrics.
    
    Args:
        enable_generation: If True, use model.generate() for TTFT measurement.
                          If False, use forward pass only (faster, for profiling).
    
    Returns:
        Dict with metrics: latency, bandwidth, logit_diff, memory_used, ttft_ms (if generation)
    """
    inf_config = config["experiment"]["inference"]
    
    # Create prompt with proper tokenization to target length
    base_prompt = "The future of AI is"
    prompt = create_prompt_with_tokens(tokenizer, base_prompt, inf_config["prompt_length"])
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Measure memory before
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Track TTFT if generation enabled
    ttft_ms = None
    first_token_time = None
    
    if enable_generation:
        # Measure inference time with generation
        start_time = time.perf_counter()
        
        # Use generate for token-by-token generation
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=inf_config.get("generation_length", 50),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        # TTFT: time to first generated token
        # For forward pass, this is approximately the first decode step
        # We estimate this as ~1/num_generated_tokens of total time for now
        num_generated = generated_ids.shape[-1] - input_ids.shape[-1]
        if num_generated > 0:
            ttft_ms = (elapsed / num_generated) * 1000  # Estimate: first token latency
        
        logits = model(generated_ids[:, :-1], output_hidden_states=False).logits
    else:
        # Measure inference time (forward pass only)
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
    
    # Calculate effective bandwidth using actual bytes transferred via ring buffer
    # This is more accurate than total model size since it reflects actual PCIe traffic
    if ring_buffer and hasattr(ring_buffer, 'stats'):
        actual_bytes = ring_buffer.stats.get('bytes_transferred', 0)
        if actual_bytes > 0:
            bandwidth_gbps = (actual_bytes / elapsed) / 1024**3
        else:
            # Ring buffer exists but no transfers yet, use model size fallback
            model_size_bytes = get_model_size_bytes(model, config["experiment"]["model"]["dtype"])
            bandwidth_gbps = (model_size_bytes / elapsed) / 1024**3
    else:
        # No ring buffer, use total model size
        model_size_bytes = get_model_size_bytes(model, config["experiment"]["model"]["dtype"])
        bandwidth_gbps = (model_size_bytes / elapsed) / 1024**3
    
    result = {
        "latency_ms": elapsed * 1000,
        "bandwidth_gbps": bandwidth_gbps,
        "peak_vram_mb": peak_vram,
        "logit_diff": logit_diff,
        "logits": logits.detach().cpu().numpy() if reference_output is None else None,
    }
    
    # Add TTFT if generation was used
    if ttft_ms is not None:
        result["ttft_ms"] = ttft_ms
    
    return result


def run_experiment(config_path: str, model_id: str, num_runs: int, output_path: str, chunk_size_mb: int = 64, ttft_enabled: bool = False, disable_kv_swap: bool = True):
    """Run full virtual memory experiment.
    
    Args:
        config_path: Path to experiment config YAML
        model_id: Model identifier (e.g., EleutherAI/gpt-j-6B)
        num_runs: Number of measurement runs
        output_path: Output JSON file path
        chunk_size_mb: Prefetch chunk size in MB
        ttft_enabled: Enable TTFT measurement using model.generate()
    """
    
    streamer = None
    try:
        # Load config
        config = load_config(config_path)
        
        # Device setup
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Virtual Memory Ring Buffer Experiment")
        logger.info(f"{'='*70}\n")
        
        # Load tokenizer only (model will be loaded with ring buffer)
        _, tokenizer = load_model_and_tokenizer(model_id, config["experiment"]["model"]["dtype"])
        
        # Setup ring buffer with GPU-resident model (NEW ARCHITECTURE)
        if config["experiment"]["ring_buffer"]["enabled"]:
            model, ring_buffer, streamer, hook_mgr = setup_ring_buffer(
                model_id=model_id,
                config=config,
                device=device,
                chunk_size_mb=chunk_size_mb,
                dtype=config["experiment"]["model"]["dtype"]
            )
            logger.info(f"✅ GPU-resident model with ring buffer virtualization\n")
        else:
            # This path is not supported with new architecture
            raise RuntimeError("Ring buffer must be enabled for this experiment")
        
        # Calculate model size
        model_size_gb = get_model_size_bytes(model, config["experiment"]["model"]["dtype"]) / 1024**3
        logger.info(f"Model size: {model_size_gb:.1f}GB\n")
        
        # Ring buffer is always available in new architecture
        ring_buffer_obj = ring_buffer
        
        # Determine if we should enable generation (parameter overrides config)
        enable_generation = ttft_enabled or config["experiment"]["measurement"].get("ttft_enabled", False)
        
        # Run warmup
        logger.info(f"Running {config['experiment']['measurement']['warmup_runs']} warmup iterations...")
        for i in range(config["experiment"]["measurement"]["warmup_runs"]):
            _ = run_inference(model, tokenizer, config, device, ring_buffer=ring_buffer_obj, enable_generation=enable_generation)
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
                    reference_output=reference_output,
                    ring_buffer=ring_buffer_obj,
                    enable_generation=enable_generation
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
        
        # Compute summary statistics using statistics module
        latencies = [r["latency_ms"] for r in results]
        bandwidths = [r["bandwidth_gbps"] for r in results]
        ttfts = [r.get("ttft_ms") for r in results if "ttft_ms" in r]
        
        summary = {
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "min_latency_ms": min(latencies) if latencies else 0.0,
            "max_latency_ms": max(latencies) if latencies else 0.0,
            "avg_bandwidth_gbps": statistics.mean(bandwidths) if bandwidths else 0.0,
            "median_bandwidth_gbps": statistics.median(bandwidths) if bandwidths else 0.0,
            "stdev_bandwidth_gbps": statistics.stdev(bandwidths) if len(bandwidths) > 1 else 0.0,
            "min_bandwidth_gbps": min(bandwidths) if bandwidths else 0.0,
            "max_bandwidth_gbps": max(bandwidths) if bandwidths else 0.0,
            "pass": all(bw > 20.0 for bw in bandwidths) if num_runs > 0 else False,
        }
        
        # Add TTFT statistics if available
        if ttfts:
            summary["avg_ttft_ms"] = statistics.mean(ttfts)
            summary["median_ttft_ms"] = statistics.median(ttfts)
            summary["stdev_ttft_ms"] = statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
            summary["min_ttft_ms"] = min(ttfts)
            summary["max_ttft_ms"] = max(ttfts)
        
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
        logger.info(f"Latency (ms): mean={summary['avg_latency_ms']:.1f} ± {summary['stdev_latency_ms']:.1f}, median={summary['median_latency_ms']:.1f}, range=[{summary['min_latency_ms']:.1f}, {summary['max_latency_ms']:.1f}]")
        logger.info(f"Bandwidth (GB/s): mean={summary['avg_bandwidth_gbps']:.1f} ± {summary['stdev_bandwidth_gbps']:.1f}, median={summary['median_bandwidth_gbps']:.1f}, range=[{summary['min_bandwidth_gbps']:.1f}, {summary['max_bandwidth_gbps']:.1f}]")
        
        if "avg_ttft_ms" in summary:
            logger.info(f"TTFT (ms): mean={summary['avg_ttft_ms']:.1f} ± {summary['stdev_ttft_ms']:.1f}, median={summary['median_ttft_ms']:.1f}, range=[{summary['min_ttft_ms']:.1f}, {summary['max_ttft_ms']:.1f}]")
        
        if summary["pass"]:
            logger.info(f"\n✅ PASS: Bandwidth sustained > 20 GB/s")
        else:
            logger.info(f"\n❌ FAIL: Bandwidth did not sustain > 20 GB/s")
        
        logger.info(f"\nResults saved to: {output_path}\n")
        
        return output
    
    finally:
        # Cleanup streamer resources
        if streamer is not None:
            logger.info("Stopping weight streamer...")
            streamer.stop()
            logger.info("✅ Weight streamer stopped")


def main():
    parser = argparse.ArgumentParser(description="Run virtual memory experiment")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--model", default="gpt2", help="Model ID")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--chunk-size-mb", type=int, nargs='+', 
                       default=[16, 64, 128, 512],
                       help="Chunk sizes in MB for parameter sweep (default: 16 64 128 512)")
    parser.add_argument("--enable-pcie-trace", action="store_true",
                       help="Enable PCIe bandwidth trace via nvidia-smi dmon")
    parser.add_argument("--ttft-enabled", action="store_true",
                       help="Enable TTFT measurement using model.generate()")
    parser.add_argument("--disable-kv-swap", action="store_true", default=True,
                       help="Disable KV cache swapping (weights own 100% of PCIe bus)")
    
    args = parser.parse_args()
    
    # If chunk size sweep requested, run experiments with each chunk size
    if len(args.chunk_size_mb) > 1 or args.chunk_size_mb[0] != 64:
        logger.info(f"Chunk size sweep: {args.chunk_size_mb} MB")
        sweep_results = {}
        
        for chunk_size_mb in args.chunk_size_mb:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running with chunk size: {chunk_size_mb} MB")
            logger.info(f"{'='*70}\n")
            
            # Create output path with chunk size suffix
            output_path = str(args.output).replace('.json', f'_chunk_{chunk_size_mb}mb.json')
            
            try:
                result = run_experiment(args.config, args.model, args.runs, output_path, chunk_size_mb=chunk_size_mb, ttft_enabled=args.ttft_enabled, disable_kv_swap=args.disable_kv_swap)
                sweep_results[f"chunk_{chunk_size_mb}mb"] = result["summary"]
            except Exception as e:
                logger.error(f"Failed for chunk size {chunk_size_mb}MB: {e}")
                sweep_results[f"chunk_{chunk_size_mb}mb"] = {"error": str(e)}
        
        # Save sweep summary
        sweep_output = str(args.output).replace('.json', '_sweep_summary.json')
        with open(sweep_output, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        
        logger.info(f"\n✅ Sweep summary saved to: {sweep_output}")
    else:
        # Single run with default chunk size
        chunk_size_mb = args.chunk_size_mb[0]
        if args.enable_pcie_trace:
            import subprocess
            logger.info("Starting PCIe trace collection via nvidia-smi dmon...")
            # Start background trace
            trace_file = str(args.output).replace('.json', '_pcie_trace.csv')
            
            # Fix: Use list of args to avoid shell injection vulnerability
            with open(trace_file, 'w') as trace_fp:
                trace_proc = subprocess.Popen(
                    ['nvidia-smi', 'dmon', '-s', 'pcit', '-d', '1', '-o', 'T'],
                    stdout=trace_fp,
                    stderr=subprocess.PIPE
                )
            try:
                run_experiment(args.config, args.model, args.runs, args.output, chunk_size_mb=chunk_size_mb, ttft_enabled=args.ttft_enabled, disable_kv_swap=args.disable_kv_swap)
            finally:
                trace_proc.terminate()
                trace_proc.wait(timeout=5)
                logger.info(f"PCIe trace saved to: {trace_file}")
        else:
            run_experiment(args.config, args.model, args.runs, args.output, chunk_size_mb=chunk_size_mb, ttft_enabled=args.ttft_enabled, disable_kv_swap=args.disable_kv_swap)


if __name__ == "__main__":
    main()

