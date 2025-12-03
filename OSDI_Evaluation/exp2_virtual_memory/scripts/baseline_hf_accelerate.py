#!/usr/bin/env python3
"""
HuggingFace Accelerate Baseline: Standard Offloading

Baseline implementation using HuggingFace Accelerate's device_map="auto"
with offload_folder for handling models larger than VRAM.

Expected performance:
- Bandwidth: 8-12 GB/s (synchronous Python dispatch overhead)
- TTFT: ~30s for 70B model on L4

This represents the "naive offloading" approach without async pipelining.

Usage:
    python baseline_hf_accelerate.py \
        --model meta-llama/Llama-2-70b-hf \
        --runs 5 \
        --output results/hf_accelerate.json
"""

import argparse
import json
import logging
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_model_size_bytes(model, dtype: str = None) -> int:
    """Calculate model size in bytes.
    
    Args:
        model: PyTorch model
        dtype: Dtype string ("float16", "float32", etc.) or None to infer from model
    
    Returns:
        Total size in bytes
    """
    # Infer dtype from first parameter if not provided
    if dtype is None:
        first_param = next(model.parameters(), None)
        if first_param is not None:
            if first_param.dtype == torch.float16:
                dtype = "float16"
            elif first_param.dtype == torch.bfloat16:
                dtype = "bfloat16"
            else:
                dtype = "float32"
        else:
            dtype = "float32"
    
    # Map dtype to bytes per parameter
    dtype_to_bytes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }
    bytes_per_param = dtype_to_bytes.get(dtype, 4)
    
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * bytes_per_param


def load_model_with_accelerate(
    model_id: str,
    device_id: int = 0,
    dtype: str = "float16",
    offload_folder: str = "/tmp/hf_offload"
) -> Tuple[Any, Any]:
    """
    Load model with HuggingFace Accelerate's device_map="auto".
    
    This automatically splits model across GPU and CPU with disk offloading.
    """
    logger.info(f"Loading {model_id} with Accelerate device_map='auto'")
    logger.info(f"Offload folder: {offload_folder}")
    
    # Create offload folder
    os.makedirs(offload_folder, exist_ok=True)
    
    # Determine dtype
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with device_map="auto" and offload
    logger.info("Loading model (this may take several minutes for 70B)...")
    start = time.perf_counter()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            offload_folder=offload_folder,
            offload_state_dict=True,  # Also offload state dict
            low_cpu_mem_usage=True,
        )
        load_time = time.perf_counter() - start
        logger.info(f"✅ Model loaded in {load_time:.1f}s")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    num_iterations: int = 1,
    enable_generation: bool = False,
    generation_length: int = 50
) -> Dict[str, Any]:
    """
    Run inference and measure latency.
    
    Args:
        enable_generation: If True, use model.generate() for TTFT measurement.
                          If False, use forward pass only.
        generation_length: Number of tokens to generate
    
    Returns:
        Dict with: latency_ms, bandwidth_gbps, peak_vram_mb, ttft_ms (if generation)
    """
    model.eval()
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Measure inference
    times = []
    ttfts = []
    
    for _ in range(num_iterations):
        if enable_generation:
            # Use generate() for token-by-token generation
            start = time.perf_counter()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=generation_length,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
            
            # Estimate TTFT: first token latency
            num_generated = generated_ids.shape[-1] - input_ids.shape[-1]
            if num_generated > 0:
                ttfts.append((elapsed / num_generated) * 1000)  # Estimate: first token
        else:
            # Use forward pass only
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model(input_ids, use_cache=False)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
    
    # Get memory stats
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Calculate bandwidth (dtype is inferred from model)
    model_size_bytes = get_model_size_bytes(model)
    avg_time_s = (sum(times) / len(times)) / 1000  # Convert back to seconds
    bandwidth_gbps = (model_size_bytes / avg_time_s) / (1024**3)
    
    result = {
        "latency_ms": sum(times) / len(times),
        "bandwidth_gbps": bandwidth_gbps,
        "peak_vram_mb": peak_vram,
    }
    
    # Add TTFT if generation was used
    if ttfts:
        result["ttft_ms"] = sum(ttfts) / len(ttfts)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Accelerate Baseline")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf",
                       help="Model ID from HuggingFace")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--output", type=str, default="results/hf_accelerate.json",
                       help="Output JSON file")
    parser.add_argument("--offload-folder", type=str, default="/tmp/hf_offload",
                       help="Folder for offloading weights")
    parser.add_argument("--ttft-enabled", action="store_true",
                       help="Enable TTFT measurement using generate()")
    parser.add_argument("--generation-length", type=int, default=50,
                       help="Number of tokens to generate")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    device = torch.device(f"cuda:{args.device}")
    
    logger.info("=" * 80)
    logger.info("HuggingFace Accelerate Baseline")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Runs: {args.runs}")
    logger.info("")
    
    # Load model
    try:
        model, tokenizer = load_model_with_accelerate(
            args.model,
            device_id=args.device,
            dtype="float16",
            offload_folder=args.offload_folder
        )
        model_size_gb = get_model_size_bytes(model) / (1024**3)  # Dtype inferred from model
        logger.info(f"Model size: {model_size_gb:.1f}GB\n")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Run warmup
    logger.info("Running warmup (2 iterations)...")
    try:
        _ = run_inference(
            model, tokenizer, "The future of AI is", device,
            num_iterations=2,
            enable_generation=args.ttft_enabled,
            generation_length=args.generation_length
        )
        logger.info("✅ Warmup complete\n")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return False
    
    # Run measurements
    logger.info(f"Running {args.runs} measurement iterations...\n")
    results = []
    
    for run_id in range(args.runs):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            
            metrics = run_inference(
                model, tokenizer,
                "The future of artificial intelligence is",
                device,
                num_iterations=1,
                enable_generation=args.ttft_enabled,
                generation_length=args.generation_length
            )
            
            results.append({
                "run_id": run_id,
                **metrics
            })
            
            logger.info(
                f"Run {run_id}: latency {metrics['latency_ms']:.1f}ms, "
                f"bandwidth {metrics['bandwidth_gbps']:.1f}GB/s, "
                f"memory {metrics['peak_vram_mb']:.0f}MB"
            )
        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            return False
    
    # Compute summary
    latencies = [r["latency_ms"] for r in results]
    bandwidths = [r["bandwidth_gbps"] for r in results]
    ttfts = [r.get("ttft_ms") for r in results if "ttft_ms" in r]
    
    summary = {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": sorted(latencies)[len(latencies)//2],
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "avg_bandwidth_gbps": sum(bandwidths) / len(bandwidths),
        "median_bandwidth_gbps": sorted(bandwidths)[len(bandwidths)//2],
        "min_bandwidth_gbps": min(bandwidths),
        "max_bandwidth_gbps": max(bandwidths),
    }
    
    # Add TTFT statistics if available
    if ttfts:
        summary["avg_ttft_ms"] = sum(ttfts) / len(ttfts)
        summary["median_ttft_ms"] = sorted(ttfts)[len(ttfts)//2]
        summary["min_ttft_ms"] = min(ttfts)
        summary["max_ttft_ms"] = max(ttfts)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "baseline": "hf_accelerate",
        "model": args.model,
        "model_size_gb": model_size_gb,
        "runs": results,
        "summary": summary,
        "timestamp": time.time(),
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Average latency: {summary['avg_latency_ms']:.1f}ms")
    logger.info(f"Average bandwidth: {summary['avg_bandwidth_gbps']:.1f}GB/s")
    if "avg_ttft_ms" in summary:
        logger.info(f"Average TTFT: {summary['avg_ttft_ms']:.1f}ms")
    logger.info(f"Results saved to: {args.output}")
    
    # Cleanup
    if os.path.exists(args.offload_folder):
        logger.info(f"\nCleaning up offload folder: {args.offload_folder}")
        shutil.rmtree(args.offload_folder)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

