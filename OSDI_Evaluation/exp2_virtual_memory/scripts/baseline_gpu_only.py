#!/usr/bin/env python3
"""
GPU-Only Baseline: Measure actual GPU inference (no CPU offloading)

This is a VALID baseline for comparing against Djinn's ring buffer.
- Model must fit in GPU VRAM
- No CPU offloading
- Measures pure GPU compute + weight transfer time
- Fair comparison for Experiment 2

Usage:
    python baseline_gpu_only.py \
        --model meta-llama/Llama-2-13b-hf \
        --runs 3 \
        --output results/baseline_gpu_only.json
"""

import argparse
import json
import logging
import time
import torch
from pathlib import Path
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_model_size_gb(model) -> float:
    """Calculate model size in GB."""
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * 2) / (1024**3)  # float16


def measure_gpu_memory(model) -> tuple:
    """Get current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return allocated, reserved


def run_inference(model, tokenizer, device, max_new_tokens: int = 50) -> tuple:
    """Run inference and return (latency_sec, peak_vram_gb)."""
    prompt = "The answer to life, the universe, and everything is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if input_ids.shape[1] > 512:
        input_ids = input_ids[:, :512]
    
    input_ids = input_ids.to(device)
    
    # Clear cache and measure
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=min(max_new_tokens, 50),
            do_sample=False,
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    return elapsed, peak_memory


def main():
    parser = argparse.ArgumentParser(description="GPU-Only Baseline (Valid Comparison)")
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-hf", help="Model ID")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--output", default="baseline_gpu_only.json", help="Output JSON file")
    parser.add_argument("--ttft-enabled", action="store_true", help="Measure TTFT")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("GPU-Only Baseline (Valid for Experiment 2)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: cuda:0 (GPU-only, no offloading)")
    logger.info(f"Runs: {args.runs}")
    logger.info("")
    
    device = torch.device("cuda:0")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False, resume_download=True)
    
    # Load model directly to GPU
    logger.info("Loading model to GPU (device_map='cuda:0')...")
    logger.info("NOTE: For large models (70B), this will likely OOM - that is expected and validates the problem!")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # GPU-only, no offloading
            local_files_only=False,
            resume_download=True
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"✅ OOM as expected for large models")
            logger.error(f"   Model {args.model} is too large for {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB GPU")
            logger.error(f"   This proves the problem Djinn solves with memory virtualization")
            
            # Save OOM result
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                "model": args.model,
                "baseline": "gpu_only",
                "device": "cuda:0",
                "status": "OOM",
                "error": str(e),
                "interpretation": "Model too large for GPU VRAM - this is why Djinn's ring buffer is needed"
            }
            
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ OOM result saved to: {output_path}")
            return
        else:
            raise
    
    model_size_gb = get_model_size_gb(model)
    allocated, reserved = measure_gpu_memory(model)
    
    logger.info(f"✅ Model loaded on GPU")
    logger.info(f"Model size: {model_size_gb:.1f}GB")
    logger.info(f"GPU memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
    logger.info("")
    
    # Warmup
    logger.info("Running warmup (2 iterations)...")
    for i in range(2):
        latency, peak = run_inference(model, tokenizer, device, max_new_tokens=10)
        logger.info(f"  Warmup {i+1}: {latency*1000:.1f}ms, Peak: {peak:.1f}GB")
    logger.info("✅ Warmup complete")
    logger.info("")
    
    # Measurement runs
    logger.info(f"Running {args.runs} measurement iterations...")
    results = {
        "model": args.model,
        "baseline": "gpu_only",
        "device": "cuda:0",
        "model_size_gb": model_size_gb,
        "gpu_memory_allocated_gb": allocated,
        "gpu_memory_reserved_gb": reserved,
        "runs": [],
        "summary": {}
    }
    
    latencies = []
    peak_memories = []
    
    for i in range(args.runs):
        logger.info(f"  Run {i+1}/{args.runs}...")
        latency, peak = run_inference(model, tokenizer, device, max_new_tokens=50)
        latencies.append(latency)
        peak_memories.append(peak)
        results["runs"].append({
            "run": i+1,
            "latency_ms": latency * 1000,
            "peak_vram_gb": peak,
        })
        logger.info(f"    Latency: {latency*1000:.1f}ms, Peak VRAM: {peak:.1f}GB")
    
    # Compute statistics
    avg_latency = sum(latencies) / len(latencies)
    avg_peak = sum(peak_memories) / len(peak_memories)
    
    results["summary"] = {
        "avg_latency_ms": avg_latency * 1000,
        "min_latency_ms": min(latencies) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "avg_peak_vram_gb": avg_peak,
        "effective_bandwidth_gbps": model_size_gb / avg_latency,
    }
    
    logger.info("")
    logger.info("Results:")
    logger.info(f"  Avg latency: {avg_latency*1000:.1f}ms")
    logger.info(f"  Min latency: {min(latencies)*1000:.1f}ms")
    logger.info(f"  Max latency: {max(latencies)*1000:.1f}ms")
    logger.info(f"  Avg peak VRAM: {avg_peak:.1f}GB")
    logger.info(f"  Effective bandwidth: {model_size_gb/avg_latency:.2f} GB/s")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info(f"✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()


