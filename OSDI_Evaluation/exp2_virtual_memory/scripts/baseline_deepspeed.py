#!/usr/bin/env python3
"""
DeepSpeed Inference Baseline: Optimized C++ Runtime

Baseline using DeepSpeed's inference engine with kernel injection
for optimized performance.

Expected performance:
- Bandwidth: 15-23 GB/s (specialized C++ with kernel fusions)
- TTFT: ~2-5s for 70B model on L4

This is the "speed of light" baseline that shows what's possible with
purpose-built inference engines.

Usage:
    python baseline_deepspeed.py \
        --model meta-llama/Llama-2-13b-hf \
        --runs 3 \
        --output results/deepspeed.json
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


def get_model_size_bytes(model) -> int:
    """Calculate model size in bytes."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 2  # float16


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> float:
    """Run inference and measure time."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if input_ids.shape[1] > 512:
        input_ids = input_ids[:, :512]
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Measure time
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=min(max_new_tokens, 50),
            do_sample=False,
        )
    
    elapsed = time.time() - start
    return elapsed


def load_model_with_deepspeed(model_id: str):
    """Load model with DeepSpeed optimization if available."""
    try:
        import deepspeed
        logger.info(f"Loading {model_id} with DeepSpeed...")
        
        # Determine dtype
        torch_dtype = torch.float16
        
        # Load model to CPU first
        logger.info("  Step 1/3: Loading model checkpoint to CPU...")
        logger.info("  (Using local cache only - model must be cached)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=False,  # Try to use cache first, download if needed with resume
            resume_download=True,
        )
        
        logger.info("  Step 2/3: Initializing DeepSpeed inference engine...")
        
        # Wrap with DeepSpeed for kernel injection
        model = deepspeed.init_inference(
            model,
            mp_size=1,
            dtype=torch_dtype,
            replace_with_kernel_inject=True,
            enable_cuda_graph=False,
        )
        
        # Extract module if wrapped
        model = model.module if hasattr(model, 'module') else model
        
        logger.info("  Step 3/3: Model ready")
        return model, True
        
    except ImportError:
        logger.warning("DeepSpeed not available, falling back to standard inference")
        logger.info(f"Loading {model_id} with standard PyTorch...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=False,
            resume_download=True,
        )
        return model, False
    except Exception as e:
        logger.warning(f"DeepSpeed initialization failed: {e}")
        logger.info(f"Loading {model_id} with standard PyTorch...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=False,
            resume_download=True,
        )
        return model, False


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed Baseline")
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-hf", help="Model ID")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--output", default="deepspeed.json", help="Output JSON file")
    parser.add_argument("--ttft-enabled", action="store_true", help="Measure TTFT")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("DeepSpeed Inference Baseline")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: cuda:0")
    logger.info(f"Runs: {args.runs}")
    logger.info("")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False, resume_download=True)
    
    # Load model with DeepSpeed
    model, deepspeed_enabled = load_model_with_deepspeed(args.model)
    
    model_size_gb = get_model_size_bytes(model) / (1024**3)
    logger.info(f"✅ Model loaded (DeepSpeed: {deepspeed_enabled})")
    logger.info(f"Model size: {model_size_gb:.1f}GB")
    logger.info("")
    
    # Warmup
    logger.info("Running warmup (2 iterations)...")
    prompt = "The answer to life, the universe, and everything is"
    for _ in range(2):
        run_inference(model, tokenizer, prompt, max_new_tokens=10)
    logger.info("✅ Warmup complete")
    logger.info("")
    
    # Measurement runs
    logger.info(f"Running {args.runs} measurement iterations...")
    results = {
        "model": args.model,
        "baseline": "deepspeed",
        "device": "cuda:0",
        "model_size_gb": model_size_gb,
        "deepspeed_enabled": deepspeed_enabled,
        "runs": [],
        "summary": {}
    }
    
    latencies = []
    
    for i in range(args.runs):
        logger.info(f"  Run {i+1}/{args.runs}...")
        latency = run_inference(model, tokenizer, prompt, max_new_tokens=50)
        latencies.append(latency)
        results["runs"].append({
            "run": i+1,
            "latency_ms": latency * 1000,
        })
    
    # Compute statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    results["summary"] = {
        "avg_latency_ms": avg_latency * 1000,
        "min_latency_ms": min_latency * 1000,
        "max_latency_ms": max_latency * 1000,
        "effective_bandwidth_gbps": model_size_gb / avg_latency,
    }
    
    logger.info("")
    logger.info("Results:")
    logger.info(f"  Avg latency: {avg_latency*1000:.1f}ms")
    logger.info(f"  Min latency: {min_latency*1000:.1f}ms")
    logger.info(f"  Max latency: {max_latency*1000:.1f}ms")
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


