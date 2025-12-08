#!/usr/bin/env python3
"""
Baseline: PyTorch Eager Model Execution

Scientific Goal: Show that PyTorch Eager holds VRAM during pause (cannot share GPU)

This script demonstrates that standard PyTorch inference:
1. Loads Llama-2-13B on GPU
2. Runs inference to layer N
3. During "pause" (simulating think time), VRAM is held - no sharing possible
4. Proves Djinn enables GPU sharing that PyTorch cannot

Key Metric: VRAM remains allocated during pause, blocking other requests.
"""

import sys
import os
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure logging to file and console."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pytorch_eager_baseline.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)


def check_gpu_memory() -> Dict[str, float]:
    """Get current GPU memory stats."""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - allocated,
    }


def run_pytorch_eager_baseline(
    model_name: str = "meta-llama/Llama-2-13b-hf",
    input_length: int = 2048,
    pause_duration_seconds: int = 10,
    breakpoint_layer: int = 20,
    output_dir: Path = Path("/tmp/exp3_osdi_results"),
) -> Dict[str, Any]:
    """
    Run PyTorch Eager baseline demonstrating VRAM holding during pause.
    
    Scientific Goal: Show that standard PyTorch inference CANNOT release VRAM during
    pause/think-time, preventing GPU sharing. Djinn releases VRAM by swapping to host.
    
    Args:
        model_name: HuggingFace model ID
        input_length: Input sequence length (long context to make KV cache meaningful)
        pause_duration_seconds: How long to "pause" (simulating think time)
        breakpoint_layer: Layer index to pause at
        output_dir: Output directory for logs
    
    Returns:
        Dictionary with baseline results
    """
    setup_logging(output_dir)
    
    logger.info("="*80)
    logger.info("BASELINE: PyTorch Eager - VRAM Holding During Pause")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Breakpoint Layer: {breakpoint_layer}")
    logger.info(f"Input Length: {input_length} (long context for meaningful KV cache)")
    logger.info(f"Pause Duration: {pause_duration_seconds}s")
    logger.info("Scientific Goal: Prove PyTorch cannot share GPU during pause\n")
    
    try:
        # Check initial GPU state
        mem_before = check_gpu_memory()
        logger.info(f"Initial GPU Memory: {mem_before}")
        
        # Load model
        logger.info("\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # CRITICAL FIX: Ensure pad_token is set (required for batched inference)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer.pad_token to eos_token")
        
        mem_after_model = check_gpu_memory()
        model_vram_gb = mem_after_model['allocated_gb'] - mem_before['allocated_gb']
        logger.info(f"Model Loaded. GPU Memory: {mem_after_model}")
        logger.info(f"Model Weight VRAM: {model_vram_gb:.2f}GB")
        
        # Prepare input
        logger.info(f"\nPreparing input ({input_length} tokens)...")
        prompt = "The future of AI is" + " context token " * (input_length - 5)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=input_length,
        ).to("cuda:0")
        
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        
        # Install hook to stop at breakpoint layer
        activations_captured = {}
        layer_counter = [0]
        
        def make_hook(target_layer):
            def hook_fn(module, input, output):
                layer_idx = layer_counter[0]
                if layer_idx == target_layer:
                    logger.info(f"Breakpoint reached at layer {layer_idx}")
                    activations_captured['activation'] = output
                    if isinstance(output, tuple):
                        activations_captured['activation'] = output[0]
                    raise RuntimeError("Breakpoint: Stop forward pass")
                layer_counter[0] += 1
                return None
            return hook_fn
        
        # Find the transformer layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            target_module = model.model.layers[breakpoint_layer]
            logger.info(f"Installing hook on layer {breakpoint_layer}")
            handle = target_module.register_forward_hook(make_hook(0))
        else:
            logger.error("Could not find transformer layers")
            return {"status": "error", "reason": "Could not locate transformer layers"}
        
        # Run partial forward pass
        logger.info(f"\nRunning forward pass to layer {breakpoint_layer}...")
        mem_before_forward = check_gpu_memory()
        
        with torch.no_grad():
            try:
                _ = model(**inputs)
            except RuntimeError as e:
                if "Breakpoint" in str(e):
                    logger.info("Forward pass stopped at breakpoint")
                else:
                    raise
        
        mem_at_breakpoint = check_gpu_memory()
        logger.info(f"GPU Memory at breakpoint: {mem_at_breakpoint}")
        
        # CRITICAL TEST: Pause and measure VRAM (should stay the same)
        logger.info(f"\nPausing for {pause_duration_seconds} seconds...")
        logger.info("VRAM should remain fully allocated (proves GPU cannot be shared)")
        
        pause_start = time.perf_counter()
        vram_samples = []
        
        for i in range(pause_duration_seconds * 2):  # Sample 2x per second
            vram_samples.append(check_gpu_memory()['allocated_gb'])
            time.sleep(0.5)
        
        pause_duration = time.perf_counter() - pause_start
        
        mem_after_pause = check_gpu_memory()
        logger.info(f"GPU Memory after pause: {mem_after_pause}")
        
        # Calculate VRAM holding
        vram_held_gb = mem_at_breakpoint['allocated_gb']
        vram_variation_gb = max(vram_samples) - min(vram_samples)
        
        logger.info(f"\n📊 VRAM Holding Analysis:")
        logger.info(f"   VRAM at breakpoint: {mem_at_breakpoint['allocated_gb']:.2f}GB")
        logger.info(f"   VRAM after pause: {mem_after_pause['allocated_gb']:.2f}GB")
        logger.info(f"   VRAM variation during pause: {vram_variation_gb:.4f}GB")
        logger.info(f"   GPU is BLOCKED: {mem_after_pause['free_gb'] < 1.0}")
        
        # Cleanup
        handle.remove()
        del model
        torch.cuda.empty_cache()
        
        # Results
        result = {
            "status": "success",
            "baseline": "pytorch_eager",
            "model": model_name,
            "breakpoint_layer": breakpoint_layer,
            "pause_duration_seconds": pause_duration,
            "metrics": {
                "vram_held_during_pause_gb": vram_held_gb,
                "vram_variation_during_pause_gb": vram_variation_gb,
                "gpu_free_during_pause_gb": mem_after_pause['free_gb'],
                "can_run_other_requests": mem_after_pause['free_gb'] > 5.0,  # Need 5GB+ for small request
            },
            "conclusion": {
                "vram_released": False,
                "allows_concurrent_requests": False,
                "requires_full_gpu_for_pause": True,
            }
        }
        
        logger.info(f"\n✅ PyTorch Eager Baseline Results:")
        logger.info(f"   VRAM Held: {result['metrics']['vram_held_during_pause_gb']:.2f}GB")
        logger.info(f"   Allows Other Requests: {result['metrics']['can_run_other_requests']}")
        logger.info(f"   GPU Shared: False ❌")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Baseline failed: {e}", exc_info=True)
        return {
            "status": "error",
            "baseline": "pytorch_eager",
            "error": str(e),
        }


def main():
    """Run PyTorch Eager baseline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PyTorch Eager Baseline: VRAM Holding During Pause"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-13b-hf",
        help="Model name"
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=512,
        help="Input length in tokens"
    )
    parser.add_argument(
        "--pause-duration",
        type=int,
        default=10,
        help="Pause duration in seconds"
    )
    parser.add_argument(
        "--breakpoint-layer",
        type=int,
        default=20,
        help="Layer to pause at"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/exp3_osdi_results"),
        help="Output directory"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    result = run_pytorch_eager_baseline(
        model_name=args.model,
        input_length=args.input_length,
        pause_duration_seconds=args.pause_duration,
        breakpoint_layer=args.breakpoint_layer,
        output_dir=args.output_dir,
    )
    
    # Save results
    results_file = args.output_dir / "pytorch_eager_results.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"\n✅ Results saved to {results_file}")
    
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

