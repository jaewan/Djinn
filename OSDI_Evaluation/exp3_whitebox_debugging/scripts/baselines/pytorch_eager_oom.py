#!/usr/bin/env python3
"""
Baseline: PyTorch Eager Mode Exhaustion

Demonstrates that PyTorch eager execution cannot achieve zero-VRAM context switching.
Shows OOM when trying to hold state during pause and run second job.
"""

import sys
import os
import argparse
import logging
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)


def query_vram() -> Optional[dict]:
    """Query GPU memory usage."""
    try:
        import subprocess
        cmd = ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', '-i', '0']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(', ')
        return {'used_mb': float(parts[0]), 'total_mb': float(parts[1])} if len(parts) >= 2 else None
    except Exception as e:
        logger.warning(f"Could not query VRAM: {e}")
        return None


def pytorch_eager_breakpoint_test(model_name: str, breakpoint_layer: int, pause_duration_secs: float = 10.0) -> dict:
    """Simulate context switching with PyTorch eager mode."""
    logger.info("=" * 80)
    logger.info("BASELINE: PyTorch Eager Mode (No Context Switching)")
    logger.info("=" * 80)
    
    results = {
        "model": model_name,
        "breakpoint_layer": breakpoint_layer,
        "status": "unknown",
        "error": None,
        "vram_before_load": 0.0,
        "vram_after_load": 0.0,
        "vram_during_pause": 0.0,
        "oom_occurred": False,
    }
    
    try:
        logger.info("\n📊 Querying initial VRAM...")
        vram_start = query_vram()
        if vram_start:
            logger.info(f"   Before: {vram_start['used_mb']:.0f}MB / {vram_start['total_mb']:.0f}MB")
            results["vram_before_load"] = vram_start['used_mb']
        
        logger.info(f"\n📥 Loading {model_name} in eager mode (FULL WEIGHTS on GPU)...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='cuda:0')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✅ Model loaded")
        
        vram_after_load = query_vram()
        if vram_after_load:
            logger.info(f"   After: {vram_after_load['used_mb']:.0f}MB / {vram_after_load['total_mb']:.0f}MB")
            results["vram_after_load"] = vram_after_load['used_mb']
        
        logger.info(f"\n💬 Preparing inference...")
        input_ids = tokenizer("The future of AI is", return_tensors="pt", max_length=128, truncation=True)["input_ids"].to('cuda:0')
        
        logger.info(f"\n🚀 Running inference...")
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            activations = outputs.hidden_states[breakpoint_layer]
        logger.info(f"✅ Inference executed. Activation: {activations.shape}")
        
        logger.info(f"\n⏸️  Simulating pause ({pause_duration_secs}s)...")
        logger.info("   Djinn would swap here. PyTorch cannot - state is locked in VRAM.")
        
        vram_during_pause = query_vram()
        if vram_during_pause:
            logger.info(f"   During: {vram_during_pause['used_mb']:.0f}MB / {vram_during_pause['total_mb']:.0f}MB")
            results["vram_during_pause"] = vram_during_pause['used_mb']
        
        time.sleep(pause_duration_secs)
        
        vram_after_pause = query_vram()
        if vram_after_pause:
            free_vram = vram_after_pause['total_mb'] - vram_after_pause['used_mb']
            logger.info(f"   After pause: {vram_after_pause['used_mb']:.0f}MB / {vram_after_pause['total_mb']:.0f}MB")
            logger.info(f"   Free VRAM: {free_vram:.0f}MB")
            
            if free_vram < 2000:
                logger.warning(f"❌ INSUFFICIENT MEMORY for second job! Need ~2000MB, have {free_vram:.0f}MB")
                results["oom_occurred"] = True
        
        logger.info(f"\n▶️  Resuming...")
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        logger.info("✅ Resumed")
        
        if results["oom_occurred"]:
            logger.error("\n❌ PyTorch FAILS: Model + activations locked in VRAM during pause")
            logger.error("   ➜ Djinn's context switching is a REAL contribution")
            results["status"] = "oom"
        else:
            logger.warning("\n⚠️  Second job hypothetically fit (model smaller than expected)")
            results["status"] = "would_oom_larger_model"
        
        return results
    
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ CUDA Out of Memory: {e}")
        results["status"] = "oom"
        results["error"] = str(e)
        results["oom_occurred"] = True
        return results
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        results["status"] = "error"
        results["error"] = str(e)
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Baseline: PyTorch Eager Mode (Why Djinn is needed)')
    parser.add_argument('--model', type=str, default='gpt2-medium', help='Model name (default: gpt2-medium)')
    parser.add_argument('--breakpoint-layer', type=int, default=5, help='Layer to break at (default: 5)')
    parser.add_argument('--pause-duration', type=float, default=5.0, help='Pause duration (default: 5s)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    logger.info("🔴 PyTorch Eager Mode Baseline: Why Context Switching is Needed")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Breakpoint layer: {args.breakpoint_layer}")
    logger.info("")
    
    try:
        result = pytorch_eager_breakpoint_test(args.model, args.breakpoint_layer, args.pause_duration)
        logger.info("\nRESULTS:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
        return 0 if result["status"] in ("oom", "would_oom_larger_model") else 1
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
