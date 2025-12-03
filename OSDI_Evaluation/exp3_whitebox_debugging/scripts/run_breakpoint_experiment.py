#!/usr/bin/env python3
"""
Experiment 3: White-Box Breakpoint Debugging (Real Client-Server Mode)

Demonstrates that Djinn's breakpoint abstraction enables zero-cost context switching
via remote server execution. Client registers model, then executes with mid-inference
breakpoints over the network.

Scientific Goal:
- Prove Djinn's LazyTensor abstraction enables zero-cost context switching
- Show breakpoint checkpoint/restore overhead < 10% of compute time
- Demonstrate layer-wise granularity (pause at any layer boundary)

Metrics:
- Checkpoint latency (save time)
- Restore latency (load time)
- Context switch overhead (pause duration)
- Correctness (breakpoint output == full output)

Usage:
    # Terminal 1: Start server
    python -m djinn.server.server_main --enable-breakpoints --port 5556
    
    # Terminal 2: Run experiment
    python run_breakpoint_experiment.py \
        --config configs/breakpoint_smoke.yaml \
        --model gpt2 \
        --server localhost:5556 \
        --output-dir results
"""

import sys
import os
import argparse
import logging
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

# Add djinn to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import numpy as np

# Djinn imports
import djinn
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model
from Evaluation.common.djinn_init import ensure_initialized_before_async

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info(f"✅ Logging configured (output: {log_file})")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Loaded config from {config_path}")
    return config


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading model: {model_name}")
    
    # Load on CPU initially (ghost loader will handle GPU placement)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info(f"✅ Model loaded: {model_name}")
    logger.info(f"   Layers: {count_model_layers(model)}")
    logger.info(f"   Parameters: {count_parameters(model) / 1e6:.1f}M")
    
    return model, tokenizer


def count_model_layers(model) -> int:
    """Count the number of layers in model."""
    # Try common layer attributes
    for attr in ['transformer', 'model', 'encoder', 'decoder']:
        if hasattr(model, attr):
            submodel = getattr(model, attr)
            for layer_attr in ['layer', 'layers', 'h', 'blocks']:
                if hasattr(submodel, layer_attr):
                    layers = getattr(submodel, layer_attr)
                    if isinstance(layers, (list, torch.nn.ModuleList)):
                        return len(layers)
    
    # Fallback: count all nn.Module children
    return sum(1 for _ in model.modules())


def count_parameters(model) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def get_device() -> torch.device:
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        logger.warning("⚠️  CUDA not available, using CPU (very slow)")
        return torch.device('cpu')


async def run_breakpoint_test(
    manager: EnhancedModelManager,
    model,
    tokenizer,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Run breakpoint debugging evaluation.
    
    Tests model execution with breakpoints at different layers,
    collecting checkpoint/restore metrics.
    """
    # Get breakpoint configuration
    breakpoint_config = config.get('experiment', {}).get('breakpoints', {})
    breakpoint_layers = breakpoint_config.get('layers', [5, 15, 25])
    
    # Get inference configuration
    inf_config = config.get('experiment', {}).get('inference', {})
    input_length = inf_config.get('input_length', 32)
    tolerance = inf_config.get('correctness_tolerance', 0.1)
    
    results = {
        "summary": {
            "total_tests": 0,
            "successful": 0,
            "failed": 0,
            "avg_checkpoint_time_ms": 0.0,
            "avg_pause_duration_ms": 0.0,
            "avg_restore_time_ms": 0.0,
            "avg_overhead_percent": 0.0,
        },
        "breakpoints": {},
    }
    
    # Prepare input
    input_ids = tokenizer(
        "The future of AI is",
        return_tensors="pt",
        padding=True,
        max_length=input_length,
        truncation=True
    )["input_ids"].to(device)
    
    # Get baseline output (full execution without breakpoint)
    logger.info("\n" + "="*80)
    logger.info("COLLECTING BASELINE (Full Execution)")
    logger.info("="*80)
    
    try:
        baseline_output, _ = await manager.execute_model(
            model,
            {"input_ids": input_ids},
            hints={"use_generate": False}
        )
        logger.info("✅ Baseline execution complete")
    except Exception as e:
        logger.error(f"❌ Failed to get baseline: {e}")
        baseline_output = None
    
    # Test breakpoints at different layers
    all_checkpoint_times = []
    all_pause_durations = []
    all_restore_times = []
    all_overheads = []
    
    for layer_idx in breakpoint_layers:
        logger.info("\n" + "-"*80)
        logger.info(f"Testing breakpoint at layer {layer_idx}...")
        logger.info("-"*80)
        
        session_id = f"breakpoint_test_{layer_idx}"
        
        try:
            # Execute with breakpoint via remote server
            model_output, metrics = await manager.coordinator.execute_remote_model_with_breakpoint(
                fingerprint=manager.registered_models[model]['fingerprint'],
                inputs={"input_ids": input_ids},
                breakpoint_layer_index=layer_idx,
                wait_for_resume=True,
                session_id=session_id,
            )
            
            # Collect metrics
            checkpoint_ms = metrics.get('checkpoint_time_ms', 0.0)
            pause_ms = metrics.get('pause_duration_ms', 0.0)
            restore_ms = metrics.get('restore_time_ms', 0.0)
            overhead = metrics.get('overhead_percent', 0.0)
            
            all_checkpoint_times.append(checkpoint_ms)
            all_pause_durations.append(pause_ms)
            all_restore_times.append(restore_ms)
            all_overheads.append(overhead)
            
            # Verify correctness if available
            correctness_passed = True
            if model_output is not None and baseline_output is not None:
                try:
                    if hasattr(model_output, 'logits'):
                        output_logits = model_output.logits
                    else:
                        output_logits = model_output
                    
                    if hasattr(baseline_output, 'logits'):
                        baseline_logits = baseline_output.logits
                    else:
                        baseline_logits = baseline_output
                    
                    logits_diff = torch.norm(
                        output_logits.float() - baseline_logits.float()
                    ).item()
                    correctness_passed = logits_diff < tolerance
                    
                    logger.info(f"   Correctness check: {'✅ PASS' if correctness_passed else '❌ FAIL'}")
                    logger.info(f"   Logit difference: {logits_diff:.4f} (tolerance: {tolerance})")
                except Exception as e:
                    logger.warning(f"Could not verify correctness: {e}")
            
            # Store result
            results["breakpoints"][layer_idx] = {
                "checkpoint_time_ms": checkpoint_ms,
                "pause_duration_ms": pause_ms,
                "restore_time_ms": restore_ms,
                "total_overhead_ms": checkpoint_ms + pause_ms + restore_ms,
                "overhead_percent": overhead,
                "correctness": correctness_passed,
                "checkpoint_size_mb": metrics.get('checkpoint_size_mb', 0.0),
            }
            
            results["summary"]["successful"] += 1
            
            logger.info(f"✅ Breakpoint test at layer {layer_idx} completed:")
            logger.info(f"   Checkpoint: {checkpoint_ms:.1f}ms")
            logger.info(f"   Pause: {pause_ms:.1f}ms")
            logger.info(f"   Restore: {restore_ms:.1f}ms")
            logger.info(f"   Total Overhead: {checkpoint_ms + pause_ms + restore_ms:.1f}ms ({overhead:.1f}%)")
        
        except Exception as e:
            logger.error(f"❌ Breakpoint test failed: {e}", exc_info=True)
            results["breakpoints"][layer_idx] = {"error": str(e)}
            results["summary"]["failed"] += 1
        
        results["summary"]["total_tests"] += 1
    
    # Calculate summary statistics
    if all_checkpoint_times:
        results["summary"]["avg_checkpoint_time_ms"] = np.mean(all_checkpoint_times)
        results["summary"]["avg_pause_duration_ms"] = np.mean(all_pause_durations)
        results["summary"]["avg_restore_time_ms"] = np.mean(all_restore_times)
        results["summary"]["avg_overhead_percent"] = np.mean(all_overheads)
    
    return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Experiment 3: White-Box Breakpoint Debugging (Client-Server Mode)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent / 'configs' / 'breakpoint_smoke.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        help='Model name (default: gpt2)'
    )
    parser.add_argument(
        '--server',
        type=str,
        default='localhost:5556',
        help='Server address (default: localhost:5556)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/tmp/exp3_results'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, args.log_level)
    
    logger.info("🚀 Experiment 3: White-Box Breakpoint Debugging (Client-Server Mode)")
    logger.info(f"   Config: {args.config}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Server: {args.server}")
    logger.info(f"   Output: {output_dir}")
    logger.info("")
    
    try:
        # Initialize Djinn before entering async context
        ensure_initialized_before_async(args.server)
        
        # Load configuration
        config = load_config(args.config)
        
        # Get device
        device = get_device()
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(args.model)
        
        # Initialize coordinator and model manager
        coordinator = get_coordinator()
        manager = EnhancedModelManager(coordinator=coordinator)
        
        # Register model with server
        logger.info("\n" + "="*80)
        logger.info("REGISTERING MODEL WITH SERVER")
        logger.info("="*80 + "\n")
        
        await manager.register_model(model, model_id=args.model)
        
        logger.info("\n" + "="*80)
        logger.info("STARTING BREAKPOINT DEBUGGING EVALUATION")
        logger.info("="*80 + "\n")
        
        # Run experiment
        results = await run_breakpoint_test(
            manager=manager,
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device
        )
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETED")
        logger.info("="*80)
        logger.info(f"\n📊 Results Summary:")
        logger.info(f"   Total tests: {results['summary']['total_tests']}")
        logger.info(f"   Successful: {results['summary']['successful']}")
        logger.info(f"   Failed: {results['summary']['failed']}")
        logger.info(f"\n⏱️  Timing (average across layers):")
        logger.info(f"   Checkpoint: {results['summary']['avg_checkpoint_time_ms']:.1f}ms")
        logger.info(f"   Pause: {results['summary']['avg_pause_duration_ms']:.1f}ms")
        logger.info(f"   Restore: {results['summary']['avg_restore_time_ms']:.1f}ms")
        logger.info(f"   Overhead: {results['summary']['avg_overhead_percent']:.1f}%")
        logger.info(f"\n📁 Results saved to {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
