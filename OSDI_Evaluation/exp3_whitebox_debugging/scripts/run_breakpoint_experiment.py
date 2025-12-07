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
    """Load model and tokenizer using GhostLoader for zero-memory skeleton."""
    from transformers import AutoTokenizer
    
    logger.info(f"Creating ghost model: {model_name}")
    
    # Create ghost model (zero-memory skeleton)
    # Server will download/map real weights to Text Segment
    model = create_hf_ghost_model(model_name, task="causal-lm")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Ghost model created: {model_name}")
    logger.info(f"   Ghost model size: <100MB (skeleton only)")
    logger.info(f"   Real weights: Will be loaded on server")
    
    return model, tokenizer




def get_device() -> torch.device:
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        device = torch.device('cuda:1')  # Use GPU1 to avoid conflict with other user
        torch.cuda.set_device(1)
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(1)}")
        return device
    else:
        logger.warning("⚠️  CUDA not available, using CPU (very slow)")
        return torch.device('cpu')


async def run_breakpoint_test(
    manager: EnhancedModelManager,
    model,
    tokenizer,
    config: Dict[str, Any],
    device: torch.device,
    model_name: str
) -> Dict[str, Any]:
    """
    Run breakpoint debugging evaluation with multiple trials.
    
    Tests model execution with breakpoints at different layers,
    collecting checkpoint/restore metrics with proper statistical rigor.
    """
    from common_utils import compute_statistics, format_statistics
    
    # Get breakpoint configuration
    breakpoint_config = config.get('experiment', {}).get('breakpoints', {})
    breakpoint_layers = breakpoint_config.get('layers', [5, 15, 25])
    num_trials = config.get('experiment', {}).get('num_trials', 3)
    
    # Get inference configuration
    inf_config = config.get('experiment', {}).get('inference', {})
    input_length = inf_config.get('input_length', 32)
    token_accuracy_threshold = inf_config.get('token_accuracy_threshold', 95.0)
    
    results = {
        "summary": {
            "total_tests": 0,
            "successful": 0,
            "failed": 0,
            "num_trials": num_trials,
            "num_layers": len(breakpoint_layers),
        },
        "statistics": {},  # Will contain per-layer statistics
        "breakpoints": {},  # Will contain per-trial data
    }
    
    # Prepare input
    input_ids = tokenizer(
        "The future of AI is",
        return_tensors="pt",
        padding=True,
        max_length=input_length,
        truncation=True
    )["input_ids"].to(device)
    
    # WARMUP: Execute once to eliminate cold-start effects (JIT, CUDA context init, etc.)
    logger.info("\n" + "="*80)
    logger.info("WARMUP EXECUTION (Cold-Start Elimination)")
    logger.info("="*80)
    try:
        await manager.execute_model(
            model,
            {"input_ids": input_ids},
            hints={"use_generate": False}
        )
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
        logger.info("✅ Warmup complete, cold-start effects eliminated")
    except Exception as e:
        logger.warning(f"⚠️  Warmup execution failed: {e}, continuing anyway")
    
    # Get baseline output (full execution without breakpoint)
    logger.info("\n" + "="*80)
    logger.info("COLLECTING BASELINE (Full Execution)")
    logger.info("="*80)
    
    try:
        baseline_output = await manager.execute_model(
            model,
            {"input_ids": input_ids},
            hints={"use_generate": False}
        )
        logger.info("✅ Baseline execution complete")
    except Exception as e:
        logger.error(f"❌ Failed to get baseline: {e}")
        baseline_output = None
    
    # Multi-trial testing
    per_layer_data = {layer: {
        'checkpoint_times': [],
        'pause_durations': [],
        'restore_times': [],
        'overheads': [],
        'token_accuracies': [],
        'checkpoint_sizes': [],
    } for layer in breakpoint_layers}
    
    for trial_num in range(num_trials):
        logger.info("\n" + "="*80)
        logger.info(f"TRIAL {trial_num + 1} / {num_trials}")
        logger.info("="*80)
        
        for layer_idx in breakpoint_layers:
            logger.info("\n" + "-"*80)
            logger.info(f"Trial {trial_num + 1}: Testing breakpoint at layer {layer_idx}...")
            logger.info("-"*80)
            
            session_id = f"breakpoint_test_t{trial_num}_l{layer_idx}_{int(__import__('time').time())}"
            
            try:
                # Compute fingerprint to look up registered model
                # (registered_models dict keys are fingerprints, not model names)
                fingerprint, _ = manager._compute_fingerprint(model, model_id=model_name)
                
                if fingerprint not in manager.registered_models:
                    logger.error(f"❌ Model not registered: fingerprint={fingerprint[:8]}, model_name={model_name}")
                    logger.error(f"   Available fingerprints: {list(manager.registered_models.keys())}")
                    raise RuntimeError(f"Model {model_name} (fingerprint: {fingerprint[:8]}) not registered")
                
                logger.debug(f"Using fingerprint {fingerprint[:8]} for model {model_name}")

                model_output, metrics = await manager.coordinator.execute_remote_model_with_breakpoint(
                    fingerprint=fingerprint,
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
                
                per_layer_data[layer_idx]['checkpoint_times'].append(checkpoint_ms)
                per_layer_data[layer_idx]['pause_durations'].append(pause_ms)
                per_layer_data[layer_idx]['restore_times'].append(restore_ms)
                per_layer_data[layer_idx]['overheads'].append(overhead)
                
                # Verify correctness at token level (more robust than logit norm)
                correctness_passed = True
                token_accuracy = 0.0

                # DEBUG: Check what we got
                logger.info(f"   DEBUG - model_output type: {type(model_output)}, baseline_output type: {type(baseline_output)}")
                logger.info(f"   DEBUG - model_output is None: {model_output is None}, baseline_output is None: {baseline_output is None}")

                if model_output is not None and baseline_output is not None:
                    try:
                        # Handle both dict and object-based outputs
                        if isinstance(model_output, dict):
                            output_logits = model_output.get('logits', model_output)
                        elif hasattr(model_output, 'logits'):
                            output_logits = model_output.logits
                        else:
                            output_logits = model_output
                        
                        if isinstance(baseline_output, dict):
                            baseline_logits = baseline_output.get('logits', baseline_output)
                        elif hasattr(baseline_output, 'logits'):
                            baseline_logits = baseline_output.logits
                        else:
                            baseline_logits = baseline_output
                        
                        # Compare at token level (more meaningful for OSDI)
                        output_tokens = output_logits.argmax(dim=-1)
                        baseline_tokens = baseline_logits.argmax(dim=-1)

                        # DEBUG: Log shapes and first few tokens
                        logger.info(f"   DEBUG - Output logits shape: {output_logits.shape}, baseline shape: {baseline_logits.shape}")
                        logger.info(f"   DEBUG - First 5 output tokens: {output_tokens[0][:5].tolist()}")
                        logger.info(f"   DEBUG - First 5 baseline tokens: {baseline_tokens[0][:5].tolist()}")
                        logger.info(f"   DEBUG - Output logits sample: {output_logits[0][0][:3].tolist()}")
                        logger.info(f"   DEBUG - Baseline logits sample: {baseline_logits[0][0][:3].tolist()}")

                        token_matches = (output_tokens == baseline_tokens).sum().item()
                        total_tokens = output_tokens.numel()
                        token_accuracy = 100.0 * token_matches / total_tokens if total_tokens > 0 else 0.0
                        
                        correctness_passed = token_accuracy >= token_accuracy_threshold
                        
                        logger.info(f"   Correctness check: {'✅ PASS' if correctness_passed else '❌ FAIL'}")
                        logger.info(f"   Token accuracy: {token_accuracy:.1f}% ({token_matches}/{total_tokens})")
                    except Exception as e:
                        logger.warning(f"Could not verify correctness: {e}")
                
                per_layer_data[layer_idx]['token_accuracies'].append(token_accuracy)
                per_layer_data[layer_idx]['checkpoint_sizes'].append(metrics.get('checkpoint_size_mb', 0.0))
                
                results["summary"]["successful"] += 1
                
                logger.info(f"✅ Layer {layer_idx} completed:")
                logger.info(f"   Checkpoint: {checkpoint_ms:.1f}ms")
                logger.info(f"   Pause: {pause_ms:.1f}ms")
                logger.info(f"   Restore: {restore_ms:.1f}ms")
                logger.info(f"   Overhead: {overhead:.1f}%")
            
            except Exception as e:
                logger.error(f"❌ Breakpoint test failed: {e}", exc_info=True)
                results["summary"]["failed"] += 1
            
            results["summary"]["total_tests"] += 1
    
    # Compute per-layer statistics
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL ANALYSIS (Across Trials)")
    logger.info("="*80)
    
    for layer_idx in breakpoint_layers:
        data = per_layer_data[layer_idx]
        
        stats = {
            'checkpoint_time': compute_statistics(data['checkpoint_times']),
            'pause_duration': compute_statistics(data['pause_durations']),
            'restore_time': compute_statistics(data['restore_times']),
            'overhead_percent': compute_statistics(data['overheads']),
            'token_accuracy': compute_statistics(data['token_accuracies']),
        }
        
        results["statistics"][layer_idx] = stats
        
        logger.info(f"\nLayer {layer_idx}:")
        logger.info(f"  Checkpoint: {format_statistics(stats['checkpoint_time'])}")
        logger.info(f"  Pause:      {format_statistics(stats['pause_duration'])}")
        logger.info(f"  Restore:    {format_statistics(stats['restore_time'])}")
        logger.info(f"  Overhead:   {format_statistics(stats['overhead_percent'])}")
        logger.info(f"  Accuracy:   {format_statistics(stats['token_accuracy'])}")
    
    # Summary statistics across all layers
    all_overheads = [v for data in per_layer_data.values() for v in data['overheads']]
    all_accuracies = [v for data in per_layer_data.values() for v in data['token_accuracies']]
    
    if all_overheads:
        results["summary"]["overhead_stats"] = compute_statistics(all_overheads)
        logger.info(f"\nOverall Overhead (all layers & trials):")
        logger.info(f"  {format_statistics(results['summary']['overhead_stats'])}")
    
    if all_accuracies:
        results["summary"]["accuracy_stats"] = compute_statistics(all_accuracies)
        logger.info(f"\nOverall Token Accuracy (all layers & trials):")
        logger.info(f"  {format_statistics(results['summary']['accuracy_stats'])}")
    
    return results


def main():
    """Main entry point - following exp2 pattern exactly."""
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

    # Initialize Djinn BEFORE entering async context (following exp2 pattern exactly)
    logger.info(f"[Djinn] Initializing client...")
    ensure_initialized_before_async(args.server)

    # Verify coordinator is available
    coordinator = get_coordinator()
    if coordinator is None:
        raise RuntimeError("Failed to initialize Djinn coordinator. Check server connection.")

    # Coordinator will be started in async context if needed
    logger.info("[Djinn] Client initialized successfully (coordinator will start in async context if needed)")

    # Run the experiment (pass coordinator to avoid async context issues)
    success = asyncio.run(run_experiment(args, coordinator))

    return 0 if success else 1


async def run_experiment(args, coordinator):
    """Run experiment in async context - following exp2 pattern exactly."""
    try:
        # Load configuration
        config = load_config(args.config)

        # Get device
        device = get_device()

        # Load model
        model, tokenizer = load_model_and_tokenizer(args.model)

        # Initialize model manager with the coordinator
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
            device=device,
            model_name=args.model
        )

        # Save results
        results_file = args.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETED")
        logger.info("="*80)
        logger.info(f"\n📊 Results Summary:")
        logger.info(f"   Total tests: {results['summary']['total_tests']}")
        logger.info(f"   Successful: {results['summary']['successful']}")
        logger.info(f"   Failed: {results['summary']['failed']}")
        if 'accuracy_stats' in results['summary']:
            logger.info(f"   Token accuracy: {results['summary']['accuracy_stats']['mean']:.1f}% ± {results['summary']['accuracy_stats']['std']:.1f}%")
        if 'overhead_stats' in results['summary']:
            logger.info(f"   Overhead: {results['summary']['overhead_stats']['mean']:.1f}% ± {results['summary']['overhead_stats']['std']:.1f}%")
        logger.info(f"\n📁 Results saved to {results_file}")

        return True

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
