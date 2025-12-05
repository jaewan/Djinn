#!/usr/bin/env python3
"""
Step 1: Start Breakpoint Session (Client Process A)

Connects to Djinn server, executes model to breakpoint layer, then exits.
The state persists on the server via session_id.

This demonstrates the key property: process termination does NOT lose state.
The checkpoint and session state live on the server.

Usage:
    source .venv/bin/activate
    python scripts/start_breakpoint.py \
        --model llama-3-8b \
        --breakpoint-layer 15 \
        --output /tmp/exp3_session.txt

Output:
    Prints session_id to stdout and /tmp/exp3_session.txt for resume step.
"""

import sys
import os
import argparse
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add djinn to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

import torch

# Djinn imports
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model
from Evaluation.common.djinn_init import ensure_initialized_before_async

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)


def get_device() -> torch.device:
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        logger.warning("CUDA not available, using CPU")
        return torch.device('cpu')


async def start_breakpoint_session(
    model_name: str,
    breakpoint_layer: int,
    server_addr: str = "localhost:5556"
) -> Dict[str, Any]:
    """
    Start a breakpoint session and execute to breakpoint.
    
    Returns:
        Dict with session_id, checkpoint_id, and metadata
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Start Breakpoint Session (Client Process A)")
    logger.info("=" * 80)
    
    # Get device
    device = get_device()
    
    # Create ghost model (zero-memory skeleton)
    logger.info(f"\n📦 Creating ghost model: {model_name}")
    model = create_hf_ghost_model(model_name, task="causal-lm")
    logger.info("✅ Ghost model created (zero-memory skeleton)")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    logger.info(f"\n📝 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("✅ Tokenizer loaded")
    
    # Initialize coordinator and manager
    logger.info(f"\n🔗 Connecting to server: {server_addr}")
    coordinator = get_coordinator()
    manager = EnhancedModelManager(coordinator=coordinator)
    logger.info("✅ Connected to server")
    
    # Register model with server
    logger.info(f"\n📤 Registering model with server")
    await manager.register_model(model, model_id=model_name)
    logger.info("✅ Model registered")
    
    # Prepare input (keep on CPU for thin-client semantics)
    input_text = "The future of AI is"
    logger.info(f"\n💬 Input text: '{input_text}'")
    
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        max_length=128,
        truncation=True
    )["input_ids"]  # IMPORTANT: Keep on CPU, not GPU (thin-client architecture)
    
    logger.info(f"✅ Input prepared: shape {input_ids.shape}")
    logger.info(f"   Input device: {input_ids.device} (thin-client: should be CPU)")
    
    # Execute with breakpoint
    logger.info(f"\n🚀 Executing model with breakpoint at layer {breakpoint_layer}")
    logger.info("   This will pause at the breakpoint and save activation checkpoint...")
    
    session_id = f"breakpoint_session_{breakpoint_layer}_{int(__import__('time').time())}"
    
    try:
        # Compute fingerprint to look up registered model
        fingerprint, _ = manager._compute_fingerprint(model, model_id=model_name)
        
        if fingerprint not in manager.registered_models:
            logger.error(f"❌ Model not registered: fingerprint={fingerprint[:8]}, model_name={model_name}")
            raise RuntimeError(f"Model {model_name} not registered")
        
        model_output, metrics = await manager.coordinator.execute_remote_model_with_breakpoint(
            fingerprint=fingerprint,
            inputs={"input_ids": input_ids},
            breakpoint_layer_index=breakpoint_layer,
            wait_for_resume=False,  # KEY: Don't wait! Exit after breakpoint is hit
            session_id=session_id,
        )
        
        logger.info(f"\n✅ Breakpoint triggered!")
        logger.info(f"   Session ID: {session_id}")
        
        # Collect metrics
        checkpoint_id = metrics.get('checkpoint_id', '')
        checkpoint_size_mb = metrics.get('checkpoint_size_mb', 0.0)
        checkpoint_time_ms = metrics.get('checkpoint_time_ms', 0.0)
        
        logger.info(f"   Checkpoint ID: {checkpoint_id}")
        logger.info(f"   Checkpoint size: {checkpoint_size_mb:.1f}MB")
        logger.info(f"   Checkpoint time: {checkpoint_time_ms:.1f}ms")
        
        result = {
            "session_id": session_id,
            "checkpoint_id": checkpoint_id,
            "model_name": model_name,
            "breakpoint_layer": breakpoint_layer,
            "checkpoint_size_mb": checkpoint_size_mb,
            "checkpoint_time_ms": checkpoint_time_ms,
            "input_text": input_text,
        }
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Failed to execute: {e}", exc_info=True)
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Step 1: Start Breakpoint Session (Process A)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama-3-8b',
        help='Model name (default: llama-3-8b)'
    )
    parser.add_argument(
        '--breakpoint-layer',
        type=int,
        default=15,
        help='Layer to break at (default: 15)'
    )
    parser.add_argument(
        '--server',
        type=str,
        default='localhost:5556',
        help='Server address (default: localhost:5556)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/tmp/exp3_session.txt'),
        help='Output file for session_id'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    args.output.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output.parent, args.log_level)
    
    logger.info("🚀 Start Breakpoint Session (Process A)")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Breakpoint layer: {args.breakpoint_layer}")
    logger.info(f"   Server: {args.server}")
    logger.info("")
    
    try:
        # Initialize Djinn
        ensure_initialized_before_async(args.server)
        
        # Start session
        result = await start_breakpoint_session(
            model_name=args.model,
            breakpoint_layer=args.breakpoint_layer,
            server_addr=args.server
        )
        
        # Save session info to file for resume step
        logger.info(f"\n📁 Saving session info to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ SESSION STARTED - CLIENT PROCESS EXITING")
        logger.info("=" * 80)
        logger.info(f"\nSession ID: {result['session_id']}")
        logger.info(f"Checkpoint ID: {result['checkpoint_id']}")
        logger.info(f"\nState is now persisted on server!")
        logger.info(f"Next step: Run verify_vram_freed.py, then resume_breakpoint.py")
        logger.info(f"Resume file: {args.output}\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
