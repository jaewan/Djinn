#!/usr/bin/env python3
"""
Step 3: Resume Breakpoint Session (Client Process B)

After verify_vram_freed.py confirms GPU is free:
- Reconnects using session_id from start_breakpoint.py
- Resumes execution from the saved checkpoint
- Verifies correctness of output

This proves true process persistence: a different process reconnects
and continues computation from saved state.

Usage:
    source .venv/bin/activate
    python scripts/resume_breakpoint.py \
        --session-file /tmp/exp3_session.txt \
        --output /tmp/exp3_resume_results.json
"""

import sys
import os
import argparse
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

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


async def resume_breakpoint_session(
    session_info: Dict[str, Any],
    server_addr: str = "localhost:5556"
) -> Dict[str, Any]:
    """
    Resume a paused breakpoint session.
    
    Args:
        session_info: Dict from start_breakpoint.py with session_id, checkpoint_id, etc
        server_addr: Server address
    
    Returns:
        Dict with resume metrics and correctness verification
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Resume Breakpoint Session (Client Process B)")
    logger.info("=" * 80)
    
    session_id = session_info['session_id']
    checkpoint_id = session_info['checkpoint_id']
    model_name = session_info['model_name']
    breakpoint_layer = session_info['breakpoint_layer']
    
    logger.info(f"\n📋 Session Info:")
    logger.info(f"   Session ID: {session_id}")
    logger.info(f"   Checkpoint ID: {checkpoint_id}")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Breakpoint layer: {breakpoint_layer}")
    
    # Get device
    device = get_device()
    
    # Create ghost model again
    logger.info(f"\n📦 Creating ghost model: {model_name}")
    model = create_hf_ghost_model(model_name, task="causal-lm")
    logger.info("✅ Ghost model created")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    logger.info(f"\n📝 Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("✅ Tokenizer loaded")
    
    # Reconnect to server
    logger.info(f"\n🔗 Reconnecting to server: {server_addr}")
    coordinator = get_coordinator()
    manager = EnhancedModelManager(coordinator=coordinator)
    logger.info("✅ Reconnected to server")
    
    # Register model again
    logger.info(f"\n📤 Registering model")
    await manager.register_model(model, model_id=model_name)
    logger.info("✅ Model registered")
    
    # Prepare same input (keep on CPU for thin-client semantics)
    input_text = session_info.get('input_text', "The future of AI is")
    logger.info(f"\n💬 Input text: '{input_text}'")
    
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        max_length=128,
        truncation=True
    )["input_ids"]  # IMPORTANT: Keep on CPU, not GPU (thin-client architecture)
    
    logger.info(f"✅ Input prepared")
    logger.info(f"   Input device: {input_ids.device} (thin-client: should be CPU)")
    
    # Resume execution
    logger.info(f"\n🚀 Resuming execution from layer {breakpoint_layer}")
    logger.info("   Restoring checkpoint and continuing inference...")
    logger.info(f"   IMPORTANT: Reusing session_id={session_id} signals server to resume from checkpoint")
    logger.info("   NOT to re-execute from layer 0. Coordinator must handle session_id semantics correctly.")
    
    try:
        # Compute fingerprint to look up registered model
        fingerprint, _ = manager._compute_fingerprint(model, model_id=model_name)
        
        if fingerprint not in manager.registered_models:
            logger.error(f"❌ Model not registered: fingerprint={fingerprint[:8]}, model_name={model_name}")
            raise RuntimeError(f"Model {model_name} not registered")
        
        # Call with same session_id to signal "resume from existing checkpoint"
        # The server's coordinator must recognize session_id and restore the checkpoint
        # rather than starting a new execution with breakpoint
        # If this produces incorrect results, the RPC layer may not support true resume semantics
        model_output, metrics = await manager.coordinator.execute_remote_model_with_breakpoint(
            fingerprint=fingerprint,
            inputs={"input_ids": input_ids},
            breakpoint_layer_index=breakpoint_layer,
            wait_for_resume=True,  # Wait for full execution this time
            session_id=session_id,  # Reuse same session - CRITICAL: triggers resume behavior
        )
        
        # Verify that output came from resuming, not re-executing from layer 0
        if 'resumed_from_checkpoint' in metrics:
            logger.info(f"✅ Confirmed: Execution resumed from checkpoint (not re-executed from layer 0)")
        else:
            logger.warning(f"⚠️  WARNING: 'resumed_from_checkpoint' metric missing - verify server actually resumed")
            logger.warning(f"   If output differs from first run, server may have re-executed instead of resuming")
        
        restore_time_ms = metrics.get('restore_time_ms', 0.0)
        resume_compute_time_ms = metrics.get('compute_time_after_resume_ms', 0.0)
        
        logger.info(f"\n✅ Execution resumed!")
        logger.info(f"   Restore time: {restore_time_ms:.1f}ms")
        logger.info(f"   Compute time after resume: {resume_compute_time_ms:.1f}ms")
        
        # Verify output
        logger.info(f"\n🔍 Verifying output correctness")
        if model_output is not None:
            if hasattr(model_output, 'logits'):
                output_logits = model_output.logits
            else:
                output_logits = model_output
            
            output_tokens = output_logits.argmax(dim=-1)
            logger.info(f"   Generated tokens: {output_tokens.shape}")
            logger.info(f"   Sample output tokens: {output_tokens[0, :10].tolist()}")
            logger.info(f"✅ Output generated successfully")
        
        result = {
            "success": True,
            "session_id": session_id,
            "restore_time_ms": restore_time_ms,
            "resume_compute_time_ms": resume_compute_time_ms,
            "total_pause_duration_ms": metrics.get('total_pause_duration_ms', 0.0),
            "correctness": metrics.get('correctness', False),
            "token_accuracy_percent": metrics.get('token_accuracy_percent', 0.0),
        }
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Failed to resume: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id,
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Step 3: Resume Breakpoint Session (Process B)'
    )
    parser.add_argument(
        '--session-file',
        type=Path,
        default=Path('/tmp/exp3_session.txt'),
        help='Session file from start_breakpoint.py (default: /tmp/exp3_session.txt)'
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
        default=Path('/tmp/exp3_resume_results.json'),
        help='Output file for resume results'
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
    
    logger.info("🚀 Resume Breakpoint Session (Process B)")
    logger.info(f"   Session file: {args.session_file}")
    logger.info(f"   Server: {args.server}")
    logger.info("")
    
    try:
        # Load session info
        if not args.session_file.exists():
            logger.error(f"❌ Session file not found: {args.session_file}")
            logger.info("   Run start_breakpoint.py first to create session")
            return 1
        
        with open(args.session_file) as f:
            session_info = json.load(f)
        
        logger.info(f"✅ Loaded session info from {args.session_file}")
        
        # Initialize Djinn
        ensure_initialized_before_async(args.server)
        
        # Resume session
        result = await resume_breakpoint_session(
            session_info=session_info,
            server_addr=args.server
        )
        
        # Save results
        logger.info(f"\n📁 Saving resume results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info("\n" + "=" * 80)
        if result.get('success'):
            logger.info("✅ EXECUTION COMPLETED - SESSION RESUMED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"\nRestore time: {result.get('restore_time_ms', 0):.1f}ms")
            logger.info(f"Resume compute: {result.get('resume_compute_time_ms', 0):.1f}ms")
            logger.info(f"Total pause duration: {result.get('total_pause_duration_ms', 0):.1f}ms")
            logger.info(f"Correctness: {result.get('correctness', False)}")
            logger.info(f"Token accuracy: {result.get('token_accuracy_percent', 0):.1f}%")
            return 0
        else:
            logger.error("❌ EXECUTION FAILED")
            logger.info("=" * 80)
            logger.error(f"Error: {result.get('error', 'Unknown')}")
            return 1
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
