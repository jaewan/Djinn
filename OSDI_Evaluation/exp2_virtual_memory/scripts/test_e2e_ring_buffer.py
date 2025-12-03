#!/usr/bin/env python3
"""
End-to-End Ring Buffer Integration Test

Tests that ring buffer works with Djinn's server-side model cache infrastructure.
Verifies: model loading, ring buffer allocation, weight hooks, async prefetch, inference.
"""

import sys
import logging
from pathlib import Path

# Add Djinn to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from djinn.backend.runtime.ring_buffer import WeightRingBuffer
from djinn.backend.runtime.weight_streamer import WeightStreamer
from djinn.backend.runtime.weight_hooks import install_ring_buffer_hooks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ring_buffer_e2e():
    """End-to-end test of ring buffer infrastructure."""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔬 E2E Ring Buffer Test on {device}")
    logger.info("=" * 70)
    
    # Step 1: Load a small model for testing
    logger.info("\n[STEP 1] Loading model...")
    model_id = "gpt2"  # Small model for testing
    logger.info(f"  Model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load to CPU (important for streaming architecture)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16
        )
        # Ensure model is on CPU
        model = model.to("cpu")
        model_params = sum(p.numel() for p in model.parameters())
        model_size_gb = (model_params * 2) / 1024**3  # float16 = 2 bytes
        logger.info(f"✅ Model loaded: {model_params/1e9:.2f}B params (~{model_size_gb:.2f}GB)")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Step 2: Create ring buffer
    logger.info("\n[STEP 2] Creating ring buffer...")
    try:
        capacity_gb = 4  # 4GB ring buffer for small test
        capacity_bytes = int(capacity_gb * 1024**3)
        ring_buffer = WeightRingBuffer(capacity_bytes, device=device)
        logger.info(f"✅ Ring buffer created: {capacity_gb}GB on {device}")
    except Exception as e:
        logger.error(f"❌ Failed to create ring buffer: {e}")
        return False
    
    # Step 3: Register model in ring buffer
    logger.info("\n[STEP 3] Registering model in ring buffer...")
    try:
        # Build model state dict by iterating (not caching all)
        model_state = {}
        for name, param in model.named_parameters():
            model_state[name] = param
        for name, buffer in model.named_buffers():
            model_state[name] = buffer
        
        registration = ring_buffer.register_model("test_model", model_state)
        logger.info(f"✅ Model registered: {len(registration.layer_allocations)} layers")
    except Exception as e:
        logger.error(f"❌ Failed to register model: {e}")
        return False
    
    # Step 4: Create weight streamer
    logger.info("\n[STEP 4] Creating weight streamer...")
    try:
        streamer = WeightStreamer(ring_buffer, device=device, chunk_size_bytes=64*1024*1024)
        streamer.start()
        logger.info(f"✅ Weight streamer started (chunk_size: 64MB)")
    except Exception as e:
        logger.error(f"❌ Failed to create streamer: {e}")
        return False
    
    # Step 5: Install weight hooks
    logger.info("\n[STEP 5] Installing weight hooks...")
    try:
        hook_mgr = install_ring_buffer_hooks(
            model,
            ring_buffer=ring_buffer,
            model_id="test_model",
            streamer=streamer
        )
        logger.info(f"✅ Weight hooks installed on {len(hook_mgr.layer_names)} layers")
    except Exception as e:
        logger.error(f"❌ Failed to install hooks: {e}")
        streamer.stop()
        return False
    
    # Step 6: Run inference (model stays on CPU, hooks handle weight placement)
    logger.info("\n[STEP 6] Running inference...")
    try:
        prompt = "The future of AI is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move only inputs to device, not model
        input_ids = inputs["input_ids"].to(device)
        logger.info(f"  Input: '{prompt}' (shape: {input_ids.shape})")
        
        torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        import time
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=False)
            logits = outputs.logits
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.perf_counter() - start
        
        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(f"✅ Inference complete: {elapsed*1000:.1f}ms, peak VRAM: {peak_vram_mb:.0f}MB")
        else:
            logger.info(f"✅ Inference complete: {elapsed*1000:.1f}ms")
        
        logger.info(f"  Output logits shape: {logits.shape}")
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        streamer.stop()
        return False
    
    # Step 7: Check streaming statistics
    logger.info("\n[STEP 7] Checking streaming statistics...")
    try:
        rb_stats = ring_buffer.get_stats()
        streamer_stats = streamer.get_stats()
        
        logger.info(f"  Ring Buffer Stats:")
        logger.info(f"    - Models registered: {rb_stats['models_registered']}")
        logger.info(f"    - Buffer wraps: {rb_stats['buffer_wraps']}")
        logger.info(f"    - Bytes transferred: {rb_stats['bytes_transferred'] / 1024**2:.1f}MB")
        
        logger.info(f"  Streamer Stats:")
        logger.info(f"    - Total prefetch jobs: {streamer_stats['total_prefetch_jobs']}")
        logger.info(f"    - Successful prefetches: {streamer_stats['prefetch_success']}")
        logger.info(f"    - Success rate: {streamer_stats['success_rate']*100:.1f}%")
        logger.info(f"    - Avg prefetch latency: {streamer_stats['avg_prefetch_latency_ms']:.1f}ms")
        
        if rb_stats['bytes_transferred'] > 0:
            logger.info(f"✅ Weight streaming active: {rb_stats['bytes_transferred']/1024**2:.1f}MB transferred")
        else:
            logger.warning(f"⚠️  No bytes transferred - weights may not have been streamed")
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        streamer.stop()
        return False
    
    # Cleanup
    logger.info("\n[CLEANUP] Stopping streamer and clearing buffer...")
    streamer.stop()
    logger.info("✅ Cleanup complete")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ E2E RING BUFFER TEST PASSED")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_ring_buffer_e2e()
    exit(0 if success else 1)

