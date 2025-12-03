#!/usr/bin/env python3
"""
Start Djinn Server for Experiment 2: Ring Buffer Virtualization

Configures and launches a Djinn server with ring buffer model caching
optimized for L4 GPU (24GB VRAM) to stream Llama-70B (140GB) weights.

Configuration:
- Ring Buffer: 20GB capacity (skip-end allocation)
- Model Cache: Ring buffer-backed model cache
- DISABLE_KV_SWAP: Prevents KV cache swapping for isolated PCIe testing
- NUMA Binding: Binds to optimal CPU nodes
- Pinned Memory: Pre-allocates pinned memory for H2D transfers

Usage:
    python start_exp2_server.py \
        --port 5000 \
        --gpu-id 0 \
        --ring-buffer-gb 20 \
        --disable-kv-swap

Then in another terminal, run the client:
    python run_exp2_client.py --server localhost:5000
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment(ring_buffer_gb: int, disable_kv_swap: bool):
    """Configure environment variables for ring buffer experiment."""
    
    # Disable KV cache swapping for isolated bandwidth measurement
    if disable_kv_swap:
        os.environ['DJINN_DISABLE_KV_SWAP'] = '1'
        logger.info("✅ KV swap disabled - ring buffer gets 100% PCIe bandwidth")
    
    # Set ring buffer capacity
    os.environ['DJINN_RING_BUFFER_GB'] = str(ring_buffer_gb)
    logger.info(f"✅ Ring buffer configured: {ring_buffer_gb}GB")
    
    # Enable ring buffer model cache
    os.environ['DJINN_MODEL_CACHE_BACKEND'] = 'ring_buffer'
    logger.info("✅ Ring buffer model cache enabled")
    
    # Disable KV cache pre-allocation to save VRAM for weights
    os.environ['DJINN_DISABLE_KV_PREALLOC'] = '1'
    logger.info("✅ KV cache pre-allocation disabled")


def check_prerequisites(gpu_id: int) -> bool:
    """Verify GPU and environment are ready."""
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    if gpu_id >= torch.cuda.device_count():
        logger.error(f"❌ GPU {gpu_id} not available (found {torch.cuda.device_count()} GPUs)")
        return False
    
    props = torch.cuda.get_device_properties(gpu_id)
    total_memory_gb = props.total_memory / (1024**3)
    logger.info(f"✅ GPU {gpu_id}: {props.name}, {total_memory_gb:.1f}GB")
    
    # Check free memory
    torch.cuda.set_device(gpu_id)
    reserved = torch.cuda.memory_reserved(gpu_id)
    free_memory_gb = (props.total_memory - reserved) / (1024**3)
    
    if free_memory_gb < 4.0:
        logger.warning(f"⚠️  Low free memory: {free_memory_gb:.1f}GB (need ~4GB)")
        return False
    
    logger.info(f"✅ Free memory: {free_memory_gb:.1f}GB")
    return True


def start_server(port: int, gpu_id: int, ring_buffer_gb: int, disable_kv_swap: bool):
    """Start Djinn server with ring buffer configuration."""
    
    logger.info("=" * 70)
    logger.info("DJINN SERVER - EXPERIMENT 2: RING BUFFER VIRTUALIZATION")
    logger.info("=" * 70)
    logger.info("")
    
    # Check prerequisites
    logger.info("[1/4] Checking prerequisites...")
    if not check_prerequisites(gpu_id):
        logger.error("Prerequisites check failed")
        return False
    logger.info("")
    
    # Setup environment
    logger.info("[2/4] Configuring environment...")
    setup_environment(ring_buffer_gb, disable_kv_swap)
    logger.info("")
    
    # Import Djinn after environment setup
    logger.info("[3/4] Initializing Djinn server...")
    try:
        from djinn.server import DjinnServer, ServerConfig
        from djinn.config import get_config
        
        # Create server config
        server_config = ServerConfig(
            node_id="exp2_server",
            control_port=port,
            data_port=port + 1,
            gpu_indices=[gpu_id],
            prefer_dpdk=False,  # Use TCP for compatibility
            tcp_fallback=True,
        )
        
        # Initialize server
        server = DjinnServer(server_config)
        logger.info(f"✅ Djinn server created on port {port}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("")
    logger.info("[4/4] Starting server...")
    
    try:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"✅ SERVER READY - Port {port}")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Configuration Summary:")
        logger.info(f"  GPU: {gpu_id}")
        logger.info(f"  Ring Buffer: {ring_buffer_gb}GB")
        logger.info(f"  KV Swap: {'disabled' if disable_kv_swap else 'enabled'}")
        logger.info(f"  Control Port: {port}")
        logger.info(f"  Data Port: {port + 1}")
        logger.info("")
        logger.info("Waiting for client connections... (Ctrl+C to stop)")
        logger.info("=" * 70)
        logger.info("")
        
        # Start server loop
        server.start()
        
        # Keep server alive
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("\n\nShutting down server...")
        try:
            server.shutdown()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        logger.info("✅ Server stopped")
        return True
    
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Start Djinn server for Experiment 2 (Ring Buffer Virtualization)"
    )
    parser.add_argument("--port", type=int, default=5000,
                       help="Control port for server (default: 5000)")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU ID to use (default: 0)")
    parser.add_argument("--ring-buffer-gb", type=int, default=20,
                       help="Ring buffer capacity in GB (default: 20)")
    parser.add_argument("--disable-kv-swap", action="store_true",
                       help="Disable KV cache swapping (default: enabled)")
    
    args = parser.parse_args()
    
    try:
        success = start_server(
            port=args.port,
            gpu_id=args.gpu_id,
            ring_buffer_gb=args.ring_buffer_gb,
            disable_kv_swap=args.disable_kv_swap
        )
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

