#!/usr/bin/env python3
"""
Pinned Memory Bandwidth Benchmark

Validates that Host-to-Device (H2D) transfer bandwidth meets expectations:
- Pinned memory: Target >22 GB/s (PCIe Gen4 saturation)
- Pageable memory: Expected ~4 GB/s (slow path)

This is critical for Experiment 2: if bandwidth is low, ring buffer prefetching
will be bottlenecked and async pipelining won't hide latency.

Usage:
    python benchmarks/shm_bandwidth.py [--size_mb SIZE] [--device DEVICE]
"""

import argparse
import logging
import time
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def measure_bandwidth(
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
    device: torch.device = torch.device('cuda:0')
) -> float:
    """
    Measure bandwidth for copying src_tensor to dst_tensor.
    
    Args:
        src_tensor: Source tensor (CPU)
        dst_tensor: Destination tensor (GPU)
        num_iterations: Number of measurement iterations
        warmup_iterations: Number of warmup iterations
        device: GPU device
    
    Returns:
        Bandwidth in GB/s
    """
    # Clear cache once before all warmup iterations (expensive syscall, only call once)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Warmup (no cache clearing between iterations)
    for _ in range(warmup_iterations):
        dst_tensor.copy_(src_tensor, non_blocking=True)
        torch.cuda.synchronize()
    
    # Measure (cache already cleared above)
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        dst_tensor.copy_(src_tensor, non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Calculate bandwidth
    size_bytes = src_tensor.numel() * src_tensor.element_size()
    avg_time = np.mean(times[1:])  # Skip first measurement
    bandwidth_gbs = (size_bytes / avg_time) / (1024**3)
    
    return bandwidth_gbs


def benchmark_pinned_vs_pageable(size_mb: int = 1000, device_id: int = 0):
    """Run benchmark comparing pinned vs pageable memory."""
    
    device = torch.device(f'cuda:{device_id}')
    
    logger.info("=" * 80)
    logger.info("Pinned Memory Bandwidth Benchmark")
    logger.info("=" * 80)
    logger.info(f"GPU: {torch.cuda.get_device_name(device_id)}")
    logger.info(f"Transfer size: {size_mb} MB")
    logger.info("")
    
    size_bytes = size_mb * 1024 * 1024
    num_elements = size_bytes // 4  # float32
    
    # Test 1: Pinned memory (should be fast)
    logger.info("Test 1: Pinned Memory H2D Transfer")
    logger.info("-" * 80)
    
    try:
        src_pinned = torch.zeros(num_elements, dtype=torch.float32, pin_memory=True)
        dst_gpu = torch.zeros(num_elements, dtype=torch.float32, device=device)
        
        bandwidth_pinned = measure_bandwidth(src_pinned, dst_gpu, device=device)
        
        logger.info(f"✅ Pinned bandwidth: {bandwidth_pinned:.1f} GB/s")
        if bandwidth_pinned > 22.0:
            logger.info(f"   ✅ PASS: Excellent (>22 GB/s, saturating PCIe Gen4)")
        elif bandwidth_pinned > 15.0:
            logger.info(f"   ⚠️  PASS: Acceptable (15-22 GB/s)")
        else:
            logger.warning(f"   ❌ FAIL: Low bandwidth (<15 GB/s)")
        
        del src_pinned, dst_gpu
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"❌ Pinned memory test failed: {e}")
        bandwidth_pinned = None
    
    logger.info("")
    
    # Test 2: Pageable memory (should be slow)
    logger.info("Test 2: Pageable Memory H2D Transfer")
    logger.info("-" * 80)
    
    try:
        src_pageable = torch.zeros(num_elements, dtype=torch.float32, pin_memory=False)
        dst_gpu = torch.zeros(num_elements, dtype=torch.float32, device=device)
        
        bandwidth_pageable = measure_bandwidth(src_pageable, dst_gpu, device=device)
        
        logger.info(f"✅ Pageable bandwidth: {bandwidth_pageable:.1f} GB/s")
        if bandwidth_pageable > 4.0:
            logger.info(f"   ✅ PASS: Expected (>4 GB/s)")
        else:
            logger.warning(f"   ⚠️  WARN: Very slow (<4 GB/s, may indicate OS issues)")
        
        del src_pageable, dst_gpu
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"❌ Pageable memory test failed: {e}")
        bandwidth_pageable = None
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    if bandwidth_pinned and bandwidth_pageable:
        speedup = bandwidth_pinned / bandwidth_pageable
        logger.info(f"Pinned vs Pageable speedup: {speedup:.1f}x")
    
    logger.info("")
    logger.info("Recommendations for Experiment 2:")
    logger.info("-" * 80)
    
    if bandwidth_pinned is None:
        logger.error("❌ Pinned memory test failed - ring buffer won't work")
        return False
    elif bandwidth_pinned < 15.0:
        logger.warning("⚠️  Low PCIe bandwidth - consider NUMA binding")
        logger.warning("   Run: numactl --cpunodebind=0 --membind=0 python ...")
        return False
    else:
        logger.info("✅ System ready for Experiment 2")
        logger.info("   Pinned bandwidth sufficient for async pipelining")
        return True


def main():
    parser = argparse.ArgumentParser(description="Pinned memory bandwidth benchmark")
    parser.add_argument('--size-mb', type=int, default=1000, help='Transfer size in MB')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    success = benchmark_pinned_vs_pageable(size_mb=args.size_mb, device_id=args.device)
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

