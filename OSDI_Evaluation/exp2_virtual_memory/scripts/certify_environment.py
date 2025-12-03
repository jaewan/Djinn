#!/usr/bin/env python3
"""
Environment Certification Script for Experiment 2

Validates hardware and software prerequisites for Ring Buffer experiments:
- GPU memory and capabilities
- NUMA topology
- Pinned memory limits
- PCIe bandwidth capability
- PyTorch configuration

Usage:
    python certify_environment.py --verbose
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_cuda_availability() -> bool:
    """Check if CUDA is available."""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    logger.info(f"✅ CUDA available: {torch.version.cuda}")
    return True


def check_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    results = {}
    
    try:
        device_count = torch.cuda.device_count()
        logger.info(f"✅ GPU count: {device_count}")
        results["gpu_count"] = device_count
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            logger.info(f"   GPU {i}: {props.name}, {total_memory_gb:.1f}GB")
            
            results[f"gpu_{i}_name"] = props.name
            results[f"gpu_{i}_memory_gb"] = total_memory_gb
        
        return results
    except Exception as e:
        logger.error(f"❌ Failed to get GPU info: {e}")
        return {}


def check_free_memory(device: int = 0, min_free_gb: float = 6.0) -> bool:
    """Check free GPU memory."""
    try:
        props = torch.cuda.get_device_properties(device)
        reserved = torch.cuda.memory_reserved(device)
        total = props.total_memory
        free_gb = (total - reserved) / (1024**3)
        
        logger.info(f"✅ Free GPU memory: {free_gb:.1f}GB (required: {min_free_gb:.1f}GB)")
        
        if free_gb >= min_free_gb:
            return True
        else:
            logger.warning(f"⚠️  Free memory {free_gb:.1f}GB < required {min_free_gb:.1f}GB")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to check free memory: {e}")
        return False


def check_pytorch_config() -> Dict[str, Any]:
    """Check PyTorch configuration."""
    results = {}
    
    # Check deterministic behavior
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        logger.info("✅ CUBLAS workspace configured")
        results["cublas_configured"] = True
    except Exception as e:
        logger.warning(f"⚠️  Could not configure CUBLAS: {e}")
        results["cublas_configured"] = False
    
    # Check memory growth
    try:
        torch.cuda.empty_cache()
        logger.info("✅ CUDA cache can be cleared")
        results["cache_clearable"] = True
    except Exception as e:
        logger.warning(f"⚠️  Could not clear CUDA cache: {e}")
        results["cache_clearable"] = False
    
    return results


def check_pinned_memory() -> Dict[str, Any]:
    """Check pinned memory configuration."""
    results = {}
    
    try:
        # Try to allocate pinned memory
        test_size_mb = 256
        test_tensor = torch.cuda.FloatTensor(test_size_mb * 1024 * 1024 // 4)
        pinned_tensor = torch.randn(test_size_mb * 1024 * 1024 // 4, 
                                     pin_memory=True, device='cpu')
        
        logger.info(f"✅ Pinned memory allocation works ({test_size_mb}MB test)")
        results["pinned_memory_available"] = True
        
        # Clean up
        del test_tensor
        del pinned_tensor
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        if "not enough memory" in str(e).lower():
            logger.warning(f"⚠️  Insufficient pinned memory: {e}")
            results["pinned_memory_available"] = False
        else:
            logger.warning(f"⚠️  Pinned memory error: {e}")
            results["pinned_memory_available"] = False
    
    return results


def check_numa_topology() -> Dict[str, Any]:
    """Check NUMA topology."""
    results = {}
    
    try:
        # Try to get NUMA info via lscpu
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        output = result.stdout
        
        # Extract NUMA info
        for line in output.split('\n'):
            if 'NUMA node' in line:
                logger.info(f"   {line.strip()}")
        
        if "NUMA node" in output:
            logger.info("✅ NUMA topology detected")
            results["numa_available"] = True
        else:
            logger.warning("⚠️  No NUMA topology detected")
            results["numa_available"] = False
            
    except FileNotFoundError:
        logger.warning("⚠️  lscpu not available for NUMA detection")
        results["numa_available"] = None
    except Exception as e:
        logger.warning(f"⚠️  Could not check NUMA topology: {e}")
        results["numa_available"] = None
    
    return results


def check_pcie_bandwidth() -> Dict[str, Any]:
    """Check PCIe bandwidth capability."""
    results = {}
    
    try:
        # Try simple H2D transfer test
        size_mb = 256
        host_tensor = torch.randn(size_mb * 1024 * 1024 // 4, pin_memory=True)
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        device_tensor = host_tensor.cuda()
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        
        bandwidth_gbps = (size_mb / (elapsed_ms / 1000.0)) / 1024
        logger.info(f"✅ H2D bandwidth test: {bandwidth_gbps:.1f} GB/s ({size_mb}MB transfer)")
        
        results["h2d_bandwidth_gbps"] = bandwidth_gbps
        
        # Clean up
        del host_tensor
        del device_tensor
        torch.cuda.empty_cache()
        
        if bandwidth_gbps > 10.0:
            logger.info(f"   PCIe bandwidth looks healthy (>{10.0}GB/s)")
            results["bandwidth_sufficient"] = True
        else:
            logger.warning(f"   PCIe bandwidth low (<10GB/s), expect degraded performance")
            results["bandwidth_sufficient"] = False
        
    except Exception as e:
        logger.error(f"❌ PCIe bandwidth test failed: {e}")
        results["h2d_bandwidth_gbps"] = 0.0
        results["bandwidth_sufficient"] = False
    
    return results


def check_model_loading(small_model_id: str = "gpt2") -> bool:
    """Test model loading capability."""
    try:
        from transformers import AutoModelForCausalLM
        
        logger.info(f"Testing model loading with {small_model_id}...")
        model = AutoModelForCausalLM.from_pretrained(small_model_id, torch_dtype=torch.float16)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model loading works ({num_params/1e9:.2f}B parameters)")
        
        del model
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Certify environment for Experiment 2")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-model-test", action="store_true", help="Skip model loading test")
    parser.add_argument("--min-free-gb", type=float, default=6.0, help="Minimum free GPU memory")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("ENVIRONMENT CERTIFICATION FOR EXPERIMENT 2")
    logger.info("=" * 70)
    logger.info("")
    
    checks = {
        "cuda": False,
        "gpu_info": False,
        "free_memory": False,
        "pytorch": False,
        "pinned_memory": False,
        "numa": False,
        "pcie": False,
        "model_loading": False,
    }
    
    # Run checks
    logger.info("[1/8] Checking CUDA availability...")
    checks["cuda"] = check_cuda_availability()
    logger.info("")
    
    logger.info("[2/8] Getting GPU information...")
    gpu_info = check_gpu_info()
    checks["gpu_info"] = bool(gpu_info)
    logger.info("")
    
    logger.info("[3/8] Checking free GPU memory...")
    checks["free_memory"] = check_free_memory(min_free_gb=args.min_free_gb)
    logger.info("")
    
    logger.info("[4/8] Checking PyTorch configuration...")
    pytorch_config = check_pytorch_config()
    checks["pytorch"] = all(pytorch_config.values())
    logger.info("")
    
    logger.info("[5/8] Checking pinned memory...")
    pinned_info = check_pinned_memory()
    checks["pinned_memory"] = pinned_info.get("pinned_memory_available", False)
    logger.info("")
    
    logger.info("[6/8] Checking NUMA topology...")
    numa_info = check_numa_topology()
    checks["numa"] = numa_info.get("numa_available", True)  # Optional, not critical
    logger.info("")
    
    logger.info("[7/8] Testing PCIe bandwidth...")
    pcie_info = check_pcie_bandwidth()
    checks["pcie"] = pcie_info.get("bandwidth_sufficient", False)
    logger.info("")
    
    if not args.skip_model_test:
        logger.info("[8/8] Testing model loading...")
        checks["model_loading"] = check_model_loading()
        logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info("CERTIFICATION SUMMARY")
    logger.info("=" * 70)
    
    critical_checks = ["cuda", "gpu_info", "free_memory", "pytorch"]
    recommended_checks = ["pinned_memory", "pcie", "model_loading"]
    
    critical_pass = all(checks[c] for c in critical_checks)
    recommended_pass = all(checks[c] for c in recommended_checks if c in checks and checks[c] is not None)
    
    for check, passed in checks.items():
        if passed is None:
            symbol = "⊘"
        else:
            symbol = "✅" if passed else "❌"
        logger.info(f"{symbol} {check}: {passed}")
    
    logger.info("")
    
    if critical_pass:
        logger.info("✅ CRITICAL CHECKS PASSED - Environment ready for Experiment 2")
        if recommended_pass:
            logger.info("✅ RECOMMENDED CHECKS PASSED - Optimal performance expected")
        else:
            logger.warning("⚠️  Some recommended checks failed - Performance may be degraded")
        return 0
    else:
        logger.error("❌ CRITICAL CHECKS FAILED - Environment not ready")
        return 1


if __name__ == "__main__":
    sys.exit(main())

