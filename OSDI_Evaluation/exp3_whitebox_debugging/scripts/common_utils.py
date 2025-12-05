#!/usr/bin/env python3
"""
Shared utilities for Experiment 3 scripts.

Provides:
- VRAM monitoring via pynvml (direct API, <1ms)
- Logging setup (console + file)
- Device detection
- Statistics computation
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Try to import pynvml, fallback to nvidia-smi if not available
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, will fall back to nvidia-smi")

import torch


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure logging to console and file."""
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


def query_vram_pynvml() -> Optional[Dict[str, float]]:
    """
    Query GPU memory using pynvml (direct API, <1ms).
    
    Returns:
        Dict with keys: gpu_memory_used_mb, gpu_memory_total_mb, utilization_percent
        Or None if query fails
    """
    if not PYNVML_AVAILABLE:
        return None
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        return {
            'gpu_memory_used_mb': mem_info.used / 1024 / 1024,
            'gpu_memory_total_mb': mem_info.total / 1024 / 1024,
            'utilization_percent': util_info.gpu
        }
    except Exception as e:
        logger.error(f"pynvml query failed: {e}")
        return None


def query_vram_nvidia_smi() -> Optional[Dict[str, float]]:
    """
    Query GPU memory using nvidia-smi (fallback).
    
    Returns:
        Dict with keys: gpu_memory_used_mb, gpu_memory_total_mb, utilization_percent
        Or None if query fails
    """
    import subprocess
    
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits',
            '-i', '0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return None
        
        parts = result.stdout.strip().split(', ')
        if len(parts) < 3:
            return None
        
        return {
            'gpu_memory_used_mb': float(parts[0]),
            'gpu_memory_total_mb': float(parts[1]),
            'utilization_percent': float(parts[2])
        }
    except Exception as e:
        logger.error(f"nvidia-smi query failed: {e}")
        return None


def query_vram() -> Optional[Dict[str, float]]:
    """
    Query GPU memory (pynvml preferred, fallback to nvidia-smi).
    
    Returns:
        Dict with gpu_memory_used_mb, gpu_memory_total_mb, utilization_percent
    """
    if PYNVML_AVAILABLE:
        result = query_vram_pynvml()
        if result:
            return result
    
    return query_vram_nvidia_smi()


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive statistics.
    
    Args:
        values: List of measurements
    
    Returns:
        Dict with: mean, std, min, max, ci_lower, ci_upper (95%)
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'count': 0,
        }
    
    values_array = np.array(values)
    mean = float(np.mean(values_array))
    std = float(np.std(values_array, ddof=1) if len(values) > 1 else 0.0)
    
    # 95% confidence interval
    if len(values) > 1:
        sem = scipy_stats.sem(values_array)  # Standard error of mean
        margin = sem * scipy_stats.t.ppf(0.975, len(values) - 1)  # t-value for 95%
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = mean
        ci_upper = mean
    
    return {
        'mean': mean,
        'std': std,
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'count': len(values),
    }


def format_statistics(stats: Dict[str, float]) -> str:
    """Format statistics dict for logging."""
    return (
        f"mean={stats['mean']:.2f}ms, "
        f"std={stats['std']:.2f}ms, "
        f"min={stats['min']:.2f}ms, "
        f"max={stats['max']:.2f}ms, "
        f"CI=[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]ms (95%)"
    )


def initialize_pynvml() -> None:
    """Initialize pynvml (if available)."""
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            logger.info("pynvml initialized (high-precision VRAM monitoring enabled)")
        except Exception as e:
            logger.warning(f"pynvml initialization failed: {e}, will use nvidia-smi")


def shutdown_pynvml() -> None:
    """Shutdown pynvml (if available)."""
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
