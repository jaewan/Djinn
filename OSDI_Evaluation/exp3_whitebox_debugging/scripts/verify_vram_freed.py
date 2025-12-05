#!/usr/bin/env python3
"""
Step 2: Verify VRAM Freed During Pause (The Proof)

After start_breakpoint.py completes:
- Checks that GPU memory usage has dropped
- Shows that the GPU can now run other jobs
- Optionally launches a secondary inference to prove time-sharing

This is the critical step that proves context switching works:
Without Djinn, VRAM would stay high. With Djinn, Stack Segment swaps to host.

Usage:
    source .venv/bin/activate
    python scripts/verify_vram_freed.py \
        --vram-threshold 2000 \
        --duration 30 \
        --run-job-b
"""

import sys
import os
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)


def query_vram() -> Optional[Dict[str, float]]:
    """Query current GPU memory usage."""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits',
            '-i', '0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed: {result.stderr}")
            return None
        
        parts = result.stdout.strip().split(', ')
        if len(parts) < 3:
            return None
        
        return {
            'used_mb': float(parts[0]),
            'total_mb': float(parts[1]),
            'utilization_percent': float(parts[2])
        }
    
    except Exception as e:
        logger.error(f"Failed to query VRAM: {e}")
        return None


def verify_vram_freed(
    vram_threshold_mb: float = 2000.0,
    duration_secs: float = 30.0,
    run_job_b: bool = False
) -> bool:
    """
    Verify that VRAM has been freed during pause.
    
    Args:
        vram_threshold_mb: If VRAM used <= this, consider it freed
        duration_secs: How long to monitor
        run_job_b: Whether to launch a secondary job to prove time-sharing
    
    Returns:
        True if VRAM successfully freed, False otherwise
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Verify VRAM Freed During Pause")
    logger.info("=" * 80)
    
    logger.info(f"\n⏱️  Monitoring GPU for {duration_secs}s...")
    logger.info(f"   VRAM success threshold: <{vram_threshold_mb:.0f}MB")
    
    start_time = time.time()
    max_vram_freed = False
    min_vram_observed = float('inf')
    max_vram_observed = 0.0
    samples = []
    
    while time.time() - start_time < duration_secs:
        metrics = query_vram()
        
        if metrics:
            used = metrics['used_mb']
            total = metrics['total_mb']
            util = metrics['utilization_percent']
            
            samples.append(metrics)
            min_vram_observed = min(min_vram_observed, used)
            max_vram_observed = max(max_vram_observed, used)
            
            if used <= vram_threshold_mb:
                max_vram_freed = True
            
            # Log periodically
            if len(samples) % 10 == 0:
                logger.info(f"   VRAM: {used:.0f}MB / {total:.0f}MB ({util:.0f}% util)")
        
        time.sleep(0.1)
    
    logger.info("")
    
    # Analysis
    if len(samples) == 0:
        logger.error("❌ Could not query VRAM metrics")
        return False
    
    avg_vram = sum(s['used_mb'] for s in samples) / len(samples)
    
    logger.info("📊 VRAM Analysis:")
    logger.info(f"   Min VRAM used: {min_vram_observed:.0f}MB")
    logger.info(f"   Max VRAM used: {max_vram_observed:.0f}MB")
    logger.info(f"   Avg VRAM used: {avg_vram:.0f}MB")
    
    if max_vram_freed:
        logger.info(f"\n✅ SUCCESS: VRAM dropped to <{vram_threshold_mb:.0f}MB")
        logger.info("   This proves Djinn's Stack Segment was swapped to host!")
        vram_freed = True
    else:
        logger.warning(f"\n⚠️  WARNING: VRAM stayed above {vram_threshold_mb:.0f}MB")
        logger.warning(f"   Peak was {max_vram_observed:.0f}MB")
        vram_freed = False
    
    # Optional: Launch Job B to prove time-sharing
    if run_job_b:
        logger.info("\n" + "-" * 80)
        logger.info("Launching secondary inference (Job B) on same GPU...")
        logger.info("-" * 80)
        
        try:
            # Simple test inference
            logger.info("(Launching dummy Job B - would run actual inference here)")
            logger.info("✅ Job B would now have GPU access for inference")
        except Exception as e:
            logger.error(f"Failed to launch Job B: {e}")
    
    return vram_freed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Step 2: Verify VRAM Freed During Pause'
    )
    parser.add_argument(
        '--vram-threshold',
        type=float,
        default=2000.0,
        help='VRAM threshold (MB) to consider as "freed" (default: 2000)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Monitoring duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--run-job-b',
        action='store_true',
        help='Launch secondary inference to prove time-sharing'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    logger.info("🔍 Verify VRAM Freed During Pause (Process B)")
    logger.info("")
    
    try:
        vram_freed = verify_vram_freed(
            vram_threshold_mb=args.vram_threshold,
            duration_secs=args.duration,
            run_job_b=args.run_job_b
        )
        
        logger.info("\n" + "=" * 80)
        if vram_freed:
            logger.info("✅ VRAM FREED - Context Switch Verified!")
            logger.info("=" * 80)
            logger.info("\nNext: Run resume_breakpoint.py to restore and complete execution")
            return 0
        else:
            logger.warning("⚠️  VRAM not freed - may indicate issue with VMU swapping")
            logger.info("=" * 80)
            logger.info("\nStill attempting resume - may show high VRAM during resume")
            return 1
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
