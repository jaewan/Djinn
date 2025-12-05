#!/usr/bin/env python3
"""
VRAM Monitor Sidecar: Capture GPU memory timeline during breakpoint experiment.

Runs in parallel with breakpoint experiment to generate the "money plot" (Figure 7):
Shows GPU memory usage over time, demonstrating VRAM dip during context switch.

Uses pynvml for high-precision (<1ms) monitoring (default 10ms interval).
Falls back to nvidia-smi if pynvml unavailable.

Usage:
    # Terminal 1 (10ms resolution by default)
    python scripts/monitor_vram.py \
        --output /tmp/exp3_vram.csv \
        --interval 0.01 \
        --duration 300
    
    # Terminal 2 (in parallel)
    python scripts/start_breakpoint.py
    python scripts/verify_vram_freed.py
    python scripts/resume_breakpoint.py

Output:
    CSV with columns: timestamp, gpu_memory_used_mb, gpu_memory_total_mb, utilization_percent
"""

import sys
import os
import argparse
import csv
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent dir to path to import common_utils
sys.path.insert(0, os.path.dirname(__file__))
from common_utils import query_vram, initialize_pynvml, shutdown_pynvml

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
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
    
    logger.info(f"Logging configured at {level}")


def monitor_vram(
    output_file: Path,
    interval_secs: float = 0.01,  # 10ms default (10x better resolution than before)
    duration_secs: float = 300.0,
    verbose: bool = False
) -> None:
    """
    Monitor VRAM and write to CSV with high precision.
    
    Args:
        output_file: Path to write CSV
        interval_secs: Polling interval (default 0.01s = 10ms for high resolution)
        duration_secs: Total monitoring duration
        verbose: Print each sample
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting VRAM monitor (pynvml-based, high precision)")
    logger.info(f"  Interval: {interval_secs*1000:.1f}ms")
    logger.info(f"  Duration: {duration_secs}s")
    logger.info(f"  Output: {output_file}")
    
    # Initialize pynvml if available
    initialize_pynvml()
    
    # CSV header
    fieldnames = [
        'timestamp',
        'datetime',
        'gpu_memory_used_mb',
        'gpu_memory_total_mb',
        'utilization_percent'
    ]
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()
            
            start_time = time.time()
            sample_count = 0
            last_error_logged = 0
            
            while True:
                elapsed = time.time() - start_time
                
                if elapsed > duration_secs:
                    logger.info(f"Monitoring complete. Collected {sample_count} samples at {1000*interval_secs:.1f}ms resolution.")
                    break
                
                metrics = query_vram()
                
                if metrics:
                    # Add timestamp
                    timestamp = time.time()
                    row = {
                        'timestamp': timestamp,
                        'datetime': datetime.utcnow().isoformat(),
                        'gpu_memory_used_mb': metrics.get('gpu_memory_used_mb', ''),
                        'gpu_memory_total_mb': metrics.get('gpu_memory_total_mb', ''),
                        'utilization_percent': metrics.get('utilization_percent', ''),
                    }
                    
                    writer.writerow(row)
                    csvfile.flush()
                    sample_count += 1
                    
                    if verbose and sample_count % 100 == 0:  # Log every 100 samples (~1s at 10ms)
                        logger.info(
                            f"[{elapsed:.1f}s] VRAM: {metrics.get('gpu_memory_used_mb', 0):.0f}MB / "
                            f"{metrics.get('gpu_memory_total_mb', 0):.0f}MB "
                            f"({metrics.get('utilization_percent', 0):.0f}%)"
                        )
                else:
                    # Log error but don't spam
                    current_time = time.time()
                    if current_time - last_error_logged > 5.0:
                        logger.warning("Failed to query VRAM metrics")
                        last_error_logged = current_time
                
                time.sleep(interval_secs)
    
    except KeyboardInterrupt:
        logger.info(f"Monitoring interrupted. Collected {sample_count} samples.")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}", exc_info=True)
        raise
    finally:
        shutdown_pynvml()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='VRAM Monitor: Capture GPU memory timeline during Djinn experiment'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=0.01,
        help='Polling interval in seconds (default: 0.01 = 10ms for high precision)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=300.0,
        help='Monitoring duration in seconds (default: 300)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print each sample'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.output.parent, args.log_level)
    
    logger.info("VRAM Monitor for Djinn Experiment 3")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Interval: {args.interval}s")
    logger.info(f"  Duration: {args.duration}s")
    logger.info("")
    
    try:
        monitor_vram(
            output_file=args.output,
            interval_secs=args.interval,
            duration_secs=args.duration,
            verbose=args.verbose
        )
        logger.info("✅ VRAM monitoring complete")
        return 0
    
    except Exception as e:
        logger.error(f"❌ VRAM monitoring failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
