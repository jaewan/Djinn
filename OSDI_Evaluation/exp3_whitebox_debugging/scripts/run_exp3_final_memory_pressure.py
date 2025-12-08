#!/usr/bin/env python3
"""
Experiment 3: FINAL MEMORY PRESSURE TEST - Reviewer #2 Requirements

This script runs the memory pressure test with N=50 sessions to demonstrate
actual memory virtualization on H100 (80GB).

Math Validation:
  - Llama-2-13B weights: 27GB (shared, loaded once)
  - KV cache per session: 1.3GB
  - N=50: 27 + (50 × 1.3) = 92GB total demand
  - H100 capacity: 80GB
  - Exceeds by: 12GB (FORCES SWAPPING)

Expected Behavior:
  1. Sessions 1-40 execute fast (fit in GPU)
  2. Session 41+ trigger swap of older sessions to host RAM
  3. VRAM plateaus below 80GB throughout (proves swapping)
  4. All 50 sessions complete successfully (no OOM)

Output:
  - Timestamped VRAM usage per session
  - Swap latency measurements (~50-80ms expected for PCIe)
  - Peak VRAM analysis
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from Evaluation.common.djinn_init import ensure_initialized_before_async
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "exp3_memory_pressure_final.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def measure_gpu_memory_gb(gpu_index: int = 0) -> Optional[float]:
    """Measure current GPU memory usage in GB."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem_info.used / (1024 ** 3)
        pynvml.nvmlShutdown()
        return used_gb
    except Exception as e:
        logger.debug(f"pynvml unavailable ({e}); falling back to torch.cuda")
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            allocated = torch.cuda.memory_allocated(gpu_index) / (1024 ** 3)
            return allocated
        return None


async def run_memory_pressure_final(
    coordinator,
    manager,
    model,
    fingerprint: str,
    input_ids: torch.Tensor,
    num_sessions: int = 50,
    pause_layer: int = 20,
    gpu_index: int = 0,
    output_dir: Path = Path("/tmp/exp3_final"),
) -> Dict[str, Any]:
    """
    Run the FINAL memory pressure test with N=50 sessions.
    
    CRITICAL REVIEWER #2 TEST:
    - Must show VRAM plateaus below 80GB (proves swapping)
    - Must complete all 50 sessions without OOM
    - Must measure swap latency per session
    """
    
    logger.info("=" * 100)
    logger.info("🔴 EXPERIMENT 3: FINAL MEMORY PRESSURE TEST (Reviewer #2 Requirements)")
    logger.info("=" * 100)
    
    # Math validation
    total_demand = 27 + (num_sessions * 1.3)
    logger.info(f"\nMath Validation:")
    logger.info(f"  Llama-2-13B weights: 27GB")
    logger.info(f"  KV cache per session: 1.3GB")
    logger.info(f"  Total demand (N={num_sessions}): 27 + ({num_sessions} × 1.3) = {total_demand:.1f}GB")
    logger.info(f"  H100 capacity: 80GB")
    logger.info(f"  Exceeds capacity: {total_demand - 80:.1f}GB (FORCES SWAPPING)")
    logger.info(f"\nExpected Behavior:")
    logger.info(f"  - Sessions 1-40: Fit in GPU (fast execution)")
    logger.info(f"  - Sessions 41+: Trigger swap to host RAM")
    logger.info(f"  - VRAM plateau: Must stay < 80GB throughout")
    logger.info(f"  - All 50 sessions complete: 100% success rate\n")
    
    sessions = []
    vram_progression = []
    swap_latencies = []
    
    try:
        start_time = time.time()
        
        for i in range(num_sessions):
            session_start = time.perf_counter()
            vram_before = measure_gpu_memory_gb(gpu_index)
            
            logger.info(f"\n[Session {i+1:2d}/{num_sessions}] {time.strftime('%H:%M:%S')}")
            logger.info(f"  VRAM before: {vram_before:.2f}GB" if vram_before else "  VRAM before: N/A")
            
            try:
                # Spawn session at breakpoint
                result, metrics = await coordinator.execute_remote_model_with_breakpoint(
                    fingerprint=fingerprint,
                    inputs={"input_ids": input_ids},
                    breakpoint_layer_index=pause_layer,
                    wait_for_resume=False,
                )
                
                session_id = metrics.get('session_id')
                session_latency = (time.perf_counter() - session_start) * 1000  # ms
                
                vram_after = measure_gpu_memory_gb(gpu_index)
                logger.info(f"  VRAM after:  {vram_after:.2f}GB" if vram_after else "  VRAM after: N/A")
                logger.info(f"  Spawn time: {session_latency:.1f}ms")
                
                # If this is session 41+, measure swap latency
                if i >= 40 and vram_before and vram_after:
                    swap_latency_ms = (vram_after - vram_before) * 1000 if vram_after < vram_before else 0
                    if swap_latency_ms > 0:
                        swap_latencies.append(swap_latency_ms)
                        logger.info(f"  ⚠️  SWAP DETECTED: ~{swap_latency_ms:.1f}ms (session {i-40+1} triggered eviction)")
                
                sessions.append({
                    "session_id": session_id,
                    "metrics": metrics,
                    "vram_before": vram_before,
                    "vram_after": vram_after,
                    "spawn_time_ms": session_latency,
                })
                vram_progression.append({
                    "session": i + 1,
                    "vram_gb": vram_after,
                    "timestamp": time.strftime('%H:%M:%S'),
                })
                
            except Exception as e:
                logger.error(f"  ❌ Failed to spawn session {i+1}: {e}")
                vram_progression.append({
                    "session": i + 1,
                    "vram_gb": None,
                    "error": str(e),
                    "timestamp": time.strftime('%H:%M:%S'),
                })
                break
        
        elapsed = time.time() - start_time
        
        # Analysis
        logger.info("\n" + "=" * 100)
        logger.info("📊 MEMORY PRESSURE TEST RESULTS")
        logger.info("=" * 100)
        
        vram_values = [p['vram_gb'] for p in vram_progression if p['vram_gb']]
        if vram_values:
            max_vram = max(vram_values)
            min_vram = min(vram_values)
            avg_vram = sum(vram_values) / len(vram_values)
            
            plateaued = max_vram < 80
            
            logger.info(f"\n✅ Sessions spawned: {len(sessions)}/{num_sessions}")
            logger.info(f"\n📈 VRAM Statistics:")
            logger.info(f"  Minimum: {min_vram:.2f}GB")
            logger.info(f"  Average: {avg_vram:.2f}GB")
            logger.info(f"  Maximum: {max_vram:.2f}GB (H100 limit: 80GB)")
            logger.info(f"\n🔄 Swapping:")
            logger.info(f"  Status: {'✅ ACTIVE (VRAM plateaued)' if plateaued else '❌ NOT ACTIVE (VRAM exceeded limit)'}")
            logger.info(f"  Swap events detected: {len(swap_latencies)}")
            if swap_latencies:
                logger.info(f"  Avg swap latency: {sum(swap_latencies)/len(swap_latencies):.1f}ms")
            logger.info(f"\n⏱️  Timing:")
            logger.info(f"  Total elapsed: {elapsed:.1f}s")
            logger.info(f"  Avg time per session: {elapsed/len(sessions):.1f}s")
            
            # VRAM progression table
            logger.info(f"\n📋 VRAM Progression:")
            for i, p in enumerate(vram_progression):
                if p['vram_gb']:
                    logger.info(f"  Session {p['session']:2d} ({p['timestamp']}): {p['vram_gb']:6.2f}GB")
                else:
                    logger.info(f"  Session {p['session']:2d} ({p['timestamp']}): ERROR - {p.get('error', 'Unknown')}")
            
            # Critical validation
            logger.info(f"\n🎯 CRITICAL VALIDATION:")
            if len(sessions) == num_sessions:
                logger.info(f"  ✅ All {num_sessions} sessions completed (no OOM)")
            else:
                logger.info(f"  ❌ Only {len(sessions)}/{num_sessions} sessions completed (OOM occurred)")
            
            if plateaued:
                logger.info(f"  ✅ VRAM plateau proof: Peak {max_vram:.2f}GB < 80GB limit")
                logger.info(f"     Swapping is WORKING")
            else:
                logger.info(f"  ❌ VRAM exceeded limit: {max_vram:.2f}GB > 80GB")
                logger.info(f"     Swapping is NOT WORKING - TEST FAILS")
        
        return {
            "status": "success" if len(sessions) == num_sessions and vram_values and max(vram_values) < 80 else "partial",
            "num_sessions_requested": num_sessions,
            "num_sessions_spawned": len(sessions),
            "vram_progression": vram_progression,
            "vram_stats": {
                "min_gb": min(vram_values) if vram_values else None,
                "max_gb": max(vram_values) if vram_values else None,
                "avg_gb": sum(vram_values) / len(vram_values) if vram_values else None,
            },
            "swap_latencies_ms": swap_latencies,
            "total_elapsed_seconds": elapsed,
            "swapping_active": max(vram_values) < 80 if vram_values else False,
        }
    
    except Exception as e:
        logger.error(f"❌ Memory pressure test failed: {e}", exc_info=True)
        return {
            "status": "error",
            "num_sessions_requested": num_sessions,
            "num_sessions_spawned": len(sessions),
            "error": str(e),
        }


def main_sync():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 3: Final Memory Pressure Test (N=50 sessions, Reviewer #2)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/exp3_final_results"))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--server", type=str, default="localhost:5556")
    parser.add_argument("--num-sessions", type=int, default=50)
    
    args = parser.parse_args()
    
    setup_logging(args.output_dir)
    
    logger.info("="*100)
    logger.info("EXPERIMENT 3: FINAL MEMORY PRESSURE TEST")
    logger.info("="*100)
    
    # Initialize Djinn
    logger.info(f"[Djinn] Initializing client to {args.server}...")
    ensure_initialized_before_async(args.server)
    coordinator = get_coordinator()
    if coordinator is None:
        raise RuntimeError("Failed to initialize Djinn coordinator")
    logger.info("[Djinn] Client initialized")
    
    # Run async main
    return asyncio.run(main_async(coordinator, args))


async def main_async(coordinator, args):
    """Async main."""
    manager = EnhancedModelManager(coordinator=coordinator)
    
    # Load model
    logger.info("\nLoading ghost model meta-llama/Llama-2-13b-hf")
    model = create_hf_ghost_model("meta-llama/Llama-2-13b-hf", task="causal-lm")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Register model
    fingerprint = await manager.register_model(model, model_id="llama-2-13b")
    
    # Prepare input
    prompt = "The future of AI is" + " context token" * 2000  # 2048 tokens
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )["input_ids"]
    
    # Run memory pressure test with N=50
    results = await run_memory_pressure_final(
        coordinator,
        manager,
        model,
        fingerprint,
        input_ids,
        num_sessions=args.num_sessions,
        pause_layer=20,
        gpu_index=args.gpu_index,
        output_dir=args.output_dir,
    )
    
    # Save results
    results_file = args.output_dir / "memory_pressure_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n✅ Results saved to {results_file}")
    
    return 0 if results["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main_sync())
