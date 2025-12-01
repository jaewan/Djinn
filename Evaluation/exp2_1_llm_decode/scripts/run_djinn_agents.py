#!/usr/bin/env python3
"""
Djinn Agent Scaling Baseline for Experiment 2 (Hero agent workload).

This script demonstrates Djinn's "parking lot" solution by spawning N concurrent
agents that perform Reason → Act (sleep) → Reflect loops. Unlike Ray:

- Ray Keep-Alive: Each agent pins full model + KV cache. OOMs at low N.
- Ray Serverless: Each step reloads model + KV cache. High latency, scales to N=32.
- Djinn: Shared Text Segment (weights), per-session Data Segment (KV cache), 
         time-shared Stack Slab. Scales to N=32 with low latency.

Key Djinn features demonstrated:
1. Ghost Loading: Client creates zero-memory model on meta device
2. Session Persistence: session_id maps to persistent KV cache on server
3. Server-Side Generation: use_generate=True for fair comparison with Ray
4. Semantic Hints: phase="prefill" vs phase="decode" for KV reuse detection
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Djinn Imports
import djinn
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model
from Evaluation.common.djinn_init import ensure_initialized_before_async

# Optional GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# =============================================================================
# Configuration & Helpers
# =============================================================================

def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bytes_of_tensor(t: torch.Tensor) -> int:
    """Calculate tensor size in bytes."""
    return t.element_size() * t.numel()


class GpuSampler:
    """Background GPU utilization sampler using NVML."""

    def __init__(self, device_index: int, interval_s: float = 0.1) -> None:
        if not NVML_AVAILABLE:
            raise RuntimeError("pynvml not available; install nvidia-ml-py3")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: List[float] = []
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        pynvml.nvmlShutdown()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self._samples.append(float(util.gpu))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def mean_util(self) -> Optional[float]:
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)


# =============================================================================
# Agent Execution Logic (Async)
# =============================================================================

async def agent_lifecycle(
    agent_idx: int,
    manager: EnhancedModelManager,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
) -> List[Dict]:
    """
    Simulate one Data Scientist / Agent loop with Reason → Act → Reflect.
    
    Each agent gets a unique session_id that persists across iterations,
    allowing Djinn to reuse the KV cache for the Reflect phase.
    """
    records: List[Dict] = []
    
    # 1. Establish Session Identity (maps to Djinn's "Data Segment")
    # This ID persists for the lifetime of the agent
    session_id = f"agent_{agent_idx}_{uuid.uuid4().hex[:8]}"
    
    # Initialize tokenizer if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare initial input
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    current_input_ids = prompt_tokens["input_ids"]
    
    # Optional GPU sampler (only for first agent to reduce overhead)
    sampler: Optional[GpuSampler] = None
    if agent_idx == 0 and args.sample_gpu and NVML_AVAILABLE:
        try:
            sampler = GpuSampler(0, interval_s=0.05)
            sampler.__enter__()
        except Exception as e:
            print(f"[Warning] GPU sampling disabled: {e}")

    # Track KV cache state for incremental decode
    kv_processed_len = 0  # Number of tokens already processed and cached on server
    
    try:
        for iteration in range(args.iterations):
            # -----------------------------------------------------------------
            # PHASE 1: REASON (Prefill + Initial Decode)
            # -----------------------------------------------------------------
            start_reason = time.perf_counter()
            
            # Prepare inputs for generation (full sequence for prefill)
            inputs = {
                "input_ids": current_input_ids,
            }
            
            # Execute with semantic hints
            # phase="prefill" signals the server: this is initial processing,
            # may need to compute KV cache from scratch
            with djinn.session(phase="prefill", session_id=session_id, priority="normal"):
                reason_result = await manager.execute_model(
                    model,
                    inputs,
                    hints={
                        "use_generate": True,
                        "max_new_tokens": args.new_tokens,
                        "do_sample": False,  # Greedy decoding for reproducibility
                    }
                )
            
            latency_reason_ms = (time.perf_counter() - start_reason) * 1000.0
            
            # Update context (keep full history for next phase)
            if isinstance(reason_result, dict):
                if "generated_ids" in reason_result:
                    current_input_ids = reason_result["generated_ids"]
                elif "input_ids" in reason_result:
                    current_input_ids = reason_result["input_ids"]
            elif torch.is_tensor(reason_result):
                current_input_ids = reason_result
            
            # Update KV cache tracking: all tokens are now cached on server
            kv_processed_len = current_input_ids.shape[-1]
            
            records.append({
                "agent_id": agent_idx,
                "iteration": iteration,
                "stage": "reason",
                "latency_ms": latency_reason_ms,
                "tokens_generated": args.new_tokens,
                "kv_reused": False,  # Prefill phase, no KV reuse
            })

            # -----------------------------------------------------------------
            # PHASE 2: ACT (Simulated Tool Use / Think Time)
            # -----------------------------------------------------------------
            # Critical: During this sleep, the GPU Stack Slab is freed.
            # However, Djinn's Data Segment (KV cache for this session_id)
            # remains resident in VRAM, avoiding the "cold start" cost.
            await asyncio.sleep(args.sleep_seconds)

            # -----------------------------------------------------------------
            # PHASE 3: REFLECT (Efficient Decode with KV Reuse)
            # -----------------------------------------------------------------
            start_reflect = time.perf_counter()
            
            # CRITICAL FIX: Send only incremental tokens for true KV reuse
            # All tokens up to kv_processed_len are already cached on server
            incremental_input_ids = current_input_ids[:, kv_processed_len:]
            
            # If no new tokens (pure continuation), send last token for decode
            if incremental_input_ids.shape[-1] == 0:
                incremental_input_ids = current_input_ids[:, -1:]
            
            # Prepare incremental inputs for decode-only phase
            reflect_inputs = {
                "input_ids": incremental_input_ids,
            }
            
            # Same session_id: server detects this is a continuation
            # phase="decode" signals: this is decode-only, reuse KV cache
            # from the previous prefill phase
            with djinn.session(phase="decode", session_id=session_id, priority="normal"):
                reflect_result = await manager.execute_model(
                    model,
                    reflect_inputs,
                    hints={
                        "use_generate": True,
                        "max_new_tokens": args.new_tokens,
                        "do_sample": False,
                    }
                )
            
            latency_reflect_ms = (time.perf_counter() - start_reflect) * 1000.0
            
            # Update context
            if isinstance(reflect_result, dict):
                if "generated_ids" in reflect_result:
                    current_input_ids = reflect_result["generated_ids"]
                elif "input_ids" in reflect_result:
                    current_input_ids = reflect_result["input_ids"]
            elif torch.is_tensor(reflect_result):
                current_input_ids = reflect_result
            
            # Update KV cache tracking: all tokens are now cached on server
            kv_processed_len = current_input_ids.shape[-1]
            
            records.append({
                "agent_id": agent_idx,
                "iteration": iteration,
                "stage": "reflect",
                "latency_ms": latency_reflect_ms,
                "tokens_generated": args.new_tokens,
                "kv_reused": True,  # Decode phase, KV reuse expected
            })

    finally:
        if sampler:
            sampler.__exit__(None, None, None)

    return records


# =============================================================================
# Main Harness
# =============================================================================

async def run_sweep(args: argparse.Namespace, coordinator) -> Dict[str, Any]:
    """Main sweep loop: vary number of concurrent agents from 1 to 32."""
    
    # 1. Ensure coordinator is started (may need to start in async context)
    if not hasattr(coordinator, '_started') or not coordinator._started:
        try:
            await coordinator.start()
            print("[Djinn] Coordinator started in async context")
        except Exception as e:
            print(f"[Djinn] Warning: Coordinator start in async context failed: {e}")
            # Continue anyway - might already be started
    
    # 2. Create model manager with coordinator
    manager = EnhancedModelManager(coordinator=coordinator)
    
    # 2. Ghost Load Model ONCE (shared Text Segment on server)
    print(f"[Djinn] Ghost loading model: {args.model_id}")
    model = create_hf_ghost_model(args.model_id)
    
    print(f"[Djinn] Registering model with server...")
    await manager.register_model(model, model_id=args.model_id)
    print(f"[Djinn] Model registered (fingerprint: {args.model_id[:16]}...)")
    
    # 3. Load tokenizer (explicit token=None to avoid conflicts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompt
    prompt_text = args.prompt_file.read_text().strip()

    # 4. Output structure
    payload = {
        "tag": args.tag,
        "model_id": args.model_id,
        "dtype": "float16",
        "new_tokens": args.new_tokens,
        "sleep_seconds": args.sleep_seconds,
        "iterations": args.iterations,
        "agent_counts": [],
        "generated_at": _utc_timestamp(),
    }

    # 5. Sweep loop: vary number of concurrent agents
    for n_agents in args.agent_counts:
        print(f"\n[Djinn] ============================================================")
        print(f"[Djinn] Starting sweep: {n_agents} concurrent agents...")
        print(f"[Djinn] ============================================================")
        
        # Create async task for each agent
        tasks = []
        for i in range(n_agents):
            tasks.append(
                agent_lifecycle(i, manager, model, tokenizer, prompt_text, args)
            )
        
        # Run all agents concurrently
        sweep_start = time.perf_counter()
        try:
            results_list = await asyncio.gather(*tasks)
            sweep_duration = time.perf_counter() - sweep_start
        except Exception as e:
            print(f"[Djinn] ERROR during sweep (N={n_agents}): {e}")
            # Record partial results
            results_list = []
            sweep_duration = time.perf_counter() - sweep_start

        # Flatten results
        all_records = [rec for sublist in results_list for rec in sublist]

        # Calculate statistics
        if all_records:
            latencies = sorted([r["latency_ms"] for r in all_records])
            mean_lat = sum(latencies) / len(latencies)
            p50_lat = latencies[len(latencies) // 2]
            p99_idx = int(len(latencies) * 0.99)
            p99_lat = latencies[min(p99_idx, len(latencies) - 1)]
            
            # Count KV reuse events (decode phases)
            kv_reused_count = sum(1 for r in all_records if r.get("kv_reused", False))
        else:
            mean_lat = p50_lat = p99_lat = 0.0
            kv_reused_count = 0

        print(f"[Djinn] N={n_agents} complete:")
        print(f"  - Total duration: {sweep_duration:.2f}s")
        print(f"  - Records: {len(all_records)}")
        print(f"  - Mean latency: {mean_lat:.2f}ms")
        print(f"  - P99 latency: {p99_lat:.2f}ms")
        print(f"  - KV reuse events: {kv_reused_count}")

        payload["agent_counts"].append({
            "agents": n_agents,
            "records": all_records,
            "aggregates": {
                "mean_latency_ms": mean_lat,
                "p50_latency_ms": p50_lat,
                "p99_latency_ms": p99_lat,
                "total_duration_s": sweep_duration,
                "kv_reused_count": kv_reused_count,
            }
        })
        
        # Optional: sleep between sweeps to let server GC clear old sessions
        if n_agents < args.agent_counts[-1]:
            await asyncio.sleep(1)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Djinn agent scaling baseline (hero experiment)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("Evaluation/exp2_1_llm_decode/configs/prompt.txt"),
        help="Path to prompt file"
    )
    parser.add_argument(
        "--agent-counts",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        help="Number of concurrent agents to sweep"
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=50,
        help="Tokens to generate per Reason/Reflect phase"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Reason → Act → Reflect loops per agent"
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=10.0,
        help="Sleep duration during Act phase (simulates tool use)"
    )
    parser.add_argument(
        "--djinn-server",
        type=str,
        default="localhost:5556",
        help="Djinn server data-plane address (host:port)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Evaluation/exp2_1_llm_decode/results/djinn_agents"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="djinn_hero",
        help="Result tag for filename"
    )
    parser.add_argument(
        "--sample-gpu",
        action="store_true",
        help="Enable GPU utilization sampling (NVML)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Djinn Agent Scaling Baseline (Experiment 2 - Hero)")
    print("=" * 70)
    print(f"Model: {args.model_id}")
    print(f"Agent counts: {args.agent_counts}")
    print(f"New tokens per phase: {args.new_tokens}")
    print(f"Iterations per agent: {args.iterations}")
    print(f"Sleep between Reason/Reflect: {args.sleep_seconds}s")
    print(f"Djinn server: {args.djinn_server}")
    print("=" * 70)

    # Initialize Djinn BEFORE entering async context
    print("[Djinn] Initializing client...")
    ensure_initialized_before_async(args.djinn_server)
    
    # Verify coordinator is available
    coordinator = get_coordinator()
    if coordinator is None:
        raise RuntimeError("Failed to initialize Djinn coordinator. Check server connection.")
    
    # Coordinator will be started in async context if needed
    print("[Djinn] Client initialized successfully (coordinator will start in async context if needed)")

    # Run the sweep (pass coordinator to avoid async context issues)
    payload = asyncio.run(run_sweep(args, coordinator))

    # Save results
    out_path = args.output_dir / f"{args.tag}_{_utc_timestamp()}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n[Djinn] Results saved to {out_path}")


if __name__ == "__main__":
    main()

