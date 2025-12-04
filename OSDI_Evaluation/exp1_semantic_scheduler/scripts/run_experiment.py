#!/usr/bin/env python3
"""
Run Experiment 1: Semantic Scheduler - Agent Scalability (Phase 3).

Demonstrates that semantic idle detection + KV swapping enables 50 concurrent
agents to share a single GPU without OOM.

Usage:
    python scripts/run_experiment.py \
        --config configs/agent_scaling_smoke.yaml \
        --model-id meta-llama/Llama-2-7b-hf \
        --output-dir results
"""

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoTokenizer

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import djinn
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model
from Evaluation.common.djinn_init import ensure_initialized_before_async
import yaml


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


async def agent_lifecycle(
    agent_idx: int,
    manager: EnhancedModelManager,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    config: Dict[str, Any],
) -> List[Dict]:
    """
    Run single agent through Reason → Act → Reflect loop.
    
    Returns list of records per iteration.
    """
    records: List[Dict] = []
    
    session_id = f"agent_{agent_idx}_{uuid.uuid4().hex[:8]}"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    current_input_ids = prompt_tokens["input_ids"]
    kv_processed_len = 0
    
    try:
        for iteration in range(config.get("iterations", 1)):
            # PHASE 1: REASON (Prefill + Initial Decode)
            start_reason = time.perf_counter()
            
            inputs = {"input_ids": current_input_ids}
            
            with djinn.session(phase="prefill", session_id=session_id, priority="normal"):
                reason_result = await manager.execute_model(
                    model,
                    inputs,
                    hints={
                        "use_generate": True,
                        "max_new_tokens": config.get("new_tokens", 50),
                        "do_sample": False,
                    }
                )
            
            latency_reason_ms = (time.perf_counter() - start_reason) * 1000.0
            
            # Update context
            if isinstance(reason_result, dict) and "generated_ids" in reason_result:
                current_input_ids = reason_result["generated_ids"]
            elif torch.is_tensor(reason_result):
                current_input_ids = reason_result
            
            kv_processed_len = current_input_ids.shape[-1]
            
            records.append({
                "agent_id": agent_idx,
                "iteration": iteration,
                "stage": "reason",
                "latency_ms": latency_reason_ms,
                "tokens_generated": config.get("new_tokens", 50),
                "kv_reused": False,
            })
            
            # PHASE 2: ACT (Simulated Tool Use / Idle Period)
            # Randomized sleep to prevent "Thundering Herd" synchronization artifacts
            # Per Evaluation Plan: uniform(8s, 12s)
            if "sleep_seconds" in config:
                sleep_time = config["sleep_seconds"]
            else:
                # Default: randomized uniform(8s, 12s) as per Evaluation Plan
                sleep_time = random.uniform(8.0, 12.0)
            await asyncio.sleep(sleep_time)
            
            # PHASE 3: REFLECT (Decode with KV Reuse)
            start_reflect = time.perf_counter()
            
            # Incremental input for decode
            incremental_input_ids = current_input_ids[:, kv_processed_len:]
            if incremental_input_ids.shape[-1] == 0:
                incremental_input_ids = current_input_ids[:, -1:]
            
            reflect_inputs = {"input_ids": incremental_input_ids}
            
            with djinn.session(phase="decode", session_id=session_id, priority="normal"):
                reflect_result = await manager.execute_model(
                    model,
                    reflect_inputs,
                    hints={
                        "use_generate": True,
                        "max_new_tokens": config.get("new_tokens", 50),
                        "do_sample": False,
                    }
                )
            
            latency_reflect_ms = (time.perf_counter() - start_reflect) * 1000.0
            
            # Update context
            if isinstance(reflect_result, dict) and "generated_ids" in reflect_result:
                current_input_ids = reflect_result["generated_ids"]
            elif torch.is_tensor(reflect_result):
                current_input_ids = reflect_result
            
            kv_processed_len = current_input_ids.shape[-1]
            
            records.append({
                "agent_id": agent_idx,
                "iteration": iteration,
                "stage": "reflect",
                "latency_ms": latency_reflect_ms,
                "tokens_generated": config.get("new_tokens", 50),
                "kv_reused": True,
            })
    
    except Exception as e:
        print(f"[Agent {agent_idx}] ERROR: {e}")
        records.append({
            "agent_id": agent_idx,
            "error": str(e),
            "stage": "error",
        })
    
    return records


async def run_sweep(args: argparse.Namespace, coordinator, config: Dict) -> Dict[str, Any]:
    """Run agent scaling sweep."""
    
    # Ensure coordinator started
    if not hasattr(coordinator, '_started') or not coordinator._started:
        try:
            await coordinator.start()
        except:
            pass
    
    manager = EnhancedModelManager(coordinator=coordinator)
    
    # Ghost load model
    print(f"[Exp1] Loading model: {args.model_id}")
    model = create_hf_ghost_model(args.model_id)
    await manager.register_model(model, model_id=args.model_id)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompt
    prompt_text = "This is a test prompt for agent scaling."
    
    payload = {
        "tag": "exp1_semantic_scheduler",
        "model_id": args.model_id,
        "config": config,
        "generated_at": _utc_timestamp(),
        "agent_counts": [],
    }
    
    # Sweep through agent counts
    for n_agents in config["workload"]["agent_counts"]:
        print(f"\n[Exp1] Starting sweep: {n_agents} agents...")
        
        tasks = []
        for i in range(n_agents):
            tasks.append(
                agent_lifecycle(i, manager, model, tokenizer, prompt_text, config["workload"])
            )
        
        sweep_start = time.perf_counter()
        try:
            results_list = await asyncio.gather(*tasks)
            sweep_duration = time.perf_counter() - sweep_start
        except Exception as e:
            print(f"[Exp1] ERROR during sweep (N={n_agents}): {e}")
            results_list = []
            sweep_duration = time.perf_counter() - sweep_start
        
        all_records = [rec for sublist in results_list for rec in sublist]
        
        # Calculate stats
        if all_records:
            latencies = sorted([r.get("latency_ms", 0) for r in all_records if "latency_ms" in r])
            mean_lat = sum(latencies) / len(latencies) if latencies else 0
            p50_lat = latencies[len(latencies) // 2] if latencies else 0
            p99_idx = int(len(latencies) * 0.99)
            p99_lat = latencies[min(p99_idx, len(latencies) - 1)] if latencies else 0
            kv_reused = sum(1 for r in all_records if r.get("kv_reused", False))
            errors = sum(1 for r in all_records if r.get("error"))
        else:
            mean_lat = p50_lat = p99_lat = 0.0
            kv_reused = 0
            errors = 0
        
        success = len(all_records) == (n_agents * config["workload"]["iterations"] * 2)
        
        print(f"[Exp1] N={n_agents}: duration={sweep_duration:.2f}s, records={len(all_records)}, "
              f"p99={p99_lat:.1f}ms, success={success}")
        
        payload["agent_counts"].append({
            "agents": n_agents,
            "records": all_records,
            "aggregates": {
                "mean_latency_ms": mean_lat,
                "p50_latency_ms": p50_lat,
                "p99_latency_ms": p99_lat,
                "total_duration_s": sweep_duration,
                "kv_reuse_events": kv_reused,
                "errors": errors,
                "success": success,
            }
        })
        
        if n_agents < config["workload"]["agent_counts"][-1]:
            await asyncio.sleep(1)
    
    return payload


def main():
    parser = argparse.ArgumentParser(description="Exp1: Semantic Scheduler Agent Scaling")
    parser.add_argument("--config", type=Path, required=True, help="Config file")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--djinn-server", type=str, default="localhost:5556")
    parser.add_argument("--output-dir", type=Path, default=Path("OSDI_Evaluation/exp1_semantic_scheduler/results"))
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("Exp1: Semantic Scheduler - Agent Scalability (Phase 3)")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Model: {args.model_id}")
    print(f"Server: {args.djinn_server}")
    
    # Initialize Djinn
    print("[Exp1] Initializing Djinn client...")
    ensure_initialized_before_async(args.djinn_server)
    
    coordinator = get_coordinator()
    if coordinator is None:
        print("[ERROR] Failed to initialize coordinator")
        sys.exit(1)
    
    # Run experiment
    payload = asyncio.run(run_sweep(args, coordinator, config))
    
    # Save results
    out_path = args.output_dir / f"exp1_semantic_scheduler_{_utc_timestamp()}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n[Exp1] Results saved to {out_path}")


if __name__ == "__main__":
    main()

