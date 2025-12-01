#!/usr/bin/env python3
"""
Ray serverless baseline for Experiment 2 (Hero agent workload).

Each Reason/Reflect step is executed by a short-lived Ray task that loads the
model, performs generation, and releases GPU memory immediately afterward.
This emulates the cold-start behavior of stateless serverless runtimes that
cannot retain weights or KV cache between user inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to path for module imports
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.metrics import summarize_fields


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


@ray.remote
def serverless_step(
    model_id: str,
    dtype: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = _resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs: Dict = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"text": decoded, "tokens_generated": tokens_generated}


def run_serverless_loop(
    prompt: str,
    agent_count: int,
    iterations: int,
    sleep_s: float,
    model_id: str,
    dtype: str,
    new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict]:
    contexts = [prompt for _ in range(agent_count)]
    records: List[Dict] = []
    for iteration in range(iterations):
        for stage in ("reason", "reflect"):
            pending = []
            for agent_id in range(agent_count):
                start = time.perf_counter()
                ref = serverless_step.remote(
                    model_id,
                    dtype,
                    contexts[agent_id],
                    new_tokens,
                    temperature,
                    top_p,
                )
                pending.append((agent_id, start, ref))
            for agent_id, start, ref in pending:
                result = ray.get(ref)
                latency_ms = (time.perf_counter() - start) * 1000.0
                contexts[agent_id] = result["text"]
                records.append(
                    {
                        "agent_id": agent_id,
                        "iteration": iteration,
                        "stage": stage,
                        "latency_ms": latency_ms,
                        "tokens_generated": result["tokens_generated"],
                    }
                )
            if stage == "reason":
                time.sleep(sleep_s)
                for idx in range(agent_count):
                    contexts[idx] += "\nObservation: external action completed."
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray serverless baseline (hero experiment)")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prompt-file", type=Path, default=Path("Evaluation/exp2_1_llm_decode/configs/prompt.txt"))
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_1_llm_decode/results/ray_serverless"))
    parser.add_argument("--tag", type=str, default="ray_serverless")
    parser.add_argument("--ray-address", type=str, help="Optional Ray cluster address (default: local).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt_file.read_text().strip()
    # Initialize Ray with minimal dependencies
    # OSDI FIX: Handle remote Ray cluster connection properly
    ray_address = args.ray_address
    if ray_address:
        server_ip, server_port = ray_address.split(':')
        print(f"[ray-serverless] Connecting to Ray cluster at: {ray_address}")
        
        # Test connectivity to required ports
        import subprocess
        import socket
        print(f"[ray-serverless] Testing connectivity...")
        
        def test_port(host, port, timeout=2):
            """Test if a port is reachable."""
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, int(port)))
                sock.close()
                return result == 0
            except Exception:
                return False
        
        # Test GCS port
        if test_port(server_ip, server_port):
            print(f"[ray-serverless] ✓ GCS port {server_port} is reachable")
        else:
            print(f"[ray-serverless] ✗ GCS port {server_port} is NOT reachable!")
            print(f"[ray-serverless]   Fix: Open port {server_port} in AWS Security Group")
        
        # Test Raylet/Object Store ports (critical for worker communication)
        test_ports = [10001, 10005, 10010, 10015]
        all_ports_ok = True
        failed_ports = []
        for port in test_ports:
            if not test_port(server_ip, port):
                all_ports_ok = False
                failed_ports.append(port)
        
        if all_ports_ok:
            print(f"[ray-serverless] ✓ Raylet/Object Store ports (10001-10020) are reachable")
        else:
            print(f"[ray-serverless] ✗ Some Raylet ports are NOT reachable: {failed_ports}")
            print(f"[ray-serverless]   Fix: Open ports 10001-10020 in AWS Security Group")
            print(f"[ray-serverless]   This is required for Ray worker communication")
            print(f"[ray-serverless]   Without these ports, Ray will try to start locally and fail")
    
    try:
        # OSDI FIX: Clean up any existing Ray sessions before connecting
        # This prevents "node_ip_address.json" errors from stale sessions
        import os
        import shutil
        ray_session_dir = "/tmp/ray"
        if os.path.exists(ray_session_dir):
            # Don't delete everything, just log that we're connecting fresh
            print(f"[ray-serverless] Note: Existing Ray sessions in {ray_session_dir} (will use remote cluster)")
        
        # OSDI FIX: Explicitly configure for remote cluster connection
        # When connecting to remote cluster, we're a pure client (no local worker)
        init_kwargs = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }
        
        if ray_address:
            init_kwargs["address"] = ray_address
            # Set a longer timeout for remote connections
            init_kwargs["_system_config"] = {
                "object_timeout_milliseconds": 200000,
            }
        
        ray.init(**init_kwargs)
        
        # Verify we're connected to the remote cluster
        cluster_resources = ray.cluster_resources()
        print(f"[ray-serverless] Successfully connected to Ray cluster")
        print(f"[ray-serverless] Cluster resources: {cluster_resources}")
    except Exception as e:
        print(f"[ray-serverless] ERROR: Failed to connect to Ray cluster at {ray_address}")
        print(f"[ray-serverless] Error: {e}")
        print(f"[ray-serverless] Troubleshooting:")
        print(f"  1. Verify server is running: ray status (on server machine)")
        print(f"  2. Check port matches: server used --port=XXX, client uses SERVER_IP:XXX")
        print(f"  3. Test connectivity: nc -zv {ray_address.split(':')[0] if ray_address else 'SERVER_IP'} {ray_address.split(':')[1] if ray_address else 'PORT'}")
        raise
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_timestamp()
    payload = {
        "tag": args.tag,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "prompt_file": str(args.prompt_file),
        "new_tokens": args.new_tokens,
        "sleep_seconds": args.sleep_seconds,
        "iterations": args.iterations,
        "agent_counts": [],
        "generated_at": timestamp,
    }

    for agent_count in args.agent_counts:
        print(f"[ray-serverless] Running {agent_count} agents...")
        records = run_serverless_loop(
            prompt,
            agent_count,
            args.iterations,
            args.sleep_seconds,
            args.model_id,
            args.dtype,
            args.new_tokens,
            args.temperature,
            args.top_p,
        )
        aggregates = summarize_fields(records, ["latency_ms", "tokens_generated"])
        payload["agent_counts"].append(
            {
                "agents": agent_count,
                "records": records,
                "aggregates": aggregates,
            }
        )

    output_path = args.output_dir / f"{args.tag}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[ray-serverless] Saved results to {output_path}")


if __name__ == "__main__":
    main()


