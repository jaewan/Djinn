#!/usr/bin/env python3
"""
Ray keep-alive baseline for Experiment 2 (Hero agent workload).

Spawns long-lived Ray actors, each pinning a full LLM (e.g., Llama-2-7B) on GPU
memory, and executes the Reason → Act (sleep) → Reflect loop for a configurable
number of concurrent agents. This emulates the "persistent actor" pattern where
every tenant holds onto weights + KV cache for the entire session.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

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
class LLMKeepAliveActor:
    def __init__(self, model_id: str, dtype: str, device: str):
        torch_dtype = _resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(device)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"OOM during model loading: {e}")
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs: Dict = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
        try:
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"OOM during generation: {e}")
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])
        return {"text": decoded, "tokens_generated": tokens_generated}


def run_reason_act_reflect(
    actors: List[ray.actor.ActorHandle],
    prompt: str,
    new_tokens: int,
    sleep_s: float,
    iterations: int,
    temperature: float,
    top_p: float,
) -> List[Dict]:
    contexts = [prompt for _ in actors]
    all_records: List[Dict] = []
    for iteration in range(iterations):
        for stage in ("reason", "reflect"):
            pending = []
            for agent_id, actor in enumerate(actors):
                start = time.perf_counter()
                ref = actor.generate.remote(contexts[agent_id], new_tokens, temperature, top_p)
                pending.append((agent_id, start, ref))
            for agent_id, start, ref in pending:
                result = ray.get(ref)
                latency_ms = (time.perf_counter() - start) * 1000.0
                contexts[agent_id] = result["text"]
                all_records.append(
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
                for idx in range(len(contexts)):
                    contexts[idx] += "\nObservation: external action completed."
    return all_records


def summarize_agent_runs(records: List[Dict]) -> Dict[str, Dict]:
    return summarize_fields(records, ["latency_ms", "tokens_generated"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray keep-alive baseline (hero experiment)")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prompt-file", type=Path, default=Path("Evaluation/exp2_1_llm_decode/configs/prompt.txt"))
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=1, help="Reason→Act→Reflect loops per agent.")
    parser.add_argument("--new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_1_llm_decode/results/ray_keepalive"))
    parser.add_argument("--tag", type=str, default="ray_keepalive")
    parser.add_argument("--ray-address", type=str, help="Optional Ray cluster address (default: local).")
    parser.add_argument("--gpu-per-actor", type=float, default=0.01, help="GPU fraction per actor (0.01 forces physical OOM at low N by bypassing Ray scheduler)")
    parser.add_argument("--stop-on-oom", action="store_true", help="Stop sweep when OOM is encountered (for hero comparison).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt_file.read_text().strip()
    # Initialize Ray with minimal dependencies
    # OSDI FIX: Handle remote Ray cluster connection properly
    ray_address = args.ray_address
    if ray_address:
        server_ip, server_port = ray_address.split(':')
        print(f"[ray-keepalive] Connecting to Ray cluster at: {ray_address}")
        
        # Test connectivity to required ports
        import socket
        print(f"[ray-keepalive] Testing connectivity...")
        
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
            print(f"[ray-keepalive] ✓ GCS port {server_port} is reachable")
        else:
            print(f"[ray-keepalive] ✗ GCS port {server_port} is NOT reachable!")
            print(f"[ray-keepalive]   Fix: Open port {server_port} in AWS Security Group")
        
        # Test Raylet/Object Store ports (critical for worker communication)
        test_ports = [10001, 10005, 10010, 10015]
        all_ports_ok = True
        failed_ports = []
        for port in test_ports:
            if not test_port(server_ip, port):
                all_ports_ok = False
                failed_ports.append(port)
        
        if all_ports_ok:
            print(f"[ray-keepalive] ✓ Raylet/Object Store ports (10001-10020) are reachable")
        else:
            print(f"[ray-keepalive] ✗ Some Raylet ports are NOT reachable: {failed_ports}")
            print(f"[ray-keepalive]   Fix 1: Open ports 10001-10020 in AWS Security Group")
            print(f"[ray-keepalive]   Fix 2: Check ufw on SERVER machine:")
            print(f"[ray-keepalive]     sudo ufw allow 10001:10020/tcp")
            print(f"[ray-keepalive]   This is required for Ray worker communication")
            print(f"[ray-keepalive]   Without these ports, Ray will try to start locally and fail")
    
    try:
        # OSDI FIX: Clean up any existing Ray sessions before connecting
        # This prevents "node_ip_address.json" errors from stale sessions
        import os
        ray_session_dir = "/tmp/ray"
        if os.path.exists(ray_session_dir):
            # Don't delete everything, just log that we're connecting fresh
            print(f"[ray-keepalive] Note: Existing Ray sessions in {ray_session_dir} (will use remote cluster)")
        
        # OSDI FIX: Explicitly configure for remote cluster connection
        # When connecting to remote cluster, we're a pure client (no local worker)
        init_kwargs = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }
        
        if ray_address:
            init_kwargs["address"] = ray_address
            # NOTE: Cannot use _system_config when connecting to existing cluster
            # Ray will use the cluster's system config automatically
        
        ray.init(**init_kwargs)
        
        # Verify we're connected to the remote cluster
        cluster_resources = ray.cluster_resources()
        print(f"[ray-keepalive] Successfully connected to Ray cluster")
        print(f"[ray-keepalive] Cluster resources: {cluster_resources}")
    except Exception as e:
        print(f"[ray-keepalive] ERROR: Failed to connect to Ray cluster at {ray_address}")
        print(f"[ray-keepalive] Error: {e}")
        print(f"[ray-keepalive] Troubleshooting:")
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

    oom_encountered = False
    for agent_count in args.agent_counts:
        if oom_encountered:
            print(f"[ray-keepalive] Skipping N={agent_count} (OOM already encountered)")
            payload["agent_counts"].append({
                "agents": agent_count,
                "records": [],
                "aggregates": {},
                "oom": True,
            })
            continue
        
        print(f"[ray-keepalive] Running {agent_count} concurrent agents...")
        actors = []
        try:
            # Try to create and initialize actors
            for i in range(agent_count):
                try:
                    actor = LLMKeepAliveActor.options(num_gpus=args.gpu_per_actor).remote(
                        args.model_id,
                        args.dtype,
                        "cuda" if torch.cuda.is_available() else "cpu",
                    )
                    # Test actor by calling a simple method to catch OOM early
                    ray.get(actor.generate.remote(prompt, 1, args.temperature, args.top_p))
                    actors.append(actor)
                except Exception as e:
                    if "OutOfMemoryError" in str(e) or "OOM" in str(e):
                        print(f"[ray-keepalive] OOM at agent {i} (N={agent_count}): {e}")
                        oom_encountered = True
                        # Clean up any actors we created
                        for a in actors:
                            ray.kill(a)
                        payload["agent_counts"].append({
                            "agents": agent_count,
                            "records": [],
                            "aggregates": {},
                            "oom": True,
                            "oom_at_agent": i,
                        })
                        break
                    else:
                        raise
            else:
                # All actors created successfully, run the loop
                records = run_reason_act_reflect(
                    actors,
                    prompt,
                    args.new_tokens,
                    args.sleep_seconds,
                    args.iterations,
                    args.temperature,
                    args.top_p,
                )
                aggregates = summarize_agent_runs(records)
                payload["agent_counts"].append(
                    {
                        "agents": agent_count,
                        "records": records,
                        "aggregates": aggregates,
                    }
                )
                # Terminate actors gracefully
                for actor in actors:
                    ray.kill(actor)
                ray.wait([actor.__ray_terminate__.remote() for actor in actors], timeout=5)
        except Exception as e:
            if "OutOfMemoryError" in str(e) or "OOM" in str(e):
                print(f"[ray-keepalive] OOM during execution (N={agent_count}): {e}")
                oom_encountered = True
                payload["agent_counts"].append({
                    "agents": agent_count,
                    "records": [],
                    "aggregates": {},
                    "oom": True,
                })
                if args.stop_on_oom:
                    break
            else:
                raise

    output_path = args.output_dir / f"{args.tag}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[ray-keepalive] Saved results to {output_path}")


if __name__ == "__main__":
    main()


