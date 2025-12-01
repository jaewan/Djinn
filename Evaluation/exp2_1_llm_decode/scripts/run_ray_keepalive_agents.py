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
    ray.init(
        address=args.ray_address or None,
        ignore_reinit_error=True,
        include_dashboard=False,
    )
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


