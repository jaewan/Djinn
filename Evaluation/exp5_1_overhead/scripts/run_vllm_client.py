#!/usr/bin/env python3
"""
vLLM baseline runner for Experiment 5.1 microbenchmarks.

Sends prompts to a running vLLM OpenAI-compatible server and records latency +
token statistics so the results can be compared with Djinn and other baselines.

IMPORTANT (OSDI Review - Critique 3):
  vLLM uses highly optimized C++ PagedAttention kernels and kernel-level optimizations
  that are NOT present in the vanilla HuggingFace `model.generate()` used by Djinn/RPC baselines.
  
  Therefore: vLLM comparison is provided for context on specialized kernel optimizations,
  NOT for direct architectural comparison. For fair architectural comparison (RPC overhead),
  see: native_pytorch vs pytorch_rpc vs semantic_blind vs full_djinn (all using same HF code).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests
import yaml

from Evaluation.common.metrics import summarize_fields


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM baseline for overhead experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml"),
        help="Workload config (same as Djinn microbench).",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        help="Optional workload name filter (only hf_causal_lm entries are supported).",
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="dummy")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp5_1_overhead/results/vllm"))
    parser.add_argument("--tag", type=str, default="vllm_baseline")
    return parser.parse_args()


def load_workloads(config_path: Path, only: List[str] | None) -> List[Dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    workloads = []
    for workload in payload.get("workloads", []):
        if workload["implementation"] != "hf_causal_lm":
            continue
        if only and workload["name"] not in only:
            continue
        workloads.append(workload)
    return workloads


def build_prompt_text(spec: Dict) -> str:
    if "prompt_text" in spec and spec["prompt_text"]:
        return spec["prompt_text"]
    if "prompt_file" in spec and spec["prompt_file"]:
        return Path(spec["prompt_file"]).read_text().strip()
    return "Describe the Djinn Tensor Operating System in one paragraph."


def send_vllm_request(
    base_url: str,
    api_key: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> Dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    # Use completions API (not chat) for models without chat template
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    response = requests.post(f"{base_url}/completions", headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def run_workload(args: argparse.Namespace, workload: Dict) -> Dict:
    spec = workload.get("params", {})
    prompt = build_prompt_text(spec)
    new_tokens = spec.get("new_tokens", 32)
    model_id = spec["model_id"]
    
    # OSDI FIX: Warn about HTTP overhead dominating small model latency
    # vLLM HTTP overhead (~2-5ms) can exceed tiny model compute time (<1ms)
    # See: OSDI Review - Critique 2 (HTTP Overhead in vLLM Baseline)
    small_models = ["tiny", "small", "mini", "nano", "micro"]
    if any(marker in model_id.lower() for marker in small_models):
        print(f"[vllm] WARNING: Model '{model_id}' may be too small for accurate benchmarking.")
        print(f"[vllm]          HTTP overhead (~2-5ms) may dominate compute time (<1ms).")
        print(f"[vllm]          Consider using larger models (e.g., Llama-2-7B) for valid comparison.")
    
    runs = []
    for run_id in range(1, args.runs + 1):
        start = time.perf_counter()
        result = send_vllm_request(args.base_url, args.api_key, model_id, prompt, new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", new_tokens)
        prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
        total_tokens = usage.get("total_tokens", completion_tokens + prompt_tokens)
        runs.append(
            {
                "run_id": run_id,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
    aggregates = summarize_fields(runs, ["latency_ms", "completion_tokens", "total_tokens"])
    return {
        "workload": workload["name"],
        "model_id": model_id,
        "prompt": prompt,
        "runs": runs,
        "aggregates": aggregates,
    }


def main() -> None:
    args = parse_args()
    workloads = load_workloads(args.config, args.workloads)
    if not workloads:
        raise SystemExit("No matching hf_causal_lm workloads found in config.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for workload in workloads:
        print(f"[vllm] Running workload {workload['name']} ({workload['params']['model_id']})")
        results.append(run_workload(args, workload))
    payload = {
        "tag": args.tag,
        "base_url": args.base_url,
        "generated_at": _utc_timestamp(),
        "results": results,
    }
    output_path = args.output_dir / f"{args.tag}_{payload['generated_at']}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[vllm] Saved results to {output_path}")


if __name__ == "__main__":
    main()


