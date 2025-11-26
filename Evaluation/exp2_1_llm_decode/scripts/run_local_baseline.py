#!/usr/bin/env python3
"""
Native PyTorch baseline runner for Experiment 2.1 (LLM decode).

This script times HuggingFace generation locally on a single GPU/CPU, producing
per-run and aggregate metrics compatible with docs/EvaluationPlan.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Evaluation.common.djinn_remote import configure_remote_backend

try:
    import pynvml  # Optional, used for GPU sampling

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pynvml = None
    _NVML_AVAILABLE = False


DEFAULT_PROMPT = (
    "The disaggregated accelerator stack is orchestrated by Djinn because "
    "it understands semantic intent and keeps KV caches resident across decode phases."
)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


@dataclass
class GpuSampleSummary:
    gpu_util_pct: Optional[float] = None
    mem_util_pct: Optional[float] = None
    mem_used_mb: Optional[float] = None


class NvmlSampler:
    """Background GPU utilization sampler using NVML."""

    def __init__(self, device_index: int, interval_s: float = 0.01) -> None:
        if not _NVML_AVAILABLE:
            raise RuntimeError("pynvml not available; install nvidia-ml-py3")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: List[Dict[str, float]] = []
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "NvmlSampler":
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
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self._samples.append(
                {
                    "gpu_util_pct": float(util.gpu),
                    "mem_util_pct": float(util.memory),
                    "mem_used_mb": float(mem.used) / (1024**2),
                    "timestamp": time.perf_counter(),
                }
            )
            self._stop.wait(self.interval_s)

    def summary(self) -> GpuSampleSummary:
        if not self._samples:
            return GpuSampleSummary()
        gpu_util = [sample["gpu_util_pct"] for sample in self._samples]
        mem_util = [sample["mem_util_pct"] for sample in self._samples]
        mem_used = [sample["mem_used_mb"] for sample in self._samples]
        return GpuSampleSummary(
            gpu_util_pct=sum(gpu_util) / len(gpu_util),
            mem_util_pct=sum(mem_util) / len(mem_util),
            mem_used_mb=sum(mem_used) / len(mem_used),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-hf", help="HF model identifier")
    parser.add_argument("--prompt-file", type=Path, help="Path to prompt text file")
    parser.add_argument("--prompt-text", help="Inline prompt override")
    parser.add_argument("--prompt-length", type=int, default=72, help="Prompt token length (padded/truncated)")
    parser.add_argument("--new-tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5, help="Number of measured runs")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Unmeasured warmup runs")
    parser.add_argument("--device", default="cuda:0", help="torch device string")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--backend", choices=["local", "djinn"], default="local", help="Choose execution backend")
    parser.add_argument("--djinn-server", help="Djinn data-plane address host:port (default localhost:5556)")
    parser.add_argument("--djinn-device-index", type=int, default=0, help="Remote device index on Djinn node")
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_1_llm_decode/results/native_pytorch"))
    parser.add_argument("--tag", default="native_pytorch", help="Baseline label for the output JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-gpu", action="store_true", help="Collect GPU utilization via NVML (if available)")
    parser.add_argument("--nvml-interval", type=float, default=0.01, help="Sampling interval for NVML (seconds)")
    parser.add_argument("--save-generated", action="store_true", help="Persist generated text for manual inspection")
    parser.add_argument("--semantic-aware", action="store_true", default=True, help="Use semantic hints (default: True)")
    parser.add_argument("--no-semantic-aware", dest="semantic_aware", action="store_false", help="Disable semantic hints (semantic-blind mode)")
    parser.add_argument("--semantic-session", dest="semantic_use_session", action="store_true", default=True, help="Reuse session IDs for KV persistence (default: True)")
    parser.add_argument("--no-semantic-session", dest="semantic_use_session", action="store_false", help="Disable session reuse for semantics")
    return parser.parse_args()


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_text:
        return args.prompt_text.strip()
    if args.prompt_file:
        return args.prompt_file.read_text().strip()
    return DEFAULT_PROMPT


def prepare_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_inputs(tokenizer, prompt: str, prompt_len: int, batch_size: int):
    encoded = tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        max_length=prompt_len,
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def load_model(model_id: str, dtype: str, device: str):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return model


def bytes_of_tensor(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.numel()


def run_generation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    ids = input_ids.clone().to(device)
    mask = attention_mask.clone().to(device)
    gen_kwargs = dict(
        max_new_tokens=args.new_tokens,
        use_cache=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    sampler: Optional[NvmlSampler] = None
    if args.sample_gpu and device.type == "cuda" and _NVML_AVAILABLE:
        sampler = NvmlSampler(device.index or 0, interval_s=args.nvml_interval)

    with torch.no_grad():
        if sampler:
            sampler.__enter__()
        start = time.perf_counter()
        generated = model.generate(input_ids=ids, attention_mask=mask, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - start) * 1000.0
        if sampler:
            sampler.__exit__(None, None, None)

    tokens_generated = generated.shape[-1] - ids.shape[-1]
    per_token_ms = total_ms / max(tokens_generated, 1)
    throughput_tps = (tokens_generated / (total_ms / 1000.0)) if tokens_generated else 0.0

    generated_cpu = generated.to("cpu")

    if args.save_generated:
        generated_text = tokenizer.batch_decode(generated_cpu, skip_special_tokens=True)
    else:
        generated_text = None

    gpu_summary = sampler.summary() if sampler else GpuSampleSummary()

    host_to_device_mb = bytes_of_tensor(ids) / (1024**2)
    device_to_host_mb = bytes_of_tensor(generated_cpu) / (1024**2)

    return {
        "total_ms": total_ms,
        "per_token_ms": per_token_ms,
        "tokens_generated": tokens_generated,
        "throughput_tokens_per_s": throughput_tps,
        "host_to_device_mb": host_to_device_mb,
        "device_to_host_mb": device_to_host_mb,
        "gpu_util_pct": gpu_summary.gpu_util_pct,
        "mem_util_pct": gpu_summary.mem_util_pct,
        "mem_used_mb": gpu_summary.mem_used_mb,
        "generated_text": generated_text,
    }


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = sorted(run["total_ms"] for run in runs)
    per_token = sorted(run["per_token_ms"] for run in runs)
    throughput = sorted(run["throughput_tokens_per_s"] for run in runs)

    def stats(values):
        return {
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": _percentile(values, 95.0),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "total_ms": stats(totals),
        "per_token_ms": stats(per_token),
        "throughput_tokens_per_s": stats(throughput),
    }


def _extract_logits_from_result(result: Any) -> torch.Tensor:
    if torch.is_tensor(result):
        return result.cpu()
    if isinstance(result, dict):
        if "logits" in result and torch.is_tensor(result["logits"]):
            return result["logits"].cpu()
        for value in result.values():
            if torch.is_tensor(value):
                return value.cpu()
    raise RuntimeError("Unable to extract logits tensor from remote execution result.")


async def run_remote_llm_baseline(args: argparse.Namespace) -> None:
    import djinn
    from djinn.core.coordinator import get_coordinator
    from djinn.core.enhanced_model_manager import EnhancedModelManager

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt = resolve_prompt(args)
    tokenizer = prepare_tokenizer(args.model_id)
    input_ids, attention_mask = prepare_inputs(
        tokenizer, prompt, args.prompt_length, args.batch_size
    )
    model = load_model(args.model_id, args.dtype, device="cpu")

    coordinator = get_coordinator()
    if coordinator is None:
        raise RuntimeError("Djinn coordinator unavailable. Ensure djinn.init() ran successfully.")

    manager = EnhancedModelManager(coordinator=coordinator)
    manager.use_model_cache = True
    await manager.register_model(model, model_id=args.model_id)

    warmups = args.warmup_runs
    for _ in range(warmups):
        await _remote_run_once(manager, model, tokenizer, input_ids, attention_mask, args)

    runs: List[Dict[str, Any]] = []
    for run_id in range(1, args.runs + 1):
        run_result = await _remote_run_once(
            manager, model, tokenizer, input_ids, attention_mask, args
        )
        run_result["run_id"] = run_id
        runs.append(run_result)
        print(
            f"[run {run_id}/{args.runs}] total={run_result['total_ms']:.2f}ms "
            f"({run_result['per_token_ms']:.2f} ms/token, "
            f"throughput={run_result['throughput_tokens_per_s']:.2f} tok/s)"
        )

    aggregates = aggregate_runs(runs)

    device_desc = f"Djinn remote ({getattr(args, 'djinn_server_address', 'default')})"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"llm_decode_{args.tag}_{timestamp}.json"

    payload = {
        "baseline": args.tag,
        "model_id": args.model_id,
        "prompt_tokens": args.prompt_length,
        "new_tokens": args.new_tokens,
        "batch_size": args.batch_size,
        "runs": runs,
        "aggregates": aggregates,
        "device": device_desc,
        "dtype": args.dtype,
        "backend": args.backend,
        "djinn_server": getattr(args, "djinn_server_address", None),
        "seed": args.seed,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
        },
        "meta": {
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
            "prompt_text": prompt if args.prompt_text else None,
            "sample_gpu": args.sample_gpu and _NVML_AVAILABLE,
        },
    }
    with output_file.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved remote results to {output_file}")
    print("Aggregate latency (ms): mean={mean:.2f}, p95={p95:.2f}".format(
        mean=aggregates["total_ms"]["mean"],
        p95=aggregates["total_ms"]["p95"],
    ))
    print("Per-token latency (ms): mean={mean:.2f}, p95={p95:.2f}".format(
        mean=aggregates["per_token_ms"]["mean"],
        p95=aggregates["per_token_ms"]["p95"],
    ))


async def _remote_run_once(
    manager: EnhancedModelManager,
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    start = time.perf_counter()
    generated_ids, bytes_sent, bytes_received = await _remote_generate_sequence(
        manager,
        model,
        tokenizer,
        prompt_ids,
        attention_mask,
        args.new_tokens,
        args,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    tokens_generated = args.new_tokens
    throughput = tokens_generated / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
    per_token_avg = latency_ms / max(tokens_generated, 1)
    prompt_len = prompt_ids.shape[-1]
    generated_text = ""
    if tokens_generated > 0:
        generated_tokens = generated_ids[:, prompt_len:]
        generated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0].strip()

    return {
        "total_ms": latency_ms,
        "per_token_ms": per_token_avg,
        "tokens_generated": tokens_generated,
        "throughput_tokens_per_s": throughput,
        "host_to_device_mb": bytes_sent / (1024**2),
        "device_to_host_mb": bytes_received / (1024**2),
        "gpu_util_pct": None,
        "mem_util_pct": None,
        "mem_used_mb": None,
        "generated_text": generated_text,
    }


async def _remote_generate_sequence(
    manager: EnhancedModelManager,
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    new_tokens: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int, int]:
    generated = prompt_ids.clone()
    mask = attention_mask.clone()
    bytes_sent = 0
    bytes_received = 0

    if new_tokens <= 0:
        return generated, bytes_sent, bytes_received

    # Use semantic hints for decode phase (if enabled)
    import djinn
    import uuid
    
    semantic_aware = getattr(args, 'semantic_aware', True)
    semantic_session_enabled = getattr(args, 'semantic_use_session', True)
    can_persist = semantic_aware and semantic_session_enabled
    decode_session_id = f"decode_{uuid.uuid4().hex[:12]}" if can_persist else None
    
    with torch.no_grad():
        for token_idx in range(new_tokens):
            if can_persist:
                if token_idx == 0:
                    # Prefill: send entire prompt
                    inputs = {
                        "input_ids": generated,
                        "attention_mask": mask,
                    }
                    phase = "prefill"
                else:
                    # Decode: send only the most recent token
                    inputs = {
                        "input_ids": generated[:, -1:],
                        "attention_mask": mask[:, -1:],
                    }
                    phase = "decode"
            else:
                # Semantic-blind: always send full sequence
                inputs = {
                    "input_ids": generated,
                    "attention_mask": mask,
                }
                phase = "decode"
            
            bytes_sent += bytes_of_tensor(inputs["input_ids"])
            bytes_sent += bytes_of_tensor(inputs["attention_mask"])

            if semantic_aware:
                with djinn.session(
                    phase=phase,
                    priority="normal",
                    session_id=decode_session_id if can_persist else None,
                    expected_tokens=new_tokens - token_idx,
                ):
                    result = await manager.execute_model(model, inputs)
            else:
                result = await manager.execute_model(model, inputs)
            
            logits = _extract_logits_from_result(result)
            bytes_received += bytes_of_tensor(logits)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=-1)

    return generated, bytes_sent, bytes_received


def main() -> None:
    args = parse_args()
    configure_remote_backend(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.backend == "djinn":
        asyncio.run(run_remote_llm_baseline(args))
        return

    prompt = resolve_prompt(args)
    tokenizer = prepare_tokenizer(args.model_id)
    input_ids, attention_mask = prepare_inputs(
        tokenizer, prompt, args.prompt_length, args.batch_size
    )
    model = load_model(args.model_id, args.dtype, args.device)

    device_obj = torch.device(args.device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        idx = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        device_desc = torch.cuda.get_device_name(idx)
    else:
        device_desc = str(device_obj)

    # Warmup
    for _ in range(args.warmup_runs):
        run_generation(model, tokenizer, input_ids, attention_mask, args)

    runs: List[Dict[str, Any]] = []
    for run_id in range(1, args.runs + 1):
        result = run_generation(model, tokenizer, input_ids, attention_mask, args)
        result["run_id"] = run_id
        runs.append(result)
        print(
            f"[run {run_id}/{args.runs}] total={result['total_ms']:.2f}ms "
            f"({result['per_token_ms']:.2f} ms/token, throughput={result['throughput_tokens_per_s']:.2f} tok/s)"
        )

    aggregates = aggregate_runs(runs)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"llm_decode_{args.tag}_{timestamp}.json"
    payload = {
        "baseline": args.tag,
        "model_id": args.model_id,
        "prompt_tokens": args.prompt_length,
        "new_tokens": args.new_tokens,
        "batch_size": args.batch_size,
        "runs": runs,
        "aggregates": aggregates,
        "device": device_desc,
        "dtype": args.dtype,
        "backend": args.backend,
        "djinn_server": getattr(args, "djinn_server_address", None),
        "seed": args.seed,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
        },
        "meta": {
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
            "prompt_text": prompt if args.prompt_text else None,
            "sample_gpu": args.sample_gpu and _NVML_AVAILABLE,
        },
    }
    with output_file.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to {output_file}")
    print(
        "Aggregate latency (ms): mean={mean:.2f}, p95={p95:.2f}".format(
            mean=aggregates["total_ms"]["mean"],
            p95=aggregates["total_ms"]["p95"],
        )
    )
    print(
        "Per-token latency (ms): mean={mean:.2f}, p95={p95:.2f}".format(
            mean=aggregates["per_token_ms"]["mean"],
            p95=aggregates["per_token_ms"]["p95"],
        )
    )


if __name__ == "__main__":
    main()

