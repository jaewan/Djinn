#!/usr/bin/env python3
"""
Multi-turn conversation baseline for Experiment 2.3.

Runs a scripted dialogue using HuggingFace generation, measuring per-turn latency,
data transfer, and throughput. Serves as the native PyTorch reference before
comparing against Djinn semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from djinn.core.ghost_loader import create_hf_ghost_model

from Evaluation.common.djinn_remote import configure_remote_backend

try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover
    pynvml = None
    _NVML_AVAILABLE = False


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
    def __init__(self, device_index: int, interval_s: float = 0.02) -> None:
        if not _NVML_AVAILABLE:
            raise RuntimeError("pynvml (nvidia-ml-py3) required for GPU sampling")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, float]] = []

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
    parser.add_argument("--model-id", default="EleutherAI/gpt-j-6b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--backend", choices=["local", "djinn"], default="local", help="Choose execution backend")
    parser.add_argument("--djinn-server", help="Djinn data-plane address host:port (default localhost:5556)")
    parser.add_argument("--djinn-device-index", type=int, default=0, help="Remote device index on Djinn node")
    parser.add_argument("--conversation-file", type=Path, default=Path("Evaluation/exp2_3_conversation/prompts/conversation.json"), help="Path to conversation JSON file")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_3_conversation/results/native_pytorch"))
    parser.add_argument("--tag", default="native_pytorch")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--nvml-interval", type=float, default=0.02)
    parser.add_argument("--save-turn-text", action="store_true")
    parser.add_argument("--semantic-aware", action="store_true", default=True, help="Use semantic hints (default: True)")
    parser.add_argument("--no-semantic-aware", dest="semantic_aware", action="store_false", help="Disable semantic hints (semantic-blind mode)")
    parser.add_argument("--semantic-session", dest="semantic_use_session", action="store_true", default=True, help="Reuse session IDs for KV persistence (default: True)")
    parser.add_argument("--no-semantic-session", dest="semantic_use_session", action="store_false", help="Disable session reuse for semantics")
    return parser.parse_args()


def load_conversation(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text())
    return data


def prepare_model(args: argparse.Namespace):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.backend == "djinn":
        model = create_hf_ghost_model(args.model_id, task="causal-lm")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        device = args.device
        if device != "cpu":
            model.to(device)
    model.eval()
    # For remote execution we keep tensors on CPU; ghost models have no weights.
    return model, tokenizer


def bytes_of_tensor(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.numel()


def build_prompt(history: List[Dict[str, str]]) -> str:
    text_parts = []
    for msg in history:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            text_parts.append(f"[System] {content}\n")
        elif role == "user":
            text_parts.append(f"User: {content}\n")
        elif role == "assistant":
            text_parts.append(f"Assistant: {content}\n")
    text_parts.append("Assistant:")
    return "\n".join(text_parts)


def generate_turn(
    model,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
    sampler: Optional[NvmlSampler],
) -> Dict[str, Any]:
    device = torch.device(args.device)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)
    host_to_device_mb = bytes_of_tensor(input_ids) / (1024**2)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    if sampler and not sampler._thread:
        sampler.__enter__()

    with torch.no_grad():
        start = time.perf_counter()
        output_ids = model.generate(input_ids=input_ids, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - start) * 1000.0

    generated_ids = output_ids[:, input_ids.shape[-1] :]
    generated_cpu = generated_ids.to("cpu")
    tokens = generated_cpu.shape[-1]
    per_token_ms = total_ms / max(tokens, 1)
    device_to_host_mb = bytes_of_tensor(generated_cpu) / (1024**2)

    text = tokenizer.batch_decode(generated_cpu, skip_special_tokens=True)[0].strip()

    return {
        "latency_ms": total_ms,
        "per_token_ms": per_token_ms,
        "tokens": tokens,
        "generated_text": text,
        "host_to_device_mb": host_to_device_mb,
        "device_to_host_mb": device_to_host_mb,
    }


def run_dialogue(
    model,
    tokenizer,
    conversation_template: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    history: List[Dict[str, str]] = []
    turn_records: List[Dict[str, Any]] = []

    sampler: Optional[NvmlSampler] = None
    device = torch.device(args.device)
    if args.sample_gpu and device.type == "cuda" and _NVML_AVAILABLE:
        sampler = NvmlSampler(device.index or 0, interval_s=args.nvml_interval)
        sampler.__enter__()

    run_start = time.perf_counter()

    for msg in conversation_template:
        history.append({"role": msg["role"], "content": msg.get("content", "")})
        if msg["role"] == "assistant" and not msg.get("content"):
            prompt_text = build_prompt(history[:-1])  # exclude current placeholder
            result = generate_turn(model, tokenizer, prompt_text, args, sampler)
            history[-1]["content"] = result["generated_text"]
            turn_records.append(
                {
                    "turn_id": len(turn_records) + 1,
                    "prompt": prompt_text,
                    "response": result["generated_text"] if args.save_turn_text else None,
                    **{k: result[k] for k in ("latency_ms", "per_token_ms", "tokens", "host_to_device_mb", "device_to_host_mb")},
                }
            )

    total_time_ms = (time.perf_counter() - run_start) * 1000.0
    if sampler:
        sampler.__exit__(None, None, None)
        gpu_summary = sampler.summary()
    else:
        gpu_summary = GpuSampleSummary()

    return {
        "turns": turn_records,
        "total_time_ms": total_time_ms,
        "history": history if args.save_turn_text else None,
        "gpu": gpu_summary.__dict__,
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


async def generate_turn_remote(
    manager,
    model,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Remote execution path for a single turn."""
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = encoded["input_ids"]
    host_to_device_mb = bytes_of_tensor(input_ids) / (1024**2)

    start = time.perf_counter()
    generated_ids, bytes_sent, bytes_received = await _remote_generate_sequence(
        manager,
        model,
        tokenizer,
        input_ids,
        args.max_new_tokens,
        args,
    )
    total_ms = (time.perf_counter() - start) * 1000.0

    generated_tokens = generated_ids[:, input_ids.shape[-1]:]
    generated_cpu = generated_tokens.to("cpu")
    tokens = generated_cpu.shape[-1]
    per_token_ms = total_ms / max(tokens, 1)
    device_to_host_mb = bytes_received / (1024**2)

    text = tokenizer.batch_decode(generated_cpu, skip_special_tokens=True)[0].strip()

    return {
        "latency_ms": total_ms,
        "per_token_ms": per_token_ms,
        "tokens": tokens,
        "generated_text": text,
        "host_to_device_mb": host_to_device_mb,
        "device_to_host_mb": device_to_host_mb,
    }


async def _remote_generate_sequence(
    manager,
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int, int]:
    """Generate sequence token-by-token using remote execution."""
    generated = prompt_ids.clone()
    bytes_sent = 0
    bytes_received = 0

    if new_tokens <= 0:
        return generated, bytes_sent, bytes_received

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
                    }
                    phase = "prefill"
                else:
                    # Decode: send only the most recent token
                    inputs = {
                        "input_ids": generated[:, -1:],
                    }
                    phase = "decode"
            else:
                # Semantic-blind: always send full sequence
                inputs = {
                    "input_ids": generated,
                }
                phase = "decode"
            
            bytes_sent += bytes_of_tensor(inputs["input_ids"])

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

    return generated, bytes_sent, bytes_received


async def run_dialogue_remote(
    manager,
    model,
    tokenizer,
    conversation_template: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Remote execution path for dialogue."""
    history: List[Dict[str, str]] = []
    turn_records: List[Dict[str, Any]] = []
    run_start = time.perf_counter()

    for msg in conversation_template:
        history.append({"role": msg["role"], "content": msg.get("content", "")})
        if msg["role"] == "assistant" and not msg.get("content"):
            prompt_text = build_prompt(history[:-1])  # exclude current placeholder
            result = await generate_turn_remote(manager, model, tokenizer, prompt_text, args)
            history[-1]["content"] = result["generated_text"]
            turn_records.append(
                {
                    "turn_id": len(turn_records) + 1,
                    "prompt": prompt_text,
                    "response": result["generated_text"] if args.save_turn_text else None,
                    **{k: result[k] for k in ("latency_ms", "per_token_ms", "tokens", "host_to_device_mb", "device_to_host_mb")},
                }
            )

    total_time_ms = (time.perf_counter() - run_start) * 1000.0
    return {
        "turns": turn_records,
        "total_time_ms": total_time_ms,
        "history": history if args.save_turn_text else None,
        "gpu": {"gpu_util_pct": None, "mem_util_pct": None, "mem_used_mb": None},
    }


async def run_remote_conversation_baseline(args: argparse.Namespace) -> None:
    """Remote execution path for conversation baseline."""
    from djinn.core.coordinator import get_coordinator
    from djinn.core.enhanced_model_manager import EnhancedModelManager
    from djinn.backend.runtime.initialization import get_coordinator as get_runtime_coordinator, _runtime_state

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    conversation = load_conversation(args.conversation_file)
    model, tokenizer = prepare_model(args)

    # Ensure coordinator is available - wait for initialization if needed
    coordinator = None
    try:
        coordinator = get_runtime_coordinator()
    except RuntimeError:
        pass
    
    if coordinator is None:
        try:
            coordinator = get_coordinator()
        except RuntimeError:
            pass
    
    if coordinator is None:
        # Wait for initialization task if it's running
        if _runtime_state.initialization_task:
            try:
                await _runtime_state.initialization_task
                coordinator = get_runtime_coordinator()
            except Exception as e:
                raise RuntimeError(f"Djinn coordinator unavailable after initialization: {e}")
        
        if coordinator is None:
            raise RuntimeError("Djinn coordinator unavailable. Ensure djinn.init() ran successfully.")

    manager = EnhancedModelManager(coordinator=coordinator, server_address=args.djinn_server)
    manager.use_model_cache = True
    await manager.register_model(model, model_id=args.model_id)

    # Warmup
    for _ in range(args.warmup_runs):
        await run_dialogue_remote(manager, model, tokenizer, conversation, args)

    runs: List[Dict[str, Any]] = []
    for run_id in range(1, args.runs + 1):
        result = await run_dialogue_remote(manager, model, tokenizer, conversation, args)
        aggregates = aggregate_turns(result["turns"])
        runs.append(
            {
                "run_id": run_id,
                "total_time_ms": result["total_time_ms"],
                "turns": result["turns"],
                "aggregates": aggregates,
                "gpu": result["gpu"],
            }
        )
        print(
            f"[run {run_id}/{args.runs}] total={result['total_time_ms']:.2f}ms "
            f"(mean turn latency={aggregates['turn_latency_ms']['mean']:.2f} ms)"
        )

    device_desc = f"Djinn remote ({getattr(args, 'djinn_server_address', 'default')})"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"conversation_{args.tag}_{timestamp}.json"

    payload = {
        "baseline": args.tag,
        "model_id": args.model_id,
        "device": device_desc,
        "dtype": args.dtype,
        "backend": args.backend,
        "djinn_server": getattr(args, "djinn_server_address", None),
        "runs": runs,
        "conversation_file": str(args.conversation_file),
        "library_versions": {
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
        },
    }

    with output_file.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved remote conversation baseline results to {output_file}")


def aggregate_turns(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = sorted(t["latency_ms"] for t in turns)
    per_token = sorted(t["per_token_ms"] for t in turns)
    data_host = sum(t["host_to_device_mb"] for t in turns)
    data_dev = sum(t["device_to_host_mb"] for t in turns)

    def stats(values):
        return {
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": _percentile(values, 95.0),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "turn_latency_ms": stats(latencies),
        "turn_per_token_ms": stats(per_token),
        "total_host_to_device_mb": data_host,
        "total_device_to_host_mb": data_dev,
    }


def main() -> None:
    args = parse_args()
    configure_remote_backend(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.backend == "djinn":
        asyncio.run(run_remote_conversation_baseline(args))
        return

    conversation = load_conversation(args.conversation_file)
    model, tokenizer = prepare_model(args)

    device_obj = torch.device(args.device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        idx = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        device_desc = torch.cuda.get_device_name(idx)
    else:
        device_desc = str(device_obj)

    for _ in range(args.warmup_runs):
        run_dialogue(model, tokenizer, conversation, args)

    runs: List[Dict[str, Any]] = []
    for run_id in range(1, args.runs + 1):
        result = run_dialogue(model, tokenizer, conversation, args)
        aggregates = aggregate_turns(result["turns"])
        runs.append(
            {
                "run_id": run_id,
                "total_time_ms": result["total_time_ms"],
                "turns": result["turns"],
                "aggregates": aggregates,
                "gpu": result["gpu"],
            }
        )
        print(
            f"[run {run_id}/{args.runs}] total={result['total_time_ms']:.2f}ms "
            f"(mean turn latency={aggregates['turn_latency_ms']['mean']:.2f} ms)"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"conversation_{args.tag}_{timestamp}.json"

    payload = {
        "baseline": args.tag,
        "model_id": args.model_id,
        "device": device_desc,
        "dtype": args.dtype,
        "backend": args.backend,
        "djinn_server": getattr(args, "djinn_server_address", None),
        "runs": runs,
        "conversation_file": str(args.conversation_file),
        "library_versions": {
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
        },
    }

    with output_file.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved conversation baseline results to {output_file}")


if __name__ == "__main__":
    main()

