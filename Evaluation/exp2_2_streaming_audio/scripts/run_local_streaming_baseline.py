#!/usr/bin/env python3
"""
Native PyTorch streaming audio baseline for Experiment 2.2.

Transcribes long-form audio via Whisper, chunked into overlapping windows to
mimic real-time streaming. Generates per-chunk latency/data statistics and saves
results for comparison with Djinn-based baselines.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
            raise RuntimeError("pynvml (nvidia-ml-py3) is required for GPU sampling")
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


def build_decoder_prompt_tensor(forced_decoder_ids: List[List[int]]) -> torch.LongTensor:
    """Convert forced decoder ids ([[pos, token], ...]) into an input tensor."""
    tokens = [token for _, token in forced_decoder_ids]
    return torch.tensor([tokens], dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="openai/whisper-large-v3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--backend", choices=["local", "djinn"], default="local", help="Choose execution backend")
    parser.add_argument("--djinn-server", help="Djinn data-plane address host:port (default localhost:5556)")
    parser.add_argument("--djinn-device-index", type=int, default=0, help="Remote device index on Djinn node")
    parser.add_argument("--audio-file", type=Path, help="Path to WAV/FLAC/etc input audio")
    parser.add_argument("--dummy-audio", type=float, default=10.0, help="Generate synthetic sine audio of N seconds (default: 10.0)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_2_streaming_audio/results/native_pytorch"))
    parser.add_argument("--tag", default="native_pytorch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--nvml-interval", type=float, default=0.02)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--save-chunk-text", action="store_true")
    parser.add_argument("--semantic-aware", action="store_true", default=True, help="Use semantic hints (default: True)")
    parser.add_argument("--no-semantic-aware", dest="semantic_aware", action="store_false", help="Disable semantic hints (semantic-blind mode)")
    parser.add_argument("--semantic-session", dest="semantic_use_session", action="store_true", default=True, help="Reuse session IDs for state persistence (default: True)")
    parser.add_argument("--no-semantic-session", dest="semantic_use_session", action="store_false", help="Disable session reuse for semantics")
    return parser.parse_args()


def load_audio(args: argparse.Namespace) -> Tuple[torch.Tensor, int]:
    target_sr = args.sample_rate
    if args.dummy_audio:
        duration = args.dummy_audio
        t = torch.linspace(0, duration, int(target_sr * duration), dtype=torch.float32)
        waveform = 0.2 * torch.sin(2 * math.pi * 440 * t)
        return waveform.unsqueeze(0), target_sr

    if not args.audio_file:
        raise ValueError("Provide --audio-file or --dummy-audio")
    waveform, sr = torchaudio.load(str(args.audio_file))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def chunk_waveform(
    waveform: torch.Tensor,
    sr: int,
    chunk_seconds: float,
    stride_seconds: float,
) -> Iterable[Tuple[torch.Tensor, float, float]]:
    total_samples = waveform.size(-1)
    chunk_samples = max(int(chunk_seconds * sr), 1)
    stride_samples = max(int(stride_seconds * sr), 1)
    start = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_samples)
        chunk = waveform[:, start:end]
        yield chunk, start / sr, end / sr
        if end == total_samples:
            break
        start += stride_samples


def prepare_model(args: argparse.Namespace):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    # For remote execution, keep model on CPU
    device = args.device if args.backend != "djinn" else "cpu"
    if device != "cpu":
        model.to(device)
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    return model, processor, forced_decoder_ids


def bytes_of_tensor(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.numel()


def _stage_payload_to_tensor(payload: Any) -> torch.Tensor:
    """Extract tensor from stage execution result payload."""
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        # Check for SecureSerializer tensor format (_is_tensor metadata without data)
        # This happens when tensor data is in binary section but not reconstructed
        if payload.get('_is_tensor') and 'shape' in payload and 'dtype' in payload:
            # This is metadata-only, tensor data should be in binary section
            # For now, raise error to indicate we need to handle this properly
            raise ValueError(
                f"Tensor metadata found but data not reconstructed. "
                f"This indicates SecureSerializer deserialization issue. "
                f"Metadata: {list(payload.keys())}"
            )
        # Check for serialized tensor format (from SecureSerializer with data)
        if 'data' in payload and 'shape' in payload:
            return _deserialize_stage_tensor(payload)
        # Check for 'result' key (server response format)
        if 'result' in payload:
            result = payload['result']
            if torch.is_tensor(result):
                return result
            # If result is a dict, check for serialized tensor format first
            if isinstance(result, dict):
                # Check for SecureSerializer format with data
                if 'data' in result and 'shape' in result:
                    return _deserialize_stage_tensor(result)
                # Check for SecureSerializer metadata-only format
                if result.get('_is_tensor') and 'shape' in result:
                    raise ValueError(
                        f"Result tensor metadata found but data not reconstructed. "
                        f"Keys: {list(result.keys())}"
                    )
                # Check for 'logits' key (common in HuggingFace outputs)
                if 'logits' in result:
                    logits = result['logits']
                    if torch.is_tensor(logits):
                        return logits
                    if isinstance(logits, dict):
                        if 'data' in logits and 'shape' in logits:
                            return _deserialize_stage_tensor(logits)
                        if logits.get('_is_tensor'):
                            raise ValueError(f"Logits tensor metadata found but data not reconstructed")
                # Check for any tensor values in the dict
                for key, value in result.items():
                    if torch.is_tensor(value):
                        return value
                    if isinstance(value, dict):
                        if 'data' in value and 'shape' in value:
                            return _deserialize_stage_tensor(value)
                        if value.get('_is_tensor'):
                            raise ValueError(f"Tensor metadata for '{key}' found but data not reconstructed")
            # Recursively search
            return _stage_payload_to_tensor(result)
        # Check for 'logits' key directly
        if 'logits' in payload:
            logits = payload['logits']
            if torch.is_tensor(logits):
                return logits
            if isinstance(logits, dict):
                if 'data' in logits and 'shape' in logits:
                    return _deserialize_stage_tensor(logits)
                if logits.get('_is_tensor'):
                    raise ValueError(f"Logits tensor metadata found but data not reconstructed")
        # Recursively search for tensors or serialized tensors
        for value in payload.values():
            try:
                return _stage_payload_to_tensor(value)
            except ValueError:
                continue
    if isinstance(payload, (list, tuple)):
        for value in payload:
            try:
                return _stage_payload_to_tensor(value)
            except ValueError:
                continue
    raise ValueError(f"Unsupported stage payload type: {type(payload)}")


def _deserialize_stage_tensor(serialized: Dict[str, Any]) -> torch.Tensor:
    data = serialized.get('data')
    shape = serialized.get('shape', [])
    dtype_str = serialized.get('dtype', 'float32')
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("Serialized tensor missing binary data")
    if isinstance(dtype_str, str) and dtype_str.startswith('torch.'):
        dtype_str = dtype_str.replace('torch.', '')
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'bfloat16': np.float16,
        'int64': np.int64,
        'int32': np.int32,
        'bool': np.bool_,
    }
    numpy_dtype = dtype_map.get(dtype_str, np.float32)
    np_array = np.frombuffer(data, dtype=numpy_dtype).copy()
    if shape:
        np_array = np_array.reshape(shape)
    return torch.from_numpy(np_array)


def transcribe_chunks(
    model,
    processor,
    forced_decoder_ids,
    waveform: torch.Tensor,
    sr: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    chunk_records: List[Dict[str, Any]] = []
    decoder_prompt = build_decoder_prompt_tensor(forced_decoder_ids)

    sampler: Optional[NvmlSampler] = None
    if args.sample_gpu and device.type == "cuda" and _NVML_AVAILABLE:
        sampler = NvmlSampler(device.index or 0, interval_s=args.nvml_interval)

    if sampler:
        sampler.__enter__()
    run_start = time.perf_counter()

    for chunk_id, (chunk, start_s, end_s) in enumerate(
        chunk_waveform(waveform, sr, args.chunk_seconds, args.stride_seconds), start=1
    ):
        inputs = processor(chunk.squeeze(0), sampling_rate=sr, return_tensors="pt")
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        input_features = inputs.input_features.to(device).to(torch_dtype)
        host_to_device_mb = bytes_of_tensor(input_features) / (1024**2)

        with torch.no_grad():
            start = time.perf_counter()
            generated_ids = model.generate(
                input_features,
                max_new_tokens=args.max_new_tokens,
                forced_decoder_ids=forced_decoder_ids,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_ms = (time.perf_counter() - start) * 1000.0

        generated_cpu = generated_ids.to("cpu")
        prompt_len = decoder_prompt.shape[-1]
        decoded_slice = generated_cpu[:, prompt_len:]
        tokens = decoded_slice.shape[-1]
        per_token_ms = total_ms / max(tokens, 1)
        device_to_host_mb = bytes_of_tensor(generated_cpu) / (1024**2)

        chunk_text = (
            processor.batch_decode(decoded_slice, skip_special_tokens=True)[0].strip()
            if args.save_chunk_text
            else None
        )

        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "start_s": start_s,
                "end_s": end_s,
                "chunk_duration_s": end_s - start_s,
                "latency_ms": total_ms,
                "per_token_ms": per_token_ms,
                "tokens": tokens,
                "host_to_device_mb": host_to_device_mb,
                "device_to_host_mb": device_to_host_mb,
                "transcript": chunk_text,
            }
        )

    total_time_ms = (time.perf_counter() - run_start) * 1000.0
    if sampler:
        sampler.__exit__(None, None, None)
        gpu_summary = sampler.summary()
    else:
        gpu_summary = GpuSampleSummary()

    return {
        "chunks": chunk_records,
        "total_time_ms": total_time_ms,
        "gpu": gpu_summary.__dict__,
    }


async def transcribe_chunks_remote(
    manager,
    model: WhisperForConditionalGeneration,
    processor,
    decoder_prompt: torch.LongTensor,
    waveform: torch.Tensor,
    sr: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Remote execution path using EnhancedModelManager with encoder state caching."""
    chunk_records: List[Dict[str, Any]] = []
    run_start = time.perf_counter()

    encoder_handle = None
    remote_session: Optional[str] = None

    for chunk_id, (chunk, start_s, end_s) in enumerate(
        chunk_waveform(waveform, sr, args.chunk_seconds, args.stride_seconds), start=1
    ):
        inputs_dict = processor(chunk.squeeze(0), sampling_rate=sr, return_tensors="pt")
        # Match model dtype (model is loaded with args.dtype, default float16)
        model_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(args.dtype, torch.float16)
        input_features = inputs_dict.input_features.to(dtype=model_dtype).contiguous()

        semantic_aware = getattr(args, 'semantic_aware', True)
        semantic_use_session = getattr(args, 'semantic_use_session', True)
        
        if encoder_handle is None:
            encoder_inputs = {
                "input_features": input_features,
            }
            encode_start = time.perf_counter()
            import djinn
            if semantic_aware and semantic_use_session:
                with djinn.session(phase="encode", priority="normal"):
                    encoder_handle, remote_session = await manager.execute_encoder_stage(
                        model,
                        encoder_inputs=encoder_inputs,
                        model_id=args.model_id,
                        session_id=remote_session,
                        handle_metadata={"chunk_id": chunk_id},
                    )
            else:
                encoder_handle, remote_session = await manager.execute_encoder_stage(
                    model,
                    encoder_inputs=encoder_inputs,
                    model_id=args.model_id,
                    session_id=remote_session if semantic_use_session else None,
                    handle_metadata={"chunk_id": chunk_id},
                )
            encode_total_ms = (time.perf_counter() - encode_start) * 1000.0
            host_to_device_mb = bytes_of_tensor(input_features) / (1024**2)
            device_to_host_mb = 0.0
            latency_ms = encode_total_ms
            per_token_ms = 0.0
            tokens = 0
            chunk_text = None
        else:
            decoder_input_ids = decoder_prompt.clone()
            start = time.perf_counter()
            generated_ids = decoder_input_ids.clone()
            import djinn
            for token_idx in range(args.max_new_tokens):
                inputs = {
                    "decoder_input_ids": generated_ids,
                }
                if semantic_aware and semantic_use_session:
                    with djinn.session(phase="decode", priority="normal", expected_tokens=args.max_new_tokens - token_idx):
                        result_payload, remote_session = await manager.execute_decoder_stage(
                            model,
                            decoder_inputs=inputs,
                            state_handle=encoder_handle,
                            model_id=args.model_id,
                            session_id=remote_session,
                        )
                else:
                    result_payload, remote_session = await manager.execute_decoder_stage(
                        model,
                        decoder_inputs=inputs,
                        state_handle=encoder_handle,
                        model_id=args.model_id,
                        session_id=remote_session if semantic_use_session else None,
                    )
                logits = _stage_payload_to_tensor(result_payload)

                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if processor.tokenizer.eos_token_id is not None and torch.all(
                    next_token == processor.tokenizer.eos_token_id
                ):
                    break
            total_ms = (time.perf_counter() - start) * 1000.0

            generated_cpu = generated_ids.to("cpu")
            tokens = generated_cpu.shape[-1]
            per_token_ms = total_ms / max(tokens, 1)
            host_to_device_mb = 0.0  # encoder state already resident on server
            device_to_host_mb = bytes_of_tensor(generated_cpu) / (1024**2)
            chunk_text = (
                processor.batch_decode(generated_cpu, skip_special_tokens=True)[0].strip()
                if args.save_chunk_text
                else None
            )
            latency_ms = total_ms

        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "start_s": start_s,
                "end_s": end_s,
                "chunk_duration_s": end_s - start_s,
                "latency_ms": latency_ms,
                "per_token_ms": per_token_ms,
                "tokens": tokens,
                "host_to_device_mb": host_to_device_mb,
                "device_to_host_mb": device_to_host_mb,
                "transcript": chunk_text,
            }
        )

    total_time_ms = (time.perf_counter() - run_start) * 1000.0
    return {
        "chunks": chunk_records,
        "total_time_ms": total_time_ms,
        "gpu": {"gpu_util_pct": None, "mem_util_pct": None, "mem_used_mb": None},
    }


async def run_remote_streaming_baseline(args: argparse.Namespace) -> None:
    """Remote execution path for streaming audio."""
    from djinn.core.coordinator import get_coordinator
    from djinn.core.enhanced_model_manager import EnhancedModelManager

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    waveform, sr = load_audio(args)
    model, processor, forced_decoder_ids = prepare_model(args)
    # Configure model to return logits only so Djinn receives tensors
    model.config.return_dict = False
    model.config.output_hidden_states = False
    model.config.use_cache = False
    decoder_prompt = build_decoder_prompt_tensor(forced_decoder_ids)

    coordinator = get_coordinator()
    if coordinator is None:
        raise RuntimeError("Djinn coordinator unavailable. Ensure djinn.init() ran successfully.")

    manager = EnhancedModelManager(coordinator=coordinator)
    manager.use_model_cache = True

    await manager.register_model(model, model_id=args.model_id)

    # Warmup
    for _ in range(args.warmup_runs):
        await transcribe_chunks_remote(
            manager,
            model,
            processor,
            decoder_prompt,
            waveform,
            sr,
            args,
        )

    runs: List[Dict[str, Any]] = []
    for run_id in range(1, args.runs + 1):
        result = await transcribe_chunks_remote(
            manager,
            model,
            processor,
            decoder_prompt,
            waveform,
            sr,
            args,
        )
        aggregates = aggregate(result["chunks"])
        run_entry = {
            "run_id": run_id,
            "total_time_ms": result["total_time_ms"],
            "chunks": result["chunks"],
            "aggregates": aggregates,
            "gpu": result["gpu"],
        }
        runs.append(run_entry)
        print(
            f"[run {run_id}/{args.runs}] total={run_entry['total_time_ms']:.2f}ms "
            f"(mean chunk latency={aggregates['chunk_latency_ms']['mean']:.2f} ms)"
        )

    device_desc = f"Djinn remote ({getattr(args, 'djinn_server_address', 'default')})"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"streaming_audio_{args.tag}_{timestamp}.json"

    payload = {
        "baseline": args.tag,
        "model_id": args.model_id,
        "device": device_desc,
        "dtype": args.dtype,
        "backend": args.backend,
        "djinn_server": getattr(args, "djinn_server_address", None),
        "sample_rate": sr,
        "chunk_seconds": args.chunk_seconds,
        "stride_seconds": args.stride_seconds,
        "runs": runs,
        "library_versions": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": sys.modules["transformers"].__version__,
        },
        "meta": {
            "audio_file": str(args.audio_file) if args.audio_file else None,
            "dummy_audio_seconds": args.dummy_audio,
            "language": args.language,
            "task": args.task,
            "sample_gpu": False,  # GPU sampling not available for remote
        },
    }

    with output_file.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved remote streaming baseline results to {output_file}")


def aggregate(chunk_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = sorted(record["latency_ms"] for record in chunk_records)
    per_token = sorted(record["per_token_ms"] for record in chunk_records)
    throughput = []
    for record in chunk_records:
        duration = max(record["chunk_duration_s"], 1e-6)
        throughput.append(duration / (record["latency_ms"] / 1000.0))

    def stats(values):
        return {
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": _percentile(values, 95.0),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "chunk_latency_ms": stats(latencies),
        "chunk_per_token_ms": stats(per_token),
        "chunk_throughput_audio_sec_per_sec": stats(throughput),
    }


def main() -> None:
    args = parse_args()
    configure_remote_backend(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.backend == "djinn":
        from djinn.backend.runtime import initialization as djinn_runtime_init

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(run_remote_streaming_baseline(args))
        finally:
            loop.run_until_complete(djinn_runtime_init.shutdown())
            if not loop.is_running():
                loop.close()
        return

    waveform, sr = load_audio(args)
    model, processor, forced_decoder_ids = prepare_model(args)

    device_obj = torch.device(args.device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        idx = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        device_desc = torch.cuda.get_device_name(idx)
    else:
        device_desc = str(device_obj)

    # Warmup (single pass)
    for _ in range(args.warmup_runs):
        transcribe_chunks(model, processor, forced_decoder_ids, waveform, sr, args)

    runs: List[Dict[str, Any]] = []
    for run_id in range(1, args.runs + 1):
        result = transcribe_chunks(model, processor, forced_decoder_ids, waveform, sr, args)
        aggregates = aggregate(result["chunks"])
        run_entry = {
            "run_id": run_id,
            "total_time_ms": result["total_time_ms"],
            "chunks": result["chunks"],
            "aggregates": aggregates,
            "gpu": result["gpu"],
        }
        runs.append(run_entry)
        print(
            f"[run {run_id}/{args.runs}] total={run_entry['total_time_ms']:.2f}ms "
            f"(mean chunk latency={aggregates['chunk_latency_ms']['mean']:.2f} ms)"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"streaming_audio_{args.tag}_{timestamp}.json"

    payload = {
        "baseline": args.tag,
        "model_id": args.model_id,
        "device": device_desc,
        "dtype": args.dtype,
        "backend": args.backend,
        "djinn_server": getattr(args, "djinn_server_address", None),
        "sample_rate": sr,
        "chunk_seconds": args.chunk_seconds,
        "stride_seconds": args.stride_seconds,
        "runs": runs,
        "library_versions": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": sys.modules["transformers"].__version__,
        },
        "meta": {
            "audio_file": str(args.audio_file) if args.audio_file else None,
            "dummy_audio_seconds": args.dummy_audio,
            "language": args.language,
            "task": args.task,
            "sample_gpu": args.sample_gpu and _NVML_AVAILABLE,
        },
    }

    with output_file.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved streaming baseline results to {output_file}")


if __name__ == "__main__":
    main()

