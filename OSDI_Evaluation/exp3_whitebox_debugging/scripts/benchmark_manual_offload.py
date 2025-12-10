#!/usr/bin/env python3
"""
Experiment 3 Baseline: Manual CPU Offload (Resume Latency vs. Depth)

Measures the time to move activations + KV cache at breakpoint layer L from CPU (pinned)
back to GPU, representing a hand-written tensor.to('cpu') / tensor.to('cuda')
approach. Serves as the "speed-of-light" PCIe baseline.

Data transferred: Hidden state + cumulative KV cache up to layer L
  - Hidden state: [batch=1, seq_len, hidden_dim] = [1, 2048, 5120] @ fp16 (~10.56 MB)
  - KV cache per layer: 2 * [batch=1, seq_len, num_heads, head_dim] @ fp16
    - Total for L layers: grows with L (approximately 4 * seq_len * hidden_dim * L bytes)
Compute cost: O(1) (just DMA transfer)
Memory cost: O(seq_len * hidden_dim * L)
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel


def prepare_inputs(tokenizer, prompt: str, max_length: int, device: torch.device):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in encoded.items()}


@torch.no_grad()
def compute_activation_at_layer(
    llama_model: LlamaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer: int,
) -> tuple:
    """
    Forward to target_layer and return (hidden_states, kv_cache).
    KV cache contains accumulated key-value pairs up to target_layer.
    """
    device = input_ids.device

    batch, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

    attn_mask = None
    if hasattr(llama_model, "_prepare_decoder_attention_mask"):
        attn_mask = llama_model._prepare_decoder_attention_mask(
            attention_mask,
            (batch, seq_len),
            input_ids,
            device,
            input_ids.dtype,
        )

    hidden_states = llama_model.embed_tokens(input_ids)
    rotary_emb = llama_model.rotary_emb
    position_embeddings = rotary_emb(hidden_states, position_ids)

    # Collect KV cache as we run through layers
    kv_cache = []
    for idx, layer in enumerate(llama_model.layers):
        output = layer(
            hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,  # Enable caching
            cache_position=None,
            position_embeddings=position_embeddings,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        past_key_value = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if past_key_value is not None:
            kv_cache.append(past_key_value)
        
        if (idx + 1) >= target_layer:
            break

    hidden_states = llama_model.norm(hidden_states)
    return hidden_states, kv_cache


def benchmark_offload(
    llama_model: LlamaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer: int,
) -> tuple:
    """
    Compute activation + KV cache at layer L, move to CPU (pinned),
    then measure CPU->GPU transfer time (non_blocking + sync).
    Returns (latency_ms, total_data_mb).
    """
    device = input_ids.device

    # Compute activation + KV cache on GPU
    act_gpu, kv_cache_gpu = compute_activation_at_layer(
        llama_model,
        input_ids,
        attention_mask,
        target_layer=target_layer,
    )

    # Calculate total data size (hidden state + KV cache)
    hidden_state_bytes = act_gpu.numel() * act_gpu.element_size()
    kv_cache_bytes = 0
    for kv_pair in kv_cache_gpu:
        if isinstance(kv_pair, tuple) and len(kv_pair) == 2:
            k, v = kv_pair
            kv_cache_bytes += k.numel() * k.element_size()
            kv_cache_bytes += v.numel() * v.element_size()
    total_bytes = hidden_state_bytes + kv_cache_bytes
    total_mb = total_bytes / 1e6

    # Move to pinned CPU memory
    act_cpu = act_gpu.detach().to("cpu", non_blocking=True).pin_memory()
    kv_cache_cpu = []
    for kv_pair in kv_cache_gpu:
        if isinstance(kv_pair, tuple) and len(kv_pair) == 2:
            k, v = kv_pair
            k_cpu = k.detach().to("cpu", non_blocking=True).pin_memory()
            v_cpu = v.detach().to("cpu", non_blocking=True).pin_memory()
            kv_cache_cpu.append((k_cpu, v_cpu))

    torch.cuda.synchronize(device)
    start = time.perf_counter()

    # Move back to GPU (simulate resume)
    _ = act_cpu.to(device, non_blocking=True)
    for k_cpu, v_cpu in kv_cache_cpu:
        _ = k_cpu.to(device, non_blocking=True)
        _ = v_cpu.to(device, non_blocking=True)

    torch.cuda.synchronize(device)
    end = time.perf_counter()

    return (end - start) * 1000.0, total_mb  # ms, MB


def run_benchmark(
    model_name: str,
    layers: List[int],
    max_length: int,
    output_path: Path,
    warmup_iters: int,
    repeat_iters: int,
):
    assert torch.cuda.is_available(), "CUDA GPU required for this benchmark."
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    llama_model: LlamaModel = model.model
    llama_model.eval()

    prompt = "The future of AI is" + " context token" * (max_length // 4)
    inputs = prepare_inputs(tokenizer, prompt, max_length, device)

    results: List[Dict] = []
    for layer in layers:
        # Warmup
        for _ in range(warmup_iters):
            _ = benchmark_offload(
                llama_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                target_layer=layer,
            )

        measurements = [
            benchmark_offload(
                llama_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                target_layer=layer,
            )
            for _ in range(repeat_iters)
        ]
        latencies = [m[0] for m in measurements]
        data_sizes = [m[1] for m in measurements]
        
        mean_ms = statistics.mean(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        mean_mb = statistics.mean(data_sizes)

        results.append(
            {
                "layer": layer,
                "resume_latency_ms": mean_ms,
                "resume_latency_std_ms": std_ms,
                "num_samples": repeat_iters,
                "data_transferred_mb": mean_mb,
                "note": "Includes hidden state + cumulative KV cache",
            }
        )
        print(f"[Manual Offload] Layer {layer}: mean={mean_ms:.1f} ms, std={std_ms:.1f} ms over {repeat_iters} runs (data ~{mean_mb:.2f} MB)")

    # Gather environment metadata for reproducibility
    import sys
    import platform
    metadata = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "platform": platform.platform(),
    }
    
    output = {
        "model": model_name,
        "layers": layers,
        "results": results,
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Manual offload benchmark saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Manual CPU offload baseline: resume latency vs depth")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/exp3_resume_results/manual_offload_latency.json"),
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per layer")
    parser.add_argument("--repeat", type=int, default=5, help="Measured iterations per layer")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        layers=args.layers,
        max_length=args.max_length,
        output_path=args.output,
        warmup_iters=max(args.warmup, 0),
        repeat_iters=max(args.repeat, 1),
    )


if __name__ == "__main__":
    main()
