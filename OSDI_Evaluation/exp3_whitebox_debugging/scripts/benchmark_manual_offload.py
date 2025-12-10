#!/usr/bin/env python3
"""
Experiment 3 Baseline: Manual CPU Offload (Resume Latency vs. Depth)

Measures the time to move activations at breakpoint layer L from CPU (pinned)
back to GPU, representing a hand-written tensor.to('cpu') / tensor.to('cuda')
approach. Serves as the "speed-of-light" PCIe baseline.

CRITICAL AUDIT: Djinn only checkpoints the hidden state activation, NOT KV cache.
To ensure apple-to-apple comparison, this benchmark ALSO transfers hidden state only.

Data transferred: Hidden state activation only (matches Djinn checkpoint scope)
  - Shape: [batch=1, seq_len, hidden_dim] = [1, 2048, 5120] @ fp16
  - Size: ~10.56 MB (constant, independent of depth L)
Compute cost: O(1) (just DMA transfer)
Memory cost: O(seq_len * hidden_dim) - constant with depth
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
) -> torch.Tensor:
    """
    Forward to target_layer and return hidden_states activation.
    MATCHES DJINN SCOPE: Djinn only checkpoints hidden state, not KV cache.
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

    # Run through layers, but do NOT collect KV cache (matches Djinn)
    for idx, layer in enumerate(llama_model.layers):
        output = layer(
            hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,  # Do NOT cache (to match Djinn checkpoint scope)
            cache_position=None,
            position_embeddings=position_embeddings,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        if (idx + 1) >= target_layer:
            break

    hidden_states = llama_model.norm(hidden_states)
    return hidden_states


def benchmark_offload(
    llama_model: LlamaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer: int,
) -> tuple:
    """
    Compute activation at layer L (hidden state only, no KV cache),
    move to CPU (pinned), then measure CPU->GPU transfer time.
    Returns (latency_ms, data_size_mb).
    
    MATCHES DJINN SCOPE: Only transfers hidden state activation.
    """
    device = input_ids.device

    # Compute activation on GPU (hidden state only)
    act_gpu = compute_activation_at_layer(
        llama_model,
        input_ids,
        attention_mask,
        target_layer=target_layer,
    )

    # Calculate data size (hidden state only)
    total_bytes = act_gpu.numel() * act_gpu.element_size()
    total_mb = total_bytes / 1e6

    # Move to pinned CPU memory
    act_cpu = act_gpu.detach().to("cpu", non_blocking=True).pin_memory()

    torch.cuda.synchronize(device)
    start = time.perf_counter()

    # Move back to GPU (simulate resume)
    _ = act_cpu.to(device, non_blocking=True)

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
        # Warmup (PCIe drivers are lazy, multiple iterations stabilize)
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
        median_ms = statistics.median(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        mean_mb = statistics.mean(data_sizes)
        
        # Check for outliers (PCIe timing can be noisy)
        outlier_count = sum(1 for lat in latencies if abs(lat - mean_ms) > 2 * std_ms) if std_ms > 0 else 0

        results.append(
            {
                "layer": layer,
                "resume_latency_ms": mean_ms,
                "resume_latency_median_ms": median_ms,
                "resume_latency_std_ms": std_ms,
                "num_samples": repeat_iters,
                "outlier_count": outlier_count,
                "data_transferred_mb": mean_mb,
                "note": "Hidden state only (matches Djinn checkpoint scope)",
            }
        )
        print(f"[Manual Offload] Layer {layer}: mean={mean_ms:.1f} ms, median={median_ms:.1f} ms, std={std_ms:.1f} ms over {repeat_iters} runs (data ~{mean_mb:.2f} MB, outliers={outlier_count})")

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
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per layer (PCIe drivers need stabilization)")
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
