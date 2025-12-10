#!/usr/bin/env python3
"""
Experiment 3 Baseline: Stateless Recompute (Resume Latency vs. Depth)

Measures the time to recompute activations from input up to a breakpoint layer L.
This represents a "serverless" approach: delete intermediates on pause, recompute
from the start on resume. Used to compare against manual offload and Djinn.

Data transferred: NONE (recomputes from scratch)
Compute cost: O(L) where L = layer depth
Memory cost: O(1) (no checkpointing)
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

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
def recompute_to_layer(
    llama_model: LlamaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer: int,
) -> float:
    """
    Recompute from input -> target_layer.
    Returns elapsed time in milliseconds.
    """
    device = input_ids.device

    # Prepare masks/positions (mirrors HF Llama forward)
    batch, seq_len = input_ids.shape

    # Synchronize before starting timer to include all work (positions, masks, embeddings, layers)
    torch.cuda.synchronize(device)
    start = time.perf_counter()

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

    # Embeddings
    hidden_states = llama_model.embed_tokens(input_ids)
    rotary_emb = llama_model.rotary_emb
    position_embeddings = rotary_emb(hidden_states, position_ids)

    for idx, layer in enumerate(llama_model.layers):
        output = layer(
            hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        if (idx + 1) >= target_layer:
            break

    hidden_states = llama_model.norm(hidden_states)

    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return (end - start) * 1000.0  # ms


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
        # Warmup to stabilize CUDA kernels
        for _ in range(warmup_iters):
            _ = recompute_to_layer(
                llama_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                target_layer=layer,
            )

        latencies = [
            recompute_to_layer(
                llama_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                target_layer=layer,
            )
            for _ in range(repeat_iters)
        ]
        mean_ms = statistics.mean(latencies)
        median_ms = statistics.median(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        
        # Check for outliers
        outlier_count = sum(1 for lat in latencies if abs(lat - mean_ms) > 2 * std_ms) if std_ms > 0 else 0

        results.append(
            {
                "layer": layer,
                "resume_latency_ms": mean_ms,
                "resume_latency_median_ms": median_ms,
                "resume_latency_std_ms": std_ms,
                "num_samples": repeat_iters,
                "outlier_count": outlier_count,
            }
        )
        print(f"[Recompute] Layer {layer}: mean={mean_ms:.1f} ms, median={median_ms:.1f} ms, std={std_ms:.1f} ms over {repeat_iters} runs (outliers={outlier_count})")

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
    print(f"✅ Recompute benchmark saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Recompute baseline: resume latency vs depth")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/exp3_resume_results/recompute_latency.json"),
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per layer (increased for stability)")
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
