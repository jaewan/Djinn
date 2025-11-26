"""
Utilities for estimating transformer KV cache sizes using architecture metadata.
"""

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Union

import torch

DTYPE_STR_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "single": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
}

DTYPE_SIZE_MAP = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}


@dataclass(frozen=True)
class TransformerKVSpec:
    """Normalized transformer dimensions used for KV cache estimates."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int


def _pick(config: Dict[str, Any], aliases: Iterable[str]) -> Optional[int]:
    for key in aliases:
        value = config.get(key)
        if value is not None:
            return value
    return None


def _dtype_to_bytes(dtype: Optional[Union[str, torch.dtype]], default: int = 2) -> int:
    if dtype is None:
        return default

    if isinstance(dtype, torch.dtype):
        return DTYPE_SIZE_MAP.get(dtype, default)

    if isinstance(dtype, str):
        resolved = DTYPE_STR_ALIASES.get(dtype.lower())
        if resolved:
            return DTYPE_SIZE_MAP.get(resolved, default)

    return default


def normalize_config_sources(*configs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for config in configs:
        if not config:
            continue
        for key, value in config.items():
            if value is None:
                continue
            merged[key] = value
    return merged


def build_transformer_kv_spec(
    config: Dict[str, Any],
    *,
    default_dtype_bytes: int = 2,
) -> Optional[TransformerKVSpec]:
    """
    Normalize raw architecture config into a TransformerKVSpec.
    """

    num_layers = _pick(config, ["num_layers", "num_hidden_layers", "n_layer"])
    hidden_size = _pick(config, ["hidden_size", "n_embd", "d_model", "embed_dim"])
    num_heads = _pick(config, ["num_heads", "num_attention_heads", "n_head"])
    head_dim = _pick(config, ["head_dim", "attention_head_size"])
    num_kv_heads = _pick(
        config,
        ["num_key_value_heads", "num_kv_heads", "n_head_kv", "n_kv_head"],
    )

    if head_dim is None and hidden_size and num_heads:
        with contextlib.suppress(Exception):
            head_dim = hidden_size // num_heads

    if num_kv_heads is None and num_heads:
        num_kv_heads = num_heads

    if not all([num_layers, num_heads, num_kv_heads, head_dim]):
        return None

    dtype_value = config.get("dtype") or config.get("torch_dtype")
    dtype_bytes = int(config.get("dtype_bytes") or _dtype_to_bytes(dtype_value, default_dtype_bytes))

    return TransformerKVSpec(
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        dtype_bytes=dtype_bytes,
    )


def kv_bytes_per_token(spec: TransformerKVSpec) -> int:
    """
    Base KV cache bytes required for a single token for the provided spec.
    """
    return 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * spec.dtype_bytes


def estimate_transformer_kv_bytes(
    config: Dict[str, Any],
    expected_tokens: int,
    *,
    overhead_ratio: float = 0.2,
) -> Optional[int]:
    """
    Estimate total KV cache bytes for a transformer workload.
    """
    spec = build_transformer_kv_spec(config)
    if not spec or expected_tokens <= 0:
        return None

    base = kv_bytes_per_token(spec) * expected_tokens
    return int(base * (1 + overhead_ratio))

