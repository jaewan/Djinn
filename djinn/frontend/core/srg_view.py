"""
SRG View utilities.

Provides helpers to materialize a Semantically Rich Graph (SRG) view from
LazyTensor outputs and compute lightweight summaries for tooling (profilers,
capability reports, etc.) without duplicating the underlying DAG structures.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Set

import torch

from .lazy_tensor import LazyTensor


def _resolve_dtype(dtype_value: Any) -> Optional[torch.dtype]:
    """Best-effort conversion from stored dtype metadata to torch.dtype."""
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        token = dtype_value.split(".")[-1]
        return getattr(torch, token, None)
    return None


def _flatten_lazy_tensors(obj: Any) -> Iterable[LazyTensor]:
    """Yield LazyTensor instances from nested structures."""
    if isinstance(obj, LazyTensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _flatten_lazy_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_lazy_tensors(item)


def build_srg_view(root: Any, max_nodes: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Build a normalized SRG view starting from the provided LazyTensor outputs.

    Args:
        root: LazyTensor or nested structure containing LazyTensors.
        max_nodes: Optional cap to avoid traversing unbounded graphs.

    Returns:
        List of node descriptors with semantic metadata.
    """
    nodes: List[Dict[str, Any]] = []
    visited: Set[int] = set()
    queue = deque(_flatten_lazy_tensors(root))

    while queue:
        tensor = queue.popleft()
        if not isinstance(tensor, LazyTensor):
            continue

        tensor_id = getattr(tensor, "_tensor_id", id(tensor))
        if tensor_id in visited:
            continue
        visited.add(tensor_id)

        entry = {
            "id": tensor_id,
            "operation": getattr(tensor, "_operation", "unknown"),
            "shape": tuple(tensor.shape) if hasattr(tensor, "shape") and tensor.shape else None,
            "dtype": str(getattr(tensor, "dtype", "")) if hasattr(tensor, "dtype") else None,
            "semantic_class": getattr(tensor, "semantic_class", None),
            "lifecycle": getattr(tensor, "lifecycle", None),
            "phase": getattr(tensor, "detected_phase", None),
            "compute_cost": getattr(tensor, "compute_cost", None),
            "input_ids": [],
        }

        inputs = getattr(tensor, "_inputs", [])
        for inp in inputs:
            if isinstance(inp, LazyTensor):
                inp_id = getattr(inp, "_tensor_id", id(inp))
                entry["input_ids"].append(inp_id)
                queue.append(inp)

        nodes.append(entry)
        if max_nodes is not None and len(nodes) >= max_nodes:
            break

    return nodes


def summarize_srg_view(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics for an SRG view."""
    total = len(nodes)
    by_semantic = Counter()
    by_phase = Counter()
    lifecycle_bytes = Counter()

    for node in nodes:
        sem = node.get("semantic_class")
        if sem:
            by_semantic[str(sem)] += 1

        phase = node.get("phase")
        if phase:
            by_phase[phase] += 1

        lifecycle = node.get("lifecycle")
        shape = node.get("shape")
        dtype_obj = _resolve_dtype(node.get("dtype"))
        elem_size = dtype_obj.itemsize if dtype_obj is not None else 0

        if lifecycle and shape and elem_size:
            numel = 1
            for dim in shape:
                numel *= dim
            lifecycle_bytes[lifecycle] += numel * elem_size

    return {
        "total_nodes": total,
        "semantic_breakdown": dict(by_semantic),
        "phase_breakdown": dict(by_phase),
        "lifecycle_bytes": {k: v for k, v in lifecycle_bytes.items()},
    }

