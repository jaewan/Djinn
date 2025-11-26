"""
Common statistics helpers for evaluation harnesses.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Optional


def percentile(values: List[float], pct: float) -> Optional[float]:
    """Return the requested percentile via linear interpolation."""
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


def summarize(values: Iterable[float]) -> Dict[str, Optional[float]]:
    """Compute summary stats used in evaluation payloads."""
    arr = list(values)
    if not arr:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(arr),
        "median": statistics.median(arr),
        "p95": percentile(arr, 95.0),
        "min": min(arr),
        "max": max(arr),
    }


def summarize_fields(rows: List[Dict[str, float]], fields: Iterable[str]) -> Dict[str, Dict]:
    """Return summaries for each requested numeric field."""
    summary: Dict[str, Dict] = {}
    for field in fields:
        summary[field] = summarize(row[field] for row in rows if field in row)
    return summary


__all__ = ["percentile", "summarize", "summarize_fields"]


