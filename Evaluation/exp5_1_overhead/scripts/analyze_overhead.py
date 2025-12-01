#!/usr/bin/env python3
"""
Aggregate Experiment 5.1 results into CSV or Markdown tables.

Reads the JSON payloads emitted by `run_overhead_sweep.py` and summarizes the
mean/p95 latency, data transfer, throughput, and derived metrics (speedup,
overhead, semantic efficiency).  Intended for quick iteration on the dev box
before producing publication-quality plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Evaluation/exp5_1_overhead/results"),
        help="Directory containing workload subdirectories with JSON outputs.",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        help="Optional workload name filter (default: include all).",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "markdown"),
        default="csv",
        help="Output format. CSV writes to file/stdout; Markdown prints a table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for CSV output. Defaults to stdout.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="If set, only analyze the most recent JSON per workload.",
    )
    return parser.parse_args()


def collect_result_files(results_dir: Path, workloads: Optional[Sequence[str]], latest_only: bool) -> List[Path]:
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")
    targets = []
    workload_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    for workload_dir in workload_dirs:
        name = workload_dir.name
        if workloads and name not in workloads:
            continue
        json_files = sorted(workload_dir.glob("*.json"))
        if not json_files:
            continue
        if latest_only:
            targets.append(json_files[-1])
        else:
            targets.extend(json_files)
    if workloads and not targets:
        raise SystemExit(f"No matching results for workloads: {', '.join(workloads)}")
    return targets


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    experiment = payload.get("experiment", {})
    workload_block = payload["workload"]
    workload_name = workload_block["workload"]
    rows: List[Dict[str, Any]] = []
    
    # Find native baseline latency for overhead calculation
    # See: OSDI Review - Critique 2 (Metric Inconsistency)
    native_latency_ms = None
    for baseline in workload_block["results"]:
        if baseline["baseline"] == "native_pytorch":
            aggregates = baseline["aggregates"]
            native_latency_ms = _safe_get(aggregates, "latency_ms", "mean")
            break
    
    for baseline in workload_block["results"]:
        aggregates = baseline["aggregates"]
        derived = baseline.get("derived", {})
        latency_ms = _safe_get(aggregates, "latency_ms", "mean")
        
        # CRITICAL: latency_delta_vs_native_ms shows the difference from native baseline
        # Positive = slower than native (overhead), Negative = faster than native (improvement)
        # Note: "Faster than native" typically indicates different computations being measured
        # See: OSDI Review - Critique 2
        latency_delta_vs_native_ms = None
        if latency_ms is not None and native_latency_ms is not None:
            latency_delta_vs_native_ms = latency_ms - native_latency_ms
        
        row = {
            "workload": workload_name,
            "category": workload_block.get("category"),
            "baseline": baseline["baseline"],
            "runner_type": baseline.get("runner_type"),
            "latency_mean_ms": latency_ms,
            "latency_p95_ms": _safe_get(aggregates, "latency_ms", "p95"),
            "latency_delta_vs_native_ms": latency_delta_vs_native_ms,
            "data_mean_mb": _safe_get(aggregates, "total_data_mb", "mean"),
            "throughput_mean_units_s": _safe_get(aggregates, "throughput_units_per_s", "mean"),
        }
        for key, value in derived.items():
            row[key] = value
        _attach_profiling_columns(row, aggregates)
        row["source_file"] = str(path)
        row["experiment_tag"] = experiment.get("tag")
        rows.append(row)
    return rows


def _safe_get(aggregates: Dict[str, Dict], section: str, field: str) -> Optional[float]:
    block = aggregates.get(section)
    if not block:
        return None
    return block.get(field)


def _attach_profiling_columns(row: Dict[str, Any], aggregates: Dict[str, Dict[str, Any]]) -> None:
    mapping = {
        "client_serialize_mean_ms": ("client_serialize_ms", "mean"),
        "client_deserialize_mean_ms": ("client_deserialize_ms", "mean"),
        "client_network_c2s_mean_ms": ("client_network_c2s_ms", "mean"),
        "client_network_s2c_mean_ms": ("client_network_s2c_ms", "mean"),
        "server_duration_mean_ms": ("server_duration_ms", "mean"),
        "server_executor_time_mean_ms": ("server_executor_time_ms", "mean"),
        "server_queue_latency_mean_ms": ("server_queue_latency_ms", "mean"),
        "server_plan_mean_ms": ("server_plan_ms", "mean"),
        "server_placement_mean_ms": ("server_placement_ms", "mean"),
        "server_execution_mean_ms": ("server_execution_ms", "mean"),
        "server_skeletonization_mean_ms": ("server_skeletonization_ms", "mean"),
        "server_cleanup_mean_ms": ("server_cleanup_ms", "mean"),
    }
    for column, (section, field) in mapping.items():
        row[column] = _safe_get(aggregates, section, field)


def write_csv(rows: List[Dict[str, Any]], output: Optional[Path]) -> None:
    if not rows:
        print("No rows to write.")
        return
    fieldnames = [
        "workload",
        "category",
        "baseline",
        "latency_mean_ms",
        "latency_p95_ms",
        "latency_delta_vs_native_ms",  # OSDI FIX: Renamed from 'overhead' - can be negative
        "data_mean_mb",
        "throughput_mean_units_s",
        "latency_overhead_pct_vs_native_pytorch",
        "speedup_vs_semantic_blind",
        "data_savings_pct_vs_semantic_blind",
        "semantic_efficiency_ratio",
        "client_serialize_mean_ms",
        "client_deserialize_mean_ms",
        "client_network_c2s_mean_ms",
        "client_network_s2c_mean_ms",
        "server_duration_mean_ms",
        "server_executor_time_mean_ms",
        "server_queue_latency_mean_ms",
        "server_plan_mean_ms",
        "server_placement_mean_ms",
        "server_execution_mean_ms",
        "server_skeletonization_mean_ms",
        "server_cleanup_mean_ms",
        "source_file",
    ]
    fh = output.open("w", newline="") if output else sys.stdout
    close_fh = output is not None
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})
    if output:
        print(f"Wrote CSV summary to {output}")
    if close_fh:
        fh.close()


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No rows to display.")
        return
    header = [
        "Workload",
        "Baseline",
        "Latency (ms)",
        "P95 (ms)",
        "Delta vs Native (ms)",  # OSDI FIX: Renamed - can be negative
        "Data (MB)",
        "Speedup vs Blind",
        "Data Savings (%)",
        "Semantic Eff. Ratio",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- "] * len(header)) + "|")
    for row in rows:
        # Use .3f for sub-millisecond precision to avoid losing information
        # See: OSDI Review - Code Nit #3 (Markdown Precision)
        delta = row.get("latency_delta_vs_native_ms", 0.0) or 0.0
        # Format delta with sign for clarity (positive = slower, negative = faster)
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(
            "| {workload} | {baseline} | {lat:.3f} | {p95:.3f} | {delta} | {data:.3f} | {speedup:.2f} | {savings:.2f} | {eff:.2f} |".format(
                workload=row["workload"],
                baseline=row["baseline"],
                lat=row.get("latency_mean_ms", 0.0) or 0.0,
                p95=row.get("latency_p95_ms", 0.0) or 0.0,
                delta=delta_str,
                data=row.get("data_mean_mb", 0.0) or 0.0,
                speedup=row.get("speedup_vs_semantic_blind") or 0.0,
                savings=row.get("data_savings_pct_vs_semantic_blind") or 0.0,
                eff=row.get("semantic_efficiency_ratio") or 0.0,
            )
        )


def main() -> None:
    args = parse_args()
    targets = collect_result_files(args.results_dir, args.workloads, args.latest_only)
    all_rows: List[Dict[str, Any]] = []
    for path in targets:
        all_rows.extend(load_rows(path))
    if args.format == "csv":
        write_csv(all_rows, args.output)
    else:
        print_markdown(all_rows)


if __name__ == "__main__":
    main()


