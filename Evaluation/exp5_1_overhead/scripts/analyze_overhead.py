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
    for baseline in workload_block["results"]:
        aggregates = baseline["aggregates"]
        derived = baseline.get("derived", {})
        row = {
            "workload": workload_name,
            "category": workload_block.get("category"),
            "baseline": baseline["baseline"],
            "runner_type": baseline.get("runner_type"),
            "latency_mean_ms": _safe_get(aggregates, "latency_ms", "mean"),
            "latency_p95_ms": _safe_get(aggregates, "latency_ms", "p95"),
            "data_mean_mb": _safe_get(aggregates, "total_data_mb", "mean"),
            "throughput_mean_units_s": _safe_get(aggregates, "throughput_units_per_s", "mean"),
        }
        for key, value in derived.items():
            row[key] = value
        row["source_file"] = str(path)
        row["experiment_tag"] = experiment.get("tag")
        rows.append(row)
    return rows


def _safe_get(aggregates: Dict[str, Dict], section: str, field: str) -> Optional[float]:
    block = aggregates.get(section)
    if not block:
        return None
    return block.get(field)


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
        "data_mean_mb",
        "throughput_mean_units_s",
        "latency_overhead_pct_vs_native_pytorch",
        "speedup_vs_semantic_blind",
        "data_savings_pct_vs_semantic_blind",
        "semantic_efficiency_ratio",
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
        "Data (MB)",
        "Speedup vs Blind",
        "Data Savings (%)",
        "Semantic Eff. Ratio",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- "] * len(header)) + "|")
    for row in rows:
        print(
            "| {workload} | {baseline} | {lat:.2f} | {p95:.2f} | {data:.3f} | {speedup:.2f} | {savings:.2f} | {eff:.2f} |".format(
                workload=row["workload"],
                baseline=row["baseline"],
                lat=row.get("latency_mean_ms", 0.0) or 0.0,
                p95=row.get("latency_p95_ms", 0.0) or 0.0,
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


