#!/usr/bin/env python3
"""
Summarize Experiment 6.1 generality results.

Consumes the JSON emitted by `run_generality_suite.py` and prints Markdown/CSV
tables for both per-workload baselines and per-group semantic efficiency
statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-file",
        type=Path,
        default=Path("Evaluation/exp6_1_generality/results/smoke.json"),
        help="Path to the JSON result bundle.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "markdown"),
        default="markdown",
        help="Output format for the per-group summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path when --format=csv (stdout otherwise).",
    )
    return parser.parse_args()


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def flatten_workloads(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        name = group["group"]
        for workload in group["workloads"]:
            for baseline in workload["results"]:
                aggregates = baseline["aggregates"]
                row = {
                    "group": name,
                    "workload": workload["workload"],
                    "baseline": baseline["baseline"],
                    "latency_mean_ms": _safe_get(aggregates, "latency_ms", "mean"),
                    "latency_p95_ms": _safe_get(aggregates, "latency_ms", "p95"),
                    "data_mean_mb": _safe_get(aggregates, "total_data_mb", "mean"),
                    "speedup_vs_semantic_blind": baseline.get("derived", {}).get("speedup_vs_semantic_blind"),
                    "data_savings_pct_vs_semantic_blind": baseline.get("derived", {}).get("data_savings_pct_vs_semantic_blind"),
                    "semantic_efficiency_ratio": baseline.get("derived", {}).get("semantic_efficiency_ratio"),
                }
                rows.append(row)
    return rows


def _safe_get(aggregates: Dict[str, Dict], section: str, field: str):
    block = aggregates.get(section)
    if not block:
        return None
    return block.get(field)


def write_group_summary(groups: List[Dict[str, Any]], fmt: str, output: Path | None) -> None:
    summary_rows = []
    for summary in groups:
        metrics = summary.get("metrics", {})
        summary_rows.append(
            {
                "group": summary["group"],
                "num_workloads": summary["num_workloads"],
                "speedup_mean": _metric(metrics, "speedup_vs_blind", "mean"),
                "speedup_p95": _metric(metrics, "speedup_vs_blind", "p95"),
                "data_savings_mean": _metric(metrics, "data_savings_pct_vs_blind", "mean"),
                "efficiency_mean": _metric(metrics, "semantic_efficiency_ratio", "mean"),
            }
        )

    if fmt == "csv":
        fieldnames = ["group", "num_workloads", "speedup_mean", "speedup_p95", "data_savings_mean", "efficiency_mean"]
        fh = output.open("w", newline="") if output else sys.stdout
        close_fh = output is not None
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        if close_fh:
            fh.close()
        if output:
            print(f"Wrote group summary CSV to {output}")
    else:
        print("| Group | # Workloads | Speedup (mean) | Speedup (p95) | Data Savings (%) | Semantic Eff. |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in summary_rows:
            print(
                f"| {row['group']} | {row['num_workloads']} | "
                f"{row['speedup_mean'] or 0:.2f} | {row['speedup_p95'] or 0:.2f} | "
                f"{row['data_savings_mean'] or 0:.2f} | {row['efficiency_mean'] or 0:.2f} |"
            )


def _metric(metrics: Dict[str, Dict[str, float]], field: str, stat: str) -> float | None:
    entry = metrics.get(field)
    if not entry:
        return None
    return entry.get(stat)


def main() -> None:
    args = parse_args()
    payload = load_payload(args.result_file)
    groups = payload["groups"]
    workloads = flatten_workloads(groups)

    print("Per-workload baselines:")
    print("| Group | Workload | Baseline | Latency (ms) | Data (MB) | Speedup | Data Savings | Sem. Eff. |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in workloads:
        print(
            "| {group} | {workload} | {baseline} | {lat:.2f} | {data:.3f} | {speedup:.2f} | {savings:.2f} | {eff:.2f} |".format(
                group=row["group"],
                workload=row["workload"],
                baseline=row["baseline"],
                lat=row["latency_mean_ms"] or 0.0,
                data=row["data_mean_mb"] or 0.0,
                speedup=row.get("speedup_vs_semantic_blind") or 0.0,
                savings=row.get("data_savings_pct_vs_semantic_blind") or 0.0,
                eff=row.get("semantic_efficiency_ratio") or 0.0,
            )
        )

    print("\nPer-group summary:")
    write_group_summary(payload["summaries"], args.format, args.output)


if __name__ == "__main__":
    main()


