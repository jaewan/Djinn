#!/usr/bin/env python3
"""
Experiment 6.1 – Cross-workload generality harness.

Runs grouped workloads (sequential / parallel / hybrid) and aggregates speedup,
data-savings, and semantic efficiency metrics per group.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.djinn_init import ensure_initialized
from Evaluation.common.experiment_runner import ExperimentRunner
from Evaluation.common.metrics import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Evaluation/exp6_1_generality/configs/generality_smoke.yaml"),
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        help="Optional list of workload group names to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file override.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional label for output filename.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _resolve_server_address(exp_cfg: Dict[str, Any]) -> Optional[str]:
    return exp_cfg.get("djinn_server_address") or os.environ.get("GENIE_SERVER_ADDRESS")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    experiment_cfg = cfg["experiment"]
    baselines_cfg = cfg["baselines"]
    groups_cfg = cfg["workload_groups"]

    if args.groups:
        selected = set(args.groups)
        groups_cfg = [group for group in groups_cfg if group["name"] in selected]
        if not groups_cfg:
            raise SystemExit("No workload groups matched the provided filter.")

    server_address = _resolve_server_address(experiment_cfg)
    ensure_initialized(server_address)

    runner = ExperimentRunner(experiment_cfg, baselines_cfg)
    group_outputs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for group in groups_cfg:
        workloads = group["workloads"]
        print(f"\n=== Group: {group['name']} ({len(workloads)} workloads) ===")
        workload_results = runner.run_workloads(workloads)
        for result in workload_results:
            _print_workload_summary(result)
        summary = summarize_group_metrics(group["name"], workload_results, experiment_cfg)
        summaries.append(summary)
        group_outputs.append({"group": group["name"], "description": group.get("description"), "workloads": workload_results})
        _print_group_summary(summary)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or experiment_cfg.get("tag", "run")
    output_path = args.output or Path(experiment_cfg.get("result_file", f"Evaluation/exp6_1_generality/results/{tag}_{timestamp}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(args.config),
        "generated_at": timestamp,
        "experiment": experiment_cfg,
        "groups": group_outputs,
        "summaries": summaries,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved generality results to {output_path}")


def summarize_group_metrics(group_name: str, workload_results: List[Dict[str, Any]], experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    target = experiment_cfg.get("target_baseline")
    blind = experiment_cfg.get("blind_baseline")
    metrics = {
        "speedup_vs_blind": [],
        "data_savings_pct_vs_blind": [],
        "semantic_efficiency_ratio": [],
    }
    for workload in workload_results:
        baseline_map = {res["baseline"]: res for res in workload["results"]}
        if not target or target not in baseline_map:
            continue
        derived = baseline_map[target].get("derived", {})
        if blind:
            speedup = derived.get(f"speedup_vs_{blind}")
            data_savings = derived.get(f"data_savings_pct_vs_{blind}")
            if speedup is not None:
                metrics["speedup_vs_blind"].append(speedup)
            if data_savings is not None:
                metrics["data_savings_pct_vs_blind"].append(data_savings)
        efficiency = derived.get("semantic_efficiency_ratio")
        if efficiency is not None:
            metrics["semantic_efficiency_ratio"].append(efficiency)

    summary = {
        "group": group_name,
        "num_workloads": len(workload_results),
        "metrics": {
            name: summarize(values)
            for name, values in metrics.items()
            if values
        },
    }
    return summary


def _print_workload_summary(workload_result: Dict[str, Any]) -> None:
    name = workload_result["workload"]
    print(f"Workload {name}:")
    for baseline in workload_result["results"]:
        agg = baseline["aggregates"]["latency_ms"]
        data = baseline["aggregates"]["total_data_mb"]
        mean_latency = agg["mean"]
        mean_data = data["mean"]
        print(f"  - {baseline['baseline']}: latency={mean_latency:.2f} ms, data={mean_data:.3f} MB")


def _print_group_summary(summary: Dict[str, Any]) -> None:
    print(f"\nGroup summary for {summary['group']}:")
    for metric, stats in summary["metrics"].items():
        mean = stats.get("mean")
        p95 = stats.get("p95")
        print(f"  {metric}: mean={mean:.2f} (p95={p95:.2f})")


if __name__ == "__main__":
    main()


