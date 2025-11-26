#!/usr/bin/env python3
"""
Experiment 5.1 – Overhead analysis harness.

This script runs the workloads described in `configs/overhead_smoke.yaml` (or a
custom YAML) and records latency/data metrics for each configured baseline.  For
now the smoke test relies on synthetic workloads so we can iterate on a dev L4
GPU; later we can swap in real Djinn runners by adding new baseline types to the
config without touching this script.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Evaluation/exp5_1_overhead/configs/overhead_smoke.yaml"),
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        help="Optional workload name filter (run only matching workloads).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override results directory (otherwise taken from config).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag to include in output filenames (defaults to config experiment.tag or 'run').",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def filter_workloads(workloads: List[Dict], names: List[str]) -> List[Dict]:
    name_set = set(names)
    return [wl for wl in workloads if wl["name"] in name_set]


def _resolve_server_address(exp_cfg: Dict[str, Any]) -> Optional[str]:
    return exp_cfg.get("djinn_server_address") or os.environ.get("GENIE_SERVER_ADDRESS")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    experiment_cfg = cfg["experiment"]
    baselines_cfg = cfg["baselines"]
    workloads_cfg = cfg["workloads"]

    server_address = _resolve_server_address(experiment_cfg)
    ensure_initialized(server_address)

    if args.workloads:
        workloads_cfg = filter_workloads(workloads_cfg, args.workloads)
        if not workloads_cfg:
            raise SystemExit("No workloads matched the provided filter.")

    runner = ExperimentRunner(experiment_cfg, baselines_cfg)
    workload_results = runner.run_workloads(workloads_cfg)

    output_dir = args.output_dir or Path(experiment_cfg.get("result_dir", "Evaluation/exp5_1_overhead/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or experiment_cfg.get("tag", "run")

    manifest = []
    for workload_result in workload_results:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        workload_dir = output_dir / workload_result["workload"]
        workload_dir.mkdir(parents=True, exist_ok=True)
        output_file = workload_dir / f"{tag}_{timestamp}.json"
        payload = {
            "config": str(args.config),
            "experiment": experiment_cfg,
            "workload": workload_result,
            "generated_at": timestamp,
        }
        with output_file.open("w") as f:
            json.dump(payload, f, indent=2)
        manifest.append(str(output_file))
        _print_summary(workload_result)

    print("\nSaved results:")
    for path in manifest:
        print(f"  - {path}")


def _print_summary(workload_result: Dict[str, Any]) -> None:
    name = workload_result["workload"]
    print(f"\nWorkload: {name}")
    for baseline in workload_result["results"]:
        agg = baseline["aggregates"]["latency_ms"]
        data = baseline["aggregates"]["total_data_mb"]
        speed = baseline["aggregates"]["throughput_units_per_s"]
        mean_latency = agg["mean"]
        mean_data = data["mean"]
        mean_speed = speed["mean"]
        print(
            f"  - {baseline['baseline']}: latency={mean_latency:.2f} ms, "
            f"data={mean_data:.3f} MB, throughput={mean_speed:.2f} units/s"
        )


if __name__ == "__main__":
    main()


