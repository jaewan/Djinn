#!/usr/bin/env python3
"""
Render Experiment 3.1 ablation matrix into a concrete run plan.

Used as a lightweight test to ensure the configuration files are valid before
executing heavy workloads on the production cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def merge_config(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = defaults.copy()
    merged.update(override or {})
    return merged


def build_plan(matrix_path: Path, workloads_path: Path) -> Dict[str, Any]:
    matrix = load_yaml(matrix_path)
    workload_spec = load_yaml(workloads_path)

    defaults = matrix.get("defaults", {})
    configurations = matrix.get("configurations", {})
    run_plan = matrix.get("run_plan", {})
    workload_defs = workload_spec.get("workloads", {})

    expanded = []
    for workload_name in run_plan.get("workloads", []):
        workload_cfg = workload_defs.get(workload_name)
        if not workload_cfg:
            raise ValueError(f"Unknown workload '{workload_name}'")
        for config_name in run_plan.get("configurations", []):
            config_override = configurations.get(config_name)
            if not config_override:
                raise ValueError(f"Unknown configuration '{config_name}'")
            merged_cfg = merge_config(defaults, config_override)
            expanded.append(
                {
                    "workload": workload_name,
                    "model_id": workload_cfg["model_id"],
                    "workload_type": workload_cfg["type"],
                    "config_name": config_name,
                    "config": merged_cfg,
                }
            )
    return {"runs": expanded, "run_count": len(expanded)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="Path to ablation_matrix.yaml")
    parser.add_argument("--workloads", type=Path, required=True, help="Path to workloads.yaml")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    plan = build_plan(args.matrix, args.workloads)
    payload = json.dumps(plan, indent=2)

    if args.output:
        args.output.write_text(payload + "\n")
        print(f"Saved run matrix to {args.output} ({plan['run_count']} combinations).")
    else:
        print(payload)


if __name__ == "__main__":
    main()

