#!/usr/bin/env python3
"""
Experiment 3: Resume Latency Runner

Runs three baselines on H100:
  1) Stateless Recompute
  2) Manual CPU Offload (pinned, non_blocking)
  3) Djinn Resume (IO_WAIT -> ready)

Outputs combined JSON for plotting crossover curves.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_cmd(cmd, cwd: Path):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3 resume-latency baselines")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--server", type=str, default="127.0.0.1:5556")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per layer")
    parser.add_argument("--repeat", type=int, default=5, help="Measured iterations per layer")
    parser.add_argument("--sleep-between", type=int, default=5, help="Seconds to sleep between runs for GPU release")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/exp3_resume_results"),
    )
    parser.add_argument("--skip-recompute", action="store_true")
    parser.add_argument("--skip-offload", action="store_true")
    parser.add_argument("--skip-djinn", action="store_true")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Recompute
    recompute_path = output_dir / "recompute_latency.json"
    if not args.skip_recompute:
        run_cmd(
            [
                "python3",
                "benchmark_recompute.py",
                "--model",
                args.model,
                "--layers",
                *map(str, args.layers),
                "--max-length",
                str(args.max_length),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--output",
                str(recompute_path),
            ],
            scripts_dir,
        )
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    # 2) Manual Offload
    offload_path = output_dir / "manual_offload_latency.json"
    if not args.skip_offload:
        run_cmd(
            [
                "python3",
                "benchmark_manual_offload.py",
                "--model",
                args.model,
                "--layers",
                *map(str, args.layers),
                "--max-length",
                str(args.max_length),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--output",
                str(offload_path),
            ],
            scripts_dir,
        )
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    # 3) Djinn Resume
    djinn_path = output_dir / "djinn_resume_latency.json"
    if not args.skip_djinn:
        run_cmd(
            [
                "python3",
                "benchmark_djinn_resume.py",
                "--model",
                args.model,
                "--layers",
                *map(str, args.layers),
                "--max-length",
                str(args.max_length),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--output",
                str(djinn_path),
                "--server",
                args.server,
            ],
            scripts_dir,
        )
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    # Aggregate
    combined = {"model": args.model, "layers": args.layers, "results": {}}
    for name, path in [
        ("recompute", recompute_path),
        ("manual_offload", offload_path),
        ("djinn", djinn_path),
    ]:
        if path.exists():
            with open(path) as f:
                combined["results"][name] = json.load(f)

    combined_path = output_dir / "resume_latency_combined.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n✅ Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
