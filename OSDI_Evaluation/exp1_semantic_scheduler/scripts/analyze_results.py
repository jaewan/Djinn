#!/usr/bin/env python3
"""
Analyze results from Experiment 1 (Semantic Scheduler).

Usage:
    python scripts/analyze_results.py \
        --results results/*.json \
        --output results/analysis.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics


def analyze_single_result(result: Dict) -> Dict[str, Any]:
    """Analyze a single result file."""
    analysis = {
        "model_id": result.get("model_id"),
        "config": result.get("config"),
        "agent_sweeps": []
    }
    
    for sweep in result.get("agent_counts", []):
        agents = sweep["agents"]
        aggregates = sweep.get("aggregates", {})
        records = sweep.get("records", [])
        
        # Extract per-stage latencies
        reason_latencies = [r["latency_ms"] for r in records if r.get("stage") == "reason"]
        reflect_latencies = [r["latency_ms"] for r in records if r.get("stage") == "reflect"]
        
        sweep_analysis = {
            "agents": agents,
            "success": aggregates.get("success", False),
            "total_duration_s": aggregates.get("total_duration_s"),
            "p99_latency_ms": aggregates.get("p99_latency_ms"),
            "mean_latency_ms": aggregates.get("mean_latency_ms"),
            "kv_reuse_events": aggregates.get("kv_reuse_events"),
            "errors": aggregates.get("errors"),
            "reason_latencies": {
                "mean": statistics.mean(reason_latencies) if reason_latencies else 0,
                "median": statistics.median(reason_latencies) if reason_latencies else 0,
                "p99": sorted(reason_latencies)[int(len(reason_latencies)*0.99)] if reason_latencies else 0,
            },
            "reflect_latencies": {
                "mean": statistics.mean(reflect_latencies) if reflect_latencies else 0,
                "median": statistics.median(reflect_latencies) if reflect_latencies else 0,
                "p99": sorted(reflect_latencies)[int(len(reflect_latencies)*0.99)] if reflect_latencies else 0,
            },
        }
        analysis["agent_sweeps"].append(sweep_analysis)
    
    return analysis


def generate_summary(analyses: List[Dict]) -> Dict[str, Any]:
    """Generate summary across all runs."""
    summary = {
        "total_runs": len(analyses),
        "scaling_analysis": {},
        "key_findings": []
    }
    
    # Group by agent count
    by_agent_count = {}
    for analysis in analyses:
        for sweep in analysis.get("agent_sweeps", []):
            agents = sweep["agents"]
            if agents not in by_agent_count:
                by_agent_count[agents] = []
            by_agent_count[agents].append(sweep)
    
    # Analyze per agent count
    for agents in sorted(by_agent_count.keys()):
        sweeps = by_agent_count[agents]
        
        p99_values = [s["p99_latency_ms"] for s in sweeps if s.get("p99_latency_ms")]
        success_rate = sum(1 for s in sweeps if s.get("success")) / len(sweeps) * 100
        
        summary["scaling_analysis"][f"agents_{agents}"] = {
            "runs": len(sweeps),
            "p99_latency_ms": {
                "mean": statistics.mean(p99_values) if p99_values else 0,
                "min": min(p99_values) if p99_values else 0,
                "max": max(p99_values) if p99_values else 0,
            },
            "success_rate": success_rate,
        }
    
    # Key findings
    max_successful_agents = 0
    for agents in sorted(by_agent_count.keys()):
        sweeps = by_agent_count[agents]
        if sum(1 for s in sweeps if s.get("success")) / len(sweeps) > 0.8:  # >80% success
            max_successful_agents = agents
    
    summary["key_findings"].append(f"Maximum agents (>80% success): {max_successful_agents}")
    
    # Latency degradation
    if 1 in by_agent_count and max_successful_agents in by_agent_count:
        p99_single = statistics.mean([s["p99_latency_ms"] for s in by_agent_count[1]])
        p99_max = statistics.mean([s["p99_latency_ms"] for s in by_agent_count[max_successful_agents]])
        degradation = (p99_max - p99_single) / p99_single * 100
        summary["key_findings"].append(f"Latency degradation (1→{max_successful_agents}): {degradation:.1f}%")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze Exp1 results")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Result files")
    parser.add_argument("--output", type=Path, default=Path("results/analysis.json"))
    
    args = parser.parse_args()
    
    # Load all results
    analyses = []
    for result_file in args.results:
        print(f"Loading {result_file}...")
        with open(result_file) as f:
            result = json.load(f)
            analysis = analyze_single_result(result)
            analyses.append(analysis)
    
    # Generate summary
    summary = generate_summary(analyses)
    
    # Save analysis
    output_data = {
        "analyses": analyses,
        "summary": summary
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Analysis saved to {args.output}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for finding in summary["key_findings"]:
        print(f"• {finding}")
    
    print("\nScaling Analysis:")
    for agents, data in sorted(summary["scaling_analysis"].items()):
        print(f"  {agents}: p99={data['p99_latency_ms']['mean']:.1f}ms, success={data['success_rate']:.0f}%")


if __name__ == "__main__":
    main()

