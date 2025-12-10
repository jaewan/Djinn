#!/usr/bin/env python3
"""
Serverless Emulator Baseline for OSDI Evaluation.

Emulates the performance of serverless systems (AWS Lambda, Google Cloud Run, K-Serve).

Key assumption: Serverless deletes state between requests (cold starts).
For each request: Latency = T_load (weight loading) + T_compute (inference)

This emulator measures ACTUAL cold start time by loading Llama-2-13B
from disk to GPU and then running inference.

Expected behavior: Flat ~30-35s latency per request (dominated by cold start).
This proves that serverless solutions solve memory capacity but kill latency.

Djinn's advantage: Keeps state "warm" (swapped to host) instead of "cold" (deleted).
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerlessEmulator:
    """
    Emulates ideal serverless system.
    
    Assumptions:
    - Infinite container capacity (no memory limit)
    - Zero scheduling overhead
    - State is deleted between requests (cold start required)
    - PCIe bandwidth available for weight loading
    """

    def __init__(self, model_id: str):
        """
        Measure cold start and inference times once, then simulate for all requests.
        
        Args:
            model_id: Model to load
        """
        self.model_id = model_id
        logger.info(f"ServerlessEmulator: Measuring cold start for {model_id}")

        # Measure cold start (load from disk to GPU)
        self.cold_start_time_ms = self._measure_cold_start()

        # Measure inference time (single inference with loaded model)
        self.inference_time_ms = self._measure_inference()

        logger.info(f"Cold start: {self.cold_start_time_ms:.0f}ms")
        logger.info(f"Inference: {self.inference_time_ms:.0f}ms")
        logger.info(f"Total per request: {self.cold_start_time_ms + self.inference_time_ms:.0f}ms")

    def _measure_cold_start(self) -> float:
        """
        Measure actual time to load model from disk to GPU.
        
        This includes:
        - torch.load from disk (if not cached)
        - Model instantiation
        - Weight transfer to GPU
        
        Returns:
            Time in milliseconds
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="cuda",  # Load to GPU
                trust_remote_code=True,
            )
            torch.cuda.synchronize()  # Wait for GPU to finish
        finally:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()

        elapsed_s = time.perf_counter() - start
        return elapsed_s * 1000

    def _measure_inference(self) -> float:
        """
        Measure inference latency with model already loaded.
        
        Returns:
            Time in milliseconds
        """
        logger.info("Measuring inference time...")
        
        torch.cuda.empty_cache()

        # Load model once
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create a prompt
        prompt = "We the People of the United States, in Order to form a more perfect Union, " \
                 "establish Justice, insure domestic Tranquility, provide for the common defence, " \
                 "promote the general Welfare, and secure the Blessings of Liberty to ourselves " \
                 "and our Posterity, do ordain and establish this Constitution for the United States of America."

        try:
            # Warm up
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=50)
            torch.cuda.synchronize()

            # Measure
            start = time.perf_counter()
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start

            return elapsed_s * 1000

        finally:
            del model
            torch.cuda.empty_cache()

    def serve_request(self, agent_id: int, prompt: str) -> Dict[str, Any]:
        """
        Simulate serving a single request in serverless.
        
        In serverless: State is deleted between requests.
        Therefore: Each request incurs cold start + inference.
        
        Args:
            agent_id: Agent identifier
            prompt: Input prompt (unused, but included for consistency)
        
        Returns:
            Result dictionary
        """
        return {
            "agent_id": agent_id,
            "cold_start_ms": self.cold_start_time_ms,
            "inference_ms": self.inference_time_ms,
            "total_latency_ms": self.cold_start_time_ms + self.inference_time_ms,
            "status": "success",
        }


def run_serverless_baseline(
    trace: Dict[str, Any],
    model_id: str = "meta-llama/Llama-2-13b-hf",
) -> Dict[str, Any]:
    """
    Run serverless emulator on trace.
    
    Expected: Flat ~30-35s latency for every request.
    
    Args:
        trace: Workload trace
        model_id: Model to emulate
    
    Returns:
        Results dictionary
    """
    n_agents = trace["n_agents"]
    logger.info(f"\n{'='*80}")
    logger.info(f"SERVERLESS EMULATOR BASELINE: N={n_agents}")
    logger.info(f"{'='*80}\n")

    start_time = time.perf_counter()

    try:
        emulator = ServerlessEmulator(model_id)
    except Exception as e:
        logger.error(f"Failed to initialize emulator: {e}")
        return {
            "system": "serverless_emulator",
            "model_id": model_id,
            "n_agents": n_agents,
            "status": "error",
            "error": str(e),
        }

    # Serve all requests
    results = []
    for agent_id in range(n_agents):
        result = emulator.serve_request(agent_id, trace["prompts"][agent_id])
        results.append(result)

        if (agent_id + 1) % 10 == 0:
            logger.info(f"  Simulated {agent_id + 1}/{n_agents} agents")

    duration = time.perf_counter() - start_time

    # Calculate statistics
    latencies = [r["total_latency_ms"] for r in results]
    latencies_sorted = sorted(latencies)

    result = {
        "system": "serverless_emulator",
        "model_id": model_id,
        "n_agents": n_agents,
        "cold_start_ms": emulator.cold_start_time_ms,
        "inference_ms": emulator.inference_time_ms,
        "duration_s": duration,
        "latency_stats": {
            "mean_ms": np.mean(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p99_ms": np.percentile(latencies, 99),
        },
        "results": results,
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"SERVERLESS SUMMARY: N={n_agents}")
    logger.info(f"{'='*80}")
    logger.info(f"Cold start: {emulator.cold_start_time_ms:.0f}ms (per request)")
    logger.info(f"Inference: {emulator.inference_time_ms:.0f}ms (per request)")
    logger.info(f"Total per request: {emulator.cold_start_time_ms + emulator.inference_time_ms:.0f}ms")
    logger.info(f"P99 Latency: {result['latency_stats']['p99_ms']:.0f}ms")
    logger.info(f"Note: Latency is FLAT (all requests ~same), unlike Djinn which improves with N")
    logger.info(f"{'='*80}\n")

    return result


def main():
    """Test serverless baseline at multiple N values."""
    import argparse
    from trace_generator import load_trace

    parser = argparse.ArgumentParser()
    # Use absolute paths relative to script location
    script_dir = Path(__file__).parent
    exp_dir = script_dir.parent
    parser.add_argument("--trace-dir", type=Path, default=exp_dir / "traces")
    parser.add_argument("--results-dir", type=Path, default=exp_dir / "results")
    parser.add_argument("--n-agents", type=int, default=None, help="Run single N value")

    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.n_agents is not None:
        # Single N
        trace_file = args.trace_dir / f"trace_{args.n_agents}.json"
        if not trace_file.exists():
            logger.error(f"Trace file not found: {trace_file}")
            sys.exit(1)
        
        trace = load_trace(trace_file)
        result = run_serverless_baseline(trace)
        
        # Save result
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = args.results_dir / f"serverless_baseline_{args.n_agents}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result to {output_file}")
    else:
        # Sweep
        all_results = {}
        for n in [10, 20, 30, 40, 50, 60, 70, 80]:
            trace_file = args.trace_dir / f"trace_{n}.json"
            if not trace_file.exists():
                logger.warning(f"Trace file not found: {trace_file}, skipping")
                continue
            
            trace = load_trace(trace_file)
            result = run_serverless_baseline(trace)
            all_results[n] = result
        
        # Save sweep results
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = args.results_dir / f"serverless_baseline_sweep_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved sweep results to {output_file}")


if __name__ == "__main__":
    main()
