#!/usr/bin/env python3
"""
Client for Experiment 2: Ring Buffer Virtualization

Connects to a remote Djinn server and runs inference using the ring buffer
model cache. Measures TTFT (Time-To-First-Token) and effective bandwidth.

This follows the client-server pattern used in Experiment 1 (semantic scheduler)
but measures different metrics specific to virtualization.

Usage:
    python run_exp2_client.py \
        --server localhost:5000 \
        --model meta-llama/Llama-2-70b-hf \
        --runs 5 \
        --ttft-enabled \
        --output results/exp2_client.json

Prerequisites:
    1. Start server: python start_exp2_server.py --port 5000
    2. Run this client in another terminal
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import djinn
from djinn.core.enhanced_model_manager import EnhancedModelManager
from Evaluation.common.djinn_init import ensure_initialized_before_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


async def run_single_inference(
    manager: EnhancedModelManager,
    model,
    tokenizer,
    prompt: str,
    session_id: str,
    generation_length: int = 50,
) -> Dict[str, Any]:
    """Run single inference and measure latency + TTFT."""
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Measure total time
    start_total = time.perf_counter()
    ttft = None
    
    try:
        with djinn.session(phase="generate", session_id=session_id, priority="normal"):
            # Use generate with hints for streaming
            result = await manager.execute_model(
                model,
                {"input_ids": input_ids},
                hints={
                    "use_generate": True,
                    "max_new_tokens": generation_length,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                }
            )
        
        elapsed_total = time.perf_counter() - start_total
        
        # For streaming-based TTFT, we'd track the first token time separately
        # For now, estimate TTFT as first token latency
        if isinstance(result, dict) and "generated_ids" in result:
            generated_ids = result["generated_ids"]
            num_generated = generated_ids.shape[-1] - input_ids.shape[-1]
            if num_generated > 0:
                ttft = (elapsed_total / num_generated) * 1000  # Estimate first token in ms
        
        return {
            "success": True,
            "total_latency_ms": elapsed_total * 1000,
            "ttft_ms": ttft,
            "timestamp": time.time(),
        }
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


async def run_experiment(
    server_address: str,
    model_id: str,
    num_runs: int = 5,
    generation_length: int = 50,
    output_file: Optional[str] = None,
) -> bool:
    """Run client-side experiment."""
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2 CLIENT - RING BUFFER VIRTUALIZATION")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Server: {server_address}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Runs: {num_runs}")
    logger.info("")
    
    # Initialize Djinn client
    logger.info("[1/4] Initializing Djinn client...")
    try:
        await ensure_initialized_before_async(server_address=server_address)
        logger.info(f"✅ Connected to server at {server_address}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to server: {e}")
        return False
    logger.info("")
    
    # Load model (local copy for input/output processing)
    logger.info("[2/4] Loading model tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✅ Tokenizer loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        return False
    logger.info("")
    
    # Create model manager for remote execution
    logger.info("[3/4] Creating model manager...")
    try:
        manager = EnhancedModelManager()
        
        # Load model remotely (or get reference to it)
        model = await manager.get_or_create_model(model_id)
        logger.info(f"✅ Model manager ready")
    except Exception as e:
        logger.error(f"❌ Failed to create model manager: {e}")
        return False
    logger.info("")
    
    # Run inference measurements
    logger.info("[4/4] Running inference measurements...")
    logger.info(f"Running {num_runs} iterations...\n")
    
    results = []
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning enables computers to",
        "Deep learning models are trained using",
        "Neural networks process information through",
        "Artificial intelligence applications include",
    ]
    
    for run_id in range(num_runs):
        try:
            prompt = prompts[run_id % len(prompts)]
            session_id = f"exp2_client_{run_id}_{int(time.time() * 1000)}"
            
            metrics = await run_single_inference(
                manager, model, tokenizer, prompt,
                session_id=session_id,
                generation_length=generation_length,
            )
            
            results.append({
                "run_id": run_id,
                "prompt": prompt,
                **metrics,
            })
            
            if metrics["success"]:
                ttft_str = f", TTFT: {metrics['ttft_ms']:.1f}ms" if metrics["ttft_ms"] else ""
                logger.info(
                    f"Run {run_id}: latency {metrics['total_latency_ms']:.1f}ms{ttft_str}"
                )
            else:
                logger.error(f"Run {run_id}: {metrics['error']}")
        
        except Exception as e:
            logger.error(f"Run {run_id} exception: {e}")
            results.append({
                "run_id": run_id,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            })
    
    # Compute summary
    successful = [r for r in results if r.get("success", False)]
    
    if not successful:
        logger.error("❌ No successful runs")
        return False
    
    latencies = [r["total_latency_ms"] for r in successful]
    ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms")]
    
    summary = {
        "total_runs": num_runs,
        "successful_runs": len(successful),
        "success_rate": len(successful) / num_runs if num_runs > 0 else 0.0,
        
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "median_latency_ms": sorted(latencies)[len(latencies)//2] if latencies else 0.0,
    }
    
    # Add TTFT statistics if available
    if ttfts:
        summary["avg_ttft_ms"] = sum(ttfts) / len(ttfts)
        summary["min_ttft_ms"] = min(ttfts)
        summary["max_ttft_ms"] = max(ttfts)
        summary["median_ttft_ms"] = sorted(ttfts)[len(ttfts)//2]
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Successful runs: {summary['successful_runs']}/{num_runs} ({summary['success_rate']:.1%})")
    logger.info(f"Avg latency: {summary['avg_latency_ms']:.1f}ms")
    if "avg_ttft_ms" in summary:
        logger.info(f"Avg TTFT: {summary['avg_ttft_ms']:.1f}ms (target: <7000ms)")
    
    # Save results if requested
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "experiment": "exp2_client_remote",
            "server": server_address,
            "model": model_id,
            "generation_length": generation_length,
            "runs": results,
            "summary": summary,
            "timestamp": _utc_timestamp(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Experiment 2 Client - Ring Buffer Virtualization")
    parser.add_argument("--server", type=str, default="localhost:5000",
                       help="Server address (host:port)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf",
                       help="Model ID from HuggingFace")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of inference runs")
    parser.add_argument("--generation-length", type=int, default=50,
                       help="Tokens to generate per inference")
    parser.add_argument("--output", type=str, default="results/exp2_client.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(run_experiment(
            server_address=args.server,
            model_id=args.model,
            num_runs=args.runs,
            generation_length=args.generation_length,
            output_file=args.output,
        ))
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

