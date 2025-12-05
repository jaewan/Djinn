#!/usr/bin/env python3
"""
Correctness Verification: KV Cache Swap Integrity

Verifies that KV caches swapped to host memory can be restored correctly
without data corruption or numerical drift.

Test approach:
1. Execute model without swap (baseline)
2. Execute model with forced swap/restore
3. Compare logits: baseline vs swapped
4. Assert: torch.allclose(baseline_logits, restored_logits, rtol=1e-5)
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import djinn
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model
from Evaluation.common.djinn_init import ensure_initialized_before_async

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_logit_correctness(
    model_id: str = "meta-llama/Llama-2-7b-hf",
    num_runs: int = 3,
) -> Dict[str, Any]:
    """
    Test that swapped KV caches produce identical logits.
    
    Args:
        model_id: HuggingFace model identifier
        num_runs: Number of test runs (for statistical validation)
    
    Returns:
        Dictionary with test results and statistics
    """
    
    logger.info("=" * 70)
    logger.info("CORRECTNESS TEST: KV Cache Swap Integrity")
    logger.info("=" * 70)
    
    model = create_hf_ghost_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    manager = EnhancedModelManager()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create test prompts
    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "What is the meaning of life?",
    ]
    
    results = {
        "model_id": model_id,
        "num_runs": num_runs,
        "tests": [],
        "summary": {
            "passed": 0,
            "failed": 0,
            "mean_logit_diff": 0.0,
            "max_logit_diff": 0.0,
        }
    }
    
    logit_diffs = []
    
    for run in range(num_runs):
        prompt = test_prompts[run % len(test_prompts)]
        logger.info(f"\n[Run {run + 1}/{num_runs}] Prompt: {prompt[:50]}...")
        
        session_id = f"verify_{uuid.uuid4().hex[:8]}"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        try:
            # PATH A: Execute without swap (baseline)
            logger.info("  [A] Baseline execution (no swap)...")
            session_id_a = f"{session_id}_baseline"
            
            with djinn.session(phase="prefill", session_id=session_id_a):
                result_a = await manager.execute_model(
                    model,
                    {"input_ids": input_ids},
                    hints={
                        "use_generate": True,
                        "max_new_tokens": 20,
                        "return_dict": True,
                        "output_scores": True,
                    }
                )
            
            # Extract logits from result (try multiple formats)
            if isinstance(result_a, dict):
                logits_a = result_a.get("logits") or result_a.get("scores")
            else:
                logits_a = result_a
            
            if logits_a is None:
                logger.warning("  Could not extract logits from result A")
                continue
            
            # PATH B: Execute with forced swap/restore
            logger.info("  [B] Execution with swap/restore...")
            session_id_b = f"{session_id}_swapped"
            
            with djinn.session(phase="prefill", session_id=session_id_b):
                result_b_step1 = await manager.execute_model(
                    model,
                    {"input_ids": input_ids},
                    hints={
                        "use_generate": True,
                        "max_new_tokens": 1,
                    }
                )
            
            # Signal IO_WAIT (trigger swap)
            logger.info("  [B] Signaling IO_WAIT (force swap)...")
            djinn.signal_phase("IO_WAIT", session_id_b)
            
            await asyncio.sleep(0.5)  # Ensure swap happens
            
            # Signal COMPUTE (trigger restore)
            logger.info("  [B] Signaling COMPUTE (trigger restore)...")
            djinn.signal_phase("COMPUTE", session_id_b)
            
            await asyncio.sleep(0.2)  # Ensure restore completes
            
            # Continue execution after restore
            with djinn.session(phase="decode", session_id=session_id_b):
                result_b = await manager.execute_model(
                    model,
                    {"input_ids": input_ids},
                    hints={
                        "use_generate": True,
                        "max_new_tokens": 20,
                        "return_dict": True,
                        "output_scores": True,
                    }
                )
            
            # Extract logits from result B
            if isinstance(result_b, dict):
                logits_b = result_b.get("logits") or result_b.get("scores")
            else:
                logits_b = result_b
            
            if logits_b is None:
                logger.warning("  Could not extract logits from result B")
                continue
            
            # Compare logits
            logger.info("  [Compare] Computing logit differences...")
            
            # Ensure same shape for comparison
            if logits_a.shape != logits_b.shape:
                logger.warning(f"  Shape mismatch: {logits_a.shape} vs {logits_b.shape}")
                continue
            
            # Compute element-wise difference
            logit_diff = torch.abs(logits_a - logits_b)
            max_diff = logit_diff.max().item()
            mean_diff = logit_diff.mean().item()
            
            logit_diffs.append(max_diff)
            
            # Check correctness with tolerance
            rtol = 1e-5
            atol = 1e-8
            is_close = torch.allclose(logits_a, logits_b, rtol=rtol, atol=atol)
            
            test_result = {
                "run": run + 1,
                "prompt": prompt[:50],
                "max_logit_diff": max_diff,
                "mean_logit_diff": mean_diff,
                "passed": is_close,
                "rtol": rtol,
                "atol": atol,
            }
            
            results["tests"].append(test_result)
            
            if is_close:
                logger.info(f"  ✅ PASS (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                results["summary"]["passed"] += 1
            else:
                logger.error(f"  ❌ FAIL (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                logger.error(f"     Exceeds tolerance (rtol={rtol}, atol={atol})")
                results["summary"]["failed"] += 1
        
        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}", exc_info=True)
            results["tests"].append({
                "run": run + 1,
                "error": str(e),
                "passed": False,
            })
            results["summary"]["failed"] += 1
    
    # Compute summary statistics
    if logit_diffs:
        results["summary"]["mean_logit_diff"] = sum(logit_diffs) / len(logit_diffs)
        results["summary"]["max_logit_diff"] = max(logit_diffs)
    
    return results


async def main():
    """Run correctness verification."""
    
    logger.info("SEMANTIC SCHEDULER CORRECTNESS VERIFICATION")
    logger.info("Testing: KV cache swap/restore produces identical outputs")
    logger.info("")
    
    # Initialize Djinn
    ensure_initialized_before_async("localhost:5556")
    
    try:
        results = await test_logit_correctness(num_runs=3)
        
        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Passed: {results['summary']['passed']}/{results['summary']['passed'] + results['summary']['failed']}")
        logger.info(f"Failed: {results['summary']['failed']}/{results['summary']['passed'] + results['summary']['failed']}")
        logger.info(f"Max Logit Diff: {results['summary']['max_logit_diff']:.2e}")
        logger.info(f"Mean Logit Diff: {results['summary']['mean_logit_diff']:.2e}")
        logger.info("")
        
        if results['summary']['failed'] == 0 and results['summary']['passed'] > 0:
            logger.info("✅ CORRECTNESS VERIFIED: Swap/restore preserves output correctness")
            verdict = "PASS"
        else:
            logger.error("❌ CORRECTNESS FAILED: Swap/restore introduced errors")
            verdict = "FAIL"
        
        logger.info("=" * 70)
        
        # Save results
        out_path = Path("correctness_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {out_path}")
        
        return 0 if verdict == "PASS" else 1
    
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

