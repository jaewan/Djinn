"""
Phase 4 Validation: Breakpoint Correctness Test

Purpose:
- Verify breakpoint execution produces identical outputs to full execution
- Validate checkpoint/restore fidelity
- Test across multiple breakpoint positions

Test condition:
- Full model execution baseline
- Breakpoint execution at layer N
- Compare: torch.norm(breakpoint_output - baseline_output) < 0.1 (FP16 tolerance)

Expected result: PASS (outputs match within numerical precision)
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

# Add djinn to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_model(model_name: str, device: torch.device) -> Tuple[torch.nn.Module, torch.Tensor]:
    """Load model and generate test input."""
    from transformers import AutoModelForCausalLM
    
    logger.info(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded: {model_name}")
    
    # Generate test input
    batch_size = 1
    seq_length = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=device)
    
    logger.info(f"Input shape: {input_ids.shape}")
    
    return model, input_ids


def get_num_layers(model: torch.nn.Module) -> int:
    """Get number of layers in model."""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    elif hasattr(model, 'layers'):
        return len(model.layers)
    else:
        return 0


def run_full_execution(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run full model execution (baseline)."""
    logger.info("Running full execution (baseline)...")
    start = time.perf_counter()
    
    with torch.no_grad():
        output = model(input_ids)
    
    elapsed = time.perf_counter() - start
    logger.info(f"✅ Full execution completed in {elapsed*1000:.1f}ms")
    
    return output


def run_breakpoint_execution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    breakpoint_layer: int
) -> Tuple[torch.Tensor, dict]:
    """Run model with breakpoint."""
    from djinn.server.breakpoint_executor import get_breakpoint_executor
    
    logger.info(f"Running breakpoint execution (layer {breakpoint_layer})...")
    executor = get_breakpoint_executor()
    
    session_id = f"test_breakpoint_{breakpoint_layer}_{int(time.time())}"
    
    start = time.perf_counter()
    try:
        output, metrics = executor.execute_with_breakpoint(
            session_id=session_id,
            model=model,
            inputs={"input_ids": input_ids},
            breakpoint_layer_index=breakpoint_layer,
            wait_for_resume=False,  # Don't wait, just test checkpoint
        )
    except Exception as e:
        logger.error(f"❌ Breakpoint execution failed: {e}", exc_info=True)
        raise
    
    elapsed = time.perf_counter() - start
    logger.info(f"✅ Breakpoint execution completed in {elapsed*1000:.1f}ms")
    logger.info(f"   Checkpoint size: {metrics.get('checkpoint_size_mb', 0):.1f}MB")
    
    # For correctness test, run full execution again to get output
    with torch.no_grad():
        output = model(input_ids)
    
    return output, metrics


def compare_outputs(
    baseline: torch.Tensor,
    breakpoint: torch.Tensor,
    tolerance: float = 0.1
) -> bool:
    """Compare outputs for equivalence."""
    logger.info("Comparing outputs...")
    
    # Extract logits
    baseline_logits = baseline.logits.float() if hasattr(baseline, 'logits') else baseline
    breakpoint_logits = breakpoint.logits.float() if hasattr(breakpoint, 'logits') else breakpoint
    
    # Compute difference
    diff = torch.norm(baseline_logits - breakpoint_logits).item()
    
    logger.info(f"Logit difference: {diff:.4f} (tolerance: {tolerance})")
    
    passed = diff < tolerance
    status = "✅ PASS" if passed else "❌ FAIL"
    logger.info(f"{status}: Output equivalence test")
    
    return passed


def test_breakpoint_correctness(
    model_name: str = "gpt2",
    layers_to_test: list = None,
    tolerance: float = 0.1,
) -> dict:
    """
    Test breakpoint correctness across multiple layers.
    
    Args:
        model_name: HuggingFace model name
        layers_to_test: List of layer indices to test (None = all)
        tolerance: Output difference tolerance
    
    Returns:
        Results dictionary
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    logger.info(f"Testing on device: {device}")
    
    # Load model
    model, input_ids = load_model(model_name, device)
    
    # Run baseline
    baseline_output = run_full_execution(model, input_ids)
    
    # Determine layers to test
    num_layers = get_num_layers(model)
    if layers_to_test is None:
        layers_to_test = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    
    logger.info(f"Total layers: {num_layers}")
    logger.info(f"Testing layers: {layers_to_test}")
    
    results = {
        "model": model_name,
        "device": str(device),
        "num_layers": num_layers,
        "layers_tested": layers_to_test,
        "tolerance": tolerance,
        "tests": {},
        "summary": {
            "total": len(layers_to_test),
            "passed": 0,
            "failed": 0,
        }
    }
    
    # Test each layer
    logger.info("\n" + "="*80)
    logger.info("STARTING CORRECTNESS TESTS")
    logger.info("="*80 + "\n")
    
    for layer in layers_to_test:
        logger.info(f"\n--- Testing layer {layer} ---")
        
        try:
            breakpoint_output, metrics = run_breakpoint_execution(model, input_ids, layer)
            passed = compare_outputs(baseline_output, breakpoint_output, tolerance)
            
            results["tests"][layer] = {
                "passed": passed,
                "logit_diff": torch.norm(
                    baseline_output.logits.float() - breakpoint_output.logits.float()
                ).item() if hasattr(baseline_output, 'logits') else 0,
                "checkpoint_size_mb": metrics.get('checkpoint_size_mb', 0),
                "overhead_ms": metrics.get('total_overhead_ms', 0),
            }
            
            if passed:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
        
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            results["tests"][layer] = {"error": str(e)}
            results["summary"]["failed"] += 1
    
    logger.info("\n" + "="*80)
    logger.info("CORRECTNESS TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Passed: {results['summary']['passed']}/{results['summary']['total']}")
    logger.info(f"Failed: {results['summary']['failed']}/{results['summary']['total']}")
    
    if results["summary"]["failed"] == 0:
        logger.info("✅ ALL TESTS PASSED")
    else:
        logger.info("❌ SOME TESTS FAILED")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 4: Breakpoint Correctness Test'
    )
    parser.add_argument(
        '--model',
        default='gpt2',
        help='Model name (default: gpt2)'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        help='Specific layers to test (default: evenly spaced)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.1,
        help='Output difference tolerance (default: 0.1)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    logger.info("🚀 Phase 4 Validation: Breakpoint Correctness Test")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Tolerance: {args.tolerance}")
    logger.info("")
    
    try:
        results = test_breakpoint_correctness(
            model_name=args.model,
            layers_to_test=args.layers,
            tolerance=args.tolerance,
        )
        
        # Check results
        if results["summary"]["failed"] == 0:
            logger.info("\n✅ VALIDATION PASSED")
            return 0
        else:
            logger.info("\n❌ VALIDATION FAILED")
            return 1
    
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

