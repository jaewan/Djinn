#!/usr/bin/env python3
"""
Verify that real vision and multimodal models can register and execute remotely.

Tests:
- ResNet-50 (vision)
- ViT-Base (vision)
- CLIP (multimodal)

This validates that Phase 4 load test workloads will work.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.djinn_init import ensure_initialized_before_async
from Evaluation.common.workloads import build_workload


async def test_model(model_name: str, implementation: str, model_id: str, server_address: str):
    """Test a single model registration and execution."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} ({model_id})")
    print(f"{'='*60}")
    
    try:
        # Initialize Djinn (must be done before building workload)
        from djinn.backend.runtime.initialization import init_async
        result = await init_async(
            server_address=server_address,
            auto_connect=True,
            profiling=False,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"Djinn init failed: {result.get('error')}")
        
        # Build workload
        spec = {
            "model_id": model_id,
            "batch_size": 2,  # Small batch for testing
            "image_size": 224,
        }
        if implementation == "hf_multimodal":
            spec["text_length"] = 77
        
        print(f"Building workload...")
        workload = build_workload(implementation, spec, "cpu", "float16")
        model = workload.model
        print(f"✅ Workload built: {type(model).__name__}")
        
        # Initialize manager
        from djinn.core.enhanced_model_manager import EnhancedModelManager
        from djinn.backend.runtime.initialization import get_coordinator
        
        coordinator = get_coordinator()
        if coordinator is None:
            raise RuntimeError("Coordinator unavailable after initialization")
        
        manager = EnhancedModelManager(coordinator=coordinator, server_address=server_address)
        print(f"✅ Manager initialized")
        
        # Register model using the actual HuggingFace model ID for proper server-side loading
        print(f"Registering model as {model_id}...")
        fingerprint = await manager.register_model(model, model_id=model_id)
        print(f"✅ Model registered: {fingerprint[:16]}...")
        
        # Prepare inputs
        inputs = workload.prepare_inputs()
        print(f"✅ Inputs prepared: {list(inputs.keys())}")
        
        # Execute remotely
        import djinn
        if implementation == "hf_vision":
            with djinn.session(phase="vision", priority="normal"):
                print(f"Executing remotely...")
                result = await manager.execute_model(model, inputs)
        elif implementation == "hf_multimodal":
            with djinn.session(phase="generic", priority="normal"):
                print(f"Executing remotely...")
                result = await manager.execute_model(model, inputs)
        else:
            print(f"Executing remotely...")
            result = await manager.execute_model(model, inputs)
        
        # Check result
        if isinstance(result, dict):
            if "logits" in result:
                output_shape = result["logits"].shape
            else:
                output_shape = list(result.values())[0].shape
        else:
            output_shape = result.shape
        
        print(f"✅ Execution successful: output shape {output_shape}")
        
        # Get metrics
        metrics = manager.last_execution_metrics
        if metrics:
            duration_ms = metrics.get("duration_ms", 0)
            print(f"✅ Execution metrics: {duration_ms:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description="Verify real models work remotely")
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:5556",
        help="Djinn server address",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "vit", "clip", "all"],
        default="all",
        help="Model to test",
    )
    args = parser.parse_args()
    
    tests = []
    
    if args.model in ("resnet50", "all"):
        tests.append(("ResNet-50", "hf_vision", "microsoft/resnet-50", args.server))
    
    if args.model in ("vit", "all"):
        tests.append(("ViT-Base", "hf_vision", "google/vit-base-patch16-224", args.server))
    
    if args.model in ("clip", "all"):
        tests.append(("CLIP", "hf_multimodal", "openai/clip-vit-base-patch32", args.server))
    
    print(f"Testing {len(tests)} model(s)...")
    print(f"Server: {args.server}")
    
    results = []
    for model_name, impl, model_id, server in tests:
        success = await test_model(model_name, impl, model_id, server)
        results.append((model_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {model_name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n✅ All models verified successfully!")
        return 0
    else:
        print(f"\n❌ Some models failed verification")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

