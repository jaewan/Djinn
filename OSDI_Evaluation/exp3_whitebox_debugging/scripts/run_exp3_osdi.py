#!/usr/bin/env python3
"""
Experiment 3: OSDI Major Revision - White-Box Interactivity

This script orchestrates the full Experiment 3 evaluation:
1. PyTorch Eager baseline (VRAM held during pause)
2. vLLM API capabilities baseline (no breakpoint API)
3. Djinn breakpoint correctness (multi-layer trials)
4. Activation steering demo (modify + resume)
5. Concurrent request demo (Request B executes while A is paused)
6. Comparative results report with real metrics
"""

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from Evaluation.common.djinn_init import ensure_initialized_before_async
from djinn.core.coordinator import get_coordinator
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.core.ghost_loader import create_hf_ghost_model
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "exp3_osdi.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"✅ Loaded config from {config_path}")
    return config


def build_prompt(base: str, total_tokens: int) -> str:
    suffix = " context token"
    tokens_needed = max(0, total_tokens - len(base.split()))
    repeated = (suffix * tokens_needed) if tokens_needed else ""
    return f"{base}{repeated}"


def extract_logits(model_output: Any) -> torch.Tensor:
    if model_output is None:
        raise ValueError("Model output is None")
    if isinstance(model_output, dict):
        if 'logits' in model_output:
            return model_output['logits']
        for value in model_output.values():
            if torch.is_tensor(value):
                return value
    if hasattr(model_output, 'logits'):
        return model_output.logits
    if torch.is_tensor(model_output):
        return model_output
    raise ValueError(f"Unsupported model output type: {type(model_output)}")


def compute_token_accuracy(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    ref = reference_logits.argmax(dim=-1)
    cand = candidate_logits.argmax(dim=-1)
    if ref.shape != cand.shape:
        raise ValueError(f"Token shapes differ: {ref.shape} vs {cand.shape}")
    matches = (ref == cand).float().mean().item()
    return matches * 100.0


def measure_gpu_memory_gb(gpu_index: int = 0) -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem_info.used / (1024 ** 3)
        pynvml.nvmlShutdown()
        return used_gb
    except Exception as e:
        logger.debug(f"pynvml unavailable ({e}); falling back to torch.cuda")
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            allocated = torch.cuda.memory_allocated(gpu_index) / (1024 ** 3)
            return allocated
        return None


def format_optional(value: Optional[float], suffix: str = "GB") -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f} {suffix}"


# ---------------------------------------------------------------------------
# Djinn evaluation workflows
# ---------------------------------------------------------------------------

async def run_breakpoint_trials(
    coordinator,
    manager,
    model,
    fingerprint: str,
    input_ids: torch.Tensor,
    baseline_logits: torch.Tensor,
    layers: List[int],
    num_trials: int,
) -> Dict[str, Any]:
    accuracy_entries: List[Dict[str, Any]] = []
    for trial in range(num_trials):
        for layer in layers:
            start = time.perf_counter()
            model_output, metrics = await coordinator.execute_remote_model_with_breakpoint(
                fingerprint=fingerprint,
                inputs={"input_ids": input_ids},
                breakpoint_layer_index=layer,
                wait_for_resume=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            logits = extract_logits(model_output).detach().cpu()
            token_accuracy = compute_token_accuracy(baseline_logits, logits)
            accuracy_entries.append({
                "trial": trial + 1,
                "layer": layer,
                "token_accuracy": token_accuracy,
                "latency_ms": elapsed_ms,
                "checkpoint_time_ms": metrics.get('checkpoint_time_ms', 0.0),
                "restore_time_ms": metrics.get('restore_time_ms', 0.0),
                "checkpoint_size_mb": metrics.get('checkpoint_size_mb', 0.0),
            })
            logger.info(
                f"Djinn trial {trial+1}/{num_trials}, layer {layer}: "
                f"token_accuracy={token_accuracy:.2f}%"
            )
    accuracies = [entry["token_accuracy"] for entry in accuracy_entries]
    latency = [entry["latency_ms"] for entry in accuracy_entries]
    summary = {
        "entries": accuracy_entries,
        "token_accuracy_mean": statistics.mean(accuracies) if accuracies else 0.0,
        "token_accuracy_std": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "latency_mean_ms": statistics.mean(latency) if latency else 0.0,
        "latency_std_ms": statistics.pstdev(latency) if len(latency) > 1 else 0.0,
    }
    return summary


async def run_activation_steering_demo(
    coordinator,
    fingerprint: str,
    input_ids: torch.Tensor,
    layer: int,
    scale: float,
) -> Dict[str, Any]:
    async def _capture_checkpoint() -> Tuple[torch.Tensor, Dict[str, Any]]:
        result, metrics = await coordinator.execute_remote_model_with_breakpoint(
            fingerprint=fingerprint,
            inputs={"input_ids": input_ids},
            breakpoint_layer_index=layer,
            wait_for_resume=False,
        )
        # Try to get checkpoint_activation from metrics first (new behavior), then from result dict (fallback)
        checkpoint_activation = metrics.get('checkpoint_activation')
        if checkpoint_activation is None and isinstance(result, dict):
            checkpoint_activation = result.get('checkpoint_activation')
        if checkpoint_activation is None:
            raise RuntimeError("Checkpoint activation not returned by server (check breakpoint_executor output)")
        session_id = metrics.get('session_id')
        if not session_id:
            raise RuntimeError("Missing session_id in metrics")
        return checkpoint_activation, metrics

    # Resume without modifications
    baseline_activation, baseline_metrics = await _capture_checkpoint()
    session_id = baseline_metrics['session_id']
    baseline_output, baseline_resume_metrics = await coordinator.resume_from_checkpoint(
        fingerprint=fingerprint,
        session_id=session_id,
        modified_activation=baseline_activation,
        layer_index=layer,
    )

    # Resume with steering (scaled activation)
    steering_activation, steering_metrics = await _capture_checkpoint()
    session_id_steer = steering_metrics['session_id']
    steered_activation = steering_activation * scale
    steered_output, steered_resume_metrics = await coordinator.resume_from_checkpoint(
        fingerprint=fingerprint,
        session_id=session_id_steer,
        modified_activation=steered_activation,
        layer_index=layer,
    )

    baseline_logits = extract_logits(baseline_output).detach().cpu()
    steered_logits = extract_logits(steered_output).detach().cpu()
    baseline_tokens = baseline_logits.argmax(dim=-1)
    steered_tokens = steered_logits.argmax(dim=-1)
    changed = bool(torch.any(baseline_tokens != steered_tokens).item())

    logger.info(
        "Activation steering result: changed=%s (scale=%.2f)",
        changed,
        scale,
    )

    return {
        "layer": layer,
        "scale": scale,
        "output_changed": changed,
        "baseline_restore_time_ms": baseline_resume_metrics.get('restore_time_ms', 0.0),
        "steered_restore_time_ms": steered_resume_metrics.get('restore_time_ms', 0.0),
        "token_diff_percent": float(((baseline_tokens != steered_tokens).float().mean().item()) * 100.0),
    }


async def run_concurrent_demo(
    coordinator,
    manager,
    model,
    fingerprint: str,
    input_ids: torch.Tensor,
    layer: int,
    gpu_index: int,
) -> Dict[str, Any]:
    pause_result, pause_metrics = await coordinator.execute_remote_model_with_breakpoint(
        fingerprint=fingerprint,
        inputs={"input_ids": input_ids},
        breakpoint_layer_index=layer,
        wait_for_resume=False,
    )
    checkpoint_activation = pause_result.get('checkpoint_activation') if isinstance(pause_result, dict) else None
    if checkpoint_activation is None:
        raise RuntimeError("Concurrent demo: missing checkpoint_activation")
    session_id = pause_metrics.get('session_id')
    vram_before = measure_gpu_memory_gb(gpu_index)
    logger.info(f"Concurrent demo: VRAM before Request B: {format_optional(vram_before)}")

    start_b = time.perf_counter()
    b_output = await manager.execute_model(model, {"input_ids": input_ids})
    request_b_latency_ms = (time.perf_counter() - start_b) * 1000
    vram_during = measure_gpu_memory_gb(gpu_index)
    logger.info(f"Concurrent demo: VRAM during pause: {format_optional(vram_during)}")
    b_logits = extract_logits(b_output)

    resume_output, resume_metrics = await coordinator.resume_from_checkpoint(
        fingerprint=fingerprint,
        session_id=session_id,
        modified_activation=checkpoint_activation,
        layer_index=layer,
    )
    resume_latency = resume_metrics.get('restore_time_ms', 0.0) if isinstance(resume_metrics, dict) else 0.0

    return {
        "request_b_latency_ms": request_b_latency_ms,
        "vram_before_pause_gb": vram_before,
        "vram_during_pause_gb": vram_during,
        "resume_restore_time_ms": resume_latency,
        "request_b_tokens": int(b_logits.argmax(dim=-1).shape[-1]) if torch.is_tensor(b_logits) else None,
    }


async def run_djinn_breakpoint_tests(coordinator, config: Dict[str, Any], output_dir: Path, gpu_index: int) -> Dict[str, Any]:
    # Use the coordinator passed from main
    manager = EnhancedModelManager(coordinator=coordinator)

    model_name = config['model']['name']
    logger.info(f"Loading ghost model {model_name}")
    model = create_hf_ghost_model(model_name, task="causal-lm")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fingerprint = await manager.register_model(model, model_id=model_name)
    input_len = config['experiment']['inference']['input_length']
    prompt = build_prompt("The future of AI is", input_len)
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=input_len,
    )["input_ids"]

    logger.info("Running baseline Djinn execution (no breakpoint) for accuracy reference")
    baseline_output = await manager.execute_model(model, {"input_ids": input_ids})
    baseline_logits = extract_logits(baseline_output).detach().cpu()

    breakpoint_layers = config['experiment']['breakpoints']['layers']
    num_trials = config['experiment']['num_trials']
    trial_summary = await run_breakpoint_trials(
        coordinator,
        manager,
        model,
        fingerprint,
        input_ids,
        baseline_logits,
        breakpoint_layers,
        num_trials,
    )

    steering_cfg = config['experiment']['activation_steering']
    steering_results: Dict[str, Any] = {}
    if steering_cfg.get('enabled', True):
        # Use steering layer from config, fallback to mid-layer from breakpoints
        steering_layer = steering_cfg.get('steering_layer', breakpoint_layers[1] if len(breakpoint_layers) > 1 else 6)
        try:
            steering_results = await run_activation_steering_demo(
                coordinator,
                fingerprint,
                input_ids,
                steering_layer,
                steering_cfg.get('modification_factor', 0.9),
            )
        except Exception as e:
            logger.warning(f"Activation steering demo skipped: {e}")
            steering_results = {}

    concurrent_cfg = config['experiment']['concurrent_demo']
    concurrent_results: Dict[str, Any] = {}
    if concurrent_cfg.get('enabled', True):
        # Use mid-layer from breakpoints
        concurrent_layer = breakpoint_layers[1] if len(breakpoint_layers) > 1 else breakpoint_layers[0]
        try:
            concurrent_results = await run_concurrent_demo(
                coordinator,
                manager,
                model,
                fingerprint,
                input_ids,
                concurrent_layer,
                gpu_index,
            )
        except Exception as e:
            logger.warning(f"Concurrent demo skipped: {e}")
            concurrent_results = {}

    djinn_results = {
        "status": "success",
        "token_accuracy": trial_summary,
        "activation_steering": steering_results,
        "concurrent_demo": concurrent_results,
        "baseline_prompt": prompt,
    }

    results_path = output_dir / "djinn_breakpoint_results.json"
    with open(results_path, 'w') as f:
        json.dump(djinn_results, f, indent=2, default=str)
    logger.info(f"Djinn breakpoint results saved to {results_path}")

    return djinn_results


# ---------------------------------------------------------------------------
# Comparative report
# ---------------------------------------------------------------------------

def generate_comparative_report(
    pytorch_results: Dict[str, Any],
    vllm_results: Dict[str, Any],
    djinn_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    def _djinn_vram() -> str:
        demo = djinn_results.get('concurrent_demo', {})
        return format_optional(demo.get('vram_during_pause_gb'))

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baselines": {
            "pytorch_eager": pytorch_results,
            "vllm": vllm_results,
            "djinn": djinn_results,
        },
        "summary_table": {
            "breakpoint_support": {
                "pytorch_eager": "Manual (holds VRAM)",
                "vllm": "NOT POSSIBLE",
                "djinn": "Native pause/resume",
            },
            "vram_during_pause_gb": {
                "pytorch_eager": format_optional(pytorch_results.get('metrics', {}).get('vram_held_during_pause_gb')),
                "vllm": "N/A",
                "djinn": _djinn_vram(),
            },
            "concurrent_requests": {
                "pytorch_eager": "NO",
                "vllm": "NO",
                "djinn": "YES" if djinn_results.get('concurrent_demo') else "N/A",
            },
            "activation_steering": {
                "pytorch_eager": "Manual, blocks GPU",
                "vllm": "NOT POSSIBLE",
                "djinn": "Steering demo completed" if djinn_results.get('activation_steering', {}).get('output_changed') else "No change",
            },
            "token_accuracy_percent": {
                "pytorch_eager": "100 (reference)",
                "vllm": "N/A",
                "djinn": f"{djinn_results.get('token_accuracy', {}).get('token_accuracy_mean', 0.0):.2f}",
            },
        },
    }

    logger.info("\n📊 COMPARATIVE RESULTS TABLE:\n")
    logger.info(f"{'Metric':<30} {'PyTorch Eager':<25} {'vLLM':<25} {'Djinn':<25}")
    logger.info("-" * 110)
    for metric, values in report["summary_table"].items():
        pytorch_val = str(values.get("pytorch_eager", "N/A"))[:24]
        vllm_val = str(values.get("vllm", "N/A"))[:24]
        djinn_val = str(values.get("djinn", "N/A"))[:24]
        logger.info(f"{metric:<30} {pytorch_val:<25} {vllm_val:<25} {djinn_val:<25}")

    report_file = output_dir / "comparative_results.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Comparative report saved to {report_file}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main_sync():
    """Synchronous main - handles initialization before async context."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: OSDI Major Revision")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "exp3_osdi_full.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/exp3_osdi_results"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-djinn", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for VRAM sampling")
    parser.add_argument("--server", type=str, default="localhost:5556", help="Djinn server address")

    args = parser.parse_args()

    setup_logging(args.output_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("EXPERIMENT 3: OSDI WHITE-BOX INTERACTIVITY")
    logger.info("=" * 80)

    config = load_config(args.config)

    # Initialize Djinn BEFORE entering async context
    logger.info(f"[Djinn] Initializing client to {args.server}...")
    ensure_initialized_before_async(args.server)
    coordinator = get_coordinator()
    if coordinator is None:
        raise RuntimeError("Failed to initialize Djinn coordinator")
    logger.info("[Djinn] Client initialized")

    # Run async main
    return asyncio.run(main_async(coordinator, args, config))


async def main_async(coordinator, args, config):
    """Async main - runs experiments with initialized coordinator."""
    pytorch_results = {}
    vllm_results = {}
    djinn_results = {}

    if not args.skip_pytorch:
        logger.info("\n🔵 Running PyTorch Eager baseline...")
        try:
            from baselines.pytorch_eager_baseline import run_pytorch_eager_baseline
            pytorch_results = run_pytorch_eager_baseline(
                model_name=config['model']['name'],
                input_length=config['experiment']['inference']['input_length'],
                breakpoint_layer=config['experiment']['breakpoints']['layers'][1],
                pause_duration_seconds=config['experiment']['concurrent_demo'].get('pause_duration_seconds', 5),
                output_dir=args.output_dir,
            )
        except Exception as e:
            logger.warning(f"PyTorch baseline skipped: {e}")
            pytorch_results = {"status": "error", "error": str(e)}

    if not args.skip_vllm:
        logger.info("\n🟢 Running vLLM API test...")
        try:
            from baselines.vllm_breakpoint_test import test_vllm_api_capabilities
            vllm_results = test_vllm_api_capabilities(
                model_name=config['model']['name'],
                output_dir=args.output_dir,
            )
        except Exception as e:
            logger.warning(f"vLLM baseline skipped: {e}")
            vllm_results = {"status": "error", "error": str(e)}

    if not args.skip_djinn:
        logger.info("\n🟡 Running Djinn breakpoint tests...")
        try:
            djinn_results = await run_djinn_breakpoint_tests(coordinator, config, args.output_dir, args.gpu_index)
        except Exception as e:
            logger.error(f"Djinn tests failed: {e}", exc_info=True)
            djinn_results = {"status": "error", "error": str(e)}

    generate_comparative_report(pytorch_results, vllm_results, djinn_results, args.output_dir)
    logger.info("\n✅ EXPERIMENT 3 EVALUATION COMPLETE")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    exit_code = main_sync()
    sys.exit(exit_code or 0)
