"""
Helper to initialize Djinn just once for evaluation harnesses.

The goal is to mirror what real users do: call `djinn.init(server_address=...)`
before running workloads so remote Accel devices are available.  We keep init
outside of the measured workloads so timing aggregates remain untouched.
"""

from __future__ import annotations

from typing import Optional
import asyncio
import os
import logging

logger = logging.getLogger(__name__)


_initialized = False


def check_gpu_memory(min_free_gb: Optional[float] = None, device: str = "cuda:0") -> None:
    """
    Ensure there is enough free GPU memory before initializing Djinn.

    Args:
        min_free_gb: Minimum free memory required (defaults to env or 6 GB).
        device: CUDA device string.

    Raises:
        RuntimeError: If available memory is below threshold.
    """
    threshold = min_free_gb if min_free_gb is not None else float(os.getenv("GENIE_EVAL_MIN_FREE_GB", 6.0))
    if threshold <= 0:
        return

    try:
        import torch
    except ImportError:
        logger.debug("Torch unavailable; skipping GPU memory check.")
        return

    if not torch.cuda.is_available():
        logger.debug("CUDA not available; skipping GPU memory check.")
        return

    cuda_device = torch.device(device)
    props = torch.cuda.get_device_properties(cuda_device)
    reserved = torch.cuda.memory_reserved(cuda_device)
    free_gb = (props.total_memory - reserved) / 1024**3

    if free_gb < threshold:
        raise RuntimeError(
            f"Not enough free GPU memory for Djinn VMU "
            f"(required {threshold:.1f} GB, available {free_gb:.1f} GB). "
            "Free GPU memory or adjust GENIE_EVAL_MIN_FREE_GB."
        )
    logger.debug(
        "GPU memory check passed: %.1f GB free (threshold %.1f GB)",
        free_gb,
        threshold,
    )


def ensure_initialized(server_address: Optional[str]) -> None:
    """
    Initialize Djinn if not already done.

    Args:
        server_address: Remote Djinn server data port (e.g., "localhost:5556").
                        If None we rely on env vars or default config.
    """
    global _initialized
    if _initialized:
        return

    min_free = os.getenv("GENIE_EVAL_MIN_FREE_GB")
    if min_free is not None:
        try:
            check_gpu_memory(float(min_free))
        except RuntimeError as exc:
            logger.error("GPU memory check failed: %s", exc)
            raise
    else:
        # default check
        check_gpu_memory()

    try:
        import djinn  # noqa: F401 - side effect init
    except ImportError as exc:
        logger.warning("Djinn module unavailable; skipping initialization: %s", exc)
        return

    try:
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # In async context - can't use sync init, but initialization should already be done
            # Just verify it's initialized
            from djinn.backend.runtime.initialization import _runtime_state
            if not _runtime_state.initialized:
                logger.warning(
                    "In async context but Djinn not initialized. "
                    "Call ensure_initialized_before_async() before asyncio.run()"
                )
            else:
                _initialized = True
        except RuntimeError:
            # Not in async context - use sync init
            result = djinn.init(server_address=server_address, auto_connect=True, profiling=False)
            if result.get("status") == "success":
                logger.info("Djinn init succeeded (server=%s)", server_address or "default")
                _initialized = True
            else:
                logger.warning("Djinn init reported failure: %s", result.get("error"))
    except Exception as exc:
        logger.error("Djinn init() raised: %s", exc)


def ensure_initialized_before_async(server_address: Optional[str]) -> None:
    """
    Initialize Djinn synchronously before entering an async context.
    This should be called from main() or other sync contexts before
    asyncio.run() is called.

    Args:
        server_address: Remote Djinn server data port (e.g., "localhost:5556").
                        If None we rely on env vars or default config.
    """
    # Evaluation harnesses run repeated short-lived jobs; skip remote warmup to
    # avoid 5s stalls when the server is offline.
    os.environ.setdefault("GENIE_SKIP_REMOTE_WARMUP", "1")
    ensure_initialized(server_address)


def export_evaluation_metrics(output_path: Optional[str] = None) -> None:
    """
    Export current Djinn metrics to a JSON file for evaluation analysis.

    This captures VMU utilization, session counts, memory pressure events,
    and other metrics that evaluation scripts should record.

    Args:
        output_path: Path to write metrics JSON. If None, uses
                    GENIE_EVAL_METRICS_PATH env var or defaults to
                    "evaluation_metrics.json"
    """
    if output_path is None:
        output_path = os.getenv("GENIE_EVAL_METRICS_PATH", "evaluation_metrics.json")

    try:
        from djinn.server.memory_metrics import export_metrics_to_json
        export_metrics_to_json(output_path)
        logger.info(f"Exported evaluation metrics to {output_path}")
    except Exception as exc:
        logger.warning(f"Failed to export evaluation metrics: {exc}")
        # Don't raise - evaluation shouldn't fail if metrics export fails

