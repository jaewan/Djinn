"""
Centralized baseline runner implementations for OSDI/SOSP evaluation.

This module provides a unified interface for all baseline types used in experiments.
Each baseline runner extends BaselineRunner and implements the same interface:
- run(workload_cfg) -> BaselineResult

Baseline Types:
- LocalSynthetic: Native PyTorch (local GPU execution)
- RemoteDjinn: Djinn with semantic awareness on/off
- PytorchRpc: PyTorch RPC baseline (fair comparison layer)
- vLLM: Specialized LLM serving engine (context, not architecture)
- Ray: Ray Actors for distributed execution

All baselines use the same workload implementations (from common.workloads)
and metric aggregation (from common.metrics).
"""

from .vllm_runner import vLLMBaselineRunner

__all__ = ["vLLMBaselineRunner"]
