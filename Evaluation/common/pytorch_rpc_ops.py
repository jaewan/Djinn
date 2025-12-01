"""
Shared helpers for the PyTorch RPC baseline.

This module hosts the RPC entrypoints and executor registry so that both the
server script and the evaluation harness can import the same symbol when
issuing rpc.rpc_sync(...) calls.
"""

from __future__ import annotations

from typing import Dict

import torch

EXECUTOR_REGISTRY: Dict[str, "RpcWorkloadExecutor"] = {}


def register_executor(name: str, executor: "RpcWorkloadExecutor") -> None:
    EXECUTOR_REGISTRY[name] = executor


def clear_executors() -> None:
    EXECUTOR_REGISTRY.clear()


def rpc_forward(workload_name: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if workload_name not in EXECUTOR_REGISTRY:
        raise KeyError(f"Workload '{workload_name}' is not registered on the RPC server.")
    return EXECUTOR_REGISTRY[workload_name].execute(inputs)


class RpcWorkloadExecutor:
    """Protocol definition for type checking."""

    def execute(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


