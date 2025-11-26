"""
Helpers for constructing lightweight "ghost" models that describe HuggingFace
architectures without instantiating their weights locally.

These ghosts align the implementation with the documented Ghost Interception
path: the client only supplies semantic metadata (model_id, config, task),
while the Djinn server performs the heavyweight weight download directly into
the VMU text segment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Optional, Dict

import torch.nn as nn


# Supported task aliases for clarity in callers.
HfTask = Literal["causal-lm", "seq2seq", "vision-classification"]


@dataclass(frozen=True)
class GhostMetadata:
    model_id: str
    model_class: str
    framework: str
    task: HfTask
    config_dict: Dict
    revision: Optional[str] = None

    def architecture_hash(self) -> str:
        """Deterministic hash for the architecture description."""
        arch_desc = {
            "model_id": self.model_id,
            "model_class": self.model_class,
            "framework": self.framework,
            "task": self.task,
            "config": self.config_dict,
        }
        arch_json = json.dumps(arch_desc, sort_keys=True)
        return hashlib.sha256(arch_json.encode("utf-8")).hexdigest()[:16]


class HuggingFaceGhostModel(nn.Module):
    """
    Minimal nn.Module placeholder that carries HuggingFace metadata but no weights.

    The object satisfies the pieces of the client stack that expect a PyTorch
    nn.Module (for fingerprinting, registration bookkeeping, etc.) while
    guaranteeing that no heavy state_dict is resident on the workstation.
    """

    def __init__(self, metadata: GhostMetadata):
        super().__init__()
        self.config = metadata.config_dict or {}
        self.model_id = metadata.model_id
        self._djinn_ghost_metadata = {
            "ghost": True,
            "model_id": metadata.model_id,
            "model_class": metadata.model_class,
            "framework": metadata.framework,
            "config": metadata.config_dict,
            "task": metadata.task,
            "revision": metadata.revision,
            "descriptor": {
                "framework": metadata.framework,
                "model_class": metadata.model_class,
                "config": metadata.config_dict,
                "ghost": True,
            },
        }
        self._djinn_ghost_arch_hash = metadata.architecture_hash()

    def forward(self, *args, **kwargs):  # pragma: no cover - should never run locally
        raise RuntimeError("Ghost models cannot execute locally. Use Djinn remote execution.")


def _resolve_model_class(task: HfTask) -> str:
    if task == "seq2seq":
        return "transformers.AutoModelForSeq2SeqLM"
    if task == "vision-classification":
        return "transformers.AutoModelForImageClassification"
    return "transformers.AutoModelForCausalLM"


def create_hf_ghost_model(
    model_id: str,
    *,
    task: HfTask = "causal-lm",
    revision: Optional[str] = None,
    config_dict: Optional[Dict] = None,
) -> HuggingFaceGhostModel:
    """
    Instantiate a HuggingFace ghost model for the given model_id.

    Args:
        model_id: HuggingFace repo id (e.g., "EleutherAI/gpt-j-6b").
        task: Logical task type to hint the server loader.
        revision: Optional HF revision for reproducibility.

    Returns:
        HuggingFaceGhostModel ready for Djinn registration.
    """

    model_class = _resolve_model_class(task)
    metadata = GhostMetadata(
        model_id=model_id,
        model_class=model_class,
        framework="transformers",
        task=task,
        config_dict=config_dict or {},
        revision=revision,
    )
    ghost = HuggingFaceGhostModel(metadata=metadata)
    return ghost

