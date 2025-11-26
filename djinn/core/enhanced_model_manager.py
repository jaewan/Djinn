"""
Enhanced Model Manager: Client-side integration for model cache system.

Handles explicit model registration, fingerprinting, and execution routing for
Djinn's semantic cache pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .fingerprint_policy import FingerprintPolicy

logger = logging.getLogger(__name__)


class ModelNotRegisteredError(RuntimeError):
    """Raised when executing a model that hasn't been registered."""


class _UsageTracker:
    """Simple usage tracker to back future cache policies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: Dict[str, int] = defaultdict(int)

    def record(self, fingerprint: str) -> None:
        with self._lock:
            self._usage[fingerprint] += 1

    def reset(self, fingerprint: str) -> None:
        with self._lock:
            self._usage[fingerprint] = 0

    def get(self, fingerprint: str) -> int:
        with self._lock:
            return self._usage.get(fingerprint, 0)


class _FingerprintAdapter:
    """Wrap FingerprintPolicy with a legacy-compatible .compute() shim."""

    def __init__(self, policy: FingerprintPolicy):
        self._policy = policy

    def compute(self, model: nn.Module, model_id: Optional[str] = None) -> str:
        fingerprint, _ = self._policy.compute_fingerprint(
            model=model,
            model_id=model_id,
        )
        return fingerprint


class EnhancedModelManager:
    """
    Client-side model manager for the semantic cache system.
    """

    def __init__(self, coordinator=None, server_address: Optional[str] = None):
        from .model_registry import get_model_registry
        from .cache_query import get_cache_query_client

        self.fingerprint_policy = FingerprintPolicy()
        self.fingerprint = _FingerprintAdapter(self.fingerprint_policy)
        self.registry = get_model_registry()
        self.cache_client = get_cache_query_client()
        self._server_address = server_address

        if coordinator is None:
            from .coordinator import get_coordinator

            try:
                self.coordinator = get_coordinator()
            except RuntimeError:
                self.coordinator = None
                logger.warning("Coordinator not available, will use direct TCP")
        else:
            self.coordinator = coordinator

        self.registered_models: Dict[str, Dict[str, Any]] = {}
        self._connection_pool: Dict[str, List[tuple]] = {}
        self._connection_lock = asyncio.Lock()
        self._connection_timeout = 60.0
        self._max_connection_errors = 3
        self._max_connections_per_target = 20

        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        debug_mode = bool(loop and loop.get_debug())
        max_workers = 4 if debug_mode else 8
        self._serialization_executor = ThreadPoolExecutor(
            max_workers=min(20, max_workers),
            thread_name_prefix="djinn-serialize",
        )

        self.usage_tracker = _UsageTracker()
        self._server_address_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
        self._cache_lock = asyncio.Lock()
        self._last_execution_metrics: Dict[str, Any] = {}

    def _optimize_tcp_socket(self, sock) -> None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_WINDOW_CLAMP, 64 * 1024 * 1024)
            except (AttributeError, OSError):
                pass
            actual_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            logger.debug(
                "✅ TCP optimized: NODELAY=1, SNDBUF=%sMB, RCVBUF=%sMB",
                f"{actual_sndbuf/(1024*1024):.1f}",
                f"{actual_rcvbuf/(1024*1024):.1f}",
            )
        except Exception as exc:  # pragma: no cover - safety path
            logger.warning(f"⚠️  Failed to optimize TCP socket: {exc}")

    def __del__(self):
        if hasattr(self, "_serialization_executor"):
            try:
                self._serialization_executor.shutdown(wait=False)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    async def register_model(self, model: nn.Module, model_id: Optional[str] = None) -> str:
        fingerprint, metadata = self._compute_fingerprint(model, model_id=model_id)

        if fingerprint in self.registered_models:
            logger.debug("Model %s already registered locally", fingerprint[:8])
            return fingerprint

        ghost_metadata = getattr(model, "_djinn_ghost_metadata", None)
        self.registered_models[fingerprint] = {
            "model_id": model_id,
            "model": model,
            "registered_at": 0,
            "fingerprint_meta": metadata,
            "ghost_metadata": ghost_metadata,
        }

        if self.coordinator:
            try:
                await self.coordinator.register_remote_model(
                    fingerprint=fingerprint,
                    model=model,
                    model_id=model_id,
                    ghost_metadata=ghost_metadata,
                )
                logger.info("✅ Model registered on server: %s", fingerprint[:8])
            except Exception as exc:
                logger.warning("Server registration failed: %s", exc)
        else:
            logger.info("✅ Model registered locally: %s", fingerprint[:8])

        return fingerprint

    async def execute_model(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        model_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        fingerprint, _ = self._compute_fingerprint(model, model_id=model_id)

        if fingerprint not in self.registered_models:
            logger.warning(
                "⚠️  Model %s not registered; auto-registering (first run may be slow)...",
                fingerprint[:16],
            )
            await self.register_model(model, model_id)
            logger.info("✅ Model %s auto-registered", fingerprint[:8])

        return await self._execute_via_cache(
            fingerprint=fingerprint,
            inputs=inputs,
            hints=hints,
            profile_id=profile_id,
        )

    async def _execute_via_graph(self, model: nn.Module, inputs: Dict[str, Any]):
        logger.info("Executing via graph fallback (SLOW)...")
        await asyncio.sleep(0.5)
        if isinstance(model, nn.Module):
            if "input_ids" in inputs:
                return model(inputs["input_ids"])
            if "x" in inputs:
                return model(inputs["x"])
            return model(**inputs)
        return torch.zeros(1)

    async def _execute_via_cache(
        self,
        fingerprint: str,
        inputs: Dict[str, Any],
        hints: Optional[Dict[str, Any]] = None,
        profile_id: Optional[str] = None,
    ) -> torch.Tensor:
        if fingerprint not in self.registered_models:
            raise ModelNotRegisteredError(f"Model {fingerprint} not registered")

        if self.coordinator:
            try:
                qos_hints = hints or {}
                result, metrics = await self.coordinator.execute_remote_model(
                    fingerprint=fingerprint,
                    inputs=inputs,
                    profile_id=profile_id,
                    qos_class=qos_hints.get("qos_class"),
                    deadline_ms=qos_hints.get("deadline_ms"),
                    return_metrics=True,
                )
                self._last_execution_metrics = metrics or {}
                return result
            except Exception as exc:
                logger.warning("Remote execution failed: %s", exc)

        raise RuntimeError(
            f"Cannot execute model {fingerprint}: no coordinator available. "
            "Ensure Djinn server is running and client is connected."
        )

    async def execute_encoder_stage(
        self,
        model: nn.Module,
        encoder_inputs: Dict[str, Any],
        model_id: Optional[str] = None,
        session_id: Optional[str] = None,
        handle_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        if not self.coordinator:
            raise RuntimeError("Coordinator unavailable for stage execution")

        fingerprint, _ = self._compute_fingerprint(model, model_id=model_id)
        if fingerprint not in self.registered_models:
            await self.register_model(model, model_id)

        result = await self.coordinator.execute_remote_stage(
            fingerprint=fingerprint,
            stage="encoder",
            inputs=encoder_inputs,
            session_id=session_id,
            stage_options=handle_metadata,
        )
        return result.get("state_handle"), result.get("session_id") or session_id

    async def execute_decoder_stage(
        self,
        model: nn.Module,
        decoder_inputs: Dict[str, Any],
        state_handle: Optional[Dict[str, Any]],
        model_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[Any, Optional[str]]:
        if not self.coordinator:
            raise RuntimeError("Coordinator unavailable for stage execution")

        fingerprint, _ = self._compute_fingerprint(model, model_id=model_id)
        if fingerprint not in self.registered_models:
            await self.register_model(model, model_id)

        result = await self.coordinator.execute_remote_stage(
            fingerprint=fingerprint,
            stage="decoder",
            inputs=decoder_inputs,
            session_id=session_id,
            state_handle=state_handle,
        )
        return result.get("result"), result.get("session_id") or session_id

    def _compute_fingerprint(
        self,
        model: nn.Module,
        model_id: Optional[str] = None,
        tenant_id: str = "default",
        version_tag: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        return self.fingerprint_policy.compute_fingerprint(
            model=model,
            model_id=model_id,
            tenant_id=tenant_id,
            version_tag=version_tag,
        )

    def _detect_framework(self, model: nn.Module) -> Optional[str]:
        module_name = model.__class__.__module__
        if module_name.startswith("transformers"):
            return "transformers"
        if module_name.startswith("torchvision"):
            return "torchvision"
        return None

    def _serialize_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        tensor_cpu = tensor.detach().cpu()
        return {
            "data": tensor_cpu.tolist(),
            "shape": list(tensor_cpu.shape),
            "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
        }

    def _serialize_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        return {name: self._serialize_tensor(tensor) for name, tensor in weights.items()}

    def _serialize_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        return {name: self._serialize_tensor(tensor) for name, tensor in inputs.items()}

    def _deserialize_tensor(self, data: Dict[str, Any]) -> torch.Tensor:
        dtype = self._dtype_from_string(data.get("dtype", "float32"))
        tensor = torch.tensor(data["data"], dtype=dtype)
        return tensor.reshape(data["shape"])

    @staticmethod
    def _dtype_from_string(dtype_str: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
            "double": torch.float64,
            "int64": torch.int64,
            "long": torch.int64,
            "int32": torch.int32,
            "int": torch.int32,
            "int16": torch.int16,
            "short": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        return mapping.get(dtype_str, torch.float32)

    @property
    def last_execution_metrics(self) -> Dict[str, Any]:
        return self._last_execution_metrics


def get_model_manager() -> EnhancedModelManager:
    global _global_manager
    try:
        manager = _global_manager
    except NameError:
        manager = None
    if manager is None:
        manager = EnhancedModelManager()
        globals()["_global_manager"] = manager
    return manager


_global_manager: Optional[EnhancedModelManager] = None
