"""
Server-side encoder state cache for streaming workloads (Phase 1).

Stores encoder outputs in the VMU Data Segment, keyed by session + state ID,
and integrates with the SessionManager for automatic cleanup.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, Set

import torch

from djinn.backend.runtime.unified_vmu import UnifiedVMU, get_vmu
from djinn.server.session_manager import SessionManager, get_session_manager

logger = logging.getLogger(__name__)


@dataclass
class StageHandle:
    """Wire-format friendly reference to cached stage outputs."""

    session_id: str
    state_id: str
    stage: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'state_id': self.state_id,
            'stage': self.stage,
            'metadata': self.metadata,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "StageHandle":
        if not payload:
            raise ValueError("Cannot build StageHandle from empty payload")
        return StageHandle(
            session_id=payload.get('session_id', ''),
            state_id=payload.get('state_id', ''),
            stage=payload.get('stage', ''),
            metadata=payload.get('metadata') or {}
        )


@dataclass
class CachedStateRecord:
    """Metadata describing a cached encoder state."""

    session_id: str
    state_id: str
    stage: str
    tensor: torch.Tensor
    shape: Tuple[int, ...]
    dtype: torch.dtype
    num_bytes: int
    offset: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EncoderStateView:
    """
    Lightweight view returned to decoders.

    Provides a `last_hidden_state` attribute (matching HuggingFace outputs)
    and preserves any metadata captured during storage.
    """

    def __init__(self, last_hidden_state: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        self.last_hidden_state = last_hidden_state
        self._metadata = metadata or {}
        for key, value in self._metadata.items():
            if key == "last_hidden_state":
                continue
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation."""
        data = dict(self._metadata)
        data["last_hidden_state"] = self.last_hidden_state
        return data


class StateCache:
    """
    Server-side cache for encoder intermediates.

    Responsibilities:
    - Store encoder outputs on the VMU Data Segment (session-private memory)
    - Provide fast lookup by (session_id, state_id)
    - Integrate with SessionManager for cleanup on disconnect/timeout
    """

    def __init__(
        self,
        vmu: Optional[UnifiedVMU] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        self.vmu = vmu or get_vmu()
        self.session_manager = session_manager or get_session_manager()
        self.device = getattr(self.vmu, "device", torch.device("cpu"))

        self.lock = threading.Lock()
        self.states: Dict[Tuple[str, str], CachedStateRecord] = {}
        self.session_index: Dict[str, Set[str]] = {}
        self.stats = {
            "states_cached": 0,
            "states_released": 0,
            "sessions_cleaned": 0,
        }

        self._registered_cleanup = False
        self._register_cleanup_hook()

    def store_encoder_state(
        self,
        session_id: str,
        state_id: str,
        encoder_outputs: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        stage: str = "encoder",
        handle_metadata: Optional[Dict[str, Any]] = None,
    ) -> StageHandle:
        """
        Store encoder outputs for a session.

        Args:
            session_id: Session owner
            state_id: Unique identifier for this encoder result
            encoder_outputs: Tensor or object with `.last_hidden_state`
            metadata: Optional metadata dictionary preserved for retrieval
        """
        if not session_id:
            raise ValueError("session_id is required")
        if not state_id:
            raise ValueError("state_id is required")

        hidden, extracted_metadata = self._extract_last_hidden_state(encoder_outputs)
        hidden = hidden.detach().contiguous()
        if hidden.device != self.device:
            hidden = hidden.to(self.device, non_blocking=True)
        metadata = {**extracted_metadata, **(metadata or {})}
        handle_metadata = handle_metadata or {}

        storage_tensor, offset, num_bytes = self._write_to_data_segment(
            session_id=session_id,
            state_id=state_id,
            tensor=hidden,
        )

        record = CachedStateRecord(
            session_id=session_id,
            state_id=state_id,
            stage=stage,
            tensor=storage_tensor,
            shape=tuple(hidden.shape),
            dtype=hidden.dtype,
            num_bytes=num_bytes,
            offset=offset,
            metadata=metadata,
        )

        with self.lock:
            previous = self.states.get((session_id, state_id))
            if previous:
                logger.warning(
                    "Overwriting existing encoder state %s for session %s",
                    state_id,
                    session_id,
                )
            self.states[(session_id, state_id)] = record
            self.session_index.setdefault(session_id, set()).add(state_id)
            self.stats["states_cached"] += 1

        self.session_manager.register_ref(session_id, state_id, size_bytes=num_bytes)

        logger.info(
            "Stored encoder state %s (%.2f MB) for session %s",
            state_id,
            num_bytes / 1024**2,
            session_id,
        )
        return StageHandle(
            session_id=session_id,
            state_id=state_id,
            stage=stage,
            metadata={**handle_metadata, **metadata},
        )

    def get_encoder_state(
        self,
        session_id: str,
        state_id: str,
        expected_stage: Optional[str] = None,
    ) -> Optional[EncoderStateView]:
        """
        Retrieve cached encoder outputs.

        Returns:
            EncoderStateView or None if not found.
        """
        with self.lock:
            record = self.states.get((session_id, state_id))

        if record is None:
            return None

        if expected_stage and record.stage != expected_stage:
            raise ValueError(
                f"State {state_id} stored for stage {record.stage}, expected {expected_stage}"
            )

        tensor = record.tensor.view(record.shape)
        return EncoderStateView(last_hidden_state=tensor, metadata=dict(record.metadata))

    def get_state_by_handle(self, handle: StageHandle) -> EncoderStateView:
        """Retrieve cached state using an opaque handle."""
        view = self.get_encoder_state(
            session_id=handle.session_id,
            state_id=handle.state_id,
            expected_stage=handle.stage or None,
        )
        if view is None:
            raise RuntimeError(
                f"State {handle.state_id} not found for session {handle.session_id}"
            )
        return view

    def release_state(self, session_id: str, state_id: str) -> bool:
        """
        Release a cached encoder state (manual cleanup).
        """
        with self.lock:
            key = (session_id, state_id)
            record = self.states.pop(key, None)
            if not record:
                return False

            session_states = self.session_index.get(session_id)
            if session_states and state_id in session_states:
                session_states.discard(state_id)
                if not session_states:
                    self.session_index.pop(session_id, None)

            self.stats["states_released"] += 1

        self.session_manager.release_ref(session_id, state_id)
        logger.debug("Released encoder state %s for session %s", state_id, session_id)
        return True

    def cleanup_session(self, session_id: str) -> None:
        """
        Cleanup hook invoked when SessionManager kills a session.

        Removes in-memory references and releases VMU metadata for the session.
        """
        with self.lock:
            state_ids = list(self.session_index.pop(session_id, set()))
            self.stats["sessions_cleaned"] += 1

        for state_id in state_ids:
            self.release_state(session_id, state_id)

        if hasattr(self.vmu, "free_session_data"):
            try:
                self.vmu.free_session_data(session_id)
            except Exception as exc:
                logger.error(
                    "Failed to free VMU data for session %s: %s", session_id, exc
                )

        logger.info("Cleaned encoder states for session %s (%d states)", session_id, len(state_ids))

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self.lock:
            return {
                **self.stats,
                "states_active": len(self.states),
                "sessions_tracked": len(self.session_index),
            }

    # Internal helpers -------------------------------------------------
    def _write_to_data_segment(
        self,
        session_id: str,
        state_id: str,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[int], int]:
        num_bytes = tensor.element_size() * tensor.numel()

        if not hasattr(self.vmu, "allocate_session_data"):
            # Fallback: clone tensor without VMU integration (should not happen in production).
            logger.warning("VMU missing allocate_session_data; falling back to tensor clone.")
            clone = tensor.to(self.device).clone()
            return clone, None, num_bytes

        offset = self.vmu.allocate_session_data(
            session_id=session_id,
            size_bytes=num_bytes,
            name=f"encoder_state:{state_id}",
        )
        view = self.vmu.get_session_data_view(
            session_id=session_id,
            offset=offset,
            size=num_bytes,
            dtype=tensor.dtype,
        )
        view = view.view(tensor.shape)

        copy_non_blocking = tensor.device.type == "cuda" and self.device.type == "cuda"

        view.copy_(tensor, non_blocking=copy_non_blocking)

        return view, offset, num_bytes

    def _extract_last_hidden_state(self, encoder_outputs: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        extras: Dict[str, Any] = {}
        if encoder_outputs is None:
            raise ValueError("encoder_outputs cannot be None")
        if isinstance(encoder_outputs, torch.Tensor):
            return encoder_outputs, extras
        if hasattr(encoder_outputs, "last_hidden_state"):
            hidden_state = encoder_outputs.last_hidden_state
            if hidden_state is None:
                raise ValueError("encoder_outputs.last_hidden_state is None")
            extras = self._collect_metadata(getattr(encoder_outputs, "__dict__", {}))
            return hidden_state, extras
        if isinstance(encoder_outputs, dict) and "last_hidden_state" in encoder_outputs:
            hidden_state = encoder_outputs.get("last_hidden_state")
            if hidden_state is None:
                raise ValueError("encoder_outputs['last_hidden_state'] is None")
            extras = {
                key: value
                for key, value in encoder_outputs.items()
                if key != "last_hidden_state"
            }
            return hidden_state, extras
        if isinstance(encoder_outputs, (list, tuple)) and encoder_outputs:
            first = encoder_outputs[0]
            if first is None:
                raise ValueError("encoder_outputs[0] is None")
            if isinstance(first, torch.Tensor):
                return first, extras
        raise ValueError(f"encoder_outputs must provide a last_hidden_state tensor, got type {type(encoder_outputs)}")

    @staticmethod
    def _collect_metadata(source: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key, value in source.items():
            if key.startswith("_") or key == "last_hidden_state":
                continue
            metadata[key] = value
        return metadata

    def _register_cleanup_hook(self) -> None:
        if self._registered_cleanup:
            return
        try:
            self.session_manager.register_cleanup_callback(self.cleanup_session)
            self._registered_cleanup = True
            logger.debug("StateCache registered session cleanup callback")
        except Exception as exc:
            logger.error(f"Failed to register cleanup callback: {exc}")


_global_state_cache: Optional[StateCache] = None


def get_state_cache() -> StateCache:
    """Global accessor for the StateCache singleton."""
    global _global_state_cache
    if _global_state_cache is None:
        _global_state_cache = StateCache()
    return _global_state_cache

