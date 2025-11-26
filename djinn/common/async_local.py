"""
Async-friendly replacement for threading.local().

ContextVar-backed storage ensures state survives across asyncio context switches
while still isolating data between OS threads when necessary.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict


class AsyncLocal:
    """
    Lightweight drop-in replacement for threading.local() that works with asyncio.

    Attributes set on an AsyncLocal instance are scoped to the current context
    (task or thread) and automatically preserved across `await` boundaries.
    """

    __slots__ = ("_context_var", "_name")

    def __init__(self, name: str):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_context_var", ContextVar(name, default=None))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_state(self) -> Dict[str, Any]:
        state = self._context_var.get()
        if state is None:
            state = {}
            self._context_var.set(state)
        return state

    def _replace_state(self, state: Dict[str, Any]) -> None:
        # Always store a copy to avoid sharing mutable state between contexts
        self._context_var.set(dict(state))

    # ------------------------------------------------------------------ #
    # threading.local()-style API
    # ------------------------------------------------------------------ #
    def __getattr__(self, item: str) -> Any:
        state = self._ensure_state()
        if item in state:
            return state[item]
        raise AttributeError(f"{self._name} has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key in AsyncLocal.__slots__:
            object.__setattr__(self, key, value)
            return
        state = dict(self._ensure_state())
        state[key] = value
        self._replace_state(state)

    def __delattr__(self, item: str) -> None:
        state = dict(self._ensure_state())
        if item not in state:
            raise AttributeError(f"{self._name} has no attribute '{item}'")
        del state[item]
        self._replace_state(state)

    def clear(self) -> None:
        """Remove all attributes for the current context."""
        self._context_var.set({})

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the current context state."""
        return dict(self._ensure_state())

    def __repr__(self) -> str:
        return f"AsyncLocal(name={self._name!r}, state_keys={list(self._ensure_state().keys())})"

