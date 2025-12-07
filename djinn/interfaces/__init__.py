"""
Djinn extensibility interfaces.

These abstract base classes define the contract for pluggable components:
- SwappableState: Any state that can be swapped GPU<->CPU
- PhaseHandler: Workload-specific phase signal handling
- EvictionPolicy: Policy for selecting which sessions to evict
- InferenceBackend: Framework-agnostic inference execution
"""

from .swappable_state import SwappableState
from .phase_handler import PhaseHandler
from .eviction_policy import EvictionPolicy, EvictionCandidate
from .inference_backend import InferenceBackend

__all__ = [
    "SwappableState",
    "PhaseHandler",
    "EvictionPolicy",
    "EvictionCandidate",
    "InferenceBackend",
]

