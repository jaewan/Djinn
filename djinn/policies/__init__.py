"""
Eviction policies: Different strategies for selecting which sessions to evict.

- SignalDrivenPolicy: Only evict sessions that explicitly signaled readiness
- LRUPolicy: Least-Recently-Used eviction
"""

from .signal_driven import SignalDrivenPolicy
from .lru import LRUPolicy

__all__ = ["SignalDrivenPolicy", "LRUPolicy"]

