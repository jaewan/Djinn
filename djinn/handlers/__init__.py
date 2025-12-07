"""
Phase handlers: Workload-specific semantic phase implementations.

Different workloads handle phases differently. These handlers implement
workload-specific policies without modifying Djinn core.
"""

from .agent_handler import AgentPhaseHandler

__all__ = ["AgentPhaseHandler"]

