"""Shared helpers for evaluation experiments."""

# Intentionally avoid importing submodules here so optional deps stay lazy.  Users
# can import the helper they need explicitly.

__all__ = ["gpu", "metrics", "workloads", "experiment_runner", "djinn_init"]



