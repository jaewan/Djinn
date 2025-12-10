"""
OSDI Ablation Study Package

This package contains four ablation studies for Section 5.1 (System Microbenchmarks)
of the OSDI paper submission.

Structure:
  scripts/
    - ablation_os_tax.py: Ablation 1 - OS Tax (Dispatch Overhead)
    - ablation_session_arena.py: Ablation 2 - Session Arena Decomposition
    - ablation_plan_cache.py: Ablation 3 - Plan Cache Effectiveness
    - ablation_semantic_signals.py: Ablation 4 - Semantic Signal Value
    - run_all_ablations.py: Master runner for all ablations
    - generate_ablation_figures.py: Figure generation

Quick Start:
  python scripts/run_all_ablations.py
"""

__version__ = "1.0"
__author__ = "Djinn Project"
