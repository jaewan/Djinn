"""
Phase 4: Metric Fidelity & Correctness Validation

This module contains three key validation tests:

1. **Logit Equivalence**: Verify ring buffer output matches PyTorch
2. **PCIe Flatline**: Verify ring buffer saturates PCIe bandwidth
3. **Agent Sleep/Resume**: Verify semantic scheduler lifecycle

See README.md for detailed documentation.
"""

__version__ = "1.0"
__all__ = [
    "test_logit_equivalence",
    "test_pcie_flatline",
    "test_agent_sleep_resume",
    "run_all_validations",
]

