import warnings
from packaging import version
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import subprocess


def validate_environment(strict: bool = False) -> bool:
	"""Validate core runtime versions for Djinn Phase 1."""
	success = True

	if not HAS_TORCH:
		warnings.warn("PyTorch not installed. Some features may be unavailable.")
		return not strict
	
	# PyTorch version
	torch_ver = version.parse(torch.__version__.split("+")[0])
	# Support both legacy (2.2–2.5) and new (2.8) tracks
	if not ((version.parse("2.2.0") <= torch_ver < version.parse("2.6.0")) or torch_ver >= version.parse("2.8.0")):
		warnings.warn(f"Djinn targets PyTorch 2.2.x–2.5.x or 2.8+; found {torch.__version__}")
		success = False

	# CUDA availability is optional in Phase 1

	# Optional: DPDK presence (warn if missing)
	try:
		res = subprocess.run(["pkg-config", "--modversion", "libdpdk"], capture_output=True, text=True)
		if res.returncode != 0:
			warnings.warn("DPDK not found; zero-copy features disabled")
	except Exception:
		warnings.warn("Could not verify DPDK version")

	return success or not strict


