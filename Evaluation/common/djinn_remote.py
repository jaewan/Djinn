from __future__ import annotations

import logging
import os
from typing import Optional

from Evaluation.common.djinn_init import ensure_initialized, ensure_initialized_before_async

logger = logging.getLogger(__name__)

REMOTE_DEVICE_PREFIX = "privateuseone"


def resolve_server_address(explicit: Optional[str]) -> Optional[str]:
    """
    Determine which Djinn server address to use.
    """
    return explicit or os.environ.get("GENIE_SERVER_ADDRESS")


def configure_remote_backend(args) -> Optional[str]:
    """
    Prepare the Djinn backend if requested.

    Returns the resolved device string when the backend is djinn, otherwise None.
    """
    backend = getattr(args, "backend", "local")
    if backend != "djinn":
        return None

    server_address = resolve_server_address(getattr(args, "djinn_server", None))
    os.environ.setdefault("DJINN_ENABLE_PRIVATEUSE1_DEVICE", "1")
    logger.info("Initializing Djinn runtime (backend=%s, server=%s)", backend, server_address or "default")
    
    # Use ensure_initialized_before_async if we're about to enter async context
    # This ensures sync init happens before asyncio.run()
    ensure_initialized_before_async(server_address)

    device_index = getattr(args, "djinn_device_index", 0)
    device = f"{REMOTE_DEVICE_PREFIX}:{device_index}"

    setattr(args, "device", device)
    setattr(args, "djinn_server_address", server_address)

    logger.info("Using remote device %s for Djinn backend", device)
    return device

