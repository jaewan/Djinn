#!/usr/bin/env python3
"""
Djinn initialization utilities for OSDI Evaluation scripts.

Provides client initialization before entering async contexts.
"""

import logging
import sys
import os

logger = logging.getLogger(__name__)


def ensure_initialized_before_async(server_address: str) -> None:
    """
    Initialize Djinn client before entering async context.

    This ensures the coordinator and transports are set up correctly
    before any async operations begin.

    Args:
        server_address: Server address in format "host:port"
    """
    try:
        # Add current directory to path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Evaluation directory
        root_dir = os.path.dirname(parent_dir)     # Djinn root

        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        # Import and initialize Djinn
        import djinn
        from djinn.backend.runtime.initialization import init

        # Initialize with server address
        init(server_address=server_address)

        logger.info(f"Djinn init succeeded (server={server_address})")

    except Exception as e:
        logger.error(f"Djinn init failed: {e}")
        raise