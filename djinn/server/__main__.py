"""
Entry point for running Djinn server as a module.

Usage: python -m djinn.server --port 5556 --gpu 0
"""

import asyncio
from djinn.server.server_main import main

if __name__ == "__main__":
    asyncio.run(main())

