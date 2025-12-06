"""
Entry point for running Djinn server as a module.

Usage: python -m djinn.server --node-id 0
"""

from djinn.server.server import main

if __name__ == "__main__":
    main()

