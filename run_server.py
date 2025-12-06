#!/usr/bin/env python
"""
Djinn Server Runner - Standalone entry point.

This script ensures proper module resolution and clean startup for background execution.

Usage:
    python run_server.py --node-id 0
    nohup python run_server.py --node-id 0 > server.log 2>&1 &
    setsid python run_server.py --node-id 0 > server.log 2>&1 < /dev/null &
"""

import sys
import os
import asyncio

# Ensure the package root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def main():
    """Main entry point for server."""
    from djinn.server.server import DjinnServer, ServerConfig
    import argparse
    import logging
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Djinn Server")
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--control-port", type=int, default=5555, help="Control port")
    parser.add_argument("--data-port", type=int, default=5556, help="Data port")
    parser.add_argument("--gpus", nargs="*", type=int, help="GPU indices")
    parser.add_argument("--no-dpdk", action="store_true", help="Disable DPDK")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create server config
    config = ServerConfig(
        node_id=args.node_id,
        control_port=args.control_port,
        data_port=args.data_port,
        gpu_indices=args.gpus,
        tcp_fallback=True
    )
    
    async def run():
        """Run the server."""
        server = DjinnServer(config)
        success = await server.start()
        if not success:
            sys.exit(1)
        
        try:
            # Keep server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.stop()
            sys.exit(0)
    
    # Run server
    asyncio.run(run())

if __name__ == "__main__":
    main()

