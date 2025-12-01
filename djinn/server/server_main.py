"""
Djinn Server Main Entry Point

Starts the Djinn server for remote model execution.
"""

import asyncio
import argparse
import logging
import sys

def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('asyncio').setLevel(level)
    logging.getLogger('asyncio').propagate = False

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Djinn Server')
    parser.add_argument('--port', type=int, default=5556, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--log-level', default='warning', help='Server logging level')
    args = parser.parse_args()
    configure_logging(args.log_level)
    
    logger.info(f"🚀 Starting Djinn Server v2.3")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info(f"   GPU:  {args.gpu}")
    
    # Initialize server components
    from .server import DjinnServer, ServerConfig
    
    config = ServerConfig(
        node_id='djinn-server',
        control_port=args.port,
        gpu_indices=[args.gpu]
    )
    
    server = DjinnServer(config)
    
    # Start server
    success = await server.start()
    if not success:
        logger.error("❌ Failed to start server")
        sys.exit(1)
    
    logger.info(f"✅ Server running on {args.host}:{args.port}")
    logger.info("   Press Ctrl+C to stop")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
        # TODO: Add proper shutdown
    
    logger.info("✅ Server stopped")


if __name__ == "__main__":
    asyncio.run(main())

