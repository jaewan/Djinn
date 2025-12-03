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
    parser.add_argument('--max-concurrent', type=int, default=64, 
                        help='Max concurrent requests (default: 64 for high-concurrency experiments)')
    parser.add_argument('--max-vram-gb', type=float, default=80.0,
                        help='Max VRAM per tenant in GB (default: 80 for A100-80GB)')
    
    # Ring buffer flags
    parser.add_argument('--ring-buffer', action='store_true', 
                        help='Enable ring buffer mode for oversized models')
    parser.add_argument('--ring-buffer-gb', type=float, default=48.0,
                        help='Ring buffer capacity in GB (default: 48)')
    parser.add_argument('--ring-buffer-workers', type=int, default=1,
                        help='Number of prefetch workers (default: 1)')
    
    args = parser.parse_args()
    configure_logging(args.log_level)
    
    logger.info(f"🚀 Starting Djinn Server v2.3")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info(f"   GPU:  {args.gpu}")
    logger.info(f"   Max concurrent requests: {args.max_concurrent}")
    logger.info(f"   Max VRAM per tenant: {args.max_vram_gb} GB")
    
    # Ring buffer configuration
    if args.ring_buffer:
        logger.info(f"   Ring Buffer: ENABLED")
        logger.info(f"   Ring Buffer capacity: {args.ring_buffer_gb} GB")
        logger.info(f"   Ring Buffer workers: {args.ring_buffer_workers}")
    else:
        logger.info(f"   Ring Buffer: DISABLED")
    
    # Initialize server components
    from .server import DjinnServer, ServerConfig
    import os
    
    # Set environment variables to control server configuration
    os.environ['GENIE_QOS_MAX_CONCURRENCY'] = str(args.max_concurrent)
    
    # Set ring buffer environment variables
    if args.ring_buffer:
        os.environ['GENIE_VMU_RING_BUFFER'] = '1'
        os.environ['GENIE_VMU_RING_BUFFER_GB'] = str(args.ring_buffer_gb)
        os.environ['GENIE_VMU_RING_BUFFER_WORKERS'] = str(args.ring_buffer_workers)
    
    config = ServerConfig(
        node_id='djinn-server',
        control_port=args.port,
        gpu_indices=[args.gpu]
    )
    
    server = DjinnServer(config)
    
    # Configure tenant limits with CLI parameters
    from .tenant_resource_policy import TenantLimits
    server.tenant_resource_policy.configure_tenant('default', TenantLimits(
        max_vram_gb=args.max_vram_gb,
        max_concurrent_requests=args.max_concurrent,
        priority=1,
    ))
    
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

