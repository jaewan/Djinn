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
    
    # Phase 3: Semantic Scheduler flags
    parser.add_argument('--enable-semantic-scheduler', action='store_true',
                        help='Enable semantic idle detection and KV swapping (Phase 3)')
    parser.add_argument('--idle-threshold-seconds', type=float, default=1.0,
                        help='Mark session idle after this duration (default: 1.0s)')
    parser.add_argument('--host-swap-pool-gb', type=float, default=32.0,
                        help='Host swap pool size in GB (default: 32)')
    parser.add_argument('--no-lifo-on-overload', action='store_false', dest='lifo_on_overload',
                        help='Disable LIFO scheduling during overload (default: enabled)')
    
    # Experiment 3: Breakpoint debugging flags
    parser.add_argument('--enable-breakpoints', action='store_true',
                        help='Enable breakpoint debugging (Experiment 3: white-box debugging)')
    parser.add_argument('--checkpoint-pool-gb', type=float, default=64.0,
                        help='Activation checkpoint pool size in GB (default: 64)')
    
    args = parser.parse_args()
    configure_logging(args.log_level)
    
    logger.info(f"🚀 Starting Djinn Server v2.3")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info(f"   GPU:  {args.gpu}")
    logger.info(f"   Max concurrent requests: {args.max_concurrent}")
    logger.info(f"   Max VRAM per tenant: {args.max_vram_gb} GB")
    
    # Set CUDA_VISIBLE_DEVICES to ensure server uses correct GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logger.info(f"   CUDA_VISIBLE_DEVICES: {args.gpu}")
    
    # Ring buffer configuration
    if args.ring_buffer:
        logger.info(f"   Ring Buffer: ENABLED")
        logger.info(f"   Ring Buffer capacity: {args.ring_buffer_gb} GB")
        logger.info(f"   Ring Buffer workers: {args.ring_buffer_workers}")
    else:
        logger.info(f"   Ring Buffer: DISABLED")
    
    # Phase 3: Semantic Scheduler configuration
    if args.enable_semantic_scheduler:
        logger.info(f"   Semantic Scheduler: ENABLED")
        logger.info(f"   Idle threshold: {args.idle_threshold_seconds}s")
        logger.info(f"   Host swap pool: {args.host_swap_pool_gb} GB")
        logger.info(f"   LIFO on overload: {args.lifo_on_overload}")
    else:
        logger.info(f"   Semantic Scheduler: DISABLED")
    
    # Experiment 3: Breakpoint debugging configuration
    if args.enable_breakpoints:
        logger.info(f"   Breakpoint Debugging: ENABLED")
        logger.info(f"   Checkpoint pool: {args.checkpoint_pool_gb} GB")
    else:
        logger.info(f"   Breakpoint Debugging: DISABLED")
    
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
    
    # Phase 3: Initialize semantic scheduler components
    print(f"DEBUG: enable_semantic_scheduler = {args.enable_semantic_scheduler}")
    if args.enable_semantic_scheduler:
        from .semantic_idle_detector import get_activity_tracker
        from .host_swap_pool_v2 import get_swap_pool
        from .multi_tenant.kv_session_manager import get_kv_session_manager
        
        # Initialize activity tracker
        activity_tracker = get_activity_tracker(
            idle_threshold_seconds=args.idle_threshold_seconds,
            enabled=True
        )
        
        # Initialize host swap pool
        swap_pool = get_swap_pool(pool_size_gb=args.host_swap_pool_gb)
        
        # **CRITICAL**: Wire semantic scheduler components together
        # Register KV manager callbacks for idle/resume events
        kv_manager = get_kv_session_manager()
        activity_tracker.register_idle_callback(kv_manager.evict_kv_to_host)
        activity_tracker.register_resume_callback(kv_manager.restore_kv_from_host)
        
        # Start activity tracker (pass current event loop for async callbacks)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        activity_tracker.start(event_loop=loop)
        
        logger.info(f"   ✅ SemanticActivityTracker started (idle_threshold={args.idle_threshold_seconds}s)")
        logger.info(f"   ✅ HostSwapPool initialized ({args.host_swap_pool_gb} GB)")
        logger.info(f"   ✅ Callbacks wired: idle→evict, resume→restore")
    
    # Experiment 3: Initialize breakpoint debugging components
    if args.enable_breakpoints:
        try:
            logger.info("Initializing breakpoint debugging components...")
            from .activation_checkpointer import get_activation_checkpointer
            logger.info("✓ Imported get_activation_checkpointer")
            from .breakpoint_manager import get_breakpoint_manager
            logger.info("✓ Imported get_breakpoint_manager")
            from .breakpoint_executor import get_breakpoint_executor
            logger.info("✓ Imported get_breakpoint_executor")
            
            # Initialize checkpointer
            logger.info(f"Initializing ActivationCheckpointer (pool_size_gb={args.checkpoint_pool_gb})...")
            checkpointer = get_activation_checkpointer(pool_size_gb=args.checkpoint_pool_gb)
            logger.info(f"✓ ActivationCheckpointer initialized")
            
            # Initialize breakpoint manager
            logger.info("Initializing BreakpointManager...")
            manager = get_breakpoint_manager()
            logger.info("✓ BreakpointManager initialized")
            
            # Initialize executor
            logger.info("Initializing BreakpointExecutor...")
            executor = get_breakpoint_executor()
            executor.activation_checkpointer = checkpointer
            executor.breakpoint_manager = manager
            logger.info("✓ BreakpointExecutor initialized")
            
            logger.info(f"   ✅ ActivationCheckpointer initialized ({args.checkpoint_pool_gb} GB)")
            logger.info(f"   ✅ BreakpointManager initialized")
            logger.info(f"   ✅ BreakpointExecutor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize breakpoint components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
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
        # Graceful shutdown: close server connections
        if server:
            await server.shutdown()
            logger.info("✅ Server connections closed")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info("✅ Server stopped")


if __name__ == "__main__":
    asyncio.run(main())

