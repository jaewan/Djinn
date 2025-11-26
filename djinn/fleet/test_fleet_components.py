"""
Simple test for fleet components.

Tests basic functionality of GlobalFleetCoordinator components.
"""

import asyncio
import logging
from djinn.fleet import (
    GlobalFleetCoordinator,
    GlobalModelRegistry,
    ServerHealthTracker,
    SemanticLoadBalancer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_model_registry():
    """Test GlobalModelRegistry."""
    logger.info("Testing GlobalModelRegistry...")
    
    registry = GlobalModelRegistry(backend='memory')
    
    # Register models
    await registry.register_model("model1", "server1:5556")
    await registry.register_model("model1", "server2:5556")
    await registry.register_model("model2", "server1:5556")
    
    # Find cached servers
    servers = await registry.find_cached_servers("model1")
    assert "server1:5556" in servers
    assert "server2:5556" in servers
    logger.info(f"✅ Found {len(servers)} servers for model1")
    
    # Unregister
    await registry.unregister_model("model1", "server1:5556")
    servers = await registry.find_cached_servers("model1")
    assert "server1:5556" not in servers
    assert "server2:5556" in servers
    logger.info(f"✅ After unregister: {len(servers)} servers")
    
    logger.info("✅ ModelRegistry test passed")


async def test_server_health():
    """Test ServerHealthTracker."""
    logger.info("Testing ServerHealthTracker...")
    
    tracker = ServerHealthTracker()
    
    # Update health
    await tracker.update_health("server1:5556", {
        'memory_used_gb': 8.0,
        'memory_total_gb': 16.0,
        'gpu_utilization': 50.0,
        'active_requests': 10
    })
    
    # Check health
    assert await tracker.is_healthy("server1:5556")
    metrics = await tracker.get_load_metrics("server1:5556")
    assert metrics['memory_utilization'] == 0.5
    logger.info(f"✅ Server health: {metrics}")
    
    # Filter healthy
    healthy = await tracker.filter_healthy(["server1:5556", "server2:5556"])
    assert "server1:5556" in healthy
    assert "server2:5556" not in healthy
    logger.info(f"✅ Healthy servers: {healthy}")
    
    logger.info("✅ ServerHealthTracker test passed")


async def test_load_balancer():
    """Test SemanticLoadBalancer."""
    logger.info("Testing SemanticLoadBalancer...")
    
    balancer = SemanticLoadBalancer()
    tracker = ServerHealthTracker()
    
    # Add servers
    await tracker.update_health("server1:5556", {
        'memory_used_gb': 8.0,
        'memory_total_gb': 16.0,
        'gpu_utilization': 50.0,
        'active_requests': 10
    })
    await tracker.update_health("server2:5556", {
        'memory_used_gb': 4.0,
        'memory_total_gb': 16.0,
        'gpu_utilization': 25.0,
        'active_requests': 5
    })
    
    # Select best server
    best = await balancer.select_best_server(
        candidates=["server1:5556", "server2:5556"],
        health_tracker=tracker
    )
    assert best in ["server1:5556", "server2:5556"]
    logger.info(f"✅ Selected server: {best}")
    
    logger.info("✅ LoadBalancer test passed")


async def test_global_coordinator():
    """Test GlobalFleetCoordinator."""
    logger.info("Testing GlobalFleetCoordinator...")
    
    coordinator = GlobalFleetCoordinator()
    
    # Register servers
    await coordinator.register_server("server1:5556", {'memory_total_gb': 16.0})
    await coordinator.register_server("server2:5556", {'memory_total_gb': 16.0})
    
    # Update health
    await coordinator.update_server_health("server1:5556", {
        'memory_used_gb': 8.0,
        'memory_total_gb': 16.0,
        'gpu_utilization': 50.0,
        'active_requests': 10
    })
    
    # Route request (no cache)
    server = await coordinator.route_request(
        fingerprint="model1",
        inputs={'x': None}  # Placeholder
    )
    assert server in ["server1:5556", "server2:5556"]
    logger.info(f"✅ Routed to: {server}")
    
    # Get fleet status
    status = await coordinator.get_fleet_status()
    assert status.total_servers == 2
    assert status.healthy_servers == 2
    logger.info(f"✅ Fleet status: {status}")
    
    logger.info("✅ GlobalCoordinator test passed")


async def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("Testing Fleet Components")
    logger.info("="*80)
    
    try:
        await test_model_registry()
        await test_server_health()
        await test_load_balancer()
        await test_global_coordinator()
        
        logger.info("="*80)
        logger.info("✅ All tests passed!")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())

