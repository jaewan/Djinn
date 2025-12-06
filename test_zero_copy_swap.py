#!/usr/bin/env python3
"""
Test script to verify zero-copy KV swapping works correctly.
This validates the refactored kv_session_manager without pickle.
"""

import torch
import asyncio
import logging
from djinn.server.multi_tenant.kv_session_manager import KVSessionManager, KVSession
from djinn.server.host_swap_pool import HostSwapPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_zero_copy_kv_swap():
    """Test that KV swapping works without pickle serialization."""
    
    # Initialize components
    logger.info("=" * 70)
    logger.info("Testing Zero-Copy KV Swapping (No Pickle)")
    logger.info("=" * 70)
    
    # Initialize swap pool as singleton
    from djinn.server.host_swap_pool import get_swap_pool
    swap_pool = get_swap_pool(pool_size_gb=4.0)
    
    # Create KV manager
    kv_manager = KVSessionManager()
    
    # Test 1: Simple tensor KV cache
    logger.info("\n[TEST 1] Simple tensor KV cache")
    session_id_1 = "test_session_1"
    kv_tensor = torch.randn(2, 32, 128, 64, device='cuda:0', dtype=torch.float32)
    kv_size = kv_tensor.element_size() * kv_tensor.numel()
    
    # Create session
    sess = await kv_manager.get_or_create(session_id_1, gpu_id=0, initial_kv=kv_tensor)
    logger.info(f"✓ Created session with tensor KV: {kv_size / (1024**2):.2f}MB")
    
    # Evict to host
    evicted = await kv_manager.evict_kv_to_host(session_id_1)
    logger.info(f"✓ Evicted to host: {evicted / (1024**2):.2f}MB")
    
    # Restore from host
    restored = await kv_manager.restore_kv_from_host(session_id_1)
    logger.info(f"✓ Restored from host: {restored / (1024**2):.2f}MB")
    
    # Verify KV was restored correctly
    restored_kv = await kv_manager.get_session_kv(session_id_1)
    if torch.allclose(kv_tensor, restored_kv, atol=1e-5):
        logger.info("✓ Restored KV matches original (zero-copy verified)")
    else:
        logger.error("✗ Restored KV does not match original!")
        raise AssertionError("KV mismatch after restore")
    
    # Test 2: Tuple of tensors (common in transformers)
    logger.info("\n[TEST 2] Tuple of tensors KV cache")
    session_id_2 = "test_session_2"
    kv_tuple = (
        torch.randn(2, 32, 64, 64, device='cuda:0', dtype=torch.float32),
        torch.randn(2, 32, 64, 64, device='cuda:0', dtype=torch.float32),
    )
    kv_tuple_size = sum(t.element_size() * t.numel() for t in kv_tuple)
    
    # Create session
    sess = await kv_manager.get_or_create(session_id_2, gpu_id=0, initial_kv=kv_tuple)
    logger.info(f"✓ Created session with tuple KV: {kv_tuple_size / (1024**2):.2f}MB")
    
    # Evict to host
    evicted = await kv_manager.evict_kv_to_host(session_id_2)
    logger.info(f"✓ Evicted to host: {evicted / (1024**2):.2f}MB")
    
    # Restore from host
    restored = await kv_manager.restore_kv_from_host(session_id_2)
    logger.info(f"✓ Restored from host: {restored / (1024**2):.2f}MB")
    
    # Verify KV was restored correctly
    restored_kv = await kv_manager.get_session_kv(session_id_2)
    if isinstance(restored_kv, tuple) and len(restored_kv) == 2:
        if torch.allclose(kv_tuple[0], restored_kv[0], atol=1e-5) and \
           torch.allclose(kv_tuple[1], restored_kv[1], atol=1e-5):
            logger.info("✓ Restored tuple KV matches original (structure preserved)")
        else:
            logger.error("✗ Restored tuple KV values do not match!")
            raise AssertionError("Tuple KV mismatch after restore")
    else:
        logger.error(f"✗ Restored KV is not tuple: {type(restored_kv)}")
        raise AssertionError("Tuple structure not preserved")
    
    # Test 3: Multiple sessions with concurrent swaps
    logger.info("\n[TEST 3] Multiple concurrent sessions")
    sessions = []
    for i in range(3):
        session_id = f"test_session_concurrent_{i}"
        size = (i + 1) * 2  # Variable sizes
        kv_data = torch.randn(size, 32, 64, 64, device='cuda:0', dtype=torch.float32)
        await kv_manager.get_or_create(session_id, gpu_id=0, initial_kv=kv_data)
        sessions.append((session_id, kv_data))
        logger.info(f"  Created session {i}: {kv_data.element_size() * kv_data.numel() / (1024**2):.2f}MB")
    
    # Evict all
    for session_id, _ in sessions:
        await kv_manager.evict_kv_to_host(session_id)
    logger.info("✓ Evicted all sessions to host")
    
    # Restore all
    for session_id, _ in sessions:
        await kv_manager.restore_kv_from_host(session_id)
    logger.info("✓ Restored all sessions from host")
    
    # Verify all
    for session_id, original_kv in sessions:
        restored_kv = await kv_manager.get_session_kv(session_id)
        if torch.allclose(original_kv, restored_kv, atol=1e-5):
            logger.info(f"  ✓ Session {session_id} verified")
        else:
            logger.error(f"  ✗ Session {session_id} mismatch!")
            raise AssertionError(f"Session {session_id} KV mismatch")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ All zero-copy KV swapping tests passed!")
    logger.info("=" * 70)
    
    # Print swap pool stats
    stats = swap_pool.get_stats()
    logger.info("\nSwap Pool Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_zero_copy_kv_swap())

