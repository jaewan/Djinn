#!/usr/bin/env python3
"""
Simple test to verify Djinn server and client can communicate.
"""

import asyncio
import subprocess
import sys
import time

async def test_connection():
    # Start server with output visible
    print("🚀 Starting Djinn server...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "djinn.server", "--port", "5556", "--gpu", "0"],
        stdout=sys.stdout,  # Show server output
        stderr=sys.stderr,
    )
    
    print(f"Server PID: {server_proc.pid}")
    print("Waiting 20 seconds for server to initialize...")
    await asyncio.sleep(20)
    
    if server_proc.poll() is not None:
        print("❌ Server died!")
        return False
    
    print("✅ Server is running")
    
    try:
        # Try to connect
        print("\n📡 Testing connection...")
        from djinn.backend.runtime.initialization import init_async
        from djinn.config import DjinnConfig
        from djinn.core.coordinator import get_coordinator
        
        config = DjinnConfig()
        config.network.remote_server_address = "127.0.0.1:5556"
        await init_async(config)
        coordinator = get_coordinator()
        
        print("✅ Client initialized")
        
        # Try to get capabilities
        try:
            caps = await coordinator.get_capabilities()
            print(f"✅ Got capabilities: {caps.get('gpu_count', 'N/A')} GPUs")
        except Exception as e:
            print(f"⚠️  get_capabilities failed: {e}")
        
        print("\n✅ Connection test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\n🛑 Stopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except:
            server_proc.kill()

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)


