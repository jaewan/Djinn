#!/usr/bin/env python3
"""
Test if semantic scheduler triggers swaps when agents are actually idle.
"""
import asyncio
import time
from djinn.core import djinn
from transformers import AutoTokenizer

async def test_single_agent_with_idle():
    """Test that a single agent triggers swap after going idle."""
    
    model_id = "meta-llama/Llama-2-13b-hf"
    
    # Initialize djinn
    djinn.init(model_id, server="localhost:5556")
    
    # Prefill: Register session and do one inference
    print("\n[TEST] Starting prefill phase...")
    with djinn.session(phase="llm_prefill", session_id="test_idle_agent"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = "The quick brown fox jumps over the " + "lazy dog " * 50  # Long prompt to use KV cache
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print(f"[TEST] Prefill input shape: {input_ids.shape}")
        
        output = djinn.generate(input_ids, max_new_tokens=5)
        print(f"[TEST] Prefill complete. Output: {output}")
    
    # Now GO IDLE: Session exists but no new requests for 3 seconds
    print("\n[TEST] Entering IDLE phase (3 seconds, no activity)...")
    print("[TEST] Idle detector threshold is 1.0s, should trigger swap after ~1-2 seconds")
    
    for i in range(3):
        time.sleep(1)
        print(f"[TEST] Idle for {i+1}s...")
        
        # Check metrics from server
        import requests
        try:
            resp = requests.get("http://localhost:5556/metrics", timeout=1)
            if resp.status_code == 200:
                metrics = resp.json()
                swaps = metrics.get("swaps", 0)
                print(f"    Swaps so far: {swaps}")
        except:
            pass
    
    # RESUME: Do another operation (should trigger restore if swap happened)
    print("\n[TEST] Resuming DECODE phase...")
    with djinn.session(phase="llm_decode", session_id="test_idle_agent"):
        output = djinn.generate(input_ids, max_new_tokens=5)
        print(f"[TEST] Decode complete. Output: {output}")
    
    print("\n[TEST] Test complete!")

if __name__ == "__main__":
    asyncio.run(test_single_agent_with_idle())

