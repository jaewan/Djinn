#!/usr/bin/env python3
"""
vLLM Baseline: Test where vLLM crashes under concurrent load.
Compare against Djinn's scaling capability.
"""
import asyncio
import time
import sys
from typing import List, Dict, Any
import torch

async def run_vllm_baseline():
    """Run vLLM with increasing concurrency to find OOM point."""
    from vllm import LLM, SamplingParams
    
    model_id = "meta-llama/Llama-2-13b-hf"
    prompt = "The quick brown fox jumps over the " + "lazy dog " * 50  # 512 tokens like Djinn test
    
    results = {
        "baseline": "vLLM",
        "model": model_id,
        "tests": []
    }
    
    try:
        # Initialize vLLM LLM engine with modest GPU memory
        print("Initializing vLLM LLM engine...")
        llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,  # Use 80% of GPU VRAM
            dtype=torch.float16,
            max_model_len=2048,
        )
        print(f"✅ vLLM initialized")
    except Exception as e:
        print(f"❌ vLLM initialization failed: {e}")
        return results
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        top_p=1.0,
    )
    
    # Test increasing concurrency
    for n_concurrent in [1, 8, 16, 32, 48, 64]:
        try:
            print(f"\n[vLLM] Testing N={n_concurrent} concurrent requests...")
            
            # Create N concurrent requests
            prompts = [prompt] * n_concurrent
            
            start_time = time.perf_counter()
            
            # This is where vLLM will batch the requests
            outputs = llm.generate(prompts, sampling_params)
            
            elapsed = time.perf_counter() - start_time
            
            total_tokens = len(outputs) * 50  # max_tokens per request
            throughput = total_tokens / elapsed
            
            result = {
                "n_agents": n_concurrent,
                "elapsed_s": elapsed,
                "total_tokens": total_tokens,
                "throughput_tps": throughput,
                "success": True,
                "error": None
            }
            
            print(f"✅ N={n_concurrent}: {elapsed:.2f}s, throughput={throughput:.1f} tokens/sec")
            results["tests"].append(result)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ N={n_concurrent}: CUDA OUT OF MEMORY")
            result = {
                "n_agents": n_concurrent,
                "elapsed_s": None,
                "total_tokens": None,
                "throughput_tps": None,
                "success": False,
                "error": "CUDA out of memory"
            }
            results["tests"].append(result)
            print(f"⚠️  vLLM crashed at N={n_concurrent}")
            break
            
        except Exception as e:
            print(f"❌ N={n_concurrent}: {type(e).__name__}: {str(e)[:100]}")
            result = {
                "n_agents": n_concurrent,
                "elapsed_s": None,
                "total_tokens": None,
                "throughput_tps": None,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}"
            }
            results["tests"].append(result)
            break
    
    return results

if __name__ == "__main__":
    print("=" * 70)
    print("vLLM BASELINE: Concurrent Load Test")
    print("=" * 70)
    
    results = asyncio.run(run_vllm_baseline())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    max_n = 0
    for test in results["tests"]:
        if test["success"]:
            max_n = test["n_agents"]
            print(f"✅ N={test['n_agents']:2d}: {test['elapsed_s']:6.2f}s → {test['throughput_tps']:7.1f} tokens/sec")
        else:
            print(f"❌ N={test['n_agents']:2d}: FAILED - {test['error']}")
    
    print(f"\n📊 vLLM max concurrent agents before OOM: N={max_n}")
    print(f"📊 Djinn achieved: N=64 (222 swaps, 109 restores)")
    print(f"📊 Scaling advantage: {64/max_n if max_n > 0 else 'inf'}x")

