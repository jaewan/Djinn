"""
End-to-end KV cache roundtrip correctness test (Phase 4.2).

Validates that swapped and restored KV cache produces identical outputs
to non-swapped cache, ensuring the semantic scheduler doesn't break inference.
"""

import asyncio
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_kv_roundtrip_correctness():
    """Test that swapped/restored KV produces identical logits to non-swapped."""
    
    logger.info("=" * 80)
    logger.info("KV Roundtrip Correctness Test")
    logger.info("=" * 80)
    
    # Load model
    model_id = "meta-llama/Llama-2-7b-hf"
    logger.info(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare input
    prompt = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    logger.info(f"Input shape: {input_ids.shape}")
    
    # Step 1: Run model with KV cache (no swap) - BASELINE
    logger.info("\n1️⃣ Step 1: Baseline inference (no swap)")
    model.eval()
    with torch.no_grad():
        output_baseline = model(input_ids, use_cache=True)
        kv_baseline = output_baseline.past_key_values
        logits_baseline = output_baseline.logits.clone()
    
    logger.info(f"Baseline logits shape: {logits_baseline.shape}")
    logger.info(f"Baseline KV type: {type(kv_baseline)}")
    
    # Step 2: Convert DynamicCache to legacy tuple format (simulate eviction)
    logger.info("\n2️⃣ Step 2: Convert KV to legacy tuple (simulate eviction)")
    if isinstance(kv_baseline, DynamicCache):
        kv_legacy = kv_baseline.to_legacy_cache()
        logger.info(f"Converted DynamicCache to legacy: {len(kv_legacy)} layers")
    else:
        kv_legacy = kv_baseline
    
    # Step 3: Move to CPU (simulate host swap)
    logger.info("\n3️⃣ Step 3: Move KV to CPU (simulate host swap)")
    def move_to_cpu(data):
        if isinstance(data, torch.Tensor):
            data_safe = data.clone().contiguous()
            cpu_buf = torch.empty(data_safe.shape, dtype=data_safe.dtype, device='cpu', pin_memory=True)
            cpu_buf.copy_(data_safe, non_blocking=True)
            return cpu_buf
        elif isinstance(data, (tuple, list)):
            return type(data)(move_to_cpu(item) for item in data)
        else:
            return data
    
    kv_cpu = move_to_cpu(kv_legacy)
    logger.info("✅ KV moved to pinned CPU memory")
    
    # Step 4: Restore KV from CPU (simulate restore)
    logger.info("\n4️⃣ Step 4: Restore KV from CPU (simulate restore)")
    def move_to_gpu(data):
        if isinstance(data, torch.Tensor):
            gpu_tensor = torch.empty_like(data, device='cuda:0')
            gpu_tensor.copy_(data, non_blocking=True)
            return gpu_tensor
        elif isinstance(data, (tuple, list)):
            return type(data)(move_to_gpu(item) for item in data)
        else:
            return data
    
    torch.cuda.current_stream().synchronize()
    kv_restored = move_to_gpu(kv_cpu)
    torch.cuda.current_stream().synchronize()
    logger.info("✅ KV restored to GPU")
    
    # Step 5: Reconstruct DynamicCache from legacy tuple
    logger.info("\n5️⃣ Step 5: Reconstruct DynamicCache from restored tuple")
    kv_reconstructed = DynamicCache.from_legacy_cache(kv_restored)
    
    # Validate reconstruction
    if hasattr(kv_reconstructed, 'get_seq_length'):
        seq_len = kv_reconstructed.get_seq_length()
        logger.info(f"✅ Reconstructed DynamicCache: seq_len={seq_len}")
    else:
        logger.error(f"❌ Reconstructed KV lacks get_seq_length(): type={type(kv_reconstructed)}")
        return False
    
    # Step 6: Run model with restored KV
    logger.info("\n6️⃣ Step 6: Inference with restored KV cache")
    with torch.no_grad():
        output_restored = model(input_ids, past_key_values=kv_reconstructed, use_cache=False)
        logits_restored = output_restored.logits
    
    logger.info(f"Restored logits shape: {logits_restored.shape}")
    
    # Step 7: Verify equivalence
    logger.info("\n7️⃣ Step 7: Verify logits equivalence")
    atol = 1e-3
    rtol = 1e-3
    
    if torch.allclose(logits_baseline, logits_restored, atol=atol, rtol=rtol):
        logger.info(f"✅ PASS: Logits are equivalent (atol={atol}, rtol={rtol})")
        logger.info(f"  Max absolute difference: {(logits_baseline - logits_restored).abs().max().item():.2e}")
        logger.info(f"  Mean absolute difference: {(logits_baseline - logits_restored).abs().mean().item():.2e}")
        return True
    else:
        logger.error(f"❌ FAIL: Logits differ beyond tolerance")
        diff = (logits_baseline - logits_restored).abs()
        logger.error(f"  Max difference: {diff.max().item():.2e}")
        logger.error(f"  Mean difference: {diff.mean().item():.2e}")
        logger.error(f"  >1% difference: {(diff > 0.01).sum().item()} / {diff.numel()}")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_kv_roundtrip_correctness())
    exit(0 if result else 1)


