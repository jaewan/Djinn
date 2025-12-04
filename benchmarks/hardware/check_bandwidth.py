import torch
import time
import psutil
import os

def check_bandwidth(device_id=0, size_gb=2):
    device = torch.device(f"cuda:{device_id}")
    props = torch.cuda.get_device_properties(device)
    print(f"\n--- Checking GPU {device_id}: {props.name} ---")
    
    # 1. Check System RAM
    mem_info = psutil.virtual_memory()
    print(f"Host RAM Available: {mem_info.available / 1024**3:.2f} GB")
    
    # 2. Allocation (Pinned)
    print(f"Allocating {size_gb}GB Pinned Memory...")
    num_elements = size_gb * 1024**3 // 4  # float32 = 4 bytes
    try:
        # Create CPU tensor and pin it
        x_host = torch.empty(num_elements, dtype=torch.float32).pin_memory()
        x_device = torch.empty(num_elements, dtype=torch.float32, device=device)
    except RuntimeError as e:
        print(f"❌ FAIL: Allocation failed. {e}")
        return

    # 3. Warmup
    print("Warming up...")
    torch.cuda.synchronize()
    for _ in range(5):
        x_device.copy_(x_host, non_blocking=True)
    torch.cuda.synchronize()

    # 4. Benchmark Host -> Device (H2D)
    print("Benchmarking H2D (Host -> Device)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iters = 20
    start_event.record()
    for _ in range(iters):
        x_device.copy_(x_host, non_blocking=True)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    total_gb = iters * size_gb
    bandwidth = total_gb / (elapsed_ms / 1000.0)
    
    print(f"✅ H2D Bandwidth: {bandwidth:.2f} GB/s")

    # 5. Interpret
    if bandwidth > 24.0:
        print("Verdict: 🚀 EXCELLENT (PCIe Gen 4 x16)")
    elif bandwidth > 12.0:
        print("Verdict: ⚠️ MEDIOCRE (Likely Gen 3 x16 or Gen 4 x8)")
    else:
        print("Verdict: ❌ FAIL (Something is wrong)")

if __name__ == "__main__":
    # Check GPU 0
    check_bandwidth(0)
