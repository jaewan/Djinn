from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .dpdk_bindings import get_dpdk

@dataclass
class DMABuffer:
    data_ptr: int
    size: int
    dma_handle: Optional[dict]
    pool_name: str


class DPDKAllocator:
    """Shim allocator with pinned-memory fallback.

    In Phase 2/3 this returns pinned-memory backed tensors for efficient host
    transfer. In Phase 4, integrate DPDK EAL and gpudev registration to
    populate `dma_handle` with IOVA/lkey/rkey.
    """

    def __init__(self) -> None:
        self.eal_initialized = False
        self.memory_pools: Dict[str, object] = {}
        self.huge_page_size = "2MB"
        self._dpdk = get_dpdk()
        self.initialize_dpdk()

    def initialize_dpdk(self) -> None:
        # Attempt to initialize DPDK EAL via C++ runtime if available
        try:
            import djinn._runtime as _rt  # type: ignore
            if _rt.dpdk_available() and _rt.eal_init([]):
                _rt.create_default_pool("genie_pool", 8192, 4096)
                self.eal_initialized = True
            else:
                self.eal_initialized = False
        except Exception:
            # Fallback to Python-side probe
            if self._dpdk.is_available():
                self.eal_initialized = self._dpdk.init_eal()
            else:
                self.eal_initialized = False
        self.create_memory_pools()

    def create_memory_pools(self) -> None:
        # Create logical pools for sizing; real pools will come with DPDK binding
        self.memory_pools = {
            "small": object(),
            "medium": object(),
            "large": object(),
            "huge": object(),
        }

    def select_pool(self, size: int) -> str:
        if size <= 256 << 10:
            return "small"
        if size <= 2 << 20:
            return "medium"
        if size <= 32 << 20:
            return "large"
        return "huge"

    def allocate(self, size: int, device: torch.device) -> DMABuffer:
        # If DPDK is available, allocate an mbuf and return a handle
        try:
            import djinn._runtime as _rt  # type: ignore
            if _rt.dpdk_available() and self.eal_initialized:
                mbuf_ptr = _rt.alloc_mbuf(0)
                if mbuf_ptr:
                    pool_name = self.select_pool(size)
                    return DMABuffer(
                        data_ptr=int(mbuf_ptr),
                        size=size,
                        dma_handle={"rte_mbuf": int(mbuf_ptr)},
                        pool_name=pool_name,
                    )
        except Exception:
            pass

        # For now, prefer pinned tensor when CUDA available; otherwise, CPU tensor
        numel = (size + 3) // 4
        use_pin = False
        try:
            use_pin = torch.cuda.is_available()
        except Exception:
            use_pin = False
        if use_pin:
            try:
                t = torch.empty(numel, dtype=torch.int32, pin_memory=True)
            except Exception:
                t = torch.empty(numel, dtype=torch.int32)
        else:
            t = torch.empty(numel, dtype=torch.int32)
        data_ptr = t.data_ptr()
        # Keep a reference to avoid GC of the underlying storage
        if not hasattr(self, "_keepalive"):
            self._keepalive = []  # type: ignore[attr-defined]
        self._keepalive.append(t)  # type: ignore[attr-defined]
        pool_name = self.select_pool(size)
        return DMABuffer(data_ptr=data_ptr, size=size, dma_handle=None, pool_name=pool_name)


