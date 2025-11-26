"""
Server state management using singleton pattern.

Replaces module-level globals with a centralized state manager.
This improves testability, thread-safety, and maintainability.
"""

import logging
import threading
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)


class ServerState:
    """
    Singleton server state manager.
    
    Manages all global server state including:
    - Device (CPU/GPU)
    - GPU cache
    - Graph cache
    - Statistics
    - Optimization executor
    
    Thread-safe singleton pattern.
    """
    
    _instance: Optional['ServerState'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize server state (private - use get_instance())."""
        if ServerState._instance is not None:
            raise RuntimeError("ServerState is a singleton. Use get_instance() instead.")
        
        self.device: Optional[torch.device] = None
        self.gpu_cache: Optional[Any] = None
        self.graph_cache: Optional[Any] = None
        self.optimization_executor: Optional[Any] = None
        
        self.stats: Dict[str, Any] = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'subgraph_requests': 0,
            'start_time': None,
            'gpu_cache_hits': 0,
            'gpu_cache_misses': 0,
            'graph_cache_hits': 0,
            'graph_cache_misses': 0,
        }
        
        self._initialized = False
        self._gpu_warmed = False  # ✅ Track if GPU has been warmed up (once-only)
        self._warmup_lock = threading.Lock()  # Thread-safe warmup
    
    @classmethod
    def get_instance(cls) -> 'ServerState':
        """
        Get or create the singleton instance.
        
        Thread-safe singleton pattern.
        
        Returns:
            ServerState instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def initialize(self, gpu_id: int = 0) -> None:
        """
        Initialize server components.
        
        Args:
            gpu_id: GPU device ID (default: 0)
        """
        if self._initialized:
            logger.warning("ServerState already initialized, skipping")
            return
        
        # Set device
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"🚀 Server starting with GPU: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️  No GPU available, using CPU")
        
        # Initialize GPU cache
        try:
            from .gpu_cache import get_global_cache
            self.gpu_cache = get_global_cache(max_models=5)
            logger.info("✅ GPU cache initialized (max_models=5)")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize GPU cache: {e}")
            self.gpu_cache = None
        
        # Initialize graph cache
        try:
            from .graph_cache import get_graph_cache
            self.graph_cache = get_graph_cache()
            logger.info("✅ Graph cache initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize graph cache: {e}")
            self.graph_cache = None
        
        # Initialize optimization executor (optional)
        try:
            from .optimizations.optimization_executor import OptimizationExecutor
            self.optimization_executor = OptimizationExecutor(gpu_id=gpu_id)
            logger.info("✅ Optimization executor initialized")
        except Exception as e:
            logger.debug(f"Optimization executor not available: {e}")
            self.optimization_executor = None
        
        # Initialize stats
        import time
        self.stats['start_time'] = time.time()
        
        self._initialized = True
        logger.info("✅ Server state initialized")
    
    def warmup_gpu(self) -> bool:
        """
        Warm up GPU by triggering CUDA kernel JIT compilation (once-only).
        
        This is a one-time operation that benefits all clients. Subsequent calls
        return immediately if already warmed up.
        
        Returns:
            True if warmup succeeded (or already done), False otherwise
        """
        with self._warmup_lock:
            if self._gpu_warmed:
                logger.debug("GPU already warmed up, skipping")
                return True
            
            if self.device is None or self.device.type != 'cuda':
                logger.warning("No CUDA device available, skipping GPU warmup")
                return False
            
            logger.info("🔥 Warming up GPU (JIT compilation)...")
            import time
            warmup_start = time.perf_counter()
            
            try:
                # Create a simple dummy computation to trigger JIT compilation
                # This warms up common CUDA kernels (matmul, elementwise ops, etc.)
                dummy = torch.randn(100, 100, device=self.device)
                _ = torch.matmul(dummy, dummy)  # Matmul kernel
                _ = dummy + dummy  # Elementwise kernel
                _ = torch.relu(dummy)  # Activation kernel
                _ = torch.sum(dummy)  # Reduction kernel
                
                # Synchronize to ensure compilation completes
                torch.cuda.synchronize()
                
                warmup_time = (time.perf_counter() - warmup_start) * 1000
                self._gpu_warmed = True
                logger.info(f"✅ GPU warmed up (JIT compilation): {warmup_time:.1f}ms")
                return True
            except Exception as e:
                warmup_time = (time.perf_counter() - warmup_start) * 1000
                logger.warning(f"GPU warmup failed (took {warmup_time:.1f}ms): {e}")
                return False
    
    def is_gpu_warmed(self) -> bool:
        """Check if GPU has been warmed up."""
        return self._gpu_warmed
    
    def reset(self) -> None:
        """Reset server state (for testing)."""
        self.device = None
        self.gpu_cache = None
        self.graph_cache = None
        self.optimization_executor = None
        self.stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'subgraph_requests': 0,
            'start_time': None,
            'gpu_cache_hits': 0,
            'gpu_cache_misses': 0,
            'graph_cache_hits': 0,
            'graph_cache_misses': 0,
        }
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.stats.copy()
    
    def increment_stat(self, key: str, value: int = 1) -> None:
        """Increment a statistic."""
        if key in self.stats:
            self.stats[key] += value
        else:
            self.stats[key] = value

