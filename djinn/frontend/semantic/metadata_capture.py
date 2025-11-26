"""
Metadata capture during LazyTensor construction with Phase 2.2 parallelization.

Key insight: We can extract semantic info from:
1. Call stack (module path)
2. Operation sequence (pattern hints)
3. Tensor properties (shape, dtype)

Phase 2.2 Enhancement:
- Support batch metadata capture
- Optional parallel annotation for large graphs
- Statistics tracking for optimization monitoring
"""

import inspect
import torch.nn as nn
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
import threading


# ============================================================================
# P0 OPTIMIZATION: Parallel Metadata Annotation (Phase 2.2)
# ============================================================================

class MetadataAnnotationStats:
    """Track metadata annotation performance."""
    
    def __init__(self):
        self.total_nodes = 0
        self.total_time_ms = 0
        self.batch_count = 0
        self._lock = threading.Lock()
    
    def record(self, num_nodes: int, time_ms: float):
        """Record annotation metrics."""
        with self._lock:
            self.total_nodes += num_nodes
            self.total_time_ms += time_ms
            self.batch_count += 1
    
    def get_stats(self) -> Dict:
        """Get annotation statistics."""
        with self._lock:
            avg_time = (self.total_time_ms / self.batch_count) if self.batch_count > 0 else 0
            avg_per_node = (self.total_time_ms / self.total_nodes) if self.total_nodes > 0 else 0
            return {
                'total_nodes': self.total_nodes,
                'total_time_ms': f"{self.total_time_ms:.2f}",
                'batch_count': self.batch_count,
                'avg_batch_time_ms': f"{avg_time:.2f}",
                'avg_per_node_ms': f"{avg_per_node:.3f}",
            }


# Global stats tracker
_annotation_stats = MetadataAnnotationStats()


def get_metadata_annotation_stats() -> Dict:
    """Get metadata annotation statistics."""
    return _annotation_stats.get_stats()


class MetadataCapture:
    """Captures semantic metadata during graph construction with optional parallelization."""

    def __init__(self, use_parallel: bool = True, max_workers: int = 4, lazy_metadata: bool = True):
        self._module_stack = []  # Track nn.Module hierarchy
        self._operation_sequence = []  # Track recent operations
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.lazy_metadata = lazy_metadata  # ✅ NEW: Lazy metadata capture flag
        self._metadata_cache = {}  # ✅ NEW: Cache metadata per operation

    def capture_metadata(self, operation: str, inputs: list, kwargs: dict) -> Dict:
        """
        Capture metadata for a LazyTensor operation.

        Called from LazyTensor.__torch_dispatch__ and __torch_function__.
        
        ✅ OPTIMIZATION P1.2: Lazy metadata capture
        - Only capture full metadata when needed (pattern matching, scheduling)
        - Fast path returns minimal metadata
        """
        # ✅ NEW P1.2 OPTIMIZATION: Fast path for common operations
        if self.lazy_metadata:
            # Fast path: return minimal metadata immediately
            cache_key = (operation, id(inputs[0]) if inputs else None)
            if cache_key in self._metadata_cache:
                return self._metadata_cache[cache_key]
            
            # Minimal metadata (very fast)
            metadata = {'operation': operation}
            
            # Only capture module context on first 10% of operations (sample-based)
            import random
            if random.random() < 0.1 or 'attention' in operation.lower() or 'conv' in operation.lower():
                module_path = self._get_module_context()
                if module_path:
                    metadata['module_path'] = module_path
                    metadata['module_type'] = self._get_module_type(module_path)
            
            # Cache this minimal metadata
            if len(self._metadata_cache) < 1000:  # Bound cache size
                self._metadata_cache[cache_key] = metadata
            
            return metadata
        
        # Original full metadata capture (slow but complete)
        metadata = {}

        # 1. Capture module context (if inside nn.Module)
        module_path = self._get_module_context()
        if module_path:
            metadata['module_path'] = module_path
            metadata['module_type'] = self._get_module_type(module_path)

        # 2. Infer semantic role from operation
        semantic_role = self._infer_semantic_role(operation, inputs)
        if semantic_role:
            metadata['semantic_role'] = semantic_role

        # 3. Track operation sequence for pattern hints
        self._operation_sequence.append(operation)
        if len(self._operation_sequence) > 10:
            self._operation_sequence.pop(0)

        # 4. Check for common patterns in recent operations
        pattern_hints = self._detect_pattern_hints()
        if pattern_hints:
            metadata['pattern_hints'] = pattern_hints

        # 5. Infer modality from tensor properties
        modality = self._infer_modality(inputs, kwargs)
        if modality:
            metadata['modality'] = modality

        return metadata

    def annotate_graph_batch(self, nodes: List, max_workers: Optional[int] = None) -> List[Dict]:
        """
        PHASE 2.2 OPTIMIZATION: Batch annotate multiple nodes in parallel.
        
        Args:
            nodes: List of nodes to annotate (each node has op, inputs, kwargs)
            max_workers: Number of threads (default: self.max_workers)
        
        Returns:
            List of metadata dicts for each node
        
        Usage:
            metadata_list = capture.annotate_graph_batch(graph_nodes, max_workers=4)
        """
        import time
        start_time = time.perf_counter()
        
        if not self.use_parallel or len(nodes) < 5:
            # For small batches, sequential is faster (no thread overhead)
            return [self._annotate_single_node(node) for node in nodes]
        
        # Parallel annotation for large batches
        max_workers = max_workers or self.max_workers
        results = [None] * len(nodes)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._annotate_single_node, node): i
                for i, node in enumerate(nodes)
            }
            
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Fallback to empty metadata on error
                    results[idx] = {}
        
        # Record statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _annotation_stats.record(len(nodes), elapsed_ms)
        
        return results
    
    def _annotate_single_node(self, node) -> Dict:
        """
        Annotate a single node (thread-safe, called in parallel).
        
        Each node is independent, so parallelization is safe.
        """
        # Extract operation info from node
        if isinstance(node, dict):
            op = node.get('operation', 'unknown')
            inputs = node.get('inputs', [])
            kwargs = node.get('kwargs', {})
        else:
            # LazyTensor or similar object
            op = getattr(node, 'operation', 'unknown')
            inputs = getattr(node, 'inputs', [])
            kwargs = getattr(node, 'kwargs', {})
        
        # Use existing capture logic
        return self.capture_metadata(op, inputs, kwargs)

    def _get_module_context(self) -> Optional[str]:
        """Extract current nn.Module path from call stack."""
        stack = inspect.stack()

        for frame_info in stack:
            frame_locals = frame_info.frame.f_locals

            # Look for 'self' that's an nn.Module
            if 'self' in frame_locals:
                obj = frame_locals['self']
                if isinstance(obj, nn.Module):
                    # Found the module - return its path
                    return self._get_module_path(obj)

        return None

    def _get_module_path(self, module: nn.Module) -> str:
        """Get hierarchical path of module (e.g., 'encoder.layer.0.attention')."""
        # Try to get from module's _modules dict if it's a submodule
        if hasattr(module, '_forward_hooks'):
            # This is a registered submodule
            for name, mod in module.named_modules():
                if mod is module:
                    return name

        # Fallback: use class name
        return module.__class__.__name__

    def _get_module_type(self, module_path: str) -> str:
        """Get module type from path."""
        if 'attention' in module_path.lower():
            return 'MultiHeadAttention'
        elif 'conv' in module_path.lower():
            return 'Convolution'
        elif 'linear' in module_path.lower():
            return 'Linear'
        return 'Unknown'

    def _infer_semantic_role(self, operation: str, inputs: list) -> Optional[str]:
        """Infer semantic role from operation."""
        op_lower = operation.lower()

        if 'matmul' in op_lower or 'bmm' in op_lower:
            # Could be attention or linear layer
            if len(inputs) >= 2:
                # Check shapes for attention pattern
                # Q@K.T has matching inner dims
                return 'matmul'  # Will be refined by pattern detection

        elif 'softmax' in op_lower:
            return 'attention_softmax'

        elif 'conv' in op_lower:
            return 'convolution'

        elif 'relu' in op_lower or 'gelu' in op_lower:
            return 'activation'

        return None

    def _detect_pattern_hints(self) -> Optional[Dict]:
        """Detect pattern hints from recent operation sequence."""
        recent_ops = self._operation_sequence[-5:]  # Last 5 ops

        # Attention pattern: matmul → softmax → matmul
        if len(recent_ops) >= 3:
            if ('matmul' in recent_ops[-3] and
                'softmax' in recent_ops[-2] and
                'matmul' in recent_ops[-1]):
                return {'likely_pattern': 'attention'}

        # Conv-BN-ReLU pattern
        if len(recent_ops) >= 3:
            if ('conv' in recent_ops[-3] and
                'batch_norm' in recent_ops[-2] and
                'relu' in recent_ops[-1]):
                return {'likely_pattern': 'conv_bn_relu'}

        return None

    def _infer_modality(self, inputs: list, kwargs: dict) -> Optional[str]:
        """Infer modality from tensor properties."""
        for inp in inputs:
            if hasattr(inp, 'shape') and len(inp.shape) >= 4:
                # 4D tensor likely image: [N, C, H, W]
                return 'vision'
            elif hasattr(inp, 'shape') and len(inp.shape) == 2:
                # 2D tensor likely text embeddings: [batch, seq_len]
                return 'text'

        return None


# Global metadata capture instance
_metadata_capture = MetadataCapture(use_parallel=True, max_workers=4)


def get_metadata_capture() -> MetadataCapture:
    """Get global metadata capture instance."""
    return _metadata_capture
