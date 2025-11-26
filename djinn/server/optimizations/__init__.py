"""
Optimization Components for Djinn Server

This package contains optimization components for the Djinn execution system.

PRODUCTION COMPONENTS (actively used):
- block_compiler: Block compilation for graph-based execution (used in execute_model API)
- smart_subgraph_builder: Cost-aware subgraph fragmentation (used in executor)
- materialization_optimizer: Optimized materialization with CUDA streams (used in lazy_tensor)
- optimization_executor: Wraps SubgraphExecutor with optimizations (used in server)
- tensor_registry: Weight/KV cache caching (used via OptimizationExecutor)
- fusion_compiler: Pattern-based operation grouping (used via OptimizationExecutor)
- batch_compiler: Batch operation optimization (used in SimpleExecutor)
- tensorrt_compiler: TensorRT compilation (used via OptimizationExecutor)
- torchscript_synthesizer: TorchScript synthesis (used by TensorRT compiler)

EXPERIMENTAL COMPONENTS (see experimental/ subdirectory):
- adaptive_budget_tuner: Adaptive memory budget tuning (complete but not integrated)
- block_serializer: Block serialization for Phase 3 (not yet implemented)

PUBLIC API COMPONENTS (exported in djinn.semantic):
- phase_executor: Phase-aware execution strategies (exported for external use, has tests)

NOTE: These components are primarily used by the old graph-based execution system.
The new model cache system (MemoryAwareModelCache) has its own optimizations.
Both systems coexist in production with MigrationController managing gradual rollout.
"""

# Production components (actively used)
from .block_compiler import BlockCompiler, BlockExecutor, get_block_compiler
from .smart_subgraph_builder import SmartSubgraphBuilder, FragmentationConfig
from .materialization_optimizer import MaterializationOptimizer
from .optimization_executor import OptimizationExecutor, OptimizationStats
from .tensor_registry import SmartTensorRegistry, RemoteHandle
from .fusion_compiler import SRGFusionCompiler, FusionStrategy
from .batch_compiler import BatchCompiler, get_batch_compiler
from .tensorrt_compiler import TensorRTCompiler, get_tensorrt_compiler
from .torchscript_synthesizer import TorchScriptSynthesizer, get_synthesizer

__all__ = [
    # Block compilation
    'BlockCompiler',
    'BlockExecutor',
    'get_block_compiler',
    # Subgraph building
    'SmartSubgraphBuilder',
    'FragmentationConfig',
    # Materialization
    'MaterializationOptimizer',
    # Optimization executor
    'OptimizationExecutor',
    'OptimizationStats',
    # Tensor registry
    'SmartTensorRegistry',
    'RemoteHandle',
    # Fusion compiler
    'SRGFusionCompiler',
    'FusionStrategy',
    # Batch compiler
    'BatchCompiler',
    'get_batch_compiler',
    # TensorRT
    'TensorRTCompiler',
    'get_tensorrt_compiler',
    # TorchScript
    'TorchScriptSynthesizer',
    'get_synthesizer',
]
