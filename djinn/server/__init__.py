"""
Server components for Djinn disaggregated GPU cluster.

Core Components:
    - DjinnServer: Main server implementation
    - TCP Server: TCP-based server for remote execution
    - Model Cache: Memory-aware model caching system (redesigned)
    - Architecture Registry: Model architecture reconstruction
    - Profiling: Performance profiling infrastructure

Usage:
    # Start server programmatically
    from djinn.server import DjinnServer, ServerConfig
    config = ServerConfig()
    server = DjinnServer(config)
    await server.start()
    
    # Or use TCP server directly
    from djinn.server.tcp_server import start_server
    await start_server(host="0.0.0.0", port=5556)
"""

# ============================================================================
# Core Server Components
# ============================================================================
from .server import DjinnServer, ServerConfig
from .capability_provider import CapabilityProvider

# ============================================================================
# Model Cache System (Redesigned - Production)
# ============================================================================
from .memory_aware_model_cache import MemoryAwareModelCache
from .resilient_model_handler import ResilientModelHandler
from .architecture_registry import HybridArchitectureRegistry
from .model_security import ModelSecurityValidator

# ============================================================================
# Profiling Infrastructure
# ============================================================================
from .profiling_context import ProfilingContext, record_phase, get_profiler

# ============================================================================
# Memory Management
# ============================================================================
from .semantic_memory_manager import (
    PhaseAwareMemoryManager,
    ExecutionPhase,
    EvictionPriority
)
from .memory_pressure_handler import MemoryPressureHandler
from .memory_metrics import MetricsCollector, get_metrics

# ============================================================================
# Caching Systems
# ============================================================================
from .gpu_cache import SimpleGPUCache, get_global_cache
from .graph_cache import GraphCache, get_graph_cache
# Alias for backward compatibility with tcp_server.py
get_global_graph_cache = get_graph_cache
from .subgraph_cache import SubgraphCache, get_subgraph_cache

# ============================================================================
# Subgraph Execution (Graph-based execution path)
# ============================================================================
from .subgraph_builder import SubgraphBuilder, RemoteSubgraph
from .subgraph_executor import SubgraphExecutor
from .batch_executor import BatchExecutor, get_batch_executor

# ============================================================================
# Serialization
# ============================================================================
from .serialization import serialize_tensor, deserialize_tensor, deserialize_tensor_from_dict

# ============================================================================
# Utilities
# ============================================================================
from .device_mapper import DeviceMapper
from .error_responses import ErrorResponseBuilder, ErrorCode
from .type_utils import is_tensor, is_tensor_fast, TensorTypeCache
from .server_state import ServerState
from .handler_registry import MessageHandlerRegistry, MessageType
from .exceptions import (
    DjinnServerError,
    ModelNotFoundError,
    ExecutionError,
    OutOfMemoryError,
    TimeoutError,
    SecurityError,
    InvalidRequestError
)

# ============================================================================
# Performance Monitoring
# ============================================================================
from .performance_monitor import PerformanceMonitor

__all__ = [
    # Core server
    'DjinnServer',
    'ServerConfig',
    'CapabilityProvider',
    
    # Model cache system (production)
    'MemoryAwareModelCache',
    'ResilientModelHandler',
    'HybridArchitectureRegistry',
    'ModelSecurityValidator',
    
    # Profiling
    'ProfilingContext',
    'record_phase',
    'get_profiler',
    
    # Memory management
    'PhaseAwareMemoryManager',
    'ExecutionPhase',
    'EvictionPriority',
    'MemoryPressureHandler',
    'MetricsCollector',
    'get_metrics',
    
    # Caching
    'SimpleGPUCache',
    'get_global_cache',
    'GraphCache',
    'get_graph_cache',
    'get_global_graph_cache',  # Alias for backward compatibility
    'SubgraphCache',
    'get_subgraph_cache',
    
    # Subgraph execution
    'SubgraphBuilder',
    'RemoteSubgraph',
    'SubgraphExecutor',
    'BatchExecutor',
    'get_batch_executor',
    
    # Serialization
    'serialize_tensor',
    'deserialize_tensor',
    'deserialize_tensor_from_dict',
    
    # Utilities
    'DeviceMapper',
    'ErrorResponseBuilder',
    'ErrorCode',
    'is_tensor',
    'is_tensor_fast',
    'TensorTypeCache',
    
    # Architecture
    'ServerState',
    'MessageHandlerRegistry',
    'MessageType',
    'DjinnServerError',
    'ModelNotFoundError',
    'ExecutionError',
    'OutOfMemoryError',
    'TimeoutError',
    'SecurityError',
    'InvalidRequestError',
    
    # Performance monitoring
    'PerformanceMonitor',
]
