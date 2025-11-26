"""
Metadata registry for storing semantic annotations.

Provides thread-safe storage for all semantic metadata about graph nodes.
"""

import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class NodeMetadata:
    """Semantic metadata for a graph node."""
    node_id: str
    
    # Core metadata
    phase: Optional[str] = None  # ExecutionPhase
    modality: Optional[str] = None  # vision, text, audio, fusion
    semantic_role: Optional[str] = None  # attention, conv, linear, etc.
    
    # Cost estimates
    compute_flops: float = 0.0
    memory_bytes: float = 0.0
    operational_intensity: float = 0.0
    data_movement_bytes: float = 0.0
    
    # Optimization hints
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    co_location_required: bool = False
    co_location_targets: list = field(default_factory=list)
    
    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'phase': self.phase,
            'modality': self.modality,
            'semantic_role': self.semantic_role,
            'compute_flops': self.compute_flops,
            'memory_bytes': self.memory_bytes,
            'operational_intensity': self.operational_intensity,
            'data_movement_bytes': self.data_movement_bytes,
            'optimization_hints': self.optimization_hints,
            'co_location_required': self.co_location_required,
            'co_location_targets': self.co_location_targets,
            'extra': self.extra,
        }


class MetadataRegistry:
    """
    Thread-safe registry of semantic metadata for all nodes.
    """
    
    def __init__(self):
        self._metadata: Dict[str, NodeMetadata] = {}
        self._lock = threading.RLock()
    
    def register_metadata(self, node_id: str, metadata: NodeMetadata):
        """Register metadata for a node."""
        with self._lock:
            self._metadata[node_id] = metadata
    
    def get_metadata(self, node_id: str) -> Optional[NodeMetadata]:
        """Get metadata for a node."""
        with self._lock:
            return self._metadata.get(node_id)
    
    def update_metadata(self, node_id: str, updates: Dict[str, Any]):
        """Update metadata for a node."""
        with self._lock:
            if node_id in self._metadata:
                metadata = self._metadata[node_id]
                for key, value in updates.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)
    
    def get_all_metadata(self) -> Dict[str, NodeMetadata]:
        """Get all registered metadata."""
        with self._lock:
            return dict(self._metadata)
    
    def clear(self):
        """Clear all metadata."""
        with self._lock:
            self._metadata.clear()


# Global metadata registry
_global_metadata_registry = MetadataRegistry()


def get_metadata_registry() -> MetadataRegistry:
    """Get the global metadata registry."""
    return _global_metadata_registry

