"""Device placement logic for semantic-aware execution.

This module implements device placement strategies based on semantic metadata
and workload characteristics.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch

from ...core.types import ExecutionPhase, MemoryPattern

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of compute devices."""
    CPU = "cpu"
    GPU = "gpu"
    REMOTE_GPU = "remote_gpu"
    SPECIALIZED = "specialized"  # TPU, NPU, etc.


@dataclass
class DeviceCapabilities:
    """Capabilities of a compute device."""
    device_id: str
    device_type: DeviceType
    memory_gb: float
    compute_tflops: float
    bandwidth_gbps: float
    supports_fp16: bool = True
    supports_int8: bool = False
    is_available: bool = True
    current_memory_used: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class PlacementDecision:
    """Placement decision for a node or group."""
    node_name: str
    device_id: str
    reasoning: str
    priority: int = 0
    memory_required: float = 0.0
    compute_required: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class PlacementPlan:
    """Complete placement plan for a graph."""
    decisions: Dict[str, PlacementDecision] = field(default_factory=dict)
    device_assignments: Dict[str, List[str]] = field(default_factory=dict)  # device_id -> node_names
    colocation_groups: Dict[str, str] = field(default_factory=dict)  # group_id -> device_id
    total_devices_used: int = 0
    strategy: str = "balanced"
    metadata: Dict = field(default_factory=dict)


class PlacementEngine:
    """Engine for making device placement decisions."""
    
    def __init__(self, devices: Optional[List[DeviceCapabilities]] = None):
        """Initialize placement engine.
        
        Args:
            devices: List of available devices
        """
        self.devices = devices or self._detect_devices()
        self.device_map = {d.device_id: d for d in self.devices}
        self._placement_cache = {}
    
    def _detect_devices(self) -> List[DeviceCapabilities]:
        """Detect available devices.
        
        Returns:
            List of DeviceCapabilities
        """
        devices = []
        
        # Always have CPU
        devices.append(DeviceCapabilities(
            device_id="cpu:0",
            device_type=DeviceType.CPU,
            memory_gb=16.0,  # Placeholder
            compute_tflops=0.5,
            bandwidth_gbps=50.0
        ))
        
        # Check for GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(DeviceCapabilities(
                    device_id=f"cuda:{i}",
                    device_type=DeviceType.GPU,
                    memory_gb=props.total_memory / (1024**3),
                    compute_tflops=10.0,  # Placeholder
                    bandwidth_gbps=600.0,  # Placeholder
                    supports_fp16=True,
                    supports_int8=True
                ))
        
        # Add simulated remote GPUs for testing
        devices.append(DeviceCapabilities(
            device_id="remote_gpu:0",
            device_type=DeviceType.REMOTE_GPU,
            memory_gb=24.0,
            compute_tflops=15.0,
            bandwidth_gbps=100.0,  # Network bandwidth
            metadata={"latency_ms": 5}
        ))
        
        return devices
    
    def create_placement_plan(self, graph, optimization_plan=None, 
                            schedule=None) -> PlacementPlan:
        """Create placement plan for a graph.
        
        Args:
            graph: FX GraphModule
            optimization_plan: Optional optimization plan
            schedule: Optional execution schedule
            
        Returns:
            PlacementPlan with device assignments
        """
        plan = PlacementPlan()
        
        # Extract placement hints from optimization plan
        hints = self._extract_placement_hints(optimization_plan)
        
        # Make placement decisions for each node
        for node in graph.graph.nodes:
            if node.op in ['call_function', 'call_method', 'call_module']:
                decision = self._make_placement_decision(node, hints, plan)
                plan.decisions[node.name] = decision
                
                # Update device assignments
                if decision.device_id not in plan.device_assignments:
                    plan.device_assignments[decision.device_id] = []
                plan.device_assignments[decision.device_id].append(node.name)
        
        # Handle colocation groups
        if optimization_plan and hasattr(optimization_plan, 'colocation_groups'):
            for group_id, node_names in optimization_plan.colocation_groups.items():
                # Place all nodes in group on same device
                device_id = self._select_device_for_group(node_names, plan)
                plan.colocation_groups[group_id] = device_id
                
                # Update decisions
                for node_name in node_names:
                    if node_name in plan.decisions:
                        plan.decisions[node_name].device_id = device_id
                        plan.decisions[node_name].reasoning = f"Co-located with group {group_id}"
        
        # Update statistics
        plan.total_devices_used = len(plan.device_assignments)
        plan.metadata['total_nodes'] = len(plan.decisions)
        
        logger.info(f"Created placement plan using {plan.total_devices_used} devices")
        
        return plan
    
    def _extract_placement_hints(self, optimization_plan) -> Dict:
        """Extract placement hints from optimization plan.
        
        Args:
            optimization_plan: Optimization plan
            
        Returns:
            Dictionary of placement hints
        """
        hints = {}
        
        if not optimization_plan:
            return hints
        
        # Extract node placement hints
        if hasattr(optimization_plan, 'node_placement'):
            hints['node_placement'] = optimization_plan.node_placement
        
        # Extract priority nodes
        if hasattr(optimization_plan, 'metadata'):
            hints['high_priority'] = optimization_plan.metadata.get('high_priority_ops', [])
        
        return hints
    
    def _make_placement_decision(self, node, hints: Dict, 
                                current_plan: PlacementPlan) -> PlacementDecision:
        """Make placement decision for a single node.
        
        Args:
            node: FX Node
            hints: Placement hints
            current_plan: Current placement plan
            
        Returns:
            PlacementDecision for the node
        """
        # Check for explicit placement hint
        if 'placement_hint' in node.meta:
            device_id = self._resolve_device_hint(node.meta['placement_hint'])
            return PlacementDecision(
                node_name=node.name,
                device_id=device_id,
                reasoning="Explicit placement hint",
                priority=node.meta.get('priority', 0)
            )
        
        # Check semantic metadata
        if 'semantic' in node.meta:
            return self._semantic_placement(node, hints)
        
        # Default placement based on operation type
        return self._default_placement(node)
    
    def _semantic_placement(self, node, hints: Dict) -> PlacementDecision:
        """Make placement based on semantic metadata.
        
        Args:
            node: FX Node with semantic metadata
            hints: Placement hints
            
        Returns:
            PlacementDecision
        """
        metadata = node.meta['semantic']
        
        # Check execution phase
        if hasattr(metadata, 'execution_phase'):
            phase = metadata.execution_phase
            
            # LLM decode phase - place near KV cache
            if phase == ExecutionPhase.DECODE:
                device_id = self._find_kv_cache_device() or "cuda:0"
                return PlacementDecision(
                    node_name=node.name,
                    device_id=device_id,
                    reasoning="LLM decode phase - co-locate with KV cache",
                    priority=8
                )
            
            # Vision backbone - prefer GPU
            elif phase == ExecutionPhase.VISION_BACKBONE:
                device_id = self._select_best_gpu()
                return PlacementDecision(
                    node_name=node.name,
                    device_id=device_id,
                    reasoning="Vision backbone - GPU optimized",
                    priority=6
                )
            
            # Multi-modal fusion - place at fusion point
            elif phase == ExecutionPhase.MULTIMODAL_FUSION:
                device_id = self._select_fusion_device()
                return PlacementDecision(
                    node_name=node.name,
                    device_id=device_id,
                    reasoning="Multi-modal fusion point",
                    priority=7
                )
        
        # Check memory pattern
        if hasattr(metadata, 'memory_pattern'):
            pattern = metadata.memory_pattern
            
            # Persistent memory - avoid remote devices
            if pattern == MemoryPattern.PERSISTENT:
                device_id = self._select_local_device()
                return PlacementDecision(
                    node_name=node.name,
                    device_id=device_id,
                    reasoning="Persistent memory pattern - local device",
                    priority=5
                )
            
            # Streaming - can use remote
            elif pattern == MemoryPattern.STREAMING:
                device_id = self._select_any_gpu()
                return PlacementDecision(
                    node_name=node.name,
                    device_id=device_id,
                    reasoning="Streaming pattern - any GPU",
                    priority=3
                )
        
        # Check compute intensity
        if hasattr(metadata, 'compute_intensity'):
            if metadata.compute_intensity > 8.0:
                # High compute - prefer powerful GPU
                device_id = self._select_best_gpu()
                return PlacementDecision(
                    node_name=node.name,
                    device_id=device_id,
                    reasoning=f"High compute intensity ({metadata.compute_intensity})",
                    priority=6
                )
        
        # Default GPU placement
        return self._default_placement(node)
    
    def _default_placement(self, node) -> PlacementDecision:
        """Default placement strategy.
        
        Args:
            node: FX Node
            
        Returns:
            PlacementDecision
        """
        # Check operation type
        op_name = str(node.target).lower() if node.op == 'call_function' else ""
        
        # Compute-intensive ops go to GPU
        if any(op in op_name for op in ['conv', 'matmul', 'linear', 'attention']):
            device_id = self._select_any_gpu()
            reasoning = "Compute-intensive operation"
        # Memory ops can stay on CPU
        elif any(op in op_name for op in ['reshape', 'view', 'transpose', 'cat']):
            device_id = "cpu:0"
            reasoning = "Memory operation - CPU sufficient"
        else:
            # Default to least loaded device
            device_id = self._select_least_loaded_device()
            reasoning = "Load balancing"
        
        return PlacementDecision(
            node_name=node.name,
            device_id=device_id,
            reasoning=reasoning,
            priority=0
        )
    
    def _resolve_device_hint(self, hint: str) -> str:
        """Resolve a device hint to actual device ID.
        
        Args:
            hint: Device hint string
            
        Returns:
            Actual device ID
        """
        if hint == "kv_cache_device":
            return self._find_kv_cache_device() or "cuda:0"
        elif hint == "fusion_device":
            return self._select_fusion_device()
        elif hint in self.device_map:
            return hint
        else:
            # Try to match device type
            for device_id, device in self.device_map.items():
                if hint.lower() in device_id.lower():
                    return device_id
            
            # Default
            return "cuda:0" if torch.cuda.is_available() else "cpu:0"
    
    def _find_kv_cache_device(self) -> Optional[str]:
        """Find device with KV cache.
        
        Returns:
            Device ID or None
        """
        # For now, assume first GPU has KV cache
        for device_id, device in self.device_map.items():
            if device.device_type == DeviceType.GPU:
                return device_id
        return None
    
    def _select_best_gpu(self) -> str:
        """Select best available GPU.
        
        Returns:
            Device ID
        """
        best_device = None
        best_score = -1
        
        for device_id, device in self.device_map.items():
            if device.device_type in [DeviceType.GPU, DeviceType.REMOTE_GPU]:
                if device.is_available:
                    # Score based on compute and memory
                    score = device.compute_tflops + (device.memory_gb / 10)
                    # Penalize remote devices
                    if device.device_type == DeviceType.REMOTE_GPU:
                        score *= 0.8
                    
                    if score > best_score:
                        best_score = score
                        best_device = device_id
        
        return best_device or "cpu:0"
    
    def _select_any_gpu(self) -> str:
        """Select any available GPU.
        
        Returns:
            Device ID
        """
        # Prefer local GPUs
        for device_id, device in self.device_map.items():
            if device.device_type == DeviceType.GPU and device.is_available:
                return device_id
        
        # Then remote GPUs
        for device_id, device in self.device_map.items():
            if device.device_type == DeviceType.REMOTE_GPU and device.is_available:
                return device_id
        
        return "cpu:0"
    
    def _select_local_device(self) -> str:
        """Select a local (non-remote) device.
        
        Returns:
            Device ID
        """
        # Prefer local GPU
        for device_id, device in self.device_map.items():
            if device.device_type == DeviceType.GPU and device.is_available:
                return device_id
        
        return "cpu:0"
    
    def _select_fusion_device(self) -> str:
        """Select device for multi-modal fusion.
        
        Returns:
            Device ID
        """
        # Fusion typically needs good memory bandwidth
        best_device = None
        best_bandwidth = -1
        
        for device_id, device in self.device_map.items():
            if device.device_type in [DeviceType.GPU, DeviceType.REMOTE_GPU]:
                if device.is_available and device.bandwidth_gbps > best_bandwidth:
                    best_bandwidth = device.bandwidth_gbps
                    best_device = device_id
        
        return best_device or "cuda:0"
    
    def _select_least_loaded_device(self) -> str:
        """Select least loaded device.
        
        Returns:
            Device ID
        """
        best_device = None
        min_usage = float('inf')
        
        for device_id, device in self.device_map.items():
            if device.is_available:
                usage = device.current_memory_used / max(device.memory_gb, 1.0)
                if usage < min_usage:
                    min_usage = usage
                    best_device = device_id
        
        return best_device or "cpu:0"
    
    def _select_device_for_group(self, node_names: List[str], 
                                plan: PlacementPlan) -> str:
        """Select device for a colocation group.
        
        Args:
            node_names: Nodes in the group
            plan: Current placement plan
            
        Returns:
            Device ID
        """
        # Check if any nodes already placed
        for node_name in node_names:
            if node_name in plan.decisions:
                return plan.decisions[node_name].device_id
        
        # Otherwise select best GPU
        return self._select_best_gpu()
    
    def update_device_usage(self, device_id: str, memory_delta: float):
        """Update device memory usage.
        
        Args:
            device_id: Device ID
            memory_delta: Change in memory usage (GB)
        """
        if device_id in self.device_map:
            device = self.device_map[device_id]
            device.current_memory_used += memory_delta
            
            # Mark as unavailable if out of memory
            if device.current_memory_used >= device.memory_gb * 0.95:
                device.is_available = False
                logger.warning(f"Device {device_id} marked unavailable due to memory pressure")
