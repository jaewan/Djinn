"""
djinn/testing/performance_profiler.py

Framework for profiling graph execution performance.

This gates shadow sync deployment: we must measure actual
graph execution performance before enabling auto-registration.
"""

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import logging
import pickle
import io

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of profiling one model."""
    model_id: str
    model_category: str  # 'small_hf', 'large_hf', 'vision', 'custom'
    
    # Timings (milliseconds)
    capture_ms: float
    serialize_ms: float
    transfer_ms: float
    deserialize_ms: float
    execute_ms: float
    total_ms: float
    
    # Sizes
    model_size_mb: float
    graph_size_kb: float
    input_size_kb: float
    
    # Metadata
    num_operations: int
    num_parameters: int
    
    def meets_target(self, target_ms: float) -> bool:
        """Check if total time meets target."""
        return self.total_ms < target_ms
    
    def identify_bottleneck(self) -> Tuple[str, float]:
        """Identify the slowest component."""
        components = {
            'capture': self.capture_ms,
            'serialize': self.serialize_ms,
            'transfer': self.transfer_ms,
            'deserialize': self.deserialize_ms,
            'execute': self.execute_ms,
        }
        bottleneck = max(components.items(), key=lambda x: x[1])
        return bottleneck


def simulate_network_transfer(data: bytes, latency_ms: float = 1.0, bandwidth_mbps: float = 1000.0) -> bytes:
    """
    Simulate network transfer with latency and bandwidth constraints.
    
    Args:
        data: Data to transfer
        latency_ms: Network latency in milliseconds
        bandwidth_mbps: Bandwidth in megabits per second
    
    Returns:
        Same data (simulated transfer)
    """
    # Simulate latency
    time.sleep(latency_ms / 1000.0)
    
    # Simulate bandwidth (data size / bandwidth)
    data_size_mb = len(data) * 8 / (1024 * 1024)  # Convert bytes to megabits
    transfer_time = data_size_mb / bandwidth_mbps  # Seconds
    time.sleep(transfer_time)
    
    return data


def serialize_graph(graph) -> bytes:
    """
    Serialize graph to binary format.
    
    For now, uses pickle. In production, this would use
    the optimized binary protocol.
    """
    buffer = io.BytesIO()
    pickle.dump(graph, buffer)
    return buffer.getvalue()


def deserialize_graph(data: bytes):
    """
    Deserialize graph from binary format.
    """
    buffer = io.BytesIO(data)
    return pickle.load(buffer)


def execute_graph(graph, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Execute graph with given inputs.
    
    Uses SimpleExecutor for local execution.
    """
    from djinn.server.executor import SimpleExecutor
    
    executor = SimpleExecutor()
    
    # For LazyTensor graphs, we need to materialize the output
    # The graph should be a LazyDAGAdapter with a root LazyTensor
    if hasattr(graph, 'root_tensor'):
        # Execute the graph by materializing the root tensor
        result = executor.execute_subgraph(graph.root_tensor)
        return result
    else:
        # Fallback: try to execute directly
        raise NotImplementedError("Graph execution not implemented for this graph type")


class PerformanceGate:
    """
    Performance validation gate for shadow sync deployment.
    
    IMPORTANT: These thresholds are GUIDELINES, not hard requirements.
    The purpose is to:
    1. Identify bottlenecks (which component is slow?)
    2. Guide optimization priorities
    3. Inform shadow sync threshold (N) selection
    
    If targets are not met:
    - Document which components need optimization
    - Adjust shadow sync threshold N (e.g., N=5 or N=10 instead of N=3)
    - Proceed with implementation but with adjusted expectations
    
    Target guidelines (milliseconds):
    - Small models (<1B params): <500ms total
    - Large models (>1B params): <1500ms total
    - Deserialization: <100ms for all models (bottleneck risk if exceeded)
    
    Network simulation note:
    - simulate_network_transfer() must reflect actual deployment environment
    - Use realistic latency/bandwidth for your network (LAN vs WAN)
    - Consider compression if enabled in production
    """
    
    # Target latencies (milliseconds) - GUIDELINES, not hard requirements
    TARGETS = {
        'small_hf': 500,
        'large_hf': 1500,
        'vision': 400,
        'custom': 800,
    }
    
    # Component-specific targets - GUIDELINES
    MAX_DESERIALIZE_MS = 100
    
    def __init__(self, network_latency_ms: float = 1.0, network_bandwidth_mbps: float = 1000.0):
        """
        Initialize performance gate.
        
        Args:
            network_latency_ms: Simulated network latency in milliseconds
            network_bandwidth_mbps: Simulated network bandwidth in megabits per second
        """
        self.results: List[ProfileResult] = []
        self.network_latency_ms = network_latency_ms
        self.network_bandwidth_mbps = network_bandwidth_mbps
    
    def profile_model(
        self,
        model: nn.Module,
        model_id: str,
        category: str,
        sample_inputs: Dict[str, torch.Tensor],
    ) -> ProfileResult:
        """
        Profile graph execution for one model.
        
        Measures all components of graph execution:
        1. Capture (LazyTensor DAG construction)
        2. Serialize (graph → binary)
        3. Transfer (network simulation)
        4. Deserialize (binary → graph)
        5. Execute (actual computation)
        """
        import djinn
        from djinn.frontend.core.capture import capture
        from djinn.frontend.core.graph_builder import get_global_builder
        
        # Warm up
        with torch.no_grad():
            _ = model(**sample_inputs)
        
        # Reset graph builder
        builder = get_global_builder()
        builder.root_tensor = None
        
        # 1. Capture
        start = time.perf_counter()
        with capture():
            output = model(**sample_inputs)
        capture_ms = (time.perf_counter() - start) * 1000
        
        # Get graph
        builder = get_global_builder()
        if builder.root_tensor is None:
            raise RuntimeError("Failed to capture graph. Make sure model operations are captured.")
        
        graph = builder.build_from_capture()
        
        # Count operations (approximate - count LazyTensor nodes)
        num_operations = self._count_operations(builder.root_tensor)
        
        # 2. Serialize
        start = time.perf_counter()
        serialized = serialize_graph(graph)
        serialize_ms = (time.perf_counter() - start) * 1000
        graph_size_kb = len(serialized) / 1024
        
        # 3. Transfer (simulate network)
        start = time.perf_counter()
        transferred = simulate_network_transfer(
            serialized,
            latency_ms=self.network_latency_ms,
            bandwidth_mbps=self.network_bandwidth_mbps
        )
        transfer_ms = (time.perf_counter() - start) * 1000
        
        # 4. Deserialize
        start = time.perf_counter()
        deserialized_graph = deserialize_graph(transferred)
        deserialize_ms = (time.perf_counter() - start) * 1000
        
        # 5. Execute
        start = time.perf_counter()
        with torch.no_grad():
            result = execute_graph(deserialized_graph, sample_inputs)
        execute_ms = (time.perf_counter() - start) * 1000
        
        total_ms = capture_ms + serialize_ms + transfer_ms + deserialize_ms + execute_ms
        
        # Metadata
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        num_parameters = sum(p.numel() for p in model.parameters())
        
        input_size_kb = sum(
            t.numel() * t.element_size() for t in sample_inputs.values()
        ) / 1024
        
        result = ProfileResult(
            model_id=model_id,
            model_category=category,
            capture_ms=capture_ms,
            serialize_ms=serialize_ms,
            transfer_ms=transfer_ms,
            deserialize_ms=deserialize_ms,
            execute_ms=execute_ms,
            total_ms=total_ms,
            model_size_mb=model_size_mb,
            graph_size_kb=graph_size_kb,
            input_size_kb=input_size_kb,
            num_operations=num_operations,
            num_parameters=num_parameters,
        )
        
        self.results.append(result)
        return result
    
    def _count_operations(self, root_tensor) -> int:
        """
        Count operations in LazyTensor DAG (approximate).
        
        Traverses the DAG starting from root_tensor.
        """
        visited = set()
        count = 0
        
        def traverse(tensor):
            nonlocal count
            if tensor is None:
                return
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            
            # Count this operation
            if hasattr(tensor, '_operation') and tensor._operation:
                count += 1
            
            # Traverse inputs
            if hasattr(tensor, '_inputs'):
                for inp in tensor._inputs:
                    if isinstance(inp, torch.Tensor):
                        traverse(inp)
        
        traverse(root_tensor)
        return count
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate all profiling results against target guidelines.
        
        NOTE: "Failures" are optimization opportunities, not blockers.
        The gate helps identify:
        1. Which components need optimization
        2. Whether shadow sync threshold N should be adjusted
        3. Expected performance characteristics
        
        Returns:
            (passed, report)
            - passed: True if all targets met (guideline)
            - report: Detailed breakdown with bottlenecks identified
        """
        if not self.results:
            return (False, "No profiling results")
        
        target_misses = []  # Renamed from "failures" to clarify these are guidelines
        warnings = []
        
        for result in self.results:
            target = self.TARGETS.get(result.model_category, 1000)
            
            # Check total latency (guideline)
            if not result.meets_target(target):
                target_misses.append(
                    f"{result.model_id}: {result.total_ms:.0f}ms > {target}ms guideline "
                    f"(consider optimizing or increasing shadow sync threshold N)"
                )
            
            # Check deserialization specifically (bottleneck risk)
            if result.deserialize_ms > self.MAX_DESERIALIZE_MS:
                warnings.append(
                    f"{result.model_id}: Deserialize {result.deserialize_ms:.0f}ms "
                    f"> {self.MAX_DESERIALIZE_MS}ms guideline (bottleneck risk)"
                )
            
            # Identify bottlenecks
            bottleneck_name, bottleneck_ms = result.identify_bottleneck()
            if bottleneck_ms > 0.5 * result.total_ms:
                warnings.append(
                    f"{result.model_id}: {bottleneck_name} is bottleneck "
                    f"({bottleneck_ms:.0f}ms = {bottleneck_ms/result.total_ms*100:.0f}%)"
                )
        
        # Generate report
        report_lines = ["Performance Gate Validation Report", "=" * 50]
        report_lines.append("NOTE: Targets are GUIDELINES for optimization, not blockers")
        report_lines.append("")
        
        for result in self.results:
            target = self.TARGETS.get(result.model_category, 1000)
            passed = "✓" if result.meets_target(target) else "⚠"
            
            report_lines.append(f"\n{passed} {result.model_id} ({result.model_category})")
            report_lines.append(f"  Total: {result.total_ms:.0f}ms (guideline: {target}ms)")
            report_lines.append(f"  Breakdown:")
            report_lines.append(f"    Capture:     {result.capture_ms:6.1f}ms")
            report_lines.append(f"    Serialize:   {result.serialize_ms:6.1f}ms")
            report_lines.append(f"    Transfer:    {result.transfer_ms:6.1f}ms")
            report_lines.append(f"    Deserialize: {result.deserialize_ms:6.1f}ms")
            report_lines.append(f"    Execute:     {result.execute_ms:6.1f}ms")
            
            bottleneck = result.identify_bottleneck()
            report_lines.append(f"  Bottleneck: {bottleneck[0]} ({bottleneck[1]:.0f}ms)")
        
        if target_misses:
            report_lines.append("\n" + "=" * 50)
            report_lines.append("TARGET GUIDELINES NOT MET (optimization opportunities):")
            for miss in target_misses:
                report_lines.append(f"  ⚠ {miss}")
            report_lines.append("\n  RECOMMENDATION:")
            report_lines.append("    - Optimize identified bottlenecks, OR")
            report_lines.append("    - Increase shadow sync threshold N (e.g., N=5 or N=10)")
            report_lines.append("    - Document expected performance characteristics")
        
        if warnings:
            report_lines.append("\n" + "=" * 50)
            report_lines.append("WARNINGS (bottleneck risks):")
            for warning in warnings:
                report_lines.append(f"  ⚠ {warning}")
        
        report = "\n".join(report_lines)
        passed = len(target_misses) == 0
        
        return (passed, report)
    
    def save_results(self, filepath: str):
        """Save profiling results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(
                [asdict(r) for r in self.results],
                f,
                indent=2
            )

