"""
Capability Interlock: Client-side safety system for fallback behavior

Implements the v2.3 architecture feature: "Capability Interlock"

Problem:
- Ghost model has NO weights (on meta device)
- If remote execution fails, local fallback requires emergency materialization
- But client may not have enough RAM to materialize large models
- Need intelligent fallback logic

Solution:
- Before attempting fallback, audit local resources
- Check if model can fit in available RAM
- Estimate materialization overhead (1.5x model size)
- Gracefully reject fallback if resources insufficient
- Provide clear diagnostics to user

Capability States:
- REMOTE_AVAILABLE (fast path): Use remote execution
- FALLBACK_CAPABLE (slow path): Can materialize locally
- FALLBACK_BLOCKED (error path): Cannot materialize, must fail

Example:
    try:
        # Fast path: Remote execution
        result = engine.execute(ghost_model, inputs)
    except RemoteBusyError:
        # Try fallback
        result = engine.execute_with_fallback(ghost_model, inputs)
        # Will fail with ResourceError if not enough RAM
"""

import logging
import psutil
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CapabilityState(Enum):
    """States of execution capability."""
    REMOTE_AVAILABLE = "remote_available"  # Remote available, use it
    FALLBACK_CAPABLE = "fallback_capable"  # Can materialize locally
    FALLBACK_BLOCKED = "fallback_blocked"  # Cannot materialize
    UNKNOWN = "unknown"                     # Not yet determined


@dataclass
class ResourceAudit:
    """Results of resource audit for fallback decision."""
    total_system_ram_gb: float
    available_ram_gb: float
    model_size_gb: float
    estimated_overhead_gb: float
    total_needed_gb: float
    can_materialize: bool
    safety_margin_gb: float = 2.0  # Keep 2GB free for OS
    
    def __str__(self) -> str:
        """Detailed resource report."""
        return (
            f"Resource Audit:\n"
            f"  Total RAM: {self.total_system_ram_gb:.1f} GB\n"
            f"  Available: {self.available_ram_gb:.1f} GB\n"
            f"  Model size: {self.model_size_gb:.1f} GB\n"
            f"  Estimated overhead (1.5x): {self.estimated_overhead_gb:.1f} GB\n"
            f"  Total needed: {self.total_needed_gb:.1f} GB\n"
            f"  Safety margin: {self.safety_margin_gb:.1f} GB\n"
            f"  Can materialize: {self.can_materialize}"
        )


class CapabilityEngine:
    """
    Client-side execution capability checker.
    
    Decides whether to attempt remote execution, local fallback, or fail gracefully.
    """
    
    def __init__(self, safety_margin_gb: float = 2.0, overhead_multiplier: float = 1.5):
        """
        Initialize capability engine.
        
        Args:
            safety_margin_gb: Minimum free RAM to keep (default: 2GB for OS)
            overhead_multiplier: Materialization overhead factor (default: 1.5x)
        """
        self.safety_margin_gb = safety_margin_gb
        self.overhead_multiplier = overhead_multiplier
        self.state = CapabilityState.UNKNOWN
    
    def estimate_model_size(self, model: nn.Module) -> float:
        """
        Estimate model size in GB.
        
        Args:
            model: PyTorch model
        
        Returns:
            Estimated size in GB
        """
        total_params = sum(p.numel() for p in model.parameters())
        # Assume mixed precision (16 or 32 bit)
        # Average 24 bits (3 bytes) per parameter
        total_bytes = total_params * 4  # Use 32-bit estimate (conservative)
        total_gb = total_bytes / (1024**3)
        return total_gb
    
    def audit_resources(self, model: nn.Module) -> ResourceAudit:
        """
        Audit system resources for fallback capability.
        
        Args:
            model: Model to materialize
        
        Returns:
            ResourceAudit with detailed resource information
        """
        # Get system memory info
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024**3)
        available_ram_gb = memory.available / (1024**3)
        
        # Estimate model size
        model_size_gb = self.estimate_model_size(model)
        
        # Estimate overhead (materialize causes ~1.5x memory spike)
        overhead_gb = model_size_gb * (self.overhead_multiplier - 1.0)
        total_needed_gb = model_size_gb + overhead_gb
        
        # Account for safety margin
        available_for_model_gb = available_ram_gb - self.safety_margin_gb
        
        # Determine if we can materialize
        can_materialize = (available_for_model_gb >= total_needed_gb)
        
        audit = ResourceAudit(
            total_system_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            model_size_gb=model_size_gb,
            estimated_overhead_gb=overhead_gb,
            total_needed_gb=total_needed_gb,
            can_materialize=can_materialize,
            safety_margin_gb=self.safety_margin_gb
        )
        
        logger.info(f"\n{audit}")
        
        # Update state
        if can_materialize:
            self.state = CapabilityState.FALLBACK_CAPABLE
        else:
            self.state = CapabilityState.FALLBACK_BLOCKED
        
        return audit

    def ensure_safe_fallback(self, model_size_bytes: int) -> None:
        """
        Simple v2.3.15 capability interlock: prevent local crashes during fallback.

        Requires 1.5x headroom to prevent swap thrashing.

        Args:
            model_size_bytes: Estimated model size in bytes

        Raises:
            ResourceError: If fallback would cause resource exhaustion
        """
        import psutil

        # Require 1.5x headroom (matches v2.3.15 spec)
        safety_margin = model_size_bytes * 1.5
        available = psutil.virtual_memory().available

        if available < safety_margin:
            raise ResourceError(
                f"Safety Interlock: Fallback denied. "
                f"Req {safety_margin/1e9:.2f}GB > Avail {available/1e9:.2f}GB"
            )
    
    def can_execute_remotely(self, coordinator) -> bool:
        """
        Check if remote execution is available.
        
        Args:
            coordinator: Djinn coordinator instance
        
        Returns:
            True if remote server is available and responsive
        """
        if coordinator is None:
            return False
        
        try:
            # Quick health check to server
            # (implement based on coordinator interface)
            return hasattr(coordinator, 'execute_remote_subgraph')
        except Exception as e:
            logger.warning(f"Remote health check failed: {e}")
            return False
    
    def execute(self, 
                ghost_model: nn.Module,
                inputs: Dict[str, Any],
                coordinator=None) -> torch.Tensor:
        """
        Execute with intelligent fallback logic.
        
        Strategy:
        1. Try remote execution (fast path)
        2. On failure, check if fallback is possible
        3. If possible, materialize and execute locally
        4. If not possible, fail with clear error
        
        Args:
            ghost_model: Ghost model (on meta device)
            inputs: Input tensors/data
            coordinator: Djinn coordinator
        
        Returns:
            Output tensor
        
        Raises:
            RuntimeError: If execution fails and fallback not possible
        """
        logger.info("🚀 Capability-aware execution starting")
        
        # Primary: Try remote execution
        if self.can_execute_remotely(coordinator):
            try:
                logger.info("✅ Remote execution available (fast path)")
                self.state = CapabilityState.REMOTE_AVAILABLE
                
                # Execute remotely
                from ...server.optimizations.smart_subgraph_builder import SmartSubgraphBuilder
                result = coordinator.execute_remote_subgraph(
                    ghost_model, inputs
                )
                return result
            
            except Exception as e:
                logger.warning(f"Remote execution failed: {e}")
                # Fall through to fallback logic
        
        # Secondary: Check if fallback is possible
        logger.info("📊 Checking fallback capability...")
        audit = self.audit_resources(ghost_model)
        
        if not audit.can_materialize:
            # Cannot materialize - fail gracefully
            raise RuntimeError(
                f"❌ Fallback not possible: Insufficient RAM\n\n{audit}\n\n"
                f"Solution: Use a server with more GPU memory, or request only needed outputs"
            )
        
        logger.info("✅ Local fallback possible (slow path)")
        
        # Materialize model locally
        try:
            logger.info(f"📥 Materializing model ({audit.model_size_gb:.1f}GB)...")
            
            # Disable Djinn interception during materialization
            from .interception_control import disable_interception, InterceptionContext
            
            with disable_interception(InterceptionContext.MATERIALIZATION):
                # Load actual model from HuggingFace or cache
                from transformers import AutoModel
                real_model = AutoModel.from_pretrained(
                    ghost_model.config.model_type,
                    torch_dtype=torch.float32
                )
                real_model = real_model.eval()
            
            logger.info("✅ Model materialized")
            
            # Execute locally
            logger.info("⚙️  Executing locally...")
            with torch.no_grad():
                output = real_model(**inputs)
            
            return output.logits if hasattr(output, 'logits') else output
        
        except Exception as e:
            logger.error(f"❌ Fallback execution failed: {e}")
            raise RuntimeError(
                f"Fallback execution failed after remote unavailability: {e}"
            )
    
    def get_capability_state(self) -> CapabilityState:
        """Get current capability state."""
        return self.state
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed capability diagnostics."""
        memory = psutil.virtual_memory()
        return {
            'state': self.state.value,
            'total_ram_gb': memory.total / (1024**3),
            'available_ram_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'safety_margin_gb': self.safety_margin_gb,
            'overhead_multiplier': self.overhead_multiplier,
        }


# Global capability engine instance
_global_capability_engine: Optional[CapabilityEngine] = None


def get_capability_engine() -> CapabilityEngine:
    """Get or create global capability engine."""
    global _global_capability_engine
    
    if _global_capability_engine is None:
        _global_capability_engine = CapabilityEngine()
    
    return _global_capability_engine

