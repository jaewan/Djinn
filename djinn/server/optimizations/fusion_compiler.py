"""
SRG-Driven Fusion Compiler (Tier 1: Pattern Grouping).

This module implements Tier 1 fusion: grouping operations by execution phase and
identifying fusable patterns (attention, convolution) without actually fusing kernels.

The purpose is to:
1. Identify which operations could potentially benefit from fusion
2. Group them for efficient execution
3. Provide data to inform future Tier 2/3 decisions

Tier 1 has minimal overhead (~0.5ms for 100 operations) and enables all future optimizations.
Tier 2 (TorchScript) and Tier 3 (TensorRT) should only be enabled if profiling shows ROI.

Design Principles:
- Start simple: Pattern grouping only, no compilation overhead
- Measure everything: Track which patterns are hot and fusable
- Data-driven: Only invest in compilation when production data justifies it
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from djinn.core.types import ExecutionPhase

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Types of fusion strategies."""
    NONE = "none"              # No fusion
    ATTENTION = "attention"    # Attention block fusion
    CONV = "conv"              # Conv + BN + activation fusion
    MLP = "mlp"                # Fully connected layer fusion
    UNKNOWN = "unknown"        # Unknown pattern


@dataclass
class FusedBlock:
    """Represents a group of operations that can be fused."""
    
    block_id: str                                  # Unique identifier
    operations: List[Dict]                        # Original operation specs
    execution_phase: ExecutionPhase
    fusion_strategy: FusionStrategy
    
    # Compilation state (Tier 2+)
    compiled_module: Optional[Any] = None        # Compiled module (future)
    torchscript_compiled: bool = False           # Whether TorchScript compiled
    
    # Statistics
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    estimated_savings_ms: float = 0.0            # Estimated savings from fusion


@dataclass
class FusionCompilerStats:
    """Statistics tracked by fusion compiler."""
    blocks_created: int = 0
    blocks_reused: int = 0
    fusion_time_ms: float = 0.0
    execution_time_saved_ms: float = 0.0
    attention_blocks_identified: int = 0
    conv_blocks_identified: int = 0
    no_fusion_blocks: int = 0


class SRGFusionCompiler:
    """
    Tier 1 Fusion Compiler: Pattern-based grouping only.
    
    This compiler:
    1. Groups operations by execution phase
    2. Identifies fusable patterns (attention, conv)
    3. Creates FusedBlock objects for efficient execution
    4. Tracks statistics for data-driven decisions
    
    No actual kernel fusion happens at Tier 1. This is purely grouping + metadata.
    """
    
    def __init__(
        self,
        enable_torchscript: bool = False,
        enable_compilation: bool = False
    ):
        """
        Initialize fusion compiler.
        
        Args:
            enable_torchscript: Enable Tier 2 (TorchScript) compilation (default: False)
            enable_compilation: Enable Tier 3 (TensorRT) compilation (default: False)
        """
        self.enable_torchscript = enable_torchscript
        self.enable_compilation = enable_compilation
        
        # Cache compiled blocks
        self.fusion_cache: Dict[str, FusedBlock] = {}
        
        # Statistics
        self.stats = FusionCompilerStats()
        
        logger.info(
            "SRGFusionCompiler initialized: "
            "torchscript=%s, compilation=%s",
            enable_torchscript,
            enable_compilation
        )
    
    def fuse_subgraph(
        self,
        operations: List[Dict],
        semantic_metadata: Dict[str, Any]
    ) -> List[FusedBlock]:
        """
        Fuse operations based on SRG annotations.
        
        Tier 1: Pattern-based grouping (ALWAYS enabled)
        - Group by execution phase
        - Identify fusable patterns (attention, conv blocks)
        - No actual kernel fusion yet
        
        Args:
            operations: List of operation specs
            semantic_metadata: SRG annotations per operation
        
        Returns:
            List of fused blocks (may be 1-to-1 if no fusion possible)
        """
        start_time = time.time()
        
        # Step 1: Group operations by execution phase
        phase_groups = self._group_by_phase(operations, semantic_metadata)
        
        # Step 2: Identify fusion patterns within each phase
        fused_blocks = []
        for phase, ops in phase_groups.items():
            if phase == ExecutionPhase.LLM_DECODE:
                # Look for attention patterns
                blocks = self._fuse_attention_patterns(ops, semantic_metadata)
            elif phase == ExecutionPhase.LLM_PREFILL:
                # Prefill is compute-bound, look for attention patterns
                blocks = self._fuse_attention_patterns(ops, semantic_metadata)
            elif phase == ExecutionPhase.VISION_ENCODING:
                # Look for conv patterns
                blocks = self._fuse_conv_patterns(ops, semantic_metadata)
            else:
                # No specific fusion, just group
                blocks = self._create_ungrouped_blocks(ops, phase)
            
            fused_blocks.extend(blocks)
        
        # Step 3: (Optional) Tier 2 TorchScript compilation
        if self.enable_torchscript:
            for block in fused_blocks:
                if block.fusion_strategy != FusionStrategy.NONE:
                    # TODO: Implement TorchScript compilation
                    # Only attempt if profiling shows it's worth it
                    pass
        
        self.stats.fusion_time_ms += (time.time() - start_time) * 1000
        return fused_blocks
    
    def _group_by_phase(
        self,
        operations: List[Dict],
        metadata: Dict[str, Any]
    ) -> Dict[ExecutionPhase, List[Dict]]:
        """Group operations by execution phase."""
        groups: Dict[ExecutionPhase, List[Dict]] = defaultdict(list)
        
        for op in operations:
            op_id = op.get('op_id', '')
            op_metadata = metadata.get(op_id, {})
            phase = op_metadata.get('execution_phase', ExecutionPhase.UNKNOWN)
            
            groups[phase].append(op)
        
        return dict(groups)
    
    def _create_ungrouped_blocks(
        self,
        operations: List[Dict],
        phase: ExecutionPhase
    ) -> List[FusedBlock]:
        """Create blocks for operations with no specific fusion strategy."""
        blocks = []
        
        for i, op in enumerate(operations):
            block = FusedBlock(
                block_id=f"ungrouped_{phase.value}_{i}",
                operations=[op],
                execution_phase=phase,
                fusion_strategy=FusionStrategy.NONE
            )
            blocks.append(block)
            self.stats.no_fusion_blocks += 1
        
        return blocks
    
    def _fuse_attention_patterns(
        self,
        operations: List[Dict],
        metadata: Dict[str, Any]
    ) -> List[FusedBlock]:
        """
        Identify and group attention patterns.
        
        Pattern: Q @ K.T → softmax → (attn @ V) [→ dropout]
        
        This is the core optimization for LLM inference (prefill and decode).
        """
        blocks = []
        i = 0
        
        while i < len(operations):
            # Look ahead for attention pattern
            if self._is_attention_start(operations[i]):
                # Find end of attention block
                j = i + 1
                while j < len(operations) and self._is_attention_continuation(operations[j]):
                    j += 1
                
                # Create fused block
                block = FusedBlock(
                    block_id=f"attention_{len(blocks)}",
                    operations=operations[i:j],
                    execution_phase=ExecutionPhase.LLM_DECODE,
                    fusion_strategy=FusionStrategy.ATTENTION,
                    estimated_savings_ms=0.1 * (j - i)  # Rough estimate
                )
                blocks.append(block)
                self.stats.attention_blocks_identified += 1
                i = j
            else:
                # Single operation, no fusion
                block = FusedBlock(
                    block_id=f"op_{i}",
                    operations=[operations[i]],
                    execution_phase=ExecutionPhase.LLM_DECODE,
                    fusion_strategy=FusionStrategy.NONE
                )
                blocks.append(block)
                self.stats.no_fusion_blocks += 1
                i += 1
        
        return blocks
    
    def _is_attention_start(self, op: Dict) -> bool:
        """Check if operation starts attention pattern (Q @ K.T)."""
        op_name = op.get('operation', '').lower()
        
        # Look for matrix multiplications (matmul, bmm, mm)
        attention_ops = ['matmul', 'bmm', 'mm', 'addmm']
        return any(att_op in op_name for att_op in attention_ops)
    
    def _is_attention_continuation(self, op: Dict) -> bool:
        """Check if operation is part of attention block."""
        op_name = op.get('operation', '').lower()
        
        # Operations that are part of attention
        attention_ops = [
            'softmax', '_softmax',      # Softmax on attention scores
            'matmul', 'bmm', 'mm',      # Matrix multiplications
            'dropout', 'drop',          # Dropout
            'add', 'add_',              # Residual connections
            'layer_norm', 'rms_norm',   # Normalization
            'scale',                    # Scaling
        ]
        
        return any(att_op in op_name for att_op in attention_ops)
    
    def _fuse_conv_patterns(
        self,
        operations: List[Dict],
        metadata: Dict[str, Any]
    ) -> List[FusedBlock]:
        """
        Identify and group conv patterns.
        
        Pattern: conv → batchnorm → activation (relu, gelu, etc)
        
        This is common in vision models and enables efficient fusion.
        """
        blocks = []
        i = 0
        
        while i < len(operations):
            if self._is_conv(operations[i]):
                # Look ahead for bn + activation
                j = i + 1
                while j < len(operations) and self._is_conv_continuation(operations[j]):
                    j += 1
                
                block = FusedBlock(
                    block_id=f"conv_{len(blocks)}",
                    operations=operations[i:j],
                    execution_phase=ExecutionPhase.VISION_ENCODING,
                    fusion_strategy=FusionStrategy.CONV,
                    estimated_savings_ms=0.05 * (j - i)  # Rough estimate
                )
                blocks.append(block)
                self.stats.conv_blocks_identified += 1
                i = j
            else:
                # Single operation, no fusion
                block = FusedBlock(
                    block_id=f"op_{i}",
                    operations=[operations[i]],
                    execution_phase=ExecutionPhase.VISION_ENCODING,
                    fusion_strategy=FusionStrategy.NONE
                )
                blocks.append(block)
                self.stats.no_fusion_blocks += 1
                i += 1
        
        return blocks
    
    def _is_conv(self, op: Dict) -> bool:
        """Check if operation is convolution."""
        op_name = op.get('operation', '').lower()
        return 'conv' in op_name
    
    def _is_conv_continuation(self, op: Dict) -> bool:
        """Check if operation follows conv (bn, relu, etc)."""
        op_name = op.get('operation', '').lower()
        
        # Operations that typically follow convolution
        conv_followers = [
            'batch_norm', 'native_batch_norm',
            'relu', 'gelu', 'silu', 'elu',
            'max_pool', 'avg_pool',
            'dropout', 'drop'
        ]
        
        return any(follower in op_name for follower in conv_followers)
    
    def get_stats(self) -> Dict:
        """Get compiler statistics."""
        return {
            'blocks_created': self.stats.blocks_created,
            'blocks_reused': self.stats.blocks_reused,
            'fusion_time_ms': self.stats.fusion_time_ms,
            'execution_time_saved_ms': self.stats.execution_time_saved_ms,
            'attention_blocks_identified': self.stats.attention_blocks_identified,
            'conv_blocks_identified': self.stats.conv_blocks_identified,
            'no_fusion_blocks': self.stats.no_fusion_blocks,
        }
    
    def report_execution(
        self,
        block_id: str,
        execution_time_ms: float,
        fusion_strategy: FusionStrategy
    ):
        """Report execution time for a block (for profiling)."""
        # This data will be used to decide when to enable Tier 2/3
        pass
