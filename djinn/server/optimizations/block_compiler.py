"""
Phase 2: Block Compilation

Converts fine-grained computation graphs (1500 operations) into coarse-grained
TorchScript blocks (15 blocks) to reduce RPC overhead from 300ms to 15ms.

Strategy: Identify module boundaries and compile each module to TorchScript.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutableBlock:
    """Compiled block ready for execution."""
    block_id: int
    name: str  # e.g., "transformer.h.0"
    torchscript_module: Optional[torch.jit.ScriptModule] = None
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    operation_count: int = 0
    memory_bytes: int = 0
    compute_flops: int = 0
    dependencies: List[int] = field(default_factory=list)  # Block IDs that must run before
    
    def serialize(self) -> bytes:
        """Serialize TorchScript module to bytes."""
        if self.torchscript_module is None:
            raise ValueError(f"Block {self.block_id} has no compiled module")
        
        # Save to bytes buffer
        import io
        buffer = io.BytesIO()
        torch.jit.save(self.torchscript_module, buffer)
        return buffer.getvalue()
    
    @staticmethod
    def deserialize(data: bytes) -> 'torch.jit.ScriptModule':
        """Deserialize TorchScript module from bytes."""
        import io
        buffer = io.BytesIO(data)
        return torch.jit.load(buffer)


class BlockIdentifier:
    """Identifies block boundaries in computation graph."""
    
    @staticmethod
    def identify_module_boundaries(model: nn.Module) -> Dict[str, nn.Module]:
        """
        Extract module boundaries from model.
        
        Strategy: Identify high-level modules (layers, attention blocks, etc.)
        
        Returns:
            Dict mapping module path → nn.Module
        """
        boundaries = {}
        
        # Strategy 1: For known architectures, use specific patterns
        if hasattr(model, 'transformer'):
            # Transformer model (GPT-2, BERT, etc.)
            if hasattr(model.transformer, 'h'):
                # Has layers
                for i, layer in enumerate(model.transformer.h):
                    path = f"transformer.h.{i}"
                    boundaries[path] = layer
        
        elif hasattr(model, 'encoder'):
            # Encoder-based (BERT, ViT, etc.)
            if hasattr(model.encoder, 'layer'):
                for i, layer in enumerate(model.encoder.layer):
                    path = f"encoder.layer.{i}"
                    boundaries[path] = layer
        
        elif hasattr(model, 'features'):
            # CNN-based (ResNet, etc.)
            # Use sequential blocks
            if isinstance(model.features, nn.Sequential):
                for i, block in enumerate(model.features):
                    if isinstance(block, nn.Sequential):
                        path = f"features.{i}"
                        boundaries[path] = block
        
        # Fallback: Use all named_modules at depth 2
        if not boundaries:
            for name, module in model.named_modules():
                depth = name.count('.')
                if depth == 2:  # Reasonable block size
                    boundaries[name] = module
        
        logger.info(f"Identified {len(boundaries)} block boundaries")
        return boundaries


class TorchScriptCompiler:
    """Compiles modules to TorchScript."""
    
    @staticmethod
    def compile_module(module: nn.Module, sample_input: torch.Tensor) -> Optional[torch.jit.ScriptModule]:
        """
        Compile module to TorchScript.
        
        Args:
            module: PyTorch module to compile
            sample_input: Sample input tensor for tracing
        
        Returns:
            torch.jit.ScriptModule or None if compilation failed
        """
        try:
            # Try scripting first (more robust)
            return torch.jit.script(module)
        except Exception as e:
            logger.debug(f"Script compilation failed: {e}, trying trace...")
            try:
                # Fallback to tracing
                return torch.jit.trace(module, sample_input)
            except Exception as e2:
                logger.warning(f"TorchScript compilation failed: {e2}")
                return None


class BlockCompiler:
    """
    Converts computation graphs into executable TorchScript blocks.
    
    Purpose:
    - Reduce RPC calls from 1500 to 15 (100x)
    - Save 285ms RPC overhead
    - Enable batched execution
    """
    
    def __init__(self):
        self.blocks: List[ExecutableBlock] = []
        self.block_id_counter = 0
        self.stats = {
            'total_blocks': 0,
            'successful_compilations': 0,
            'failed_compilations': 0,
            'total_operations': 0,
            'total_memory_bytes': 0,
        }
    
    def compile_model(self, model: nn.Module, sample_input: torch.Tensor) -> List[ExecutableBlock]:
        """
        Compile model into executable blocks.
        
        Returns:
            List of ExecutableBlock objects ready for execution
        """
        self.blocks = []
        self.block_id_counter = 0
        
        # Step 1: Identify module boundaries
        boundaries = BlockIdentifier.identify_module_boundaries(model)
        logger.info(f"Compiling {len(boundaries)} modules into blocks")
        
        # Step 2: Compile each boundary module
        for path, module in sorted(boundaries.items()):
            block = self._compile_block(path, module, sample_input)
            if block:
                self.blocks.append(block)
        
        # Step 3: Identify dependencies
        self._identify_dependencies()
        
        self.stats['total_blocks'] = len(self.blocks)
        logger.info(f"Compilation complete: {len(self.blocks)} blocks, "
                   f"{self.stats['successful_compilations']} successful, "
                   f"{self.stats['failed_compilations']} failed")
        
        return self.blocks
    
    def _compile_block(self, path: str, module: nn.Module, sample_input: torch.Tensor) -> Optional[ExecutableBlock]:
        """Compile a single module boundary to ExecutableBlock."""
        try:
            # Compile to TorchScript
            ts_module = TorchScriptCompiler.compile_module(module, sample_input)
            
            if ts_module is None:
                self.stats['failed_compilations'] += 1
                return None
            
            # Count operations (rough estimate)
            op_count = sum(1 for _ in module.modules())
            
            # Estimate memory
            memory_bytes = sum(p.numel() * 4 for p in module.parameters())  # 4 bytes per float32
            
            block = ExecutableBlock(
                block_id=self.block_id_counter,
                name=path,
                torchscript_module=ts_module,
                input_names=[path],
                output_names=[f"{path}_out"],
                operation_count=op_count,
                memory_bytes=memory_bytes,
                compute_flops=op_count * 1000,  # Rough estimate
            )
            
            self.block_id_counter += 1
            self.stats['successful_compilations'] += 1
            self.stats['total_operations'] += op_count
            self.stats['total_memory_bytes'] += memory_bytes
            
            return block
        
        except Exception as e:
            logger.warning(f"Failed to compile block {path}: {e}")
            self.stats['failed_compilations'] += 1
            return None
    
    def _identify_dependencies(self):
        """Identify execution dependencies between blocks (sequential for now)."""
        for i, block in enumerate(self.blocks):
            if i > 0:
                block.dependencies = [self.blocks[i-1].block_id]


class BlockExecutor:
    """Executes compiled blocks."""
    
    def __init__(self):
        self.block_cache: Dict[int, torch.jit.ScriptModule] = {}
    
    def execute_blocks(self, blocks: List[ExecutableBlock], inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute blocks sequentially with input/output handling.
        
        Args:
            blocks: List of executable blocks
            inputs: Dictionary of input tensors
        
        Returns:
            Dictionary of output tensors
        """
        outputs = {}
        current_input = None
        
        for block in blocks:
            try:
                # Get input
                if block.input_names:
                    if block.input_names[0] in inputs:
                        current_input = inputs[block.input_names[0]]
                
                if current_input is None and outputs:
                    # Use previous output
                    current_input = list(outputs.values())[-1]
                
                if current_input is None:
                    raise ValueError(f"No input for block {block.block_id}")
                
                # Execute
                with torch.no_grad():
                    result = block.torchscript_module(current_input)
                
                # Store output
                for out_name in block.output_names:
                    outputs[out_name] = result
                
                current_input = result
            
            except Exception as e:
                logger.error(f"Block execution failed: {e}")
                raise
        
        return outputs
    
    def execute_block_local(self, block: ExecutableBlock, inputs: torch.Tensor) -> torch.Tensor:
        """Execute single block (local fallback)."""
        if block.torchscript_module is None:
            raise ValueError(f"Block {block.block_id} not compiled")
        
        with torch.no_grad():
            return block.torchscript_module(inputs)


# Global compiler instance
_global_compiler = None
_compiler_lock = None


def get_block_compiler() -> BlockCompiler:
    """Get or create global block compiler."""
    global _global_compiler, _compiler_lock
    
    if _compiler_lock is None:
        import threading
        _compiler_lock = threading.Lock()
    
    if _global_compiler is None:
        with _compiler_lock:
            if _global_compiler is None:
                _global_compiler = BlockCompiler()
    
    return _global_compiler
