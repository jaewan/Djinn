"""Module context tracking for semantic metadata enrichment."""

from typing import Optional, List, Dict, Any
import torch.nn as nn
from dataclasses import dataclass, field

from ...common.async_local import AsyncLocal


@dataclass 
class ModuleContext:
    """Context information about the current module execution."""
    module_path: str
    module_type: str
    layer_depth: int
    parent_modules: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


class ModuleContextTracker:
    """Track module execution context for semantic enrichment.
    
    This class maintains a thread-local stack of module contexts to track
    which nn.Module is currently executing, enabling rich semantic metadata.
    """
    
    _thread_local = AsyncLocal("module_context_tracker")
    
    def __init__(self):
        self._module_hooks = {}
        self._module_paths = {}
        self._active = False
        
    @classmethod
    def get_instance(cls) -> "ModuleContextTracker":
        """Get thread-local instance."""
        if not hasattr(cls._thread_local, "instance"):
            cls._thread_local.instance = cls()
        return cls._thread_local.instance
    
    def activate(self, model: nn.Module) -> None:
        """Activate module tracking for a model."""
        if self._active:
            return
            
        # Build module path mapping
        self._build_module_paths(model)
        
        # Register forward hooks
        for name, module in model.named_modules():
            handle = module.register_forward_pre_hook(self._pre_forward_hook)
            self._module_hooks[name] = handle
            
        self._active = True
        
    def deactivate(self) -> None:
        """Deactivate module tracking."""
        for handle in self._module_hooks.values():
            handle.remove()
        self._module_hooks.clear()
        self._module_paths.clear()
        self._active = False
        
    def _build_module_paths(self, model: nn.Module, prefix: str = "") -> None:
        """Build mapping of module objects to their paths."""
        for name, module in model.named_modules():
            full_path = f"{prefix}.{name}" if prefix else name
            self._module_paths[id(module)] = full_path
            
    def _pre_forward_hook(self, module: nn.Module, input) -> None:
        """Hook called before module forward pass."""
        module_path = self._module_paths.get(id(module), "unknown")
        module_type = module.__class__.__name__
        
        # Calculate layer depth
        layer_depth = module_path.count('.')
        
        # Get parent modules
        parts = module_path.split('.')
        parent_modules = ['.'.join(parts[:i+1]) for i in range(len(parts)-1)]
        
        context = ModuleContext(
            module_path=module_path,
            module_type=module_type,
            layer_depth=layer_depth,
            parent_modules=parent_modules,
            attributes=self._extract_module_attributes(module)
        )
        
        self._push_context(context)
        
        # Register post-hook to pop context
        def post_hook(module, input, output):
            self._pop_context()
            
        module.register_forward_hook(post_hook, always_call=True)
        
    def _extract_module_attributes(self, module: nn.Module) -> Dict[str, Any]:
        """Extract relevant attributes from module."""
        attrs = {}
        
        # Common attributes
        if hasattr(module, 'in_features'):
            attrs['in_features'] = module.in_features
        if hasattr(module, 'out_features'):
            attrs['out_features'] = module.out_features
        if hasattr(module, 'hidden_size'):
            attrs['hidden_size'] = module.hidden_size
        if hasattr(module, 'num_heads'):
            attrs['num_heads'] = module.num_heads
        if hasattr(module, 'dropout'):
            attrs['dropout'] = module.dropout
            
        # Layer type hints
        if isinstance(module, nn.MultiheadAttention):
            attrs['is_attention'] = True
            attrs['attention_type'] = 'multihead'
        elif 'attention' in module.__class__.__name__.lower():
            attrs['is_attention'] = True
            attrs['attention_type'] = 'custom'
            
        return attrs
        
    def _push_context(self, context: ModuleContext) -> None:
        """Push context onto stack."""
        if not hasattr(self._thread_local, "context_stack"):
            self._thread_local.context_stack = []
        self._thread_local.context_stack.append(context)
        
    def _pop_context(self) -> Optional[ModuleContext]:
        """Pop context from stack."""
        if hasattr(self._thread_local, "context_stack") and self._thread_local.context_stack:
            return self._thread_local.context_stack.pop()
        return None
        
    def get_current_context(self) -> Optional[ModuleContext]:
        """Get current module context."""
        if hasattr(self._thread_local, "context_stack") and self._thread_local.context_stack:
            return self._thread_local.context_stack[-1]
        return None
        
    def get_context_stack(self) -> List[ModuleContext]:
        """Get full context stack."""
        if hasattr(self._thread_local, "context_stack"):
            return list(self._thread_local.context_stack)
        return []
        
    def detect_execution_phase(self, context: Optional[ModuleContext] = None) -> str:
        """Detect execution phase based on module context."""
        if context is None:
            context = self.get_current_context()
            
        if context is None:
            return "unknown"
            
        module_path = context.module_path.lower()
        module_type = context.module_type.lower()
        
        # LLM phases
        if 'kv_cache' in module_path or 'past_key' in module_path:
            return "decode"  # KV cache operations indicate decode phase
        elif 'embedding' in module_path and context.layer_depth == 0:
            return "prefill"  # Initial embedding lookup
            
        # Vision phases
        if 'backbone' in module_path or 'feature' in module_path:
            return "vision_backbone"
        elif 'head' in module_path or 'classifier' in module_path:
            return "vision_head"
            
        # Multi-modal
        if 'cross' in module_path and 'attention' in module_type:
            return "multimodal_fusion"
        elif 'fusion' in module_path or 'combine' in module_path:
            return "multimodal_fusion"
            
        # Attention patterns
        if context.attributes.get('is_attention'):
            if 'self' in module_path:
                return "self_attention"
            elif 'cross' in module_path:
                return "cross_attention"
                
        return "unknown"
        
    def infer_semantic_role(self, operation: str, context: Optional[ModuleContext] = None) -> str:
        """Infer semantic role of operation based on context."""
        if context is None:
            context = self.get_current_context()
            
        if context is None:
            return "unknown"
            
        # Attention-related roles
        if context.attributes.get('is_attention'):
            if 'query' in operation or 'q_proj' in context.module_path:
                return "attention_query_projection"
            elif 'key' in operation or 'k_proj' in context.module_path:
                return "attention_key_projection"
            elif 'value' in operation or 'v_proj' in context.module_path:
                return "attention_value_projection"
            elif 'out' in operation or 'o_proj' in context.module_path:
                return "attention_output_projection"
                
        # Layer norm roles
        if 'layernorm' in context.module_type.lower() or 'norm' in context.module_path:
            return "normalization"
            
        # FFN roles
        if 'mlp' in context.module_path or 'ffn' in context.module_path:
            if 'gate' in context.module_path:
                return "gated_activation"
            elif context.layer_depth % 2 == 0:
                return "ffn_up_projection"
            else:
                return "ffn_down_projection"
                
        # Vision-specific roles
        if 'conv' in context.module_type.lower():
            return "convolutional_feature_extraction"
        elif 'pool' in context.module_type.lower():
            return "spatial_pooling"
            
        return f"{context.module_type}_operation"


# Global instance for easy access
_global_tracker = None


def get_module_context_tracker() -> ModuleContextTracker:
    """Get global module context tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ModuleContextTracker()
    return _global_tracker
