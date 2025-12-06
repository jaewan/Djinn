"""
InferenceBackend: Framework-agnostic interface for inference execution.

Enables Djinn to work with different inference engines:
- HuggingFace Transformers (current)
- vLLM (future - continuous batching)
- TensorRT-LLM (future - maximum performance)
- Custom backends (user-defined)

The key abstraction: the backend manages model execution and KV state,
while Djinn manages multi-session orchestration and memory.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from .swappable_state import SwappableState


class InferenceBackend(ABC):
    """
    Abstract interface for inference execution backends.
    
    Djinn is a scheduling and memory management layer. The backend handles
    the actual model inference. This separation enables:
    1. Backend upgrades without changing Djinn
    2. Multi-model execution (swap different backends)
    3. Comparative evaluation (same Djinn, different backends)
    """
    
    @abstractmethod
    async def execute(
        self, 
        model_id: str,
        inputs: Dict[str, Any],
        session_state: Optional[SwappableState] = None
    ) -> Tuple[Any, Optional[SwappableState]]:
        """
        Execute inference on the model.
        
        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            inputs: Input dict with 'input_ids', 'attention_mask', etc.
            session_state: Optional KV cache from previous steps
                          (restored by Djinn before calling execute)
        
        Returns:
            Tuple of:
            - output: Model output (logits, predictions, etc.)
            - updated_state: Updated KV cache (or None if not applicable)
        
        Example (HuggingFace):
            output = model(input_ids=inputs['input_ids'], 
                          past_key_values=session_state)
            return output.logits, output.past_key_values
        
        Example (vLLM):
            request_id = str(uuid.uuid4())
            future = self.llm_engine.add_request(request_id, ...)
            output = await future
            return output, None  # vLLM manages KV internally
        """
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str, **kwargs) -> None:
        """
        Load a model into the backend.
        
        Called once per unique model_id. Should be idempotent.
        
        Args:
            model_id: Model to load
            **kwargs: Backend-specific options
        
        Example (HuggingFace):
            from transformers import AutoModelForCausalLM
            self.models[model_id] = AutoModelForCausalLM.from_pretrained(model_id)
        """
        pass
    
    @abstractmethod
    def get_session_state(self, session_id: str) -> Optional[SwappableState]:
        """
        Get the current KV cache state for a session.
        
        Called by Djinn when preparing to evict (swap to CPU).
        
        Returns:
            SwappableState for the session, or None if session not found
        """
        pass
    
    @abstractmethod
    def set_session_state(self, session_id: str, state: Optional[SwappableState]) -> None:
        """
        Set the KV cache state for a session.
        
        Called by Djinn after restoring from CPU (swap in).
        
        Args:
            session_id: Session to update
            state: Restored KV cache, or None to clear
        """
        pass
    
    @abstractmethod
    def get_loaded_models(self) -> Dict[str, Any]:
        """
        Return dict of {model_id: model_object} for currently loaded models.
        
        Used for lifecycle management and cleanup.
        """
        pass

