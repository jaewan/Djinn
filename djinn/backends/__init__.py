"""
Pluggable backend implementations.

Different inference engines can be used as backends:
- huggingface: Standard HuggingFace Transformers (current default)
- vllm: vLLM for batched inference (future)
- tensorrt_llm: TensorRT-LLM for maximum performance (future)
"""

from .huggingface import HuggingFaceKVCache, HuggingFaceBackend

__all__ = [
    "HuggingFaceKVCache",
    "HuggingFaceBackend",
]

