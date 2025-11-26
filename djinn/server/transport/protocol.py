"""
TCP Protocol Constants for Djinn.

Centralized message type definitions to avoid magic numbers and improve maintainability.
"""

from enum import IntEnum


class MessageType(IntEnum):
    """TCP message type constants."""
    # Tensor transfer types (legacy protocol)
    SINGLE_TENSOR = 0x01
    MULTI_TENSOR = 0x02

    # Cache query protocol
    CACHE_QUERY = 0x04

    # Model cache protocol
    REGISTER_MODEL = 0x05
    EXECUTE_MODEL = 0x06
    INIT_MODEL = 0x07  # Explicit model initialization (warmup)
    WARMUP_GPU = 0x08  # ✅ NEW: One-time GPU warmup (server-wide)
    REGISTER_MODEL_CHUNKED = 0x09
    REGISTER_MODEL_CHUNK = 0x0A
    REGISTER_MODEL_FINALIZE = 0x0B
    QUERY_RESULT = 0x0C  # ✅ NEW: Query result of async task
    EXECUTE_BATCH = 0x0D  # ✅ NEW: Execute batch of models
    EXECUTE_STAGE = 0x0E  # Stage-level execution (encoder/decoder)

    # Error response
    ERROR = 0xFF
    
    @classmethod
    def is_chunked_protocol(cls, msg_type: int) -> bool:
        """Check if message type is part of chunked protocol (keep-alive)."""
        return msg_type in (cls.REGISTER_MODEL_CHUNKED, cls.REGISTER_MODEL_CHUNK, cls.REGISTER_MODEL_FINALIZE)

