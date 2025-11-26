"""
Control plane message types and message handling.

Defines message types, status enums, and the ControlMessage dataclass
for serialization/deserialization of control plane messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Dict, Any


class MessageType(IntEnum):
    """Control plane message types"""
    # Connection management
    HELLO = 1
    CAPABILITY_EXCHANGE = 2
    HEARTBEAT = 3
    GOODBYE = 4

    # Transfer coordination
    TRANSFER_REQUEST = 10
    TRANSFER_READY = 11
    TRANSFER_START = 12
    TRANSFER_COMPLETE = 13
    TRANSFER_ERROR = 14
    TRANSFER_CANCEL = 15

    # Status and monitoring
    STATUS_REQUEST = 20
    STATUS_RESPONSE = 21
    NODE_LIST_REQUEST = 22
    NODE_LIST_RESPONSE = 23


class TransferStatus(IntEnum):
    """Transfer status values"""
    PENDING = 0
    NEGOTIATING = 1
    READY = 2
    IN_PROGRESS = 3
    COMPLETED = 4
    FAILED = 5
    CANCELLED = 6


@dataclass
class ControlMessage:
    """Control plane message"""
    type: MessageType
    sender: str
    timestamp: float
    message_id: str
    payload: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        data = asdict(self)
        data['type'] = int(self.type)
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ControlMessage':
        """Deserialize message from JSON string."""
        data = json.loads(json_str)
        data['type'] = MessageType(data['type'])
        return cls(**data)

