"""
Protocol constants for Djinn transport.

⚠️ IMPORTANT: Packet processing is implemented in C++ (src/data_plane/).
This file ONLY contains constants for Python-C++ interface.
"""

from enum import IntEnum
from typing import NamedTuple

# Protocol constants (must match C++)
GENIE_MAGIC = 0x47454E49  # "GENI"
GENIE_VERSION = 1
DEFAULT_MTU = 1500
DEFAULT_UDP_PORT = 5556

class PacketType(IntEnum):
    """Packet types (must match C++ enum)"""
    DATA = 0
    ACK = 1
    NACK = 2
    HEARTBEAT = 3
    CONTROL = 4

class PacketFlags(IntEnum):
    """Packet flags bitfield"""
    NONE = 0x00
    FRAGMENTED = 0x01
    LAST_FRAGMENT = 0x02
    RETRANSMIT = 0x04

class NetworkNode(NamedTuple):
    """Network node information."""
    id: str
    ip: str
    mac: str
    port: int = DEFAULT_UDP_PORT

# Header sizes (for reference)
ETHERNET_HEADER_SIZE = 14
IPV4_HEADER_SIZE = 20
UDP_HEADER_SIZE = 8
GENIE_HEADER_SIZE = 64
TOTAL_HEADER_SIZE = 106
MAX_PAYLOAD_SIZE = DEFAULT_MTU - TOTAL_HEADER_SIZE  # 1394 bytes

# For backward compatibility with tests, import legacy implementations
try:
    from legacy.protocol_python_reference import PacketBuilder, PacketParser
except ImportError:
    PacketBuilder = PacketParser = None
