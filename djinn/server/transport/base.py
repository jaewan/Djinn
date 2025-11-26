"""
Transport abstraction. All transports (TCP, DPDK) implement this.
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch

class Transport(ABC):
    """Abstract base for all transports."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Transport name (for logging)."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize transport. Return True if successful."""
        pass
    
    @abstractmethod
    async def send(
        self, 
        tensor: torch.Tensor,
        target: str,
        transfer_id: str,
        metadata: Dict
    ) -> bool:
        """Send tensor to target. Return True if successful."""
        pass
    
    @abstractmethod
    async def receive(
        self,
        transfer_id: str,
        metadata: Dict
    ) -> torch.Tensor:
        """Receive tensor. Return received tensor."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if transport is available on this system."""
        pass