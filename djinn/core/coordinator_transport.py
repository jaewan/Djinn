"""
Transport Selection and Management for Djinn Coordinator.

Handles transport selection, tensor sending, and metadata extraction.
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class TransportSelector:
    """
    Selects appropriate transport for tensor transfers.
    
    Strategy:
    1. If GPU tensor + DPDK available → use DPDK (100 Gbps)
    2. Otherwise → use TCP (10 Gbps, but always works)
    """
    
    def __init__(self, transports: Dict):
        """
        Initialize transport selector.
        
        Args:
            transports: Dictionary of available transports (e.g., {'dpdk': ..., 'tcp': ...})
        """
        self.transports = transports
    
    def select_transport(self, tensor: torch.Tensor, metadata: Dict) -> Any:
        """
        Select best transport based on context.
        
        Args:
            tensor: PyTorch tensor to transfer
            metadata: Transfer metadata
            
        Returns:
            Transport instance
        """
        # Prefer DPDK for GPU tensors (zero-copy)
        if tensor.is_cuda and 'dpdk' in self.transports:
            return self.transports['dpdk']
        
        # Otherwise use TCP
        if 'tcp' not in self.transports:
            raise RuntimeError("No available transport for CPU tensor")
        return self.transports['tcp']
    
    def select_transport_for_metadata(self, metadata: Dict) -> Any:
        """
        Select transport based on metadata.
        
        Args:
            metadata: Transfer metadata
            
        Returns:
            Transport instance
        """
        # For Phase 1, always use TCP
        if 'tcp' not in self.transports:
            raise RuntimeError("TCP transport not available")
        return self.transports['tcp']


class MetadataExtractor:
    """Extracts and enriches semantic metadata from tensors."""
    
    def __init__(self, scheduler=None):
        """
        Initialize metadata extractor.
        
        Args:
            scheduler: Optional scheduler for phase inference
        """
        self.scheduler = scheduler
    
    def extract_metadata(self, tensor: torch.Tensor, user_metadata: Optional[Dict] = None) -> Dict:
        """
        Extract semantic metadata from tensor.
        
        Args:
            tensor: PyTorch tensor
            user_metadata: Optional user-provided metadata
            
        Returns:
            Dictionary with enriched metadata
        """
        metadata = {
            'dtype': str(tensor.dtype),
            'shape': list(tensor.shape),
            'size_bytes': tensor.numel() * tensor.element_size(),
            'device': str(tensor.device),
            'is_gpu': tensor.is_cuda,
        }
        
        # Add user-provided semantic metadata
        if user_metadata:
            metadata.update(user_metadata)
        
        # Infer phase if not provided (from shape heuristics)
        if 'phase' not in metadata:
            metadata['phase'] = self._infer_phase(tensor)
        
        return metadata
    
    def _infer_phase(self, tensor: torch.Tensor) -> str:
        """
        Infer execution phase from tensor characteristics.
        
        Simple heuristic: large batch = prefill, small = decode
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Phase string ('prefill', 'decode', or 'unknown')
        """
        # Simple heuristic: large batch = prefill, small = decode
        if len(tensor.shape) >= 2:
            batch_size = tensor.shape[0]
            if batch_size >= 32:
                return "prefill"
            elif batch_size == 1:
                return "decode"
        return "unknown"

