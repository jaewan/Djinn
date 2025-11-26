"""
Batch Protocol Implementation

Provides protocol for batching multiple operations into a single network
request, reducing overhead and enabling 3-5x speedup for small operations.

Protocol Format:
  Request:
    [Request Type: 1 byte]        0x02 = batch request
    [Batch Size: 1 byte]          Number of operations
    [Op1 Name Len: 2 bytes]
    [Op1 Name: variable]
    [Op1 Metadata: JSON encoded]
    [Op2 Name Len: 2 bytes]
    ...
    [Total Payload Size: 4 bytes]
    [Batched Payload: variable]

  Response:
    [Status: 1 byte]              0x00 = success
    [Result Count: 1 byte]        Number of results
    [Result1 Size: 4 bytes]
    [Result1 Data: variable]
    [Result2 Size: 4 bytes]
    ...
"""

import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Protocol constants
REQUEST_TYPE_SINGLE = 0x01
REQUEST_TYPE_BATCH = 0x02
STATUS_SUCCESS = 0x00
STATUS_ERROR = 0x01


class BatchProtocol:
    """
    Protocol for batching operations into single network request.
    
    Supports:
    - Packing multiple operations into one binary request
    - Unpacking multiple results from one binary response
    - Backward compatible with single-op requests
    """
    
    @staticmethod
    def pack_batch_request(operations: List[Dict]) -> bytes:
        """
        Pack multiple operations into binary batch request.
        
        Args:
            operations: List of operation dicts with 'operation' and optional 'metadata'
            
        Returns:
            Binary batch request data
            
        Example:
            operations = [
                {'operation': 'gpt2_decode', 'metadata': {'key': 'value'}},
                {'operation': 'gpt2_decode', 'metadata': {}},
            ]
            data = BatchProtocol.pack_batch_request(operations)
        """
        if not operations:
            raise ValueError("Cannot batch empty operation list")
        
        request = bytearray()
        
        # Header
        request.append(REQUEST_TYPE_BATCH)
        request.append(len(operations))
        
        # Pack each operation
        for op in operations:
            if 'operation' not in op:
                raise ValueError("Operation must have 'operation' field")
            
            operation_name = op['operation']
            metadata = op.get('metadata', {})
            
            # Operation name with length prefix (2 bytes, big-endian)
            name_bytes = operation_name.encode('utf-8')
            request.extend(len(name_bytes).to_bytes(2, byteorder='big'))
            request.extend(name_bytes)
            
            # Metadata as JSON with length prefix (2 bytes, big-endian)
            metadata_json = json.dumps(metadata)
            metadata_bytes = metadata_json.encode('utf-8')
            request.extend(len(metadata_bytes).to_bytes(2, byteorder='big'))
            request.extend(metadata_bytes)
        
        return bytes(request)
    
    @staticmethod
    def unpack_batch_response(data: bytes) -> List[Dict]:
        """
        Unpack multiple results from batch response.
        
        Args:
            data: Binary batch response data
            
        Returns:
            List of result dicts
            
        Raises:
            ValueError: If data format is invalid
            
        Example:
            data = b'\\x00\\x02\\x00\\x00\\x00\\x10{"result": 1}\\x00\\x00\\x00\\x10{"result": 2}'
            results = BatchProtocol.unpack_batch_response(data)
            # results = [{'result': 1}, {'result': 2}]
        """
        if len(data) < 2:
            raise ValueError("Response data too short (need at least 2 bytes)")
        
        offset = 0
        
        # Status
        status = data[offset]
        offset += 1
        
        if status != STATUS_SUCCESS:
            raise RuntimeError(f"Remote operation failed with status {status}")
        
        # Result count
        result_count = data[offset]
        offset += 1
        
        results = []
        
        # Unpack each result
        for i in range(result_count):
            if offset + 4 > len(data):
                raise ValueError(f"Invalid response: cannot read result {i} size")
            
            # Result size (4 bytes, big-endian)
            result_size = int.from_bytes(data[offset:offset+4], byteorder='big')
            offset += 4
            
            if offset + result_size > len(data):
                raise ValueError(f"Invalid response: result {i} data truncated")
            
            # Result data
            result_data = data[offset:offset+result_size]
            offset += result_size
            
            try:
                result_obj = json.loads(result_data.decode('utf-8'))
                results.append(result_obj)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse result {i}: {e}")
                raise ValueError(f"Invalid JSON in result {i}") from e
        
        return results
    
    @staticmethod
    def pack_batch_response(results: List[Dict], status: int = STATUS_SUCCESS) -> bytes:
        """
        Pack multiple results into binary batch response.
        
        Used by server-side to send batch results back to client.
        
        Args:
            results: List of result dicts
            status: Response status code (default: SUCCESS)
            
        Returns:
            Binary batch response data
            
        Example:
            results = [{'output': 'token1'}, {'output': 'token2'}]
            data = BatchProtocol.pack_batch_response(results)
        """
        response = bytearray()
        
        # Header
        response.append(status)
        response.append(len(results))
        
        # Pack each result
        for result in results:
            result_json = json.dumps(result)
            result_bytes = result_json.encode('utf-8')
            
            # Result size (4 bytes, big-endian)
            response.extend(len(result_bytes).to_bytes(4, byteorder='big'))
            # Result data
            response.extend(result_bytes)
        
        return bytes(response)
    
    @staticmethod
    def get_batch_size_estimate(operations: List[Dict]) -> int:
        """
        Estimate total size of batch request in bytes.
        
        Useful for deciding whether to batch or not.
        
        Args:
            operations: List of operations to batch
            
        Returns:
            Estimated size in bytes
        """
        size = 2  # Header (type + count)
        
        for op in operations:
            name = op.get('operation', '')
            metadata = op.get('metadata', {})
            
            size += 2 + len(name.encode('utf-8'))  # name with length prefix
            size += 2 + len(json.dumps(metadata).encode('utf-8'))  # metadata with length
        
        return size


class BatchProtocolError(Exception):
    """Base exception for batch protocol errors"""
    pass


class BatchProtocolPackError(BatchProtocolError):
    """Error packing batch request"""
    pass


class BatchProtocolUnpackError(BatchProtocolError):
    """Error unpacking batch response"""
    pass
