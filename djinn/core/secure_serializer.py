"""
Secure request/response serializer for TCP protocol.

Replaces unsafe pickle with JSON + binary protocol:
- JSON for metadata (safe, human-readable)
- Binary for tensor data (efficient)
- No code execution risk

Protocol format:
[1 byte: version] [4 bytes: metadata_len] [N bytes: JSON metadata] [tensor data...]
"""

import json
import struct
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Protocol version
PROTOCOL_VERSION = 1  # JSON + binary format

# Maximum metadata size (prevent DoS)
MAX_METADATA_SIZE = 10 * 1024 * 1024  # 10MB


class SecureSerializer:
    """Secure serializer for request/response messages."""
    
    @staticmethod
    def serialize_request(request: Dict[str, Any]) -> bytes:
        """
        Serialize request to secure binary format.
        
        Format:
        [1 byte: version] [4 bytes: metadata_len] [N bytes: JSON metadata] [tensor data...]
        
        ✅ OPTIMIZATION: If request has 'weights_binary', use direct binary protocol:
        [1 byte: version] [4 bytes: metadata_len] [N bytes: JSON metadata] [weights_binary]
        
        Args:
            request: Request dictionary (JSON-serializable metadata + optional tensors)
            
        Returns:
            Serialized bytes
        """
        # ✅ OPTIMIZATION: Check for direct binary weights (new protocol)
        # Supports both 'weights_binary' (full model) and 'chunk_data_binary' (chunks)
        has_binary_weights = ('weights_binary' in request and isinstance(request['weights_binary'], bytes))
        has_binary_chunk = ('chunk_data_binary' in request and isinstance(request['chunk_data_binary'], bytes))
        
        if has_binary_weights or has_binary_chunk:
            # Direct binary protocol - skip JSON for weights/chunks
            # Handle architecture_data specially (it's bytes, can't be in JSON)
            architecture_data = request.get('architecture_data')
            # Determine which binary field to use
            binary_field = 'weights_binary' if has_binary_weights else 'chunk_data_binary'
            binary_data = request[binary_field]
            
            metadata = {k: v for k, v in request.items() 
                       if k not in ('weights_binary', 'chunk_data_binary', 'architecture_data')}
            # ✅ TD-2 FIX: Ensure binary protocol flag is always set and preserved
            # Use multiple indicators to ensure detection works even if flag is lost
            metadata['_binary_protocol'] = True
            metadata['_has_binary_weights'] = has_binary_weights  # Additional indicator
            metadata['_has_binary_chunk'] = has_binary_chunk    # Additional indicator
            # Store architecture_data size in metadata (actual data sent separately)
            if architecture_data:
                metadata['_architecture_data_size'] = len(architecture_data)
            
            # ✅ TD-2 FIX: Validate flag will be preserved in JSON
            metadata_json = json.dumps(metadata, default=str).encode('utf-8')
            # Verify flag is in JSON (it should be, but check to be sure)
            metadata_decoded = json.loads(metadata_json.decode('utf-8'))
            if not metadata_decoded.get('_binary_protocol'):
                logger.warning("⚠️  Binary protocol flag lost during JSON encoding, using fallback detection")
            
            if len(metadata_json) > MAX_METADATA_SIZE:
                raise ValueError(f"Metadata too large: {len(metadata_json)} bytes (max: {MAX_METADATA_SIZE})")
            
            # Protocol: [version] [metadata_len] [metadata_json] [architecture_data_len] [architecture_data] [binary_data]
            # ✅ IMPORTANT: For binary protocol, we DON'T include tensor_count/binary_data_count headers
            # The server will detect this by checking _binary_protocol flag in metadata
            msg = bytearray()
            msg.append(PROTOCOL_VERSION)
            msg.extend(struct.pack('>I', len(metadata_json)))
            msg.extend(metadata_json)
            
            # Send architecture_data if present (before binary_data)
            if architecture_data:
                msg.extend(struct.pack('>Q', len(architecture_data)))  # 8 bytes: size
                msg.extend(architecture_data)  # N bytes: architecture data
            else:
                msg.extend(struct.pack('>Q', 0))  # 8 bytes: size = 0 (no architecture data)
            
            msg.extend(binary_data)  # Direct binary, no JSON, no tensor count headers
            
            return bytes(msg)
        
        # Legacy path: Separate metadata from tensor data
        metadata = {}
        tensor_data = {}
        
        for key, value in request.items():
            if isinstance(value, torch.Tensor):
                # Store tensor metadata, data will be serialized separately
                tensor_data[key] = value
                metadata[key] = {
                    '_is_tensor': True,
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device)
                }
            elif isinstance(value, bytes):
                # ✅ IMPROVEMENT: For large binary data, send separately (like tensors)
                # Base64 adds 33% overhead, so avoid for large data
                BINARY_SIZE_THRESHOLD = 1024  # 1KB threshold
                
                if len(value) > BINARY_SIZE_THRESHOLD:
                    # Large binary data - send separately (like tensors)
                    tensor_data[key] = value  # Store as bytes
                    metadata[key] = {
                        '_is_binary': True,
                        'size': len(value)
                    }
                else:
                    # Small binary data - base64 is fine (overhead acceptable)
                    import base64
                    metadata[key] = {
                        '_is_bytes': True,
                        'data': base64.b64encode(value).decode('utf-8')
                    }
            elif isinstance(value, dict):
                # Check if it's a dict of tensors (like uncached_weights)
                has_tensors = any(isinstance(v, torch.Tensor) for v in value.values())
                if has_tensors:
                    # Dict of tensors - serialize each tensor separately
                    tensor_data[key] = value  # Store as special dict
                    # Store minimal metadata (just keys, not full tensor metadata)
                    # Full tensor metadata will be stored per-tensor in binary section
                    metadata[key] = {
                        '_is_tensor_dict': True,
                        'keys': list(value.keys())
                        # Don't store tensor_metadata here - it's too large for JSON
                        # Tensor metadata will be stored per-tensor in binary section
                    }
                # ✅ BUG FIX: Check for dict of serialized tensor dicts (like execution inputs)
                # Pattern: {'input_ids': {'data': bytes, 'shape': [...], 'format': '...'}, ...}
                elif len(value) > 0 and all(
                    isinstance(v, dict) and 
                    'data' in v and 
                    'format' in v and 
                    isinstance(v.get('data'), bytes)
                    for v in value.values()
                ):
                    # Dict of serialized tensor dicts (like execution inputs from _serialize_inputs)
                    # Send each as binary data, not JSON
                    tensor_data[key] = value  # Store as special dict
                    metadata[key] = {
                        '_is_serialized_dict': True,
                        'keys': list(value.keys()),
                        # Store metadata for each tensor (shape, dtype) for reconstruction
                        'tensors': {
                            k: {
                                'shape': v.get('shape', []),
                                'dtype': v.get('dtype', 'float32')
                            }
                            for k, v in value.items()
                        }
                    }
                elif isinstance(value, dict) and 'data' in value and 'format' in value and isinstance(value.get('data'), bytes):
                    # ✅ FIX: Single serialized tensor dict (from _serialize_tensor)
                    # Pattern: {'data': bytes, 'shape': [...], 'dtype': '...', 'format': '...'}
                    # Extract bytes directly and send as binary data
                    tensor_data[key] = value['data']  # Store as bytes
                    metadata[key] = {
                        '_is_binary': True,
                        'size': len(value['data']),
                        'shape': value.get('shape', []),
                        'dtype': value.get('dtype', 'float32'),
                        'format': value.get('format', 'numpy_binary'),
                        'keys': [key],
                        'tensors': {
                            key: {
                                'shape': value.get('shape', []),
                                'dtype': value.get('dtype', 'float32')
                            }
                        }
                    }
                # Legacy dict-based serialization removed - use binary protocol instead
                # If you see this code path, it means something is still using old dict format
                # This should not happen - all weights should use _serialize_weights_binary()
                else:
                    # Regular dict - JSON-serializable
                    metadata[key] = value
            elif isinstance(value, (list, str, int, float, bool, type(None))):
                # JSON-serializable
                metadata[key] = value
            else:
                # Try to convert to JSON-serializable
                try:
                    json.dumps(value)  # Test if serializable
                    metadata[key] = value
                except (TypeError, ValueError):
                    logger.warning(f"Value for key '{key}' is not JSON-serializable, converting to string")
                    metadata[key] = str(value)
        
        # Serialize metadata to JSON
        try:
            metadata_json = json.dumps(metadata, default=str).encode('utf-8')
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize metadata to JSON: {e}")
        
        if len(metadata_json) > MAX_METADATA_SIZE:
            raise ValueError(f"Metadata too large: {len(metadata_json)} bytes (max: {MAX_METADATA_SIZE})")
        
        # Phase 3: Only binary protocol or simple JSON requests supported
        # No legacy tensor/binary data serialization

        # Build message
        msg = bytearray()
        msg.append(PROTOCOL_VERSION)  # 1 byte: version
        msg.extend(struct.pack('>I', len(metadata_json)))  # 4 bytes: metadata length
        msg.extend(metadata_json)  # N bytes: JSON metadata

        # Append binary data for serialized tensors (e.g., execution inputs)
        tensor_entries = []
        for key, entry in metadata.items():
            if not isinstance(entry, dict):
                continue

            if entry.get('_is_tensor'):
                # Handle _is_tensor entries: serialize tensor to bytes
                tensor = tensor_data.get(key)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    # Serialize tensor to numpy bytes
                    cpu_tensor = tensor.cpu().contiguous()
                    tensor_bytes = cpu_tensor.numpy().tobytes()
                    tensor_entries.append((key, tensor_bytes))

            elif entry.get('_is_serialized_dict'):
                dict_bytes = tensor_data.get(key)
                if dict_bytes is None:
                    continue
                for tensor_key in entry.get('keys', []):
                    tensor_payload = dict_bytes.get(tensor_key)
                    if tensor_payload is None:
                        continue
                    tensor_entries.append((tensor_key, tensor_payload['data']))

            elif entry.get('_is_binary'):
                tensor_bytes = tensor_data.get(key)
                if tensor_bytes is None:
                    continue
                # Support multiple keys if provided (rare)
                keys = entry.get('keys') or [key]
                for tensor_key in keys:
                    tensor_entries.append((tensor_key, tensor_bytes))

        if tensor_entries:
            msg.extend(struct.pack('>I', len(tensor_entries)))
            for tensor_key, tensor_bytes in tensor_entries:
                key_bytes = tensor_key.encode('utf-8')
                msg.extend(struct.pack('>I', len(key_bytes)))
                msg.extend(key_bytes)
                msg.extend(struct.pack('>Q', len(tensor_bytes)))
                msg.extend(tensor_bytes)

        return bytes(msg)

    @staticmethod
    def deserialize_request(data: bytes) -> Dict[str, Any]:
        """
        Deserialize request from secure binary format (Phase 3 simplified).

        Only supports:
        - Binary protocol (REGISTER_MODEL with weights_binary)
        - Simple JSON requests (no additional data)

        Args:
            data: Serialized request bytes

        Returns:
            Deserialized request dictionary
        """
        if len(data) < 5:
            raise ValueError(f"Invalid request data: too short ({len(data)} bytes)")

        offset = 0

        # Read version
        version = data[offset]
        offset += 1

        if version != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported protocol version: {version} (expected: {PROTOCOL_VERSION})")

        # Read metadata length
        metadata_len = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4

        if metadata_len > MAX_METADATA_SIZE:
            raise ValueError(f"Metadata too large: {metadata_len} bytes (max: {MAX_METADATA_SIZE})")

        # Read metadata
        metadata_json = data[offset:offset+metadata_len].decode('utf-8')
        offset += metadata_len

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse metadata JSON: {e}")

        request_type = metadata.get('type', '')

        # Check for binary protocol
        is_register_model = request_type == 'REGISTER_MODEL'
        is_register_chunk = request_type == 'REGISTER_MODEL_CHUNK'
        is_execute_model = request_type == 'EXECUTE_MODEL'
        # ✅ FIX: Check for binary protocol flags in metadata (weights_binary won't be in metadata JSON)
        has_binary_flag = metadata.get('_binary_protocol')
        has_binary_weights_flag = metadata.get('_has_binary_weights')
        has_binary_chunk_flag = metadata.get('_has_binary_chunk')
        has_arch_data_size = metadata.get('_architecture_data_size') is not None
        is_binary_protocol = (
            has_binary_flag or
            has_binary_weights_flag or
            has_binary_chunk_flag or
            has_arch_data_size
        ) and (is_register_model or is_register_chunk)
        
        # Always log for REGISTER_MODEL to debug
        if is_register_model:
            logger.info(
                f"🔍 REGISTER_MODEL deserialization: "
                f"request_type={request_type}, "
                f"_binary_protocol={has_binary_flag}, "
                f"_has_binary_weights={has_binary_weights_flag}, "
                f"_has_binary_chunk={has_binary_chunk_flag}, "
                f"_architecture_data_size={metadata.get('_architecture_data_size')}, "
                f"is_binary_protocol={is_binary_protocol}, "
                f"remaining_bytes={len(data)-offset}"
            )

        tensor_data = {}
        has_serialized_inputs = (
            is_execute_model and
            'inputs' in metadata and
            isinstance(metadata['inputs'], dict) and
            metadata['inputs'].get('_is_serialized_dict')
        )
        binary_entries_required = any(
            isinstance(entry, dict) and (
                entry.get('_is_serialized_dict') or entry.get('_is_binary') or entry.get('_is_tensor')
            )
            for entry in metadata.values()
        )

        # Parse appended binary data for serialized tensors (inputs or results)
        if not is_binary_protocol and offset < len(data):
            if not binary_entries_required:
                raise ValueError(
                    f"Unexpected binary payload (no metadata declares serialized tensors). "
                    f"metadata keys: {list(metadata.keys())}"
                )
            tensor_count = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            for i in range(tensor_count):
                if len(data) < offset + 4:
                    raise ValueError(f"Invalid tensor payload: missing key length for tensor {i}")
                key_len = struct.unpack('>I', data[offset:offset+4])[0]
                offset += 4

                if len(data) < offset + key_len:
                    raise ValueError(f"Invalid tensor payload: missing key for tensor {i}")
                key = data[offset:offset+key_len].decode('utf-8')
                offset += key_len

                if len(data) < offset + 8:
                    raise ValueError(f"Invalid tensor payload: missing data length for tensor {i}")
                data_len = struct.unpack('>Q', data[offset:offset+8])[0]
                offset += 8

                if len(data) < offset + data_len:
                    raise ValueError(f"Invalid tensor payload: missing data for tensor {i}")
                tensor_bytes = data[offset:offset+data_len]
                offset += data_len

                tensor_data[key] = tensor_bytes

        if is_binary_protocol:
            # Binary protocol: read architecture_data size, then architecture_data, then binary data
            if len(data) < offset + 8:
                raise ValueError("Invalid binary protocol data: missing architecture_data size")

            arch_data_size = struct.unpack('>Q', data[offset:offset+8])[0]
            offset += 8

            architecture_data = None
            if arch_data_size > 0:
                if len(data) < offset + arch_data_size:
                    raise ValueError(f"Invalid binary protocol data: missing architecture_data ({arch_data_size} bytes)")
                architecture_data = data[offset:offset+arch_data_size]
                offset += arch_data_size

            # Remaining data is binary
            remaining_data = data[offset:]
            logger.info(f"🔍 Binary weights/chunk data: {len(remaining_data)} bytes")
            request = metadata.copy()

            # Set the appropriate binary field based on request type
            if is_register_model:
                request['weights_binary'] = remaining_data
                logger.info(f"✅ Set weights_binary in request: {len(remaining_data)} bytes, model_id={request.get('model_id', 'N/A')[:30]}")
            elif is_register_chunk:
                request['chunk_data_binary'] = remaining_data

            if architecture_data:
                request['architecture_data'] = architecture_data

            return request

        elif has_serialized_inputs:
            # EXECUTE_MODEL with serialized inputs - extract tensor data from binary section
            logger.debug(f"Deserializing EXECUTE_MODEL request with serialized inputs")

            # Reconstruct the inputs dict with deserialized tensors
            inputs_metadata = metadata['inputs']
            reconstructed_inputs = {}
            tensor_keys = inputs_metadata['keys']
            tensor_metadata = inputs_metadata['tensors']

            for key in tensor_keys:
                if key in tensor_data:
                    reconstructed_inputs[key] = {
                        'data': tensor_data[key],
                        'shape': tensor_metadata[key]['shape'],
                        'dtype': tensor_metadata[key]['dtype'],
                        'format': 'numpy_binary'  # Assume numpy_binary format
                    }

            request = metadata.copy()
            request['inputs'] = reconstructed_inputs
        else:
            request = metadata.copy()
            if offset < len(data):
                if not binary_entries_required:
                    raise ValueError(
                        f"Unsupported request format. Only binary protocol is supported in Phase 3. "
                        f"Legacy dict-based and pickle formats have been removed. "
                        f"Request type: {request_type}, "
                        f"metadata keys: {list(metadata.keys())}, "
                        f"unexpected data: {len(data) - offset} bytes after JSON"
                    )

        # Reconstruct any generic serialized entries (e.g., results)
        if tensor_data:
            for key, entry in metadata.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get('_is_tensor') and key in tensor_data:
                    # Reconstruct tensor from binary data
                    shape = entry.get('shape', [])
                    dtype_str = entry.get('dtype', 'float32')
                    tensor_bytes = tensor_data[key]
                    # Convert dtype string to torch dtype
                    dtype_map = {
                        'torch.float32': torch.float32,
                        'torch.float16': torch.float16,
                        'torch.bfloat16': torch.bfloat16,
                        'torch.int64': torch.int64,
                        'torch.int32': torch.int32,
                        'torch.bool': torch.bool,
                        'float32': torch.float32,
                        'float16': torch.float16,
                        'bfloat16': torch.bfloat16,
                        'int64': torch.int64,
                        'int32': torch.int32,
                        'bool': torch.bool,
                    }
                    dtype = dtype_map.get(dtype_str, torch.float32)
                    # Reconstruct tensor from bytes
                    import numpy as np
                    np_dtype_map = {
                        torch.float32: np.float32,
                        torch.float16: np.float16,
                        torch.bfloat16: np.float16,  # bfloat16 not directly supported
                        torch.int64: np.int64,
                        torch.int32: np.int32,
                        torch.bool: np.bool_,
                    }
                    np_dtype = np_dtype_map.get(dtype, np.float32)
                    np_array = np.frombuffer(tensor_bytes, dtype=np_dtype)
                    if shape:
                        np_array = np_array.reshape(shape)
                    tensor = torch.from_numpy(np_array.copy())
                    request[key] = tensor
                elif entry.get('_is_binary') and key in tensor_data:
                    request[key] = {
                        'data': tensor_data[key],
                        'shape': entry.get('shape', entry.get('tensors', {}).get(key, {}).get('shape', [])),
                        'dtype': entry.get('dtype', entry.get('tensors', {}).get(key, {}).get('dtype', 'float32')),
                        'format': entry.get('format', 'numpy_binary')
                    }
                elif entry.get('_is_serialized_dict') and key not in ('inputs',):
                    sub_entries = {}
                    tensor_keys = entry.get('keys', [])
                    tensor_metadata = entry.get('tensors', {})
                    for tensor_key in tensor_keys:
                        if tensor_key in tensor_data:
                            sub_entries[tensor_key] = {
                                'data': tensor_data[tensor_key],
                                'shape': tensor_metadata.get(tensor_key, {}).get('shape', []),
                                'dtype': tensor_metadata.get(tensor_key, {}).get('dtype', 'float32'),
                                'format': entry.get('format', 'numpy_binary')
                            }
                    if sub_entries:
                        request[key] = sub_entries

        return request

    @staticmethod
    def serialize_response(response: Dict[str, Any]) -> bytes:
        """Serialize response (same format as request)."""
        return SecureSerializer.serialize_request(response)
    
    @staticmethod
    def deserialize_response(data: bytes) -> Dict[str, Any]:
        """Deserialize response (same format as request)."""
        return SecureSerializer.deserialize_request(data)

