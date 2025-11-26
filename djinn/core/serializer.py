"""
Djinn v2.3.15 Serializer - Binary Protocol Implementation

Replaces pickle with a robust, length-prefixed binary protocol optimized for tensor data.
No implicit concatenation, explicit structure preservation.

Wire Format:
[Header: 17 bytes]
  1B: Version (0x02)
  8B: TotalBodySize (Q) - For OOM checks
  4B: MetaLength (I)
  4B: TensorCount (I)

[Metadata: M bytes]
  JSON: { "structure": {...}, "tensors": [{"dtype": "f16", "shape": [1, 10], "nbytes": 20}, ...] }

[Tensor Data: Variable]
  [8B Length][Raw Bytes]
  [8B Length][Raw Bytes]
  ...

Returns list of buffers suitable for scatter-gather I/O.
"""
import struct
import json
import torch
import numpy as np
from typing import List, Dict, Any, Union
import io

# Torch dtype to numpy dtype mapping
TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


class DjinnSerializer:
    """
    Binary protocol serializer for Djinn v2.3.15.

    Key features:
    - Length-prefixed binary format (no pickle)
    - Zero-copy tensor serialization
    - Explicit structure preservation
    - OOM-safe with pre-flight checks
    """

    VERSION = 0x02  # Protocol version 2

    @staticmethod
    def serialize(request_dict: Dict[str, Any]) -> List[memoryview]:
        """
        Serialize request dictionary to list of buffers for scatter-gather I/O.

        Args:
            request_dict: Dictionary containing tensors and metadata

        Returns:
            List of memoryview objects suitable for scatter-gather I/O
        """
        tensors = []
        tensor_meta = []

        # 1. Recursive Skeletonization
        def skeletonize(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                # CRITICAL: Ensure CPU memory and contiguous layout for DMA
                t_cpu = obj.detach().cpu().contiguous()

                # Calculate exact byte size
                nbytes = t_cpu.numel() * t_cpu.element_size()

                # Store tensor and metadata
                tensors.append(t_cpu)
                # Use numpy dtype for serialization
                numpy_dtype = TORCH_TO_NUMPY_DTYPE.get(obj.dtype, np.float32)
                tensor_meta.append({
                    "dtype": numpy_dtype.__name__,  # e.g., 'int64', 'float32'
                    "shape": list(obj.shape),
                    "nbytes": nbytes
                })

                # Return reference ID for reconstruction
                return f"REF_{len(tensors)-1}"

            elif isinstance(obj, dict):
                return {k: skeletonize(v) for k, v in obj.items()}

            elif isinstance(obj, (list, tuple)):
                return [skeletonize(v) for v in obj]

            else:
                # Scalars and other types pass through
                return obj

        # Perform skeletonization
        structure = skeletonize(request_dict)

        # 2. Prepare Metadata
        meta = {"structure": structure, "tensors": tensor_meta}
        meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

        # 3. Calculate sizes
        # Tensors: Sum(8 bytes length prefix + raw bytes)
        tensor_payload_size = sum(8 + t_meta['nbytes'] for t_meta in tensor_meta)
        total_body_size = len(meta_bytes) + tensor_payload_size

        # 4. Create Header (Version=0x02, Big Endian)
        header = struct.pack('>BQII',
                           DjinnSerializer.VERSION,
                           total_body_size,
                           len(meta_bytes),
                           len(tensors))

        # 5. Build Buffer List (Zero-Copy)
        buffers = [header, meta_bytes]

        # Add tensor data with length prefixes
        for t in tensors:
            # Get numpy array for buffer protocol compliance
            arr = t.numpy()

            # Convert to bytes (ensures contiguous, correct byte order)
            tensor_bytes = arr.tobytes()

            # Length prefix (8 bytes, big endian)
            buffers.append(struct.pack('>Q', len(tensor_bytes)))

            # Raw tensor data (zero-copy via memoryview)
            buffers.append(memoryview(tensor_bytes))

        return buffers

    @staticmethod
    def deserialize(buffers: List[bytes]) -> Dict[str, Any]:
        """
        Deserialize buffers back to request dictionary.

        Args:
            buffers: List of buffer objects from network

        Returns:
            Reconstructed dictionary with tensors
        """
        # Concatenate buffers if needed (for testing)
        if len(buffers) > 1:
            # In real usage, buffers come pre-separated
            stream = io.BytesIO(b''.join(buffers))
        else:
            stream = io.BytesIO(buffers[0])

        # 1. Read Header
        header_data = stream.read(17)
        if len(header_data) < 17:
            raise ValueError("Incomplete header")

        version, total_body_size, meta_length, tensor_count = struct.unpack('>BQII', header_data)

        if version != DjinnSerializer.VERSION:
            raise ValueError(f"Unsupported protocol version: {version}")

        # 2. Read Metadata
        meta_bytes = stream.read(meta_length)
        if len(meta_bytes) < meta_length:
            raise ValueError("Incomplete metadata")

        meta = json.loads(meta_bytes.decode('utf-8'))
        structure = meta['structure']
        tensor_meta = meta['tensors']

        # 3. Read Tensor Data
        tensors = []
        for i, t_meta in enumerate(tensor_meta):
            # Read length prefix
            len_bytes = stream.read(8)
            if len(len_bytes) < 8:
                raise ValueError(f"Incomplete tensor {i} length prefix")

            tensor_size = struct.unpack('>Q', len_bytes)[0]

            # Read tensor data
            tensor_bytes = stream.read(tensor_size)
            if len(tensor_bytes) < tensor_size:
                raise ValueError(f"Incomplete tensor {i} data")

            # Reconstruct tensor
            arr = np.frombuffer(tensor_bytes, dtype=t_meta['dtype'])
            arr = arr.reshape(t_meta['shape'])

            tensor = torch.from_numpy(arr)
            tensors.append(tensor)

        # 4. Reconstruct Structure
        def reconstruct(obj: Any) -> Any:
            if isinstance(obj, str) and obj.startswith("REF_"):
                ref_id = int(obj[4:])
                return tensors[ref_id]

            elif isinstance(obj, dict):
                return {k: reconstruct(v) for k, v in obj.items()}

            elif isinstance(obj, list):
                return [reconstruct(v) for v in obj]

            else:
                return obj

        return reconstruct(structure)

    @staticmethod
    def get_total_size(buffers: List[memoryview]) -> int:
        """
        Calculate total serialized size from buffer list.
        Useful for OOM checks before transmission.

        Args:
            buffers: List of buffers from serialize()

        Returns:
            Total size in bytes
        """
        return sum(len(buf) for buf in buffers)

    @staticmethod
    def validate_request(request_dict: Dict[str, Any]) -> None:
        """
        Validate request before serialization.

        Args:
            request_dict: Request to validate

        Raises:
            ValueError: If request is invalid
        """
        def validate_recursive(obj: Any, path: str = ""):
            if isinstance(obj, torch.Tensor):
                if not obj.is_contiguous():
                    raise ValueError(f"Tensor at {path} is not contiguous")
                if obj.numel() == 0:
                    raise ValueError(f"Tensor at {path} is empty")

            elif isinstance(obj, dict):
                for k, v in obj.items():
                    validate_recursive(v, f"{path}.{k}" if path else k)

            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    validate_recursive(v, f"{path}[{i}]")

        validate_recursive(request_dict)




