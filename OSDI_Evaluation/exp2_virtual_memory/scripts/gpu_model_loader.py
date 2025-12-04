"""
GPU-Resident Model Loader with Ring Buffer Views

Implements the redesigned architecture where:
1. Model parameters ARE ring buffer views from initialization
2. Eliminates device mismatch by keeping all computation on GPU
3. Enables async weight streaming into ring buffer (H2D transfer)

Key insight: Instead of loading model to CPU then swapping pointers,
we load the model structure (skeleton) and replace meta tensors
with ring buffer views directly.
"""

import logging
import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext

logger = logging.getLogger(__name__)


def load_model_skeleton_to_meta(model_id: str, dtype: torch.dtype = torch.float16):
    """
    Load model structure to 'meta' device (zero memory).
    
    Uses HuggingFace's meta device to create model structure without
    allocating actual tensor memory.
    
    Args:
        model_id: Model identifier (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')
        dtype: Data type for parameters
        
    Returns:
        Model with parameters on 'meta' device (no actual memory allocated)
    """
    logger.info(f"Loading model skeleton to 'meta' device: {model_id}")
    
    try:
        # Try using init_empty_weights from accelerate
        from accelerate import init_empty_weights
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
    except ImportError:
        # Fallback: manually create on meta device
        logger.warning("accelerate not available, using manual meta device loading")
        with torch.device('meta'):
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
    
    logger.info(f"✅ Model skeleton loaded to meta device: {model.num_parameters()/1e9:.1f}B params")
    return model


def load_actual_weights_from_checkpoint(model_id: str, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
    """
    Load actual weights from model checkpoint (CPU memory).
    
    Loads the full model to CPU temporarily to extract weights, then immediately
    allows it to be garbage collected.
    
    Args:
        model_id: Model identifier
        dtype: Data type
        
    Returns:
        Dict mapping parameter names to CPU tensors
    """
    logger.info(f"Loading actual weights from checkpoint: {model_id}")
    
    # Load full model to CPU to get weights
    model_full = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu"
    )
    model_full.eval()
    
    # Extract all parameters and buffers
    weights_dict = {}
    for name, param in model_full.named_parameters():
        # Store on CPU as float32 or specified dtype
        weights_dict[name] = param.detach().cpu()
    
    for name, buffer in model_full.named_buffers():
        weights_dict[name] = buffer.detach().cpu()
    
    logger.info(f"✅ Loaded {len(weights_dict)} weight tensors from checkpoint")
    
    # Delete model to free CPU memory
    del model_full
    import gc
    gc.collect()
    
    return weights_dict


def collect_all_parameters_info(model: nn.Module) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
    """
    Collect information about all parameters and buffers that need allocation.
    
    Returns:
        Dict mapping parameter name -> (shape, dtype)
    """
    params_info = {}
    
    # Collect all parameters
    for name, param in model.named_parameters():
        params_info[name] = (param.shape, param.dtype)
    
    # Collect all buffers (e.g., layer norm weights)
    for name, buffer in model.named_buffers():
        params_info[name] = (buffer.shape, buffer.dtype)
    
    logger.info(f"Collected {len(params_info)} parameters/buffers for allocation")
    return params_info


def allocate_parameters_in_ring_buffer(
    ring_buffer,
    params_info: Dict[str, Tuple[torch.Size, torch.dtype]],
    device: torch.device
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Size, torch.dtype]]]:
    """
    Allocate space in ring buffer for as many parameters as fit.

    For virtualization, we allocate what fits in the ring buffer and leave
    the rest for on-demand streaming.

    Args:
        ring_buffer: WeightRingBuffer instance
        params_info: Parameter information from collect_all_parameters_info
        device: Target device (GPU)

    Returns:
        Tuple of:
        - Dict mapping parameter name -> ring buffer view tensor (allocated)
        - Dict mapping parameter name -> (shape, dtype) (not allocated, for streaming)
    """
    logger.info(f"Allocating parameters in {ring_buffer.capacity_bytes / 1024**3:.1f}GB ring buffer")

    param_views = {}
    remaining_params = {}
    total_allocated = 0

    # Sort parameters by size (largest first) for better packing
    sorted_params = sorted(params_info.items(), key=lambda x: math.prod(x[1][0]), reverse=True)

    for param_name, (shape, dtype) in sorted_params:
        try:
            # Try to allocate view in ring buffer
            view = ring_buffer.allocate_view(
                param_name=param_name,
                shape=shape,
                dtype=dtype,
                device=device
            )
            param_views[param_name] = view

            # Track allocation
            num_elements = math.prod(shape)
            bytes_allocated = num_elements * torch.tensor([], dtype=dtype).element_size()
            total_allocated += bytes_allocated

        except RuntimeError as e:
            if "Ring buffer full" in str(e):
                # Ring buffer is full, add to remaining for streaming
                remaining_params[param_name] = (shape, dtype)
                logger.debug(f"Ring buffer full, {param_name} will be streamed on-demand")
            else:
                logger.error(f"Failed to allocate {param_name}: {e}")
                raise

    allocated_count = len(param_views)
    remaining_count = len(remaining_params)

    logger.info(f"✅ Allocated {allocated_count} parameters ({total_allocated / 1024**3:.2f}GB)")
    logger.info(f"📤 {remaining_count} parameters will be streamed on-demand")

    if remaining_count > 0:
        logger.info(f"   Ring buffer utilization: {total_allocated / ring_buffer.capacity_bytes * 100:.1f}%")
        logger.info(f"   Model virtualization: {(remaining_count / (allocated_count + remaining_count)) * 100:.1f}% of parameters streamed")

    return param_views, remaining_params


def replace_meta_tensors_with_views(
    model: nn.Module,
    param_views: Dict[str, torch.Tensor]
) -> None:
    """
    Replace meta device tensors with ring buffer views.
    
    This converts the model from having 'meta' device parameters to having
    GPU-resident parameters (via ring buffer views).
    
    Args:
        model: Model with meta tensors
        param_views: Dict of ring buffer views from allocate_parameters_in_ring_buffer
    """
    logger.info("Replacing meta tensors with ring buffer views...")
    
    replaced_count = 0
    
    # Replace parameters
    for name, param in model.named_parameters():
        if name in param_views:
            view = param_views[name]
            # Replace the parameter with the ring buffer view
            # This makes PyTorch track it as a GPU tensor
            set_module_tensor_to_device(model, name, view.device, tensor=view)
            replaced_count += 1
    
    # Replace buffers
    for name, buffer in model.named_buffers():
        if name in param_views:
            view = param_views[name]
            set_module_tensor_to_device(model, name, view.device, tensor=view)
            replaced_count += 1
    
    logger.info(f"✅ Replaced {replaced_count} tensors with ring buffer views")


def initialize_weights_in_ring_buffer(
    ring_buffer,
    param_views: Dict[str, torch.Tensor],
    weights_dict: Dict[str, torch.Tensor],
    device: torch.device,
    model_id: str = "default"
) -> None:
    """
    Initialize allocated ring buffer views with actual weights from checkpoint.

    Only initializes parameters that were successfully allocated to the ring buffer.
    Remaining parameters will be streamed on-demand during inference.

    Args:
        ring_buffer: WeightRingBuffer instance (for tracking)
        param_views: Ring buffer views that were allocated (GPU tensors)
        weights_dict: Actual weights from checkpoint (CPU tensors)
        device: GPU device
        model_id: Model ID in ring buffer
    """
    logger.info("Initializing allocated ring buffer views with weights...")

    import time
    start_time = time.perf_counter()

    initialized = 0
    total_bytes = 0

    for name, view in param_views.items():
        weight_source = name

        # Handle tied weights (e.g., lm_head.weight -> transformer.wte.weight in GPT-2)
        if name not in weights_dict:
            # Try common weight sharing patterns
            if name == "lm_head.weight" and "transformer.wte.weight" in weights_dict:
                weight_source = "transformer.wte.weight"
                logger.debug(f"Using {weight_source} for {name} (tied weights)")
            else:
                logger.debug(f"Weight {name} not found in checkpoint, skipping")
                continue

        weight = weights_dict[weight_source]

        # Ensure shapes match
        if view.shape != weight.shape:
            logger.error(f"Shape mismatch for {name}: view {view.shape} vs weight {weight.shape}")
            continue

        # Ensure dtypes match
        if view.dtype != weight.dtype:
            weight = weight.to(view.dtype)

        try:
            # Copy to GPU view (this performs H2D transfer)
            # Pin memory for faster transfer if on CPU
            if weight.device.type == 'cpu' and device.type == 'cuda':
                # Pin the tensor if not already pinned for faster H2D transfer
                if not weight.is_pinned():
                    weight = weight.pin_memory()

            # Synchronous copy to ensure data is available
            view.copy_(weight, non_blocking=False)

            total_bytes += weight.numel() * weight.element_size()
            initialized += 1

        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            continue

    # Synchronize to ensure all H2D transfers complete
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time
    bandwidth_gbps = (total_bytes / 1024**3) / elapsed if elapsed > 0 else 0

    logger.info(f"✅ Initialized {initialized} resident weights ({total_bytes / 1024**3:.2f}GB in {elapsed:.2f}s)")
    logger.info(f"   H2D bandwidth: {bandwidth_gbps:.1f}GB/s")


def set_module_tensor_to_device(model: nn.Module, tensor_name: str, device: torch.device, tensor: torch.Tensor = None):
    """
    Set a tensor in a module to a specific device.
    
    Args:
        model: Model containing the tensor
        tensor_name: Full tensor name (e.g., 'layer.weight')
        device: Target device
        tensor: Optional tensor to set (otherwise moves existing tensor)
    """
    # Split name into module path and tensor name
    parts = tensor_name.rsplit('.', 1)
    
    if len(parts) == 1:
        # Top-level parameter
        module = model
        attr_name = parts[0]
    else:
        # Navigate to parent module
        module_path, attr_name = parts
        module = model
        for part in module_path.split('.'):
            module = getattr(module, part)
    
    # Set the tensor
    if tensor is not None:
        setattr(module, attr_name, nn.Parameter(tensor, requires_grad=False) if isinstance(getattr(module, attr_name), nn.Parameter) else tensor)
    else:
        existing = getattr(module, attr_name)
        if isinstance(existing, nn.Parameter):
            setattr(module, attr_name, nn.Parameter(existing.to(device), requires_grad=False))
        else:
            setattr(module, attr_name, existing.to(device))


def load_model_with_ring_buffer(
    model_id: str,
    ring_buffer,
    device: torch.device,
    dtype: str = "float16"
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, Tuple[torch.Size, torch.dtype]]]:
    """
    Load model with parameters as ring buffer views (with virtualization).

    This is the main entry point for the redesigned architecture with virtualization support.

    Flow:
    1. Load model skeleton to 'meta' device (0 memory)
    2. Collect parameter information
    3. Allocate ring buffer space for as many parameters as fit
    4. Replace meta tensors with ring buffer views (resident parameters)
    5. Load actual weights from checkpoint to CPU
    6. Stream weights CPU→GPU into resident ring buffer views
    7. Register model with ring buffer for hook tracking
    8. Model is GPU-resident with resident weights initialized, others streamed on-demand

    Args:
        model_id: Model identifier
        ring_buffer: WeightRingBuffer instance
        device: Target GPU device
        dtype: Data type ('float16', 'float32', etc.)

    Returns:
        (model, param_views, remaining_params) where:
        - model: GPU-resident model with ring buffer parameters
        - param_views: Dict of allocated resident views
        - remaining_params: Dict of parameters that will be streamed on-demand
    """
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    logger.info("=" * 80)
    logger.info("Loading Model with GPU Ring Buffer Architecture (Virtualization)")
    logger.info("=" * 80)

    # Step 1: Load skeleton
    model = load_model_skeleton_to_meta(model_id, dtype=torch_dtype)

    # Step 2: Collect parameter info
    params_info = collect_all_parameters_info(model)

    # Step 3: Allocate what fits in ring buffer
    param_views, remaining_params = allocate_parameters_in_ring_buffer(ring_buffer, params_info, device)

    # Step 4: Replace meta tensors with resident ring buffer views
    replace_meta_tensors_with_views(model, param_views)

    # Step 5: Load actual weights from checkpoint
    logger.info("\nStep 5: Loading actual weights from checkpoint...")
    weights_dict = load_actual_weights_from_checkpoint(model_id, dtype=torch_dtype)

    # Step 6: Stream weights to resident GPU ring buffer views
    logger.info("Step 6: Initializing resident weights in ring buffer...")
    initialize_weights_in_ring_buffer(ring_buffer, param_views, weights_dict, device, model_id="default")

    # Step 7: Register model with ring buffer for hook tracking
    logger.info("Step 7: Registering model with ring buffer...")
    try:
        ring_buffer.register_model(model_id="default", state_dict=weights_dict)
        logger.debug("✅ Model registered with ring buffer")
    except Exception as e:
        logger.debug(f"Model registration note: {e}")

    # Clean up CPU weights (keep for potential future streaming)
    # Note: In full implementation, we'd keep weights_dict for on-demand streaming
    # del weights_dict
    # import gc
    # gc.collect()

    # Verify model is on GPU
    first_param = next(model.parameters())
    resident_params = len(param_views)
    streaming_params = len(remaining_params)
    total_params = resident_params + streaming_params

    logger.info(f"\n✅ Model loaded with virtualization:")
    logger.info(f"   Device: {first_param.device}, dtype: {first_param.dtype}")
    logger.info(f"   Resident parameters: {resident_params}/{total_params} ({resident_params/total_params*100:.1f}%)")
    logger.info(f"   Streaming parameters: {streaming_params}/{total_params} ({streaming_params/total_params*100:.1f}%)")

    logger.info("=" * 80)

    return model, param_views, remaining_params

