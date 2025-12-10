#!/usr/bin/env python3
"""
Djinn Ring Buffer Measurement: Direct Integration

This script directly uses Djinn's ring buffer for weight streaming.
This is the ACTUAL Djinn measurement, not a proxy.

Architecture:
1. Load model checkpoint to CPU
2. Initialize WeightRingBuffer with 20GB capacity
3. Register model in ring buffer (triggers skip-end allocation)
4. Run inference with hooks that manage weight streaming
5. Measure TTFT/Decode/E2E

Expected performance on L4 (24GB) with Llama-2-13B (26GB):
- TTFT: ~1.1s (async streaming of 6GB delta)
- Decode: ~705ms/token (streaming per token)
- E2E: ~35s (TTFT + 50 tokens)

This demonstrates the ring buffer's ability to stream weights
while keeping 20GB resident, reducing effective model reload time.
"""

import argparse
import json
import logging
import time
import torch
from pathlib import Path
from typing import Optional, List
import sys

# Import measurement protocol
sys.path.insert(0, str(Path(__file__).parent))
from measurement_protocol import (
    MeasurementProtocol,
    ExperimentConfig,
    create_standard_config
)

# Import Djinn components
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DjinnRingBufferMeasurement(MeasurementProtocol):
    """
    Direct measurement of Djinn's ring buffer for weight streaming.
    
    This measures the actual ring buffer performance, not a proxy.
    The ring buffer is integrated into the inference path via hooks.
    """
    
    def __init__(self, model_id: str, config: ExperimentConfig, ring_buffer_gb: float = 20.0):
        """
        Initialize Djinn ring buffer measurement.
        
        Args:
            model_id: HuggingFace model ID
            config: Experiment configuration
            ring_buffer_gb: Ring buffer capacity in GB
        """
        super().__init__(model_id, config)
        self.ring_buffer_gb = ring_buffer_gb
        self.model_device = None
        self.ring_buffer = None
        self.model_registration = None
        self.streamer = None
        self.hook_manager = None
    
    def load_model(self):
        """Load model and initialize ring buffer."""
        logger.info("=" * 70)
        logger.info("Djinn Ring Buffer Measurement")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_id}")
        logger.info(f"Ring buffer capacity: {self.ring_buffer_gb}GB")
        logger.info("")
        
        load_start = time.time()
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            local_files_only=False,
            resume_download=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("✅ Tokenizer loaded")
        
        # Load model to CPU
        logger.info("Loading model to CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=False,
            resume_download=True,
        )
        logger.info("✅ Model loaded to CPU")
        
        # Initialize Djinn Ring Buffer
        logger.info("Initializing Djinn Ring Buffer...")
        try:
            from djinn.backend.runtime.ring_buffer import WeightRingBuffer
            from djinn.backend.runtime.weight_streamer import WeightStreamer
            from djinn.backend.runtime.weight_hooks import install_ring_buffer_hooks
            
            # Create ring buffer with specified capacity
            ring_buffer_bytes = int(self.ring_buffer_gb * 1024**3)
            self.ring_buffer = WeightRingBuffer(
                capacity_bytes=ring_buffer_bytes,
                device='cuda:0'
            )
            logger.info(f"  Ring buffer created: {self.ring_buffer_gb}GB on CUDA")
            
            # Register model in ring buffer
            logger.info("Registering model in ring buffer...")
            
            # Get state dict and layer order
            state_dict = self.model.state_dict()
            layer_names = [name for name in state_dict.keys()]
            
            # Register model (triggers skip-end allocation and weight copying)
            self.model_registration = self.ring_buffer.register_model(
                model_id=self.model_id,
                state_dict=state_dict,
                layer_order=layer_names
            )
            logger.info(f"✅ Model registered in ring buffer")
            logger.info(f"  Total weights in buffer: {self.model_registration.total_bytes / (1024**3):.1f}GB")
            
            # Replace model weights with appropriate tensors:
            # - Resident weights: ring buffer views (already on GPU)
            # - Streamed weights: tiny GPU placeholders (hooks will redirect to slots)
            logger.info("Replacing model weights with ring buffer views and placeholders...")
            resident_replaced = 0
            streamed_replaced = 0
            for name, param in self.model.named_parameters():
                if name in self.model_registration.layer_allocations:
                    allocation = self.model_registration.layer_allocations[name]
                    
                    if allocation.is_resident:
                        # Resident: point directly to ring buffer (already on GPU)
                        ring_view = self.ring_buffer.buffer[
                            allocation.offset : allocation.end_offset()
                        ].view(allocation.dtype).view(allocation.shape)
                        param.data = ring_view
                        resident_replaced += 1
                    else:
                        # Streamed: create tiny GPU placeholder
                        # Actual data is in host_weights (pinned CPU memory)
                        # Hooks will redirect to streaming slots during forward pass
                        placeholder = torch.empty(1, dtype=param.dtype, device='cuda:0')
                        param.data = placeholder
                        streamed_replaced += 1
            
            logger.info(f"✅ Replaced {resident_replaced} resident weights with ring buffer views")
            logger.info(f"✅ Replaced {streamed_replaced} streamed weights with GPU placeholders")
            
            # Initialize weight streamer for async pipelining
            logger.info("Initializing weight streamer...")
            self.streamer = WeightStreamer(
                ring_buffer=self.ring_buffer,
                device=torch.device('cuda:0'),
                prefetch_queue_size=16,
                chunk_size_bytes=64 * 1024 * 1024  # 64MB chunks
            )
            self.streamer.start()
            logger.info("✅ Weight streamer started")
            
        except ImportError as e:
            logger.error(f"❌ Djinn ring buffer not available: {e}")
            logger.error("Cannot run ring buffer measurement without Djinn runtime")
            raise RuntimeError("Djinn ring buffer required for this measurement")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ring buffer: {e}")
            raise
        
        # Move model to GPU
        # This works now because:
        # - Resident weights already point to ring buffer views (on GPU) - no allocation
        # - Streamed weights are tiny placeholders (1 element each) - minimal allocation
        # - Actual streamed data is in pinned host memory, will be streamed to slots
        logger.info("Moving model to GPU...")
        self.model = self.model.to('cuda:0')
        logger.info("✅ Model on GPU")
        
        # Install ring buffer hooks to redirect weights during forward pass
        logger.info("Installing ring buffer hooks...")
        self.hook_manager = install_ring_buffer_hooks(
            model=self.model,
            ring_buffer=self.ring_buffer,
            model_id=self.model_id,
            streamer=self.streamer,
            layer_names=None  # Let hook manager extract module names
        )
        logger.info(f"✅ Hooks installed")
        
        self.model_size_gb = self.get_model_size_gb()
        self.load_time_ms = (time.time() - load_start) * 1000
        self.model_device = 'cuda:0'
        
        logger.info(f"✅ Ring buffer setup complete")
        logger.info(f"  Model size: {self.model_size_gb:.1f}GB")
        logger.info(f"  Resident fraction: {(self.ring_buffer_gb / self.model_size_gb * 100):.1f}%")
        logger.info(f"  Streaming delta: {(self.model_size_gb - self.ring_buffer_gb):.1f}GB")
        logger.info(f"  Load time: {self.load_time_ms:.1f}ms")
        
        # Log block-level information if available
        if self.model_registration.blocks:
            resident_blocks = [b for b in self.model_registration.blocks if b.is_resident]
            streamed_blocks = [b for b in self.model_registration.blocks if not b.is_resident]
            logger.info(f"  Block-granularity streaming:")
            logger.info(f"    Total blocks: {len(self.model_registration.blocks)}")
            logger.info(f"    Resident blocks: {len(resident_blocks)}")
            logger.info(f"    Streamed blocks: {len(streamed_blocks)}")
            logger.info(f"    Streaming slots: {len(self.model_registration.streaming_slots)}")
        logger.info("")
    
    def get_model_size_gb(self) -> float:
        """Calculate total model size in GB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        # float16 = 2 bytes per parameter
        return (total_params * 2) / (1024**3)
    
    def cleanup(self):
        """Clean up resources."""
        if self.streamer is not None:
            logger.info("Stopping weight streamer...")
            self.streamer.stop()
        if self.hook_manager is not None:
            logger.info("Removing hooks...")
            self.hook_manager.remove_hooks()
    
    def run_inference(self, max_new_tokens: int = 1) -> float:
        """
        Run inference with ring buffer and return elapsed time.
        
        The ring buffer automatically streams weights as needed during
        the forward pass via hooks.
        
        Args:
            max_new_tokens: Number of tokens to generate
            
        Returns:
            elapsed_ms: Execution time in milliseconds
        """
        # Prepare input
        input_ids = self.tokenizer.encode(
            self.prompt,
            return_tensors="pt"
        )
        
        # Pad to prompt_tokens length
        if input_ids.shape[1] < self.config.prompt_tokens:
            padding_length = self.config.prompt_tokens - input_ids.shape[1]
            padding = torch.full(
                (1, padding_length),
                self.tokenizer.pad_token_id,
                dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :self.config.prompt_tokens]
        
        # Move to model device (GPU)
        input_ids = input_ids.to(self.model_device)
        
        # Synchronize before measurement
        torch.cuda.synchronize()
        
        # Measure inference time
        start = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=min(max_new_tokens, 50),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Synchronize to ensure completion
        torch.cuda.synchronize()
        
        elapsed = time.time() - start
        return elapsed * 1000  # Convert to milliseconds


def save_results(
    ttft_results, decode_results, e2e_results, output_file: str, ring_buffer_gb: float
):
    """Save all measurement results to JSON."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_size_gb = ttft_results.model_size_gb
    streaming_delta_gb = model_size_gb - ring_buffer_gb
    resident_fraction = ring_buffer_gb / model_size_gb if model_size_gb > 0 else 0
    
    # Compile all results
    results = {
        "baseline": "djinn_ring_buffer",
        "model": ttft_results.model_id,
        "model_size_gb": model_size_gb,
        "ring_buffer_gb": ring_buffer_gb,
        "resident_fraction": resident_fraction,
        "streaming_delta_gb": streaming_delta_gb,
        "load_time_ms": ttft_results.load_time_ms,
        "measurements": {
            "ttft": {
                "individual_runs_ms": [r.latency_ms for r in ttft_results.ttft_results],
                "average_ms": ttft_results.ttft_avg_ms,
                "min_ms": ttft_results.ttft_min_ms,
                "max_ms": ttft_results.ttft_max_ms,
                "stddev_ms": ttft_results.ttft_stddev_ms,
            },
            "decode": {
                "individual_runs_ms": [r.latency_ms for r in decode_results.decode_results],
                "average_ms_per_token": decode_results.decode_avg_ms,
                "min_ms_per_token": decode_results.decode_min_ms,
                "max_ms_per_token": decode_results.decode_max_ms,
                "total_tokens_generated": decode_results.config.generated_tokens,
            },
            "e2e": {
                "individual_runs_ms": [r.latency_ms for r in e2e_results.e2e_results],
                "average_ms": e2e_results.e2e_avg_ms,
                "min_ms": e2e_results.e2e_min_ms,
                "max_ms": e2e_results.e2e_max_ms,
                "total_tokens": e2e_results.config.prompt_tokens + e2e_results.config.generated_tokens,
                "avg_tokens_per_second": (e2e_results.config.prompt_tokens + e2e_results.config.generated_tokens) * 1000 / e2e_results.e2e_avg_ms if e2e_results.e2e_avg_ms > 0 else 0,
            }
        },
        "notes": f"Djinn Ring Buffer: {ring_buffer_gb}GB resident ({resident_fraction*100:.0f}%), streaming {streaming_delta_gb:.1f}GB delta"
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Djinn Ring Buffer Measurement"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-13b-hf",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--ring-buffer-gb",
        type=float,
        default=20.0,
        help="Ring buffer capacity in GB"
    )
    parser.add_argument(
        "--output",
        default="results/djinn_ring_buffer.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=5,
        help="Number of measurement runs"
    )
    
    args = parser.parse_args()
    
    # Create standard config
    config = create_standard_config(
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
    )
    
    # Create measurement instance
    measurement = DjinnRingBufferMeasurement(
        args.model,
        config,
        ring_buffer_gb=args.ring_buffer_gb
    )
    
    try:
        # Load model with ring buffer
        measurement.load_model()
        
        # Run all measurements
        ttft_results, decode_results, e2e_results = measurement.run_all_measurements()
        
        # Print summaries
        ttft_results.print_summary()
        decode_results.print_summary()
        e2e_results.print_summary()
        
        # Save results
        save_results(ttft_results, decode_results, e2e_results, args.output, args.ring_buffer_gb)
        
        logger.info("=" * 70)
        logger.info("✅ Djinn Ring Buffer Measurement COMPLETE")
        logger.info("=" * 70)
        
    except RuntimeError as e:
        logger.error(f"❌ Fatal error: {e}")
        measurement.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        measurement.cleanup()
        sys.exit(1)
    finally:
        measurement.cleanup()


if __name__ == "__main__":
    main()

