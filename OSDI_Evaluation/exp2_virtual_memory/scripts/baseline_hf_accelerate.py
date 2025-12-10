#!/usr/bin/env python3
"""
HuggingFace Accelerate Baseline: The Python Standard

This baseline uses HuggingFace's device_map="auto" which is the standard
approach Python practitioners use to run oversized models on small GPUs.

The mechanism:
- device_map="auto": Automatically profiles layer sizes
- Places as many layers as fit on GPU
- Offloads remaining layers to CPU with device_map strategy
- Uses synchronous blocking transfers (no pipelining)

Expected performance on L4 (24GB VRAM) with Llama-2-13B (26GB FP16):
- TTFT: 8-12 seconds (synchronous CPU→GPU transfer)
- Decode: 600-700 ms/token
- E2E: ~45-50 seconds

This is the "standard" baseline that Djinn should be compared against.
"""

import argparse
import json
import logging
import time
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import measurement protocol
import sys
sys.path.insert(0, str(Path(__file__).parent))
from measurement_protocol import (
    MeasurementProtocol, 
    ExperimentConfig, 
    create_standard_config
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HFAccelerateBaseline(MeasurementProtocol):
    """
    HuggingFace Accelerate baseline using device_map="auto".
    
    This is the Python standard for running oversized models on GPUs.
    """
    
    def __init__(self, model_id: str, config: ExperimentConfig):
        """Initialize HF Accelerate baseline."""
        super().__init__(model_id, config)
        self.model_device = None
    
    def load_model(self):
        """Load model with HF Accelerate's automatic device placement."""
        logger.info("=" * 70)
        logger.info("HuggingFace Accelerate Baseline")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_id}")
        logger.info(f"Device placement: device_map='auto' (automatic CPU/GPU split)")
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
        
        # Load model with device_map="auto"
        logger.info("Loading model with device_map='auto'...")
        logger.info("  (This will automatically split model across GPU and CPU)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",  # KEY: Automatic device placement
            offload_folder="./offload",
            offload_state_dict=True,
            local_files_only=False,
            resume_download=True,
        )
        
        self.model_size_gb = self.get_model_size_gb()
        self.load_time_ms = (time.time() - load_start) * 1000
        
        # Get the device of the first parameter
        self.model_device = next(self.model.parameters()).device
        
        logger.info(f"✅ Model loaded")
        logger.info(f"  Model size: {self.model_size_gb:.1f}GB")
        logger.info(f"  Load time: {self.load_time_ms:.1f}ms")
        logger.info(f"  Primary device: {self.model_device}")
        logger.info("")
    
    def get_model_size_gb(self) -> float:
        """Calculate total model size in GB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        # float16 = 2 bytes per parameter
        return (total_params * 2) / (1024**3)
    
    def run_inference(self, max_new_tokens: int = 1) -> float:
        """
        Run inference and return elapsed time in milliseconds.
        
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
        
        # Move to appropriate device
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
    ttft_results, decode_results, e2e_results, output_file: str
):
    """Save all measurement results to JSON."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compile all results
    results = {
        "baseline": "hf_accelerate",
        "model": ttft_results.model_id,
        "model_size_gb": ttft_results.model_size_gb,
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
        "notes": "HuggingFace Accelerate with device_map='auto' (standard Python approach)"
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HuggingFace Accelerate Baseline"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-13b-hf",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output",
        default="results/baseline_hf_accelerate.json",
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
    
    # Create baseline instance
    baseline = HFAccelerateBaseline(args.model, config)
    
    try:
        # Load model
        baseline.load_model()
        
        # Run all measurements
        ttft_results, decode_results, e2e_results = baseline.run_all_measurements()
        
        # Print summaries
        ttft_results.print_summary()
        decode_results.print_summary()
        e2e_results.print_summary()
        
        # Save results
        save_results(ttft_results, decode_results, e2e_results, args.output)
        
        logger.info("=" * 70)
        logger.info("✅ HuggingFace Accelerate Baseline COMPLETE")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

