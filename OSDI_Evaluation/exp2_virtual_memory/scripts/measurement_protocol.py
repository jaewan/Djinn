#!/usr/bin/env python3
"""
Consistent Measurement Protocol for Experiment 2

This module defines a standardized measurement protocol that all baselines
must follow. This ensures apples-to-apples comparison across:
- HuggingFace Accelerate
- DeepSpeed
- llama.cpp
- Djinn Ring Buffer

Metrics:
1. TTFT (Time-to-First-Token): Time to generate exactly 1 token from a 512-token prompt
2. Decode Latency: Time per token for generating 50 tokens (after prefill)
3. E2E Latency: Total time for 512-token prompt + 50 tokens generated

All measurements include model loading time in "end-to-end" context but
separate it in results for clarity.
"""

import time
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we measure."""
    TTFT = "ttft"
    DECODE = "decode"
    E2E = "e2e"


@dataclass
class ExperimentConfig:
    """Standard configuration for all baseline measurements."""
    
    # Prompt and generation parameters
    prompt_tokens: int = 512
    generated_tokens: int = 50
    
    # Benchmark parameters
    warmup_runs: int = 2
    measurement_runs: int = 5
    
    # System parameters
    batch_size: int = 1
    dtype: str = "float16"
    device: str = "cuda:0"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.prompt_tokens > 0, "prompt_tokens must be positive"
        assert self.generated_tokens > 0, "generated_tokens must be positive"
        assert self.warmup_runs >= 0, "warmup_runs must be non-negative"
        assert self.measurement_runs > 0, "measurement_runs must be positive"


@dataclass
class MeasurementResult:
    """Result from a single measurement run."""
    
    metric_type: MetricType
    run_number: int
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    
    def __str__(self) -> str:
        if self.metric_type == MetricType.TTFT:
            return f"Run {self.run_number}: {self.latency_ms:.1f}ms TTFT"
        elif self.metric_type == MetricType.DECODE:
            return f"Run {self.run_number}: {self.latency_ms:.1f}ms/token ({self.tokens_per_second:.2f} tok/s)"
        else:  # E2E
            return f"Run {self.run_number}: {self.latency_ms:.1f}ms E2E ({self.tokens_per_second:.2f} tok/s)"


@dataclass
class ExperimentResults:
    """Complete results from a baseline measurement."""
    
    baseline_name: str
    model_id: str
    config: ExperimentConfig
    
    # Metadata
    model_size_gb: float = 0.0
    load_time_ms: float = 0.0
    
    # Results by metric type
    ttft_results: List[MeasurementResult] = field(default_factory=list)
    decode_results: List[MeasurementResult] = field(default_factory=list)
    e2e_results: List[MeasurementResult] = field(default_factory=list)
    
    # Summary statistics
    ttft_avg_ms: float = 0.0
    ttft_min_ms: float = 0.0
    ttft_max_ms: float = 0.0
    ttft_stddev_ms: float = 0.0
    
    decode_avg_ms: float = 0.0
    decode_min_ms: float = 0.0
    decode_max_ms: float = 0.0
    
    e2e_avg_ms: float = 0.0
    e2e_min_ms: float = 0.0
    e2e_max_ms: float = 0.0
    
    def compute_statistics(self):
        """Compute summary statistics from individual runs."""
        import statistics
        
        # TTFT statistics
        if self.ttft_results:
            ttft_values = [r.latency_ms for r in self.ttft_results]
            self.ttft_avg_ms = statistics.mean(ttft_values)
            self.ttft_min_ms = min(ttft_values)
            self.ttft_max_ms = max(ttft_values)
            if len(ttft_values) > 1:
                self.ttft_stddev_ms = statistics.stdev(ttft_values)
        
        # Decode statistics
        if self.decode_results:
            decode_values = [r.latency_ms for r in self.decode_results]
            self.decode_avg_ms = statistics.mean(decode_values)
            self.decode_min_ms = min(decode_values)
            self.decode_max_ms = max(decode_values)
        
        # E2E statistics
        if self.e2e_results:
            e2e_values = [r.latency_ms for r in self.e2e_results]
            self.e2e_avg_ms = statistics.mean(e2e_values)
            self.e2e_min_ms = min(e2e_values)
            self.e2e_max_ms = max(e2e_values)
    
    def print_summary(self):
        """Print a formatted summary of results."""
        print(f"\n{'='*70}")
        print(f"Baseline: {self.baseline_name}")
        print(f"Model: {self.model_id} ({self.model_size_gb:.1f}GB)")
        print(f"Load time: {self.load_time_ms:.1f}ms")
        print(f"{'='*70}")
        
        if self.ttft_results:
            print(f"\nTTFT (Time-to-First-Token):")
            print(f"  Average: {self.ttft_avg_ms:.1f}ms")
            print(f"  Min:     {self.ttft_min_ms:.1f}ms")
            print(f"  Max:     {self.ttft_max_ms:.1f}ms")
            if self.ttft_stddev_ms > 0:
                print(f"  StdDev:  {self.ttft_stddev_ms:.1f}ms")
        
        if self.decode_results:
            print(f"\nDecode Latency (per token):")
            print(f"  Average: {self.decode_avg_ms:.1f}ms/token")
            print(f"  Min:     {self.decode_min_ms:.1f}ms/token")
            print(f"  Max:     {self.decode_max_ms:.1f}ms/token")
        
        if self.e2e_results:
            print(f"\nEnd-to-End (512 prompt + 50 generated):")
            print(f"  Average: {self.e2e_avg_ms:.1f}ms ({(512+50)/(self.e2e_avg_ms/1000):.2f} tok/s)")
            print(f"  Min:     {self.e2e_min_ms:.1f}ms")
            print(f"  Max:     {self.e2e_max_ms:.1f}ms")
        
        print()


class MeasurementProtocol:
    """
    Base class for implementing the standard measurement protocol.
    
    Subclasses should override:
    - load_model(): Load the model with their specific framework
    - run_inference(max_new_tokens): Run inference and return execution time
    """
    
    def __init__(self, model_id: str, config: ExperimentConfig):
        """Initialize measurement protocol."""
        self.model_id = model_id
        self.config = config
        self.tokenizer = None
        self.model = None
        self.model_size_gb = 0.0
        self.load_time_ms = 0.0
        
        # Standard prompt for reproducibility
        self.prompt = "The answer to life, the universe, and everything is"
    
    def load_model(self):
        """Load model. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement load_model()")
    
    def run_inference(self, max_new_tokens: int) -> float:
        """
        Run inference and return elapsed time in milliseconds.
        Must be implemented by subclass.
        
        Args:
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            elapsed_ms: Time in milliseconds
        """
        raise NotImplementedError("Subclass must implement run_inference()")
    
    def get_model_size_gb(self) -> float:
        """Get total model size in GB. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement get_model_size_gb()")
    
    def measure_ttft(self) -> ExperimentResults:
        """
        Measure Time-to-First-Token (TTFT).
        
        Generate exactly 1 token from a 512-token prompt.
        This measures prefill performance.
        """
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 1: Time-to-First-Token (TTFT) Measurement")
        logger.info(f"{'='*70}")
        logger.info(f"Prompt tokens: {self.config.prompt_tokens}")
        logger.info(f"Generate: 1 token (first token)")
        
        results = ExperimentResults(
            baseline_name=self.__class__.__name__,
            model_id=self.model_id,
            config=self.config,
            model_size_gb=self.model_size_gb,
            load_time_ms=self.load_time_ms,
        )
        
        logger.info(f"Running {self.config.measurement_runs} measurement runs...")
        
        for run in range(self.config.measurement_runs):
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Run inference for exactly 1 token
            elapsed_ms = self.run_inference(max_new_tokens=1)
            
            result = MeasurementResult(
                metric_type=MetricType.TTFT,
                run_number=run + 1,
                latency_ms=elapsed_ms,
                tokens_generated=1,
                tokens_per_second=1000.0 / elapsed_ms if elapsed_ms > 0 else 0,
            )
            
            results.ttft_results.append(result)
            logger.info(f"  {result}")
        
        results.compute_statistics()
        logger.info(f"✅ TTFT Average: {results.ttft_avg_ms:.1f}ms")
        
        return results
    
    def measure_decode(self) -> ExperimentResults:
        """
        Measure decode latency per token.
        
        After initial prefill with 512 tokens, generate 50 tokens one-by-one
        and measure the average time per token.
        """
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 2: Decode Latency Per Token")
        logger.info(f"{'='*70}")
        logger.info(f"Setup: {self.config.prompt_tokens}-token prompt (prefill)")
        logger.info(f"Generate: {self.config.generated_tokens} tokens (one-by-one)")
        
        results = ExperimentResults(
            baseline_name=self.__class__.__name__,
            model_id=self.model_id,
            config=self.config,
            model_size_gb=self.model_size_gb,
            load_time_ms=self.load_time_ms,
        )
        
        logger.info(f"Running {self.config.measurement_runs} measurement runs...")
        
        for run in range(self.config.measurement_runs):
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Run inference to generate tokens sequentially
            # Time is measured for generating all tokens in one call
            elapsed_ms = self.run_inference(
                max_new_tokens=self.config.generated_tokens
            )
            
            # Calculate per-token latency
            latency_per_token_ms = elapsed_ms / self.config.generated_tokens
            tokens_per_second = 1000.0 / latency_per_token_ms if latency_per_token_ms > 0 else 0
            
            result = MeasurementResult(
                metric_type=MetricType.DECODE,
                run_number=run + 1,
                latency_ms=latency_per_token_ms,
                tokens_generated=self.config.generated_tokens,
                tokens_per_second=tokens_per_second,
            )
            
            results.decode_results.append(result)
            logger.info(f"  {result}")
        
        results.compute_statistics()
        logger.info(f"✅ Decode Average: {results.decode_avg_ms:.1f}ms/token")
        
        return results
    
    def measure_e2e(self) -> ExperimentResults:
        """
        Measure End-to-End (E2E) latency.
        
        Generate 512-token prompt + 50 tokens in a single inference call.
        This measures total generation time for a complete response.
        """
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 3: End-to-End Generation")
        logger.info(f"{'='*70}")
        logger.info(f"Prompt: {self.config.prompt_tokens} tokens")
        logger.info(f"Generate: {self.config.generated_tokens} tokens")
        
        results = ExperimentResults(
            baseline_name=self.__class__.__name__,
            model_id=self.model_id,
            config=self.config,
            model_size_gb=self.model_size_gb,
            load_time_ms=self.load_time_ms,
        )
        
        logger.info(f"Running {self.config.measurement_runs} measurement runs...")
        
        for run in range(self.config.measurement_runs):
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Run full inference
            elapsed_ms = self.run_inference(
                max_new_tokens=self.config.generated_tokens
            )
            
            total_tokens = self.config.prompt_tokens + self.config.generated_tokens
            tokens_per_second = (total_tokens * 1000.0) / elapsed_ms if elapsed_ms > 0 else 0
            
            result = MeasurementResult(
                metric_type=MetricType.E2E,
                run_number=run + 1,
                latency_ms=elapsed_ms,
                tokens_generated=self.config.generated_tokens,
                tokens_per_second=tokens_per_second,
            )
            
            results.e2e_results.append(result)
            logger.info(f"  {result}")
        
        results.compute_statistics()
        logger.info(f"✅ E2E Average: {results.e2e_avg_ms:.1f}ms ({(total_tokens*1000.0)/results.e2e_avg_ms:.2f} tok/s)")
        
        return results
    
    def run_all_measurements(self) -> Tuple[ExperimentResults, ExperimentResults, ExperimentResults]:
        """
        Run all three measurement phases and return results.
        
        Returns:
            (ttft_results, decode_results, e2e_results)
        """
        # Warmup
        logger.info(f"\n{'='*70}")
        logger.info(f"Warmup ({self.config.warmup_runs} iterations)")
        logger.info(f"{'='*70}")
        
        for i in range(self.config.warmup_runs):
            self.run_inference(max_new_tokens=10)
            logger.info(f"  Warmup {i+1}/{self.config.warmup_runs}")
        
        # Run measurements
        ttft_results = self.measure_ttft()
        decode_results = self.measure_decode()
        e2e_results = self.measure_e2e()
        
        return ttft_results, decode_results, e2e_results


def create_standard_config(**kwargs) -> ExperimentConfig:
    """
    Factory function to create standard experiment configuration.
    
    Args:
        **kwargs: Override default values for ExperimentConfig fields
        
    Returns:
        ExperimentConfig with standard settings
    """
    return ExperimentConfig(**kwargs)


if __name__ == "__main__":
    # Example usage
    config = create_standard_config()
    print("Standard Experiment Configuration:")
    print(f"  Prompt tokens: {config.prompt_tokens}")
    print(f"  Generated tokens: {config.generated_tokens}")
    print(f"  Warmup runs: {config.warmup_runs}")
    print(f"  Measurement runs: {config.measurement_runs}")

