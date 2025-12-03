#!/usr/bin/env python3
"""
Download models for virtual memory experiments.

Pre-downloads models to local cache to avoid counting download time during
measurements.

Usage:
    python download_model.py
    python download_model.py --model meta-llama/Llama-2-7b-hf
"""

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(model_id: str):
    """Download model and tokenizer."""
    logger.info(f"Downloading model: {model_id}")
    
    try:
        # Download tokenizer
        logger.info(f"Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info(f"✅ Tokenizer downloaded")
        
        # Download model (weights)
        logger.info(f"Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="cpu"
        )
        logger.info(f"✅ Model downloaded: {model.config.model_type}")
        logger.info(f"   Parameters: {model.num_parameters() / 1e9:.1f}B")
        
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-download models for experiments")
    parser.add_argument(
        "--model",
        default="EleutherAI/gpt-j-6B",
        help="Model to download (default: gpt-j-6b)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Model Download Utility")
    logger.info(f"{'='*70}\n")
    
    success = download_model(args.model)
    
    if success:
        logger.info(f"\n✅ Successfully downloaded {args.model}")
        logger.info(f"Model cached in: {Path.home() / '.cache' / 'huggingface' / 'hub'}")
    else:
        logger.error(f"\n❌ Failed to download {args.model}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

