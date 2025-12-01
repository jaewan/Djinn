#!/usr/bin/env python3
"""
Pre-download HuggingFace models for Djinn evaluation experiments.

This script downloads all the models used in the evaluation plan to avoid
network delays during actual experiments. Run this on the GPU server before
starting evaluations.

REQUIREMENTS:
- For LLaMA models (meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf):
  1. Apply for access at: https://huggingface.co/meta-llama
  2. Get approved by Meta (may take 1-2 weeks)
  3. Set up HuggingFace API token (see below)

HuggingFace API Token Setup:
Store your token in one of these ways:

Option A - Environment variable (recommended for scripts):
    export HF_TOKEN=your_token_here
    python Evaluation/pre_download_models.py

Option B - Token file:
    mkdir -p ~/.huggingface
    echo 'your_token_here' > ~/.huggingface/token

Option C - Config file:
    mkdir -p ~/.huggingface
    echo '[huggingface]' > ~/.huggingface/config
    echo 'token = your_token_here' >> ~/.huggingface/config

Get your token at: https://huggingface.co/settings/tokens

Usage:
    python Evaluation/pre_download_models.py

Models to download:
- LLM models: LLaMA-7B*, LLaMA-13B*, GPT-J-6B, GPT-2-Small
- Vision models: ResNet-50, ViT-Base
- Multimodal models: CLIP-ViT-Large, Whisper-Large
- Encoder models: BERT-Base, BERT-Large
  (* requires Meta approval + HF token)

All downloads will be cached in the HuggingFace cache directory.
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM,
    AutoModelForImageClassification, AutoImageProcessor,
    AutoModelForSeq2SeqLM, AutoProcessor, CLIPModel, CLIPProcessor,
    WhisperForConditionalGeneration
)
from huggingface_hub import login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations from evaluation plan
MODELS_TO_DOWNLOAD = {
    # LLM Models (Experiments 2.1, 2.3, 5.1, 6.1)
    # NOTE: LLaMA models require Meta approval - see instructions below
    "meta-llama/Llama-2-7b-hf": {
        "type": "causal_lm",
        "description": "LLaMA-7B for LLM decode experiments",
        "size_gb": 12,
        "requires_access": True,
        "access_org": "meta-llama"
    },
    "meta-llama/Llama-2-13b-hf": {
        "type": "causal_lm",
        "description": "LLaMA-13B for overhead analysis",
        "size_gb": 24,
        "requires_access": True,
        "access_org": "meta-llama"
    },
    "EleutherAI/gpt-j-6b": {
        "type": "causal_lm",
        "description": "GPT-J-6B for conversational AI",
        "size_gb": 22,
        "requires_access": False
    },
    "gpt2": {
        "type": "causal_lm",
        "description": "GPT-2-Small for overhead analysis",
        "size_gb": 0.5,
        "requires_access": False
    },

    # Encoder Models (Experiment 5.1)
    "google-bert/bert-base-uncased": {
        "type": "masked_lm",
        "description": "BERT-Base for overhead analysis",
        "size_gb": 0.4,
        "requires_access": False
    },
    "google-bert/bert-large-uncased": {
        "type": "masked_lm",
        "description": "BERT-Large for overhead analysis",
        "size_gb": 1.2,
        "requires_access": False
    },

    # Vision Models (Experiments 4.1, 6.1)
    "microsoft/resnet-50": {
        "type": "image_classification",
        "description": "ResNet-50 for vision workloads",
        "size_gb": 0.1,
        "requires_access": False
    },
    "google/vit-base-patch16-224": {
        "type": "image_classification",
        "description": "ViT-Base for vision workloads",
        "size_gb": 0.3,
        "requires_access": False
    },

    # Multimodal Models (Experiments 2.2, 6.1)
    "openai/clip-vit-large-patch14": {
        "type": "multimodal",
        "description": "CLIP-ViT-Large for multimodal workloads",
        "size_gb": 1.7,
        "requires_access": False
    },
    "openai/whisper-large-v3": {
        "type": "seq2seq",
        "description": "Whisper-Large for streaming audio transcription",
        "size_gb": 3.0,
        "requires_access": False
    }
}

def download_model(model_id: str, model_config: dict) -> bool:
    """
    Download a single model and its tokenizer/processor.

    Args:
        model_id: HuggingFace model identifier
        model_config: Configuration dict with type and description

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {model_id} ({model_config['description']})...")

        model_type = model_config['type']

        if model_type == "causal_lm":
            # Download tokenizer and model for causal language modeling
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", low_cpu_mem_usage=True)

        elif model_type == "masked_lm":
            # Download tokenizer and model for masked language modeling
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype="auto", low_cpu_mem_usage=True)

        elif model_type == "image_classification":
            # Download processor and model for image classification
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype="auto", low_cpu_mem_usage=True)

        elif model_type == "multimodal":
            # Download processor and model for CLIP
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id, torch_dtype="auto", low_cpu_mem_usage=True)

        elif model_type == "seq2seq":
            # Download processor and model for Whisper-style seq2seq workloads
            processor = AutoProcessor.from_pretrained(model_id)
            if "whisper" in model_id:
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype="auto", low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id, torch_dtype="auto", low_cpu_mem_usage=True
                )

        else:
            logger.error(f"Unknown model type: {model_type}")
            return False

        # Force model to device to ensure weights are downloaded
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
            # Move back to CPU to free GPU memory
            model = model.to("cpu")
            torch.cuda.empty_cache()

        logger.info(f"✅ Successfully downloaded {model_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {model_id}: {str(e)}")
        return False

def check_huggingface_auth():
    """Check if HuggingFace authentication is available and try to login if needed."""
    from huggingface_hub.utils import HfHubHTTPError

    # Check if any models require access
    models_requiring_access = [mid for mid, cfg in MODELS_TO_DOWNLOAD.items() if cfg.get('requires_access', False)]

    if not models_requiring_access:
        logger.info("No models require special access - proceeding without authentication")
        return True

    logger.info(f"Found {len(models_requiring_access)} models requiring access approval:")
    for model_id in models_requiring_access:
        org = MODELS_TO_DOWNLOAD[model_id]['access_org']
        logger.info(f"  - {model_id} (requires {org} approval)")

    # Check for existing authentication
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

    # Check token file
    token_file = Path.home() / '.huggingface' / 'token'
    if token_file.exists():
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read token from {token_file}: {e}")

    if not token:
        logger.error("❌ No HuggingFace token found!")
        logger.error("")
        logger.error("To download LLaMA models, you need:")
        logger.error("1. A HuggingFace account")
        logger.error("2. Meta's approval for LLaMA models (apply at: https://huggingface.co/meta-llama)")
        logger.error("3. A HuggingFace API token")
        logger.error("")
        logger.error("Store your token in one of these ways:")
        logger.error("  Option A - Environment variable:")
        logger.error("    export HF_TOKEN=your_token_here")
        logger.error("")
        logger.error("  Option B - Token file:")
        logger.error("    echo 'your_token_here' > ~/.huggingface/token")
        logger.error("")
        logger.error("  Option C - Config file:")
        logger.error("    echo '[huggingface]' > ~/.huggingface/config")
        logger.error("    echo 'token = your_token_here' >> ~/.huggingface/config")
        logger.error("")
        logger.error("Get your token at: https://huggingface.co/settings/tokens")
        return False

    # Try to authenticate
    try:
        logger.info("Authenticating with HuggingFace...")
        login(token=token)
        logger.info("✅ Successfully authenticated with HuggingFace")
        return True
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

def main():
    """Main function to download all models."""
    logger.info("Starting pre-download of HuggingFace models for Djinn evaluation")
    logger.info(f"Total models to download: {len(MODELS_TO_DOWNLOAD)}")

    # Calculate total expected size
    total_size_gb = sum(config['size_gb'] for config in MODELS_TO_DOWNLOAD.values())
    logger.info(f"Approximate download size: {total_size_gb:.1f} GB")

    # Check authentication first
    if not check_huggingface_auth():
        logger.error("Cannot proceed without HuggingFace authentication for gated models")
        return 1

    success_count = 0
    failed_models = []

    for model_id, config in MODELS_TO_DOWNLOAD.items():
        if config.get('requires_access', False):
            logger.info(f"Downloading gated model: {model_id}")
        else:
            logger.info(f"Downloading public model: {model_id}")

        if download_model(model_id, config):
            success_count += 1
        else:
            failed_models.append(model_id)

        # Small delay between downloads to be respectful
        import time
        time.sleep(1)

    logger.info(f"\nDownload Summary:")
    logger.info(f"✅ Successful: {success_count}/{len(MODELS_TO_DOWNLOAD)}")
    if failed_models:
        logger.info(f"❌ Failed: {', '.join(failed_models)}")
        logger.info("You may need to retry failed downloads or check network connectivity")
        return 1
    else:
        logger.info("🎉 All models downloaded successfully!")
        logger.info("GPU server is ready for Djinn evaluation experiments.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
