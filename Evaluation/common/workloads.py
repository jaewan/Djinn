"""
Synthetic workload primitives used by the week-5 evaluation harnesses.

The goal is to provide lightweight-yet-representative compute kernels that run on
the dev L4 GPU while exercising transformer- and CNN-style execution paths.  The
config files control dimensionality so we can scale up later without code
changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping.get(name, torch.float32)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


@dataclass
class RunMetrics:
    latency_ms: float
    input_mb: float
    output_mb: float
    units_processed: float
    throughput_units_per_s: float


class BaseSyntheticWorkload:
    """Common scaffolding for synthetic workloads."""

    def __init__(self, spec: Dict, device: str, dtype: str):
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.unit_name = spec.get("unit_name", "items")
        self.model = self._build_model().to(self.device, dtype=self.dtype)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    def _sample_inputs(self) -> torch.Tensor:
        raise NotImplementedError

    def _units_processed(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        raise NotImplementedError

    # Phase 4: expose a common interface for experiment runners
    def prepare_inputs(self) -> Dict[str, torch.Tensor]:
        """Return a dict of inputs suitable for execute_model()."""
        return {"x": self._sample_inputs()}

    def units_from_output(self, inputs: Dict[str, torch.Tensor], outputs: torch.Tensor) -> float:
        """Return logical units processed for throughput metrics."""
        tensor_input = inputs.get("x")
        if tensor_input is None:
            raise KeyError("Synthetic workload expected key 'x' in inputs")
        return self._units_processed(tensor_input, outputs)
    def run_once(self) -> RunMetrics:
        inputs = self._sample_inputs()
        with torch.no_grad():
            start = time.perf_counter()
            outputs = self.model(inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - start) * 1000.0

        input_mb = inputs.element_size() * inputs.numel() / (1024**2)
        if isinstance(outputs, torch.Tensor):
            output_mb = outputs.element_size() * outputs.numel() / (1024**2)
        else:
            output_mb = 0.0

        units = self._units_processed(inputs, outputs)
        throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

        return RunMetrics(
            latency_ms=latency_ms,
            input_mb=input_mb,
            output_mb=output_mb,
            units_processed=units,
            throughput_units_per_s=throughput,
        )

    def metadata(self) -> Dict:
        return {
            "unit_name": self.unit_name,
            "device": str(self.device),
            "dtype": str(self.dtype).split(".")[-1],
            "spec": self.spec,
        }


class SyntheticTransformerWorkload(BaseSyntheticWorkload):
    """Transformer-style workload approximating LLM decode."""

    def _build_model(self) -> nn.Module:
        hidden = self.spec.get("hidden_size", 1024)
        heads = self.spec.get("num_heads", 8)
        ff = self.spec.get("ff_size", hidden * 4)
        num_layers = self.spec.get("num_layers", 6)
        dropout = self.spec.get("dropout", 0.0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _sample_inputs(self) -> torch.Tensor:
        batch = self.spec.get("batch_size", 1)
        seq_len = self.spec.get("sequence_length", 256)
        hidden = self.spec.get("hidden_size", 1024)
        return torch.randn((batch, seq_len, hidden), device=self.device, dtype=self.dtype)

    def _units_processed(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        batch = inputs.shape[0]
        seq_len = inputs.shape[1]
        return float(batch * seq_len)


class SyntheticCnnWorkload(BaseSyntheticWorkload):
    """CNN-style workload approximating vision inference."""

    def _build_model(self) -> nn.Module:
        in_channels = self.spec.get("in_channels", 3)
        channels = self.spec.get("channels", [32, 64, 128])
        kernel_size = self.spec.get("kernel_size", 3)
        use_pool = self.spec.get("use_pooling", True)
        layers = []
        current = in_channels
        for out_ch in channels:
            layers.append(nn.Conv2d(current, out_ch, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_ch))
            if use_pool:
                layers.append(nn.MaxPool2d(2))
            current = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(current, self.spec.get("num_classes", 1000)))
        return nn.Sequential(*layers)

    def _sample_inputs(self) -> torch.Tensor:
        batch = self.spec.get("batch_size", 8)
        channels = self.spec.get("in_channels", 3)
        height = self.spec.get("image_size", 224)
        width = self.spec.get("image_size", 224)
        return torch.randn((batch, channels, height, width), device=self.device, dtype=self.dtype)

    def _units_processed(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        return float(inputs.shape[0])


class SyntheticHybridWorkload(BaseSyntheticWorkload):
    """Hybrid workload combining CNN feature extraction with a transformer head."""

    class _HybridModel(nn.Module):
        def __init__(self, spec: Dict):
            super().__init__()
            in_channels = spec.get("in_channels", 3)
            channels = spec.get("channels", [32, 64])
            kernel_size = spec.get("kernel_size", 3)
            transformer_hidden = spec.get("hidden_size", 256)
            transformer_layers = spec.get("num_layers", 2)
            heads = spec.get("num_heads", 4)
            ff = spec.get("ff_size", transformer_hidden * 4)

            conv_layers = []
            current = in_channels
            for out_ch in channels:
                conv_layers.append(nn.Conv2d(current, out_ch, kernel_size=kernel_size, padding=1))
                conv_layers.append(nn.ReLU())
                conv_layers.append(nn.BatchNorm2d(out_ch))
                conv_layers.append(nn.MaxPool2d(2))
                current = out_ch
            self.conv = nn.Sequential(*conv_layers)
            self.project = nn.Linear(current, transformer_hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_hidden,
                nhead=heads,
                dim_feedforward=ff,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.head = nn.Linear(transformer_hidden, spec.get("num_classes", 512))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.conv(x)  # (batch, C, H, W)
            batch, channels, height, width = feats.shape
            seq = feats.view(batch, channels, height * width).transpose(1, 2)  # (batch, seq, channels)
            seq = self.project(seq)
            encoded = self.transformer(seq)
            pooled = encoded.mean(dim=1)
            return self.head(pooled)

    def _build_model(self) -> nn.Module:
        return self._HybridModel(self.spec)

    def _sample_inputs(self) -> torch.Tensor:
        batch = self.spec.get("batch_size", 4)
        channels = self.spec.get("in_channels", 3)
        height = self.spec.get("image_size", 224)
        width = self.spec.get("image_size", 224)
        return torch.randn((batch, channels, height, width), device=self.device, dtype=self.dtype)

    def _units_processed(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        batch = inputs.shape[0]
        spatial = inputs.shape[2] * inputs.shape[3]
        return float(batch * spatial)


class HuggingFaceCausalLMWorkload:
    """Lightweight HuggingFace autoregressive workload for real-model smoke tests."""

    DEFAULT_PROMPT = "Djinn orchestrates disaggregated accelerators by keeping semantic context."

    def __init__(self, spec: Dict[str, Any], device: str, dtype: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # Lazy import

        self.spec = spec
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.unit_name = spec.get("unit_name", "tokens")
        self.model_id = spec["model_id"]
        prompt_text = self._resolve_prompt(spec)
        self.prompt_length = spec.get("prompt_length", 64)
        self.new_tokens = spec.get("new_tokens", 32)
        self.batch_size = spec.get("batch_size", 1)
        self.generation_params = self._sanitize_generation_params(
            spec.get(
                "generation",
                {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            )
        )

        torch_dtype = self.dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # PHASE 1.6 FIX: Disable interception during model loading to prevent accidental execution
        try:
            from djinn.frontend.core.interception_control import disable_interception, InterceptionContext
            with disable_interception(InterceptionContext.CONSTRUCTION):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                self.model.eval()
        except ImportError:
            # Djinn not available - load normally
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()

        encoded = self.tokenizer(
            [prompt_text] * self.batch_size,
            padding="max_length",
            max_length=self.prompt_length,
            truncation=True,
            return_tensors="pt",
        )
        self.base_input_ids = encoded["input_ids"]
        self.base_attention_mask = encoded["attention_mask"]

    def _resolve_prompt(self, spec: Dict[str, Any]) -> str:
        if "prompt_text" in spec and spec["prompt_text"]:
            return str(spec["prompt_text"]).strip()
        if "prompt_file" in spec and spec["prompt_file"]:
            path = Path(spec["prompt_file"])
            return path.read_text().strip()
        return self.DEFAULT_PROMPT

    def run_once(self) -> RunMetrics:
        input_ids = self.base_input_ids.clone().to(self.device)
        attention_mask = self.base_attention_mask.clone().to(self.device)
        gen_kwargs = dict(
            max_new_tokens=self.new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.generation_params,
        )

        with torch.no_grad():
            start = time.perf_counter()
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - start) * 1000.0

        tokens_generated = generated.shape[-1] - input_ids.shape[-1]
        throughput = (
            tokens_generated / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        )

        input_mb = input_ids.element_size() * input_ids.numel() / (1024**2)
        generated_cpu = generated.to("cpu")
        output_mb = generated_cpu.element_size() * generated_cpu.numel() / (1024**2)

        return RunMetrics(
            latency_ms=latency_ms,
            input_mb=input_mb,
            output_mb=output_mb,
            units_processed=float(max(tokens_generated, 0)),
            throughput_units_per_s=throughput,
        )

    # Phase 4: interoperable interface for experiment harnesses
    def prepare_inputs(self) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.base_input_ids.clone(),
            "attention_mask": self.base_attention_mask.clone(),
        }

    def units_from_output(self, inputs: Dict[str, torch.Tensor], output: torch.Tensor) -> float:
        batch = self.batch_size
        return float(batch * self.new_tokens)

    def metadata(self) -> Dict[str, Any]:
        return {
            "unit_name": self.unit_name,
            "device": str(self.device),
            "dtype": str(self.dtype).split(".")[-1],
            "spec": {
                **self.spec,
                "prompt_length": self.prompt_length,
                "new_tokens": self.new_tokens,
            },
        }

    def _sanitize_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(params)
        do_sample = sanitized.get("do_sample", False)
        if not do_sample:
            sanitized.pop("temperature", None)
            sanitized.pop("top_p", None)
        return sanitized


class HuggingFaceVisionWorkload:
    """
    HuggingFace vision classification workload for OSDI evaluation.
    
    Supports real vision models:
    - ResNet-50: "microsoft/resnet-50" or "resnet50"
    - ViT-Base: "google/vit-base-patch16-224"
    - EfficientNet: "google/efficientnet-b0"
    
    For Phase 4 load testing and OSDI evaluation.
    """

    def __init__(self, spec: Dict[str, Any], device: str, dtype: str):
        from transformers import AutoModelForImageClassification, AutoImageProcessor  # Lazy import
        from PIL import Image
        import numpy as np

        self.spec = spec
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.unit_name = spec.get("unit_name", "images")
        self.model_id = spec["model_id"]
        self.batch_size = spec.get("batch_size", 8)
        self.image_size = spec.get("image_size", 224)

        # Load model and processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        except Exception:
            # Fallback: some models don't have separate processor
            from transformers import AutoFeatureExtractor
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        
        # PHASE 1.6 FIX: Disable interception during model loading to prevent accidental execution
        try:
            from djinn.frontend.core.interception_control import disable_interception, InterceptionContext
            with disable_interception(InterceptionContext.CONSTRUCTION):
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                self.model.eval()
        except ImportError:
            # Djinn not available - load normally
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
        
        # Pre-allocate dummy images for consistent inputs
        # Create random RGB images
        dummy_images = [
            Image.fromarray(np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8))
            for _ in range(self.batch_size)
        ]
        
        # Process images
        try:
            self.base_inputs = self.processor(images=dummy_images, return_tensors="pt")
        except Exception:
            # Fallback: create pixel_values directly if processor fails
            self.base_inputs = {
                "pixel_values": torch.randn(
                    (self.batch_size, 3, self.image_size, self.image_size),
                    dtype=self.dtype
                )
            }

    def prepare_inputs(self) -> Dict[str, torch.Tensor]:
        """Prepare inputs for vision model execution."""
        # Use pre-processed inputs for consistency
        pixel_values = self.base_inputs["pixel_values"].clone().to(self.device)
        if pixel_values.dtype != self.dtype:
            pixel_values = pixel_values.to(self.dtype)
        return {"pixel_values": pixel_values}
    
    def run_once(self) -> RunMetrics:
        """Execute one forward pass for local baseline."""
        inputs = self.prepare_inputs()
        
        with torch.no_grad():
            start = time.perf_counter()
            outputs = self.model(**inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - start) * 1000.0
        
        # Extract logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs.get('logits', list(outputs.values())[0])
        else:
            logits = outputs
        
        input_mb = inputs["pixel_values"].element_size() * inputs["pixel_values"].numel() / (1024**2)
        output_mb = logits.element_size() * logits.numel() / (1024**2)
        
        units = self.units_from_output(inputs, logits)
        throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        
        return RunMetrics(
            latency_ms=latency_ms,
            input_mb=input_mb,
            output_mb=output_mb,
            units_processed=units,
            throughput_units_per_s=throughput,
        )

    def units_from_output(self, inputs: Dict[str, torch.Tensor], output: torch.Tensor) -> float:
        return float(self.batch_size)

    def metadata(self) -> Dict[str, Any]:
        return {
            "unit_name": self.unit_name,
            "device": str(self.device),
            "dtype": str(self.dtype).split(".")[-1],
            "spec": {
                **self.spec,
                "batch_size": self.batch_size,
                "image_size": self.image_size,
            },
        }


class HuggingFaceMultimodalWorkload:
    """
    HuggingFace multimodal workload (e.g., CLIP) for OSDI evaluation.
    
    Supports real multimodal models:
    - CLIP: "openai/clip-vit-base-patch32" or "openai/clip-vit-large-patch14"
    - BLIP: "Salesforce/blip-image-captioning-base"
    
    For Phase 4 load testing and OSDI evaluation (hybrid/custom class).
    """

    def __init__(self, spec: Dict[str, Any], device: str, dtype: str):
        from transformers import AutoModel, AutoProcessor  # Lazy import
        from PIL import Image
        import numpy as np

        self.spec = spec
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.unit_name = spec.get("unit_name", "pairs")
        self.model_id = spec["model_id"]
        self.batch_size = spec.get("batch_size", 4)
        self.image_size = spec.get("image_size", 224)
        self.text_length = spec.get("text_length", 77)  # CLIP default
        self.execution_mode = spec.get("execution_mode", "embed").lower()
        if self.execution_mode not in {"embed", "classification"}:
            raise ValueError(f"Unsupported execution_mode '{self.execution_mode}'")
        
        # Pre-allocate dummy inputs
        dummy_images = [
            Image.fromarray(np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8))
            for _ in range(self.batch_size)
        ]
        dummy_texts = [f"Image {i} description" for i in range(self.batch_size)]

        # PHASE 1.6 FIX: Disable interception during model loading to prevent accidental execution
        try:
            from djinn.frontend.core.interception_control import disable_interception, InterceptionContext
            interception_disabled = True
        except ImportError:
            interception_disabled = False
        
        if self.execution_mode == "classification":
            # Use image classification variant and only pixel_values
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            image_processor = AutoImageProcessor.from_pretrained(self.model_id)
            if interception_disabled:
                with disable_interception(InterceptionContext.CONSTRUCTION):
                    self.model = AutoModelForImageClassification.from_pretrained(
                        self.model_id,
                        torch_dtype=self.dtype,
                        low_cpu_mem_usage=True,
                    ).to(self.device)
                    self.model.eval()
            else:
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                self.model.eval()
            inputs = image_processor(images=dummy_images, return_tensors="pt")
            self.base_inputs = {"pixel_values": inputs["pixel_values"]}
        else:
            # Load model and processor for embedding workloads
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                use_unified_processor = True
            except Exception:
                try:
                    self.image_processor = AutoImageProcessor.from_pretrained(self.model_id)
                    self.text_processor = AutoTokenizer.from_pretrained(self.model_id)
                    use_unified_processor = False
                except Exception as e:
                    raise RuntimeError(f"Failed to load processors for {self.model_id}: {e}")
            if interception_disabled:
                with disable_interception(InterceptionContext.CONSTRUCTION):
                    self.model = AutoModel.from_pretrained(
                        self.model_id,
                        torch_dtype=self.dtype,
                        low_cpu_mem_usage=True,
                    ).to(self.device)
                    self.model.eval()
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                self.model.eval()
            if use_unified_processor:
                self.base_inputs = self.processor(images=dummy_images, text=dummy_texts, return_tensors="pt", padding=True)
            else:
                image_inputs = self.image_processor(images=dummy_images, return_tensors="pt")
                text_inputs = self.text_processor(text=dummy_texts, return_tensors="pt", padding=True)
                self.base_inputs = {**image_inputs, **text_inputs}

    def prepare_inputs(self) -> Dict[str, torch.Tensor]:
        """Prepare inputs for multimodal model execution."""
        inputs = {}
        for key, value in self.base_inputs.items():
            tensor = value.clone().to(self.device)
            if tensor.dtype != self.dtype and tensor.dtype in (torch.float32, torch.float16):
                tensor = tensor.to(self.dtype)
            inputs[key] = tensor
        return inputs
    
    def units_from_output(self, inputs: Dict[str, torch.Tensor], output: torch.Tensor) -> float:
        """Return number of image-text pairs processed."""
        return float(self.batch_size)
    
    def run_once(self) -> RunMetrics:
        """Execute one forward pass for local baseline."""
        inputs = self.prepare_inputs()
        
        with torch.no_grad():
            start = time.perf_counter()
            outputs = self.model(**inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - start) * 1000.0
        
        if self.execution_mode == "classification":
            output_tensor = getattr(outputs, "logits", outputs)
        else:
            if hasattr(outputs, "image_embeds"):
                output_tensor = outputs.image_embeds
            elif hasattr(outputs, "last_hidden_state"):
                output_tensor = outputs.last_hidden_state
            elif isinstance(outputs, dict):
                output_tensor = list(outputs.values())[0]
            else:
                output_tensor = outputs
        
        # Calculate input size (images + text)
        input_bytes = sum(t.element_size() * t.numel() for t in inputs.values())
        output_bytes = output_tensor.element_size() * output_tensor.numel()
        
        input_mb = input_bytes / (1024**2)
        output_mb = output_bytes / (1024**2)
        
        units = self.units_from_output(inputs, output_tensor)
        throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        
        return RunMetrics(
            latency_ms=latency_ms,
            input_mb=input_mb,
            output_mb=output_mb,
            units_processed=units,
            throughput_units_per_s=throughput,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "unit_name": self.unit_name,
            "device": str(self.device),
            "dtype": str(self.dtype).split(".")[-1],
            "spec": {
                **self.spec,
                "batch_size": self.batch_size,
                "image_size": self.image_size,
                "text_length": self.text_length,
                "execution_mode": self.execution_mode,
            },
        }


def build_workload(implementation: str, spec: Dict, device: str, dtype: str) -> BaseSyntheticWorkload:
    """Factory helper used by the experiment harness."""
    # PHASE 1.6 FIX: Disable Djinn interception during workload construction
    # This prevents accidental model execution during HuggingFace model loading
    try:
        from djinn.frontend.core.interception_control import disable_interception, InterceptionContext
        with disable_interception(InterceptionContext.CONSTRUCTION):
            return _build_workload_impl(implementation, spec, device, dtype)
    except ImportError:
        # Djinn not available - build normally
        return _build_workload_impl(implementation, spec, device, dtype)


def _build_workload_impl(implementation: str, spec: Dict, device: str, dtype: str) -> BaseSyntheticWorkload:
    """Internal implementation of workload building."""
    implementation = implementation.lower()
    if implementation == "synthetic_transformer":
        return SyntheticTransformerWorkload(spec, device, dtype)
    if implementation == "synthetic_cnn":
        return SyntheticCnnWorkload(spec, device, dtype)
    if implementation == "synthetic_hybrid":
        return SyntheticHybridWorkload(spec, device, dtype)
    if implementation == "hf_causal_lm":
        return HuggingFaceCausalLMWorkload(spec, device, dtype)
    if implementation in {"hf_vision", "hf_image_classification"}:
        return HuggingFaceVisionWorkload(spec, device, dtype)
    if implementation in {"hf_multimodal", "hf_clip", "hf_blip"}:
        return HuggingFaceMultimodalWorkload(spec, device, dtype)
    raise ValueError(f"Unsupported workload implementation: {implementation}")


__all__ = [
    "RunMetrics",
    "BaseSyntheticWorkload",
    "SyntheticTransformerWorkload",
    "SyntheticCnnWorkload",
    "SyntheticHybridWorkload",
    "HuggingFaceCausalLMWorkload",
    "HuggingFaceVisionWorkload",
    "HuggingFaceMultimodalWorkload",
    "build_workload",
]


