"""
Djinn v2.3 profiler that exercises the real remote execution stack using a
HuggingFace GPT-2 XL checkpoint.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.utils.server_spawner import RemoteServerManager
from djinn.backend.runtime.initialization import init_async
from djinn.core.coordinator import get_coordinator
from djinn.core.device_compatibility import RemoteAcceleratorSupport
from djinn.core.enhanced_model_manager import EnhancedModelManager
from djinn.frontend.core.lazy_tensor import LazyTensor
from djinn.frontend.core.srg_view import build_srg_view, summarize_srg_view

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
for name in ["djinn", "djinn.server", "djinn.backend"]:
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("DJINN_PROFILER_MODEL", "gpt2-xl")
DEFAULT_PROMPT = os.environ.get(
    "DJINN_PROFILER_PROMPT",
    "The disaggregated accelerator stack is orchestrated by Djinn because",
)
DEFAULT_BATCH_SIZE = int(os.environ.get("DJINN_PROFILER_BATCH", "1"))
DEFAULT_SEQ_LEN = int(os.environ.get("DJINN_PROFILER_SEQ_LEN", "32"))


@dataclass
class V23Profile:
    mode: str
    model_name: str
    batch_size: int
    registration_time_ms: float
    cold_execution_ms: float
    warm_execution_ms: float
    total_time_ms: float
    vmu_persistent_mb: float
    vmu_volatile_mb: float
    vmu_peak_mb: float
    remote_metrics: Dict[str, Any]
    logits_max_diff: float
    logits_mean_diff: float
    output_shape: Optional[tuple]
    srg_summary: Optional[Dict[str, Any]] = None


class DjinnV23Profiler:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        prompt: str = DEFAULT_PROMPT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seq_length: int = DEFAULT_SEQ_LEN,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.server_manager: Optional[RemoteServerManager] = None
        self.profiles = []
        self.capture_srg = os.environ.get("DJINN_PROFILER_CAPTURE_SRG", "0").lower() in {"1", "true", "yes"}

    def _load_model(self, device: torch.device) -> GPT2LMHeadModel:
        logger.info(f"Loading {self.model_id} on {device} ...")
        model = GPT2LMHeadModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()
        return model

    def _prepare_inputs(self) -> torch.Tensor:
        tokens = self.tokenizer(
            [self.prompt] * self.batch_size,
            padding="max_length",
            max_length=self.seq_length,
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"]

    def _extract_logits(self, output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict) and "logits" in output:
            return output["logits"]
        logits = getattr(output, "logits", None)
        if logits is not None:
            return logits
        raise ValueError(f"Unable to locate logits in output type {type(output)}")

    def _capture_srg_summary(self, input_ids: torch.Tensor) -> Optional[Dict[str, Any]]:
        if not self.capture_srg:
            return None

        logger.info("📐 Capturing SRG view for sample input...")
        try:
            RemoteAcceleratorSupport.initialize()
            analysis_model = GPT2LMHeadModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            analysis_model = analysis_model.to('remote_accelerator:0')
            with torch.no_grad():
                lazy_inputs = {'input_ids': LazyTensor.tensor(input_ids.clone())}
                lazy_output = analysis_model(**lazy_inputs)
            view = build_srg_view(lazy_output)
            summary = summarize_srg_view(view)
            logger.info(f"   SRG summary: {summary}")
            return summary
        except Exception as e:
            logger.warning(f"   ⚠️  SRG capture failed: {e}")
            return None

    def start_server(self) -> bool:
        self.server_manager = RemoteServerManager(host="127.0.0.1", port=5556, timeout=120)
        if self.server_manager.start():
            logger.info(f"✅ Djinn server started on port {self.server_manager.port}")
            return True
        logger.error("❌ Failed to launch Djinn server")
        return False

    def stop_server(self):
        if self.server_manager:
            self.server_manager.stop()
            logger.info("✅ Djinn server stopped")

    def profile_pytorch_baseline(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._load_model(device)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            start = time.perf_counter()
            outputs = model(input_ids, use_cache=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            cold_exec = (time.perf_counter() - start) * 1000

            warm_times = []
            for _ in range(2):
                start = time.perf_counter()
                _ = model(input_ids, use_cache=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warm_times.append((time.perf_counter() - start) * 1000)

        avg_warm = sum(warm_times) / len(warm_times)
        logits = self._extract_logits(outputs).detach().to("cpu")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"🔥 PyTorch baseline: cold={cold_exec:.2f}ms, warm={avg_warm:.2f}ms "
            f"(batch={self.batch_size}, seq={self.seq_length})"
        )
        return {
            "first_execution": cold_exec,
            "warm_execution": avg_warm,
            "output": logits,
        }

    async def profile_djinn_v23(self, input_ids: torch.Tensor, reference_logits: torch.Tensor) -> V23Profile:
        if not self.server_manager:
            raise RuntimeError("Server manager not started")

        server_address = f"localhost:{self.server_manager.port}"
        init_result = await init_async(server_address=server_address, auto_connect=True)
        if init_result.get("status") != "success":
            raise RuntimeError(f"Djinn init failed: {init_result}")

        manager = EnhancedModelManager(coordinator=get_coordinator())
        manager.use_model_cache = True
        remote_model = self._load_model(torch.device("cpu"))

        srg_summary = None
        if self.capture_srg:
            sample = input_ids[:1].clone()
            srg_summary = self._capture_srg_summary(sample)

        reg_start = time.perf_counter()
        fingerprint = await manager.register_model(remote_model, model_id=self.model_id)
        registration_time = (time.perf_counter() - reg_start) * 1000
        logger.info(f"📝 Model {fingerprint[:8]} registered in {registration_time:.2f}ms")

        inputs_dict = {"input_ids": input_ids.clone()}

        cold_start = time.perf_counter()
        remote_output = await manager.execute_model(remote_model, inputs_dict)
        cold_exec = (time.perf_counter() - cold_start) * 1000
        cold_metrics = manager.last_execution_metrics or {}

        warm_times = []
        warm_metrics = []
        for idx in range(2):
            start = time.perf_counter()
            _ = await manager.execute_model(remote_model, inputs_dict)
            warm_times.append((time.perf_counter() - start) * 1000)
            warm_metrics.append(manager.last_execution_metrics or {})
            logger.info(f"⚡ Warm execution {idx+1}: {warm_times[-1]:.2f}ms")

        avg_warm = sum(warm_times) / len(warm_times) if warm_times else cold_exec
        latest_metrics = warm_metrics[-1] if warm_metrics else cold_metrics

        remote_logits = self._extract_logits(remote_output).detach().to("cpu")
        max_diff = torch.max(torch.abs(remote_logits - reference_logits)).item()
        mean_diff = torch.mean(torch.abs(remote_logits - reference_logits)).item()

        profile = V23Profile(
            mode="djinn_v2.3",
            model_name=self.model_id,
            batch_size=self.batch_size,
            registration_time_ms=registration_time,
            cold_execution_ms=cold_exec,
            warm_execution_ms=avg_warm,
            total_time_ms=registration_time + cold_exec + avg_warm,
            vmu_persistent_mb=latest_metrics.get("vmu_persistent_mb", 0.0),
            vmu_volatile_mb=latest_metrics.get("vmu_volatile_mb", 0.0),
            vmu_peak_mb=latest_metrics.get("vmu_total_mb", 0.0),
            remote_metrics=latest_metrics,
            logits_max_diff=max_diff,
            logits_mean_diff=mean_diff,
            output_shape=tuple(remote_logits.shape),
            srg_summary=srg_summary,
        )

        self.profiles.append(profile)
        logger.info(
            f"✅ Djinn remote execution complete: cold={cold_exec:.2f}ms, warm={avg_warm:.2f}ms, "
            f"vmu_persistent={profile.vmu_persistent_mb:.1f}MB"
        )
        logger.info(f"🔍 Correctness diff: max={max_diff:.3e}, mean={mean_diff:.3e}")
        return profile

    def save_results(self, output_file: str):
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model_id": self.model_id,
                    "prompt": self.prompt,
                    "profiles": [asdict(p) for p in self.profiles],
                },
                f,
                indent=2,
            )
        logger.info(f"📄 Results saved to {output_file}")


async def main():
    profiler = DjinnV23Profiler()
    input_ids = profiler._prepare_inputs()

    logger.info("=== PyTorch Baseline ===")
    baseline = profiler.profile_pytorch_baseline(input_ids.clone())

    logger.info("\n=== Djinn v2.3 ===")
    if not profiler.start_server():
        return

    try:
        await asyncio.sleep(2.0)
        await profiler.profile_djinn_v23(input_ids, baseline["output"])
        profiler.save_results("djinn_v2_3_profiling_results.json")
    finally:
        profiler.stop_server()


if __name__ == "__main__":
    asyncio.run(main())

