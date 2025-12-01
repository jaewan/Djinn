"""
vLLM baseline runner for Experiment 5.1.

IMPORTANT (OSDI Review - Critique 3):
  vLLM uses highly optimized C++ PagedAttention kernels that are NOT present in 
  the vanilla HuggingFace `model.generate()` used by Djinn/RPC baselines.
  
  This baseline is provided for CONTEXT on kernel-level optimizations, 
  NOT for direct architectural comparison. For fair architectural comparison, 
  use: native_pytorch vs pytorch_rpc vs semantic_blind vs full_djinn.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional

import requests

# Import these after experiment_runner has fully initialized to avoid circular import
BaselineResult = None
_tensor_mb = None
_tensor_dict_mb = None
summarize_fields = None
build_workload = None


class vLLMBaselineRunner:
    """Run workloads against a remote vLLM OpenAI-compatible server."""

    def __init__(self, baseline_cfg: Dict, experiment_cfg: Dict):
        # Lazy import to avoid circular dependency
        global BaselineResult, _tensor_mb, _tensor_dict_mb, summarize_fields, build_workload
        if BaselineResult is None:
            from Evaluation.common.experiment_runner import BaselineResult as _BR
            from Evaluation.common.experiment_runner import _tensor_mb as _tm
            from Evaluation.common.experiment_runner import _tensor_dict_mb as _tdm
            from Evaluation.common.experiment_runner import summarize_fields as _sf
            from Evaluation.common.workloads import build_workload as _bw
            BaselineResult = _BR
            _tensor_mb = _tm
            _tensor_dict_mb = _tdm
            summarize_fields = _sf
            build_workload = _bw
        
        self.cfg = baseline_cfg
        self.exp_cfg = experiment_cfg
        self.name = baseline_cfg["name"]
        self.base_url = baseline_cfg.get("base_url", "http://localhost:8000/v1")
        self.api_key = baseline_cfg.get("api_key", "dummy")
        self.runs = baseline_cfg.get("runs", experiment_cfg.get("runs", 5))
        self.warmup_runs = baseline_cfg.get("warmup_runs", experiment_cfg.get("warmup_runs", 1))

    def run(self, workload_cfg: Dict) -> BaselineResult:
        """Execute workload against vLLM and return results."""
        if workload_cfg["implementation"] != "hf_causal_lm":
            raise NotImplementedError(f"vLLM baseline only supports hf_causal_lm, got {workload_cfg['implementation']}")

        spec = copy.deepcopy(workload_cfg.get("params", {}))
        workload = build_workload("hf_causal_lm", spec, "cpu", self.exp_cfg.get("dtype", "float16"))

        # Warmup
        for _ in range(self.warmup_runs):
            self._send_request(spec, workload)

        # Timed runs
        run_records = []
        for run_id in range(1, self.runs + 1):
            start = time.perf_counter()
            response = self._send_request(spec, workload)
            latency_ms = (time.perf_counter() - start) * 1000.0

            # Extract token counts
            usage = response.get("usage", {})
            completion_tokens = usage.get("completion_tokens", spec.get("new_tokens", 32))
            prompt_tokens = usage.get("prompt_tokens", len(workload.prompt_text.split()))
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            run_records.append({
                "run_id": run_id,
                "latency_ms": latency_ms,
                "input_mb": 0.0,  # Approximate (not serialized)
                "output_mb": 0.0,  # Approximate (not serialized)
                "total_data_mb": 0.0,
                "units_processed": float(completion_tokens),
                "throughput_units_per_s": completion_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0,
            })

        aggregates = summarize_fields(run_records, ["latency_ms", "throughput_units_per_s", "total_data_mb"])

        metadata = {
            **workload.metadata(),
            "baseline": self.name,
            "runner_type": "vllm",
            "vllm_base_url": self.base_url,
        }

        return BaselineResult(
            baseline=self.name,
            runner_type="vllm",
            runs=run_records,
            aggregates=aggregates,
            metadata=metadata,
            derived={},
        )

    def _send_request(self, spec: Dict, workload) -> Dict:
        """Send request to vLLM server."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        prompt = spec.get("prompt_text", "Describe the Djinn Tensor Operating System.")
        max_tokens = spec.get("new_tokens", 32)

        payload = {
            "model": spec["model_id"],
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        response = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

