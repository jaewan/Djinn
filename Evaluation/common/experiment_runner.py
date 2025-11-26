"""
Reusable experiment harness for Evaluation scripts.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .metrics import summarize_fields
from .workloads import RunMetrics, build_workload


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


@dataclass
class BaselineResult:
    baseline: str
    runner_type: str
    runs: List[Dict[str, float]]
    aggregates: Dict[str, Dict]
    metadata: Dict
    derived: Dict


class LocalSyntheticBaselineRunner:
    """Executes synthetic workloads locally (used for smoke tests)."""

    def __init__(self, baseline_cfg: Dict, experiment_cfg: Dict):
        self.cfg = baseline_cfg
        self.exp_cfg = experiment_cfg
        self.name = baseline_cfg["name"]
        self.device = baseline_cfg.get("device", experiment_cfg.get("device", _default_device()))
        self.dtype = baseline_cfg.get("dtype", experiment_cfg.get("dtype", "float16"))
        self.runs = baseline_cfg.get("runs", experiment_cfg.get("runs", 5))
        self.warmup_runs = baseline_cfg.get("warmup_runs", experiment_cfg.get("warmup_runs", 1))

    def run(self, workload_cfg: Dict) -> BaselineResult:
        impl = workload_cfg["implementation"]
        spec = copy.deepcopy(workload_cfg.get("params", {}))
        workload = build_workload(impl, spec, self.device, self.dtype)

        for _ in range(self.warmup_runs):
            workload.run_once()

        runs: List[Dict[str, float]] = []
        for run_id in range(1, self.runs + 1):
            metrics = workload.run_once()
            record = _metrics_to_record(run_id, metrics)
            runs.append(record)

        aggregates = summarize_fields(runs, ["latency_ms", "throughput_units_per_s", "total_data_mb"])
        metadata = {
            **workload.metadata(),
            "baseline": self.name,
            "runner_type": "local_synthetic",
        }
        return BaselineResult(
            baseline=self.name,
            runner_type="local_synthetic",
            runs=runs,
            aggregates=aggregates,
            metadata=metadata,
            derived={},
        )


class ScalingBaselineRunner:
    """Scales an existing baseline's metrics for placeholder comparisons."""

    def __init__(self, baseline_cfg: Dict, experiment_cfg: Dict):
        self.cfg = baseline_cfg
        self.exp_cfg = experiment_cfg
        self.name = baseline_cfg["name"]
        self.reference = baseline_cfg["relative_to"]
        self.latency_scale = baseline_cfg.get("latency_scale", 1.0)
        self.latency_offset_ms = baseline_cfg.get("latency_offset_ms", 0.0)
        self.input_scale = baseline_cfg.get("input_scale", baseline_cfg.get("data_scale", 1.0))
        self.output_scale = baseline_cfg.get("output_scale", baseline_cfg.get("data_scale", 1.0))
        self.input_offset_mb = baseline_cfg.get("input_offset_mb", 0.0)
        self.output_offset_mb = baseline_cfg.get("output_offset_mb", 0.0)
        self.throughput_scale = baseline_cfg.get("throughput_scale")
        self.notes = baseline_cfg.get("notes")

    def run(self, workload_cfg: Dict, prior_results: Dict[str, BaselineResult]) -> BaselineResult:
        if self.reference not in prior_results:
            raise ValueError(f"Baseline '{self.name}' depends on '{self.reference}', which has not run yet.")
        ref = prior_results[self.reference]
        runs: List[Dict[str, float]] = []
        for ref_run in ref.runs:
            latency_ms = ref_run["latency_ms"] * self.latency_scale + self.latency_offset_ms
            input_mb = ref_run["input_mb"] * self.input_scale + self.input_offset_mb
            output_mb = ref_run["output_mb"] * self.output_scale + self.output_offset_mb
            total_mb = input_mb + output_mb
            throughput = ref_run["throughput_units_per_s"]
            if self.throughput_scale is not None:
                throughput = throughput * self.throughput_scale
            elif self.latency_scale != 0:
                throughput = throughput / self.latency_scale
            runs.append(
                {
                    "run_id": ref_run["run_id"],
                    "latency_ms": latency_ms,
                    "input_mb": input_mb,
                    "output_mb": output_mb,
                    "total_data_mb": total_mb,
                    "units_processed": ref_run["units_processed"],
                    "throughput_units_per_s": throughput,
                }
            )

        aggregates = summarize_fields(runs, ["latency_ms", "throughput_units_per_s", "total_data_mb"])
        metadata = {
            "baseline": self.name,
            "runner_type": "scaling",
            "reference": self.reference,
            "notes": self.notes,
        }
        return BaselineResult(
            baseline=self.name,
            runner_type="scaling",
            runs=runs,
            aggregates=aggregates,
            metadata=metadata,
            derived={},
        )


class RemoteDjinnBaselineRunner:
    """Executes synthetic workloads remotely via Djinn server."""

    def __init__(self, baseline_cfg: Dict, experiment_cfg: Dict):
        self.cfg = baseline_cfg
        self.exp_cfg = experiment_cfg
        self.name = baseline_cfg["name"]
        self.dtype = baseline_cfg.get("dtype", experiment_cfg.get("dtype", "float16"))
        self.runs = baseline_cfg.get("runs", experiment_cfg.get("runs", 5))
        self.warmup_runs = baseline_cfg.get("warmup_runs", experiment_cfg.get("warmup_runs", 1))
        self.semantic_aware = baseline_cfg.get("semantic_aware", True)
        self.server_address = experiment_cfg.get("djinn_server_address", "localhost:5556")
        self._manager = None
        self._model = None

    async def _ensure_manager(self):
        """
        Initialize EnhancedModelManager if not already done.
        
        DESIGN NOTE (Senior Engineer):
        - Uses public APIs (init_async, get_coordinator) only
        - No direct manipulation of _runtime_state (internal implementation detail)
        - Simple, defensive logic that's easy to reason about
        - Works correctly across event loops since _runtime_state is a global singleton
        """
        if self._manager is not None:
            return self._manager

        from djinn.core.enhanced_model_manager import EnhancedModelManager
        from djinn.backend.runtime.initialization import get_coordinator, init_async, _runtime_state
        
        # Try to get coordinator (may be None if not initialized yet)
        coordinator = get_coordinator()
        
        # If coordinator is None, initialize Djinn in this async context
        # Note: init_async() may return success even if coordinator is None (e.g., connection failed)
        # So we always check coordinator after init_async()
        if coordinator is None:
            server_address = self.server_address or "localhost:5556"
            result = await init_async(
                server_address=server_address,
                auto_connect=True,
                profiling=False,
            )
            if result.get("status") != "success":
                raise RuntimeError(
                    f"Djinn init_async failed: {result.get('error')}. "
                    f"Check that the Djinn server is running at {server_address}"
                )
            
            # Get coordinator after initialization
            # Use _runtime_state directly here since get_coordinator() calls ensure_initialized()
            # which might trigger auto-init in a different way
            coordinator = _runtime_state.coordinator
            
            # If still None, try get_coordinator() as fallback
            if coordinator is None:
                coordinator = get_coordinator()
        
        # Final check - coordinator should be available now
        if coordinator is None:
            raise RuntimeError(
                f"Djinn coordinator unavailable after initialization. "
                f"Server: {self.server_address or 'localhost:5556'}. "
                f"Initialized: {_runtime_state.initialized}, "
                f"Coordinator in state: {_runtime_state.coordinator is not None}. "
                f"This usually means connection to server failed. "
                f"Check that the Djinn server is running and accessible."
            )

        self._manager = EnhancedModelManager(
            coordinator=coordinator,
            server_address=self.server_address,
        )
        self._manager.use_model_cache = True
        return self._manager

    async def _register_model(self, model: torch.nn.Module, model_id: str):
        """Register model with remote server."""
        manager = await self._ensure_manager()
        await manager.register_model(model, model_id=model_id)

    async def _run_once_remote(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, float, float, float]:
        """Execute one run remotely and return (output, input_mb, output_mb, latency_ms)."""
        manager = await self._ensure_manager()
        
        # Ensure coordinator is set on manager (it might have been None during init)
        if manager.coordinator is None:
            from djinn.backend.runtime.initialization import get_coordinator as get_rt_coord
            coordinator = get_rt_coord()
            if coordinator is None:
                try:
                    from djinn.core.coordinator import get_coordinator
                    coordinator = get_coordinator()
                except RuntimeError:
                    raise RuntimeError("Djinn coordinator unavailable in async context")
            manager.coordinator = coordinator
        
        # Track data transfer
        input_bytes = sum(t.element_size() * t.numel() for t in inputs.values())
        
        # Phase 3: Use djinn.session() for semantic hints
        import djinn
        if self.semantic_aware:
            # For semantic-aware execution, use session context manager
            with djinn.session(phase="decode", priority="normal"):
                start = time.perf_counter()
                result = await manager.execute_model(model, inputs)
        else:
            # For semantic-blind execution, don't use semantic hints
            start = time.perf_counter()
            result = await manager.execute_model(model, inputs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        
        # Extract output tensor
        if torch.is_tensor(result):
            output = result
        elif isinstance(result, dict):
            # For HuggingFace models, result might be ModelOutput with logits
            # Use explicit checks instead of 'or' to avoid tensor boolean evaluation
            if "logits" in result:
                output = result["logits"]
            elif "last_hidden_state" in result:
                output = result["last_hidden_state"]
            else:
                # Get first tensor value from dict
                for value in result.values():
                    if torch.is_tensor(value):
                        output = value
                        break
                else:
                    raise RuntimeError(f"No tensor found in result dict: {list(result.keys())}")
        else:
            raise RuntimeError(f"Unexpected result type: {type(result)}")
        
        output_bytes = output.element_size() * output.numel()
        
        input_mb = input_bytes / (1024**2)
        output_mb = output_bytes / (1024**2)
        
        return output.cpu(), input_mb, output_mb, latency_ms

    async def run_async(self, workload_cfg: Dict) -> BaselineResult:
        """Async version of run() for remote execution."""
        impl = workload_cfg["implementation"]
        spec = copy.deepcopy(workload_cfg.get("params", {}))
        
        # Build workload model on CPU (will be registered on server)
        workload = build_workload(impl, spec, "cpu", self.dtype)
        model = workload.model
        model_id = f"{workload_cfg['name']}_{self.name}"
        
        # Register model with remote server
        await self._register_model(model, model_id)
        self._model = model
        
        # Prepare inputs based on workload type
        def clone_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            cloned = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    cloned[key] = value.clone()
                else:
                    cloned[key] = value
            return cloned

        if hasattr(workload, 'prepare_inputs'):
            def get_inputs():
                prepared = workload.prepare_inputs()
                return clone_inputs(prepared)
        elif hasattr(workload, '_sample_inputs'):
            def get_inputs():
                return {"x": workload._sample_inputs()}
        else:
            raise RuntimeError(f"Workload {workload_cfg['name']} lacks prepare_inputs()")

        if hasattr(workload, 'units_from_output'):
            def get_units(inputs, output):
                return workload.units_from_output(inputs, output)
        elif hasattr(workload, '_units_processed'):
            def get_units(inputs, output):
                if "x" not in inputs:
                    raise KeyError("Synthetic workload inputs missing 'x'")
                return workload._units_processed(inputs["x"], output)
        else:
            def get_units(inputs, output):
                return 0.0
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            inputs = get_inputs()
            await self._run_once_remote(model, inputs)
        
        # Actual runs
        runs: List[Dict[str, float]] = []
        for run_id in range(1, self.runs + 1):
            inputs = get_inputs()
            output, input_mb, output_mb, latency_ms = await self._run_once_remote(model, inputs)
            
            units = get_units(inputs, output)
            throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
            
            runs.append({
                "run_id": run_id,
                "latency_ms": latency_ms,
                "input_mb": input_mb,
                "output_mb": output_mb,
                "total_data_mb": input_mb + output_mb,
                "units_processed": units,
                "throughput_units_per_s": throughput,
            })
        
        aggregates = summarize_fields(runs, ["latency_ms", "throughput_units_per_s", "total_data_mb"])
        metadata = {
            **workload.metadata(),
            "baseline": self.name,
            "runner_type": "remote_djinn",
            "semantic_aware": self.semantic_aware,
        }
        return BaselineResult(
            baseline=self.name,
            runner_type="remote_djinn",
            runs=runs,
            aggregates=aggregates,
            metadata=metadata,
            derived={},
        )

    def run(self, workload_cfg: Dict) -> BaselineResult:
        """Synchronous wrapper for async run."""
        # Note: Djinn initialization should happen before this runner is created
        # (in the main script via ensure_initialized). We verify it here but don't
        # re-initialize as that might cause issues with event loops.
        return asyncio.run(self.run_async(workload_cfg))


RUNNER_TYPES = {
    "local_synthetic": LocalSyntheticBaselineRunner,
    "scaling": ScalingBaselineRunner,
    "remote_djinn": RemoteDjinnBaselineRunner,
}


def _metrics_to_record(run_id: int, metrics: RunMetrics) -> Dict[str, float]:
    return {
        "run_id": run_id,
        "latency_ms": metrics.latency_ms,
        "input_mb": metrics.input_mb,
        "output_mb": metrics.output_mb,
        "total_data_mb": metrics.input_mb + metrics.output_mb,
        "units_processed": metrics.units_processed,
        "throughput_units_per_s": metrics.throughput_units_per_s,
    }


class ExperimentRunner:
    """Coordinates workloads and baselines for a given experiment."""

    def __init__(self, experiment_cfg: Dict, baselines_cfg: List[Dict]):
        self.experiment_cfg = experiment_cfg
        self.baselines_cfg = baselines_cfg

    def run_workloads(self, workloads: List[Dict]) -> List[Dict]:
        results = []
        for workload in workloads:
            results.append(self._run_single_workload(workload))
        return results

    def _run_single_workload(self, workload_cfg: Dict) -> Dict:
        baseline_outputs: Dict[str, BaselineResult] = {}
        ordered_results: List[Dict] = []
        for baseline_cfg in self.baselines_cfg:
            runner_type = baseline_cfg.get("type", "local_synthetic")
            runner_cls = RUNNER_TYPES.get(runner_type)
            if runner_cls is None:
                raise ValueError(f"Unsupported baseline runner type: {runner_type}")
            runner = runner_cls(baseline_cfg, self.experiment_cfg)
            if isinstance(runner, ScalingBaselineRunner):
                result = runner.run(workload_cfg, baseline_outputs)
            else:
                result = runner.run(workload_cfg)
            baseline_outputs[result.baseline] = result
            ordered_results.append(_baseline_result_to_dict(result))

        self._attach_derived_metrics(ordered_results)

        return {
            "workload": workload_cfg["name"],
            "category": workload_cfg.get("category"),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "results": ordered_results,
        }

    def _attach_derived_metrics(self, baseline_results: List[Dict]) -> None:
        ref_name = self.experiment_cfg.get("reference_baseline")
        blind_name = self.experiment_cfg.get("blind_baseline")
        target_name = self.experiment_cfg.get("target_baseline")
        result_map = {result["baseline"]: result for result in baseline_results}

        def mean_latency(name: str) -> Optional[float]:
            res = result_map.get(name)
            if not res:
                return None
            return res["aggregates"]["latency_ms"]["mean"]

        def mean_data(name: str) -> Optional[float]:
            res = result_map.get(name)
            if not res:
                return None
            return res["aggregates"]["total_data_mb"]["mean"]

        if ref_name and ref_name in result_map:
            ref_latency = mean_latency(ref_name)
            if ref_latency:
                for result in baseline_results:
                    if result["baseline"] == ref_name:
                        continue
                    latency = result["aggregates"]["latency_ms"]["mean"]
                    if latency is None:
                        continue
                    overhead_pct = (latency / ref_latency - 1.0) * 100.0
                    result.setdefault("derived", {})
                    result["derived"][f"latency_overhead_pct_vs_{ref_name}"] = overhead_pct

        if blind_name and target_name and blind_name in result_map and target_name in result_map:
            blind_latency = mean_latency(blind_name)
            target_latency = mean_latency(target_name)
            blind_data = mean_data(blind_name)
            target_data = mean_data(target_name)
            target = result_map[target_name]
            target.setdefault("derived", {})
            if blind_latency and target_latency:
                target["derived"][f"speedup_vs_{blind_name}"] = blind_latency / target_latency
            if blind_data and target_data and blind_data > 0:
                data_savings_pct = (blind_data - target_data) / blind_data * 100.0
                target["derived"][f"data_savings_pct_vs_{blind_name}"] = data_savings_pct

            ref_latency = mean_latency(self.experiment_cfg.get("reference_baseline", ""))
            if ref_latency and target_latency:
                overhead_pct = (target_latency / ref_latency - 1.0) * 100.0
                if overhead_pct != 0:
                    data_savings_pct = target["derived"].get(f"data_savings_pct_vs_{blind_name}")
                    if data_savings_pct is not None:
                        target["derived"]["semantic_efficiency_ratio"] = data_savings_pct / overhead_pct


def _baseline_result_to_dict(result: BaselineResult) -> Dict:
    return {
        "baseline": result.baseline,
        "runner_type": result.runner_type,
        "runs": result.runs,
        "aggregates": result.aggregates,
        "metadata": result.metadata,
        "derived": result.derived,
    }


__all__ = ["ExperimentRunner"]


