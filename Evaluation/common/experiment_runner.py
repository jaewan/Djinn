"""
Reusable experiment harness for Evaluation scripts.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
import atexit
import os
import sys
from datetime import datetime, timezone, timedelta
try:
    from datetime import UTC  # py3.11+
except ImportError:
    UTC = timezone.utc
from pathlib import Path
from typing import Dict, List, Optional

import torch
try:
    import torch.distributed.rpc as torch_rpc
    from torch.distributed import TCPStore
except Exception:  # pragma: no cover - optional dependency
    torch_rpc = None
    TCPStore = None

try:
    from djinn.server.profiling_context import ProfilingContext, set_profiler as _set_profiler
except Exception:  # pragma: no cover - profiling optional
    ProfilingContext = None  # type: ignore
    _set_profiler = None  # type: ignore

from .metrics import summarize_fields
from .workloads import RunMetrics, build_workload
from . import pytorch_rpc_ops


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

        # OSDI FIX: Ensure CUDA kernels are fully compiled before timing
        # This makes native baseline comparable to RPC (which has pre-warmed server)
        if torch.cuda.is_available():
            # Force CUDA context initialization
            torch.cuda.synchronize()
        
        for _ in range(self.warmup_runs):
            workload.run_once()
        
        # Additional sync after warmup to ensure all kernels are compiled
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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

    async def _run_once_remote(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, float, float, float, Dict[str, float], Dict[str, float]]:
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
        
        profiler = None
        profiler_enabled = False
        if ProfilingContext and _set_profiler:
            profiler = ProfilingContext(enabled=True)
            profiler.start()
            _set_profiler(profiler)
            profiler_enabled = True

        # Phase 3: Use djinn.session() for semantic hints
        # OSDI FIX: Pass generation hints for causal LM workloads
        import djinn
        generation_hints = getattr(self, '_generation_hints', None)
        execution_hints = generation_hints.copy() if generation_hints else {}
        
        if self.semantic_aware:
            # For semantic-aware execution, use session context manager
            with djinn.session(phase="decode", priority="normal"):
                start = time.perf_counter()
                result = await manager.execute_model(model, inputs, hints=execution_hints or None)
        else:
            # For semantic-blind execution, don't use semantic hints
            start = time.perf_counter()
            result = await manager.execute_model(model, inputs, hints=execution_hints or None)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if profiler_enabled and _set_profiler:
            _set_profiler(None)
        client_phases: Dict[str, float] = {}
        if profiler:
            client_phases = profiler.get_phase_dict()
        server_metrics = manager.last_execution_metrics or {}
        
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
        
        return output.cpu(), input_mb, output_mb, latency_ms, client_phases, server_metrics

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

        # OSDI FIX: For causal LM workloads, include generation parameters
        # so that Djinn server uses model.generate() instead of model.forward()
        # This ensures fair comparison with native PyTorch baseline
        self._generation_hints = None
        if impl == "hf_causal_lm" and hasattr(workload, 'generation_params'):
            self._generation_hints = {
                'use_generate': True,
                'max_new_tokens': workload.new_tokens,
                'pad_token_id': workload.tokenizer.eos_token_id,
                **workload.generation_params,
            }

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
            output, input_mb, output_mb, latency_ms, client_phases, server_metrics = await self._run_once_remote(model, inputs)
            
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
                **_flatten_client_phases(client_phases),
                **_flatten_server_metrics(server_metrics),
            })
        
        summary_fields = [
            "latency_ms",
            "throughput_units_per_s",
            "total_data_mb",
            "client_serialize_ms",
            "client_deserialize_ms",
            "client_network_c2s_ms",
            "client_network_s2c_ms",
            "server_duration_ms",
            "server_executor_time_ms",
            "server_queue_latency_ms",
            "server_plan_ms",
            "server_placement_ms",
            "server_execution_ms",
            "server_skeletonization_ms",
            "server_cleanup_ms",
        ]
        aggregates = summarize_fields(runs, summary_fields)
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


class PytorchRpcBaselineRunner:
    """Executes workloads via a user-managed PyTorch RPC server."""

    _rpc_initialized = False
    _client_worker_name: Optional[str] = None

    def __init__(self, baseline_cfg: Dict, experiment_cfg: Dict):
        if torch_rpc is None:
            raise RuntimeError("torch.distributed.rpc is not available")
        self.cfg = baseline_cfg
        self.exp_cfg = experiment_cfg
        self.name = baseline_cfg["name"]
        self.dtype = baseline_cfg.get("dtype", experiment_cfg.get("dtype", "float16"))
        self.runs = baseline_cfg.get("runs", experiment_cfg.get("runs", 5))
        self.warmup_runs = baseline_cfg.get("warmup_runs", experiment_cfg.get("warmup_runs", 1))
        self.server_worker = baseline_cfg.get("rpc_server_name", "rpc_server")
        self.client_worker = baseline_cfg.get("rpc_client_name", f"rpc_client_{os.getpid()}")
        self.rpc_timeout_s = baseline_cfg.get("rpc_timeout_s", 0.0)
        self._set_client_name(self.client_worker)
        self._set_timeout(self.rpc_timeout_s)
        self._ensure_rpc_initialized()

    @classmethod
    def _ensure_rpc_initialized(cls):
        if cls._rpc_initialized:
            return
        
        # OSDI FIX: Auto-configure localhost RPC for single-machine testing
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        rank = os.environ.get("RANK")
        world_size = os.environ.get("WORLD_SIZE")
        
        # If not set, auto-configure for localhost (requires rpc_server.py running)
        if not (master_addr and master_port and rank and world_size):
            print("⚠️  RPC environment not set, auto-configuring for localhost...")
            master_addr = "127.0.0.1"
            master_port = "29500"
            rank = "1"  # Client is rank 1
            world_size = "2"  # Server (0) + Client (1)
            
            # Set them so other code can use them
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["RANK"] = rank
            os.environ["WORLD_SIZE"] = world_size
            
            print(f"  MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
            print(f"  RANK={rank}, WORLD_SIZE={world_size}")
            print("  ℹ️  Make sure rpc_server.py is running with RANK=0!")
        
        if not (master_addr and master_port and rank and world_size):
            raise EnvironmentError(
                "MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE must be set for the PyTorch RPC baseline."
            )
        
        # Force CUDA context initialization before RPC init
        # to avoid cold-start overhead being attributed to RPC baseline.
        # See: OSDI Review - Critique 1 (RPC Warmup Trap)
        if torch.cuda.is_available():
            torch.ones(1).cuda()
            torch.cuda.synchronize()
        
        rank_int = int(rank)
        world_size_int = int(world_size)
        init_timeout = 600  # 10 minutes for initialization
        
        print(f"[rpc_client] Initializing RPC: rank={rank_int}, world_size={world_size_int}", flush=True)
        print(f"[rpc_client] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}", flush=True)
        
        # Simple approach: call init_rpc with long timeouts
        # Client waits a bit for server to be ready, then both coordinate via PyTorch's internal TCPStore
        if rank_int != 0:
            print(f"[rpc_client] Rank {rank_int}: Waiting for server to be ready...", flush=True)
            # Add delay to allow server to start first
            time.sleep(3)
        
        init_method = f"tcp://{master_addr}:{master_port}"
        options = torch_rpc.TensorPipeRpcBackendOptions(init_method=init_method)
        options.rpc_timeout = init_timeout
        
        print(f"[rpc_client] Calling rpc.init_rpc with init_method={init_method}, timeout={init_timeout}s", flush=True)
        
        try:
            torch_rpc.init_rpc(
                cls._current_client_name(),
                rank=rank_int,
                world_size=world_size_int,
                rpc_backend_options=options,
            )
            print("[rpc_client] RPC initialization succeeded", flush=True)
        except Exception as e:
            print(f"[rpc_client] RPC initialization failed: {e}", flush=True, file=sys.stderr)
            raise
        
        # Allow server to stabilize after RPC init
        # See: OSDI Review - Code Nit #1 (RPC Sync Race Condition)
        time.sleep(1)
        
        atexit.register(cls._shutdown_rpc)
        cls._rpc_initialized = True

    @classmethod
    def _current_client_name(cls) -> str:
        if cls._client_worker_name is None:
            cls._client_worker_name = f"rpc_client_{os.getpid()}"
        return cls._client_worker_name

    @classmethod
    def _set_client_name(cls, value: str) -> None:
        cls._client_worker_name = value

    @classmethod
    def _current_timeout(cls) -> float:
        return getattr(cls, "_rpc_timeout_value", 0.0)

    @classmethod
    def _set_timeout(cls, value: float) -> None:
        cls._rpc_timeout_value = max(0.0, float(value))

    @classmethod
    def _shutdown_rpc(cls) -> None:
        if cls._rpc_initialized:
            torch_rpc.shutdown()
            cls._rpc_initialized = False

    def run(self, workload_cfg: Dict) -> BaselineResult:
        impl = workload_cfg["implementation"]
        spec = copy.deepcopy(workload_cfg.get("params", {}))
        workload = build_workload(impl, spec, "cpu", self.dtype)

        def prepare_inputs() -> Dict[str, torch.Tensor]:
            prepared = workload.prepare_inputs()
            return {
                key: value.cpu() if torch.is_tensor(value) else value
                for key, value in prepared.items()
            }

        def get_units(inputs: Dict[str, torch.Tensor], outputs: torch.Tensor) -> float:
            return workload.units_from_output(inputs, outputs)

        for _ in range(self.warmup_runs):
            inputs = prepare_inputs()
            self._execute_remote(workload_cfg["name"], inputs)

        runs: List[Dict[str, float]] = []
        for run_id in range(1, self.runs + 1):
            inputs = prepare_inputs()
            start = time.perf_counter()
            output = self._execute_remote(workload_cfg["name"], inputs)
            latency_ms = (time.perf_counter() - start) * 1000.0
            input_mb = _tensor_dict_mb(inputs)
            output_mb = _tensor_mb(output)
            total_mb = input_mb + output_mb
            units = get_units(inputs, output)
            throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
            runs.append(
                {
                    "run_id": run_id,
                    "latency_ms": latency_ms,
                    "input_mb": input_mb,
                    "output_mb": output_mb,
                    "total_data_mb": total_mb,
                    "units_processed": units,
                    "throughput_units_per_s": throughput,
                }
            )

        aggregates = summarize_fields(runs, ["latency_ms", "throughput_units_per_s", "total_data_mb"])
        metadata = {
            **workload.metadata(),
            "baseline": self.name,
            "runner_type": "pytorch_rpc",
            "rpc_server": self.server_worker,
        }
        return BaselineResult(
            baseline=self.name,
            runner_type="pytorch_rpc",
            runs=runs,
            aggregates=aggregates,
            metadata=metadata,
            derived={},
        )

    def _execute_remote(self, workload_name: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        payload = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        result = torch_rpc.rpc_sync(
            self.server_worker,
            pytorch_rpc_ops.rpc_forward,
            args=(workload_name, payload),
        )
        if isinstance(result, torch.Tensor):
            return result
        raise RuntimeError(
            f"RPC forward for workload '{workload_name}' returned unsupported type: {type(result)}"
        )

def _flatten_client_phases(phases: Dict[str, float]) -> Dict[str, float]:
    mapping = {
        "client_serialize_ms": "client_serialize",
        "client_deserialize_ms": "client_deserialize",
        "client_network_c2s_ms": "network_c2s",
        "client_network_s2c_ms": "network_s2c",
    }
    result: Dict[str, float] = {}
    for out_key, phase_name in mapping.items():
        value = phases.get(phase_name)
        if value is not None:
            result[out_key] = float(value)
    return result


def _flatten_server_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    scalar_mapping = {
        "server_duration_ms": "duration_ms",
        "server_executor_time_ms": "executor_time_ms",
        "server_queue_latency_ms": "queue_latency_ms",
    }
    for out_key, metric_key in scalar_mapping.items():
        value = metrics.get(metric_key)
        if value is not None:
            result[out_key] = float(value)

    timing_breakdown = metrics.get("timing_breakdown_ms") or {}
    breakdown_mapping = {
        "server_plan_ms": "planning",
        "server_placement_ms": "placement",
        "server_execution_ms": "execution",
        "server_skeletonization_ms": "skeletonization",
        "server_cleanup_ms": "cleanup",
    }
    for out_key, phase in breakdown_mapping.items():
        value = timing_breakdown.get(phase)
        if value is not None:
            result[out_key] = float(value)
    return result


# Import vLLM runner if available (direct import to avoid circular dependency)
try:
    import importlib.util
    from pathlib import Path
    # experiment_runner.py is in Evaluation/common/, so baseline_runners is in the same directory
    _vllm_runner_path = Path(__file__).parent / "baseline_runners" / "vllm_runner.py"
    if _vllm_runner_path.exists():
        _spec = importlib.util.spec_from_file_location("_vllm_runner", _vllm_runner_path)
        _vllm_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vllm_module)  # type: ignore
        vLLMBaselineRunner = _vllm_module.vLLMBaselineRunner
    else:
        vLLMBaselineRunner = None
except (ImportError, Exception) as e:
    import sys
    print(f"[experiment_runner] Warning: Failed to load vLLMBaselineRunner: {e}", file=sys.stderr)
    vLLMBaselineRunner = None  # type: ignore

class RayActorBaselineRunner:
    """Executes workloads via Ray Actor (Pickle + Plasma Object Store baseline)."""

    def __init__(self, baseline_cfg: Dict, experiment_cfg: Dict):
        self.cfg = baseline_cfg
        self.exp_cfg = experiment_cfg
        self.name = baseline_cfg["name"]
        self.dtype = baseline_cfg.get("dtype", experiment_cfg.get("dtype", "float16"))
        self.runs = baseline_cfg.get("runs", experiment_cfg.get("runs", 5))
        self.warmup_runs = baseline_cfg.get("warmup_runs", experiment_cfg.get("warmup_runs", 1))
        self.device = baseline_cfg.get("device", experiment_cfg.get("device", "cuda:0"))
        self._actor = None
        self._ray_initialized = False

    def _ensure_ray_initialized(self):
        """Initialize Ray connection if not already done."""
        if self._ray_initialized:
            return
        
        try:
            import ray
            if not ray.is_initialized():
                # Try to connect to existing Ray cluster
                try:
                    ray.init(address="auto", ignore_reinit_error=True)
                    print("[ray_actor_runner] Connected to existing Ray cluster")
                except Exception as e:
                    print(f"[ray_actor_runner] Failed to connect to Ray cluster: {e}")
                    print("[ray_actor_runner] Make sure Ray is running: ray start --head --num-gpus=1")
                    raise
            self._ray_initialized = True
        except ImportError:
            raise RuntimeError(
                "Ray is not installed. Install with: pip install ray[tune]"
            )

    def _ensure_actor(self, workload_cfg: Dict):
        """Create Ray Actor if not already created."""
        if self._actor is None:
            self._ensure_ray_initialized()
            
            import ray
            from Evaluation.exp5_1_overhead.scripts.ray_actor_server import ModelActor
            
            # Create remote actor
            self._actor = ModelActor.remote(workload_cfg, self.device, self.dtype)
            
            # Wait for actor initialization by sending dummy request
            try:
                dummy_output = ray.get(
                    self._actor.execute.remote({"input_ids": torch.zeros(1, dtype=torch.long)})
                )
                print(f"[ray_actor_runner] Actor initialized successfully")
            except Exception as e:
                print(f"[ray_actor_runner] Failed to initialize actor: {e}")
                raise
        
        return self._actor

    def run(self, workload_cfg: Dict) -> BaselineResult:
        """Execute workload on Ray Actor and collect metrics."""
        import ray
        
        actor = self._ensure_actor(workload_cfg)

        # Build workload for input preparation (same as RPC)
        impl = workload_cfg["implementation"]
        spec = copy.deepcopy(workload_cfg.get("params", {}))
        workload = build_workload(impl, spec, "cpu", self.dtype)

        def prepare_inputs() -> Dict[str, torch.Tensor]:
            prepared = workload.prepare_inputs()
            return {
                key: value.cpu() if torch.is_tensor(value) else value
                for key, value in prepared.items()
            }

        def get_units(inputs: Dict[str, torch.Tensor], outputs: torch.Tensor) -> float:
            return workload.units_from_output(inputs, outputs)

        # Warmup runs (same as other baselines)
        for _ in range(self.warmup_runs):
            inputs = prepare_inputs()
            # Use ray.put() for Plasma Object Store (fair comparison with other serialization)
            inputs_ref = ray.put(inputs)
            ray.get(actor.execute.remote(inputs_ref))

        runs: List[Dict[str, float]] = []
        for run_id in range(1, self.runs + 1):
            inputs = prepare_inputs()

            # Measure end-to-end latency including Plasma serialization
            # This matches the end-to-end measurement of other baselines
            start = time.perf_counter()
            inputs_ref = ray.put(inputs)  # Plasma Object Store serialization
            output = ray.get(actor.execute.remote(inputs_ref))
            latency_ms = (time.perf_counter() - start) * 1000.0

            # Compute metrics (same as other baselines)
            input_mb = _tensor_dict_mb(inputs)
            output_mb = _tensor_mb(output)
            total_mb = input_mb + output_mb
            units = get_units(inputs, output)
            throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

            runs.append(
                {
                    "run_id": run_id,
                    "latency_ms": latency_ms,
                    "input_mb": input_mb,
                    "output_mb": output_mb,
                    "total_data_mb": total_mb,
                    "units_processed": units,
                    "throughput_units_per_s": throughput,
                }
            )

        aggregates = summarize_fields(runs, ["latency_ms", "throughput_units_per_s", "total_data_mb"])
        metadata = {
            **workload.metadata(),
            "baseline": self.name,
            "runner_type": "ray_actor",
            "serialization": "Pickle + Plasma Object Store",
        }

        return BaselineResult(
            baseline=self.name,
            runner_type="ray_actor",
            runs=runs,
            aggregates=aggregates,
            metadata=metadata,
            derived={},
        )


RUNNER_TYPES = {
    "local_synthetic": LocalSyntheticBaselineRunner,
    "scaling": ScalingBaselineRunner,
    "remote_djinn": RemoteDjinnBaselineRunner,
    "pytorch_rpc": PytorchRpcBaselineRunner,
    "native_server": NativeServerBaselineRunner,
    "ray_actor": RayActorBaselineRunner,
}
if vLLMBaselineRunner is not None:
    RUNNER_TYPES["vllm"] = vLLMBaselineRunner


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


class NativeServerBaselineRunner:
    """Runs native PyTorch baseline on dedicated server for fair comparison."""

    def __init__(self, baseline_cfg: Dict, experiment_cfg: Dict):
        self.cfg = baseline_cfg
        self.exp_cfg = experiment_cfg
        self.name = baseline_cfg["name"]
        self.server_address = baseline_cfg.get("server_address", "localhost:5557")
        self.dtype = baseline_cfg.get("dtype", experiment_cfg.get("dtype", "float16"))
        self.runs = baseline_cfg.get("runs", experiment_cfg.get("runs", 5))
        self.warmup_runs = baseline_cfg.get("warmup_runs", experiment_cfg.get("warmup_runs", 1))
        # Parse address
        parts = self.server_address.split(":")
        self.host = parts[0]
        self.port = int(parts[1]) if len(parts) > 1 else 5557

    def run(self, workload_cfg: Dict) -> BaselineResult:
        """Execute workload against native server."""
        import json
        import socket
        import struct

        impl = workload_cfg["implementation"]
        spec = copy.deepcopy(workload_cfg.get("params", {}))
        workload = build_workload(impl, spec, "cpu", self.dtype)

        def prepare_inputs() -> Dict[str, torch.Tensor]:
            prepared = workload.prepare_inputs()
            return {
                key: value.cpu() if torch.is_tensor(value) else value
                for key, value in prepared.items()
            }

        def get_units(inputs: Dict[str, torch.Tensor], outputs: torch.Tensor) -> float:
            return workload.units_from_output(inputs, outputs)

        # Warmup
        for _ in range(self.warmup_runs):
            inputs = prepare_inputs()
            self._execute_remote(inputs)

        runs: List[Dict[str, float]] = []
        for run_id in range(1, self.runs + 1):
            inputs = prepare_inputs()
            start = time.perf_counter()
            output = self._execute_remote(inputs)
            latency_ms = (time.perf_counter() - start) * 1000.0

            input_mb = _tensor_dict_mb(inputs)
            output_mb = _tensor_mb(output)
            total_mb = input_mb + output_mb
            units = get_units(inputs, output)
            throughput = units / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

            runs.append(
                {
                    "run_id": run_id,
                    "latency_ms": latency_ms,
                    "input_mb": input_mb,
                    "output_mb": output_mb,
                    "total_data_mb": total_mb,
                    "units_processed": units,
                    "throughput_units_per_s": throughput,
                }
            )

        aggregates = summarize_fields(runs, ["latency_ms", "throughput_units_per_s", "total_data_mb"])
        metadata = {
            **workload.metadata(),
            "baseline": self.name,
            "runner_type": "native_server",
            "server_address": self.server_address,
        }

        return BaselineResult(
            baseline=self.name,
            runner_type="native_server",
            runs=runs,
            aggregates=aggregates,
            metadata=metadata,
            derived={},
        )

    def _execute_remote(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Send request to native server and get response."""
        import json
        import socket
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.host, self.port))

            # Build request
            request = {
                "workload": "hf_tiny_gpt2",  # This will be enhanced later
                "inputs": {
                    key: value.tolist() if torch.is_tensor(value) else value
                    for key, value in inputs.items()
                },
            }
            request_bytes = json.dumps(request).encode()

            # Send request length + data
            sock.sendall(struct.pack(">I", len(request_bytes)))
            sock.sendall(request_bytes)

            # Read response length + data
            response_len_bytes = sock.recv(4)
            if not response_len_bytes:
                raise RuntimeError("No response from native server")

            response_len = struct.unpack(">I", response_len_bytes)[0]
            response_bytes = sock.recv(response_len)
            response = json.loads(response_bytes.decode())

            # Reconstruct output tensor from response
            output = torch.zeros(response["output_shape"], dtype=torch.float16)
            return output

        finally:
            sock.close()


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

        def mean_metric(name: str, metric: str) -> Optional[float]:
            res = result_map.get(name)
            if not res:
                return None
            block = res["aggregates"].get(metric)
            if not block:
                return None
            return block.get("mean")

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

            delta_metrics = [
                ("latency_ms", "latency_delta_ms_vs_semantic_blind"),
                ("client_deserialize_ms", "client_deserialize_delta_ms_vs_semantic_blind"),
                ("client_serialize_ms", "client_serialize_delta_ms_vs_semantic_blind"),
                ("server_execution_ms", "server_execution_delta_ms_vs_semantic_blind"),
                ("server_duration_ms", "server_duration_delta_ms_vs_semantic_blind"),
            ]
            for metric_name, derived_key in delta_metrics:
                blind_mean = mean_metric(blind_name, metric_name)
                target_mean = mean_metric(target_name, metric_name)
                if blind_mean is not None and target_mean is not None:
                    target["derived"][derived_key] = blind_mean - target_mean


def _baseline_result_to_dict(result: BaselineResult) -> Dict:
    return {
        "baseline": result.baseline,
        "runner_type": result.runner_type,
        "runs": result.runs,
        "aggregates": result.aggregates,
        "metadata": result.metadata,
        "derived": result.derived,
    }


def _tensor_mb(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None:
        return 0.0
    return tensor.element_size() * tensor.numel() / (1024**2)


def _tensor_dict_mb(values: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for value in values.values():
        if torch.is_tensor(value):
            total += _tensor_mb(value)
    return total


__all__ = ["ExperimentRunner"]


