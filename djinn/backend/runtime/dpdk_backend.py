from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is optional for this wrapper
    torch = None  # type: ignore

from . import transport_coordinator as tc
from djinn.frontend.semantic.workload import ExecutionPlan


logger = logging.getLogger(__name__)


@dataclass
class TransferHandle:
    transfer_id: str
    node_id: str
    target: str
    size: int
    completed: bool = False
    error: Optional[str] = None


class DPDKBackend:
    """DPDK-accelerated backend for plan-driven transfers.

    This wraps the existing TransportCoordinator to provide an ExecutionPlan-facing API.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = config or {}
        self._coordinator: Optional[tc.TransportCoordinator] = None
        self._active: Dict[str, TransferHandle] = {}
        self._initialized: bool = False
        self._scheduler_endpoint: Optional[str] = None

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._coordinator = await tc.initialize_transport(config=self._config)
        self._initialized = True

    def _require_coordinator(self) -> TransportCoordinator:
        if not self._coordinator:
            raise RuntimeError("Transport coordinator not initialized")
        return self._coordinator

    def close(self) -> None:
        if not self._initialized:
            return
        # Run shutdown in a private loop to avoid interfering with caller loops
        try:
            asyncio.run(tc.shutdown_transport())
        except RuntimeError:
            # If within a running loop, schedule shutdown
            loop = asyncio.get_event_loop()
            loop.create_task(tc.shutdown_transport())
        finally:
            self._initialized = False
            self._coordinator = None

    def register_with_control_plane(self, capabilities: Optional[Dict[str, Any]] = None) -> bool:
        """Register this node with the local control plane (stub for Phase 1).

        This leverages the coordinator's control server to set/update capabilities.
        Returns True if coordinator is initialized and capabilities set.
        """

        async def _run() -> bool:
            await self._ensure_initialized()
            coord = self._require_coordinator()
            if not coord.control_plane:
                # Initialize control plane if not started yet
                await coord._init_control_plane()  # type: ignore[attr-defined]
            if not coord.control_plane:
                return False
            if capabilities:
                # Update advertised capabilities on the server
                server_caps = coord.control_plane.capabilities
                for k, v in capabilities.items():
                    if hasattr(server_caps, k):
                        setattr(server_caps, k, v)
            return True

        try:
            return asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())

    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute an ExecutionPlan.

        Phase 1: only processes plan.transfers if provided. Returns a results dict
        containing transfer status keyed by transfer_id.
        """

        async def _run() -> Dict[str, Any]:
            await self._ensure_initialized()
            coord = self._require_coordinator()

            results: Dict[str, Any] = {}
            transfers: List[Dict[str, Any]] = getattr(plan, "transfers", []) or []

            for t in transfers:
                try:
                    tensor = t.get("tensor")
                    target = t.get("target") or t.get("target_node") or "local"
                    if tensor is None:
                        logger.debug("Skipping transfer without tensor payload")
                        continue
                    if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                        transfer_id = await coord.send_tensor(tensor, target)
                        handle = TransferHandle(
                            transfer_id=transfer_id,
                            node_id=plan.placement.get(t.get("fragment_id", ""), {}).get("node", "local"),
                            target=target,
                            size=tensor.numel() * tensor.element_size(),
                        )
                        self._active[transfer_id] = handle
                        results[transfer_id] = {"started": True}
                    else:
                        # Non-tensor payloads not supported in phase 1
                        logger.warning("Unsupported transfer payload; only tensor-like objects with numel/element_size are supported in phase 1")
                except Exception as e:  # Robust per-transfer handling
                    logger.error(f"Failed to start transfer: {e}")
            return results

        try:
            return asyncio.run(_run())
        except RuntimeError:
            # If caller already has an event loop running (e.g., in notebooks), run in task
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())

    def transfer_tensor(self, tensor: "torch.Tensor", target: str) -> str:
        """Convenience API to transfer a single tensor synchronously."""

        async def _run() -> str:
            await self._ensure_initialized()
            coord = self._require_coordinator()
            transfer_id = await coord.send_tensor(tensor, target)
            self._active[transfer_id] = TransferHandle(
                transfer_id=transfer_id,
                node_id="local",
                target=target,
                size=tensor.numel() * tensor.element_size(),
            )
            return transfer_id

        try:
            return asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())

    def get_active_transfers(self) -> Dict[str, TransferHandle]:
        return dict(self._active)

    def set_scheduler_endpoint(self, scheduler_url: str) -> None:
        self._scheduler_endpoint = scheduler_url

    def register_with_scheduler(self, scheduler_url: Optional[str] = None, capabilities: Optional[Dict[str, Any]] = None) -> bool:
        async def _run() -> bool:
            await self._ensure_initialized()
            url = scheduler_url or self._scheduler_endpoint
            if not url:
                return True
            try:
                import aiohttp  # type: ignore
            except Exception:
                return True
            payload: Dict[str, Any] = {
                "node_id": self._require_coordinator().node_id,
                "capabilities": capabilities or {},
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{url}/register", json=payload) as resp:
                        return 200 <= resp.status < 300
            except Exception:
                return True

        try:
            return asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())

    def send_scheduler_heartbeat(self) -> bool:
        async def _run() -> bool:
            await self._ensure_initialized()
            if not self._scheduler_endpoint:
                return True
            try:
                import aiohttp  # type: ignore
            except Exception:
                return True
            payload = {"node_id": self._require_coordinator().node_id}
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self._scheduler_endpoint}/heartbeat", json=payload) as resp:
                        return 200 <= resp.status < 300
            except Exception:
                return True

        try:
            return asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())


