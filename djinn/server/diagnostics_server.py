"""Lightweight diagnostics HTTP server for Djinn."""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DiagnosticsServer:
    """
    Minimal HTTP server that exposes diagnostic endpoints (currently VMU metrics).

    The server intentionally avoids extra dependencies (FastAPI, aiohttp) so it can run
    inside the existing asyncio event loop with negligible overhead.
    """

    def __init__(
        self,
        metrics_provider: Callable[[], Dict[str, Any]],
        *,
        host: str = "0.0.0.0",
        port: int = 9095,
    ):
        self._metrics_provider = metrics_provider
        self._host = host
        self._port = port
        self._server: Optional[asyncio.AbstractServer] = None
        self._listening_port: Optional[int] = None

    @property
    def listening_port(self) -> Optional[int]:
        return self._listening_port

    async def start(self) -> None:
        """Start the diagnostics HTTP server."""
        if self._server is not None:
            return

        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )

        sockets = getattr(self._server, "sockets", None) or []
        if sockets:
            self._listening_port = sockets[0].getsockname()[1]
        else:
            self._listening_port = self._port

        logger.info(
            "Diagnostics server listening on http://%s:%s",
            self._host,
            self._listening_port,
        )

    async def stop(self) -> None:
        """Stop the diagnostics HTTP server."""
        if not self._server:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None
        logger.info("Diagnostics server stopped")

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        except asyncio.TimeoutError:
            writer.close()
            await writer.wait_closed()
            return

        if not request_line:
            writer.close()
            await writer.wait_closed()
            return

        try:
            method, path, _ = request_line.decode("latin-1").strip().split(" ")
        except ValueError:
            await self._write_response(writer, 400, {"error": "malformed request"})
            return

        # Drain headers (we do not need them right now)
        while True:
            line = await reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break

        if method != "GET":
            await self._write_response(writer, 405, {"error": "method not allowed"})
            return

        if path in ("/metrics/vmu", "/vmu_metrics"):
            status_code, payload = self._collect_metrics_payload()
            await self._write_response(writer, status_code, payload)
        elif path in ("/healthz", "/livez"):
            await self._write_response(writer, 200, {"status": "ok", "timestamp": time.time()})
        else:
            await self._write_response(writer, 404, {"error": "not found"})

    def _collect_metrics_payload(self) -> Tuple[int, Dict[str, Any]]:
        try:
            payload = self._metrics_provider()
            return 200, payload
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Diagnostics metrics provider failed: %s", exc)
            return 500, {"status": "error", "error": str(exc), "timestamp": time.time()}

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        payload: Dict[str, Any],
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        reason = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }.get(status_code, "OK")

        headers = [
            f"HTTP/1.1 {status_code} {reason}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
            "Connection: close",
            "",
            "",
        ]
        writer.write("\r\n".join(headers).encode("latin-1") + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

