"""Prometheus metrics exporter for aries_agent (Architecture V3 §9)."""
import asyncio
import logging
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from aries_agent.hardware import get_host_ram_available_bytes, get_vram_overflow_bytes

logger = logging.getLogger("aries_agent.metrics")

EXECUTOR_STATES = ("idle", "starting", "ready", "busy", "stopping", "failed")


class AgentMetrics:
    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.agent_connected = Gauge(
            "aries_agent_connected",
            "1 when connected to the backend AgentGateway stream, 0 otherwise",
            registry=self.registry,
        )
        self.agent_reconnects_total = Counter(
            "aries_agent_reconnects_total",
            "Total stream reconnection attempts by the agent",
            registry=self.registry,
        )
        self.executor_state = Gauge(
            "aries_executor_state",
            "One-hot executor state (1 for current state, 0 otherwise)",
            ["executor", "state"],
            registry=self.registry,
        )
        self.executor_mem_used_bytes = Gauge(
            "aries_executor_mem_used_bytes",
            "Used memory on executor in bytes",
            ["executor", "device"],
            registry=self.registry,
        )
        self.executor_mem_total_bytes = Gauge(
            "aries_executor_mem_total_bytes",
            "Total memory on executor in bytes",
            ["executor", "device"],
            registry=self.registry,
        )
        self.executor_mem_overflow_bytes = Gauge(
            "aries_executor_mem_overflow_bytes",
            "Driver spill into system RAM in bytes, read from /proc/<pid>/fdinfo",
            ["executor", "device"],
            registry=self.registry,
        )
        self.host_ram_available_bytes = Gauge(
            "aries_host_ram_available_bytes",
            "Available host system RAM in bytes",
            registry=self.registry,
        )
        self.worker_load_seconds = Histogram(
            "aries_worker_load_seconds",
            "Stage worker model load duration in seconds",
            ["executor", "stage", "model"],
            registry=self.registry,
        )
        self.task_seconds = Histogram(
            "aries_task_seconds",
            "Task execution duration in seconds",
            ["executor", "stage"],
            registry=self.registry,
        )
        self.task_outcomes_total = Counter(
            "aries_task_outcomes_total",
            "Total task outcomes by executor, stage, outcome and failure class",
            ["executor", "stage", "outcome", "class"],
            registry=self.registry,
        )
        self.worker_exits_total = Counter(
            "aries_worker_exits_total",
            "Total stage worker process exits by executor, stage and class",
            ["executor", "stage", "class"],
            registry=self.registry,
        )

        self._executor_pids: Dict[str, Optional[int]] = {}
        self._executor_devices: Dict[str, str] = {}
        self._server: Optional[asyncio.Server] = None

    def init_executor(self, executor_id: str, device: str, mem_total_bytes: int) -> None:
        self._executor_devices[executor_id] = device
        self._executor_pids[executor_id] = None
        self.set_executor_state(executor_id, "idle")
        self.executor_mem_used_bytes.labels(executor=executor_id, device=device).set(0)
        self.executor_mem_total_bytes.labels(executor=executor_id, device=device).set(mem_total_bytes)
        self.executor_mem_overflow_bytes.labels(executor=executor_id, device=device).set(0)

    def set_executor_state(self, executor_id: str, active_state: str) -> None:
        for st in EXECUTOR_STATES:
            self.executor_state.labels(executor=executor_id, state=st).set(1.0 if st == active_state else 0.0)

    def set_executor_pid(self, executor_id: str, pid: Optional[int]) -> None:
        self._executor_pids[executor_id] = pid
        if not pid:
            device = self._executor_devices.get(executor_id, "")
            self.executor_mem_used_bytes.labels(executor=executor_id, device=device).set(0)
            self.executor_mem_overflow_bytes.labels(executor=executor_id, device=device).set(0)

    def set_executor_mem_used(self, executor_id: str, used_bytes: int) -> None:
        device = self._executor_devices.get(executor_id, "")
        self.executor_mem_used_bytes.labels(executor=executor_id, device=device).set(max(0, used_bytes))

    def refresh_live_gauges(self) -> None:
        self.host_ram_available_bytes.set(get_host_ram_available_bytes())
        for exec_id, pid in list(self._executor_pids.items()):
            device = self._executor_devices.get(exec_id, "")
            overflow = get_vram_overflow_bytes(pid) if pid else 0
            self.executor_mem_overflow_bytes.labels(executor=exec_id, device=device).set(overflow)

    def render_prometheus_text(self) -> bytes:
        self.refresh_live_gauges()
        return generate_latest(self.registry)

    async def start_http_server(self, host: str = "0.0.0.0", port: int = 9464) -> int:
        """Start lightweight asyncio HTTP server for /metrics."""
        self._server = await asyncio.start_server(self._handle_http, host=host, port=port)
        sockets = self._server.sockets or []
        bound_port = sockets[0].getsockname()[1] if sockets else port
        logger.info("Agent /metrics HTTP server listening on %s:%d", host, bound_port)
        return bound_port

    async def stop_http_server(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_http(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            while True:
                hdr = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break

            parts = req_line.decode("utf-8", errors="ignore").split()
            path = parts[1] if len(parts) >= 2 else "/"
            if path.startswith("/metrics"):
                body = self.render_prometheus_text()
                headers = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("utf-8")
                writer.write(headers + body)
            else:
                body = b"Not Found\n"
                headers = (
                    "HTTP/1.1 404 Not Found\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("utf-8")
                writer.write(headers + body)
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
