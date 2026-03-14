"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ██╗  ██╗███████╗██████╗ ███╗   ██╗███████╗██╗                         ║
║   ██║ ██╔╝██╔════╝██╔══██╗████╗  ██║██╔════╝██║                         ║
║   █████╔╝ █████╗  ██████╔╝██╔██╗ ██║█████╗  ██║                         ║
║   ██╔═██╗ ██╔══╝  ██╔══██╗██║╚██╗██║██╔══╝  ██║                         ║
║   ██║  ██╗███████╗██║  ██║██║ ╚████║███████╗███████╗                     ║
║   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝                     ║
║                                                                          ║
║              BIZRA SOVEREIGN NERVE CENTER v3.1.0                         ║
║                                                                          ║
║   Unified gateway + observability. Zero external deps (stdlib only).     ║
║   RPC proxy, metrics, log streaming, ops dashboard, auto-healing.        ║
║                                                                          ║
║   Port: 9740 (Sovereign Kernel HTTP)                                     ║
║   Backends: ghost_ws.py:9743, desktop_bridge.py:9742                     ║
║   State: sovereign_state/kernel.pid, kernel_initialized.json             ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import atexit
import json
import logging
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

# Note: urllib removed — desktop_bridge uses raw TCP JSON-RPC, not HTTP

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

KERNEL_VERSION = "3.1.0-NERVE-CENTER"
KERNEL_PORT = 9740
WATCHDOG_INTERVAL_S = 30
MAX_RESTART_BACKOFF_S = 300
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3
MEMORY_LOG_SIZE = 500  # Ring buffer for /api/logs
RPC_PROXY_TIMEOUT_S = 10  # Reverse proxy timeout for /rpc
METRICS_LATENCY_WINDOW = 200  # Rolling window for p95 calculation

# Resolve project root (parent of core/)
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "public"
STATE_DIR = PROJECT_ROOT / "sovereign_state"
PID_FILE = STATE_DIR / "kernel.pid"
INIT_FILE = STATE_DIR / "kernel_initialized.json"
LOG_FILE = STATE_DIR / "kernel.log"
AUTOSTART_FILE = STATE_DIR / "autostart.json"

# Backend services to manage
BACKENDS = [
    {
        "name": "ghost_ws",
        "module": "core.bridges.ghost_ws",
        "port": 9743,
        "health_path": "/health",
    },
    {
        "name": "desktop_bridge",
        "module": "core.bridges.desktop_bridge",
        "port": 9742,
        "health_path": None,  # TCP-only, no HTTP health
    },
]


# ═══════════════════════════════════════════════════════════
#  LOGGING — Structured JSON, rotating
# ═══════════════════════════════════════════════════════════


def _setup_logging() -> logging.Logger:
    """Configure structured logging with rotation."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bizra.kernel")
    logger.setLevel(logging.INFO)

    # Rotating file handler (stdlib — no loguru dependency)
    from logging.handlers import RotatingFileHandler

    fh = RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


log = _setup_logging()


# ═══════════════════════════════════════════════════════════
#  MEMORY LOG HANDLER — ring buffer for /api/logs
# ═══════════════════════════════════════════════════════════


class MemoryLogHandler(logging.Handler):
    """Thread-safe ring buffer that captures log records for the /api/logs endpoint.

    Bounded to MEMORY_LOG_SIZE entries — no memory leak.  Each entry is a
    JSON-serializable dict with timestamp, level, logger name, and message.
    """

    def __init__(self, capacity: int = MEMORY_LOG_SIZE) -> None:
        super().__init__(level=logging.DEBUG)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "seq": self._seq,
                "ts": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            with self._lock:
                self._buffer.append(entry)
                self._seq += 1
        except Exception:
            self.handleError(record)

    def recent(self, n: int = 50, level: str | None = None) -> list[dict[str, Any]]:
        """Return the most recent *n* log entries, optionally filtered by level."""
        with self._lock:
            entries = list(self._buffer)
        if level:
            ul = level.upper()
            entries = [e for e in entries if e["level"] == ul]
        return entries[-n:]

    @property
    def total(self) -> int:
        return self._seq


# Attach memory handler to the kernel logger
_memory_handler = MemoryLogHandler()
log.addHandler(_memory_handler)


# ═══════════════════════════════════════════════════════════
#  REQUEST METRICS — counters + latency for /api/metrics
# ═══════════════════════════════════════════════════════════


class RequestMetrics:
    """Thread-safe request metrics collector.

    Tracks total requests, errors, per-method counts, per-path counts,
    and a rolling latency window for p50/p95/p99 calculation.
    """

    def __init__(self, window: int = METRICS_LATENCY_WINDOW) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._errors = 0
        self._by_method: dict[str, int] = {}
        self._by_path: dict[str, int] = {}
        self._latencies: deque[float] = deque(maxlen=window)
        self._rpc_forwards = 0
        self._rpc_errors = 0

    def record(self, method: str, path: str, status: int, latency_ms: float) -> None:
        """Record a completed request."""
        with self._lock:
            self._total += 1
            self._by_method[method] = self._by_method.get(method, 0) + 1
            # Normalize path to reduce cardinality
            bucket = path.split("?")[0]
            self._by_path[bucket] = self._by_path.get(bucket, 0) + 1
            self._latencies.append(latency_ms)
            if status >= 400:
                self._errors += 1

    def record_rpc(self, success: bool) -> None:
        """Record an RPC proxy forward."""
        with self._lock:
            self._rpc_forwards += 1
            if not success:
                self._rpc_errors += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time metrics snapshot."""
        with self._lock:
            lats = sorted(self._latencies) if self._latencies else [0.0]
            n = len(lats)
            return {
                "requests_total": self._total,
                "errors_total": self._errors,
                "error_rate": round(self._errors / max(self._total, 1), 4),
                "by_method": dict(self._by_method),
                "top_paths": dict(
                    sorted(self._by_path.items(), key=lambda x: -x[1])[:10]
                ),
                "latency_ms": {
                    "p50": round(lats[n // 2], 2),
                    "p95": round(lats[int(n * 0.95)], 2),
                    "p99": round(lats[int(n * 0.99)], 2),
                    "max": round(lats[-1], 2),
                },
                "rpc_proxy": {
                    "forwards": self._rpc_forwards,
                    "errors": self._rpc_errors,
                },
                "uptime_s": _uptime(),
            }


# Global metrics instance (wired into handler)
_metrics = RequestMetrics()


# ═══════════════════════════════════════════════════════════
#  STATE MANAGEMENT — sovereign_state/ persistence
# ═══════════════════════════════════════════════════════════


class SovereignState:
    """Atomic state persistence to sovereign_state/kernel_initialized.json."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if INIT_FILE.exists():
            try:
                return json.loads(INIT_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Corrupted state file, resetting: %s", e)
        return {}

    def _persist(self) -> None:
        """Atomic write: tmp → rename."""
        tmp = INIT_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")
        tmp.replace(INIT_FILE)

    @property
    def is_initialized(self) -> bool:
        return bool(self._state.get("version") and self._state.get("timestamp"))

    def initialize(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._state = {
                "version": KERNEL_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "userName": data.get("userName", "Sovereign"),
                "lang": data.get("lang", "en"),
                "model": data.get("model", "auto"),
                "deviceProfile": data.get("deviceProfile", {}),
            }
            self._persist()
            log.info("Kernel initialized for user: %s", self._state["userName"])

    def read(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def reset(self) -> None:
        with self._lock:
            self._state = {}
            if INIT_FILE.exists():
                INIT_FILE.unlink()
            log.info("Kernel state reset -- next boot will show installer")


# ═══════════════════════════════════════════════════════════
#  PID FILE — prevent duplicate daemons
# ═══════════════════════════════════════════════════════════


def _write_pid() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    log.info("PID %d written to %s", os.getpid(), PID_FILE)


def _clear_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink(missing_ok=True)


def _check_existing_daemon() -> bool:
    """Return True if another daemon is already running."""
    if not PID_FILE.exists():
        return False

    try:
        old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False

    # Check if process is alive (Windows-compatible)
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(
            0x1000, False, old_pid
        )  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(old_pid, 0)
            return True
        except OSError:
            return False


# ═══════════════════════════════════════════════════════════
#  SUBPROCESS WATCHDOG — auto-restart backends
# ═══════════════════════════════════════════════════════════


class SubprocessWatchdog:
    """Manages backend services with health checks and exponential backoff restart."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._backoff: dict[str, float] = {}
        self._restart_count: dict[str, int] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def start_all(self) -> None:
        """Start all configured backend services."""
        self._running = True
        for backend in BACKENDS:
            self._start_one(backend)

        self._thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="watchdog"
        )
        self._thread.start()
        log.info("Watchdog started -- monitoring %d backends", len(BACKENDS))

    def _start_one(self, backend: dict[str, Any]) -> None:
        name = backend["name"]
        module = backend["module"]

        # Don't start if module doesn't exist
        module_path = PROJECT_ROOT / module.replace(".", "/") / "__init__.py"
        script_path = PROJECT_ROOT / (module.replace(".", "/") + ".py")
        if not module_path.exists() and not script_path.exists():
            log.warning("Backend module not found: %s -- skipping", module)
            return

        python = sys.executable
        cmd = [python, "-m", module]

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._processes[name] = proc
            self._backoff[name] = WATCHDOG_INTERVAL_S
            self._restart_count.setdefault(name, 0)
            log.info("Started backend: %s (PID %d)", name, proc.pid)
        except FileNotFoundError:
            log.error("Python executable not found for %s: %s", name, python)
        except OSError as e:
            log.error("Failed to start %s: %s", name, e)

    def _watchdog_loop(self) -> None:
        """Health-check loop with exponential backoff restart."""
        while self._running:
            time.sleep(WATCHDOG_INTERVAL_S)
            if not self._running:
                break

            for backend in BACKENDS:
                name = backend["name"]
                proc = self._processes.get(name)

                if proc is None:
                    continue

                if proc.poll() is not None:
                    # Process exited
                    exit_code = proc.returncode
                    self._restart_count[name] = self._restart_count.get(name, 0) + 1
                    backoff = min(
                        WATCHDOG_INTERVAL_S * (2 ** min(self._restart_count[name], 5)),
                        MAX_RESTART_BACKOFF_S,
                    )
                    log.warning(
                        "Backend %s exited (code=%s, restarts=%d) -- restarting in %.0fs",
                        name,
                        exit_code,
                        self._restart_count[name],
                        backoff,
                    )
                    time.sleep(backoff)
                    if self._running:
                        self._start_one(backend)

    def stop_all(self) -> None:
        """Graceful shutdown of all backends."""
        self._running = False
        for name, proc in self._processes.items():
            if proc.poll() is None:
                log.info("Stopping backend: %s (PID %d)", name, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning("Force-killing %s", name)
                    proc.kill()
        self._processes.clear()
        log.info("All backends stopped")

    def restart_one(self, name: str) -> bool:
        """Restart a single backend by name. Returns True on success."""
        backend = next((b for b in BACKENDS if b["name"] == name), None)
        if backend is None:
            log.warning("Cannot restart unknown backend: %s", name)
            return False

        proc = self._processes.get(name)
        if proc and proc.poll() is None:
            log.info("Stopping %s (PID %d) for restart", name, proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        self._start_one(backend)
        return name in self._processes and self._processes[name].poll() is None

    def status(self) -> list[dict[str, Any]]:
        """Return status of all managed backends."""
        result = []
        for backend in BACKENDS:
            name = backend["name"]
            proc = self._processes.get(name)
            result.append(
                {
                    "name": name,
                    "port": backend["port"],
                    "pid": proc.pid if proc and proc.poll() is None else None,
                    "alive": proc is not None and proc.poll() is None,
                    "restarts": self._restart_count.get(name, 0),
                }
            )
        return result


# ═══════════════════════════════════════════════════════════
#  HTTP REQUEST HANDLER — serves frontend + API
# ═══════════════════════════════════════════════════════════


class KernelHandler(SimpleHTTPRequestHandler):
    """
    BIZRA Sovereign Nerve Center -- Unified Gateway + Observability.

    Routes:
      GET  /                 -> installer (first run) or terminal (returning)
      GET  /installer        -> force serve installer
      GET  /terminal         -> force serve terminal
      GET  /ops              -> operations dashboard (Nerve Center)
      GET  /api/health       -> 200 OK heartbeat
      GET  /api/status       -> backend subsystem status
      GET  /api/state        -> sovereign initialization state
      GET  /api/metrics      -> request metrics + latency percentiles
      GET  /api/logs         -> recent structured log entries (ring buffer)
      POST /api/initialize   -> mark kernel as initialized
      POST /api/reset        -> reset to first-run state
      POST /api/backends     -> restart a backend  {"action":"restart","name":"..."}
      POST /rpc              -> reverse proxy to desktop_bridge (:9742)
      GET  /*                -> static files from frontend/public/
    """

    server_version = f"BIZRA-Kernel/{KERNEL_VERSION}"

    def __init__(
        self,
        *args: Any,
        state: SovereignState,
        watchdog: SubprocessWatchdog,
        **kwargs: Any,
    ) -> None:
        self.sovereign_state = state
        self.watchdog = watchdog
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Route HTTP logs through our structured logger."""
        log.debug("HTTP %s", fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        t0 = time.monotonic()
        path = self.path.split("?")[0]

        # ── API Routes ──
        if path == "/api/health":
            self._json_response(
                {"status": "alive", "version": KERNEL_VERSION, "uptime_s": _uptime()}
            )
        elif path == "/api/status":
            self._json_response(
                {
                    "kernel": {
                        "version": KERNEL_VERSION,
                        "pid": os.getpid(),
                        "uptime_s": _uptime(),
                    },
                    "backends": self.watchdog.status(),
                    "initialized": self.sovereign_state.is_initialized,
                }
            )
        elif path == "/api/state":
            self._json_response(self.sovereign_state.read())
        elif path == "/api/metrics":
            self._json_response(_metrics.snapshot())
        elif path == "/api/logs":
            # Query params: ?n=50&level=ERROR
            qs = self.path.split("?")[1] if "?" in self.path else ""
            params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
            n = min(int(params.get("n", "50")), MEMORY_LOG_SIZE)
            level = params.get("level", None)
            entries = _memory_handler.recent(n=n, level=level)
            self._json_response(
                {
                    "entries": entries,
                    "count": len(entries),
                    "total_captured": _memory_handler.total,
                }
            )
        # ── Frontend Routes ──
        elif path == "/" or path == "":
            if self.sovereign_state.is_initialized:
                self._serve_file("terminal-emulator.html")
            else:
                self._serve_file("bizra-installer.html")
        elif path == "/installer":
            self._serve_file("bizra-installer.html")
        elif path == "/terminal":
            self._serve_file("terminal-emulator.html")
        elif path == "/ops":
            self._serve_file("sovereign_ops_dashboard.html")
        else:
            # Static file serving (CSS, JS, images, fonts)
            super().do_GET()

        _metrics.record("GET", path, 200, (time.monotonic() - t0) * 1000)

    def do_POST(self) -> None:  # noqa: N802
        t0 = time.monotonic()
        path = self.path.split("?")[0]
        status = 200

        if path == "/api/initialize":
            body = self._read_body()
            self.sovereign_state.initialize(body)
            self._json_response({"status": "initialized", "version": KERNEL_VERSION})
        elif path == "/api/reset":
            self.sovereign_state.reset()
            self._json_response({"status": "reset"})
        elif path == "/api/backends":
            body = self._read_body()
            action = body.get("action", "")
            name = body.get("name", "")
            if action == "restart" and name:
                ok = self.watchdog.restart_one(name)
                self._json_response(
                    {
                        "status": "restarted" if ok else "failed",
                        "backend": name,
                        "backends": self.watchdog.status(),
                    }
                )
                status = 200 if ok else 500
            else:
                self._json_response({"error": "action and name required"}, status=400)
                status = 400
        elif path == "/rpc":
            self._proxy_rpc()
            _metrics.record("POST", path, 200, (time.monotonic() - t0) * 1000)
            return  # _proxy_rpc handles its own response
        else:
            self.send_error(404, "Not found")
            status = 404

        _metrics.record("POST", path, status, (time.monotonic() - t0) * 1000)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """CORS preflight."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ── Reverse Proxy: /rpc -> desktop_bridge :9742 (raw TCP JSON-RPC) ──

    def _proxy_rpc(self) -> None:
        """Forward JSON-RPC request to desktop_bridge over raw TCP socket.

        desktop_bridge speaks JSON-RPC over TCP (NOT HTTP), so we open a
        socket, send the JSON payload + newline, read the JSON response,
        and wrap it in an HTTP response for the browser.

        This makes port 9740 the SINGLE gateway -- frontends never need
        to know about :9742 directly.
        """
        import socket as _socket

        raw_body = b""
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length:
                raw_body = self.rfile.read(length)

            # Inject auth headers from the HTTP request into the JSON-RPC body
            # (mirrors what ghost_ws /rpc does for the HTML frontends)
            try:
                body_obj = json.loads(raw_body) if raw_body else {}
                token = self.headers.get("X-BIZRA-TOKEN", "")
                if token:
                    hdrs = body_obj.setdefault("headers", {})
                    hdrs.setdefault("X-BIZRA-TOKEN", token)
                    hdrs.setdefault("X-BIZRA-TS", str(int(time.time() * 1000)))
                    import uuid as _uuid

                    hdrs.setdefault("X-BIZRA-NONCE", str(_uuid.uuid4()))
                raw_body = json.dumps(body_obj).encode("utf-8")
            except (json.JSONDecodeError, TypeError):
                pass  # Send as-is if not valid JSON

            # Raw TCP forward to desktop_bridge
            sock = _socket.create_connection(
                ("127.0.0.1", 9742), timeout=RPC_PROXY_TIMEOUT_S
            )
            try:
                sock.sendall(raw_body + b"\n")
                # Read response (bridge sends single JSON line)
                chunks = []
                sock.settimeout(RPC_PROXY_TIMEOUT_S)
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    # JSON-RPC response is a single line — check for completeness
                    joined = b"".join(chunks)
                    if b"\n" in joined or self._is_complete_json(joined):
                        break
                resp_body = b"".join(chunks).strip()
            finally:
                sock.close()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(resp_body)
            _metrics.record_rpc(success=True)

        except (ConnectionRefusedError, _socket.timeout, OSError) as e:
            err_body = json.dumps(
                {"error": "bridge unreachable", "detail": str(e)}
            ).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(err_body)
            _metrics.record_rpc(success=False)
            log.warning("RPC proxy: bridge unreachable -- %s", e)

    @staticmethod
    def _is_complete_json(data: bytes) -> bool:
        """Quick check if data looks like a complete JSON object/array."""
        s = data.strip()
        if not s:
            return False
        return (s[0:1] == b"{" and s[-1:] == b"}") or (
            s[0:1] == b"[" and s[-1:] == b"]"
        )

    def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename: str) -> None:
        filepath = FRONTEND_DIR / filename
        if not filepath.exists():
            self.send_error(404, f"File not found: {filename}")
            return
        content = filepath.read_bytes()
        mime = mimetypes.guess_type(filename)[0] or "text/html"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(content)

    def _cors_headers(self) -> None:
        # Allow requests from file:// (null origin) and localhost variants
        origin = self.headers.get("Origin", "*")
        allowed = {"http://127.0.0.1:9740", "http://localhost:9740", "null"}
        self.send_header(
            "Access-Control-Allow-Origin",
            origin if origin in allowed else "http://127.0.0.1:9740",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}


# ═══════════════════════════════════════════════════════════
#  UPTIME TRACKER
# ═══════════════════════════════════════════════════════════

_START_TIME = time.monotonic()


def _uptime() -> float:
    return round(time.monotonic() - _START_TIME, 1)


# ═══════════════════════════════════════════════════════════
#  SIGNAL HANDLING — graceful shutdown
# ═══════════════════════════════════════════════════════════

_shutdown_event = threading.Event()


def _signal_handler(watchdog: SubprocessWatchdog, signum: int, _frame: Any) -> None:
    sig_name = (
        signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    )
    log.info("Received %s -- initiating graceful shutdown", sig_name)
    _shutdown_event.set()
    watchdog.stop_all()
    _clear_pid()


# ═══════════════════════════════════════════════════════════
#  MAIN — daemon entry point
# ═══════════════════════════════════════════════════════════


def main() -> None:
    """Start the BIZRA Sovereign Kernel Daemon."""

    log.info("=" * 60)
    log.info("  BIZRA SOVEREIGN NERVE CENTER v%s", KERNEL_VERSION)
    log.info("  PID: %d | Port: %d", os.getpid(), KERNEL_PORT)
    log.info("  Root: %s", PROJECT_ROOT)
    log.info("  Frontend: %s", FRONTEND_DIR)
    log.info("  State: %s", STATE_DIR)
    log.info("=" * 60)

    # ── Guard: prevent duplicate daemons ──
    if _check_existing_daemon():
        log.error("Another kernel daemon is already running (PID file: %s)", PID_FILE)
        sys.exit(1)

    # ── Write PID ──
    _write_pid()
    atexit.register(_clear_pid)

    # ── Initialize state manager ──
    state = SovereignState()
    if state.is_initialized:
        log.info("Kernel previously initialized for: %s", state.read().get("userName"))
    else:
        log.info("First boot -- installer will be served")

    # ── Start backend watchdog ──
    watchdog = SubprocessWatchdog()
    watchdog.start_all()

    # ── Signal handlers (main thread only) ──
    try:
        handler = partial(_signal_handler, watchdog)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, handler)  # type: ignore[attr-defined]
    except ValueError:
        # Non-main thread (e.g., test harness) — skip signal registration
        log.warning("Signal handlers skipped (non-main thread)")

    # ── Start HTTP server ──
    handler_factory = partial(KernelHandler, state=state, watchdog=watchdog)
    server = HTTPServer(("127.0.0.1", KERNEL_PORT), handler_factory)
    server.timeout = 1  # Allow shutdown check every second

    log.info("Sovereign Nerve Center listening on http://127.0.0.1:%d", KERNEL_PORT)
    log.info(
        "  /             -> %s", "terminal" if state.is_initialized else "installer"
    )
    log.info("  /ops          -> operations dashboard")
    log.info("  /rpc          -> reverse proxy to bridge (:9742)")
    log.info("  /api/health   -> health check")
    log.info("  /api/status   -> subsystem status")
    log.info("  /api/metrics  -> request metrics + latency")
    log.info("  /api/logs     -> structured log stream")

    # ── Open browser on first start ──
    try:
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{KERNEL_PORT}/")
    except Exception:
        pass  # Headless environments

    # ── Serve until shutdown ──
    try:
        while not _shutdown_event.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- shutting down")
    finally:
        server.server_close()
        watchdog.stop_all()
        _clear_pid()
        log.info("Sovereign Kernel Daemon stopped. Ihsan preserved.")


if __name__ == "__main__":
    main()
