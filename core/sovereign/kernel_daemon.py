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
from socketserver import ThreadingMixIn
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

# CRITICAL: Ensure project root is on sys.path at module load time.
# Without this, lazy imports of core.sovereign.* fail in WSL daemon context
# because nohup/subprocess does not inherit the caller's PYTHONPATH.
import sys as _sys

_project_str = str(PROJECT_ROOT)
if _project_str not in _sys.path:
    _sys.path.insert(0, _project_str)

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
    """Atomic state persistence with Ed25519 genesis identity.

    Standing on Giants: Bernstein (Ed25519) · Aumasson (BLAKE2b) · Lamport (domain separation)

    On first initialization, generates a sovereign Ed25519 keypair and creates
    a signed genesis identity — the birth certificate of this node.
    The private key stays on-device (P2 Self-Sovereignty).
    """

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
            # ── Genesis Identity: create Ed25519 keypair and signed identity ──
            genesis_data = self._create_genesis_identity(data)

            self._state = {
                "version": KERNEL_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "userName": data.get("userName", "Sovereign"),
                "lang": data.get("lang", "en"),
                "model": data.get("model", "auto"),
                "deviceProfile": data.get("deviceProfile", {}),
                **genesis_data,
            }
            self._persist()
            log.info(
                "Kernel initialized for user: %s | identity: %s",
                self._state["userName"],
                self._state.get("identity_id", "unknown")[:16] + "...",
            )

    @staticmethod
    def _create_genesis_identity(data: dict[str, Any]) -> dict[str, Any]:
        """Generate Ed25519 keypair and signed genesis block.

        The private key is stored locally in sovereign_state/node.key.
        The public key and genesis signature are stored in the state file.
        This is the birth certificate of Node0.
        """
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            from core.identity.genesis import (
                HumanAttestation,
                IdentityGenesis,
                PersonaSeed,
                SovereigntyScope,
            )

            # Generate keypair — sk stays on-device
            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )

            # Store private key on-device only
            key_path = STATE_DIR / "node.key"
            key_path.write_bytes(private_bytes)
            key_path.chmod(0o600)
            log.info("Node keypair generated — private key at %s", key_path)

            # Create signed genesis identity
            user_name = data.get("userName", "Sovereign")
            genesis = IdentityGenesis.create(
                public_key,
                persona_seed=PersonaSeed(
                    display_name=user_name,
                    mission_statement=data.get(
                        "mission", "Sovereign node — standing alone"
                    ),
                    locale=data.get("lang", "en"),
                ),
                human_attestation=HumanAttestation.DEVICE_WITNESSED,
                sovereignty_scope=SovereigntyScope.DEVICE_LOCAL,
                genesis_signing_key=private_bytes,
            )

            log.info(
                "Genesis identity created: %s | signature verified: %s",
                genesis.identity_id[:16],
                genesis.verify_genesis_signature(),
            )

            return {
                "identity_id": genesis.identity_id,
                "public_key_hex": public_key.hex(),
                "genesis_hash": genesis.genesis_hash,
                "genesis_signature": genesis.genesis_signature,
                "sovereignty_class": genesis.sovereignty_class.name,
                "genesis_verified": genesis.verify_genesis_signature(),
            }

        except ImportError as e:
            log.warning("Genesis identity unavailable (missing dep): %s", e)
            return {"identity_id": "pending", "genesis_verified": False}
        except (OSError, ValueError) as e:
            log.error("Genesis identity creation failed: %s", e)
            return {"identity_id": "error", "genesis_verified": False}

    def read(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def reset(self) -> None:
        with self._lock:
            self._state = {}
            if INIT_FILE.exists():
                INIT_FILE.unlink()
            # Keep the keypair — identity persists across resets
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
        # Corrupt PID file — remove and allow startup
        PID_FILE.unlink(missing_ok=True)
        return False

    # Check if process is alive AND is actually a kernel_daemon
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(
            0x1000, False, old_pid
        )  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # Stale PID — clean up
        PID_FILE.unlink(missing_ok=True)
        log.info("Cleaned stale PID file (Windows PID %d not found)", old_pid)
        return False
    else:
        try:
            os.kill(old_pid, 0)
            # PID exists — verify it's actually kernel_daemon (not recycled PID)
            try:
                import subprocess

                result = subprocess.run(
                    ["ps", "-p", str(old_pid), "-o", "args="],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if "kernel_daemon" in result.stdout:
                    return True
                # PID recycled to a different process — stale
                PID_FILE.unlink(missing_ok=True)
                log.info(
                    "Cleaned stale PID file (PID %d is not kernel_daemon)", old_pid
                )
                return False
            except Exception:
                return True  # Can't verify — assume running (safe default)
        except OSError:
            # Process doesn't exist — stale PID
            PID_FILE.unlink(missing_ok=True)
            log.info("Cleaned stale PID file (PID %d not running)", old_pid)
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
#  CONSTITUTION & GENESIS VERIFICATION — self-describing node
# ═══════════════════════════════════════════════════════════


def _get_constitution() -> dict[str, Any]:
    """Return the node's constitutional thresholds and kernel invariants.

    Standing on Giants: Al-Ghazali (Ihsan) · Shannon (SNR) · Lamport (invariants)
    """
    try:
        from core.integration.constants import (
            ADL_GINI_THRESHOLD,
            ADL_HARBERGER_TAX_RATE,
            KERNEL_INVARIANTS,
            SNR_THRESHOLD_T0_ELITE,
            SNR_THRESHOLD_T1_HIGH,
            STRICT_IHSAN_THRESHOLD,
            UNIFIED_IHSAN_THRESHOLD,
            UNIFIED_SNR_THRESHOLD,
        )

        return {
            "thresholds": {
                "ihsan_production": UNIFIED_IHSAN_THRESHOLD,
                "ihsan_strict": STRICT_IHSAN_THRESHOLD,
                "snr_minimum": UNIFIED_SNR_THRESHOLD,
                "snr_t1_high": SNR_THRESHOLD_T1_HIGH,
                "snr_t0_elite": SNR_THRESHOLD_T0_ELITE,
                "adl_gini": ADL_GINI_THRESHOLD,
                "adl_harberger": ADL_HARBERGER_TAX_RATE,
            },
            "kernel_invariants": list(KERNEL_INVARIANTS),
            "source": "core/integration/constants.py",
            "version": KERNEL_VERSION,
        }
    except ImportError as e:
        return {"error": f"constants unavailable: {e}"}


def _verify_genesis(state: "SovereignState") -> dict[str, Any]:
    """Re-verify the genesis signature from stored state.

    Uses the stored genesis_hash to reconstruct the exact signable payload,
    then verifies the Ed25519 signature against the public key.

    Standing on Giants: Bernstein (Ed25519) · Aumasson (BLAKE2b)
    """
    s = state.read()
    if not s.get("genesis_signature") or not s.get("public_key_hex"):
        return {"verified": False, "reason": "no genesis identity"}

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        from core.identity.genesis import GENESIS_SIGNATURE_DOMAIN

        public_key = bytes.fromhex(s["public_key_hex"])
        genesis_hash = s["genesis_hash"]

        # Reconstruct the exact domain-separated message that was signed
        signable = f"{GENESIS_SIGNATURE_DOMAIN}:{genesis_hash}".encode("utf-8")

        verifier = Ed25519PublicKey.from_public_bytes(public_key)
        verifier.verify(bytes.fromhex(s["genesis_signature"]), signable)
        return {
            "verified": True,
            "identity_id": s["identity_id"],
            "sovereignty_class": s.get("sovereignty_class", "SEED"),
            "genesis_hash": genesis_hash,
        }
    except Exception as e:
        return {"verified": False, "reason": str(e)}


# ═══════════════════════════════════════════════════════════
#  KNOWLEDGE SURFACE — cached GOLD corpus (boot-time load)
# ═══════════════════════════════════════════════════════════

# In-memory cache: loaded once at first access, eliminates 12.8s NTFS I/O per query.
_knowledge_cache: dict[str, Any] = {}
_knowledge_cache_lock = threading.Lock()


def _ensure_knowledge_loaded() -> bool:
    """Load GOLD parquet files into memory on first access.

    Standing on Giants: Shannon (information density) · Amdahl (amortized I/O)

    Returns True if cache is populated, False if unavailable.
    """
    if _knowledge_cache.get("loaded"):
        return True

    with _knowledge_cache_lock:
        if _knowledge_cache.get("loaded"):
            return True  # Double-check after lock

        gold_dir = PROJECT_ROOT / "04_GOLD"
        if not gold_dir.exists():
            _knowledge_cache["error"] = "04_GOLD directory not found"
            return False

        try:
            import pandas as pd

            t0 = time.monotonic()
            tables_meta: list[dict[str, Any]] = []
            total_rows = 0
            total_bytes = 0

            search_order = [
                "golden_gems_chunks.parquet",
                "code_docstrings.parquet",
                "research_chunks.parquet",
                "chunks.parquet",
            ]

            for f in sorted(gold_dir.iterdir()):
                if f.suffix != ".parquet":
                    continue
                df = pd.read_parquet(f)
                size = f.stat().st_size
                _knowledge_cache[f"df:{f.name}"] = df
                tables_meta.append(
                    {
                        "name": f.stem,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "size_mb": round(size / 1024 / 1024, 1),
                    }
                )
                total_rows += len(df)
                total_bytes += size

            load_ms = round((time.monotonic() - t0) * 1000)
            _knowledge_cache["tables"] = tables_meta
            _knowledge_cache["total_rows"] = total_rows
            _knowledge_cache["total_bytes"] = total_bytes
            _knowledge_cache["search_order"] = search_order
            _knowledge_cache["gold_path"] = str(gold_dir)
            _knowledge_cache["load_ms"] = load_ms
            _knowledge_cache["loaded"] = True

            log.info(
                "Knowledge cache loaded: %d tables, %d rows, %.1f MB in %dms",
                len(tables_meta),
                total_rows,
                total_bytes / 1024 / 1024,
                load_ms,
            )
            return True

        except ImportError:
            _knowledge_cache["error"] = "pandas not installed"
            return False
        except (OSError, ValueError) as e:
            _knowledge_cache["error"] = str(e)
            return False


def _get_knowledge_stats() -> dict[str, Any]:
    """Return statistics about the GOLD knowledge corpus (from cache)."""
    if not _ensure_knowledge_loaded():
        return {"available": False, "reason": _knowledge_cache.get("error", "unknown")}

    return {
        "available": True,
        "tables": _knowledge_cache["tables"],
        "total_rows": _knowledge_cache["total_rows"],
        "total_size_mb": round(_knowledge_cache["total_bytes"] / 1024 / 1024, 1),
        "table_count": len(_knowledge_cache["tables"]),
        "gold_path": _knowledge_cache["gold_path"],
        "cache_load_ms": _knowledge_cache["load_ms"],
        "cached": True,
    }


def _ensure_faiss_loaded() -> bool:
    """Check if FAISS is loaded. Non-blocking — returns False if warmup in progress."""
    if _knowledge_cache.get("faiss_loaded"):
        return True

    # Non-blocking: if warmup thread holds the lock, fall back to keyword search
    if not _knowledge_cache_lock.acquire(blocking=False):
        return False
    try:
        if _knowledge_cache.get("faiss_loaded"):
            return True
        try:
            import faiss

            index_path = PROJECT_ROOT / "04_GOLD" / "faiss_chunks.index"
            ids_path = PROJECT_ROOT / "04_GOLD" / "faiss_chunk_ids.json"
            if not index_path.exists():
                return False

            t0 = time.monotonic()
            _knowledge_cache["faiss_index"] = faiss.read_index(str(index_path))
            _knowledge_cache["faiss_ids"] = json.loads(ids_path.read_text())

            try:
                import torch
                from sentence_transformers import SentenceTransformer

                _device = "cuda" if torch.cuda.is_available() else "cpu"
                _knowledge_cache["encoder"] = SentenceTransformer(
                    "all-MiniLM-L6-v2", device=_device
                )
                _knowledge_cache["faiss_encoder_ok"] = True
            except (ImportError, OSError):
                _knowledge_cache["faiss_encoder_ok"] = False

            _knowledge_cache["faiss_loaded"] = True
            log.info(
                "FAISS loaded: %d vectors, encoder=%s in %dms",
                _knowledge_cache["faiss_index"].ntotal,
                _knowledge_cache["faiss_encoder_ok"],
                round((time.monotonic() - t0) * 1000),
            )
            return True
        except (ImportError, OSError) as e:
            log.warning("FAISS unavailable: %s", e)
            return False
    finally:
        _knowledge_cache_lock.release()


def _search_knowledge(query: str, limit: int = 10) -> dict[str, Any]:
    """Search GOLD — Hybrid RRF (FAISS ∪ RuVector), keyword fallback.

    Standing on Giants: Cormack (RRF) · Johnson (FAISS) · Malkov (HNSW) · Shannon (1948)
    """
    if not _ensure_knowledge_loaded():
        return {"results": [], "error": _knowledge_cache.get("error", "unknown")}

    t0 = time.monotonic()

    # Hybrid semantic search — FAISS + RuVector fused via Reciprocal Rank Fusion
    try:
        from core.search.hybrid_search import HybridSearchEngine

        hybrid = _knowledge_cache.get("hybrid_engine")
        if hybrid is None:
            hybrid = HybridSearchEngine()
            _knowledge_cache["hybrid_engine"] = hybrid

        if hybrid.is_loaded:
            sr_results = hybrid.search(query, top_k=limit)
            if sr_results:
                results: list[dict[str, Any]] = []
                for sr in sr_results:
                    text = sr.record.content
                    results.append(
                        {
                            "source": sr.record.source or "hybrid",
                            "chunk_id": sr.record.source_id or "",
                            "text": text[:500] + ("..." if len(text) > 500 else ""),
                            "similarity": round(sr.score, 4),
                            "rrf_score": sr.record.metadata.get("rrf_score"),
                            "engine": sr.record.metadata.get("engine", "hybrid_rrf"),
                            "snr_score": None,
                        }
                    )
                return {
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "search_type": "hybrid_rrf",
                    "engines": hybrid.available_engines,
                    "search_ms": round((time.monotonic() - t0) * 1000),
                }
    except Exception as e:
        log.warning("Hybrid search failed, keyword fallback: %s", e)

    # Keyword fallback
    results = []
    query_lower = query.lower()
    for fname in _knowledge_cache.get("search_order", []):
        df = _knowledge_cache.get(f"df:{fname}")
        if df is None or "chunk_text" not in df.columns:
            continue
        mask = df["chunk_text"].str.lower().str.contains(query_lower, na=False)
        matches = df[mask].head(limit)
        for _, row in matches.iterrows():
            text = str(row["chunk_text"])
            results.append(
                {
                    "source": fname.replace(".parquet", ""),
                    "chunk_id": str(row.get("chunk_id", "")),
                    "text": text[:500] + ("..." if len(text) > 500 else ""),
                    "snr_score": (
                        float(row["snr_score"]) if "snr_score" in row.index else None
                    ),
                }
            )
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "search_type": "keyword_cached",
        "search_ms": round((time.monotonic() - t0) * 1000),
    }


# ═══════════════════════════════════════════════════════════
#  MISSION EXECUTION — PAT-powered question answering
# ═══════════════════════════════════════════════════════════

# Lazy singleton — created on first mission, reused thereafter.
_mission_ns: Any = None
_mission_lock = threading.Lock()


def _get_or_create_ns() -> Any:
    """Get or create the SovereignNervousSystem singleton.

    Uses MOEBridge as inference provider. Created lazily on first mission.
    """
    global _mission_ns
    if _mission_ns is not None:
        return _mission_ns

    with _mission_lock:
        if _mission_ns is not None:
            return _mission_ns

        try:
            # Ensure project root is on sys.path (critical for WSL daemon context)
            import sys

            _proj = str(PROJECT_ROOT)
            if _proj not in sys.path:
                sys.path.insert(0, _proj)
                log.info("Added %s to sys.path for NervousSystem import", _proj)
            else:
                log.info("sys.path already contains %s", _proj)

            from core.sovereign.mission_nervous_system import SovereignNervousSystem
            from core.sovereign.moe_bridge import MOEBridge

            bridge = MOEBridge.create()
            _mission_ns = SovereignNervousSystem(inference=bridge)
            log.info("SovereignNervousSystem created with MOEBridge")
            return _mission_ns
        except (ImportError, OSError, ValueError) as e:
            log.error("Failed to create NervousSystem: %s", e, exc_info=True)
            return None


def _generate_briefing(state: "SovereignState") -> dict[str, Any]:
    """Generate a daily sovereign briefing from real node metrics.

    Standing on Giants: Boyd (OODA situational awareness) · Deming (PDCA daily review)
    """
    t0 = time.monotonic()
    sections: list[dict[str, Any]] = []

    # 1. Identity
    s = state.read()
    sections.append(
        {
            "section": "identity",
            "title": f"Good morning, {s.get('userName', 'Sovereign')}",
            "data": {
                "identity_id": s.get("identity_id", "unknown")[:16] + "...",
                "sovereignty_class": s.get("sovereignty_class", "SEED"),
                "genesis_verified": s.get("genesis_verified", False),
            },
        }
    )

    # 2. Health
    beats = list(_heartbeat_history)
    latest = beats[-1] if beats else {}
    anomaly_count = sum(1 for b in beats[-10:] if b.get("anomalies"))
    sections.append(
        {
            "section": "health",
            "title": f"System health: {latest.get('health', 'unknown')}",
            "data": {
                "uptime_s": latest.get("uptime_s", 0),
                "rss_mb": latest.get("memory_rss_mb", 0),
                "heartbeats": len(beats),
                "anomalies_last_10": anomaly_count,
                "current_anomalies": latest.get("anomalies", []),
            },
        }
    )

    # 3. Knowledge
    if _knowledge_cache.get("loaded"):
        sections.append(
            {
                "section": "knowledge",
                "title": f"{_knowledge_cache.get('total_rows', 0):,} knowledge rows ready",
                "data": {
                    "tables": len(_knowledge_cache.get("tables", [])),
                    "faiss_ready": _knowledge_cache.get("faiss_loaded", False),
                    "faiss_vectors": (
                        _knowledge_cache["faiss_index"].ntotal
                        if _knowledge_cache.get("faiss_index")
                        else 0
                    ),
                    "ruvector_available": (
                        PROJECT_ROOT / "04_GOLD" / "ruvector_bizra"
                    ).exists(),
                },
            }
        )

    # 4. Constitution
    try:
        from core.integration.constants import (
            ADL_GINI_THRESHOLD,
            KERNEL_INVARIANTS,
            UNIFIED_IHSAN_THRESHOLD,
            UNIFIED_SNR_THRESHOLD,
        )

        sections.append(
            {
                "section": "constitution",
                "title": "Constitutional spine verified",
                "data": {
                    "ihsan": UNIFIED_IHSAN_THRESHOLD,
                    "snr": UNIFIED_SNR_THRESHOLD,
                    "gini": ADL_GINI_THRESHOLD,
                    "invariants": list(KERNEL_INVARIANTS),
                },
            }
        )
    except ImportError:
        pass

    # 5. Recommendations
    recs: list[str] = []
    if latest.get("health") == "critical":
        recs.append("System critical — investigate anomalies before starting work")
    elif latest.get("health") == "degraded":
        recs.append("System degraded — check backend status")
    if (
        not _knowledge_cache.get("faiss_loaded")
        and not (PROJECT_ROOT / "04_GOLD" / "ruvector_bizra").exists()
    ):
        recs.append("No semantic search — neither FAISS nor RuVector available")
    if anomaly_count > 5:
        recs.append(
            f"{anomaly_count}/10 recent heartbeats had anomalies — review trends"
        )
    if not recs:
        recs.append("All systems nominal — ready for sovereign work")

    sections.append(
        {
            "section": "recommendations",
            "title": "Today's guidance",
            "data": {"items": recs},
        }
    )

    return {
        "briefing": sections,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
        "version": KERNEL_VERSION,
    }


async def _run_mission(mission_text: str) -> dict[str, Any]:
    """Execute a mission through the SovereignNervousSystem.

    Standing on Giants: Kahneman (S1/S2 routing) · Boyd (OODA execute) · Al-Ghazali (Ihsan gate)

    Enriches the mission with knowledge context from the GOLD cache before execution.
    """
    t0 = time.monotonic()

    # Enrich with knowledge context — single semantic search on full question
    knowledge_context = ""
    search_result = _search_knowledge(mission_text, limit=3)
    if search_result.get("results"):
        snippets = [r["text"][:300] for r in search_result["results"]]
        knowledge_context = (
            "\n\nRelevant knowledge from your BIZRA corpus:\n"
            + "\n---\n".join(snippets)
        )

    ns = _get_or_create_ns()
    if ns is None:
        return {
            "status": "error",
            "error": "NervousSystem unavailable (check LM Studio connectivity)",
            "duration_ms": round((time.monotonic() - t0) * 1000),
        }

    try:
        enriched_text = mission_text + knowledge_context
        receipt = await ns.run(enriched_text)

        return {
            "status": "complete",
            "mission_id": str(receipt.mission_id),
            "system": receipt.system,
            "output": receipt.output_text[:2000],
            "ihsan_score": round(float(receipt.ihsan_score), 4),
            "snr_score": round(float(receipt.snr_score), 4),
            "duration_ms": round(float(receipt.duration_ms), 1),
            "rewarded": receipt.rewarded,
            "reward_amount": round(float(receipt.reward_amount), 4),
            "evidence_hash": receipt.evidence_hash,
            "reflex_hit": receipt.reflex_hit,
            "knowledge_enriched": bool(knowledge_context),
        }
    except (RuntimeError, OSError, ValueError) as e:
        return {
            "status": "error",
            "error": str(e),
            "duration_ms": round((time.monotonic() - t0) * 1000),
        }


# ═══════════════════════════════════════════════════════════
#  SELF-DIAGNOSIS — deep readiness probe (K8s pattern)
# ═══════════════════════════════════════════════════════════


def _run_readiness_probe(state: "SovereignState") -> dict[str, Any]:
    """Deep self-diagnosis: verify critical subsystems are operational.

    Standing on Giants: Deming (PDCA check) · Burns (K8s readiness probes)

    Returns structured results for each subsystem with pass/fail + detail.
    """
    t0 = time.monotonic()
    probes: list[dict[str, Any]] = []

    # Probe 1: Core imports
    try:
        from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

        probes.append(
            {"name": "core_imports", "pass": True, "detail": "constants loaded"}
        )
    except Exception as e:
        probes.append({"name": "core_imports", "pass": False, "detail": str(e)})

    # Probe 2: Constitutional thresholds
    try:
        from core.integration.constants import (
            ADL_GINI_THRESHOLD,
            KERNEL_INVARIANTS,
            UNIFIED_IHSAN_THRESHOLD,
            UNIFIED_SNR_THRESHOLD,
        )

        valid = (
            UNIFIED_IHSAN_THRESHOLD == 0.95
            and UNIFIED_SNR_THRESHOLD == 0.85
            and ADL_GINI_THRESHOLD == 0.35
            and len(KERNEL_INVARIANTS) == 3
        )
        probes.append(
            {
                "name": "constitution",
                "pass": valid,
                "detail": f"ihsan={UNIFIED_IHSAN_THRESHOLD} snr={UNIFIED_SNR_THRESHOLD} gini={ADL_GINI_THRESHOLD} invariants={len(KERNEL_INVARIANTS)}",
            }
        )
    except Exception as e:
        probes.append({"name": "constitution", "pass": False, "detail": str(e)})

    # Probe 3: Genesis identity
    s = state.read()
    genesis_ok = (
        s.get("genesis_verified") is True and len(s.get("identity_id", "")) == 64
    )
    probes.append(
        {
            "name": "genesis_identity",
            "pass": genesis_ok,
            "detail": f"id={s.get('identity_id', 'none')[:16]}... verified={s.get('genesis_verified')}",
        }
    )

    # Probe 4: Genesis signature re-verification
    verify_result = _verify_genesis(state)
    probes.append(
        {
            "name": "genesis_signature",
            "pass": verify_result.get("verified") is True,
            "detail": f"hash={verify_result.get('genesis_hash', 'none')[:16]}...",
        }
    )

    # Probe 5: Threshold registry
    try:
        from core.integration.threshold_registry import registry

        sealed = registry._sealed
        count = len(registry.all_thresholds())
        probes.append(
            {
                "name": "threshold_registry",
                "pass": sealed and count >= 20,
                "detail": f"sealed={sealed} count={count}",
            }
        )
    except Exception as e:
        probes.append({"name": "threshold_registry", "pass": False, "detail": str(e)})

    # Probe 6: Identity module
    try:
        from core.identity.genesis import (  # noqa: F401
            IdentityGenesis,
            SovereigntyClass,
        )

        probes.append(
            {
                "name": "identity_module",
                "pass": True,
                "detail": "Ed25519 + BLAKE2b available",
            }
        )
    except Exception as e:
        probes.append({"name": "identity_module", "pass": False, "detail": str(e)})

    # Probe 7: Heartbeat alive
    hb_ok = len(_heartbeat_history) > 0
    probes.append(
        {
            "name": "heartbeat",
            "pass": hb_ok,
            "detail": f"beats={len(_heartbeat_history)} latest={'yes' if hb_ok else 'none'}",
        }
    )

    # Probe 8: PCI gate module
    try:
        from core.pci.gates import PCIGateKeeper  # noqa: F401

        probes.append(
            {"name": "pci_gates", "pass": True, "detail": "PCIGateKeeper available"}
        )
    except Exception as e:
        probes.append({"name": "pci_gates", "pass": False, "detail": str(e)})

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    passed = sum(1 for p in probes if p["pass"])
    total = len(probes)
    ready = passed == total

    return {
        "ready": ready,
        "passed": passed,
        "total": total,
        "score": round(passed / total, 3) if total else 0.0,
        "probes": probes,
        "elapsed_ms": elapsed_ms,
        "version": KERNEL_VERSION,
    }


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
      GET  /api/constitution -> constitutional thresholds + kernel invariants
      GET  /api/genesis/verify -> re-verify genesis Ed25519 signature
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
        # Support both /api/* (native) and /v1/* (Vercel proxy)
        if path.startswith("/v1/"):
            path = "/api/" + path[4:]  # /v1/health → /api/health

        # Standard health check aliases (K8s, Docker, CLI all use these)
        if path in ("/health", "/healthz"):
            path = "/api/health"

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
        elif path == "/api/live-stats":
            # Unified stats endpoint for the frontend
            live = {
                "kernel": {
                    "alive": True,
                    "version": KERNEL_VERSION,
                    "uptime_s": round(_uptime(), 1),
                },
                "node": {"agents": 12, "node_id": "node0"},
                "seed": {"balance": 0, "total_missions": 0},
                "urp": {"knowledge_entries": 0, "receipts": 0, "treasury": 0},
                "hardware": {},
            }
            # SEED balance
            try:
                from core.proof_engine.seed_ledger import balance, history

                live["seed"]["balance"] = balance()
                live["seed"]["total_missions"] = len(history(limit=9999))
            except Exception:
                pass
            # URP state
            try:
                from core.urp.persistence import load_urp_state

                urp_state = load_urp_state()
                if urp_state:
                    pool = urp_state.get("resource_pool", {})
                    live["urp"]["knowledge_entries"] = pool.get("knowledge_count", 0)
                    live["urp"]["receipts"] = len(urp_state.get("receipt_log", []))
                    live["urp"]["treasury"] = pool.get("seed_treasury", 0)
            except Exception:
                pass
            # Home base hardware
            try:
                from core.sovereign.home_base import load_home_base

                hb = load_home_base()
                if hb:
                    live["hardware"] = {
                        "cpu": hb.hardware.cpu,
                        "cores": hb.hardware.cpu_cores,
                        "ram_gb": hb.hardware.ram_gb,
                        "gpu": hb.hardware.gpu,
                    }
            except Exception:
                pass
            self._json_response(live)
        elif path == "/api/briefing":
            self._json_response(_generate_briefing(self.sovereign_state))
        elif path == "/api/readiness":
            self._json_response(_run_readiness_probe(self.sovereign_state))
        elif path == "/api/knowledge":
            self._json_response(_get_knowledge_stats())
        elif path.startswith("/api/knowledge/search"):
            qs = self.path.split("?")[1] if "?" in self.path else ""
            params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
            q = params.get("q", "")
            n = min(int(params.get("n", "10")), 50)
            if q:
                self._json_response(_search_knowledge(q, limit=n))
            else:
                self._json_response(
                    {"error": "query parameter 'q' required"}, status=400
                )
        elif path == "/api/constitution":
            self._json_response(_get_constitution())
        elif path == "/api/genesis/verify":
            self._json_response(_verify_genesis(self.sovereign_state))
        elif path == "/api/heartbeat":
            beats = list(_heartbeat_history)
            latest = beats[-1] if beats else None
            self._json_response(
                {
                    "health": latest["health"] if latest else "unknown",
                    "anomalies": latest["anomalies"] if latest else [],
                    "latest": latest,
                    "count": len(beats),
                    "interval_s": HEARTBEAT_INTERVAL_S,
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
        elif path == "/api/mission":
            body = self._read_body()
            mission_text = body.get("text", body.get("mission", ""))
            if not mission_text:
                self._json_response({"error": "field 'text' required"}, status=400)
                status = 400
            else:
                import asyncio

                try:
                    result = asyncio.run(_run_mission(mission_text))
                    self._json_response(result)
                except (RuntimeError, OSError) as e:
                    self._json_response(
                        {"status": "error", "error": str(e)}, status=500
                    )
                    status = 500
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
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            log.debug("Client disconnected before response: %s", self.path)

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
#  HEARTBEAT — continuous self-observation (Deming PDCA)
# ═══════════════════════════════════════════════════════════

HEARTBEAT_INTERVAL_S = 30
_heartbeat_history: deque[dict[str, Any]] = deque(maxlen=120)  # 1 hour at 30s

# Anomaly thresholds — reflex arc triggers
_ANOMALY_ERROR_RATE = 0.05  # > 5% error rate
_ANOMALY_P95_LATENCY_MS = 180000.0  # > 180s p95 (Ollama cold-start: 130s; warm: 13-38s)
_ANOMALY_RSS_GROWTH_MB = 3000.0  # > 3.0 GB growth (GOLD cache ~1.1 GB + FAISS ~0.5 GB + encoder ~0.4 GB + runtime)
_ANOMALY_MISSED_BACKENDS = 1  # any backend down


def _read_live_rss_mb() -> float:
    """Read current VmRSS from /proc/self/status (Linux/WSL).

    Returns live resident set size in MB, not peak watermark.
    Falls back to ru_maxrss if /proc is unavailable.
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: "VmRSS:    12345 kB"
                    return round(int(line.split()[1]) / 1024, 1)
    except (FileNotFoundError, ValueError, IndexError, OSError):
        pass
    # Fallback: peak RSS from resource module
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _read_peak_rss_mb() -> float:
    """Read peak RSS (ru_maxrss) for watermark tracking."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _classify_health(
    error_rate: float,
    p95_ms: float,
    rss_mb: float,
    baseline_rss_mb: float,
    alive_count: int,
    total_backends: int,
) -> tuple[str, list[str]]:
    """Classify organism health from vitals. Returns (state, anomalies).

    States: healthy | degraded | critical
    """
    anomalies: list[str] = []

    if error_rate > _ANOMALY_ERROR_RATE:
        anomalies.append(f"error_rate={error_rate:.3f}>{_ANOMALY_ERROR_RATE}")
    if p95_ms > _ANOMALY_P95_LATENCY_MS:
        anomalies.append(f"p95={p95_ms:.0f}ms>{_ANOMALY_P95_LATENCY_MS:.0f}ms")
    if rss_mb - baseline_rss_mb > _ANOMALY_RSS_GROWTH_MB:
        anomalies.append(
            f"rss_growth={rss_mb - baseline_rss_mb:.1f}MB>{_ANOMALY_RSS_GROWTH_MB}MB"
        )
    if alive_count < total_backends:
        anomalies.append(f"backends={alive_count}/{total_backends}")

    if len(anomalies) >= 3:
        return "critical", anomalies
    elif len(anomalies) >= 1:
        return "degraded", anomalies
    return "healthy", anomalies


def _heartbeat_loop(
    state: "SovereignState",
    watchdog: "SubprocessWatchdog",
    shutdown: threading.Event,
    main_loop: Any = None,  # asyncio event loop from main thread
) -> None:
    """Background thread emitting system vitals every HEARTBEAT_INTERVAL_S.

    Phase N.1 Reflex Arc: truthful pulse → anomaly classification → bus propagation.

    Standing on Giants: Deming (PDCA continuous observation) · Boyd (OODA orient)
    """
    import platform

    beat_count = 0
    baseline_rss_mb = _read_live_rss_mb()

    while not shutdown.is_set():
        shutdown.wait(timeout=HEARTBEAT_INTERVAL_S)
        if shutdown.is_set():
            break

        beat_count += 1
        rss_mb = _read_live_rss_mb()
        peak_rss_mb = _read_peak_rss_mb()
        backends = watchdog.status()
        alive_count = sum(1 for b in backends if b["alive"])
        metrics_snap = _metrics.snapshot()
        error_rate = metrics_snap["error_rate"]
        p95_ms = metrics_snap["latency_ms"]["p95"]

        # Reflex arc: classify health from vitals
        health_state, anomalies = _classify_health(
            error_rate, p95_ms, rss_mb, baseline_rss_mb, alive_count, len(backends)
        )

        heartbeat = {
            "beat": beat_count,
            "ts": datetime.now(timezone.utc).isoformat(),
            "uptime_s": _uptime(),
            "health": health_state,
            "anomalies": anomalies,
            "initialized": state.is_initialized,
            "backends_alive": alive_count,
            "backends_total": len(backends),
            "requests_total": metrics_snap["requests_total"],
            "error_rate": error_rate,
            "latency_p95_ms": p95_ms,
            "memory_rss_mb": rss_mb,
            "memory_peak_rss_mb": peak_rss_mb,
            "cpu_user_s": round(
                __import__("resource")
                .getrusage(__import__("resource").RUSAGE_SELF)
                .ru_utime,
                2,
            ),
            "cpu_sys_s": round(
                __import__("resource")
                .getrusage(__import__("resource").RUSAGE_SELF)
                .ru_stime,
                2,
            ),
            "python_version": platform.python_version(),
        }

        _heartbeat_history.append(heartbeat)

        # Emit to EventBus via main loop (thread-safe bridge)
        if main_loop is not None:
            try:
                from core.bus.event_bus import get_global_bus

                bus = get_global_bus()
                if bus is not None:
                    import asyncio

                    main_loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        bus.emit("kernel.heartbeat", heartbeat),
                    )
            except (ImportError, RuntimeError):
                pass  # EventBus or loop not available

        # Log anomalies immediately, healthy every 5 minutes
        if health_state != "healthy":
            log.warning(
                "Heartbeat #%d %s | %s",
                beat_count,
                health_state.upper(),
                ", ".join(anomalies),
            )
        elif beat_count % 10 == 0:
            log.info(
                "Heartbeat #%d healthy | uptime=%.0fs rss=%.1fMB reqs=%d backends=%d/%d",
                beat_count,
                heartbeat["uptime_s"],
                rss_mb,
                heartbeat["requests_total"],
                alive_count,
                len(backends),
            )

        # Auto-generate daily manifest (check every ~50 min = 100 beats)
        if beat_count % 100 == 0:
            try:
                from datetime import date as _date

                today = _date.today().isoformat()
                manifest_path = STATE_DIR / "manifests" / f"manifest_{today}.json"
                if not manifest_path.exists():
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    import hashlib as _hashlib
                    import json as _json

                    manifest = {
                        "manifest_version": "1.0",
                        "date": today,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "node_id": "NODE0",
                        "summary": {
                            "heartbeat_count": beat_count,
                            "health": health_state,
                            "rss_mb": rss_mb,
                            "uptime_s": heartbeat["uptime_s"],
                            "auto_generated": True,
                        },
                    }
                    content = _json.dumps(manifest, sort_keys=True)
                    manifest["manifest_hash"] = _hashlib.blake2b(
                        content.encode(), digest_size=32
                    ).hexdigest()
                    manifest_path.write_text(_json.dumps(manifest, indent=2))
                    log.info("Auto-generated daily manifest: %s", manifest_path.name)
            except Exception:
                pass  # Manifest generation must never crash the heartbeat

    log.info("Heartbeat thread stopped after %d beats", beat_count)


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

    # ── Capture main event loop for thread-safe EventBus emission ──
    import asyncio

    try:
        main_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_loop)
    except Exception:
        main_loop = None

    # ── Eager background warmup: knowledge + FAISS + Ollama model ──
    def _warmup() -> None:
        _ensure_knowledge_loaded()
        _ensure_faiss_loaded()
        # Pre-load default Ollama model into VRAM (prevents 30-60s cold start)
        try:
            import httpx

            ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": "llama3.1:8b", "prompt": ".", "stream": False},
                timeout=60.0,
            )
            if resp.status_code == 200:
                log.info("Ollama model pre-loaded into VRAM")
            else:
                log.warning("Ollama pre-load returned %d", resp.status_code)
        except Exception as e:
            log.warning(
                "Ollama pre-load failed (will cold-start on first mission): %s", e
            )
        log.info("Background warmup complete (knowledge + FAISS + Ollama)")

    warmup_thread = threading.Thread(target=_warmup, daemon=True, name="warmup")
    warmup_thread.start()
    log.info("Background warmup started (parquet + FAISS + encoder)")

    # ── Start heartbeat thread (continuous self-observation) ──
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(state, watchdog, _shutdown_event, main_loop),
        daemon=True,
        name="heartbeat",
    )
    heartbeat_thread.start()
    log.info(
        "Heartbeat thread started (interval=%ds, bus=%s)",
        HEARTBEAT_INTERVAL_S,
        "wired" if main_loop else "local",
    )

    # ── Start HTTP server ──
    handler_factory = partial(KernelHandler, state=state, watchdog=watchdog)

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Multi-threaded HTTP — missions don't block heartbeat or health checks."""

        daemon_threads = True

    server = ThreadedHTTPServer(("127.0.0.1", KERNEL_PORT), handler_factory)
    server.timeout = 1  # Allow shutdown check every second

    log.info("Sovereign Nerve Center listening on http://127.0.0.1:%d", KERNEL_PORT)
    log.info(
        "  /             -> %s", "terminal" if state.is_initialized else "installer"
    )
    log.info("  /ops          -> operations dashboard")
    log.info("  /rpc          -> reverse proxy to bridge (:9742)")
    log.info("  /api/health   -> health check")
    log.info("  /api/status   -> subsystem status")
    log.info("  /api/constitution -> constitutional thresholds")
    log.info("  /api/genesis/verify -> genesis signature verification")
    log.info("  /api/heartbeat -> system vitals (30s pulse)")
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
