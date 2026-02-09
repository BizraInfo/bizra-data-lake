"""
Rust Lifecycle Integration — BIZRA Proactive Sovereign Entity
==============================================================
Bridges the Python proactive system with Rust bizra-omega components.

Architecture:
    ProactiveSovereignEntity (Python)
           ↓
    RustLifecycleManager (This Module)
           ↓
    bizra-omega (Rust API + Federation + Inference)

Standing on Giants:
    - Rust/PyO3: Safe FFI with 10-100x performance
    - Axum: Async Rust HTTP server
    - Tokio: Async runtime coordination
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

# Try aiohttp, fallback to httpx or basic urllib
_AIOHTTP_AVAILABLE = False
_HTTPX_AVAILABLE = False

try:
    import aiohttp

    _AIOHTTP_AVAILABLE = True
except ImportError:
    try:
        import httpx

        _HTTPX_AVAILABLE = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)


# =============================================================================
# RUST SERVICE STATUS
# =============================================================================


class RustServiceStatus(Enum):
    """Status of Rust services."""

    UNKNOWN = auto()
    STARTING = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    STOPPED = auto()


@dataclass
class RustServiceHealth:
    """Health information for a Rust service."""

    service: str
    status: RustServiceStatus
    version: Optional[str] = None
    uptime_seconds: float = 0.0
    requests_served: int = 0
    last_check: float = field(default_factory=time.time)
    error: Optional[str] = None

    def is_healthy(self) -> bool:
        return self.status == RustServiceStatus.HEALTHY


# =============================================================================
# RUST API CLIENT
# =============================================================================


class RustAPIClient:
    """
    Async client for bizra-omega REST API.

    Endpoints:
        - /api/v1/health — Health check
        - /api/v1/status — Node status
        - /api/v1/inference/generate — LLM inference
        - /api/v1/pci/gates/check — Constitutional gate check
        - /api/v1/federation/status — P2P status

    Supports aiohttp (preferred), httpx, or falls back to sync urllib.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3001",
        timeout: float = 10.0,
        max_connections: int = 100,  # PERF: Connection pool limit
        max_connections_per_host: int = 20,  # PERF: Per-host limit
    ):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must use http:// or https:// scheme, got: {base_url}")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self._session: Optional[Any] = None
        self._connector: Optional[Any] = None
        self._client_type = "none"

        if _AIOHTTP_AVAILABLE:
            self._client_type = "aiohttp"
        elif _HTTPX_AVAILABLE:
            self._client_type = "httpx"

    async def _get_session(self) -> Any:
        """Get or create HTTP session with connection pooling."""
        if self._client_type == "aiohttp":
            if self._session is None or self._session.closed:
                # PERF FIX: Use TCPConnector with connection pooling
                self._connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_connections_per_host,
                    keepalive_timeout=30,  # Keep connections alive
                    enable_cleanup_closed=True,
                )
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                )
            return self._session
        elif self._client_type == "httpx":
            if self._session is None:
                # PERF FIX: Configure httpx connection limits
                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_connections_per_host,
                )
                self._session = httpx.AsyncClient(
                    timeout=self.timeout,
                    limits=limits,
                )
            return self._session
        return None

    async def close(self) -> None:
        """Close HTTP session and connection pool."""
        if self._session is not None:
            if self._client_type == "aiohttp" and not self._session.closed:
                await self._session.close()
                # PERF FIX: Wait for connector cleanup
                if self._connector:
                    await self._connector.close()
                    self._connector = None
            elif self._client_type == "httpx":
                await self._session.aclose()
            self._session = None

    async def _get(self, path: str) -> tuple[int, Optional[Dict[str, Any]]]:
        """Make GET request."""
        url = f"{self.base_url}{path}"

        if self._client_type == "aiohttp":
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    return resp.status, await resp.json()
                return resp.status, None
        elif self._client_type == "httpx":
            session = await self._get_session()
            resp = await session.get(url)
            if resp.status_code == 200:
                return resp.status_code, resp.json()
            return resp.status_code, None
        else:
            # Sync fallback using urllib
            import json
            import urllib.request

            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # nosec B310 — URL scheme validated in __init__
                    return resp.status, json.loads(resp.read().decode())
            except Exception:
                return 0, None

    async def _post(
        self, path: str, json_data: Dict[str, Any], timeout: Optional[float] = None
    ) -> tuple[int, Optional[Dict[str, Any]]]:
        """Make POST request."""
        url = f"{self.base_url}{path}"
        actual_timeout = timeout or self.timeout

        if self._client_type == "aiohttp":
            session = await self._get_session()
            async with session.post(
                url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=actual_timeout),
            ) as resp:
                if resp.status == 200:
                    return resp.status, await resp.json()
                return resp.status, None
        elif self._client_type == "httpx":
            session = await self._get_session()
            resp = await session.post(url, json=json_data, timeout=actual_timeout)
            if resp.status_code == 200:
                return resp.status_code, resp.json()
            return resp.status_code, None
        else:
            # Sync fallback using urllib
            import json
            import urllib.request

            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(json_data).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=actual_timeout) as resp:  # nosec B310 — URL scheme validated in __init__
                    return resp.status, json.loads(resp.read().decode())
            except Exception:
                return 0, None

    async def health_check(self) -> RustServiceHealth:
        """Check Rust API health."""
        try:
            status, data = await self._get("/api/v1/health")
            if status == 200 and data:
                return RustServiceHealth(
                    service="bizra-api",
                    status=RustServiceStatus.HEALTHY,
                    version=data.get("version"),
                )
            return RustServiceHealth(
                service="bizra-api",
                status=RustServiceStatus.DEGRADED,
                error=f"HTTP {status}",
            )
        except Exception as e:
            return RustServiceHealth(
                service="bizra-api",
                status=RustServiceStatus.UNHEALTHY,
                error=str(e),
            )

    async def get_status(self) -> Dict[str, Any]:
        """Get full node status."""
        try:
            status, data = await self._get("/api/v1/status")
            if status == 200 and data:
                return data
            return {"error": f"HTTP {status}"}
        except Exception as e:
            return {"error": str(e)}

    async def check_gates(
        self,
        content: bytes,
        snr_score: float,
        ihsan_score: float,
    ) -> Dict[str, Any]:
        """
        Check content through Rust PCI gates.

        This provides 10-100x faster gate checking than Python.
        """
        try:
            status, data = await self._post(
                "/api/v1/pci/gates/check",
                {
                    "content": content.decode("utf-8", errors="replace"),
                    "snr_score": snr_score,
                    "ihsan_score": ihsan_score,
                },
            )
            if status == 200 and data:
                return data
            return {"error": f"HTTP {status}", "passed": False}
        except Exception as e:
            return {"error": str(e), "passed": False}

    async def inference_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: str = "local",
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate inference via Rust gateway.

        Uses Rust tier selection for optimal model routing.
        """
        try:
            payload = {
                "prompt": prompt,
                "tier": tier,
                "max_tokens": max_tokens,
            }
            if system_prompt:
                payload["system"] = system_prompt

            status, data = await self._post(
                "/api/v1/inference/generate",
                payload,
                timeout=60.0,  # Longer for inference
            )
            if status == 200 and data:
                return data
            return {"error": f"HTTP {status}"}
        except Exception as e:
            return {"error": str(e)}

    async def get_federation_status(self) -> Dict[str, Any]:
        """Get federation/P2P status."""
        try:
            status, data = await self._get("/api/v1/federation/status")
            if status == 200 and data:
                return data
            return {"error": f"HTTP {status}", "connected": False}
        except Exception as e:
            return {"error": str(e), "connected": False}


# =============================================================================
# RUST PROCESS MANAGER
# =============================================================================


class RustProcessManager:
    """
    Manages Rust service processes.

    Can start/stop bizra-api and monitor its health.
    """

    def __init__(
        self,
        bizra_omega_path: Optional[str] = None,
        api_port: int = 3001,
        gossip_port: int = 7946,
    ):
        self.bizra_omega_path = bizra_omega_path or self._find_bizra_omega()
        self.api_port = api_port
        self.gossip_port = gossip_port
        self._process: Optional[subprocess.Popen] = None
        self._started_at: Optional[float] = None

    def _find_bizra_omega(self) -> str:
        """Find bizra-omega directory."""
        # Check relative to this file
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        omega_path = os.path.join(base, "bizra-omega")
        if os.path.exists(omega_path):
            return omega_path
        # Check environment variable
        return os.environ.get("BIZRA_OMEGA_PATH", omega_path)

    def is_running(self) -> bool:
        """Check if Rust process is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    async def start(self, wait_for_health: bool = True) -> bool:
        """
        Start the Rust API server.

        Returns True if started successfully.
        """
        if self.is_running():
            logger.info("Rust API already running")
            return True

        binary_path = os.path.join(
            self.bizra_omega_path, "target", "release", "bizra-api"
        )

        # Try debug binary if release doesn't exist
        if not os.path.exists(binary_path):
            binary_path = os.path.join(
                self.bizra_omega_path, "target", "debug", "bizra-api"
            )

        if not os.path.exists(binary_path):
            logger.warning("Rust binary not found at %s", binary_path)
            return False

        try:
            self._process = subprocess.Popen(
                [
                    binary_path,
                    "--port",
                    str(self.api_port),
                    "--host",
                    "0.0.0.0",  # nosec B104 — intentional: Rust API must be reachable from containers/WSL
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "RUST_LOG": "info"},
            )
            self._started_at = time.time()
            logger.info("Started Rust API on port %d", self.api_port)

            if wait_for_health:
                return await self._wait_for_health()
            return True

        except Exception as e:
            logger.error("Failed to start Rust API: %s", e)
            return False

    async def _wait_for_health(self, timeout: float = 30.0) -> bool:
        """Wait for Rust API to become healthy."""
        client = RustAPIClient(base_url=f"http://localhost:{self.api_port}")
        start = time.time()

        while time.time() - start < timeout:
            health = await client.health_check()
            if health.is_healthy():
                await client.close()
                logger.info("Rust API healthy after %.1fs", time.time() - start)
                return True
            await asyncio.sleep(0.5)

        await client.close()
        logger.warning("Rust API did not become healthy within %.1fs", timeout)
        return False

    def stop(self) -> None:
        """Stop the Rust API server."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._started_at = None
            logger.info("Stopped Rust API")

    def uptime(self) -> float:
        """Get uptime in seconds."""
        if self._started_at is None:
            return 0.0
        return time.time() - self._started_at


# =============================================================================
# RUST LIFECYCLE MANAGER
# =============================================================================


class RustLifecycleManager:
    """
    Manages the complete Rust integration lifecycle.

    Coordinates:
        - PyO3 bindings (in-process, fastest)
        - REST API client (inter-process)
        - Process management (start/stop)
        - Health monitoring
    """

    def __init__(
        self,
        api_port: int = 3001,
        use_pyo3: bool = True,
        auto_start_api: bool = False,
        health_check_interval: float = 30.0,
    ):
        self.api_port = api_port
        self.use_pyo3 = use_pyo3
        self.auto_start_api = auto_start_api
        self.health_check_interval = health_check_interval

        # Components
        self._api_client = RustAPIClient(base_url=f"http://localhost:{api_port}")
        self._process_manager = RustProcessManager(api_port=api_port)

        # State
        self._pyo3_available = False
        self._pyo3_module = None
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._last_health: Optional[RustServiceHealth] = None

        # Callbacks
        self._on_health_change: Optional[Callable[[RustServiceHealth], None]] = None

        # Try to load PyO3 bindings
        self._try_load_pyo3()

    def _try_load_pyo3(self) -> None:
        """Try to load PyO3 bindings."""
        if not self.use_pyo3:
            return

        try:
            # Try installed package first
            import bizra

            self._pyo3_module = bizra
            self._pyo3_available = True
            logger.info("PyO3 bindings loaded (10-100x faster crypto)")
        except ImportError:
            # Try from rust_bridge fallback
            try:
                from core.sovereign.rust_bridge import (  # type: ignore[import-untyped]
                    _rust_bizra,
                    is_rust_available,
                )

                if is_rust_available():
                    self._pyo3_module = _rust_bizra
                    self._pyo3_available = True
                    logger.info("PyO3 bindings loaded via rust_bridge")
            except ImportError:
                logger.info("PyO3 bindings not available, using Python fallback")

    @property
    def pyo3_available(self) -> bool:
        """Check if PyO3 bindings are available."""
        return self._pyo3_available

    @property
    def api_healthy(self) -> bool:
        """Check if REST API is healthy."""
        return self._last_health is not None and self._last_health.is_healthy()

    async def start(self) -> Dict[str, Any]:
        """
        Start the Rust lifecycle manager.

        Returns:
            Status dict with component states.
        """
        result = {
            "pyo3_available": self._pyo3_available,
            "api_started": False,
            "api_healthy": False,
        }

        if self.auto_start_api:
            result["api_started"] = await self._process_manager.start()

        # Check API health
        self._last_health = await self._api_client.health_check()
        result["api_healthy"] = self._last_health.is_healthy()

        # Start health monitoring
        self._running = True
        self._health_task = asyncio.create_task(self._health_monitor_loop())

        logger.info(
            "Rust lifecycle started: PyO3=%s, API=%s",
            self._pyo3_available,
            result["api_healthy"],
        )
        return result

    async def stop(self) -> None:
        """Stop the Rust lifecycle manager."""
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        await self._api_client.close()

        if self.auto_start_api:
            self._process_manager.stop()

        logger.info("Rust lifecycle stopped")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            try:
                new_health = await self._api_client.health_check()

                # Notify on status change
                if self._on_health_change and (
                    self._last_health is None
                    or self._last_health.status != new_health.status
                ):
                    self._on_health_change(new_health)

                self._last_health = new_health

            except Exception as e:
                logger.warning("Health check error: %s", e)

            await asyncio.sleep(self.health_check_interval)

    def set_health_callback(
        self, callback: Callable[[RustServiceHealth], None]
    ) -> None:
        """Set callback for health status changes."""
        self._on_health_change = callback

    # -------------------------------------------------------------------------
    # PyO3 Operations (In-Process, Fastest)
    # -------------------------------------------------------------------------

    def pyo3_check_ihsan(self, score: float) -> bool:
        """Check Ihsan threshold using PyO3 (fastest)."""
        if not self._pyo3_available:
            return score >= 0.95  # Python fallback

        assert self._pyo3_module is not None
        constitution = self._pyo3_module.Constitution()
        return constitution.check_ihsan(score)

    def pyo3_check_snr(self, score: float) -> bool:
        """Check SNR threshold using PyO3 (fastest)."""
        if not self._pyo3_available:
            return score >= 0.85  # Python fallback

        assert self._pyo3_module is not None
        constitution = self._pyo3_module.Constitution()
        return constitution.check_snr(score)

    def pyo3_domain_digest(self, message: bytes) -> str:
        """Compute domain-separated digest using PyO3 (20x faster)."""
        if not self._pyo3_available:
            import hashlib

            return hashlib.blake2b(b"bizra-pci-v1:" + message).hexdigest()

        assert self._pyo3_module is not None
        return self._pyo3_module.domain_separated_digest(message)

    def pyo3_sign(self, message: bytes) -> tuple[str, str]:
        """
        Sign message using PyO3 (100x faster).

        Returns:
            (signature_hex, public_key_hex)
        """
        if not self._pyo3_available:
            raise RuntimeError("PyO3 bindings not available for signing")

        assert self._pyo3_module is not None
        identity = self._pyo3_module.NodeIdentity()
        signature = identity.sign(message)
        return signature, identity.public_key

    # -------------------------------------------------------------------------
    # PyO3 Autopoiesis Operations (In-Process, 10-100x Faster)
    # -------------------------------------------------------------------------

    def pyo3_pattern_memory_learn(
        self,
        node_id: str,
        content: str,
        embedding: list[float],
        tags: list[str] | None = None,
    ) -> str | None:
        """Learn a pattern using Rust PatternMemory (10-100x faster).

        Returns pattern ID on success, None if PyO3 unavailable.
        """
        if not self._pyo3_available:
            return None

        try:
            assert self._pyo3_module is not None
            memory = self._pyo3_module.PatternMemory(node_id)
            return memory.learn(content, embedding, tags or [])
        except Exception as e:
            logger.warning("PyO3 pattern learn failed: %s", e)
            return None

    def pyo3_pattern_memory_recall(
        self,
        node_id: str,
        embedding: list[float],
        limit: int = 5,
    ) -> list[tuple[str, float, list[str]]] | None:
        """Recall similar patterns using Rust PatternMemory.

        Returns list of (content, confidence, tags) tuples, or None if unavailable.
        """
        if not self._pyo3_available:
            return None

        try:
            assert self._pyo3_module is not None
            memory = self._pyo3_module.PatternMemory(node_id)
            return memory.recall(embedding, limit)
        except Exception as e:
            logger.warning("PyO3 pattern recall failed: %s", e)
            return None

    def pyo3_preference_observe(
        self,
        pref_type: str,
        key: str,
        value: str,
    ) -> bool:
        """Observe a preference using Rust PreferenceTracker.

        Returns True on success, False if PyO3 unavailable.
        """
        if not self._pyo3_available:
            return False

        try:
            if not hasattr(self, "_pyo3_pref_tracker"):
                assert self._pyo3_module is not None
                self._pyo3_pref_tracker = self._pyo3_module.PreferenceTracker()
            self._pyo3_pref_tracker.observe(pref_type, key, value)  # type: ignore[has-type]
            return True
        except Exception as e:
            logger.warning("PyO3 preference observe failed: %s", e)
            return False

    def pyo3_preference_apply(self, prompt: str) -> str:
        """Apply learned preferences to a prompt using Rust PreferenceTracker.

        Returns modified prompt, or original if PyO3 unavailable.
        """
        if not self._pyo3_available or not hasattr(self, "_pyo3_pref_tracker"):
            return prompt

        try:
            return self._pyo3_pref_tracker.apply_to_prompt(prompt)  # type: ignore[has-type]
        except Exception as e:
            logger.warning("PyO3 preference apply failed: %s", e)
            return prompt

    # -------------------------------------------------------------------------
    # REST API Operations (Inter-Process)
    # -------------------------------------------------------------------------

    async def api_check_gates(
        self,
        content: bytes,
        snr_score: float,
        ihsan_score: float,
    ) -> Dict[str, Any]:
        """Check gates via REST API."""
        return await self._api_client.check_gates(content, snr_score, ihsan_score)

    async def api_inference(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: str = "local",
    ) -> Dict[str, Any]:
        """Generate inference via REST API."""
        return await self._api_client.inference_generate(prompt, system_prompt, tier)

    async def api_federation_status(self) -> Dict[str, Any]:
        """Get federation status via REST API."""
        return await self._api_client.get_federation_status()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Get lifecycle manager statistics."""
        return {
            "pyo3_available": self._pyo3_available,
            "api_healthy": self.api_healthy,
            "api_port": self.api_port,
            "process_running": self._process_manager.is_running(),
            "process_uptime": self._process_manager.uptime(),
            "last_health": (
                self._last_health.status.name if self._last_health else "UNKNOWN"
            ),
            "running": self._running,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


async def create_rust_lifecycle(
    api_port: int = 3001,
    use_pyo3: bool = True,
    auto_start_api: bool = False,
    auto_start: bool = True,
) -> RustLifecycleManager:
    """
    Factory function to create RustLifecycleManager.

    Args:
        api_port: Port for Rust API (default 3001)
        use_pyo3: Whether to try loading PyO3 bindings
        auto_start_api: Whether to start Rust API process
        auto_start: Whether to start lifecycle monitoring

    Returns:
        Configured RustLifecycleManager
    """
    manager = RustLifecycleManager(
        api_port=api_port,
        use_pyo3=use_pyo3,
        auto_start_api=auto_start_api,
    )

    if auto_start:
        await manager.start()

    return manager


# Backwards-compatible alias expected by tooling
RustLifecycle = RustLifecycleManager


# =============================================================================
# INTEGRATION WITH OPPORTUNITY PIPELINE
# =============================================================================


def create_rust_gate_filter(
    lifecycle: RustLifecycleManager,
) -> "ConstitutionalFilter":  # type: ignore[name-defined]
    """
    Create a constitutional filter that uses Rust for validation.

    This provides 10-100x faster gate checking.
    """
    from core.sovereign.opportunity_pipeline import ConstitutionalFilter, FilterResult

    class RustGateFilter(ConstitutionalFilter):
        """Constitutional filter backed by Rust."""

        def __init__(self, lifecycle: RustLifecycleManager):
            super().__init__("Rust Gate", weight=2.0)
            self._lifecycle = lifecycle

        async def check(self, opportunity) -> FilterResult:
            # Use PyO3 for fastest checking
            if self._lifecycle.pyo3_available:
                ihsan_ok = self._lifecycle.pyo3_check_ihsan(opportunity.ihsan_score)
                snr_ok = self._lifecycle.pyo3_check_snr(opportunity.snr_score)

                if ihsan_ok and snr_ok:
                    return FilterResult(passed=True, reason="Rust gate passed")
                else:
                    reasons = []
                    if not ihsan_ok:
                        reasons.append(f"Ihsan {opportunity.ihsan_score:.3f} < 0.95")
                    if not snr_ok:
                        reasons.append(f"SNR {opportunity.snr_score:.2f} < 0.85")
                    return FilterResult(passed=False, reason="; ".join(reasons))

            # Fall back to REST API
            if self._lifecycle.api_healthy:
                result = await self._lifecycle.api_check_gates(
                    content=opportunity.description.encode(),
                    snr_score=opportunity.snr_score,
                    ihsan_score=opportunity.ihsan_score,
                )
                if result.get("error"):
                    return FilterResult(
                        passed=False, reason=f"Rust API error: {result['error']}"
                    )
                return FilterResult(
                    passed=result.get("passed", False),
                    reason=result.get("reason", "Rust API check"),
                )

            # Python fallback
            return FilterResult(
                passed=True, reason="Rust unavailable, skipping accelerated check"
            )

    return RustGateFilter(lifecycle)
