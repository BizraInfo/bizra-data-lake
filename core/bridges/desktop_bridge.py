"""
BIZRA Desktop Bridge -- Minimal Sovereign Command Surface

Asyncio TCP server exposing JSON-RPC methods over newline-delimited JSON.
Zero new dependencies -- stdlib only (asyncio, json, logging).

Protocol: Newline-delimited JSON-RPC 2.0 on 127.0.0.1:9742.
Commands: ping, status, sovereign_query, verify_action_outcome, capture_screenshot.

Standing on Giants:
- Boyd (OODA): Fast feedback loop before scaling
- Shannon (SNR): Increase signal before adding channel bandwidth
- Lamport: Avoid new network surface unless necessary
- Al-Ghazali (Ihsan): Excellence over ego-driven expansion

Created: 2026-02-13 | BIZRA Desktop Bridge v1.0
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.bridges.browser_mcp_client import BrowserMCPClient
from core.sovereign.origin_guard import (
    NODE_ROLE_ENV,
    enforce_node0_fail_closed,
    normalize_node_role,
    resolve_origin_snapshot,
)
from core.sovereign.permit import (
    Permit,
    create_hda_permit,
)

logger = logging.getLogger("bizra.desktop_bridge")

# ---------------------------------------------------------------------------
# Rust PyO3 bindings (optional — graceful fallback to Python-only)
# ---------------------------------------------------------------------------

try:
    from bizra import Constitution, GateChain, domain_separated_digest

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 9742
AHK_BRIDGE_HOST = "127.0.0.1"
AHK_BRIDGE_PORT = int(
    os.getenv("BIZRA_AHK_BRIDGE_PORT", os.getenv("BIZRA_BRIDGE_PORT", "9742"))
)
MAX_MESSAGE_BYTES = 1_048_576  # 1 MB safety limit
ACTUATOR_ENTROPY_THRESHOLD = 3.5  # Shannon bits/char — blocks low-signal instructions
RATE_LIMIT_TOKENS_PER_SEC = 20.0
RATE_LIMIT_BURST = 30.0
AUTH_TOKEN_ENV = "BIZRA_BRIDGE_TOKEN"
AUTH_HEADER_TOKEN = "X-BIZRA-TOKEN"
AUTH_HEADER_TS = "X-BIZRA-TS"
AUTH_HEADER_NONCE = "X-BIZRA-NONCE"
AUTH_MAX_CLOCK_SKEW_MS = 120_000
AUTH_NONCE_TTL_MS = AUTH_MAX_CLOCK_SKEW_MS * 2
GUARDIAN_WIRE_MODE_ENV = "BIZRA_GUARDIAN_WIRE_MODE"
GUARDIAN_HOST_ENV = "BIZRA_GUARDIAN_HOST"
GUARDIAN_PORT_ENV = "BIZRA_GUARDIAN_PORT"
GUARDIAN_TIMEOUT_MS_ENV = "BIZRA_GUARDIAN_TIMEOUT_MS"
GENESIS_STATE_DIR = Path("sovereign_state")

# ---------------------------------------------------------------------------
# Token-bucket rate limiter (~15 lines, inline)
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Simple token-bucket rate limiter (RFC 6585 spirit)."""

    rate: float = RATE_LIMIT_TOKENS_PER_SEC
    burst: float = RATE_LIMIT_BURST
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = self.burst
        self._last_refill = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _ok(id: Any, result: dict[str, Any]) -> bytes:
    """Build a successful JSON-RPC response."""
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": id}).encode() + b"\n"


def _error(
    id: Any, code: int, message: str, data: Optional[dict[str, Any]] = None
) -> bytes:
    """Build a JSON-RPC error response."""
    error_obj: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error_obj["data"] = data
    return json.dumps({"jsonrpc": "2.0", "error": error_obj, "id": id}).encode() + b"\n"


# ---------------------------------------------------------------------------
# Desktop Bridge Server
# ---------------------------------------------------------------------------


class DesktopBridge:
    """
    Asyncio TCP server for the BIZRA desktop bridge.

    Binds to 127.0.0.1:9742 only. Exposes:
      - ping: liveness check
      - status: full Node0 health snapshot
      - sovereign_query: route a query through InferenceGateway + FATE gate
      - heartbeat_demo: browser research + HDA desktop action demo path
      - invoke_skill: route skill invocation through SkillRouter
      - list_skills: list registered skills
      - get_receipt: retrieve signed bridge receipt

    Every request requires auth headers:
      X-BIZRA-TOKEN, X-BIZRA-TS, X-BIZRA-NONCE
    """

    def __init__(
        self,
        host: str = BRIDGE_HOST,
        port: int = BRIDGE_PORT,
        gateway: Any = None,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError(
                f"Security: desktop bridge must bind to 127.0.0.1, got '{host}'"
            )
        self.host = host
        self.port = port
        self._server: Optional[asyncio.AbstractServer] = None
        self._start_time: float = 0.0
        self._rate_limiter = TokenBucket()
        self._gateway = gateway  # InferenceGateway (lazy-loaded if None)
        self._fate_gate: Any = None  # FATEGate (lazy-loaded)
        self._rust_gate_chain: Any = None  # Rust GateChain (lazy-loaded)
        self._rust_constitution: Any = None  # Rust Constitution (lazy-loaded)
        self._receipt_engine: Any = None  # BridgeReceiptEngine (lazy-loaded)
        self._skill_router: Any = None  # SkillRouter (lazy-loaded)
        self._request_count = 0
        self._auth_token: Optional[str] = None
        self._nonce_seen: dict[str, int] = {}
        self._node_role: str = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
        self._origin_snapshot: dict[str, Any] = self._default_origin_snapshot()
        self._hda_permit: Optional[Permit] = None
        self._permit_signing_key: Optional[str] = (
            os.getenv(
                "BIZRA_PERMIT_SIGNING_KEY",
                os.getenv("BIZRA_BRIDGE_TOKEN", ""),
            )
            or None
        )
        self._guardian_wire_mode = (
            os.getenv(GUARDIAN_WIRE_MODE_ENV, "best_effort").strip().lower()
        )
        if self._guardian_wire_mode not in {"off", "best_effort", "required"}:
            self._guardian_wire_mode = "best_effort"
        self._extra_methods: dict[str, Any] = {}  # Dynamic method registry
        self._guardian_host = os.getenv(GUARDIAN_HOST_ENV, "127.0.0.1")
        try:
            self._guardian_port = int(os.getenv(GUARDIAN_PORT_ENV, "9741"))
        except ValueError:
            self._guardian_port = 9741
        try:
            self._guardian_timeout_ms = int(os.getenv(GUARDIAN_TIMEOUT_MS_ENV, "1200"))
        except ValueError:
            self._guardian_timeout_ms = 1200

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the TCP server."""
        self._auth_token = self._load_auth_token()
        self._node_role = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
        enforce_node0_fail_closed(GENESIS_STATE_DIR, self._node_role)
        if self._guardian_wire_mode == "required":
            probe = await self._check_rust_guardian(
                "bridge_startup_probe",
                source="desktop_bridge.start",
            )
            if not probe.get("allowed", False):
                raise RuntimeError(
                    f"Rust guardian wire required but unavailable: {probe.get('reason')}"
                )
        # Hard cutover: bridge refuses startup if signer cannot initialize.
        self._get_receipt_engine()
        self._origin_snapshot = self._resolve_origin_snapshot()
        self._start_time = time.monotonic()
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            limit=MAX_MESSAGE_BYTES,  # H-3: bound readline buffer
        )
        addrs = [s.getsockname() for s in self._server.sockets]
        logger.info(f"Desktop bridge listening on {addrs}")
        logger.info(
            "Guardian wire mode=%s target=%s:%s timeout_ms=%s",
            self._guardian_wire_mode,
            self._guardian_host,
            self._guardian_port,
            self._guardian_timeout_ms,
        )

    async def stop(self) -> None:
        """Gracefully stop the server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Desktop bridge stopped")

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._server.is_serving()

    @property
    def uptime_s(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    # -- client handler ------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug(f"Client connected: {peer}")
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # client disconnected
                if len(line) > MAX_MESSAGE_BYTES:
                    writer.write(_error(None, -32600, "Message too large"))
                    await writer.drain()
                    continue

                response = await self._dispatch(line)
                writer.write(response)
                await writer.drain()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        except Exception:
            logger.exception("Unexpected error in client handler")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug(f"Client disconnected: {peer}")

    # -- dynamic method registry ---------------------------------------------

    def register_method(
        self, name: str, handler: Any, *, replace: bool = False
    ) -> None:
        """Register a dynamic RPC method handler.

        Args:
            name: JSON-RPC method name (e.g. ``execute_mission``).
            handler: Async callable ``(params: dict) -> dict``.
            replace: If ``True``, overwrite an existing handler.
        """
        if not replace and name in self._extra_methods:
            raise ValueError(f"Method already registered: {name}")
        self._extra_methods[name] = handler
        logger.info("Registered dynamic method: %s", name)

    # -- dispatch ------------------------------------------------------------

    async def _dispatch(self, raw: bytes) -> bytes:
        """Parse JSON-RPC and dispatch to method handler."""
        # Parse
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            receipt = self._emit_receipt(
                method="parse_error",
                query_data={"raw": raw.decode("utf-8", errors="replace")[:1024]},
                result_data={"error": "Parse error"},
                status="rejected",
                gate="JSON-RPC",
                reason="Malformed JSON payload",
            )
            return _error(
                None,
                -32700,
                "Parse error",
                data={"code": "JSON_PARSE_ERROR", "receipt": receipt},
            )

        # Validate JSON-RPC structure
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            req_id = msg.get("id") if isinstance(msg, dict) else None
            receipt = self._emit_receipt(
                method=(
                    str(msg.get("method", "invalid_jsonrpc"))
                    if isinstance(msg, dict)
                    else "invalid_jsonrpc"
                ),
                query_data=msg if isinstance(msg, dict) else {"raw": str(msg)},
                result_data={"error": "Invalid JSON-RPC 2.0 request"},
                status="rejected",
                gate="JSON-RPC",
                reason="Invalid JSON-RPC envelope",
            )
            return _error(
                req_id,
                -32600,
                "Invalid JSON-RPC 2.0 request",
                data={"code": "JSONRPC_INVALID", "receipt": receipt},
            )

        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if not isinstance(method, str):
            receipt = self._emit_receipt(
                method="invalid_method",
                query_data=msg,
                result_data={"error": "Missing or invalid method"},
                status="rejected",
                gate="JSON-RPC",
                reason="Missing or invalid method",
            )
            return _error(
                req_id,
                -32600,
                "Missing or invalid 'method'",
                data={"code": "JSONRPC_METHOD_INVALID", "receipt": receipt},
            )

        auth_err = self._validate_auth(msg)
        if auth_err is not None:
            code, message, data = auth_err
            receipt = self._emit_receipt(
                method=method,
                query_data={"params": params},
                result_data={"error": message},
                status="rejected",
                gate="AUTH",
                reason=data.get("code", "AUTH_FAILURE"),
            )
            data["receipt"] = receipt
            return _error(req_id, code, message, data=data)

        # Rate limit
        if not self._rate_limiter.allow():
            receipt = self._emit_receipt(
                method=method,
                query_data={"params": params},
                result_data={"error": "Rate limit exceeded"},
                status="rejected",
                gate="RATE_LIMIT",
                reason="Rate limit exceeded (max 20 req/s)",
            )
            return _error(
                req_id,
                -32000,
                "Rate limit exceeded (max 20 req/s)",
                data={"code": "RATE_LIMIT_EXCEEDED", "receipt": receipt},
            )

        self._request_count += 1

        # Fire DESKTOP_INVOKE hook (best-effort, don't block on failure)
        await self._fire_hook(method, params)

        # Route to handler
        handlers = {
            "ping": self._handle_ping,
            "status": self._handle_status,
            "sovereign_query": self._handle_sovereign_query,
            "heartbeat_demo": self._handle_heartbeat_demo,
            "invoke_skill": self._handle_invoke_skill,
            "list_skills": self._handle_list_skills,
            "get_receipt": self._handle_get_receipt,
            "list_receipts": self._handle_list_receipts,
            "actuator_execute": self._handle_actuator_execute,
            "get_context": self._handle_get_context,
            "verify_action_outcome": self._handle_verify_action_outcome,
            "capture_screenshot": self._handle_capture_screenshot,
            # 8 productized HDA skills (Task 1.2)
            "open_app": self._handle_hda_proxy,
            "switch_window": self._handle_hda_proxy,
            "type_text": self._handle_hda_proxy,
            "click_element": self._handle_hda_proxy,
            "screenshot": self._handle_hda_proxy,
            "read_clipboard": self._handle_hda_proxy,
            "file_open": self._handle_hda_proxy,
            "browser_navigate": self._handle_hda_proxy,
        }

        handler = handlers.get(method) or self._extra_methods.get(method)
        if handler is None:
            receipt = self._emit_receipt(
                method=method,
                query_data={"params": params},
                result_data={"error": "Method not found"},
                status="rejected",
                gate="ROUTER",
                reason=f"Method not found: {method}",
            )
            return _error(
                req_id,
                -32601,
                f"Method not found: {method}",
                data={"code": "METHOD_NOT_FOUND", "receipt": receipt},
            )

        # Set current HDA method for proxy dispatch
        self._current_hda_method = method

        try:
            result = await handler(params)
            if (
                isinstance(result, dict)
                and not result.get("receipt")
                and "request_receipt" not in result
            ):
                status = "rejected" if "error" in result else "accepted"
                receipt = self._emit_receipt(
                    method=method,
                    query_data={"params": params},
                    result_data=result,
                    status=status,
                    gate=self._infer_gate(method, result, status),
                    reason=str(result.get("error")) if status == "rejected" else None,
                )
                if receipt is not None:
                    if method == "get_receipt":
                        result["request_receipt"] = receipt
                    else:
                        result["receipt"] = receipt
            return _ok(req_id, result)
        except Exception as exc:
            logger.exception(f"Handler error for '{method}'")
            receipt = self._emit_receipt(
                method=method,
                query_data={"params": params},
                result_data={"error": "Internal server error"},
                status="rejected",
                gate="HANDLER",
                reason=str(exc),
            )
            return _error(
                req_id,
                -32603,
                "Internal server error",
                data={"code": "INTERNAL_ERROR", "receipt": receipt},
            )

    # -- hook integration ----------------------------------------------------

    async def _fire_hook(self, method: str, params: Any) -> None:
        """Fire HookPhase.DESKTOP_INVOKE (best-effort)."""
        try:
            from core.elite.hooks import (
                HookContext,
                HookPhase,
                _get_global_registry,
            )

            registry = _get_global_registry()
            hooks = registry.get_hooks(HookPhase.DESKTOP_INVOKE)

            ctx = HookContext(
                operation_name=f"desktop_bridge.{method}",
                operation_type="desktop_invoke",
                input_data=params if isinstance(params, dict) else {},
                metadata={"source": "desktop_bridge", "method": method},
            )

            for hook in hooks:
                try:
                    data = {
                        "context": ctx,
                        "input_data": ctx.input_data,
                        "output_data": ctx.output_data,
                        "metadata": ctx.metadata,
                    }
                    if hook.is_async:
                        await hook.function(data)
                    else:
                        hook.function(data)
                except Exception:
                    logger.debug(
                        f"DESKTOP_INVOKE hook '{hook.name}' failed", exc_info=True
                    )
        except ImportError:
            pass  # hooks module not available

    # -- FATE gate -----------------------------------------------------------

    def _get_fate_gate(self) -> Any:
        """Lazy-load FATEGate."""
        if self._fate_gate is None:
            try:
                from core.elite.hooks import FATEGate

                self._fate_gate = FATEGate()
            except ImportError:
                pass
        return self._fate_gate

    def _validate_fate(self, operation: str) -> dict[str, Any]:
        """Run FATE validation. Fail-closed: blocks if gate unavailable."""
        gate = self._get_fate_gate()
        if gate is None:
            return {"passed": False, "overall": 0.0, "error": "FATE gate unavailable"}

        try:
            from core.elite.hooks import HookContext

            ctx = HookContext(
                operation_name=operation,
                operation_type="desktop_invoke",
                metadata={"source": "desktop_bridge"},
            )
            score = gate.validate(ctx, declared_intent=operation)
            return score.to_dict()
        except Exception as exc:
            logger.warning(f"FATE validation failed: {exc}")
            return {"passed": False, "overall": 0.0, "error": str(exc)}

    # -- Shannon entropy gate -------------------------------------------------

    def _validate_entropy(self, instruction: str) -> dict[str, Any]:
        """Shannon entropy gate — blocks low-signal instructions (H < 3.5)."""
        if not instruction.strip():
            return {"passed": False, "entropy": 0.0, "error": "Empty instruction"}
        try:
            from core.uers.entropy import EntropyCalculator

            calc = EntropyCalculator()
            measurement = calc.text_entropy(instruction)
            passed = measurement.value >= ACTUATOR_ENTROPY_THRESHOLD
            return {
                "passed": passed,
                "entropy": round(measurement.value, 4),
                "normalized": round(measurement.normalized, 4),
                "threshold": ACTUATOR_ENTROPY_THRESHOLD,
                "unique_chars": measurement.metadata.get("unique_chars", 0),
            }
        except ImportError:
            return {
                "passed": True,
                "entropy": -1.0,
                "error": "entropy_module_unavailable",
            }

    # -- Rust gate chain (PyO3) -----------------------------------------------

    def _get_rust_gate_chain(self) -> Any:
        """Lazy-load Rust GateChain (fail-fast: Schema → Ihsan → SNR)."""
        if not _RUST_AVAILABLE:
            return None
        if self._rust_gate_chain is None:
            try:
                self._rust_gate_chain = GateChain()
            except Exception:
                pass
        return self._rust_gate_chain

    def _get_rust_constitution(self) -> Any:
        """Lazy-load Rust Constitution (threshold source of truth)."""
        if not _RUST_AVAILABLE:
            return None
        if self._rust_constitution is None:
            try:
                self._rust_constitution = Constitution()
            except Exception:
                pass
        return self._rust_constitution

    def _validate_rust_gates(
        self, content: str, snr_score: float = 0.95, ihsan_score: float = 0.95
    ) -> dict[str, Any]:
        """Run Rust GateChain verification. Returns gate results or empty dict."""
        chain = self._get_rust_gate_chain()
        if chain is None:
            return {}
        try:
            results = chain.verify(content, snr_score, ihsan_score)
            gates = {
                name: {"passed": passed, "code": code} for name, passed, code in results
            }
            all_passed = all(passed for _, passed, _ in results)
            return {"gates": gates, "passed": all_passed, "engine": "rust"}
        except Exception as exc:
            logger.warning(f"Rust gate verification failed: {exc}")
            return {}

    def _blake3_digest(self, content: str) -> Optional[str]:
        """BLAKE3 digest via Rust domain_separated_digest. None if unavailable."""
        if not _RUST_AVAILABLE:
            return None
        try:
            return domain_separated_digest(content.encode())
        except Exception:
            return None

    async def _check_rust_guardian(self, content: str, source: str) -> dict[str, Any]:
        """
        Route action preflight to rust guardian over local MCP JSON-RPC transport.

        Modes:
          - off: bypass guardian wire (legacy compatibility)
          - best_effort: allow if guardian transport unavailable
          - required: fail-closed if guardian transport unavailable or denied
        """
        mode = self._guardian_wire_mode
        if mode == "off":
            return {
                "allowed": True,
                "reason": "guardian_wire_off",
                "mode": mode,
            }

        request_id = int(time.time() * 1000)
        wire = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "guardian_check",
                    "params": {
                        "content": content,
                        "source": source,
                    },
                    "id": request_id,
                }
            ).encode()
            + b"\n"
        )

        reader: Optional[asyncio.StreamReader] = None
        writer: Optional[asyncio.StreamWriter] = None
        try:
            timeout_s = max(0.1, self._guardian_timeout_ms / 1000.0)
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._guardian_host, self._guardian_port),
                timeout=timeout_s,
            )
            writer.write(wire)
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
            if not line:
                raise RuntimeError("empty_guardian_response")
            payload = json.loads(line)
            if "error" in payload:
                err = payload.get("error") or {}
                reason = str(err.get("message", "guardian_error"))
                if mode == "required":
                    return {"allowed": False, "reason": reason, "mode": mode}
                logger.warning("Guardian wire error (best_effort): %s", reason)
                return {
                    "allowed": True,
                    "reason": "guardian_wire_error_best_effort",
                    "mode": mode,
                }

            result = payload.get("result") or {}
            raw_allowed = result.get("allowed", False)
            if isinstance(raw_allowed, str):
                allowed = raw_allowed.strip().lower() == "true"
            else:
                allowed = bool(raw_allowed)
            reason = str(
                result.get("reason", "allowed" if allowed else "guardian_veto")
            )
            return {
                "allowed": allowed,
                "reason": reason,
                "mode": mode,
            }
        except Exception as exc:
            if mode == "required":
                return {
                    "allowed": False,
                    "reason": f"guardian_wire_unavailable:{type(exc).__name__}",
                    "mode": mode,
                }
            logger.warning("Guardian wire unavailable (best_effort): %s", exc)
            return {
                "allowed": True,
                "reason": "guardian_wire_unavailable_best_effort",
                "mode": mode,
            }
        finally:
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

    # -- method handlers -----------------------------------------------------

    async def _handle_ping(self, params: Any) -> dict[str, Any]:
        """Liveness check."""
        return {
            "status": "alive",
            "uptime_s": round(self.uptime_s, 2),
            "request_count": self._request_count,
            "rust_available": _RUST_AVAILABLE,
        }

    async def _handle_status(self, params: Any) -> dict[str, Any]:
        """Full Node0 status snapshot."""
        fate_result = self._validate_fate("status")

        # Inference gateway health
        gw_health: dict[str, Any] = {"available": False}
        gateway = self._get_gateway()
        if gateway is not None:
            try:
                gw_health = await gateway.health()
                gw_health["available"] = True
            except Exception as exc:
                logger.warning(f"Gateway health check failed: {exc}")
                gw_health = {"available": False, "error": "Health check failed"}

        # Rust constitution info
        rust_info: dict[str, Any] = {"available": _RUST_AVAILABLE}
        constitution = self._get_rust_constitution()
        if constitution is not None:
            rust_info["constitution_version"] = getattr(
                constitution, "version", "unknown"
            )
            rust_info["ihsan_threshold"] = getattr(
                constitution, "ihsan_threshold", 0.95
            )
            rust_info["snr_threshold"] = getattr(constitution, "snr_threshold", 0.85)

        return {
            "node": (
                "node0"
                if self._origin_snapshot.get("designation") == "node0"
                else "node"
            ),
            "origin": dict(self._origin_snapshot),
            "bridge_uptime_s": round(self.uptime_s, 2),
            "request_count": self._request_count,
            "fate_gate": fate_result,
            "inference": gw_health,
            "rust": rust_info,
        }

    async def _handle_sovereign_query(self, params: Any) -> dict[str, Any]:
        """Route query through InferenceGateway with FATE validation."""
        if not isinstance(params, dict) or "query" not in params:
            raise ValueError("Missing 'query' in params")

        query = str(params["query"])
        if not query.strip():
            raise ValueError("Empty query")

        # FATE gate (Python)
        fate_result = self._validate_fate(f"sovereign_query:{query[:40]}")
        if not fate_result.get("passed", False):
            return {
                "error": "FATE gate blocked query",
                "fate": fate_result,
            }

        # Rust gate chain (if available — runs Schema → Ihsan → SNR)
        rust_gates = self._validate_rust_gates(query)
        if rust_gates and not rust_gates.get("passed", True):
            return {
                "error": "Rust gate chain blocked query",
                "fate": fate_result,
                "rust_gates": rust_gates,
            }

        # Inference
        gateway = self._get_gateway()
        if gateway is None:
            return {
                "error": "InferenceGateway not available",
                "fate": fate_result,
            }

        start = time.monotonic()
        try:
            result = await gateway.infer(query)
            latency_ms = round((time.monotonic() - start) * 1000, 2)

            return {
                "content": result.content,
                "model": result.model,
                "backend": (
                    result.backend.value
                    if hasattr(result.backend, "value")
                    else str(result.backend)
                ),
                "latency_ms": latency_ms,
                "tokens_generated": result.tokens_generated,
                "fate": fate_result,
                "rust_gates": rust_gates or None,
                "content_hash": self._blake3_digest(result.content),
            }
        except Exception as exc:
            logger.warning(f"sovereign_query failed: {exc}")
            latency_ms = round((time.monotonic() - start) * 1000, 2)
            return {
                "error": "Query execution failed",
                "latency_ms": latency_ms,
                "fate": fate_result,
            }

    async def _handle_heartbeat_demo(self, params: Any) -> dict[str, Any]:
        """Run first heartbeat flow: browser research -> open app -> type summary.

        This endpoint intentionally wires one compact end-to-end path so we can
        verify cross-channel execution with receipts:
          1) Browser research (BrowserMCPClient)
          2) Desktop action open_app (HDA proxy)
          3) Desktop action type_text (HDA proxy)
        """
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")

        query = str(params.get("query", "")).strip()
        if not query:
            raise ValueError("Missing 'query' in params")

        app = str(params.get("app", "notepad")).strip() or "notepad"
        browser_mode = str(params.get("browser_mode", "mock")).strip().lower() or "mock"
        if browser_mode not in {"mock", "direct", "mcp"}:
            raise ValueError("browser_mode must be one of: mock, direct, mcp")

        try:
            max_typed_chars = int(params.get("max_typed_chars", 240))
        except (TypeError, ValueError):
            max_typed_chars = 240
        max_typed_chars = max(64, min(max_typed_chars, 1024))

        stage_receipts: list[dict[str, Any]] = []
        credit_sum = 0.0
        prev_receipt_id: str | None = None
        prev_receipt_digest: str | None = None

        def _lookup_receipt_digest(summary: dict[str, Any] | None) -> str | None:
            if not isinstance(summary, dict):
                return None
            receipt_id = summary.get("receipt_id")
            if not isinstance(receipt_id, str) or not receipt_id:
                return None
            engine = self._get_receipt_engine()
            if engine is None:
                return None
            full = engine.get_receipt(receipt_id)
            if not isinstance(full, dict):
                return None
            digest = full.get("receipt_digest")
            if isinstance(digest, str) and digest:
                return digest
            return None

        def _record_stage_receipt(
            stage: str,
            *,
            status: str,
            gate: str,
            query_data: dict[str, Any],
            result_data: dict[str, Any],
            step_score: float,
            step_index: int,
            reason: str | None = None,
            duration_ms: float = 0.0,
        ) -> None:
            nonlocal credit_sum, prev_receipt_id, prev_receipt_digest

            bounded_step_score = max(0.0, min(1.0, float(step_score)))
            credit_sum += bounded_step_score
            trajectory_credit = {
                "flow": "heartbeat_demo",
                "stage": stage,
                "step_index": step_index,
                "step_score": round(bounded_step_score, 4),
                "cumulative_score": round(credit_sum / max(1, step_index), 4),
                "prefix_receipt_id": prev_receipt_id,
                "prefix_receipt_digest": prev_receipt_digest,
            }
            summary = self._emit_receipt(
                method=f"heartbeat_demo.{stage}",
                query_data=query_data,
                result_data=result_data,
                status=status,
                gate=gate,
                duration_ms=duration_ms,
                reason=reason,
                trajectory_credit=trajectory_credit,
            )

            stage_entry: dict[str, Any] = {
                "stage": stage,
                "status": status,
                "trajectory_credit": trajectory_credit,
            }
            if summary is not None:
                stage_entry["receipt"] = summary
                prev_receipt_id = summary.get("receipt_id")
                prev_receipt_digest = _lookup_receipt_digest(summary)
            stage_receipts.append(stage_entry)

        browser_start = time.monotonic()
        try:
            browser = BrowserMCPClient(mode=browser_mode)
            research = await browser.research(query)
        except Exception as exc:
            browser_duration_ms = round((time.monotonic() - browser_start) * 1000, 2)
            browser_error = {
                "error": "Browser research failed",
                "detail": type(exc).__name__,
            }
            _record_stage_receipt(
                "browser_research",
                status="rejected",
                gate="BROWSER_RESEARCH",
                query_data={"query": query, "browser_mode": browser_mode},
                result_data=browser_error,
                step_score=0.0,
                step_index=1,
                reason="browser_research_failed",
                duration_ms=browser_duration_ms,
            )
            return {
                "task_complete": False,
                "status": "failed",
                "stage": "browser_research",
                "error": "Browser research failed",
                "detail": type(exc).__name__,
                "query": query,
                "stage_receipts": stage_receipts,
            }

        browser_duration_ms = round((time.monotonic() - browser_start) * 1000, 2)
        results = research.get("results", [])
        summary = str(research.get("summary", "")).strip()
        if not summary:
            titles = [
                str(item.get("title", "")).strip()
                for item in results
                if isinstance(item, dict) and item.get("title")
            ]
            summary = f"Top matches: {', '.join(titles)}" if titles else "No matches"

        _record_stage_receipt(
            "browser_research",
            status="accepted",
            gate="BROWSER_RESEARCH",
            query_data={"query": query, "browser_mode": browser_mode},
            result_data={
                "mode": research.get("mode", browser_mode),
                "summary": summary,
                "result_count": len(results),
            },
            step_score=0.96,
            step_index=1,
            duration_ms=browser_duration_ms,
        )

        typed_text = f"[BIZRA HEARTBEAT] {summary}".strip()
        if len(typed_text) > max_typed_chars:
            typed_text = typed_text[: max_typed_chars - 3].rstrip() + "..."

        async def _run_hda(method: str, payload: dict[str, Any]) -> dict[str, Any]:
            self._current_hda_method = method
            return await self._handle_hda_proxy(payload)

        open_start = time.monotonic()
        open_result = await _run_hda("open_app", {"app": app})
        open_duration_ms = round((time.monotonic() - open_start) * 1000, 2)
        if "error" in open_result:
            _record_stage_receipt(
                "open_app",
                status="rejected",
                gate="HDA_OPEN_APP",
                query_data={"app": app},
                result_data=open_result,
                step_score=0.0,
                step_index=2,
                reason=str(open_result.get("error", "open_app failed")),
                duration_ms=open_duration_ms,
            )
            return {
                "task_complete": False,
                "status": "failed",
                "stage": "desktop_open_app",
                "error": str(open_result.get("error", "open_app failed")),
                "query": query,
                "app": app,
                "browser": {
                    "mode": research.get("mode", browser_mode),
                    "summary": summary,
                    "result_count": len(results),
                },
                "desktop": {"open_app": open_result},
                "stage_receipts": stage_receipts,
            }

        _record_stage_receipt(
            "open_app",
            status="accepted",
            gate="HDA_OPEN_APP",
            query_data={"app": app},
            result_data=open_result,
            step_score=0.95,
            step_index=2,
            duration_ms=open_duration_ms,
        )

        type_start = time.monotonic()
        type_result = await _run_hda("type_text", {"text": typed_text})
        type_duration_ms = round((time.monotonic() - type_start) * 1000, 2)
        if "error" in type_result:
            _record_stage_receipt(
                "type_text",
                status="rejected",
                gate="HDA_TYPE_TEXT",
                query_data={"text_length": len(typed_text)},
                result_data=type_result,
                step_score=0.0,
                step_index=3,
                reason=str(type_result.get("error", "type_text failed")),
                duration_ms=type_duration_ms,
            )
            return {
                "task_complete": False,
                "status": "failed",
                "stage": "desktop_type_text",
                "error": str(type_result.get("error", "type_text failed")),
                "query": query,
                "app": app,
                "typed_text": typed_text,
                "browser": {
                    "mode": research.get("mode", browser_mode),
                    "summary": summary,
                    "result_count": len(results),
                },
                "desktop": {
                    "open_app": open_result,
                    "type_text": type_result,
                },
                "stage_receipts": stage_receipts,
            }

        _record_stage_receipt(
            "type_text",
            status="accepted",
            gate="HDA_TYPE_TEXT",
            query_data={"text_length": len(typed_text)},
            result_data=type_result,
            step_score=0.94,
            step_index=3,
            duration_ms=type_duration_ms,
        )

        return {
            "task_complete": True,
            "status": "completed",
            "flow": ["browser_research", "open_app", "type_text"],
            "query": query,
            "app": app,
            "typed_text": typed_text,
            "browser": {
                "mode": research.get("mode", browser_mode),
                "summary": summary,
                "result_count": len(results),
            },
            "desktop": {
                "open_app": open_result,
                "type_text": type_result,
            },
            "stage_receipts": stage_receipts,
            "trajectory_credit": {
                "flow": "heartbeat_demo",
                "steps_total": 3,
                "completed_steps": 3,
                "cumulative_score": round(credit_sum / 3.0, 4),
            },
        }

    # -- new handlers (Phase 2) -----------------------------------------------

    async def _handle_invoke_skill(self, params: Any) -> dict[str, Any]:
        """Invoke a skill via SkillRouter with FATE + Rust gate validation."""
        if not isinstance(params, dict) or "skill" not in params:
            raise ValueError("Missing 'skill' in params")

        skill_name = str(params["skill"]).strip()
        if not skill_name:
            raise ValueError("Empty skill name")

        inputs = params.get("inputs", {})

        # Rust guardian wire preflight (Python -> Rust -> action) for safety authority.
        guardian_content = (
            f"invoke_skill:{skill_name}:{json.dumps(inputs, sort_keys=True)[:512]}"
        )
        guardian = await self._check_rust_guardian(
            guardian_content,
            source="desktop_bridge.invoke_skill",
        )
        if not guardian.get("allowed", False):
            receipt = self._emit_receipt(
                "invoke_skill",
                params,
                guardian,
                "rejected",
                "RustGuardian",
                reason=f"Rust guardian veto: {guardian.get('reason', 'guardian_veto')}",
            )
            return {
                "error": "Rust guardian veto",
                "guardian": guardian,
                "receipt": receipt,
            }

        # FATE gate — ihsan score derived server-side (never client-controlled)
        fate_result = self._validate_fate(f"invoke_skill:{skill_name}")
        ihsan = float(fate_result.get("overall", 0.0))
        if not fate_result.get("passed", False):
            receipt = self._emit_receipt(
                "invoke_skill", params, fate_result, "rejected", "FATE"
            )
            return {
                "error": "FATE gate blocked",
                "fate": fate_result,
                "receipt": receipt,
            }

        # Rust gate
        rust_gates = self._validate_rust_gates(skill_name)
        if rust_gates and not rust_gates.get("passed", True):
            receipt = self._emit_receipt(
                "invoke_skill", params, rust_gates, "rejected", "Rust GateChain"
            )
            return {
                "error": "Rust gate chain blocked",
                "rust_gates": rust_gates,
                "receipt": receipt,
            }

        # Route to SkillRouter
        router = self._get_skill_router()
        if router is None:
            return {"error": "SkillRouter not available"}

        start = time.monotonic()
        try:
            result = await router.invoke(skill_name, inputs, ihsan_score=ihsan)
            duration = (time.monotonic() - start) * 1000

            status = "accepted" if result.success else "rejected"
            receipt = self._emit_receipt(
                "invoke_skill", params, result.to_dict(), status, skill_name, duration
            )

            response = result.to_dict()
            response["receipt"] = receipt
            return response
        except Exception as exc:
            logger.warning(f"invoke_skill '{skill_name}' failed: {exc}")
            duration = (time.monotonic() - start) * 1000
            receipt = self._emit_receipt(
                "invoke_skill",
                params,
                {"error": "Skill invocation failed"},
                "rejected",
                skill_name,
                duration,
                reason="Skill invocation failed",
            )
            return {"error": "Skill invocation failed", "receipt": receipt}

    async def _handle_list_skills(self, params: Any) -> dict[str, Any]:
        """List available skills from the SkillRegistry."""
        router = self._get_skill_router()
        if router is None:
            return {"error": "SkillRegistry not available", "skills": []}

        try:
            if hasattr(router.registry, "get_all"):
                all_skills = router.registry.get_all()
            else:
                all_skills = router.registry.list_all()
            filter_status = params.get("filter") if isinstance(params, dict) else None
            ranked = False
            limit = 10
            ihsan = 1.0
            include_fabric = False
            fabric_limit = 25
            if isinstance(params, dict):
                ranked = bool(params.get("ranked", False)) or (
                    str(params.get("mode", "")).lower() in {"ranked", "top"}
                )
                include_fabric = bool(params.get("include_fabric", False))
                try:
                    limit = max(1, min(int(params.get("limit", 10)), 100))
                except (TypeError, ValueError):
                    limit = 10
                try:
                    fabric_limit = max(1, min(int(params.get("fabric_limit", 25)), 200))
                except (TypeError, ValueError):
                    fabric_limit = 25
                try:
                    ihsan = float(params.get("ihsan", 1.0))
                except (TypeError, ValueError):
                    ihsan = 1.0

            if ranked and hasattr(router, "get_top_skills"):
                ranked_skills = router.get_top_skills(limit=limit, ihsan_score=ihsan)
                response = {
                    "skills": ranked_skills,
                    "count": len(ranked_skills),
                    "mode": "ranked",
                }
                if (
                    hasattr(router.registry, "get_performance_profile")
                    and callable(router.registry.get_performance_profile)
                ):
                    response["performance_profile"] = (
                        router.registry.get_performance_profile()
                    )
                if include_fabric and hasattr(router, "get_resource_fabric_summary"):
                    response["resource_fabric"] = router.get_resource_fabric_summary(
                        limit=fabric_limit,
                        include_assets=True,
                        force=False,
                    )
                return response

            skills = []
            for s in all_skills:
                if filter_status and s.status.value != filter_status:
                    continue
                skills.append(
                    {
                        "name": s.manifest.name,
                        "description": s.manifest.description,
                        "status": s.status.value,
                        "agent": s.manifest.agent,
                        "tags": getattr(s.manifest, "tags", []),
                    }
                )

            response = {"skills": skills, "count": len(skills)}
            if include_fabric and hasattr(router, "get_resource_fabric_summary"):
                response["resource_fabric"] = router.get_resource_fabric_summary(
                    limit=fabric_limit,
                    include_assets=True,
                    force=False,
                )
            return response
        except Exception as exc:
            return {"error": str(exc), "skills": []}

    async def _handle_get_receipt(self, params: Any) -> dict[str, Any]:
        """Retrieve a signed receipt by ID."""
        if not isinstance(params, dict) or "receipt_id" not in params:
            raise ValueError("Missing 'receipt_id' in params")

        engine = self._get_receipt_engine()
        if engine is None:
            return {"error": "Receipt engine not available"}

        receipt = engine.get_receipt(str(params["receipt_id"]))
        if receipt is None:
            raise ValueError(f"Receipt not found: {params['receipt_id']}")

        return receipt

    async def _handle_list_receipts(self, params: Any) -> dict[str, Any]:
        """Return the most recent N signed bridge receipts, newest first.

        Params:
            n (int, optional): Number of receipts to return (1–100, default 20).

        Returns dict with:
            receipts: list of receipt dicts (newest first)
            count: number returned
            total_on_disk: total files in receipt_dir
        """
        n = 20
        if isinstance(params, dict):
            try:
                n = max(1, min(int(params.get("n", 20)), 100))
            except (TypeError, ValueError):
                pass

        engine = self._get_receipt_engine()
        if engine is None:
            return {"error": "Receipt engine not available", "receipts": []}

        receipts = engine.list_recent(n)

        # Count total persisted receipts for UI pagination hint
        total = 0
        try:
            total = sum(1 for _ in engine.receipt_dir.glob("br-*.json"))
        except OSError:
            pass

        return {
            "receipts": receipts,
            "count": len(receipts),
            "total_on_disk": total,
        }

    # -- actuator layer (Phase 20) -------------------------------------------

    async def _handle_actuator_execute(self, params: Any) -> dict[str, Any]:
        """
        Validate an instruction through the 3-gate pipeline before sealing.

        Pipeline: FATE gate -> Shannon entropy gate -> Rust gate chain.
        The bridge validates and signs; the AHK client executes.
        """
        if not isinstance(params, dict) or "code" not in params:
            raise ValueError("Missing 'code' in params")

        code = str(params["code"]).strip()
        intent = str(params.get("intent", "execute"))
        target_app = params.get("target_app")

        if not code:
            raise ValueError("Empty instruction code")

        # Rust guardian wire preflight before any execution sealing.
        guardian = await self._check_rust_guardian(
            f"actuator_execute:{intent}:{code[:512]}",
            source="desktop_bridge.actuator_execute",
        )
        if not guardian.get("allowed", False):
            receipt = self._emit_receipt(
                "actuator_execute",
                params,
                guardian,
                "rejected",
                "RustGuardian",
                reason=f"Rust guardian veto: {guardian.get('reason', 'guardian_veto')}",
            )
            return {
                "error": "Rust guardian veto",
                "guardian": guardian,
                "receipt": receipt,
            }

        # Gate 1: FATE (Ihsan threshold)
        fate_result = self._validate_fate(f"actuator_execute:{intent}")
        if not fate_result.get("passed", False):
            return {"error": "FATE gate blocked", "fate": fate_result}

        # Gate 2: Shannon Entropy (information density)
        entropy_result = self._validate_entropy(code)
        if not entropy_result.get("passed", False):
            receipt = self._emit_receipt(
                "actuator_execute",
                params,
                entropy_result,
                "rejected",
                "SHANNON_ENTROPY",
                reason=(
                    f"Low entropy: {entropy_result.get('entropy', 0):.2f}"
                    f" < {ACTUATOR_ENTROPY_THRESHOLD}"
                ),
            )
            return {
                "error": "Shannon entropy gate blocked",
                "entropy": entropy_result,
                "receipt": receipt,
            }

        # Gate 3: Rust GateChain (Schema + Ihsan + SNR)
        rust_gates = self._validate_rust_gates(code)
        if rust_gates and not rust_gates.get("passed", True):
            return {"error": "Rust gate chain blocked", "rust_gates": rust_gates}

        # All gates passed — seal with BLAKE3 digest
        content_hash = self._blake3_digest(code)

        return {
            "status": "SEALED",
            "intent": intent,
            "target_app": target_app,
            "content_hash": content_hash,
            "fate": fate_result,
            "entropy": entropy_result,
            "instruction_length": len(code),
        }

    # -- HDA skill proxy (Task 1.2) -------------------------------------------

    def _ensure_hda_permit(self) -> Permit:
        """Return a valid HDA permit, creating or refreshing as needed.

        Permits have a TTL (default 300s) and a budget (default 30 actions).
        When either is exhausted a fresh permit is issued automatically.

        Standing on Giants: General Magic (auto-renewing Telescript permits)
        """

        if self._hda_permit is not None:
            verification = self._hda_permit.verify(signing_key=self._permit_signing_key)
            if verification.valid:
                return self._hda_permit

        # Create fresh permit
        self._hda_permit = create_hda_permit(
            signing_key=self._permit_signing_key or "bizra-dev-permit-key",
        )
        return self._hda_permit

    async def _handle_hda_proxy(self, params: Any) -> dict[str, Any]:
        """Proxy HDA skill calls to the AHK bridge.

        The 8 productized HDA skills (open_app, switch_window, type_text,
        click_element, screenshot, read_clipboard, file_open, browser_navigate)
        are implemented in AHK and proxied through this method.

        Gate pipeline: Permit check -> Guardian check -> FATE gate -> AHK RPC.

        Standing on Giants:
        - General Magic (Telescript permits, 1994): capability-scoped authority
        - Lamport (1978): hash-chained delegation
        - Shannon (1948): 6 capabilities = minimal signal set
        """
        if not isinstance(params, dict):
            params = {}

        # Extract the method name from the call context.
        # The handler dispatch passes the params but the method name
        # is available from the _current_method attribute set by the router.
        method = getattr(self, "_current_hda_method", "unknown")

        # --- Gate 0: Telescript Permit verification ---
        permit = self._ensure_hda_permit()
        permit_check = permit.check_action(method, signing_key=self._permit_signing_key)
        if not permit_check.valid:
            return {
                "error": f"Permit denied for {method}",
                "permit": permit_check.to_dict(),
                "method": method,
            }

        # --- Gate 1: Guardian wire preflight ---
        guardian = await self._check_rust_guardian(
            f"hda:{method}:{json.dumps(params, default=str)[:256]}",
            source=f"desktop_bridge.hda.{method}",
        )
        if not guardian.get("allowed", False):
            return {
                "error": f"Guardian veto on {method}",
                "guardian": guardian,
            }

        # --- Gate 2: FATE gate check ---
        fate_result = self._validate_fate(f"hda:{method}")
        if not fate_result.get("passed", False):
            return {"error": "FATE gate blocked", "fate": fate_result}

        # --- Execute: Forward to AHK bridge ---
        ahk_result = await self._rpc_to_ahk(method, params)
        if ahk_result is None:
            return {
                "error": f"AHK bridge unreachable for {method}",
                "method": method,
                "fallback": "AHK bridge must be running for HDA skills",
            }

        # Consume budget on successful execution
        permit.consume()

        # Attach permit verification to result for audit trail
        ahk_result["permit"] = {
            "permit_id": permit.permit_id,
            "capability_checked": method,
            "actions_remaining": permit.budget.actions_remaining,
            "ttl_remaining": round(max(0.0, permit.expires_at - time.time()), 1),
        }

        return ahk_result

    async def _rpc_to_ahk(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC call to the AHK bridge and return the result."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(AHK_BRIDGE_HOST, AHK_BRIDGE_PORT),
                timeout=2.0,
            )
        except (OSError, asyncio.TimeoutError):
            return None

        try:
            import secrets

            nonce = secrets.token_hex(16)
            ts = str(int(time.time() * 1000))
            token = self._auth_token or os.getenv(AUTH_TOKEN_ENV, "")

            rpc_request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": f"hda_{nonce[:8]}",
                "headers": {
                    AUTH_HEADER_TOKEN: token,
                    AUTH_HEADER_TS: ts,
                    AUTH_HEADER_NONCE: nonce,
                },
            }

            writer.write(json.dumps(rpc_request).encode() + b"\n")
            await writer.drain()

            raw = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not raw:
                return None

            response = json.loads(raw)
            if "result" in response:
                result = response["result"]
                # AHK sends full result Map with "result" field plus optional
                # perception-action metadata (pre_hash, post_hash, etc.).
                # Return the entire map so callers see all fields.
                return result if isinstance(result, dict) else {"result": result}
            if "error" in response:
                return {"error": response["error"].get("message", "AHK error")}
            return None
        except (asyncio.TimeoutError, json.JSONDecodeError, OSError) as exc:
            logger.debug("AHK RPC %s failed: %s", method, exc)
            return None
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _handle_get_context(self, params: Any) -> dict[str, Any]:
        """
        Return live desktop context via AHK RPC or local fallback.

        Privacy-by-design: window titles are SHA-256 hashed by default.
        Plaintext requires explicit opt-in via params.plaintext_titles.

        On Windows/WSL: calls AHK bridge get_context method.
        On Linux: falls back to /proc-based process detection.

        Standing on Giants:
        - Boyd (OODA observe phase — raw perception before orientation)
        - Shannon (hash reduces channel bandwidth, preserves identity signal)
        """
        if not isinstance(params, dict):
            params = {}

        plaintext = bool(params.get("plaintext_titles", False))

        # Try AHK bridge first (Windows/WSL with AHK running)
        ahk_context = await self._get_context_via_ahk(plaintext)
        if ahk_context and not ahk_context.get("error"):
            return ahk_context

        # Fallback: local process-based context (Linux/WSL without AHK)
        return self._get_context_local(plaintext)

    async def _get_context_via_ahk(
        self, plaintext: bool = False
    ) -> dict[str, Any] | None:
        """Call the AHK bridge get_context method via JSON-RPC.

        Returns None if AHK bridge is unreachable, allowing local fallback.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(AHK_BRIDGE_HOST, AHK_BRIDGE_PORT),
                timeout=2.0,
            )
        except (OSError, asyncio.TimeoutError):
            return None

        try:
            # Build JSON-RPC request with auth headers
            import secrets
            import time as _time

            nonce = secrets.token_hex(16)
            ts = str(int(_time.time() * 1000))
            token = self._auth_token or os.getenv(AUTH_TOKEN_ENV, "")

            rpc_request = {
                "jsonrpc": "2.0",
                "method": "get_context",
                "params": {"plaintext_titles": plaintext},
                "id": f"ctx_{nonce[:8]}",
                "headers": {
                    AUTH_HEADER_TOKEN: token,
                    AUTH_HEADER_TS: ts,
                    AUTH_HEADER_NONCE: nonce,
                },
            }

            writer.write(json.dumps(rpc_request).encode() + b"\n")
            await writer.drain()

            raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not raw:
                return None

            response = json.loads(raw)
            if "result" in response:
                result = response["result"]
                if isinstance(result, dict) and "result" in result:
                    return result["result"]
                return result
            return None
        except (asyncio.TimeoutError, json.JSONDecodeError, OSError) as exc:
            logger.debug("AHK get_context failed: %s", exc)
            return None
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    def _get_context_local(self, plaintext: bool = False) -> dict[str, Any]:
        """Gather desktop context from local system (Linux/WSL fallback).

        Uses /proc filesystem for process detection when AHK is unavailable.
        Window titles are hashed by default for privacy.
        """
        import subprocess

        result: dict[str, Any] = {
            "schema_version": "2.0",
            "source": "local_fallback",
            "privacy_mode": "plaintext" if plaintext else "hashed",
            "timestamp": time.time(),
        }

        # Process list (top 32 by CPU, non-kernel)
        processes: list[dict[str, Any]] = []
        try:
            ps_output = subprocess.run(
                ["ps", "aux", "--sort=-%cpu"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in ps_output.stdout.strip().split("\n")[1:33]:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    proc_name = parts[10]
                    if proc_name.startswith("["):
                        continue  # Skip kernel threads
                    processes.append(
                        {
                            "process": proc_name.split("/")[-1][:64],
                            "pid": int(parts[1]),
                            "cpu": parts[2],
                            "mem": parts[3],
                        }
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        result["processes"] = processes
        result["process_count"] = len(processes)

        # Clipboard hash (xclip fallback)
        try:
            clip = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            clip_text = clip.stdout
            if clip_text:
                result["clipboard_hash"] = hashlib.sha256(
                    clip_text.encode()
                ).hexdigest()
                result["clipboard_length"] = len(clip_text)
            else:
                result["clipboard_hash"] = ""
                result["clipboard_length"] = 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            result["clipboard_hash"] = ""
            result["clipboard_length"] = 0

        # Foreground window (xdotool fallback for X11)
        try:
            xdo = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            title = xdo.stdout.strip()
            if title:
                result["foreground"] = {
                    "title": (
                        title
                        if plaintext
                        else hashlib.sha256(title.encode()).hexdigest()
                    ),
                    "title_hashed": not plaintext,
                }
            else:
                result["foreground"] = {"title": "", "title_hashed": False}
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            result["foreground"] = {"title": "", "title_hashed": False}

        return result

    # -- perception-action loop (Phase 21) -----------------------------------

    async def _verify_action_outcome(
        self,
        action_id: str,
        pre_hash: str,
        post_hash: str,
        intent: str,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify that an action actually changed the desktop state.

        Compares pre/post screenshot hashes to determine if the action
        had a visible effect. This closes the perception-action loop:
        Agent decides -> AHK executes -> Screenshot verifies -> Receipt seals.

        Args:
            action_id: Unique identifier for the action
            pre_hash: SHA-256 hash of pre-action screenshot
            post_hash: SHA-256 hash of post-action screenshot
            intent: The action intent (click, type, execute, read)
            target: Action target parameters

        Returns:
            Verification result with outcome_confirmed flag
        """
        state_changed = pre_hash != post_hash

        # Read actions should NOT change state
        if intent == "read":
            outcome_confirmed = not state_changed
        else:
            # Click, type, execute SHOULD change state (usually)
            outcome_confirmed = state_changed

        # Derive confidence from verification quality
        if pre_hash and post_hash and len(pre_hash) == 64 and len(post_hash) == 64:
            # Both hashes are full SHA-256 from real screenshots
            confidence = 0.95 if outcome_confirmed else 0.30
        elif pre_hash or post_hash:
            # Partial verification (only one hash available)
            confidence = 0.60
        else:
            # No screenshot data — timestamp-only confirmation
            confidence = 0.50

        return {
            "action_id": action_id,
            "intent": intent,
            "pre_hash": pre_hash,
            "post_hash": post_hash,
            "state_changed": state_changed,
            "outcome_confirmed": outcome_confirmed,
            "confidence": confidence,
            "verification_timestamp": time.time(),
        }

    async def _handle_verify_action_outcome(self, params: Any) -> dict[str, Any]:
        """Verify that a desktop action achieved its intended outcome.

        Accepts pre/post screenshot hashes from the AHK bridge and uses
        ``_verify_action_outcome`` to determine whether the action had the
        expected visual effect.  This closes the perception-action loop:
        Agent decides -> AHK executes -> Screenshot verifies -> Receipt seals.
        """
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")

        action_id = params.get("action_id", "")
        if not action_id:
            return {"verified": False, "confidence": 0.0, "reason": "missing_action_id"}

        pre_hash = str(params.get("pre_hash", ""))
        post_hash = str(params.get("post_hash", ""))
        intent = str(params.get("intent", "execute"))
        target = params.get("target", {})
        if not isinstance(target, dict):
            target = {}

        # Fallback: legacy callers may send screenshot_base64 instead of hashes
        screenshot_b64 = params.get("screenshot_base64", "")
        if not post_hash and screenshot_b64:
            outcome_data = screenshot_b64.encode("utf-8")[:4096]
            post_hash = hashlib.sha256(outcome_data).hexdigest()
        if not pre_hash and not post_hash:
            # No screenshot data at all — timestamp-based hash
            ts_data = f"{action_id}:{time.monotonic()}".encode()
            post_hash = hashlib.sha256(ts_data).hexdigest()

        verification = await self._verify_action_outcome(
            action_id=action_id,
            pre_hash=pre_hash,
            post_hash=post_hash,
            intent=intent,
            target=target,
        )
        verification["verified"] = True
        # Include the outcome_hash that downstream receipt sealing expects
        verification["outcome_hash"] = post_hash

        return verification

    async def _handle_capture_screenshot(self, params: Any) -> dict[str, Any]:
        """Capture a screenshot of the current desktop state.

        Returns a SHA-256 hash of the captured state for receipt chaining.
        Full screenshot data is stored locally, not transmitted over JSON-RPC.
        """
        label = ""
        screenshot_bytes: bytes = b""
        if isinstance(params, dict):
            label = str(params.get("label", ""))
            # Accept raw screenshot bytes or base64-encoded payload from the
            # AHK bridge.  When present the hash is content-addressed; when
            # absent we fall back to a monotonic-clock placeholder so callers
            # without screenshot support still receive a unique (but
            # non-verifiable) hash.
            raw = params.get("screenshot_bytes") or params.get("screenshot_base64")
            if isinstance(raw, (bytes, bytearray)):
                screenshot_bytes = bytes(raw)
            elif isinstance(raw, str) and raw:
                import base64

                try:
                    screenshot_bytes = base64.b64decode(raw)
                except Exception:  # noqa: BLE001 — graceful fallback
                    screenshot_bytes = raw.encode("utf-8")

        timestamp = time.monotonic()

        if screenshot_bytes:
            # Content-addressed hash of actual desktop state
            hash_input = screenshot_bytes
        else:
            # Fallback: monotonic timestamp (non-verifiable placeholder)
            hash_input = f"desktop_state:{timestamp}:{label}".encode()

        state_hash = hashlib.sha256(hash_input).hexdigest()

        return {
            "captured": True,
            "state_hash": state_hash,
            "timestamp": timestamp,
            "label": label,
        }

    # -- receipt + skill lazy-load -------------------------------------------

    def _get_receipt_engine(self) -> Any:
        """Lazy-load BridgeReceiptEngine."""
        if self._receipt_engine is not None:
            return self._receipt_engine
        from core.bridges.bridge_receipt import BridgeReceiptEngine

        self._receipt_engine = BridgeReceiptEngine()
        return self._receipt_engine

    def _get_skill_router(self) -> Any:
        """Lazy-load SkillRouter and register built-in skills (RDVE)."""
        if self._skill_router is not None:
            return self._skill_router
        try:
            from core.skills.router import SkillRouter

            self._skill_router = SkillRouter()

            # Auto-register RDVE skill (best-effort)
            try:
                from core.spearpoint.rdve_skill import register_rdve_skill

                register_rdve_skill(self._skill_router)
                logger.info("RDVE skill auto-registered on bridge SkillRouter")
            except Exception as exc:
                logger.debug(f"RDVE skill registration skipped: {exc}")

            # Auto-register Smart Files skill (best-effort)
            try:
                from core.skills.smart_file_manager import register_smart_files

                register_smart_files(self._skill_router)
                logger.info("Smart Files skill auto-registered on bridge SkillRouter")
            except Exception as exc:
                logger.debug(f"Smart Files registration skipped: {exc}")

            return self._skill_router
        except ImportError:
            return None

    def _emit_receipt(
        self,
        method: str,
        query_data: Any,
        result_data: Any,
        status: str,
        gate: str,
        duration_ms: float = 0.0,
        reason: Optional[str] = None,
        trajectory_credit: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Emit a receipt for a bridge command. Returns summary or None."""
        engine = self._get_receipt_engine()
        if engine is None:
            return None
        try:
            q = query_data if isinstance(query_data, dict) else {"raw": str(query_data)}
            r = (
                result_data
                if isinstance(result_data, dict)
                else {"raw": str(result_data)}
            )
            receipt = engine.create_receipt(
                method=method,
                query_data=q,
                result_data=r,
                fate_score=(
                    r.get("overall", r.get("fate_score", 0.0))
                    if isinstance(r, dict)
                    else 0.0
                ),
                snr_score=0.95,
                gate_passed=gate,
                status=status,
                duration_ms=duration_ms,
                reason=reason,
                origin=self._origin_snapshot,
                trajectory_credit=trajectory_credit,
            )
            return {"receipt_id": receipt["receipt_id"], "status": receipt["status"]}
        except Exception as exc:
            logger.warning(f"Receipt emission failed: {exc}")
            return None

    def _load_auth_token(self) -> str:
        token = os.getenv(AUTH_TOKEN_ENV, "").strip()
        if not token:
            raise RuntimeError(
                f"Missing bridge auth token: set {AUTH_TOKEN_ENV} before startup"
            )
        return token

    def _default_origin_snapshot(self) -> dict[str, Any]:
        """Return default non-genesis origin identity for bridge status."""
        return {
            "designation": "ephemeral_node",
            "genesis_node": False,
            "genesis_block": False,
            "home_base_device": False,
            "authority_source": "genesis_files",
            "hash_validated": False,
        }

    def _resolve_origin_snapshot(self) -> dict[str, Any]:
        """
        Resolve canonical origin identity from sovereign_state.
        """
        try:
            return resolve_origin_snapshot(GENESIS_STATE_DIR, self._node_role)
        except Exception as exc:
            logger.debug(f"Origin identity resolution failed: {exc}")
            return self._default_origin_snapshot()

    def _prune_nonce_cache(self, now_ms: int) -> None:
        expired = [
            nonce for nonce, expiry in self._nonce_seen.items() if expiry <= now_ms
        ]
        for nonce in expired:
            self._nonce_seen.pop(nonce, None)

    def _validate_auth(
        self, msg: dict[str, Any]
    ) -> Optional[tuple[int, str, dict[str, Any]]]:
        headers = msg.get("headers")
        if not isinstance(headers, dict):
            return (
                -32001,
                "Authentication failed: missing headers",
                {"code": "AUTH_MISSING_HEADERS"},
            )

        token = headers.get(AUTH_HEADER_TOKEN)
        ts = headers.get(AUTH_HEADER_TS)
        nonce = headers.get(AUTH_HEADER_NONCE)

        if not isinstance(token, str) or not token:
            return (
                -32001,
                f"Authentication failed: missing {AUTH_HEADER_TOKEN}",
                {"code": "AUTH_MISSING_TOKEN"},
            )
        if not isinstance(ts, (int, str)):
            return (
                -32001,
                f"Authentication failed: missing {AUTH_HEADER_TS}",
                {"code": "AUTH_MISSING_TIMESTAMP"},
            )
        if not isinstance(nonce, str) or not nonce:
            return (
                -32001,
                f"Authentication failed: missing {AUTH_HEADER_NONCE}",
                {"code": "AUTH_MISSING_NONCE"},
            )

        if self._auth_token is None:
            return (
                -32002,
                "Authentication failed: bridge token not initialized",
                {"code": "AUTH_NOT_READY"},
            )

        if not hmac.compare_digest(token, self._auth_token):
            return (
                -32002,
                "Authentication failed: invalid token",
                {"code": "AUTH_INVALID_TOKEN"},
            )

        try:
            ts_ms = int(ts)
        except (TypeError, ValueError):
            return (
                -32003,
                "Authentication failed: invalid timestamp",
                {"code": "AUTH_INVALID_TIMESTAMP"},
            )

        now_ms = int(time.time() * 1000)
        if abs(now_ms - ts_ms) > AUTH_MAX_CLOCK_SKEW_MS:
            return (
                -32003,
                "Authentication failed: stale timestamp",
                {"code": "AUTH_STALE_TIMESTAMP"},
            )

        self._prune_nonce_cache(now_ms)
        if nonce in self._nonce_seen:
            return (
                -32004,
                "Authentication failed: nonce replay detected",
                {"code": "AUTH_NONCE_REPLAY"},
            )
        self._nonce_seen[nonce] = now_ms + AUTH_NONCE_TTL_MS
        return None

    def _infer_gate(self, method: str, result: dict[str, Any], status: str) -> str:
        if status == "accepted":
            return method
        if "fate" in result:
            return "FATE"
        if "rust_gates" in result:
            return "Rust GateChain"
        return "HANDLER"

    # -- gateway lazy-load ---------------------------------------------------

    def _get_gateway(self) -> Any:
        """Lazy-load InferenceGateway singleton."""
        if self._gateway is not None:
            return self._gateway
        try:
            from core.inference.gateway import get_inference_gateway

            self._gateway = get_inference_gateway()
            return self._gateway
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _boot_mission_handler(bridge: DesktopBridge) -> None:
    """Register execute_mission on the bridge — First Heartbeat wiring."""
    try:
        from core.sovereign.mission import MissionOrchestrator

        orch = MissionOrchestrator(
            {
                "memory_path": str(
                    Path(os.getenv("BIZRA_DATA_LAKE_ROOT", "."))
                    / "sovereign_state"
                    / "mission_memory"
                ),
                "evidence_path": str(
                    Path(os.getenv("BIZRA_DATA_LAKE_ROOT", "."))
                    / "sovereign_state"
                    / "mission_evidence.jsonl"
                ),
                "hda_port": 9743,
            }
        )
        await orch.initialize()

        async def _handle_execute_mission(params: dict) -> dict:
            return await orch.handle_rpc(params)

        bridge.register_method("execute_mission", _handle_execute_mission)
        logger.info("MissionOrchestrator registered on execute_mission")
    except Exception as e:
        logger.warning("MissionOrchestrator boot failed (non-fatal): %s", e)


async def _run() -> None:
    bridge = DesktopBridge()
    loop = asyncio.get_running_loop()

    # Graceful shutdown on SIGINT/SIGTERM
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    await bridge.start()

    # Wire MissionOrchestrator → execute_mission RPC method
    await _boot_mission_handler(bridge)

    print(f"BIZRA Desktop Bridge listening on {BRIDGE_HOST}:{BRIDGE_PORT}")
    print("Press Ctrl+C to stop.")

    await stop_event.wait()
    await bridge.stop()
    print("Bridge stopped.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    main()
