"""
BIZRA Desktop Bridge -- Minimal Sovereign Command Surface

Asyncio TCP server exposing JSON-RPC methods over newline-delimited JSON.
Zero new dependencies -- stdlib only (asyncio, json, logging).

Protocol: Newline-delimited JSON-RPC 2.0 on 127.0.0.1:9742.
Commands: ping, status, sovereign_query.

Standing on Giants:
- Boyd (OODA): Fast feedback loop before scaling
- Shannon (SNR): Increase signal before adding channel bandwidth
- Lamport: Avoid new network surface unless necessary
- Al-Ghazali (Ihsan): Excellence over ego-driven expansion

Created: 2026-02-13 | BIZRA Desktop Bridge v1.0
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.sovereign.origin_guard import (
    NODE_ROLE_ENV,
    enforce_node0_fail_closed,
    normalize_node_role,
    resolve_origin_snapshot,
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

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the TCP server."""
        self._auth_token = self._load_auth_token()
        self._node_role = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
        enforce_node0_fail_closed(GENESIS_STATE_DIR, self._node_role)
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
            "invoke_skill": self._handle_invoke_skill,
            "list_skills": self._handle_list_skills,
            "get_receipt": self._handle_get_receipt,
            "actuator_execute": self._handle_actuator_execute,
            "get_context": self._handle_get_context,
        }

        handler = handlers.get(method)
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

    # -- new handlers (Phase 2) -----------------------------------------------

    async def _handle_invoke_skill(self, params: Any) -> dict[str, Any]:
        """Invoke a skill via SkillRouter with FATE + Rust gate validation."""
        if not isinstance(params, dict) or "skill" not in params:
            raise ValueError("Missing 'skill' in params")

        skill_name = str(params["skill"]).strip()
        if not skill_name:
            raise ValueError("Empty skill name")

        inputs = params.get("inputs", {})

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

            return {"skills": skills, "count": len(skills)}
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

    async def _handle_get_context(self, params: Any) -> dict[str, Any]:
        """
        Return the UIA context fingerprint schema.

        The actual structural data is extracted by the AHK client using
        UIA-v2 Element.DumpAll(). This method defines the expected schema.
        """
        return {
            "schema_version": "1.0",
            "expected_fields": [
                "title",
                "class",
                "process",
                "structure",
                "hash",
            ],
            "note": "Context data populated by AHK actuator client via UIA-v2",
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
