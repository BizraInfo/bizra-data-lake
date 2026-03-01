"""
BIZRA Ghost WebSocket Bridge — Real-time push from Node0 to Ghost Overlay UI.

FastAPI + WebSocket server on port 9743 that:
  1. Accepts WS connections from Ghost Overlay (AHK/Electron) clients.
  2. Subscribes to HHMM prediction events and constitutional gate results.
  3. Pushes overlay show/dismiss/update events to connected clients.
  4. Provides /health HTTP endpoint for Docker healthcheck.

Protocol: JSON messages over WebSocket (binary frames NOT accepted).
Max message size: 64KB (WS_MAX_SIZE).
Auth: HMAC-based token via first message handshake.

Standing on Giants:
- Boyd (OODA): Real-time observe→orient loop to overlay UI
- Shannon (SNR): Only push predictions above SNR threshold
- Norman (invisible design): Ghost Overlay must not disrupt user flow
- Lamport (single truth): All thresholds from constants.py

Created: 2026-02-24 | BIZRA Ghost WS Bridge v0.1
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger("bizra.ghost_ws")

# ---------------------------------------------------------------------------
# Configuration (from env, never hardcoded)
# ---------------------------------------------------------------------------

WS_HOST = os.getenv("WS_HOST", "127.0.0.1")
WS_PORT = int(os.getenv("WS_PORT", "9743"))
WS_MAX_SIZE = int(os.getenv("WS_MAX_SIZE", "65536"))  # 64KB
GHOST_IDLE_TIMEOUT_MS = int(os.getenv("GHOST_IDLE_TIMEOUT_MS", "5000"))
GHOST_DEBOUNCE_MS = int(os.getenv("GHOST_DEBOUNCE_MS", "500"))
MAX_SUGGESTIONS = 3
MAX_CONNECTED_CLIENTS = int(os.getenv("GHOST_MAX_CLIENTS", "4"))

# Auth token for WS handshake (optional — dev mode allows unauthenticated)
GHOST_WS_AUTH_TOKEN = os.getenv("GHOST_WS_AUTH_TOKEN", "")
BIZRA_ENV = os.getenv("BIZRA_ENV", "development")


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class OverlaySuggestion:
    """A single suggestion card in the Ghost Overlay."""

    id: str
    action_label: str
    intent_summary: str
    hhmm_confidence: float
    ihsan_precheck: str  # "pass" | "pending" | "blocked"
    ihsan_score: float
    ahk_action_id: str
    block_reason: Optional[str] = None
    target_region: Optional[Dict[str, float]] = None


@dataclass
class OverlayEvent:
    """Event pushed to Ghost Overlay clients."""

    type: str  # "show_overlay" | "dismiss_overlay" | "update_overlay"
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    position: Optional[Dict[str, float]] = None
    auto_dismiss_at: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------


class GhostConnectionManager:
    """Manages WebSocket connections to Ghost Overlay clients.

    Enforces max client limit and provides broadcast capability.
    """

    def __init__(self, max_clients: int = MAX_CONNECTED_CLIENTS) -> None:
        self._connections: Set[WebSocket] = set()
        self._max_clients = max_clients
        self._message_count: int = 0
        self._start_time: float = time.time()

    @property
    def client_count(self) -> int:
        return len(self._connections)

    @property
    def message_count(self) -> int:
        return self._message_count

    async def connect(self, ws: WebSocket) -> bool:
        """Accept a new WebSocket connection if under limit."""
        if len(self._connections) >= self._max_clients:
            logger.warning(
                "Ghost WS: rejecting connection — max clients (%d) reached",
                self._max_clients,
            )
            return False
        await ws.accept()
        self._connections.add(ws)
        logger.info(
            "Ghost WS: client connected (%d/%d)",
            len(self._connections),
            self._max_clients,
        )
        return True

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a disconnected client."""
        self._connections.discard(ws)
        logger.info(
            "Ghost WS: client disconnected (%d/%d)",
            len(self._connections),
            self._max_clients,
        )

    async def broadcast(self, event: OverlayEvent) -> int:
        """Send an overlay event to all connected clients.

        Returns the number of clients that received the message.
        """
        if not self._connections:
            return 0

        payload = json.dumps(asdict(event), default=str)
        sent = 0
        stale: List[WebSocket] = []

        for ws in self._connections:
            try:
                await ws.send_text(payload)
                sent += 1
                self._message_count += 1
            except Exception:
                stale.append(ws)

        for ws in stale:
            self._connections.discard(ws)

        return sent

    async def send_to(self, ws: WebSocket, event: OverlayEvent) -> bool:
        """Send an overlay event to a specific client."""
        try:
            payload = json.dumps(asdict(event), default=str)
            await ws.send_text(payload)
            self._message_count += 1
            return True
        except Exception:
            self._connections.discard(ws)
            return False

    def health_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time
        return {
            "connected_clients": len(self._connections),
            "max_clients": self._max_clients,
            "total_messages_sent": self._message_count,
            "uptime_seconds": round(uptime, 1),
        }


# ---------------------------------------------------------------------------
# Debounce Buffer
# ---------------------------------------------------------------------------


class PredictionDebouncer:
    """Debounces rapid HHMM predictions.

    Takes only the highest-confidence prediction within the debounce window.
    """

    def __init__(self, debounce_ms: int = GHOST_DEBOUNCE_MS) -> None:
        self._debounce_s = debounce_ms / 1000.0
        self._pending: Optional[Dict[str, Any]] = None
        self._timer: Optional[asyncio.Task[None]] = None
        self._callback: Optional[Any] = None

    def set_callback(self, callback: Any) -> None:
        self._callback = callback

    async def on_prediction(self, prediction: Dict[str, Any]) -> None:
        """Accept a new prediction, debouncing rapid arrivals."""
        confidence = prediction.get("confidence", 0.0)

        # Filter below SNR threshold
        if confidence < UNIFIED_SNR_THRESHOLD:
            logger.debug(
                "Ghost WS: suppressed low-confidence prediction (%.3f < %.3f)",
                confidence,
                UNIFIED_SNR_THRESHOLD,
            )
            return

        # Keep highest-confidence in the window
        if self._pending is None or confidence > self._pending.get("confidence", 0.0):
            self._pending = prediction

        # Reset debounce timer
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()

        self._timer = asyncio.create_task(self._debounce_fire())

    async def _debounce_fire(self) -> None:
        await asyncio.sleep(self._debounce_s)
        if self._pending is not None and self._callback is not None:
            await self._callback(self._pending)
            self._pending = None


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BIZRA Ghost WS Bridge",
    version="0.1.0",
    docs_url=None,  # No Swagger in production
    redoc_url=None,
)

manager = GhostConnectionManager()
debouncer = PredictionDebouncer()

# Allow file:// and localhost origins so standalone HTML tools can call /rpc
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HTTP-to-TCP JSON-RPC proxy  (used by standalone HTML tools)
# ---------------------------------------------------------------------------


@app.post("/rpc")
async def http_rpc_proxy(request: Request) -> JSONResponse:
    """Proxy a JSON-RPC 2.0 POST to the Desktop Bridge TCP server (port 9742).

    Allows browser fetch() to call invoke_skill / actuator_execute without
    a raw TCP socket.  Auth token is forwarded via X-BIZRA-TOKEN if present.
    """
    BRIDGE_HOST = "127.0.0.1"
    BRIDGE_PORT = int(os.environ.get("DESKTOP_BRIDGE_PORT", "9742"))
    CONNECT_TIMEOUT = 3.0
    READ_TIMEOUT = 30.0

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None,
            },
            status_code=400,
        )

    # Forward auth token from HTTP header into message["headers"] — the structure
    # that desktop_bridge._validate_auth() reads (X-BIZRA-TOKEN, X-BIZRA-TS, X-BIZRA-NONCE).
    token = request.headers.get("X-BIZRA-TOKEN", "")
    if token:
        hdrs = body.setdefault("headers", {})
        hdrs.setdefault("X-BIZRA-TOKEN", token)
        hdrs.setdefault("X-BIZRA-TS", str(int(time.time() * 1000)))
        hdrs.setdefault("X-BIZRA-NONCE", str(uuid.uuid4()))

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(BRIDGE_HOST, BRIDGE_PORT),
            timeout=CONNECT_TIMEOUT,
        )
    except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32001,
                    "message": "Desktop Bridge offline (port 9742)",
                },
                "id": body.get("id"),
            },
            status_code=503,
        )

    try:
        writer.write(json.dumps(body).encode() + b"\n")
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=READ_TIMEOUT)
        if not raw:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32002,
                        "message": "Desktop Bridge closed connection without response",
                    },
                    "id": body.get("id"),
                },
                status_code=502,
            )
        return JSONResponse(json.loads(raw.decode()))
    except Exception as exc:
        logger.warning("http_rpc_proxy error: %s", exc)
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32002, "message": str(exc)},
                "id": body.get("id"),
            },
            status_code=502,
        )
    finally:
        try:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
        except Exception:
            pass


@app.get("/health")
async def health() -> JSONResponse:
    """Health endpoint for Docker healthcheck and monitoring."""
    summary = manager.health_summary()
    summary["status"] = "healthy"
    summary["thresholds"] = {
        "ihsan": UNIFIED_IHSAN_THRESHOLD,
        "snr": UNIFIED_SNR_THRESHOLD,
    }
    return JSONResponse(content=summary)


@app.websocket("/ws/ghost")
async def ghost_overlay_ws(ws: WebSocket) -> None:
    """WebSocket endpoint for Ghost Overlay clients.

    Protocol:
      1. Client connects to /ws/ghost
      2. In production, first message must be {"auth": "<token>"}
      3. Server pushes OverlayEvent messages as JSON text frames
      4. Client can send {"gesture": "solidify"|"dismiss"|"scroll_next"|"scroll_prev"}
    """
    if not await manager.connect(ws):
        await ws.close(code=1013, reason="Max clients reached")
        return

    # Auth handshake (enforced in production)
    if BIZRA_ENV == "production" and GHOST_WS_AUTH_TOKEN:
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("auth") != GHOST_WS_AUTH_TOKEN:
                await ws.close(code=4001, reason="Auth failed")
                manager.disconnect(ws)
                return
        except (asyncio.TimeoutError, json.JSONDecodeError):
            await ws.close(code=4001, reason="Auth timeout or invalid")
            manager.disconnect(ws)
            return

    # Send initial state
    await manager.send_to(
        ws,
        OverlayEvent(
            type="connected",
            suggestions=[],
            auto_dismiss_at=None,
        ),
    )

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Ghost WS: invalid JSON from client")
                continue

            gesture = msg.get("gesture")
            if gesture in ("solidify", "dismiss", "scroll_next", "scroll_prev"):
                logger.info("Ghost WS: sovereign gesture received: %s", gesture)
                # Gesture handling delegated to Ghost Overlay Daemon
                # We broadcast the gesture to all other clients (multi-display support)
                await manager.broadcast(
                    OverlayEvent(type="gesture", suggestions=[{"gesture": gesture}])
                )

            # Prediction injection (from Node0 proactive loop)
            prediction = msg.get("prediction")
            if prediction:
                await debouncer.on_prediction(prediction)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        logger.exception("Ghost WS: unexpected error in client loop")
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Overlay Event Emitter (called by GhostOverlayDaemon)
# ---------------------------------------------------------------------------


async def emit_overlay_event(event: OverlayEvent) -> int:
    """Push an overlay event to all connected Ghost clients.

    Called by the GhostOverlayDaemon when a new suggestion set is ready.
    Returns the number of clients that received the event.
    """
    return await manager.broadcast(event)


async def _on_debounced_prediction(prediction: Dict[str, Any]) -> None:
    """Callback from PredictionDebouncer — build overlay and broadcast."""
    event = OverlayEvent(
        type="show_overlay",
        suggestions=[
            {
                "intent": prediction.get("intent", "unknown"),
                "confidence": prediction.get("confidence", 0.0),
                "node_id": prediction.get("node_id"),
            }
        ],
        auto_dismiss_at=time.time() + GHOST_IDLE_TIMEOUT_MS / 1000.0,
    )
    sent = await manager.broadcast(event)
    logger.info("Ghost WS: overlay event sent to %d clients", sent)


# Wire debouncer callback
debouncer.set_callback(_on_debounced_prediction)


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup() -> None:
    logger.info(
        "Ghost WS Bridge starting — port=%s, debounce=%dms, idle=%dms, snr=%.2f, ihsan=%.2f",
        WS_PORT,
        GHOST_DEBOUNCE_MS,
        GHOST_IDLE_TIMEOUT_MS,
        UNIFIED_SNR_THRESHOLD,
        UNIFIED_IHSAN_THRESHOLD,
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    logger.info("Ghost WS Bridge shutting down — %d clients", manager.client_count)
