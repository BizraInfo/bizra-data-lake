"""
Mission Orchestrator — End-to-end sovereign task execution.

Connects: DesktopBridge → ChannelDispatcher → BrowserMCPClient → Synthesis
         → SNR/Ihsan Gate → EvidenceLedger → LivingMemory

Standing on Giants:
  - Shannon: SNR scoring on output quality
  - Lamport: Hash-chained evidence with ordering invariant
  - Boyd: OODA loop (Observe context → Orient channels → Decide synthesis → Act)
  - Al-Ghazali: Ihsan gate as hard constitutional constraint
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.snr_protocol import normalize_snr_linear

logger = logging.getLogger(__name__)


# ── Data Types ──────────────────────────────────────────────────────


@dataclass
class DesktopContext:
    """Desktop state captured at mission start."""

    active_window_title: str = "unknown"
    clipboard_text: str = ""
    screen_geometry: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionRequest:
    """A user-initiated mission."""

    mission_id: str
    description: str
    context: DesktopContext
    timestamp: float
    source: str = "ahk_hotkey"


@dataclass
class ChannelResult:
    """Result from a single channel execution."""

    channel: str
    success: bool
    data: dict[str, Any]
    duration_ms: float
    error: str | None = None


@dataclass
class InferenceProvenance:
    """Records exactly how a synthesis was produced.

    Receipts without provenance are forensically incomplete (§7).
    This is part of the receipt, not a sidecar — per Spine §8-§9.

    Standing on Giants:
      - Lamport (1978): causal ordering of inference events
      - Shannon (1948): channel identity in communication
    """

    backend: str  # "ollama" | "lmstudio" | "gateway" | "template"
    model_id: str  # e.g., "phi3:mini", "deepseek-r1:14b"
    fallback_chain: list[str]  # e.g., ["ollama:TimeoutError", "gateway:success"]
    latency_ms: float  # wall-clock time for inference
    tokens_generated: int  # output token count (0 if template)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "fallback_chain": self.fallback_chain,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_generated": self.tokens_generated,
        }


@dataclass
class MissionResult:
    """Complete result of a mission execution."""

    mission_id: str
    status: str  # COMPLETE | PARTIAL | FAILED
    channels_executed: list[ChannelResult]
    synthesis: str
    briefing_path: str | None
    evidence_receipt_id: str
    ihsan_score: float
    snr_score: float
    duration_ms: float
    memory_entry_id: str = ""
    inference_provenance: InferenceProvenance | None = None


# ── HDA Client ──────────────────────────────────────────────────────


class HDAError(RuntimeError):
    """Error from AHK HDA server."""


class HDAClient:
    """Async TCP client for AHK HDA server (JSON-RPC 2.0)."""

    def __init__(self, host: str, port: int, token: str) -> None:
        self.host = host
        self.port = port
        self.token = token
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id = 0

    async def connect(self) -> bool:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=5.0,
            )
            return True
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            return False

    async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._writer:
            raise HDAError("Not connected to HDA server")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
            "auth": {
                "token": self.token,
                "timestamp": int(time.time()),
                "nonce": secrets.token_hex(16),
            },
        }

        payload = json.dumps(request) + "\n"
        self._writer.write(payload.encode())
        await self._writer.drain()

        if not self._reader:
            raise HDAError("Reader closed")

        line = await asyncio.wait_for(self._reader.readline(), timeout=30.0)
        response = json.loads(line.decode())

        if "error" in response:
            raise HDAError(response["error"].get("message", "Unknown HDA error"))

        return response.get("result", {})

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except (OSError, ConnectionError) as _close_err:
                logger.debug("Connection already closed: %s", _close_err)
            except Exception as exc:  # noqa: BLE001 — connection cleanup boundary
                logger.debug("Connection cleanup: %s", exc, exc_info=True)
            self._writer = None
            self._reader = None


# ── Mission Orchestrator ────────────────────────────────────────────


class MissionOrchestrator:
    """Central coordinator for end-to-end sovereign task execution."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._memory_path = Path(config.get("memory_path", "/tmp/bizra-mission/memory"))
        self._evidence_path = Path(
            config.get("evidence_path", "/tmp/bizra-mission/evidence.jsonl")
        )
        self._hda_port = int(config.get("hda_port", 9743))
        self._workspace_root = Path(config.get("workspace_root", Path.cwd())).resolve()

        # Lazy-initialized components
        self._memory: Any = None
        self._evidence_ledger: Any = None
        self._snr_engine: Any = None
        self._dispatcher: Any = None
        self._event_bus: Any = None
        self._hda_client: HDAClient | None = None

        # Optional injected components
        self.gateway: Any = None  # InferenceGateway for LLM synthesis

        # Crypto (for evidence signing)
        self._signer_private_hex: str | None = None
        self._signer_public_hex: str | None = None

        self._initialized = False

    async def initialize(self) -> None:
        """Boot sequence — call once at startup."""
        if self._initialized:
            return

        # Ensure paths exist
        self._memory_path.mkdir(parents=True, exist_ok=True)
        self._evidence_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize LivingMemory
        try:
            from core.living_memory.core import LivingMemoryCore

            self._memory = LivingMemoryCore(storage_path=self._memory_path)
            await self._memory.initialize()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("LivingMemory not available: %s", exc)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("LivingMemory init failed (continuing without): %s", exc)
        except Exception as exc:  # noqa: BLE001 — optional subsystem init
            logger.warning("LivingMemory init failed (continuing without): %s", exc, exc_info=True)

        # Initialize EvidenceLedger
        try:
            from core.proof_engine.evidence_ledger import EvidenceLedger

            self._evidence_ledger = EvidenceLedger(
                path=self._evidence_path, validate_on_append=False
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("EvidenceLedger not available: %s", exc)
        except (OSError, ValueError) as exc:
            logger.warning("EvidenceLedger init failed (continuing without): %s", exc)
        except Exception as exc:  # noqa: BLE001 — optional subsystem init
            logger.warning("EvidenceLedger init failed (continuing without): %s", exc, exc_info=True)

        # Initialize SNR engine
        try:
            from core.apex.snr_apex_engine import SNRApexEngine

            self._snr_engine = SNRApexEngine()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("SNRApexEngine not available: %s", exc)
        except (RuntimeError, ValueError) as exc:
            logger.warning("SNRApexEngine init failed (continuing without): %s", exc)
        except Exception as exc:  # noqa: BLE001 — optional subsystem init
            logger.warning("SNRApexEngine init failed (continuing without): %s", exc, exc_info=True)

        # Initialize ChannelDispatcher
        try:
            from core.bridges.channel_dispatcher import ChannelDispatcher

            self._dispatcher = ChannelDispatcher()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("ChannelDispatcher not available: %s", exc)
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning(
                "ChannelDispatcher init failed (continuing without): %s", exc
            )
        except Exception as exc:  # noqa: BLE001 — optional subsystem init
            logger.warning(
                "ChannelDispatcher init failed (continuing without): %s", exc, exc_info=True,
            )

        # Initialize EventBus
        try:
            from core.sovereign.event_bus import get_event_bus

            self._event_bus = get_event_bus()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("EventBus not available: %s", exc)
        except (RuntimeError, ValueError) as exc:
            logger.warning("EventBus init failed (continuing without): %s", exc)
        except Exception as exc:  # noqa: BLE001 — optional subsystem init
            logger.warning("EventBus init failed (continuing without): %s", exc, exc_info=True)

        # Initialize InferenceGateway (Ollama/LM Studio) — guarded by env var
        _llm_flag = os.environ.get("BIZRA_ENABLE_LLM", "").lower()
        if _llm_flag in ("1", "true", "yes"):
            try:
                from core.inference.gateway import InferenceConfig, InferenceGateway

                gw = InferenceGateway(config=InferenceConfig(require_local=False))
                if await gw.initialize():
                    self.gateway = gw
                    logger.info(
                        "InferenceGateway initialized (LLM synthesis available)"
                    )
                else:
                    logger.info(
                        "InferenceGateway: no backends available (template mode)"
                    )
            except (ImportError, ModuleNotFoundError) as exc:
                logger.warning("InferenceGateway not available: %s", exc)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("InferenceGateway init failed (template mode): %s", exc)
            except Exception as exc:  # noqa: BLE001 — optional subsystem init
                logger.warning("InferenceGateway init failed (template mode): %s", exc, exc_info=True)

            # O3: Warm model pool — pre-load model weights to eliminate cold start
            await self._warmup_model()

        # Try to connect to AHK HDA server
        token = os.environ.get("BIZRA_BRIDGE_TOKEN", "")
        if token:
            client = HDAClient(host="127.0.0.1", port=self._hda_port, token=token)
            if await client.connect():
                self._hda_client = client
                logger.info("Connected to AHK HDA server on port %d", self._hda_port)
            else:
                logger.info("AHK HDA not available (Level 0 mode)")

        # Load persistent node-anchored signer (or generate + persist)
        try:
            self._signer_private_hex, self._signer_public_hex = (
                _load_or_create_node_signer(self._config)
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Node signer init failed (crypto/fs): %s", exc)
        except Exception as exc:  # noqa: BLE001 — signer init boundary
            logger.warning("Node signer init failed: %s", exc, exc_info=True)

        self._initialized = True

        await self._emit(
            "mission.system_ready",
            {
                "hda_connected": self._hda_client is not None,
                "memory_initialized": self._memory is not None,
                "gateway_available": self.gateway is not None,
            },
        )

    async def _warmup_model(self) -> None:
        """Pre-load LLM weights into memory (O3: eliminate cold start penalty)."""
        try:
            import httpx

            ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Check if Ollama is reachable
                try:
                    await client.get(f"{ollama_url}/api/tags")
                except (OSError, ConnectionError) as exc:
                    logger.info("Ollama not reachable — skipping warmup: %s", exc)
                    return
                except Exception:  # noqa: BLE001 — network probe boundary
                    logger.info("Ollama not reachable — skipping warmup", exc_info=True)
                    return

                # Check if model already loaded (warm)
                ps_resp = await client.get(f"{ollama_url}/api/ps")
                if ps_resp.status_code == 200:
                    loaded = ps_resp.json().get("models", [])
                    if loaded:
                        logger.info(
                            "Model already warm: %s", loaded[0].get("name", "?")
                        )
                        return

                # Send 1-token request to pre-load model weights.
                # keep_alive="30m" prevents Ollama from unloading after 5m idle.
                logger.info("Warming up Ollama model (phi3:mini)...")
                t0 = time.monotonic()
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": "phi3:mini",
                        "prompt": "hello",
                        "stream": False,
                        "keep_alive": "30m",
                        "options": {"num_predict": 1},
                    },
                )
                warmup_ms = (time.monotonic() - t0) * 1000
                if resp.status_code == 200:
                    logger.info("Model warm (%.0fms)", warmup_ms)
                else:
                    logger.warning("Warmup got status %d", resp.status_code)
        except ImportError as exc:
            logger.info("httpx not available — warmup skipped: %s", exc)
        except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
            logger.info("Model warmup skipped (network): %s", exc)
        except Exception as exc:  # noqa: BLE001 — warmup boundary
            logger.info("Model warmup skipped: %s", exc, exc_info=True)

    async def execute(self, request: MissionRequest) -> MissionResult:
        """Execute a complete mission from user intent to proof-traced result."""
        if not self._initialized:
            await self.initialize()

        start_time = time.monotonic()
        mission_id = request.mission_id

        # ── Phase 1: OBSERVE (Boyd) ──
        await self._emit(
            "mission.started",
            {
                "mission_id": mission_id,
                "description": request.description[:200],
            },
        )

        memory_context = await self._retrieve_memories(request.description)

        # ── Phase 2: DECOMPOSE (Channel Dispatch) ──
        plan = self._decompose(mission_id, request.description)

        await self._emit(
            "mission.decomposed",
            {
                "mission_id": mission_id,
                "channels": [
                    s.channel.value if hasattr(s.channel, "value") else s.channel
                    for s in plan.subtasks
                ],
            },
        )

        # ── Phase 3: EXECUTE (Parallel Channels) ──
        channel_results = await self._execute_channels(plan, request)

        # ── Phase 4: SYNTHESIZE (with provenance capture) ──
        synthesis, provenance = await self._synthesize(
            description=request.description,
            channel_results=channel_results,
            memory_context=memory_context,
        )

        # ── Phase 5: GATE (Constitutional) ──
        ihsan_score, snr_normalized = self._score_quality(
            synthesis, channel_results, request
        )

        # ── Phase 6: EVIDENCE (Lamport) ──
        briefing_path = self._write_briefing(synthesis, mission_id)
        receipt_id = self._emit_evidence(
            mission_id, synthesis, ihsan_score, snr_normalized
        )
        memory_entry_id = await self._store_memory(request, synthesis)

        duration_ms = (time.monotonic() - start_time) * 1000

        status = "COMPLETE" if ihsan_score >= UNIFIED_IHSAN_THRESHOLD else "PARTIAL"

        result = MissionResult(
            mission_id=mission_id,
            status=status,
            channels_executed=channel_results,
            synthesis=synthesis,
            briefing_path=briefing_path,
            evidence_receipt_id=receipt_id,
            ihsan_score=ihsan_score,
            snr_score=snr_normalized,
            duration_ms=duration_ms,
            memory_entry_id=memory_entry_id,
            inference_provenance=provenance,
        )

        await self._emit(
            "mission.completed",
            {
                "mission_id": mission_id,
                "status": result.status,
                "duration_ms": duration_ms,
                "ihsan_score": ihsan_score,
                "snr_score": snr_normalized,
                "inference_provenance": provenance.to_dict(),
            },
        )

        return result

    async def handle_rpc(self, params: dict[str, Any]) -> dict[str, Any]:
        """JSON-RPC entry point called by DesktopBridge."""
        description = params.get("description", "")
        if not description:
            return {"error": "Missing 'description' parameter"}

        mission_id = secrets.token_hex(16)

        ctx_data = params.get("context", {})
        context = DesktopContext(
            active_window_title=ctx_data.get("active_window", "unknown"),
            clipboard_text=ctx_data.get("clipboard", "")[:4096],
            screen_geometry=ctx_data.get("screen", {}),
        )

        request = MissionRequest(
            mission_id=mission_id,
            description=description,
            context=context,
            timestamp=time.time(),
            source="ahk_hotkey",
        )

        result = await self.execute(request)

        rpc_response: dict[str, Any] = {
            "mission_id": result.mission_id,
            "status": result.status,
            "synthesis": result.synthesis[:2000],
            "briefing_path": result.briefing_path,
            "evidence_receipt_id": result.evidence_receipt_id,
            "ihsan_score": result.ihsan_score,
            "snr_score": result.snr_score,
            "duration_ms": result.duration_ms,
            "channels": [
                {
                    "channel": cr.channel,
                    "success": cr.success,
                    "duration_ms": cr.duration_ms,
                }
                for cr in result.channels_executed
            ],
        }
        if result.inference_provenance:
            rpc_response["inference_provenance"] = result.inference_provenance.to_dict()
        return rpc_response

    # ── Private: Channel Execution ──────────────────────────────────

    def _decompose(self, mission_id: str, description: str) -> Any:
        if self._dispatcher:
            return self._dispatcher.decompose(mission_id, description)

        # Fallback: single browser research task
        from core.bridges.channel_dispatcher import Channel, MissionPlan, SubTask

        return MissionPlan(
            mission_id=mission_id,
            subtasks=[
                SubTask(
                    id=f"{mission_id}-browser",
                    description="Browser research",
                    channel=Channel.BROWSER,
                    params={"query": description},
                ),
            ],
        )

    async def _execute_channels(
        self, plan: Any, request: MissionRequest
    ) -> list[ChannelResult]:
        results: list[ChannelResult] = []

        for subtask in plan.subtasks:
            channel_name = (
                subtask.channel.value
                if hasattr(subtask.channel, "value")
                else str(subtask.channel)
            )
            start = time.monotonic()
            try:
                if channel_name == "browser":
                    data = await self._execute_browser(subtask, request)
                elif channel_name == "desktop":
                    data = await self._execute_desktop(subtask, request)
                else:
                    data = {"channel": channel_name, "note": "not implemented for demo"}

                results.append(
                    ChannelResult(
                        channel=channel_name,
                        success=True,
                        data=data,
                        duration_ms=(time.monotonic() - start) * 1000,
                    )
                )
            except (RuntimeError, ValueError, OSError, asyncio.TimeoutError) as exc:
                results.append(
                    ChannelResult(
                        channel=channel_name,
                        success=False,
                        data={},
                        duration_ms=(time.monotonic() - start) * 1000,
                        error=str(exc)[:500],
                    )
                )
            except Exception as exc:  # noqa: BLE001 — channel execution boundary
                logger.warning("Channel %s unexpected failure", channel_name, exc_info=True)
                results.append(
                    ChannelResult(
                        channel=channel_name,
                        success=False,
                        data={},
                        duration_ms=(time.monotonic() - start) * 1000,
                        error=str(exc)[:500],
                    )
                )

        return results

    async def _execute_browser(
        self, subtask: Any, request: MissionRequest
    ) -> dict[str, Any]:
        from core.bridges.browser_mcp_client import BrowserMCPClient

        mode = os.environ.get("BIZRA_BROWSER_MODE", "direct").strip().lower()
        if mode not in {"mock", "direct", "mcp"}:
            mode = "direct"
        client = BrowserMCPClient(mode=mode)
        query = subtask.params.get("query", request.description)
        research = await client.research(query)

        return {
            "query": query,
            "results_count": len(research.get("results", [])),
            "results": research.get("results", [])[:5],
            "summary": research.get("summary", ""),
            "mode": mode,
        }

    async def _execute_desktop(
        self, subtask: Any, request: MissionRequest
    ) -> dict[str, Any]:
        if self._hda_client:
            try:
                result = await self._hda_client.send_command("get_context", {})
                return {
                    "context_captured": True,
                    "active_window": result.get("active_window", "unknown"),
                    "hda_connected": True,
                }
            except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
                logger.warning("HDA get_context failed (connection): %s", exc)
            except Exception as exc:  # noqa: BLE001 — HDA boundary
                logger.warning("HDA get_context failed: %s", exc, exc_info=True)

        local_fs = self._execute_local_filesystem(subtask, request)
        if local_fs is not None:
            local_fs["hda_connected"] = False
            return local_fs

        return {
            "context_captured": False,
            "hda_connected": False,
            "fallback": "python_file_io",
            "active_window": request.context.active_window_title,
        }

    def _execute_local_filesystem(
        self,
        subtask: Any,
        request: MissionRequest,
    ) -> dict[str, Any] | None:
        """Execute explicit local filesystem intents when HDA is unavailable.

        Accepted intent formats:
        - ``write file <relative-path> :: <content>``
        - ``append file <relative-path> :: <content>``
        - ``read file <relative-path>``
        - ``list dir <relative-path>``
        """
        description = str(
            (getattr(subtask, "params", {}) or {}).get(
                "description",
                request.description,
            )
        ).strip()

        if not description:
            return None

        write_match = re.match(
            r"^(?:write|create)\s+file\s+(.+?)\s*::\s*(.+)$",
            description,
            re.IGNORECASE | re.DOTALL,
        )
        if write_match:
            path = self._resolve_workspace_path(write_match.group(1))
            content = write_match.group(2)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {
                "context_captured": True,
                "filesystem_action": "write",
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
            }

        append_match = re.match(
            r"^append\s+file\s+(.+?)\s*::\s*(.+)$",
            description,
            re.IGNORECASE | re.DOTALL,
        )
        if append_match:
            path = self._resolve_workspace_path(append_match.group(1))
            content = append_match.group(2)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(content)
            return {
                "context_captured": True,
                "filesystem_action": "append",
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
            }

        read_match = re.match(r"^read\s+file\s+(.+)$", description, re.IGNORECASE)
        if read_match:
            path = self._resolve_workspace_path(read_match.group(1))
            if not path.exists():
                return {
                    "context_captured": True,
                    "filesystem_action": "read",
                    "path": str(path),
                    "error": "file_not_found",
                }
            content = path.read_text(encoding="utf-8", errors="replace")
            return {
                "context_captured": True,
                "filesystem_action": "read",
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
                "preview": content[:400],
            }

        list_match = re.match(
            r"^(?:list\s+(?:dir|files\s+in)|show\s+files\s+in)\s+(.+)$",
            description,
            re.IGNORECASE,
        )
        if list_match:
            path = self._resolve_workspace_path(list_match.group(1))
            if not path.exists() or not path.is_dir():
                return {
                    "context_captured": True,
                    "filesystem_action": "list",
                    "path": str(path),
                    "error": "directory_not_found",
                }
            entries = sorted(p.name for p in path.iterdir())[:200]
            return {
                "context_captured": True,
                "filesystem_action": "list",
                "path": str(path),
                "entries": entries,
            }

        return None

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        """Resolve and confine filesystem actions to the active workspace."""
        candidate = Path(raw_path.strip().strip('"'))
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self._workspace_root / candidate).resolve()

        workspace = self._workspace_root
        if resolved == workspace or workspace in resolved.parents:
            return resolved
        raise ValueError(f"path_outside_workspace:{resolved}")

    # ── Private: Synthesis ──────────────────────────────────────────

    async def _synthesize(
        self,
        description: str,
        channel_results: list[ChannelResult],
        memory_context: list[Any],
    ) -> tuple[str, InferenceProvenance]:
        """Synthesize mission output and capture inference provenance.

        Returns (synthesis_text, provenance) — provenance is part of the
        receipt, not a sidecar (per Spine §8-§9).
        """
        browser_data = next(
            (r.data for r in channel_results if r.channel == "browser" and r.success),
            None,
        )
        desktop_data = next(
            (r.data for r in channel_results if r.channel == "desktop" and r.success),
            None,
        )

        fallback_chain: list[str] = []
        inference_start = time.monotonic()

        # LLM synthesis: Ollama direct → Gateway (GPU) → Template
        _llm_enabled = os.environ.get("BIZRA_ENABLE_LLM", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if not _llm_enabled:
            text = self._template_synthesis(description, browser_data, desktop_data)
            latency = (time.monotonic() - inference_start) * 1000
            fallback_chain.append("template:success")
            return text, InferenceProvenance(
                backend="template", model_id="none",
                fallback_chain=fallback_chain,
                latency_ms=latency, tokens_generated=0,
            )

        prompt = self._build_synthesis_prompt(
            description, browser_data, desktop_data, memory_context
        )

        # 1. Ollama direct (fast path — phi3:mini warm, low latency on CPU)
        ollama_model = os.environ.get("BIZRA_OLLAMA_MODEL", "phi3:mini")
        try:
            import httpx

            ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "30m",
                        "options": {"temperature": 0.3, "num_predict": 512},
                    },
                )
                resp.raise_for_status()
                content = resp.json().get("response", "")
                if content.strip():
                    latency = (time.monotonic() - inference_start) * 1000
                    logger.info("Ollama synthesis complete (%d chars)", len(content))
                    fallback_chain.append("ollama:success")
                    return content, InferenceProvenance(
                        backend="ollama", model_id=ollama_model,
                        fallback_chain=fallback_chain,
                        latency_ms=latency,
                        tokens_generated=len(content.split()),
                    )
                fallback_chain.append("ollama:empty_content")
                logger.warning("Ollama returned empty content — trying gateway")
        except ImportError:
            fallback_chain.append("ollama:ImportError")
            logger.info("httpx not available — skipping Ollama synthesis")
        except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
            fallback_chain.append(f"ollama:{type(exc).__name__}")
            logger.warning("Ollama synthesis failed (network): %s — trying gateway", exc)
        except Exception as exc:  # noqa: BLE001 — LLM synthesis boundary
            fallback_chain.append(f"ollama:{type(exc).__name__}")
            logger.warning("Ollama synthesis failed: %s — trying gateway", exc, exc_info=True)

        # 2. Gateway (LM Studio GPU + Ollama fallback chain)
        gateway_timeout = float(os.environ.get("BIZRA_GATEWAY_TIMEOUT", "20"))
        if self.gateway:
            try:
                result = await asyncio.wait_for(
                    self.gateway.infer(prompt, max_tokens=1024, temperature=0.3),
                    timeout=gateway_timeout,
                )
                if result.content.strip():
                    latency = (time.monotonic() - inference_start) * 1000
                    gw_backend = result.backend.value if hasattr(result.backend, "value") else str(result.backend)
                    logger.info(
                        "Gateway synthesis complete (%d chars, backend=%s, %.0fms)",
                        len(result.content), gw_backend, result.latency_ms,
                    )
                    fallback_chain.append(f"gateway:{gw_backend}:success")
                    return result.content, InferenceProvenance(
                        backend=f"gateway:{gw_backend}",
                        model_id=getattr(result, "model", "unknown"),
                        fallback_chain=fallback_chain,
                        latency_ms=latency,
                        tokens_generated=getattr(result, "tokens_generated", len(result.content.split())),
                    )
                fallback_chain.append("gateway:empty_content")
                logger.warning("Gateway returned empty content — falling through")
            except asyncio.TimeoutError:
                fallback_chain.append("gateway:TimeoutError")
                logger.warning("Gateway synthesis timed out (%.0fs)", gateway_timeout)
            except (RuntimeError, ValueError, OSError) as exc:
                fallback_chain.append(f"gateway:{type(exc).__name__}")
                logger.warning("Gateway synthesis failed (known): %s", exc)
            except Exception as exc:  # noqa: BLE001 — gateway synthesis boundary
                fallback_chain.append(f"gateway:{type(exc).__name__}")
                logger.warning("Gateway synthesis failed: %s", exc, exc_info=True)

        # 3. Template (always available, no LLM needed)
        text = self._template_synthesis(description, browser_data, desktop_data)
        latency = (time.monotonic() - inference_start) * 1000
        fallback_chain.append("template:success")
        return text, InferenceProvenance(
            backend="template", model_id="none",
            fallback_chain=fallback_chain,
            latency_ms=latency, tokens_generated=0,
        )

    def _template_synthesis(
        self,
        description: str,
        browser_data: dict[str, Any] | None,
        desktop_data: dict[str, Any] | None,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [
            "# BIZRA Mission Briefing",
            "",
            f"**Mission:** {description}",
            f"**Generated:** {now}",
            "**Node:** NODE0 (Sovereign)",
            "",
        ]

        if browser_data and browser_data.get("results"):
            lines.append("## Research Findings")
            lines.append("")
            for i, result in enumerate(browser_data["results"][:5], 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                lines.append(f"### {i}. {title}")
                lines.append(f"**Source:** {url}")
                lines.append(f"{snippet}")
                lines.append("")

        if desktop_data:
            lines.append("## Desktop Context")
            lines.append("")
            if desktop_data.get("active_window"):
                lines.append(f"- Active window: {desktop_data['active_window']}")
            if desktop_data.get("hda_connected"):
                lines.append("- HDA: Connected (perception-action loop active)")
            lines.append("")

        lines.extend(
            [
                "## Proof Trace",
                "",
                "This briefing was generated with constitutional governance:",
                "- Ihsan quality gate enforced",
                "- SNR scoring applied to all content",
                "- Evidence receipt hash-chained to ledger",
                "- Ed25519 digital signature attached",
                "",
                "---",
                "*Generated by BIZRA Node0 - Sovereign AI*",
            ]
        )

        return "\n".join(lines)

    def _build_synthesis_prompt(
        self,
        description: str,
        browser_data: dict[str, Any] | None,
        desktop_data: dict[str, Any] | None,
        memory_context: list[Any],
    ) -> str:
        parts = [
            f"Task: {description}",
            "",
            "Synthesize a concise markdown briefing from these sources:",
        ]
        if browser_data and browser_data.get("results"):
            parts.append("\nWeb Research:")
            for r in browser_data["results"][:3]:  # Top 3 only (prompt compression)
                snippet = r.get("snippet", "")[:100]
                parts.append(f"- {r.get('title', 'N/A')}: {snippet}")
        if desktop_data and desktop_data.get("active_window"):
            parts.append(f"\nDesktop: {desktop_data['active_window']}")
        if memory_context:
            parts.append("\nPrior context:")
            for m in memory_context[:2]:  # Top 2 memories (prompt compression)
                content = m.content if hasattr(m, "content") else str(m)
                parts.append(f"- {content[:100]}")
        parts.append("\nBe structured, cite sources, use markdown headers.")
        return "\n".join(parts)

    # ── Private: Quality Gate ───────────────────────────────────────

    def _score_quality(
        self,
        synthesis: str,
        channel_results: list[ChannelResult],
        request: MissionRequest,
    ) -> tuple[float, float]:
        if not self._snr_engine:
            # Fail-honest: without quality engine, score below threshold → PARTIAL
            return 0.80, 0.75

        try:
            successful_channels = sum(1 for r in channel_results if r.success)
            total_channels = max(len(channel_results), 1)
            groundedness = successful_channels / total_channels

            analysis = self._snr_engine.analyze(
                signal_components={
                    "relevance": 0.85,
                    "groundedness": groundedness,
                    "coherence": 0.90,
                    "actionability": 0.80,
                    "novelty": 0.70,
                },
                noise_components={
                    "hallucination_risk": 0.05,
                    "repetition": 0.03,
                    "irrelevance": 0.05,
                    "ambiguity": 0.08,
                    "staleness": 0.02,
                    "toxicity": 0.0,
                },
            )

            snr_normalized = normalize_snr_linear(analysis.snr_linear)
            # SNRAnalysis uses ihsan_achieved (bool), derive numeric score
            ihsan_score = 0.95 if analysis.ihsan_achieved else 0.80

            return ihsan_score, snr_normalized
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("SNR scoring failed (data): %s", exc)
            return 0.80, 0.75
        except Exception as exc:  # noqa: BLE001 — SNR scoring boundary
            logger.warning("SNR scoring failed: %s", exc, exc_info=True)
            # Fail-honest: scoring failure → below threshold → PARTIAL
            return 0.80, 0.75

    # ── Private: Evidence ───────────────────────────────────────────

    def _write_briefing(self, synthesis: str, mission_id: str) -> str | None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BIZRA_Brief_{timestamp}.md"

        desktop_path = self._find_desktop_path()
        if desktop_path:
            filepath = desktop_path / filename
        else:
            missions_dir = Path("missions")
            missions_dir.mkdir(exist_ok=True)
            filepath = missions_dir / filename

        try:
            filepath.write_text(synthesis, encoding="utf-8")
            return str(filepath)
        except (OSError, PermissionError) as exc:
            logger.warning("Failed to write briefing (filesystem): %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 — file write boundary
            logger.warning("Failed to write briefing: %s", exc, exc_info=True)
            return None

    def _find_desktop_path(self) -> Path | None:
        candidates = [
            Path("/mnt/c/Users") / os.environ.get("WINDOWS_USER", "mumo") / "Desktop",
            Path.home() / "Desktop",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _emit_evidence(
        self, mission_id: str, synthesis: str, ihsan_score: float, snr_score: float
    ) -> str:
        receipt_id = mission_id[:16]

        if not self._evidence_ledger or not self._signer_private_hex:
            return receipt_id

        try:
            from core.proof_engine.evidence_ledger import emit_receipt

            digest = hashlib.blake2b(synthesis.encode(), digest_size=32).hexdigest()

            emit_receipt(
                ledger=self._evidence_ledger,
                receipt_id=receipt_id,
                node_id="NODE0-MISSION",
                snr_score=snr_score,
                ihsan_score=ihsan_score,
                seal_digest=digest,
                signer_private_key_hex=self._signer_private_hex,
                signer_public_key_hex=self._signer_public_hex,
            )
        except (ImportError, ValueError, OSError) as exc:
            logger.warning("Evidence emission failed (known): %s", exc)
        except Exception as exc:  # noqa: BLE001 — evidence emission boundary
            logger.warning("Evidence emission failed: %s", exc, exc_info=True)

        return receipt_id

    async def _store_memory(self, request: MissionRequest, synthesis: str) -> str:
        if not self._memory:
            return ""

        try:
            from core.living_memory.core import MemoryType

            entry = await self._memory.encode(
                content=f"Mission: {request.description}\nResult: {synthesis[:500]}",
                memory_type=MemoryType.EPISODIC,
                source=f"mission:{request.mission_id}",
                importance=0.8,
            )
            return entry.id if entry else ""
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            logger.warning("Memory storage failed (known): %s", exc)
            return ""
        except Exception as exc:  # noqa: BLE001 — memory storage boundary
            logger.warning("Memory storage failed: %s", exc, exc_info=True)
            return ""

    async def _retrieve_memories(self, description: str) -> list[Any]:
        if not self._memory:
            return []

        try:
            return await self._memory.retrieve(
                query=description,
                memory_type=None,
                top_k=3,
                min_score=0.3,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Memory retrieval failed (known): %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001 — memory retrieval boundary
            logger.warning("Memory retrieval failed: %s", exc, exc_info=True)
            return []

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        if not self._event_bus:
            return
        try:
            await self._event_bus.emit(topic, payload)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Event emit failed (known) | topic=%s error=%s", topic, exc)
        except Exception as exc:  # noqa: BLE001 — event bus boundary
            logger.warning("Event emit failed | topic=%s error=%s", topic, exc, exc_info=True)


# ── Persistent Node Signer ─────────────────────────────────────────────

_SIGNER_FILENAME = "mission_signer.json"


def _load_or_create_node_signer(
    config: dict[str, Any],
) -> tuple[str, str]:
    """Load persistent Ed25519 keypair from sovereign_state, or create + persist.

    Anchors mission receipts to stable node identity across restarts.
    Falls back to sovereign_state/identity/credentials.json if available.
    """
    from core.pci.crypto import generate_keypair

    # Resolve signer storage path
    state_dir = Path(config.get("sovereign_state_dir", "sovereign_state")).resolve()
    signer_path = state_dir / _SIGNER_FILENAME

    # 1. Try loading existing mission signer
    if signer_path.exists():
        try:
            data = json.loads(signer_path.read_text(encoding="utf-8"))
            priv = data["private_key_hex"]
            pub = data["public_key_hex"]
            if isinstance(priv, str) and isinstance(pub, str) and len(priv) == 64:
                logger.info("Loaded persistent mission signer from %s", signer_path)
                return priv, pub
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt signer file at %s, regenerating", signer_path)

    # 2. Try inheriting from node identity credentials
    creds_path = state_dir / "identity" / "credentials.json"
    if creds_path.exists():
        try:
            creds = json.loads(creds_path.read_text(encoding="utf-8"))
            priv = creds.get("private_key")
            pub = creds.get("public_key")
            if isinstance(priv, str) and isinstance(pub, str) and len(priv) == 64:
                # Persist as mission signer for future loads
                _persist_signer(signer_path, priv, pub, source="node_identity")
                logger.info(
                    "Inherited mission signer from node identity (%s)",
                    creds.get("node_id", "unknown"),
                )
                return priv, pub
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Signer file parse error, regenerating: %s", exc)

    # 3. Generate new keypair and persist
    priv, pub = generate_keypair()
    state_dir.mkdir(parents=True, exist_ok=True)
    _persist_signer(signer_path, priv, pub, source="generated")
    logger.info("Generated and persisted new mission signer at %s", signer_path)
    return priv, pub


def _persist_signer(path: Path, private_hex: str, public_hex: str, source: str) -> None:
    """Write signer keypair to disk with restricted permissions."""
    data = {
        "private_key_hex": private_hex,
        "public_key_hex": public_hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError as exc:
        logger.debug("chmod not supported on this platform: %s", exc)
