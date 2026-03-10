#!/usr/bin/env python3
"""BIZRA Node0 standalone lifecycle manager.

Unified entrypoint for single-node readiness before Alpha-100.

Commands:
  - activate: mint/load identity, activate URP, publish PAT/SAT awareness
  - health:   lifecycle and integration health report
  - task:     autonomous mission execution (filesystem + browser channels)
  - serve:    local API for website/UI integration
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("node0.standalone")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _tcp_open(host: str, port: int, timeout_s: float = 0.4) -> bool:
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        sock.connect((host, port))
    except OSError:
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    if sock is not None:
        return True


@dataclass
class ActivationContext:
    node_id: str
    pat_agent_ids: list[str]
    sat_agent_ids: list[str]
    private_key_hex: str


class Node0StandaloneManager:
    """Single-node activation manager for Node0 first-user readiness."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root
        self.state_root = project_root / "sovereign_state"
        self.identity_dir = self.state_root / "identity"
        self.assets_path = self.state_root / "node0_assets.json"
        self.awareness_path = self.state_root / "pat_awareness.json"
        self.urp_path = self.state_root / "urp_pledge.json"
        self.lifecycle_path = self.state_root / "node0_lifecycle.json"
        self.identity_dir.mkdir(parents=True, exist_ok=True)

    def activate(
        self,
        architect: str = "MoMo",
        strict: bool = False,
    ) -> dict[str, Any]:
        context = self._ensure_identity(architect)
        hardware = self._scan_hardware()
        urp_payload = self._activate_urp(context, hardware)
        assets = self._build_assets(context, hardware, urp_payload)
        awareness = self._build_pat_awareness(context, assets)

        integrations = assets.get("integrations", {})
        gates = {
            "identity_ready": bool(context.node_id),
            "pat_sat_ready": bool(context.pat_agent_ids)
            and bool(context.sat_agent_ids),
            "urp_signed": bool(urp_payload.get("signed", False)),
            "urp_verified": bool(urp_payload.get("signature_verified", False)),
            "assets_written": self.assets_path.exists(),
            "awareness_written": self.awareness_path.exists(),
            "desktop_bridge_reachable": bool(integrations.get("ahk_hda_bridge", False)),
            "mcp_available": bool(integrations.get("mcp", False)),
            "a2a_available": bool(integrations.get("a2a", False)),
            "telescript_available": bool(integrations.get("telescript_permit", False)),
        }
        required = [
            "identity_ready",
            "pat_sat_ready",
            "urp_signed",
            "urp_verified",
            "assets_written",
            "awareness_written",
        ]
        ready = all(gates[k] for k in required)
        if strict:
            ready = ready and all(gates.values())

        lifecycle = {
            "version": "1.0.0",
            "updated_at": _utc_now(),
            "status": "ready" if ready else "degraded",
            "strict_mode": bool(strict),
            "node_id": context.node_id,
            "identity": {
                "pat_agents": len(context.pat_agent_ids),
                "sat_agents": len(context.sat_agent_ids),
            },
            "urp": {
                "signed": urp_payload.get("signed", False),
                "verified": urp_payload.get("signature_verified", False),
                "enforced": urp_payload.get("enforced", False),
                "reason_code": urp_payload.get("reason_code"),
            },
            "artifacts": {
                "assets": str(self.assets_path),
                "pat_awareness": str(self.awareness_path),
                "urp": str(self.urp_path),
                "identity": str(self.identity_dir / "credentials.json"),
            },
            "gates": gates,
        }
        _write_json(self.lifecycle_path, lifecycle)

        return {
            "ok": ready,
            "lifecycle": lifecycle,
            "assets": assets,
            "pat_awareness": awareness,
        }

    def health(self) -> dict[str, Any]:
        lifecycle = _read_json(self.lifecycle_path, default={}) or {}
        assets = _read_json(self.assets_path, default={}) or {}
        awareness = _read_json(self.awareness_path, default={}) or {}
        urp = _read_json(self.urp_path, default={}) or {}

        pid_path = self.state_root / "proactive.pid"
        pid_alive = False
        pid = None
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
                pid_alive = True
            except (ValueError, OSError):
                pid_alive = False

        bridge_online = _tcp_open("127.0.0.1", 9742)
        lm_online = self._check_http("http://127.0.0.1:1234/v1/models")
        ollama_online = self._check_http("http://127.0.0.1:11434/v1/models")

        gates = {
            "identity_credentials": (self.identity_dir / "credentials.json").exists(),
            "lifecycle_file": self.lifecycle_path.exists(),
            "assets_file": self.assets_path.exists(),
            "pat_awareness_file": self.awareness_path.exists(),
            "urp_file": self.urp_path.exists(),
            "urp_signed": bool(urp.get("signed", False)),
            "urp_verified": bool(urp.get("signature_verified", False)),
            "bridge_online": bridge_online,
            "backend_online": lm_online or ollama_online,
        }

        overall = (
            "ready"
            if all(
                gates[k]
                for k in (
                    "identity_credentials",
                    "lifecycle_file",
                    "assets_file",
                    "pat_awareness_file",
                    "urp_file",
                    "urp_signed",
                    "urp_verified",
                )
            )
            else "degraded"
        )

        return {
            "timestamp": _utc_now(),
            "status": overall,
            "node_id": lifecycle.get("node_id")
            or awareness.get("node_id")
            or "unknown",
            "gates": gates,
            "runtime": {
                "proactive_pid": pid,
                "proactive_running": pid_alive,
                "desktop_bridge": bridge_online,
                "lm_studio": lm_online,
                "ollama": ollama_online,
            },
            "identity": lifecycle.get("identity", {}),
            "integrations": assets.get("integrations", {}),
        }

    async def run_task(
        self,
        description: str,
        source: str = "node0_standalone",
        browser_mode: str = "mock",
    ) -> dict[str, Any]:
        if not description.strip():
            raise ValueError("Task description must not be empty")
        if browser_mode not in {"mock", "direct", "mcp"}:
            raise ValueError("browser_mode must be one of: mock, direct, mcp")

        from core.sovereign.mission import (
            DesktopContext,
            MissionOrchestrator,
            MissionRequest,
        )

        orchestrator = MissionOrchestrator(
            {
                "memory_path": str(self.state_root / "memory"),
                "evidence_path": str(self.state_root / "mission_evidence.jsonl"),
                "hda_port": 9742,
                "workspace_root": str(self.project_root),
            }
        )
        request = MissionRequest(
            mission_id=uuid.uuid4().hex,
            description=description,
            context=DesktopContext(active_window_title="node0-standalone"),
            timestamp=time.time(),
            source=source,
        )

        previous_browser_mode = os.environ.get("BIZRA_BROWSER_MODE")
        os.environ["BIZRA_BROWSER_MODE"] = browser_mode
        try:
            result = await orchestrator.execute(request)
        finally:
            if previous_browser_mode is None:
                os.environ.pop("BIZRA_BROWSER_MODE", None)
            else:
                os.environ["BIZRA_BROWSER_MODE"] = previous_browser_mode

        hda_client = getattr(orchestrator, "_hda_client", None)
        if hda_client is not None:
            try:
                await hda_client.close()
            except Exception:
                pass

        channel_results = [
            {
                "channel": r.channel,
                "success": r.success,
                "duration_ms": round(r.duration_ms, 2),
                "error": r.error,
                "data": r.data,
            }
            for r in result.channels_executed
        ]

        fs_action: dict[str, Any] | None = None
        for item in channel_results:
            if item.get("channel") != "desktop":
                continue
            data = item.get("data") or {}
            action = data.get("filesystem_action")
            if not action:
                continue
            fs_action = {
                "action": action,
                "path": data.get("path"),
                "bytes": data.get("bytes"),
                "entries": data.get("entries"),
                "error": data.get("error"),
            }
            break

        payload = {
            "mission_id": result.mission_id,
            "status": result.status,
            "ihsan_score": result.ihsan_score,
            "snr_score": result.snr_score,
            "duration_ms": round(result.duration_ms, 2),
            "briefing_path": result.briefing_path,
            "evidence_receipt_id": result.evidence_receipt_id,
            "channels": channel_results,
            "filesystem_action": fs_action,
            "browser_mode": browser_mode,
            "synthesis": result.synthesis,
        }
        return payload

    def lifecycle(self) -> dict[str, Any]:
        return _read_json(self.lifecycle_path, default={}) or {}

    def assets(self) -> dict[str, Any]:
        return _read_json(self.assets_path, default={}) or {}

    def _check_http(self, url: str) -> bool:
        try:
            import httpx

            resp = httpx.get(url, timeout=1.2)
            return resp.status_code == 200
        except Exception:
            return False

    def _ensure_identity(self, architect: str) -> ActivationContext:
        from core.pat.onboarding import OnboardingWizard

        wizard = OnboardingWizard(node_dir=self.identity_dir)
        existing = wizard.load_existing_credentials()

        if existing is None:
            credentials = wizard.onboard(name=architect)
        else:
            credentials = existing

        return ActivationContext(
            node_id=credentials.node_id,
            pat_agent_ids=list(credentials.pat_agent_ids),
            sat_agent_ids=list(credentials.sat_agent_ids),
            private_key_hex=credentials.private_key,
        )

    def _scan_hardware(self) -> dict[str, Any]:
        from core.genesis.hardware import HardwareScanner

        scanner = HardwareScanner()
        hw = scanner.scan()
        return hw.to_dict()

    def _activate_urp(
        self,
        context: ActivationContext,
        hardware: dict[str, Any],
    ) -> dict[str, Any]:
        from core.genesis.urp import (
            URPPledge,
            pledge_resources,
            verify_pledge_signature,
        )

        pledge = pledge_resources(
            node_id=context.node_id,
            hardware_info=hardware,
            signing_private_key_hex=context.private_key_hex,
        )
        verified = verify_pledge_signature(pledge)

        payload = pledge.to_dict()
        payload["signature_verified"] = bool(verified)
        payload["updated_at"] = _utc_now()

        _write_json(self.urp_path, payload)

        # Defensive type reconstruction check (guards stale schema drift)
        try:
            reconstructed = URPPledge(
                **{
                    k: v
                    for k, v in payload.items()
                    if k in URPPledge.__dataclass_fields__
                }
            )
            payload["reconstructed_signed"] = bool(reconstructed.signed)
        except Exception:
            payload["reconstructed_signed"] = False

        return payload

    def _build_assets(
        self,
        context: ActivationContext,
        hardware: dict[str, Any],
        urp: dict[str, Any],
    ) -> dict[str, Any]:
        integrations = {
            "ahk_hda_bridge": _tcp_open("127.0.0.1", 9742),
            "mcp": self._module_available("core.skills.mcp_bridge"),
            "a2a": self._module_available("core.a2a.engine"),
            "telescript_permit": self._module_available("core.sovereign.permit"),
            "browser_autonomy": self._module_available(
                "core.bridges.browser_mcp_client"
            ),
        }

        roots = [
            str(self.project_root),
            str(self.state_root),
            str(self.project_root / "missions"),
            str(self.project_root / "04_GOLD"),
        ]

        payload = {
            "node_id": context.node_id,
            "updated_at": _utc_now(),
            "hardware": hardware,
            "urp_budget": urp.get("resource_budget", {}),
            "filesystem_roots": roots,
            "integrations": integrations,
            "space_awareness": {
                "workspace": str(self.project_root),
                "state": str(self.state_root),
                "identity": str(self.identity_dir),
            },
        }
        _write_json(self.assets_path, payload)
        return payload

    def _build_pat_awareness(
        self,
        context: ActivationContext,
        assets: dict[str, Any],
    ) -> dict[str, Any]:
        pat_caps = [
            "filesystem.read",
            "filesystem.write",
            "browser.research",
            "hda.execute",
            "mcp.context",
            "a2a.delegate",
            "telescript.permit",
        ]
        sat_caps = [
            "resource.pool",
            "governance.guard",
            "network.health",
            "distribution.fairness",
            "proof.validation",
        ]

        payload = {
            "node_id": context.node_id,
            "updated_at": _utc_now(),
            "assets_path": str(self.assets_path),
            "pat": [
                {
                    "agent_id": agent_id,
                    "capabilities": pat_caps,
                    "asset_awareness": {
                        "filesystem_roots": assets.get("filesystem_roots", []),
                        "integrations": assets.get("integrations", {}),
                    },
                }
                for agent_id in context.pat_agent_ids
            ],
            "sat": [
                {
                    "agent_id": agent_id,
                    "capabilities": sat_caps,
                }
                for agent_id in context.sat_agent_ids
            ],
        }

        _write_json(self.awareness_path, payload)
        return payload

    def _module_available(self, module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except Exception:
            return False

    def _maybe_execute_filesystem_action(
        self, description: str
    ) -> dict[str, Any] | None:
        text = description.strip()

        write_match = re.match(
            r"^(?:write|create)\s+file\s+(.+?)\s*::\s*(.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if write_match:
            raw_path = write_match.group(1).strip().strip('"')
            content = write_match.group(2)
            path = self._resolve_workspace_path(raw_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {
                "action": "write",
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
            }

        append_match = re.match(
            r"^append\s+file\s+(.+?)\s*::\s*(.+)$", text, re.IGNORECASE | re.DOTALL
        )
        if append_match:
            raw_path = append_match.group(1).strip().strip('"')
            content = append_match.group(2)
            path = self._resolve_workspace_path(raw_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(content)
            return {
                "action": "append",
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
            }

        read_match = re.match(r"^read\s+file\s+(.+)$", text, re.IGNORECASE)
        if read_match:
            raw_path = read_match.group(1).strip().strip('"')
            path = self._resolve_workspace_path(raw_path)
            if not path.exists():
                return {
                    "action": "read",
                    "path": str(path),
                    "error": "file_not_found",
                }
            content = path.read_text(encoding="utf-8", errors="replace")
            return {
                "action": "read",
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
                "preview": content[:400],
            }

        list_match = re.match(
            r"^(?:list\s+(?:dir|files\s+in)|show\s+files\s+in)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if list_match:
            raw_path = list_match.group(1).strip().strip('"')
            path = self._resolve_workspace_path(raw_path)
            if not path.exists() or not path.is_dir():
                return {
                    "action": "list",
                    "path": str(path),
                    "error": "directory_not_found",
                }
            entries = sorted(p.name for p in path.iterdir())[:200]
            return {
                "action": "list",
                "path": str(path),
                "entries": entries,
            }

        return None

    def _resolve_workspace_path(self, candidate: str) -> Path:
        raw = Path(candidate)
        if raw.is_absolute():
            resolved = raw.resolve()
        else:
            resolved = (self.project_root / raw).resolve()

        root = self.project_root.resolve()
        if root == resolved or root in resolved.parents:
            return resolved

        raise ValueError(f"Path outside workspace is blocked: {resolved}")


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


async def _cmd_activate(args: argparse.Namespace) -> int:
    manager = Node0StandaloneManager()
    result = manager.activate(architect=args.architect, strict=args.strict)
    _print_json(result)
    return 0 if result.get("ok") else 2


async def _cmd_health(args: argparse.Namespace) -> int:
    manager = Node0StandaloneManager()
    result = manager.health()
    _print_json(result)
    return 0 if result.get("status") == "ready" else 2


async def _cmd_task(args: argparse.Namespace) -> int:
    manager = Node0StandaloneManager()
    result = await manager.run_task(
        args.description,
        source=args.source,
        browser_mode=args.browser_mode,
    )
    _print_json(result)
    return 0 if result.get("status") in {"COMPLETE", "PARTIAL"} else 1


# ═══════════════════════════════════════════════════════════════════════════════
# API models and fleet config (module-level for FastAPI body resolution)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from pydantic import BaseModel as _BaseModel

    class _ActivateReq(_BaseModel):
        architect: str = "MoMo"
        strict: bool = False

    class _TaskReq(_BaseModel):
        description: str
        source: str = "node0_standalone_api"
        browser_mode: str = "mock"

    class _QueryReq(_BaseModel):
        prompt: str
        model: str = ""
        max_tokens: int = 1024
        temperature: float = 0.3
        route: str = "direct"  # "direct" = single model, "moe" = 5-expert routing

except ImportError:
    pass  # pydantic unavailable — _cmd_serve will catch it

# Agent → model routing map — single source of truth in constants.py
# Supports Ollama defaults + LM Studio overrides + env var overrides
# 12 agents: 7 PAT (user) + 5 SAT (system) + shared capabilities
try:
    from core.integration.constants import NODE0_MODEL_FLEET
except ImportError:
    # Fallback for standalone execution without core on sys.path
    NODE0_MODEL_FLEET: dict[str, str] = {  # type: ignore[no-redef]
        "P1-Planner": os.environ.get("BIZRA_MODEL_PLANNER", "deepseek-r1:14b"),
        "P2-Researcher": os.environ.get("BIZRA_MODEL_RESEARCHER", "qwen2.5:3b"),
        "P3-Coder": os.environ.get("BIZRA_MODEL_CODER", "mistral:latest"),
        "P4-Evaluator": os.environ.get("BIZRA_MODEL_EVALUATOR", "phi3:mini"),
        "P5-Ethicist": "frozen",
        "P6-Publisher": os.environ.get("BIZRA_MODEL_PUBLISHER", "phi3:mini"),
        "P7-DEMA": os.environ.get("BIZRA_MODEL_DEMA", "deephat-v1-7b"),
        "S1-Sentinel": "pure-code",
        "S2-Oracle": os.environ.get("BIZRA_MODEL_ORACLE", "phi3:mini"),
        "S3-Ledger": "pure-code",
        "S4-Conductor": "pure-code",
        "S5-Ambassador": "pure-code",
        "vision": os.environ.get("BIZRA_MODEL_VISION", "moondream:1.8b"),
        "embedding": os.environ.get("BIZRA_MODEL_EMBED", "nomic-embed-text:latest"),
        "default": os.environ.get("BIZRA_MODEL_DEFAULT", "phi3:mini"),
    }


def create_app(
    manager: "Node0StandaloneManager", api_key: str = ""
) -> Any:
    """Build the standalone FastAPI app for live serving and tests."""
    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as exc:
        raise SystemExit(f"Missing API dependencies: {exc}")

    api_key = api_key.strip()
    app = FastAPI(
        title="BIZRA Node0 Standalone",
        version="1.0.0",
        description="Single-node lifecycle API (activate, health, task)",
    )

    def _require_api_key(x_api_key: str | None) -> None:
        if not api_key:
            return
        if not x_api_key or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="invalid_api_key")

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "BIZRA Node0 Standalone API",
            "version": "1.1.0",
            "endpoints": [
                "GET  /health",
                "GET  /v1/models",
                "GET  /v1/agents",
                "POST /v1/query",
                "POST /activate",
                "POST /task",
                "GET  /assets",
                "GET  /lifecycle",
            ],
        }

    @app.post("/activate")
    async def activate(
        req: _ActivateReq, x_api_key: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_api_key(x_api_key)
        return manager.activate(architect=req.architect, strict=req.strict)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return manager.health()

    @app.get("/assets")
    async def assets(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        return manager.assets()

    @app.get("/lifecycle")
    async def lifecycle(
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_api_key(x_api_key)
        return manager.lifecycle()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """List available local models and agent-to-model routing."""
        ollama_models: list[str] = []
        try:
            import httpx

            ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                resp.raise_for_status()
                ollama_models = [
                    m["name"] for m in resp.json().get("models", [])
                ]
        except Exception:  # noqa: BLE001 - best-effort model listing
            pass
        return {
            "ollama_models": ollama_models,
            "agent_routing": NODE0_MODEL_FLEET,
            "ollama_url": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        }

    @app.get("/v1/agents")
    async def list_agents() -> dict[str, Any]:
        """List the 12-agent organism: 7 PAT (user's team) + 5 SAT (system gates)."""
        pat = [
            {
                "id": "P1-Planner",
                "name": "Planner",
                "role": "Strategic decomposition, goal breakdown",
                "model": NODE0_MODEL_FLEET.get("P1-Planner", "deepseek-r1:14b"),
                "type": "neural",
                "mode": "System 2",
            },
            {
                "id": "P2-Researcher",
                "name": "Researcher",
                "role": "Knowledge retrieval, domain learning",
                "model": NODE0_MODEL_FLEET.get("P2-Researcher", "qwen2.5:3b"),
                "type": "neural",
                "mode": "System 2",
            },
            {
                "id": "P3-Coder",
                "name": "Coder",
                "role": "Executable actions, Telescript generation",
                "model": NODE0_MODEL_FLEET.get("P3-Coder", "mistral:latest"),
                "type": "neural",
                "mode": "System 2",
            },
            {
                "id": "P4-Evaluator",
                "name": "Evaluator",
                "role": "Testing, simulation, outcome scoring",
                "model": NODE0_MODEL_FLEET.get("P4-Evaluator", "phi3:mini"),
                "type": "neural",
                "mode": "System 2",
            },
            {
                "id": "P5-Ethicist",
                "name": "Ethicist",
                "role": "Ihsan scoring, constitutional alignment",
                "model": "frozen",
                "type": "constitutional",
                "mode": "System 2",
            },
            {
                "id": "P6-Publisher",
                "name": "Publisher",
                "role": "Communication, formatting, user-facing output",
                "model": NODE0_MODEL_FLEET.get("P6-Publisher", "phi3:mini"),
                "type": "neural",
                "mode": "System 1",
            },
            {
                "id": "P7-DEMA",
                "name": "DEMA (Integrator)",
                "role": "Synthesis, team coordination, voice persona",
                "model": NODE0_MODEL_FLEET.get("P7-DEMA", "deephat-v1-7b"),
                "type": "neural",
                "mode": "System 2",
                "persona": "Daughter Test personified",
            },
        ]
        sat = [
            {
                "id": "S1-Sentinel",
                "name": "Sentinel",
                "role": "Real-time threat detection",
                "model": "pure-code",
                "type": "gate",
            },
            {
                "id": "S2-Oracle",
                "name": "Oracle",
                "role": "Constitutional reasoning, Shura consensus",
                "model": NODE0_MODEL_FLEET.get("S2-Oracle", "phi3:mini"),
                "type": "gate",
            },
            {
                "id": "S3-Ledger",
                "name": "Ledger",
                "role": "Evidence chain, proof-carrying inference",
                "model": "pure-code",
                "type": "gate",
            },
            {
                "id": "S4-Conductor",
                "name": "Conductor",
                "role": "Event bus routing, agent orchestration",
                "model": "pure-code",
                "type": "router",
            },
            {
                "id": "S5-Ambassador",
                "name": "Ambassador",
                "role": "Federation gossip, inter-node protocol",
                "model": "pure-code",
                "type": "federation",
            },
        ]
        return {"pat": pat, "sat": sat, "total": len(pat) + len(sat)}

    @app.post("/v1/query")
    async def query_llm(
        req: _QueryReq, x_api_key: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """LLM query — direct (single model) or MOE (5-expert routing).

        Set route="moe" to activate MOE Engine routing:
        - Input is scored against 5 experts (R/K/S/G/V)
        - Top-K experts dispatch to their specialized Ollama models
        - Results are synthesized with weighted combination
        - ihsan_tensor tracks expert contributions for learning
        """
        _require_api_key(x_api_key)
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")

        # ── MOE Route: 5-expert multi-model routing ────────────────
        if req.route == "moe":
            return await _query_moe(req)

        # ── Direct Route: single model Ollama call ─────────────────
        model = req.model or NODE0_MODEL_FLEET["default"]
        if model in NODE0_MODEL_FLEET:
            model = NODE0_MODEL_FLEET[model]

        try:
            import httpx

            ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            t0 = time.time()
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": req.prompt,
                        "stream": False,
                        "options": {
                            "temperature": req.temperature,
                            "num_predict": req.max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            latency_ms = (time.time() - t0) * 1000
            return {
                "model": model,
                "response": data.get("response", ""),
                "eval_count": data.get("eval_count", 0),
                "latency_ms": round(latency_ms, 1),
                "route": "direct",
            }
        except ImportError:
            raise HTTPException(status_code=503, detail="httpx not installed")
        except Exception as exc:  # noqa: BLE001 - query boundary
            raise HTTPException(status_code=502, detail=f"LLM query failed: {exc}")

    async def _query_moe(req: _QueryReq) -> dict[str, Any]:
        """MOE-routed query — dispatches to specialized expert models."""
        try:
            from core.sovereign.moe_bridge import MOEBridge

            ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            bridge = MOEBridge.create(ollama_url=ollama_url)

            t0 = time.time()
            response = await bridge.infer(
                req.prompt,
                context={"temperature": req.temperature, "max_tokens": req.max_tokens},
            )
            latency_ms = (time.time() - t0) * 1000

            return {
                "response": response,
                "route": "moe",
                "experts": bridge.last_ihsan_tensor,
                "stats": {
                    "expert_calls": bridge.stats.expert_calls,
                    "expert_failures": bridge.stats.expert_failures,
                    "models_used": bridge.stats.model_usage,
                },
                "latency_ms": round(latency_ms, 1),
            }
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=f"MOE Bridge not available: {type(e).__name__}",
            )
        except Exception as exc:  # noqa: BLE001 - query boundary
            raise HTTPException(
                status_code=502,
                detail=f"MOE query failed: {type(exc).__name__}",
            )

    @app.post("/task")
    async def task(
        req: _TaskReq, x_api_key: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_api_key(x_api_key)
        if not req.description.strip():
            raise HTTPException(status_code=400, detail="description is required")
        return await manager.run_task(
            req.description,
            source=req.source,
            browser_mode=req.browser_mode,
        )

    return app


async def _cmd_serve(args: argparse.Namespace) -> int:
    manager = Node0StandaloneManager()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(f"Missing API dependencies: {exc}")

    api_key = (
        os.environ.get("BIZRA_NODE0_API_KEY") or os.environ.get("BIZRA_API_KEY") or ""
    )
    api_key = api_key.strip()
    if args.host not in {"127.0.0.1", "localhost"} and not api_key:
        raise SystemExit(
            "Refusing non-loopback host without API key. "
            "Set BIZRA_NODE0_API_KEY (or BIZRA_API_KEY)."
        )

    app = create_app(manager, api_key=api_key)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BIZRA Node0 standalone lifecycle manager"
    )
    sub = parser.add_subparsers(dest="command")

    p_activate = sub.add_parser("activate", help="Activate Node0 lifecycle and assets")
    p_activate.add_argument(
        "--architect", default="MoMo", help="Founder/owner display name"
    )
    p_activate.add_argument(
        "--strict",
        action="store_true",
        help="Require every gate (including optional integrations)",
    )

    sub.add_parser("health", help="Show standalone lifecycle health")

    p_task = sub.add_parser("task", help="Run one autonomous task")
    p_task.add_argument("description", help="Mission description")
    p_task.add_argument(
        "--source", default="node0_standalone_cli", help="Task source label"
    )
    p_task.add_argument(
        "--browser-mode",
        choices=["mock", "direct", "mcp"],
        default="mock",
        help="Browser channel mode for mission research.",
    )

    p_serve = sub.add_parser("serve", help="Start local lifecycle API server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8091)

    return parser


async def _dispatch(args: argparse.Namespace) -> int:
    handlers = {
        "activate": _cmd_activate,
        "health": _cmd_health,
        "task": _cmd_task,
        "serve": _cmd_serve,
    }
    command = args.command or "health"
    return await handlers[command](args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    try:
        code = asyncio.run(_dispatch(args))
    except KeyboardInterrupt:
        code = 130

    raise SystemExit(code)


if __name__ == "__main__":
    main()
