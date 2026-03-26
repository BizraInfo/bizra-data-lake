from __future__ import annotations

import asyncio
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MissionPlan:
    macro_state: str
    steps: list[str]
    snr: float
    poi_score: float


class MissionBridge:
    """Connect node-gateway miss path to core sovereign mission pipeline."""

    def __init__(self) -> None:
        self._orch = None

    @staticmethod
    def _resolve_workspace_root() -> str:
        configured = (os.environ.get("BIZRA_MISSION_WORKSPACE_ROOT") or "").strip()
        if configured:
            return configured

        cwd = Path.cwd().resolve()
        if (cwd / "core").exists():
            return str(cwd)

        for candidate in Path(__file__).resolve().parents:
            if (candidate / "core").exists():
                return str(candidate)

        return str(cwd)

    @staticmethod
    def _enabled() -> bool:
        flag = (os.environ.get("BIZRA_MISSION_BRIDGE_ENABLED") or "1").strip().lower()
        return flag in ("1", "true", "yes", "on")

    async def run(
        self, text: str, context: dict[str, str], macro_state: str
    ) -> MissionPlan | None:
        if not self._enabled():
            return None

        try:
            from core.sovereign.mission import (
                DesktopContext,
                MissionOrchestrator,
                MissionRequest,
            )
        except Exception:
            return None

        if self._orch is None:
            self._orch = MissionOrchestrator(
                {
                    "memory_path": os.environ.get(
                        "BIZRA_MISSION_MEMORY_PATH", "/tmp/bizra-mission/memory"
                    ),
                    "evidence_path": os.environ.get(
                        "BIZRA_MISSION_EVIDENCE_PATH",
                        "/tmp/bizra-mission/evidence.jsonl",
                    ),
                    "hda_port": int(os.environ.get("BIZRA_HDA_PORT", "9743")),
                    "workspace_root": self._resolve_workspace_root(),
                }
            )
            await self._orch.initialize()

        request = MissionRequest(
            mission_id=secrets.token_hex(16),
            description=text,
            context=DesktopContext(
                active_window_title=str(context.get("active_window", "unknown")),
                clipboard_text=str(context.get("clipboard", "")),
                screen_geometry={},
            ),
            timestamp=time.time(),
            source="node_gateway",
        )

        timeout_s = float(os.environ.get("BIZRA_MISSION_TIMEOUT_SEC", "25"))
        result = await asyncio.wait_for(self._orch.execute(request), timeout=timeout_s)

        steps = self._derive_steps(
            result.synthesis, result.channels_executed, result.evidence_receipt_id
        )
        poi = max(0.0, min((result.ihsan_score + result.snr_score) / 2.0, 1.0))

        return MissionPlan(
            macro_state=macro_state,
            steps=steps,
            snr=max(0.0, min(float(result.snr_score), 1.0)),
            poi_score=poi,
        )

    @staticmethod
    def _derive_steps(
        synthesis: str, channels_executed: list, receipt_id: str
    ) -> list[str]:
        steps: list[str] = []

        for channel in channels_executed:
            status = "ok" if getattr(channel, "success", False) else "failed"
            channel_name = getattr(channel, "channel", "unknown")
            steps.append(f"Run {channel_name} channel ({status})")

        if not steps:
            steps.append("Execute mission channels")

        if synthesis:
            steps.append("Synthesize briefing with constitutional gate checks")

        if receipt_id:
            steps.append(f"Emit evidence receipt {receipt_id}")

        return steps
