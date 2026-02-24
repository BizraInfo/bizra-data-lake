"""Mission decomposition and four-channel dispatch orchestration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Channel(str, Enum):
    DESKTOP = "desktop"
    BROWSER = "browser"
    VOICE = "voice"
    PROOF = "proof"


@dataclass
class SubTask:
    id: str
    description: str
    channel: Channel
    agent: str = "coordinator"
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None


@dataclass
class MissionPlan:
    mission_id: str
    subtasks: list[SubTask]


class ChannelDispatcher:
    """Routes mission subtasks to Desktop, Browser, Voice, and Proof channels."""

    _KEYWORDS: dict[Channel, tuple[str, ...]] = {
        Channel.BROWSER: (
            "browse",
            "find",
            "lookup",
            "portfolio",
            "research",
            "search",
            "vc",
            "web",
        ),
        Channel.DESKTOP: (
            "create",
            "desktop",
            "draft",
            "email",
            "file",
            "folder",
            "organize",
            "write",
        ),
        Channel.VOICE: (
            "brief",
            "narrate",
            "speak",
            "summary",
            "voice",
        ),
        Channel.PROOF: (
            "capture",
            "evidence",
            "proof",
            "record",
            "screen",
        ),
    }

    def __init__(
        self,
        desktop_bridge: Any = None,
        browser_client: Any = None,
        voice_bridge: Any = None,
        obs_trigger: Any = None,
    ) -> None:
        self._desktop = desktop_bridge
        self._browser = browser_client
        self._voice = voice_bridge
        self._proof = obs_trigger
        self._lazy_ready = False

    def decompose(
        self,
        mission_id: str,
        description: str,
        agent_results: list[dict[str, Any]] | None = None,
    ) -> MissionPlan:
        """Create channel-routed subtasks with dependency edges."""
        lower = description.lower()
        subtasks: list[SubTask] = []

        if self._needs_channel(Channel.PROOF, lower):
            subtasks.append(
                SubTask(
                    id=f"{mission_id}-proof-start",
                    description="Start proof recording",
                    channel=Channel.PROOF,
                    params={"action": "start"},
                )
            )

        if self._needs_channel(Channel.BROWSER, lower):
            subtasks.append(
                SubTask(
                    id=f"{mission_id}-browser",
                    description="Browser research",
                    channel=Channel.BROWSER,
                    agent="researcher",
                    params={"query": description},
                )
            )

        if self._needs_channel(Channel.DESKTOP, lower):
            browser_ids = [
                task.id for task in subtasks if task.channel is Channel.BROWSER
            ]
            subtasks.append(
                SubTask(
                    id=f"{mission_id}-desktop",
                    description="Desktop execution",
                    channel=Channel.DESKTOP,
                    agent="executor",
                    params={"description": description},
                    depends_on=browser_ids,
                )
            )

        if self._needs_channel(Channel.VOICE, lower):
            deps = [
                task.id
                for task in subtasks
                if task.channel in {Channel.BROWSER, Channel.DESKTOP}
            ]
            narration = self._compose_narration(description, agent_results)
            subtasks.append(
                SubTask(
                    id=f"{mission_id}-voice",
                    description="Voice narration",
                    channel=Channel.VOICE,
                    agent="coordinator",
                    params={"text": narration},
                    depends_on=deps,
                )
            )

        if any(task.channel is Channel.PROOF for task in subtasks):
            deps = [task.id for task in subtasks if task.channel is not Channel.PROOF]
            subtasks.append(
                SubTask(
                    id=f"{mission_id}-proof-stop",
                    description="Stop proof recording",
                    channel=Channel.PROOF,
                    params={"action": "stop"},
                    depends_on=deps,
                )
            )

        if not subtasks:
            subtasks.append(
                SubTask(
                    id=f"{mission_id}-browser-default",
                    description="Default research",
                    channel=Channel.BROWSER,
                    agent="researcher",
                    params={"query": description},
                )
            )

        return MissionPlan(mission_id=mission_id, subtasks=subtasks)

    async def dispatch_all(self, plan: MissionPlan) -> dict[str, dict[str, Any]]:
        """Dispatch all tasks while honoring dependency constraints."""
        await self._ensure_lazy_init()

        pending = list(plan.subtasks)
        completed: set[str] = set()
        results: dict[str, dict[str, Any]] = {}

        for _ in range(len(plan.subtasks) + 1):
            if not pending:
                break

            ready = [
                task
                for task in pending
                if all(dep in completed for dep in task.depends_on)
            ]
            if not ready:
                ready = pending[:1]

            executions = [self._dispatch_one(task) for task in ready]
            resolved = await asyncio.gather(*executions, return_exceptions=True)

            for task, outcome in zip(ready, resolved):
                if isinstance(outcome, Exception):
                    result = {
                        "success": False,
                        "channel": task.channel.value,
                        "warning": f"exception:{type(outcome).__name__}",
                    }
                else:
                    result = outcome

                task.result = result
                results[task.id] = result
                completed.add(task.id)
                pending.remove(task)

        return results

    async def _dispatch_one(self, task: SubTask) -> dict[str, Any]:
        if task.channel is Channel.DESKTOP:
            return await self._dispatch_desktop(task)
        if task.channel is Channel.BROWSER:
            return await self._dispatch_browser(task)
        if task.channel is Channel.VOICE:
            return await self._dispatch_voice(task)
        if task.channel is Channel.PROOF:
            return await self._dispatch_proof(task)

        return {
            "success": False,
            "channel": task.channel.value,
            "warning": "unknown_channel",
        }

    async def _dispatch_desktop(self, task: SubTask) -> dict[str, Any]:
        bridge = self._desktop
        if bridge is None:
            return {
                "success": False,
                "channel": Channel.DESKTOP.value,
                "warning": "desktop_unavailable",
            }

        try:
            payload = {"query": task.params.get("description", task.description)}
            if hasattr(bridge, "send_command"):
                result = await bridge.send_command("sovereign_query", payload)
            elif hasattr(bridge, "dispatch"):
                result = await bridge.dispatch(payload)
            else:
                return {
                    "success": False,
                    "channel": Channel.DESKTOP.value,
                    "warning": "desktop_interface_mismatch",
                }
            return {
                "success": True,
                "channel": Channel.DESKTOP.value,
                "result": result,
            }
        except Exception as exc:
            return {
                "success": False,
                "channel": Channel.DESKTOP.value,
                "warning": f"desktop_error:{type(exc).__name__}",
            }

    async def _dispatch_browser(self, task: SubTask) -> dict[str, Any]:
        if self._browser is None:
            return {
                "success": False,
                "channel": Channel.BROWSER.value,
                "warning": "browser_unavailable",
            }

        try:
            query = str(task.params.get("query", task.description))
            result = await self._browser.research(query)
            return {
                "success": True,
                "channel": Channel.BROWSER.value,
                "result": result,
            }
        except Exception as exc:
            return {
                "success": False,
                "channel": Channel.BROWSER.value,
                "warning": f"browser_error:{type(exc).__name__}",
            }

    async def _dispatch_voice(self, task: SubTask) -> dict[str, Any]:
        if self._voice is None:
            return {
                "success": False,
                "channel": Channel.VOICE.value,
                "warning": "voice_unavailable",
            }

        try:
            text = str(task.params.get("text", task.description))
            output = await self._voice.speak(text=text, guardian=task.agent)
            return {
                "success": output.ihsan_passed,
                "channel": Channel.VOICE.value,
                "guardian": output.guardian,
                "duration": output.duration,
                "ihsan_score": output.ihsan_score,
                "tier": output.tier,
                "warning": output.warning,
            }
        except Exception as exc:
            return {
                "success": False,
                "channel": Channel.VOICE.value,
                "warning": f"voice_error:{type(exc).__name__}",
            }

    async def _dispatch_proof(self, task: SubTask) -> dict[str, Any]:
        if self._proof is None:
            return {
                "success": False,
                "channel": Channel.PROOF.value,
                "warning": "proof_unavailable",
            }

        action = str(task.params.get("action", "")).lower()
        try:
            if action in {"start", "start_recording"}:
                await self._proof.connect()
                ok = await self._proof.start_recording()
            elif action in {"stop", "stop_recording"}:
                ok = await self._proof.stop_recording()
                await self._proof.disconnect()
            else:
                ok = False

            return {
                "success": bool(ok),
                "channel": Channel.PROOF.value,
                "action": action,
                "warning": "" if ok else "proof_action_failed",
            }
        except Exception as exc:
            return {
                "success": False,
                "channel": Channel.PROOF.value,
                "action": action,
                "warning": f"proof_error:{type(exc).__name__}",
            }

    async def _ensure_lazy_init(self) -> None:
        if self._lazy_ready:
            return

        self._lazy_ready = True

        if self._browser is None:
            try:
                from core.bridges.browser_mcp_client import BrowserMCPClient

                self._browser = BrowserMCPClient(mode="mock")
            except Exception as exc:
                logger.debug("Browser channel unavailable: %s", exc)

        if self._voice is None:
            try:
                from core.voice.personaplex_bridge import PersonaPlexBridge

                self._voice = PersonaPlexBridge()
            except Exception as exc:
                logger.debug("Voice channel unavailable: %s", exc)

        if self._proof is None:
            try:
                from core.bridges.obs_trigger import OBSTrigger

                self._proof = OBSTrigger()
            except Exception as exc:
                logger.debug("Proof channel unavailable: %s", exc)

    def _needs_channel(self, channel: Channel, description: str) -> bool:
        tokens = self._KEYWORDS[channel]
        return any(token in description for token in tokens)

    @staticmethod
    def _compose_narration(
        description: str,
        agent_results: list[dict[str, Any]] | None,
    ) -> str:
        if not agent_results:
            return description

        snippets = []
        for item in agent_results:
            name = item.get("name") or item.get("agent") or "agent"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            snippets.append(f"{name}: {content[:180]}")

        if not snippets:
            return description

        joined = " ".join(snippets)
        return f"Mission summary: {description}. Agent outputs: {joined}"


__all__ = ["Channel", "ChannelDispatcher", "MissionPlan", "SubTask"]
