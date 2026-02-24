from __future__ import annotations

import pytest

from core.bridges.channel_dispatcher import Channel, ChannelDispatcher, MissionPlan, SubTask


class _VoiceStub:
    async def speak(self, text: str, guardian: str):
        class _Out:
            ihsan_passed = True
            guardian = "Stub Guardian"
            duration = 0.12
            ihsan_score = 0.99
            tier = "stub"
            warning = ""

        assert text
        assert guardian
        return _Out()


@pytest.mark.asyncio
async def test_dispatcher_enum_values() -> None:
    assert Channel.DESKTOP.value == "desktop"
    assert Channel.BROWSER.value == "browser"
    assert Channel.VOICE.value == "voice"
    assert Channel.PROOF.value == "proof"


@pytest.mark.asyncio
async def test_decompose_multi_channel_plan() -> None:
    dispatcher = ChannelDispatcher()
    plan = dispatcher.decompose(
        "m-001",
        "Research VCs, draft files, narrate summary, and record proof",
    )
    channels = {subtask.channel for subtask in plan.subtasks}
    assert channels == {Channel.BROWSER, Channel.DESKTOP, Channel.VOICE, Channel.PROOF}


@pytest.mark.asyncio
async def test_decompose_adds_desktop_dependency_on_browser() -> None:
    dispatcher = ChannelDispatcher()
    plan = dispatcher.decompose("m-002", "Research then draft files")

    browser_ids = [task.id for task in plan.subtasks if task.channel is Channel.BROWSER]
    desktop_tasks = [task for task in plan.subtasks if task.channel is Channel.DESKTOP]

    assert browser_ids
    assert desktop_tasks
    assert browser_ids[0] in desktop_tasks[0].depends_on


@pytest.mark.asyncio
async def test_dispatch_all_graceful_degradation_without_adapters() -> None:
    dispatcher = ChannelDispatcher(browser_client=None, voice_bridge=None, obs_trigger=None)
    plan = MissionPlan(
        mission_id="m-003",
        subtasks=[SubTask(id="t1", description="proof", channel=Channel.PROOF)],
    )

    results = await dispatcher.dispatch_all(plan)
    assert "t1" in results
    assert results["t1"]["success"] is False
    assert "warning" in results["t1"]


@pytest.mark.asyncio
async def test_dispatch_voice_with_stub() -> None:
    dispatcher = ChannelDispatcher(voice_bridge=_VoiceStub())
    plan = MissionPlan(
        mission_id="m-004",
        subtasks=[
            SubTask(
                id="voice-1",
                description="Narrate summary",
                channel=Channel.VOICE,
                agent="coordinator",
                params={"text": "Narration text"},
            )
        ],
    )

    results = await dispatcher.dispatch_all(plan)
    assert results["voice-1"]["success"] is True
    assert results["voice-1"]["channel"] == "voice"
