from __future__ import annotations

import pytest

from core.voice.personaplex_bridge import PersonaPlexBridge, VoiceOutput, VoicePersona


def test_voice_persona_defaults() -> None:
    persona = VoicePersona("Guardian", "NATF3.pt", "Protective")
    assert persona.guardian_name == "Guardian"
    assert persona.voice_code.endswith(".pt")
    assert persona.ihsan_floor == pytest.approx(0.95)


def test_bridge_has_expected_personas() -> None:
    keys = set(PersonaPlexBridge.PERSONAS.keys())
    assert keys == {
        "strategist",
        "researcher",
        "creator",
        "executor",
        "guardian",
        "coordinator",
        "analyst",
    }


def test_ihsan_gate_blocks_harmful_text() -> None:
    bridge = PersonaPlexBridge()
    persona = PersonaPlexBridge.PERSONAS["guardian"]
    passed, score = bridge.ihsan_gate("We should attack and harm people", persona)
    assert passed is False
    assert score < persona.ihsan_floor


def test_ihsan_gate_passes_safe_text() -> None:
    bridge = PersonaPlexBridge()
    persona = PersonaPlexBridge.PERSONAS["researcher"]
    passed, score = bridge.ihsan_gate(
        "Provide safe evidence-backed outreach guidance", persona
    )
    assert passed is True
    assert score >= persona.ihsan_floor


@pytest.mark.asyncio
async def test_speak_returns_blocked_output_when_ihsan_fails() -> None:
    bridge = PersonaPlexBridge()
    result = await bridge.speak("kill and hate", guardian="guardian")
    assert isinstance(result, VoiceOutput)
    assert result.ihsan_passed is False
    assert result.audio_data == b""
    assert result.tier == "blocked"


@pytest.mark.asyncio
async def test_speak_returns_voice_output_in_noop_or_engine_mode() -> None:
    bridge = PersonaPlexBridge()
    result = await bridge.speak(
        "Mission status: all channels completed with verifiable evidence",
        guardian="coordinator",
    )
    assert isinstance(result, VoiceOutput)
    assert result.ihsan_passed is True
    assert result.guardian
    assert result.tier in {"personaplex", "coqui", "noop"}


@pytest.mark.asyncio
async def test_speak_unknown_guardian_falls_back_to_coordinator() -> None:
    bridge = PersonaPlexBridge()
    result = await bridge.speak("Fallback guardian check", guardian="unknown")
    assert result.guardian == PersonaPlexBridge.PERSONAS["coordinator"].guardian_name
