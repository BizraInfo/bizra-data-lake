from __future__ import annotations

from core.sovereign.equalizer_agent import (
    EqualizerAgent,
    EqualizerCommandKind,
    HaltReason,
    OperationalMode,
    PatternCandidate,
)


def test_equalizer_saturation_triggers_halt() -> None:
    eq = EqualizerAgent()
    eq.observe(layer=1, ihsan_score=0.80, backlog=200, presence=0)

    assert eq.detect_mode() is OperationalMode.SATURATION
    cmd = eq.next_command()
    assert cmd is not None
    assert cmd.kind is EqualizerCommandKind.HALT
    assert cmd.reason == HaltReason.IHSAN_CRITICAL.value


def test_equalizer_accumulation_presence_routes_escalate() -> None:
    eq = EqualizerAgent()
    eq.pending_patterns.append(
        PatternCandidate(hash32=b"A" * 32, snr=0.80, ihsan=0.90, downstream_blocked=2)
    )
    eq.pending_patterns.append(
        PatternCandidate(hash32=b"B" * 32, snr=0.70, ihsan=0.85, downstream_blocked=7)
    )

    eq.observe(layer=1, ihsan_score=0.98, backlog=1, presence=255)
    eq.observe(layer=1, ihsan_score=0.88, backlog=40, presence=255)

    assert eq.detect_mode() is OperationalMode.ACCUMULATION
    cmd = eq.next_command()
    assert cmd is not None
    assert cmd.kind is EqualizerCommandKind.ESCALATE
    assert cmd.pattern_hash == b"B" * 32


def test_equalizer_recovery_routes_resume() -> None:
    eq = EqualizerAgent()
    eq.observe(layer=1, ihsan_score=0.88, backlog=50, presence=200)
    eq.observe(layer=1, ihsan_score=0.93, backlog=20, presence=200)

    assert eq.detect_mode() is OperationalMode.RECOVERY
    cmd = eq.next_command()
    assert cmd is not None
    assert cmd.kind is EqualizerCommandKind.RESUME


def test_equalizer_flow_no_command() -> None:
    eq = EqualizerAgent()
    eq.observe(layer=1, ihsan_score=0.99, backlog=2, presence=120)

    assert eq.detect_mode() is OperationalMode.FLOW
    assert eq.next_command() is None
