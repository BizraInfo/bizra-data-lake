"""
Tests for CROWN Layer — Three-Horizon Governance Invariant Enforcement.

Covers H0 (Ethical), H1 (Performance), H2 (Safety) horizons and
aggregate verdict logic.
"""

from __future__ import annotations

import pytest

from core.governance.crown_layer import (
    CROWNHorizon,
    CROWNLayer,
    CROWNStatus,
    SystemState,
)


@pytest.fixture
def crown() -> CROWNLayer:
    """Default CROWN layer using constitutional thresholds."""
    return CROWNLayer()


# ── Healthy system ──────────────────────────────────────────────────────


def test_all_green_system(crown: CROWNLayer) -> None:
    """All metrics healthy -> PASS, all_green=True."""
    state = SystemState(
        ihsan_score=0.97,
        snr_score=0.92,
        gini_coefficient=0.20,
        latency_ms=5000.0,
        has_riba=False,
        has_gharar=False,
        is_reversible=True,
        human_override_available=True,
        has_audit_trail=True,
    )
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.PASS
    assert verdict.all_green is True
    assert not verdict.halted
    assert verdict.warnings == []
    assert len(verdict.horizons) == 3


# ── H0 Ethical horizon ─────────────────────────────────────────────────


def test_riba_triggers_halt(crown: CROWNLayer) -> None:
    """Riba detection triggers immediate H0 HALT."""
    state = SystemState(has_riba=True)
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.HALT
    assert verdict.halted
    h0 = verdict.horizons[0]
    assert h0.horizon == CROWNHorizon.H0_ETHICAL
    assert h0.status == CROWNStatus.HALT
    assert "Riba" in h0.details


def test_gharar_triggers_halt(crown: CROWNLayer) -> None:
    """Gharar detection triggers immediate H0 HALT."""
    state = SystemState(has_gharar=True)
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.HALT
    assert verdict.halted
    h0 = verdict.horizons[0]
    assert h0.horizon == CROWNHorizon.H0_ETHICAL
    assert h0.status == CROWNStatus.HALT
    assert "Gharar" in h0.details


def test_low_ihsan_triggers_halt(crown: CROWNLayer) -> None:
    """Ihsan below threshold triggers H0 HALT."""
    state = SystemState(ihsan_score=0.80)
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.HALT
    h0 = verdict.horizons[0]
    assert h0.status == CROWNStatus.HALT
    assert "Ihsan" in h0.details
    assert h0.metrics["ihsan_score"] == 0.80


def test_high_gini_triggers_halt(crown: CROWNLayer) -> None:
    """Gini above threshold triggers H0 HALT."""
    state = SystemState(gini_coefficient=0.50)
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.HALT
    h0 = verdict.horizons[0]
    assert h0.status == CROWNStatus.HALT
    assert "Gini" in h0.details
    assert h0.metrics["gini_coefficient"] == 0.50


# ── H1 Performance horizon ─────────────────────────────────────────────


def test_high_latency_triggers_warn(crown: CROWNLayer) -> None:
    """Latency exceeding SLA triggers H1 WARN."""
    state = SystemState(latency_ms=50000.0)
    verdict = crown.render_verdict(state)

    h1 = verdict.horizons[1]
    assert h1.horizon == CROWNHorizon.H1_PERFORMANCE
    assert h1.status == CROWNStatus.WARN
    assert "Latency" in h1.details
    assert h1.metrics["latency_ms"] == 50000.0


def test_low_snr_triggers_warn(crown: CROWNLayer) -> None:
    """SNR below threshold triggers H1 WARN."""
    state = SystemState(snr_score=0.70)
    verdict = crown.render_verdict(state)

    h1 = verdict.horizons[1]
    assert h1.horizon == CROWNHorizon.H1_PERFORMANCE
    assert h1.status == CROWNStatus.WARN
    assert "SNR" in h1.details
    assert h1.metrics["snr_score"] == 0.70


# ── H2 Safety horizon ──────────────────────────────────────────────────


def test_irreversible_action_triggers_halt(crown: CROWNLayer) -> None:
    """Irreversible action triggers H2 HALT."""
    state = SystemState(is_reversible=False)
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.HALT
    h2 = verdict.horizons[2]
    assert h2.horizon == CROWNHorizon.H2_SAFETY
    assert h2.status == CROWNStatus.HALT
    assert "Irreversible" in h2.details


def test_missing_human_override_triggers_warn(crown: CROWNLayer) -> None:
    """Missing human override triggers H2 WARN."""
    state = SystemState(human_override_available=False)
    verdict = crown.render_verdict(state)

    h2 = verdict.horizons[2]
    assert h2.horizon == CROWNHorizon.H2_SAFETY
    assert h2.status == CROWNStatus.WARN
    assert "Human override" in h2.details


def test_missing_audit_trail_triggers_warn(crown: CROWNLayer) -> None:
    """Missing audit trail triggers H2 WARN."""
    state = SystemState(has_audit_trail=False)
    verdict = crown.render_verdict(state)

    h2 = verdict.horizons[2]
    assert h2.horizon == CROWNHorizon.H2_SAFETY
    assert h2.status == CROWNStatus.WARN
    assert "audit trail" in h2.details


# ── Aggregate logic ────────────────────────────────────────────────────


def test_worst_status_propagates(crown: CROWNLayer) -> None:
    """Overall status is the worst across all horizons (H0=PASS, H1=WARN, H2=HALT)."""
    state = SystemState(
        ihsan_score=0.97,       # H0 PASS
        latency_ms=50000.0,     # H1 WARN
        is_reversible=False,    # H2 HALT
    )
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.HALT
    assert verdict.halted
    assert not verdict.all_green
    assert verdict.horizons[0].status == CROWNStatus.PASS
    assert verdict.horizons[1].status == CROWNStatus.WARN
    assert verdict.horizons[2].status == CROWNStatus.HALT


def test_none_fields_skipped(crown: CROWNLayer) -> None:
    """SystemState with all None fields results in all PASS (nothing to check)."""
    state = SystemState()
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.PASS
    assert verdict.all_green is True
    for h in verdict.horizons:
        assert h.status == CROWNStatus.PASS


def test_custom_thresholds() -> None:
    """CROWNLayer with custom thresholds uses the provided values."""
    crown = CROWNLayer(ihsan_threshold=0.99)
    # 0.96 passes default (0.95) but fails custom (0.99)
    state = SystemState(ihsan_score=0.96)
    verdict = crown.render_verdict(state)

    assert verdict.status == CROWNStatus.HALT
    h0 = verdict.horizons[0]
    assert h0.status == CROWNStatus.HALT
    assert "Ihsan" in h0.details


def test_halted_property(crown: CROWNLayer) -> None:
    """verdict.halted is True when status is HALT."""
    state = SystemState(has_riba=True)
    verdict = crown.render_verdict(state)
    assert verdict.halted is True

    clean_state = SystemState(ihsan_score=0.97)
    clean_verdict = crown.render_verdict(clean_state)
    assert clean_verdict.halted is False


def test_warnings_property(crown: CROWNLayer) -> None:
    """verdict.warnings returns only the WARN horizon verdicts."""
    state = SystemState(
        latency_ms=50000.0,           # H1 WARN
        human_override_available=False,  # H2 WARN
    )
    verdict = crown.render_verdict(state)

    assert len(verdict.warnings) == 2
    horizons_with_warn = {w.horizon for w in verdict.warnings}
    assert CROWNHorizon.H1_PERFORMANCE in horizons_with_warn
    assert CROWNHorizon.H2_SAFETY in horizons_with_warn
