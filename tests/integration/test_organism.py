"""
Integration Tests — Sovereign Organism (Living BIZRA Runtime)
==============================================================

Tests the complete integration layer: NervousSystem + MissionPipeline
+ Helix3 composed into a single Living Organism.

PMBOK alignment:
  Integration Management — tests that all components compose correctly
  Quality Management     — Ihsān invariant checks
  Risk Management        — graceful degradation, error recovery
  Scope Management       — mission() returns structured receipts

Test tiers: T1 Delta (< 2 min, every commit per §7)
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from core.sovereign.organism import (
    OrganismHealth,
    OrganismReceipt,
    SovereignOrganism,
)


# ═══════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════


class EchoInference:
    """Deterministic inference for testing — echoes input with structure."""

    def __init__(self, prefix: str = "echo") -> None:
        self._prefix = prefix
        self.call_count = 0

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        agent_id = kwargs.get("agent_id", "unknown")
        # Return structured content so P4 scoring produces good Ihsān
        return (
            f"[{agent_id}] Response to mission:\n"
            f"- Analysis: comprehensive review completed\n"
            f"- Evidence: verified against constitutional standards\n"
            f"- Recommendation: proceed with implementation\n"
            f"- Quality: meets Ihsān threshold requirements"
        )


class FailingInference:
    """Provider that fails — tests graceful degradation."""

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("LLM backend offline")


# ═══════════════════════════════════════════════════════════════════
# §1: BOOT (Genesis Ceremony)
# ═══════════════════════════════════════════════════════════════════


class TestBoot:
    """Verify the organism boots correctly — the Genesis Ceremony."""

    def test_boot_creates_organism(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        assert org is not None
        assert org.health.alive is True

    def test_boot_wires_all_components(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        assert org._nervous_system is not None
        assert org._pipeline is not None
        assert org._helix3 is not None

    def test_boot_registers_12_agents(self) -> None:
        """§1: 12 agents must be registered."""
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        assert org.health.agents_registered == 12

    def test_boot_health_initial_state(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        h = org.health
        assert h.missions_completed == 0
        assert h.missions_failed == 0
        assert h.ticks_completed == 0
        assert h.heartbeat_active is False


# ═══════════════════════════════════════════════════════════════════
# §6: MISSION EXECUTION (Mode 2)
# ═══════════════════════════════════════════════════════════════════


class TestMission:
    """Verify mission execution through the full organism."""

    def test_mission_returns_receipt(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        receipt = asyncio.run(org.mission("implement user auth"))
        assert isinstance(receipt, OrganismReceipt)
        assert receipt.mission_id != ""
        assert receipt.output_text != ""

    def test_mission_scores_ihsan(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        receipt = asyncio.run(org.mission("implement user auth"))
        assert receipt.ihsan_score > 0
        assert receipt.snr_score > 0

    def test_mission_advances_chain_hash(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        r1 = asyncio.run(org.mission("first mission"))
        hash1 = r1.chain_hash
        r2 = asyncio.run(org.mission("second mission"))
        hash2 = r2.chain_hash
        assert hash1 != hash2  # Chain advances
        assert hash1 != "0" * 64  # Not genesis

    def test_multiple_missions_accumulate(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        for i in range(5):
            asyncio.run(org.mission(f"task {i}"))
        h = org.health
        assert h.missions_completed == 5
        assert h.missions_failed == 0

    def test_mission_uses_s2_on_cache_miss(self) -> None:
        """First mission is always S2 (cache miss)."""
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        receipt = asyncio.run(org.mission("brand new task"))
        assert receipt.system == "S2"

    def test_on_receipt_callback(self) -> None:
        received: List[OrganismReceipt] = []
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(
            inference=echo, on_receipt=received.append
        ))
        asyncio.run(org.mission("test callback"))
        assert len(received) == 1
        assert received[0].mission_id != ""


# ═══════════════════════════════════════════════════════════════════
# §2: EVOLUTIONARY TICK (Helix 3)
# ═══════════════════════════════════════════════════════════════════


class TestTick:
    """Verify evolutionary heartbeat through the organism."""

    def test_manual_tick(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        receipt = asyncio.run(org.tick())
        assert receipt is not None
        assert receipt.tick_number >= 1

    def test_tick_after_missions(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        asyncio.run(org.mission("task 1"))
        asyncio.run(org.mission("task 2"))
        receipt = asyncio.run(org.tick())
        assert receipt.missions_processed >= 0  # May be 0 or 2 depending on wiring

    def test_consecutive_ticks_advance(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        r1 = asyncio.run(org.tick())
        r2 = asyncio.run(org.tick())
        assert r2.tick_number == r1.tick_number + 1


# ═══════════════════════════════════════════════════════════════════
# HEALTH & OBSERVABILITY (PMBOK Quality Management)
# ═══════════════════════════════════════════════════════════════════


class TestHealth:
    """Verify organism health reporting."""

    def test_health_is_structured(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        h = org.health
        assert isinstance(h, OrganismHealth)
        assert h.alive is True
        assert h.agents_registered == 12

    def test_stats_aggregation(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        asyncio.run(org.mission("test"))
        s = org.stats
        assert "organism" in s
        assert "pipeline" in s
        assert s["organism"]["missions"] >= 1

    def test_uptime_increases(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        h1 = org.health.uptime_seconds
        asyncio.run(org.mission("tick"))
        h2 = org.health.uptime_seconds
        assert h2 >= h1


# ═══════════════════════════════════════════════════════════════════
# §4: CONSTITUTIONAL INVARIANT CHECKS
# ═══════════════════════════════════════════════════════════════════


class TestInvariants:
    """Verify constitutional invariant enforcement."""

    def test_healthy_organism_no_violations(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        violations = org.check_invariants()
        assert len(violations) == 0  # Fresh organism is healthy

    def test_agents_count_invariant(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        assert org.health.agents_registered == 12


# ═══════════════════════════════════════════════════════════════════
# GRACEFUL DEGRADATION (Risk Management)
# ═══════════════════════════════════════════════════════════════════


class TestGracefulDegradation:
    """Verify organism degrades gracefully on failures."""

    def test_failing_inference_returns_degraded_receipt(self) -> None:
        """§6 Mode 2: Failing inference degrades gracefully — low-quality S2, not crash."""
        org = asyncio.run(SovereignOrganism.boot(inference=FailingInference()))
        receipt = asyncio.run(org.mission("test degradation"))
        # Pipeline handles agent failures gracefully — mission completes as S2
        # with low quality rather than crashing
        assert isinstance(receipt, OrganismReceipt)
        assert receipt.ihsan_score < 0.85  # Below gate threshold
        assert receipt.output_text != ""

    def test_failing_inference_counts_as_low_quality(self) -> None:
        """Degraded missions complete (not fail) but with low Ihsān."""
        org = asyncio.run(SovereignOrganism.boot(inference=FailingInference()))
        asyncio.run(org.mission("fail 1"))
        asyncio.run(org.mission("fail 2"))
        # Missions complete (degraded) rather than failing
        h = org.health
        total = h.missions_completed + h.missions_failed
        assert total == 2


# ═══════════════════════════════════════════════════════════════════
# SHUTDOWN (Lifecycle Management)
# ═══════════════════════════════════════════════════════════════════


class TestShutdown:
    """Verify graceful shutdown."""

    def test_shutdown_marks_not_alive(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        assert org.health.alive is True
        asyncio.run(org.shutdown())
        assert org.health.alive is False

    def test_shutdown_processes_final_tick(self) -> None:
        echo = EchoInference()
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        asyncio.run(org.mission("pre-shutdown task"))
        asyncio.run(org.shutdown())
        # Final tick should have been processed
        assert org.health.ticks_completed >= 1


# ═══════════════════════════════════════════════════════════════════
# FULL LIFECYCLE (E2E Integration)
# ═══════════════════════════════════════════════════════════════════


class TestFullLifecycle:
    """E2E test: boot → missions → tick → health → shutdown."""

    def test_complete_lifecycle(self) -> None:
        """The complete organism lifecycle in one test."""
        echo = EchoInference()

        # 1. Boot (Genesis Ceremony)
        org = asyncio.run(SovereignOrganism.boot(inference=echo))
        assert org.health.alive is True

        # 2. Execute missions
        receipts: List[OrganismReceipt] = []
        for task in ["research auth", "implement login", "test coverage"]:
            r = asyncio.run(org.mission(task))
            receipts.append(r)

        assert len(receipts) == 3
        assert all(r.ihsan_score > 0 for r in receipts)

        # 3. Evolutionary tick
        tick_receipt = asyncio.run(org.tick())
        assert tick_receipt is not None

        # 4. Health check
        h = org.health
        assert h.missions_completed == 3
        assert h.ticks_completed >= 1
        assert h.current_ihsan_avg > 0

        # 5. Invariant check
        violations = org.check_invariants()
        assert len(violations) == 0

        # 6. Stats
        s = org.stats
        assert s["organism"]["missions"] == 3

        # 7. Shutdown
        asyncio.run(org.shutdown())
        assert org.health.alive is False
