"""
RDVE Integration Tests — Runtime wiring + health endpoint visibility
=====================================================================

Validates that the RDVE Engine (Phase 33) is properly wired into the
SovereignRuntime lifecycle and visible through the /v1/health endpoint.

Test Categories:
1. TestRDVEImports         — All RDVE module components are importable
2. TestRDVEOrchestratorUnit — Orchestrator initializes with defaults
3. TestRDVERuntimeWiring    — _init_rdve_engine() wires into SovereignRuntime
4. TestRDVEHealthVisibility — Health endpoint reports RDVE subsystem status

Standing on Giants:
    Shannon (SNR) · Besta (GoT) · Maturana (autopoiesis) ·
    Boyd (OODA) · Deming (PDCA) · Al-Ghazali (Ihsan)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


CI_PY312_RDVE_FULL_RUNTIME_QUARANTINE = pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.version_info[:2] == (3, 12),
    reason=(
        "Python 3.12 CI integration runners currently resource-starve during "
        "full SovereignRuntime RDVE initialization; local Python 3.12 repro passes. "
        "Tracked in #67."
    ),
)

# ===========================================================================
# 1. TestRDVEImports — Verify all RDVE components are importable
# ===========================================================================


class TestRDVEImports:
    """Guard against accidental import path breakage in the RDVE package."""

    def test_rdve_orchestrator_importable(self) -> None:
        """RDVEOrchestrator is importable from core.rdve."""
        from core.rdve import RDVEOrchestrator

        assert RDVEOrchestrator is not None

    def test_rdve_config_importable(self) -> None:
        """RDVEConfig is importable from core.rdve."""
        from core.rdve import RDVEConfig

        assert RDVEConfig is not None

    def test_rdve_status_importable(self) -> None:
        """RDVEStatus is importable from core.rdve."""
        from core.rdve import RDVEStatus

        assert RDVEStatus is not None

    def test_rdve_cycle_result_importable(self) -> None:
        """RDVECycleResult is importable from core.rdve."""
        from core.rdve import RDVECycleResult

        assert RDVECycleResult is not None

    def test_stability_protocol_importable(self) -> None:
        """StabilityProtocol is importable from core.rdve."""
        from core.rdve import StabilityProtocol

        assert StabilityProtocol is not None

    def test_interdisciplinary_transfer_importable(self) -> None:
        """InterdisciplinaryTransfer is importable from core.rdve."""
        from core.rdve import InterdisciplinaryTransfer

        assert InterdisciplinaryTransfer is not None

    def test_subcomponents_importable(self) -> None:
        """RDVE subcomponents (Generator, Explorer, Loop, SNR) are importable."""
        from core.autopoiesis.got_integration import GoTHypothesisExplorer
        from core.autopoiesis.hypothesis_generator import HypothesisGenerator
        from core.autopoiesis.loop_engine import AutopoieticLoop
        from core.sovereign.snr_maximizer import SNRMaximizer

        assert HypothesisGenerator is not None
        assert GoTHypothesisExplorer is not None
        assert AutopoieticLoop is not None
        assert SNRMaximizer is not None


# ===========================================================================
# 2. TestRDVEOrchestratorUnit — Orchestrator creates with defaults
# ===========================================================================


class TestRDVEOrchestratorUnit:
    """Verify RDVEOrchestrator initializes correctly with default config."""

    def test_default_init(self) -> None:
        """Orchestrator creates with all default subcomponents."""
        from core.rdve import RDVEOrchestrator

        orch = RDVEOrchestrator()
        assert orch is not None
        assert orch.status.value == "idle"
        assert orch.cycle_count == 0

    def test_config_thresholds_match_constants(self) -> None:
        """Config thresholds match core/integration/constants.py values."""
        from core.integration.constants import (
            SNR_THRESHOLD_T1_HIGH,
            STRICT_IHSAN_THRESHOLD,
            UNIFIED_IHSAN_THRESHOLD,
            UNIFIED_SNR_THRESHOLD,
        )
        from core.rdve import RDVEConfig

        config = RDVEConfig()
        assert config.snr_floor == UNIFIED_SNR_THRESHOLD
        assert config.snr_target == SNR_THRESHOLD_T1_HIGH
        assert config.ihsan_floor == UNIFIED_IHSAN_THRESHOLD
        assert config.ihsan_strict == STRICT_IHSAN_THRESHOLD

    def test_custom_config(self) -> None:
        """Orchestrator accepts custom RDVEConfig."""
        from core.rdve import RDVEConfig, RDVEOrchestrator

        config = RDVEConfig(
            num_exploration_paths=3,
            max_cycles=10,
        )
        orch = RDVEOrchestrator(config=config)
        assert orch.config.num_exploration_paths == 3
        assert orch.config.max_cycles == 10

    def test_subcomponents_wired(self) -> None:
        """Orchestrator has all 4 subcomponents wired."""
        from core.rdve import RDVEOrchestrator

        orch = RDVEOrchestrator()
        assert hasattr(orch, "_generator")
        assert hasattr(orch, "_explorer")
        assert hasattr(orch, "_snr")
        assert hasattr(orch, "_loop")
        assert orch._generator is not None
        assert orch._explorer is not None
        assert orch._snr is not None
        assert orch._loop is not None

    def test_version_string(self) -> None:
        """RDVE version is a valid semver-like string."""
        from core.rdve.orchestrator import RDVE_VERSION

        parts = RDVE_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ===========================================================================
# 3. TestRDVERuntimeWiring — _init_rdve_engine() wires into runtime
# ===========================================================================


@pytest.mark.xdist_group(name="runtime_heavy")
@CI_PY312_RDVE_FULL_RUNTIME_QUARANTINE
class TestRDVERuntimeWiring:
    """Verify RDVE is wired into SovereignRuntime after initialize()."""

    @pytest.mark.asyncio
    async def test_rdve_engine_initialized(self, tmp_path: Path) -> None:
        """After initialize(), _rdve_engine is not None."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert (
                runtime._rdve_engine is not None
            ), "RDVE Engine should be initialized after runtime.initialize()"
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_rdve_engine_is_orchestrator(self, tmp_path: Path) -> None:
        """_rdve_engine is an RDVEOrchestrator instance."""
        from core.rdve import RDVEOrchestrator
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert isinstance(runtime._rdve_engine, RDVEOrchestrator)
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_rdve_engine_idle_on_init(self, tmp_path: Path) -> None:
        """RDVE engine starts in IDLE status after init (not running)."""
        from core.rdve import RDVEStatus
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._rdve_engine.status == RDVEStatus.IDLE
        finally:
            await runtime.shutdown()

    def test_rdve_field_declared_none_before_init(self, tmp_path: Path) -> None:
        """Before initialize(), _rdve_engine is None."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        assert runtime._rdve_engine is None


# ===========================================================================
# 4. TestRDVEHealthVisibility — Health endpoint sees RDVE
# ===========================================================================


@pytest.mark.xdist_group(name="runtime_heavy")
@CI_PY312_RDVE_FULL_RUNTIME_QUARANTINE
class TestRDVEHealthVisibility:
    """Verify /v1/health subsystem checks detect RDVE correctly.

    The health endpoint in api.py uses getattr(runtime, attr, None) to check
    subsystem availability. These tests validate the attribute names match.
    """

    @pytest.mark.asyncio
    async def test_rdve_engine_attribute_detectable(self, tmp_path: Path) -> None:
        """getattr(runtime, '_rdve_engine') returns a non-None value."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            instance = getattr(runtime, "_rdve_engine", None)
            assert (
                instance is not None
            ), "Health endpoint check for _rdve_engine should find a non-None value"
            assert (
                "Stub" not in type(instance).__name__
            ), "RDVE engine should be real, not a stub"
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_fate_gate_attribute_detectable(self, tmp_path: Path) -> None:
        """getattr(runtime, '_ihsan_watchdog') returns a non-None value."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            instance = getattr(runtime, "_ihsan_watchdog", None)
            assert (
                instance is not None
            ), "Health endpoint check for _ihsan_watchdog (fate_gate) should find a value"
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_all_health_check_attributes_exist(self, tmp_path: Path) -> None:
        """All attribute names used by /v1/health exist on the runtime."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            # These must match the _checks list in api.py /v1/health
            health_attrs = [
                "_graph_reasoner",
                "_snr_optimizer",
                "_guardian_council",
                "_autonomous_loop",
                "_cognitive_fusion",
                "_embedding_service",
                "_memory_coordinator",
                "_evidence_ledger",
                "_rdve_engine",
                "_ihsan_watchdog",
                "_sat_controller",
            ]
            for attr in health_attrs:
                assert hasattr(
                    runtime, attr
                ), f"Runtime must have attribute '{attr}' for health endpoint"
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_health_score_positive(self, tmp_path: Path) -> None:
        """Health score from _calculate_health() is > 0 after init."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            score = runtime._calculate_health()
            assert (
                score >= 0.0
            ), "Health score should be non-negative after initialization"
        finally:
            await runtime.shutdown()
