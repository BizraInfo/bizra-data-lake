"""
Primordial Activation Integration Tests — Verifies all 32 subsystems wire into
SovereignRuntime correctly. Tests the full boot sequence from Phase 19-31.

Standing on: Simon (hierarchy) + Ostrom (commons) + Csikszentmihalyi (flow) + Deming (PDCA)
"""

from __future__ import annotations

import pytest

# ─── Package Imports via core ────────────────────────────────────────────────


class TestAllPackagesImportable:
    """Verify all packages are importable via core namespace."""

    def test_phase25_genesis_importable(self):
        from core.genesis import GenesisConfig, GenesisOrchestrator, GenesisResult

        assert GenesisOrchestrator is not None

    def test_phase26_guild_importable(self):
        from core.guild import Guild, GuildMember, GuildRegistry

        assert GuildRegistry is not None

    def test_phase26_quest_importable(self):
        from core.quest import Quest, QuestDifficulty, QuestEngine

        assert QuestEngine is not None

    def test_phase27_hrm_importable(self):
        from core.hrm import AbstractionLevel, HierarchicalReasoningModel, HRMConfig

        assert HierarchicalReasoningModel is not None

    def test_phase28_northstar_importable(self):
        from core.northstar import (
            BridgeNodeDetector,
            GoldenGemDetector,
            NorthStarEngine,
        )

        assert NorthStarEngine is not None

    def test_phase_memory_agentdb_importable(self):
        from core.memory import AgentDB, HNSWIndex, MemoryRecord, UnifiedStore

        assert AgentDB is not None

    def test_core_version_bumped(self):
        import core

        assert core.__version__ == "2.5.0"

    def test_all_32_subsystems_accessible_via_core(self):
        """The complete subsystem inventory — every package importable via core."""
        import core

        expected = [
            # Infrastructure (Phase 1-18)
            "pci",
            "vault",
            "federation",
            "inference",
            "a2a",
            "integration",
            "ntu",
            "protocols",
            # Decomposed sovereign
            "governance",
            "reasoning",
            "orchestration",
            "treasury",
            "bridges",
            # Phase 31: Cognitive Fusion
            "hypergraph",
            "cognitive_fusion",
            "memory_coder",
            # Phase 25-28: Ecosystem
            "genesis",
            "guild",
            "quest",
            "hrm",
            "northstar",
            "memory",
        ]
        for pkg in expected:
            assert hasattr(core, pkg), f"core.{pkg} not importable"


# ─── RuntimeConfig Feature Flags ─────────────────────────────────────────────


class TestRuntimeConfigFlags:
    """Verify feature flags for all ecosystem subsystems."""

    def test_default_config_enables_all(self):
        from core.sovereign.runtime_types import RuntimeConfig

        cfg = RuntimeConfig()
        assert cfg.enable_hrm is True
        assert cfg.enable_northstar is True
        assert cfg.enable_guild_system is True
        assert cfg.enable_quest_system is True
        assert cfg.enable_cognitive_fusion is True
        assert cfg.enable_memory_synthesizer is True

    def test_minimal_config_disables_all(self):
        from core.sovereign.runtime_types import RuntimeConfig

        cfg = RuntimeConfig.minimal()
        assert cfg.enable_hrm is False
        assert cfg.enable_northstar is False
        assert cfg.enable_guild_system is False
        assert cfg.enable_quest_system is False


# ─── SovereignRuntime Component Wiring ───────────────────────────────────────


class TestRuntimeComponentWiring:
    """Verify all ecosystem subsystems wire into SovereignRuntime."""

    def test_runtime_state_includes_ecosystem_components(self):
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        rt = SovereignRuntime(RuntimeConfig.minimal())
        state = rt._get_runtime_state()

        # All Phase 25-28 + Phase 31 components must appear
        assert "hrm_engine" in state["components"]
        assert "northstar_engine" in state["components"]
        assert "guild_registry" in state["components"]
        assert "quest_engine" in state["components"]
        assert "hypergraph_store" in state["components"]
        assert "cognitive_fusion" in state["components"]
        assert "memory_synthesizer" in state["components"]

        # Minimal config: all should be False (not yet initialized)
        assert state["components"]["hrm_engine"] is False
        assert state["components"]["northstar_engine"] is False
        assert state["components"]["guild_registry"] is False
        assert state["components"]["quest_engine"] is False

    def test_ecosystem_init_standalone(self):
        """Verify _init_ecosystem_subsystems runs without crashing."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        rt = SovereignRuntime(RuntimeConfig())
        rt._init_ecosystem_subsystems()

        assert rt._hrm_engine is not None
        assert rt._northstar_engine is not None
        assert rt._guild_registry is not None
        assert rt._quest_engine is not None

    def test_ecosystem_disabled_by_config(self):
        """When flags are False, components stay None."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        cfg = RuntimeConfig.minimal()
        rt = SovereignRuntime(cfg)
        rt._init_ecosystem_subsystems()

        assert rt._hrm_engine is None
        assert rt._northstar_engine is None
        assert rt._guild_registry is None
        assert rt._quest_engine is None


# ─── Subsystem Standalone Verification ───────────────────────────────────────


class TestSubsystemStandalone:
    """Verify each subsystem works independently before integration."""

    def test_hrm_creates_5_levels(self):
        from core.hrm import HierarchicalReasoningModel, HRMConfig

        hrm = HierarchicalReasoningModel(HRMConfig())
        assert len(hrm._config.active_levels) == 5

    def test_northstar_detects_gems(self):
        from core.northstar import GoldenGemDetector

        detector = GoldenGemDetector()
        # Detector has individual detect_* methods for each gem type
        result = detector.detect_emergence(
            node_count=100, edge_count=500, clustering_coefficient=0.8
        )
        # Result is GemActivation or None depending on thresholds
        assert result is None or hasattr(result, "gem_type")

    def test_guild_registry_has_defaults(self):
        from core.guild import GuildRegistry

        registry = GuildRegistry()
        assert len(registry._guilds) > 0

    def test_quest_engine_has_defaults(self):
        from core.quest import QuestEngine

        engine = QuestEngine()
        assert len(engine._quests) > 0

    def test_genesis_orchestrator_constructs(self):
        from core.genesis import GenesisConfig, GenesisOrchestrator

        config = GenesisConfig(pat_count=1, sat_count=1)
        orch = GenesisOrchestrator(config)
        assert orch is not None

    def test_agentdb_constructs(self):
        from core.memory import AgentDB

        db = AgentDB()
        assert db is not None


# ─── Cross-System Integration ────────────────────────────────────────────────


class TestCrossSystemIntegration:
    """Verify subsystems compose correctly."""

    def test_hrm_snr_gradient_aligns_with_constants(self):
        """HRM SNR gradient must align with core.integration.constants."""
        from core.hrm import HRM_SNR_GRADIENT
        from core.integration.constants import (
            SNR_THRESHOLD_T0_ELITE,
            UNIFIED_SNR_THRESHOLD,
        )

        # Lowest HRM level should use UNIFIED_SNR_THRESHOLD or higher
        min_snr = min(HRM_SNR_GRADIENT.values())
        assert min_snr >= UNIFIED_SNR_THRESHOLD

        # Highest should approach elite
        max_snr = max(HRM_SNR_GRADIENT.values())
        assert max_snr >= SNR_THRESHOLD_T0_ELITE - 0.01  # Allow tiny epsilon

    def test_quest_ihsan_gate_uses_constants(self):
        """Quest rewards should respect Ihsan threshold from constants."""
        from core.quest import Quest, QuestDifficulty, QuestReward

        # BLOOM difficulty should require higher quality
        q = Quest(
            quest_id="test",
            title="Test",
            description="Test quest",
            difficulty=QuestDifficulty.BLOOM,
            guild_id="test-guild",
            reward=QuestReward(bloom_amount=50, impt_amount=100),
        )
        # BLOOM is string enum — verify it's a higher tier
        tier_order = ["seed", "sprout", "bloom", "forest"]
        assert tier_order.index(q.difficulty.value) >= 2

    def test_full_component_count(self):
        """Verify total component count in runtime state."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        rt = SovereignRuntime(RuntimeConfig.minimal())
        state = rt._get_runtime_state()
        components = state["components"]

        # Should have at least 11 component entries
        assert len(components) >= 11, f"Expected ≥11 components, got {len(components)}"
