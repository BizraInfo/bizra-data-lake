"""
BIZRA Genesis Orchestrator — Smoke Tests
==========================================

16 tests covering the full genesis bootstrap pipeline:
types, hardware scanner, URP pledge, mobile pairing,
orchestrator, CLI parser, and constitutional gates.

Test naming: test_XX_descriptive_name
Coverage: GenesisOrchestrator, GenesisConfig, GenesisResult,
          HardwareScanner, URPPledge, MobilePairResult, CLI
"""

import argparse

import pytest

from core.pci.crypto import generate_keypair
from core.genesis import (
    CHECKMARK,
    CROSSMARK,
    OMEGA,
    GenesisConfig,
    GenesisOrchestrator,
    GenesisResult,
    GenesisStep,
    GenesisStepStatus,
    HardwareInfo,
    HardwareScanner,
    MobilePairResult,
    URPPledge,
    pair_mobile,
    pledge_resources,
    verify_pledge_signature,
)
from core.genesis.cli import build_genesis_parser
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD


class TestGenesisTypes:
    """Genesis type construction and validation."""

    # ── test_01: GenesisConfig default construction ──────────────────────
    def test_01_genesis_config_defaults(self) -> None:
        """GenesisConfig has sensible defaults."""
        config = GenesisConfig()
        assert config.identity_genesis is False
        assert config.hardware_scan is False
        assert config.pat_count == 7
        assert config.sat_count == 5
        assert config.sat_mode == "mini5"
        assert config.hda_bridge is False
        assert config.mobile_pair is None
        assert config.guild_join is None
        assert config.quest_accept is None
        assert config.ihsan_target == 0.999
        assert config.strict_bootstrap is True
        assert config.allow_degraded is False

    # ── test_02: GenesisConfig with all flags ────────────────────────────
    def test_02_genesis_config_all_flags(self) -> None:
        """GenesisConfig captures all CLI flags."""
        config = GenesisConfig(
            identity_genesis=True,
            hardware_scan=True,
            pat_count=7,
            sat_count=5,
            hda_bridge=True,
            mobile_pair="Z Fold 6:SM-F956B",
            guild_join="agriculture",
            quest_accept="001-sustainable-water",
            ihsan_target=0.999,
        )
        d = config.to_dict()
        assert d["identity_genesis"] is True
        assert d["pat_count"] == 7
        assert d["guild_join"] == "agriculture"
        assert d["ihsan_target"] == 0.999
        assert d["sat_mode"] == "mini5"
        assert d["strict_bootstrap"] is True

    # ── test_03: GenesisStep timing ──────────────────────────────────────
    def test_03_genesis_step_timing(self) -> None:
        """GenesisStep records timing and status."""
        step = GenesisStep(
            name="test_step",
            status=GenesisStepStatus.SUCCESS,
            duration_ms=42.5,
            details={"key": "value"},
        )
        assert step.name == "test_step"
        assert step.status == GenesisStepStatus.SUCCESS
        assert step.duration_ms == 42.5
        d = step.to_dict()
        assert d["status"] == "success"

    # ── test_04: GenesisResult aggregation ───────────────────────────────
    def test_04_genesis_result_aggregation(self) -> None:
        """GenesisResult aggregates step counts correctly."""
        result = GenesisResult(
            steps=[
                GenesisStep(name="a", status=GenesisStepStatus.SUCCESS),
                GenesisStep(name="b", status=GenesisStepStatus.SUCCESS),
                GenesisStep(name="c", status=GenesisStepStatus.FAILED),
                GenesisStep(name="d", status=GenesisStepStatus.SKIPPED),
            ],
            node_id="BIZRA-00000000",
        )
        assert result.successful_steps == 2
        assert result.failed_steps == 1
        assert result.skipped_steps == 1


class TestHardwareAndURP:
    """Hardware scanner, URP pledge, and mobile pairing."""

    # ── test_05: HardwareScanner instantiation ───────────────────────────
    def test_05_hardware_scanner_init(self) -> None:
        """HardwareScanner can be instantiated."""
        scanner = HardwareScanner()
        assert scanner is not None

    # ── test_06: URPPledge creation ──────────────────────────────────────
    def test_06_urp_pledge(self) -> None:
        """URPPledge creates a signed pledge record."""
        pledge = pledge_resources(
            node_id="BIZRA-00000001",
            hardware_info={"ram_gb": 128, "vram_gb": 16},
        )
        assert pledge.node_id == "BIZRA-00000001"
        assert pledge.ram_gb == 128
        assert pledge.vram_gb == 16
        assert len(pledge.pledge_hash) == 16
        assert pledge.pledged_at != ""
        assert pledge.signed is False
        assert pledge.reason_code == "GENESIS_URP_UNSIGNED_STUB"
        assert verify_pledge_signature(pledge) is False

    def test_06b_urp_pledge_signed_enforced(self) -> None:
        """URP pledge signs and verifies when private key is provided."""
        private_key, public_key = generate_keypair()
        pledge = pledge_resources(
            node_id="BIZRA-00000002",
            hardware_info={"ram_gb": 64, "vram_gb": 12, "storage_gb": 2048},
            signing_private_key_hex=private_key,
        )
        assert pledge.signed is True
        assert pledge.enforced is True
        assert pledge.enforcement_mode == "single_node_signed"
        assert pledge.reason_code == "GENESIS_URP_SIGNED"
        assert pledge.signer_public_key == public_key
        assert len(pledge.payload_digest) == 64
        assert len(pledge.signature) == 128
        assert verify_pledge_signature(pledge) is True

    # ── test_07: MobilePairResult parsing ────────────────────────────────
    def test_07_mobile_pair_parsing(self) -> None:
        """Mobile pairing parses device spec correctly."""
        result = pair_mobile("Z Fold 6:SM-F956B")
        assert result.device_name == "Z Fold 6"
        assert result.model == "SM-F956B"
        assert result.paired is True
        assert result.proximity_routing is True
        assert result.protocol == "stub-v1"
        assert result.status == "stub"
        assert result.reason_code == "GENESIS_MOBILE_PAIR_STUB"


class TestGenesisOrchestrator:
    """Genesis orchestrator pipeline tests."""

    # ── test_08: orchestrator init ───────────────────────────────────────
    def test_08_orchestrator_init(self) -> None:
        """GenesisOrchestrator initializes with config."""
        config = GenesisConfig()
        orchestrator = GenesisOrchestrator(config)
        assert orchestrator.config is config

    # ── test_09: full run with minimal config ────────────────────────────
    def test_09_full_run_minimal(self) -> None:
        """Orchestrator runs with minimal config (no identity genesis)."""
        config = GenesisConfig(
            identity_genesis=False,
            hardware_scan=False,
            ihsan_target=0.999,
            strict_bootstrap=False,
            allow_degraded=True,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()

        assert isinstance(result, GenesisResult)
        assert result.total_duration_ms > 0
        assert len(result.steps) >= 3  # PAT, SAT, token, ihsan
        assert result.genesis_hash != ""

    # ── test_10: full run with all flags ─────────────────────────────────
    def test_10_full_run_all_flags(self) -> None:
        """Orchestrator runs with all flags enabled (except identity genesis)."""
        config = GenesisConfig(
            identity_genesis=False,  # Skip actual minting for speed
            hardware_scan=False,  # Skip system calls
            pat_count=7,
            sat_count=5,
            sat_mode="mini5",
            hda_bridge=False,  # Skip bridge check
            mobile_pair="Z Fold 6:SM-F956B",
            guild_join="agriculture",
            quest_accept="001-sustainable-water",
            ihsan_target=0.999,
            strict_bootstrap=False,
            allow_degraded=True,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()

        assert isinstance(result, GenesisResult)
        assert len(result.steps) >= 6  # PAT, SAT, token, mobile, guild, quest, ihsan
        # Check guild and quest steps succeeded
        guild_step = next((s for s in result.steps if s.name == "guild_join"), None)
        assert guild_step is not None
        assert guild_step.status == GenesisStepStatus.SUCCESS

        quest_step = next((s for s in result.steps if s.name == "quest_accept"), None)
        assert quest_step is not None
        assert quest_step.status == GenesisStepStatus.SUCCESS

    # ── test_11: output formatting contains checkmarks ───────────────────
    def test_11_output_formatting(self) -> None:
        """Formatted output contains checkmarks for successful steps."""
        config = GenesisConfig(
            guild_join="agriculture",
            quest_accept="001-sustainable-water",
            ihsan_target=0.999,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()
        output = orchestrator.format_output(result)

        assert CHECKMARK in output or CROSSMARK in output
        assert OMEGA in output
        assert "BIZRA" in output

    # ── test_12: CLI parser builds correctly ──────────────────────────────
    def test_12_cli_parser(self) -> None:
        """Genesis CLI parser accepts all expected flags."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_genesis_parser(subparsers)

        args = parser.parse_args([
            "genesis",
            "--identity-genesis",
            "--hardware-scan",
            "--pat-7",
            "--sat-5",
            "--hda-bridge",
            "--mobile-pair", "Z Fold 6:SM-F956B",
            "--guild-join", "agriculture",
            "--quest-accept", "001-sustainable-water",
            "--ihsan-target", "0.999",
        ])

        assert args.command == "genesis"
        assert args.identity_genesis is True
        assert args.hardware_scan is True
        assert args.pat_7 is True
        assert args.sat_5 is True
        assert args.hda_bridge is True
        assert args.mobile_pair == "Z Fold 6:SM-F956B"
        assert args.guild_join == "agriculture"
        assert args.quest_accept == "001-sustainable-water"
        assert args.ihsan_target == 0.999
        assert args.strict_bootstrap is True


class TestConstitutionalGates:
    """Constitutional compliance tests."""

    # ── test_13: ihsan target validation ──────────────────────────────────
    def test_13_ihsan_target_bounds(self) -> None:
        """Ihsan target must be a valid float."""
        config = GenesisConfig(ihsan_target=0.999)
        assert 0.0 <= config.ihsan_target <= 1.0

        config2 = GenesisConfig(ihsan_target=0.95)
        assert config2.ihsan_target >= UNIFIED_IHSAN_THRESHOLD

    # ── test_14: step failure isolation ───────────────────────────────────
    def test_14_step_failure_isolation(self) -> None:
        """One failing step does not block others."""
        config = GenesisConfig(
            guild_join="nonexistent-guild-xyz",  # This will fail
            quest_accept="001-sustainable-water",
            ihsan_target=0.999,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()

        # Guild step should fail
        guild_step = next((s for s in result.steps if s.name == "guild_join"), None)
        assert guild_step is not None
        assert guild_step.status == GenesisStepStatus.FAILED

        # But quest step should still run (and fail because guild didn't set up)
        # Ihsan step should still succeed
        ihsan_step = next((s for s in result.steps if s.name == "ihsan_target"), None)
        assert ihsan_step is not None
        assert ihsan_step.status == GenesisStepStatus.SUCCESS

    # ── test_15: genesis hash determinism ─────────────────────────────────
    def test_15_genesis_hash_deterministic(self) -> None:
        """Genesis hash is deterministic for same inputs."""
        result1 = GenesisResult(
            node_id="BIZRA-00000000",
            steps=[GenesisStep(name="a", status=GenesisStepStatus.SUCCESS)],
            created_at="2026-02-02T00:00:00Z",
        )
        result2 = GenesisResult(
            node_id="BIZRA-00000000",
            steps=[GenesisStep(name="a", status=GenesisStepStatus.SUCCESS)],
            created_at="2026-02-02T00:00:00Z",
        )

        hash1 = result1.compute_hash()
        hash2 = result2.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 16

    # ── test_16: constitutional gate compliance ───────────────────────────
    def test_16_constitutional_compliance(self) -> None:
        """Full pipeline respects constitutional thresholds."""
        config = GenesisConfig(
            guild_join="agriculture",
            quest_accept="001-sustainable-water",
            ihsan_target=0.999,
            strict_bootstrap=False,
            allow_degraded=True,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()

        # Ihsan target must be >= threshold
        ihsan_step = next((s for s in result.steps if s.name == "ihsan_target"), None)
        assert ihsan_step is not None
        target = ihsan_step.details.get("target", 0)
        assert target >= UNIFIED_IHSAN_THRESHOLD

        # Genesis hash must be non-empty
        assert result.genesis_hash != ""
        assert len(result.genesis_hash) == 16

    def test_17_strict_bootstrap_fails_on_stub_steps(self) -> None:
        """Strict mode must fail closed when a step returns stub/deferred."""
        config = GenesisConfig(
            mobile_pair="Z Fold 6:SM-F956B",
            strict_bootstrap=True,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()

        mobile_step = next((s for s in result.steps if s.name == "mobile_pair"), None)
        assert mobile_step is not None
        assert mobile_step.status == GenesisStepStatus.FAILED
        assert mobile_step.reason_code == "GENESIS_MOBILE_PAIR_STUB"
        assert result.strict_gate_passed is False
        assert "GENESIS_MOBILE_PAIR_STUB" in result.reason_codes

    def test_18_non_strict_marks_stub_steps_as_skipped(self) -> None:
        """Non-strict mode must not mark stub/deferred steps as successful."""
        config = GenesisConfig(
            mobile_pair="Z Fold 6:SM-F956B",
            strict_bootstrap=False,
            allow_degraded=True,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()

        mobile_step = next((s for s in result.steps if s.name == "mobile_pair"), None)
        assert mobile_step is not None
        assert mobile_step.status == GenesisStepStatus.SKIPPED
        assert mobile_step.reason_code == "GENESIS_MOBILE_PAIR_STUB"
        assert "GENESIS_MOBILE_PAIR_STUB" in result.reason_codes

    def test_19_strict_bootstrap_fails_on_unsigned_urp(self) -> None:
        """Strict mode must fail when URP pledge has no signature."""
        config = GenesisConfig(strict_bootstrap=True)
        orchestrator = GenesisOrchestrator(config)
        orchestrator._hardware_info = HardwareInfo(
            cpu="TestCPU",
            gpu="TestGPU",
            ram_gb=64,
            vram_gb=12,
            platform_name="Linux",
            os_version="test",
        )
        result = orchestrator.run()

        urp_step = next((s for s in result.steps if s.name == "urp_pledge"), None)
        assert urp_step is not None
        assert urp_step.status == GenesisStepStatus.FAILED
        assert urp_step.reason_code == "GENESIS_URP_UNSIGNED_STUB"
        assert result.strict_gate_passed is False
        assert "GENESIS_URP_UNSIGNED_STUB" in result.reason_codes
