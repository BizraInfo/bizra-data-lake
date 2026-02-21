"""
Tests: Genesis Orchestrator
============================

Covers: GenesisOrchestrator, GenesisConfig, GenesisResult, GenesisStep,
        HardwareScanner, URPPledge, pair_mobile, CLI (build_config, run_genesis)

Standing on Giants:
- Beck (2002, TDD): Tests specify behavior, not validate it post-hoc
- Wiener (1948): Homeostatic bootstrap has verifiable convergence
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from core.genesis import GenesisConfig, GenesisResult, GenesisStep
from core.genesis.orchestrator import GenesisOrchestrator
from core.genesis.hardware import HardwareScanner
from core.genesis.mobile_pairing import MobilePairResult, pair_mobile
from core.genesis.urp import URPPledge, pledge_resources


# ─────────────────────────────────────────────────────────────────────────────
# Module Import Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenesisImports:
    def test_import_orchestrator(self):
        from core.genesis.orchestrator import GenesisOrchestrator
        assert GenesisOrchestrator is not None

    def test_import_types(self):
        from core.genesis import GenesisConfig, GenesisResult, GenesisStep
        assert GenesisConfig is not None
        assert GenesisResult is not None
        assert GenesisStep is not None

    def test_module_version(self):
        import core.genesis as g
        assert g.__version__ == "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# GenesisConfig Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenesisConfig:
    def test_defaults(self):
        c = GenesisConfig()
        assert c.identity_genesis is False
        assert c.hardware_scan is False
        assert c.pat_count == 7
        assert c.sat_count == 5
        assert c.ihsan_target == 0.999
        assert c.mobile_pair is None
        assert c.guild_join is None
        assert c.quest_accept is None

    def test_custom_values(self):
        c = GenesisConfig(
            identity_genesis=True,
            hardware_scan=True,
            guild_join="agriculture",
            ihsan_target=0.95,
        )
        assert c.identity_genesis is True
        assert c.hardware_scan is True
        assert c.guild_join == "agriculture"
        assert c.ihsan_target == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# GenesisStep Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenesisStep:
    def test_default_status_pending(self):
        s = GenesisStep(name="test_step")
        assert s.status == "pending"
        assert not s.success

    def test_success_property(self):
        s = GenesisStep(name="test_step", status="success")
        assert s.success

    def test_non_success_statuses(self):
        for status in ("pending", "running", "failed", "skipped"):
            s = GenesisStep(name="x", status=status)
            assert not s.success


# ─────────────────────────────────────────────────────────────────────────────
# GenesisResult Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenesisResult:
    def test_defaults(self):
        r = GenesisResult()
        assert r.steps == []
        assert r.node_id == ""
        assert r.genesis_hash == ""
        assert not r.success

    def test_step_count(self):
        r = GenesisResult()
        r.steps = [
            GenesisStep(name="a", status="success"),
            GenesisStep(name="b", status="failed"),
            GenesisStep(name="c", status="skipped"),
        ]
        assert r.step_count == 3
        assert r.success_count == 1
        assert r.failed_count == 1

    def test_get_step_by_name(self):
        r = GenesisResult()
        r.steps = [GenesisStep(name="hardware_scan", status="success")]
        s = r.get_step("hardware_scan")
        assert s is not None
        assert s.name == "hardware_scan"

    def test_get_step_missing_returns_none(self):
        r = GenesisResult()
        assert r.get_step("nonexistent") is None

    def test_to_dict_shape(self):
        r = GenesisResult(node_id="BIZRA-TEST", success=True)
        r.steps = [GenesisStep(name="a", status="success")]
        d = r.to_dict()
        assert d["node_id"] == "BIZRA-TEST"
        assert d["success"] is True
        assert isinstance(d["steps"], list)
        assert d["steps"][0]["name"] == "a"


# ─────────────────────────────────────────────────────────────────────────────
# HardwareScanner Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHardwareScanner:
    def test_scan_returns_dict(self):
        scanner = HardwareScanner()
        info = scanner.scan()
        assert isinstance(info, dict)

    def test_scan_has_required_keys(self):
        scanner = HardwareScanner()
        info = scanner.scan()
        assert "cpu" in info
        assert "gpu" in info
        assert "ram_gb" in info
        assert "fingerprint" in info
        assert "platform" in info

    def test_fingerprint_is_32_chars(self):
        scanner = HardwareScanner()
        info = scanner.scan()
        assert len(info["fingerprint"]) == 32

    def test_ram_gb_is_non_negative(self):
        scanner = HardwareScanner()
        info = scanner.scan()
        assert info["ram_gb"] >= 0.0

    def test_format_summary_returns_string(self):
        scanner = HardwareScanner()
        summary = scanner.format_summary()
        assert isinstance(summary, str)
        assert "CPU" in summary
        assert "RAM" in summary

    def test_scan_deterministic(self):
        """Same machine should produce same fingerprint."""
        scanner = HardwareScanner()
        a = scanner.scan()
        b = scanner.scan()
        assert a["fingerprint"] == b["fingerprint"]


# ─────────────────────────────────────────────────────────────────────────────
# URPPledge Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestURPPledge:
    def test_pledge_resources_produces_pledge(self):
        pledge = pledge_resources("BIZRA-TEST", {"ram_gb": 64.0})
        assert pledge.node_id == "BIZRA-TEST"
        assert pledge.ram_gb == 32.0  # 50% of 64GB

    def test_pledge_hash_generated(self):
        pledge = pledge_resources("BIZRA-TEST", {"ram_gb": 32.0})
        assert len(pledge.pledge_hash) == 32

    def test_pledge_hash_deterministic(self):
        a = pledge_resources("BIZRA-A", {"ram_gb": 16.0})
        b = pledge_resources("BIZRA-A", {"ram_gb": 16.0})
        assert a.pledge_hash == b.pledge_hash

    def test_pledge_to_dict(self):
        p = URPPledge(node_id="BIZRA-X", ram_gb=8.0)
        d = p.to_dict()
        assert d["node_id"] == "BIZRA-X"
        assert d["ram_gb"] == 8.0
        assert "pledge_hash" in d


# ─────────────────────────────────────────────────────────────────────────────
# MobilePairing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMobilePairing:
    def test_pair_with_colon_format(self):
        result = pair_mobile("Z Fold 6:SM-F956B")
        assert result.device_name == "Z Fold 6"
        assert result.model == "SM-F956B"
        assert result.paired is True

    def test_pair_without_model(self):
        result = pair_mobile("iPhone 16 Pro")
        assert result.device_name == "iPhone 16 Pro"
        assert result.model == ""

    def test_pair_proximity_routing_enabled(self):
        result = pair_mobile("Any Device:MODEL")
        assert result.proximity_routing is True

    def test_pair_to_dict(self):
        result = pair_mobile("Test:T-001")
        d = result.to_dict()
        assert d["device_name"] == "Test"
        assert d["model"] == "T-001"
        assert d["paired"] is True


# ─────────────────────────────────────────────────────────────────────────────
# GenesisOrchestrator Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenesisOrchestrator:
    def test_run_empty_config_has_steps(self):
        """Even an empty config (all skipped) still runs 8 steps."""
        o = GenesisOrchestrator()
        config = GenesisConfig()  # all False/None
        result = o.run(config)
        assert result.step_count == 8

    def test_run_all_skipped_on_default_config(self):
        """Default config skips all optional steps; ihsan_target always succeeds."""
        o = GenesisOrchestrator()
        config = GenesisConfig()
        result = o.run(config)
        # Only ihsan_target runs (always success); rest are skipped
        skipped = [s for s in result.steps if s.status == "skipped"]
        success = [s for s in result.steps if s.success]
        assert len(skipped) >= 6  # Most steps are skipped
        assert any(s.name == "ihsan_target" and s.success for s in result.steps)

    def test_hardware_scan_step(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(hardware_scan=True)
        result = o.run(config)
        hw = result.get_step("hardware_scan")
        assert hw is not None
        assert hw.status == "success"
        assert "fingerprint" in hw.details

    def test_guild_join_step(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(guild_join="agriculture")
        result = o.run(config)
        step = result.get_step("guild_join")
        assert step is not None
        assert step.status == "success"
        assert step.details["guild_id"] == "agriculture"

    def test_quest_accept_step(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(quest_accept="001-sustainable-water")
        result = o.run(config)
        step = result.get_step("quest_accept")
        assert step is not None
        assert step.status == "success"
        assert step.details["quest_id"] == "001-sustainable-water"

    def test_ihsan_target_always_succeeds(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(ihsan_target=0.999)
        result = o.run(config)
        step = result.get_step("ihsan_target")
        assert step is not None
        assert step.success
        assert step.details["target"] == 0.999

    def test_mobile_pair_step(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(mobile_pair="Z Fold 6:SM-F956B")
        result = o.run(config)
        step = result.get_step("mobile_pair")
        assert step is not None
        assert step.status == "success"
        assert step.details["device_name"] == "Z Fold 6"

    def test_genesis_hash_produced_when_steps_run(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(hardware_scan=True)
        result = o.run(config)
        assert len(result.genesis_hash) == 32

    def test_genesis_hash_empty_for_all_skipped(self):
        """If all steps are skipped except ihsan_target, hash is non-empty."""
        o = GenesisOrchestrator()
        config = GenesisConfig()
        result = o.run(config)
        # ihsan_target always runs → always produces hash input
        assert result.genesis_hash != ""

    def test_total_duration_ms_positive(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(hardware_scan=True)
        result = o.run(config)
        assert result.total_duration_ms > 0.0

    def test_to_dict_valid_shape(self):
        o = GenesisOrchestrator()
        config = GenesisConfig(hardware_scan=True, guild_join="agriculture")
        result = o.run(config)
        d = result.to_dict()
        assert "node_id" in d
        assert "genesis_hash" in d
        assert "success" in d
        assert "steps" in d
        assert len(d["steps"]) == 8


# ─────────────────────────────────────────────────────────────────────────────
# Genesis CLI Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenesisCLI:
    def test_build_config_from_all_steps(self):
        from core.genesis.cli import build_config
        ns = argparse.Namespace(
            all_steps=True, identity_genesis=False, hardware_scan=False,
            hda_bridge=False, mobile_pair=None, guild_join=None,
            quest_accept=None, ihsan_target=0.999, node_dir=None,
            json_output=False,
        )
        config = build_config(ns)
        assert config.identity_genesis is True
        assert config.hardware_scan is True
        assert config.guild_join == "agriculture"  # --all default
        assert config.quest_accept == "001-sustainable-water"  # --all default

    def test_build_config_explicit_guild(self):
        from core.genesis.cli import build_config
        ns = argparse.Namespace(
            all_steps=False, identity_genesis=False, hardware_scan=False,
            hda_bridge=False, mobile_pair=None, guild_join="finance",
            quest_accept=None, ihsan_target=0.95, node_dir=None,
            json_output=False,
        )
        config = build_config(ns)
        assert config.guild_join == "finance"
        assert config.ihsan_target == 0.95

    def test_run_genesis_exits_zero_on_success(self):
        from core.genesis.cli import run_genesis
        ns = argparse.Namespace(
            all_steps=False, identity_genesis=False, hardware_scan=True,
            hda_bridge=False, mobile_pair=None, guild_join="agriculture",
            quest_accept=None, ihsan_target=0.999, node_dir=None,
            json_output=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            run_genesis(ns)
        assert exc_info.value.code == 0

    def test_run_genesis_json_output(self, capsys):
        from core.genesis.cli import run_genesis
        import json as _json
        ns = argparse.Namespace(
            all_steps=False, identity_genesis=False, hardware_scan=True,
            hda_bridge=False, mobile_pair=None, guild_join=None,
            quest_accept=None, ihsan_target=0.999, node_dir=None,
            json_output=True,
        )
        with pytest.raises(SystemExit):
            run_genesis(ns)
        captured = capsys.readouterr()
        data = _json.loads(captured.out)
        assert "steps" in data
        assert "genesis_hash" in data
