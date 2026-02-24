"""
Tests for the BIZRA Node Onboarding Wizard.

Covers:
    - Non-interactive onboarding flow
    - Credential persistence and loading
    - Duplicate onboarding prevention
    - Interactive wizard (output verification)
    - Dashboard display
    - Edge cases (corrupted files, missing dirs)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.pat.onboarding import (
    DEFAULT_NODE_DIR,
    NodeCredentials,
    NodeTemplate,
    OnboardingWizard,
    get_node_credentials,
    is_onboarded,
    quick_onboard,
)


@pytest.fixture
def tmp_node_dir(tmp_path):
    """Provide a temporary node directory for each test."""
    return tmp_path / ".bizra-node-test"


@pytest.fixture
def wizard(tmp_node_dir):
    """Create an OnboardingWizard with a temp directory."""
    return OnboardingWizard(node_dir=tmp_node_dir)


class TestNodeCredentials:
    """Test NodeCredentials dataclass."""

    def test_to_dict_roundtrip(self):
        cred = NodeCredentials(
            node_id="BIZRA-A1B2C3D4",
            public_key="a" * 64,
            private_key="b" * 64,
            sovereignty_tier="seed",
            sovereignty_score=0.0,
            created_at="2026-02-07T00:00:00Z",
            pat_agent_ids=["PAT-A1B2C3D4-WRK-001"],
            sat_agent_ids=["SAT-F7E8D9C0-VAL-001"],
        )
        d = cred.to_dict()
        assert d["node_id"] == "BIZRA-A1B2C3D4"
        assert len(d["pat_agent_ids"]) == 1
        assert len(d["sat_agent_ids"]) == 1

        restored = NodeCredentials.from_dict(d)
        assert restored.node_id == cred.node_id
        assert restored.public_key == cred.public_key

    def test_from_dict_ignores_extra_fields(self):
        d = {
            "node_id": "BIZRA-A1B2C3D4",
            "public_key": "a" * 64,
            "private_key": "b" * 64,
            "sovereignty_tier": "seed",
            "sovereignty_score": 0.0,
            "created_at": "2026-02-07T00:00:00Z",
            "extra_field": "ignored",
        }
        cred = NodeCredentials.from_dict(d)
        assert cred.node_id == "BIZRA-A1B2C3D4"


class TestOnboardingWizard:
    """Test the OnboardingWizard core flow."""

    def test_fresh_node_not_onboarded(self, wizard):
        assert not wizard.is_already_onboarded()
        assert wizard.load_existing_credentials() is None

    def test_onboard_creates_identity(self, wizard):
        credentials = wizard.onboard()

        assert credentials.node_id.startswith("BIZRA-")
        assert len(credentials.node_id) == 14
        assert len(credentials.public_key) == 64
        assert len(credentials.private_key) == 64
        assert credentials.sovereignty_tier == "seed"
        assert credentials.sovereignty_score == 0.0
        assert len(credentials.pat_agent_ids) == 7
        assert len(credentials.sat_agent_ids) == 5

    def test_onboard_persists_credentials(self, wizard):
        wizard.onboard()

        assert wizard.credentials_file.exists()
        assert wizard.identity_file.exists()
        assert wizard.agents_file.exists()

        # Verify credentials file
        data = json.loads(wizard.credentials_file.read_text())
        assert data["node_id"].startswith("BIZRA-")
        assert len(data["pat_agent_ids"]) == 7

    def test_credential_file_permissions(self, wizard):
        wizard.onboard()

        if os.name != "nt":
            stat = os.stat(wizard.credentials_file)
            mode = stat.st_mode & 0o777
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_onboard_with_name(self, wizard):
        credentials = wizard.onboard(name="TestUser")

        identity_data = json.loads(wizard.identity_file.read_text())
        assert identity_data.get("metadata", {}).get("display_name") == "TestUser"

    def test_onboard_loads_back(self, wizard):
        original = wizard.onboard()

        loaded = wizard.load_existing_credentials()
        assert loaded is not None
        assert loaded.node_id == original.node_id
        assert loaded.public_key == original.public_key
        assert loaded.pat_agent_ids == original.pat_agent_ids

    def test_double_onboard_raises(self, wizard):
        wizard.onboard()

        with pytest.raises(FileExistsError, match="already onboarded"):
            wizard.onboard()

    def test_identity_card_has_valid_signatures(self, wizard):
        wizard.onboard()

        identity_data = json.loads(wizard.identity_file.read_text())
        assert identity_data["minter_signature"] is not None
        assert identity_data["self_signature"] is not None
        assert identity_data["status"] == "active"

    def test_agents_manifest_structure(self, wizard):
        wizard.onboard()

        agents_data = json.loads(wizard.agents_file.read_text())
        assert agents_data["total"] == 12
        assert len(agents_data["pat_agents"]) == 7
        assert len(agents_data["sat_agents"]) == 5

        # All PAT agents should be active (auto_activate=True)
        for agent in agents_data["pat_agents"]:
            assert agent["status"] == "active"
            assert agent["agent_id"].startswith("PAT-")
            assert agent["ownership_type"] == "user"

        # All SAT agents should be active
        for agent in agents_data["sat_agents"]:
            assert agent["status"] == "active"
            assert agent["agent_id"].startswith("SAT-")
            assert agent["ownership_type"] == "system"

    def test_pat_agent_types(self, wizard):
        wizard.onboard()

        agents_data = json.loads(wizard.agents_file.read_text())
        pat_types = [a["agent_type"] for a in agents_data["pat_agents"]]

        # Should have 2 workers + 1 each of researcher, guardian, synthesizer, validator, coordinator
        assert pat_types.count("worker") == 2
        assert "researcher" in pat_types
        assert "guardian" in pat_types
        assert "synthesizer" in pat_types
        assert "validator" in pat_types
        assert "coordinator" in pat_types

    def test_sat_agent_types(self, wizard):
        wizard.onboard()

        agents_data = json.loads(wizard.agents_file.read_text())
        sat_types = [a["agent_type"] for a in agents_data["sat_agents"]]

        assert "validator" in sat_types
        assert "guardian" in sat_types
        assert "coordinator" in sat_types
        assert "executor" in sat_types
        assert "synthesizer" in sat_types

    def test_corrupted_credentials_returns_none(self, wizard, tmp_node_dir):
        tmp_node_dir.mkdir(parents=True, exist_ok=True)
        (tmp_node_dir / "credentials.json").write_text("not json {{{")

        assert wizard.load_existing_credentials() is None


class TestInteractiveWizard:
    """Test the interactive CLI flow."""

    def test_interactive_existing_identity(self, wizard, capsys):
        wizard.onboard(name="TestUser")

        result = wizard.run_interactive()
        assert result is not None
        assert result.node_id.startswith("BIZRA-")

        captured = capsys.readouterr()
        assert "already onboarded" in captured.out.lower()

    def test_interactive_new_identity(self, wizard, capsys):
        with patch("builtins.input", return_value="NewUser"):
            result = wizard.run_interactive()

        assert result is not None
        assert result.node_id.startswith("BIZRA-")

        captured = capsys.readouterr()
        assert "Welcome to BIZRA" in captured.out
        assert result.node_id in captured.out
        assert "12 agents" in captured.out

    def test_interactive_skip_name(self, wizard, capsys):
        with patch("builtins.input", return_value=""):
            result = wizard.run_interactive()

        assert result is not None
        captured = capsys.readouterr()
        assert "Welcome to BIZRA" in captured.out


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_is_onboarded_false(self, tmp_node_dir):
        assert not is_onboarded(node_dir=tmp_node_dir)

    def test_is_onboarded_true(self, tmp_node_dir):
        wizard = OnboardingWizard(node_dir=tmp_node_dir)
        wizard.onboard()
        assert is_onboarded(node_dir=tmp_node_dir)

    def test_get_node_credentials_none(self, tmp_node_dir):
        assert get_node_credentials(node_dir=tmp_node_dir) is None

    def test_get_node_credentials_returns(self, tmp_node_dir):
        wizard = OnboardingWizard(node_dir=tmp_node_dir)
        original = wizard.onboard()

        loaded = get_node_credentials(node_dir=tmp_node_dir)
        assert loaded is not None
        assert loaded.node_id == original.node_id

    def test_quick_onboard(self, tmp_node_dir):
        cred = quick_onboard(name="QuickTest", node_dir=tmp_node_dir)

        assert cred.node_id.startswith("BIZRA-")
        assert len(cred.pat_agent_ids) == 7
        assert len(cred.sat_agent_ids) == 5

    def test_quick_onboard_double_raises(self, tmp_node_dir):
        quick_onboard(node_dir=tmp_node_dir)

        with pytest.raises(FileExistsError):
            quick_onboard(node_dir=tmp_node_dir)


class TestNodeDirDefault:
    """Test default node directory."""

    def test_default_dir_is_home(self):
        wizard = OnboardingWizard()
        assert wizard.node_dir == DEFAULT_NODE_DIR
        assert wizard.node_dir == Path.home() / ".bizra-node"

    def test_custom_dir(self, tmp_node_dir):
        wizard = OnboardingWizard(node_dir=tmp_node_dir)
        assert wizard.node_dir == tmp_node_dir


class TestNodeTemplate:
    """Test the MMORPG-inspired NodeTemplate dataclass."""

    def test_default_template_values(self):
        tmpl = NodeTemplate.default()
        assert tmpl.pat_count == 7
        assert tmpl.sat_count == 5
        assert tmpl.initial_seed_grant == 100.0
        assert tmpl.initial_bloom == 0.0
        assert tmpl.initial_impt == 0.0
        assert tmpl.ihsan_floor == 0.95
        assert tmpl.adl_gini_threshold == 0.35
        assert tmpl.snr_minimum == 0.85
        assert tmpl.desktop_rpc_enabled is True
        assert tmpl.tool_call_enabled is True
        assert tmpl.llm_call_locked is True
        assert tmpl.file_op_locked is True
        assert tmpl.browser_nav_locked is True
        assert tmpl.bootstrap_reflex_count == 4
        assert tmpl.max_cpu_percent == 25.0
        assert tmpl.max_memory_mb == 512
        assert tmpl.max_inference_tokens_per_day == 50_000
        assert tmpl.initial_stage == "seed"

    def test_alpha_100_template(self):
        tmpl = NodeTemplate.alpha_100()
        assert tmpl.initial_seed_grant == 500.0
        assert tmpl.max_inference_tokens_per_day == 100_000
        assert tmpl.bootstrap_reflex_count == 4
        # Inherits defaults for everything else
        assert tmpl.pat_count == 7
        assert tmpl.sat_count == 5
        assert tmpl.ihsan_floor == 0.95

    def test_to_dict_roundtrip(self):
        tmpl = NodeTemplate.default()
        d = tmpl.to_dict()
        assert d["pat_count"] == 7
        assert d["sat_count"] == 5
        assert d["initial_seed_grant"] == 100.0

        restored = NodeTemplate.from_dict(d)
        assert restored.pat_count == tmpl.pat_count
        assert restored.sat_count == tmpl.sat_count
        assert restored.initial_seed_grant == tmpl.initial_seed_grant
        assert restored.ihsan_floor == tmpl.ihsan_floor

    def test_from_dict_ignores_extra_fields(self):
        d = NodeTemplate.default().to_dict()
        d["unknown_field"] = "should_be_ignored"
        d["another_unknown"] = 42

        restored = NodeTemplate.from_dict(d)
        assert restored.pat_count == 7
        assert not hasattr(restored, "unknown_field")

    def test_from_dict_partial(self):
        """from_dict with a subset of fields uses defaults for the rest."""
        d = {"pat_count": 3, "sat_count": 2}
        tmpl = NodeTemplate.from_dict(d)
        assert tmpl.pat_count == 3
        assert tmpl.sat_count == 2
        # Everything else should be default
        assert tmpl.initial_seed_grant == 100.0
        assert tmpl.ihsan_floor == 0.95

    def test_fork_from_inherits_base(self):
        base = NodeTemplate.alpha_100()
        forked = NodeTemplate.fork_from(base)
        assert forked.initial_seed_grant == 500.0
        assert forked.max_inference_tokens_per_day == 100_000
        assert forked.pat_count == 7

    def test_fork_from_with_overrides(self):
        base = NodeTemplate.default()
        forked = NodeTemplate.fork_from(base, pat_count=10, sat_count=8)
        assert forked.pat_count == 10
        assert forked.sat_count == 8
        # Non-overridden values inherited from base
        assert forked.initial_seed_grant == 100.0
        assert forked.ihsan_floor == 0.95

    def test_fork_from_does_not_mutate_source(self):
        base = NodeTemplate.default()
        NodeTemplate.fork_from(base, pat_count=99)
        assert base.pat_count == 7  # Source unchanged

    def test_fork_chain(self):
        """Forks can be forked — template inheritance chain."""
        gen1 = NodeTemplate.default()
        gen2 = NodeTemplate.fork_from(gen1, initial_seed_grant=200.0)
        gen3 = NodeTemplate.fork_from(gen2, max_memory_mb=1024)
        assert gen3.initial_seed_grant == 200.0
        assert gen3.max_memory_mb == 1024
        assert gen3.pat_count == 7  # Inherited from gen1 through gen2


class TestOnboardingWithTemplate:
    """Test that OnboardingWizard respects NodeTemplate configuration."""

    def test_wizard_uses_default_template(self, tmp_node_dir):
        wizard = OnboardingWizard(node_dir=tmp_node_dir)
        assert wizard._template.pat_count == 7
        assert wizard._template.sat_count == 5

    def test_wizard_with_custom_template(self, tmp_node_dir):
        tmpl = NodeTemplate(pat_count=3, sat_count=2)
        wizard = OnboardingWizard(node_dir=tmp_node_dir, template=tmpl)
        credentials = wizard.onboard()

        assert len(credentials.pat_agent_ids) == 3
        assert len(credentials.sat_agent_ids) == 2

    def test_wizard_with_alpha_100_template(self, tmp_node_dir):
        tmpl = NodeTemplate.alpha_100()
        wizard = OnboardingWizard(node_dir=tmp_node_dir, template=tmpl)
        credentials = wizard.onboard()

        # Agent counts should still be default (7+5) since alpha_100
        # only overrides token economy and inference limits
        assert len(credentials.pat_agent_ids) == 7
        assert len(credentials.sat_agent_ids) == 5

    def test_agents_manifest_respects_template(self, tmp_node_dir):
        tmpl = NodeTemplate(pat_count=3, sat_count=2)
        wizard = OnboardingWizard(node_dir=tmp_node_dir, template=tmpl)
        wizard.onboard()

        agents_data = json.loads(wizard.agents_file.read_text())
        assert agents_data["total"] == 5
        assert len(agents_data["pat_agents"]) == 3
        assert len(agents_data["sat_agents"]) == 2

    def test_template_import_from_package(self):
        """NodeTemplate is importable from the core.pat package."""
        from core.pat import NodeTemplate as PkgNodeTemplate

        assert PkgNodeTemplate is NodeTemplate
