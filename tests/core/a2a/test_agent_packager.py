"""
Tests for core.a2a.agent_packager — Agent packaging, validation, and deployment.

Covers:
- Manifest loading from real and synthetic YAML
- Field-level validation (happy path + missing fields)
- Archive creation and ZIP integrity
- Roundtrip: package -> load -> verify
- Permit template generation
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from core.a2a.agent_packager import (
    ARCHIVE_EXTENSION,
    MANIFEST_FILENAME,
    PERMIT_TEMPLATE_FILENAME,
    README_FILENAME,
    AgentManifest,
    create_permit_template,
    load_agent,
    load_manifest,
    package_agent,
    validate_manifest,
)
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Cross-platform: resolve from repo root instead of hardcoded WSL path
_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/core/a2a → repo root
FOUNDER_OPS_MANIFEST = _REPO_ROOT / "agents" / "founder_ops" / "manifest.yaml"


@pytest.fixture
def founder_manifest() -> AgentManifest:
    """Load the real founder_ops manifest."""
    return load_manifest(FOUNDER_OPS_MANIFEST)


@pytest.fixture
def minimal_manifest(tmp_path: Path) -> Path:
    """Create a minimal valid manifest.yaml in a temp directory."""
    manifest_data = {
        "agent": {
            "name": "test-agent",
            "display_name": "Test Agent",
            "version": "0.1.0",
            "description": "A minimal test agent for unit testing.",
        },
        "capabilities": {
            "telescript": ["GO", "COMPUTE"],
        },
        "hda_skills": ["open_app", "type_text"],
        "schedule": {
            "missions": [
                {
                    "name": "heartbeat",
                    "description": "Basic health check",
                    "cron": "*/30 * * * *",
                    "auto_execute": True,
                }
            ],
        },
        "permits": {
            "default_ttl_seconds": 120,
            "default_max_actions": 10,
            "default_max_tokens": 2048,
            "signing_key_env": "BIZRA_PERMIT_SIGNING_KEY",
        },
        "onboarding": {"questions": []},
        "models": {"primary_backend": "ollama"},
        "quality": {"min_ihsan": 0.95, "min_snr": 0.85},
        "persona": {"role": "tester", "tone": "neutral"},
    }

    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()
    manifest_path = agent_dir / "manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        yaml.dump(manifest_data, fh, default_flow_style=False)

    return agent_dir


# ---------------------------------------------------------------------------
# Test: load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_load_manifest(self, founder_manifest: AgentManifest) -> None:
        """Load the real founder_ops manifest and verify core fields."""
        m = founder_manifest
        assert m.name == "founder-ops-agent"
        assert m.display_name == "Founder Ops Agent"
        assert m.version == "1.0.0"
        assert len(m.description) > 10
        assert "GO" in m.capabilities_telescript
        assert "COMPUTE" in m.capabilities_telescript
        assert "open_app" in m.hda_skills
        assert len(m.missions) >= 1
        assert m.permit_defaults["ttl_seconds"] == 300
        assert m.models.get("primary_backend") == "lm_studio"

    def test_load_manifest_missing_file(self, tmp_path: Path) -> None:
        """Loading a non-existent manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Test: validate_manifest
# ---------------------------------------------------------------------------


class TestValidateManifest:
    def test_validate_manifest_valid(self, founder_manifest: AgentManifest) -> None:
        """A well-formed manifest passes validation."""
        valid, errors = validate_manifest(founder_manifest)
        assert valid is True, f"Unexpected errors: {errors}"
        assert errors == []

    def test_validate_manifest_missing_name(
        self, founder_manifest: AgentManifest
    ) -> None:
        """Missing name fails validation."""
        broken = replace(founder_manifest, name="")
        valid, errors = validate_manifest(broken)
        assert valid is False
        assert any("name" in e.lower() for e in errors)

    def test_validate_manifest_no_capabilities(
        self, founder_manifest: AgentManifest
    ) -> None:
        """Empty Telescript capabilities list fails validation."""
        broken = replace(founder_manifest, capabilities_telescript=[])
        valid, errors = validate_manifest(broken)
        assert valid is False
        assert any("capability" in e.lower() for e in errors)

    def test_validate_manifest_unknown_capability(
        self, founder_manifest: AgentManifest
    ) -> None:
        """Unknown capability name triggers an error."""
        broken = replace(
            founder_manifest,
            capabilities_telescript=["GO", "TELEPORT"],
        )
        valid, errors = validate_manifest(broken)
        assert valid is False
        assert any("TELEPORT" in e for e in errors)

    def test_validate_manifest_ihsan_below_threshold(
        self, founder_manifest: AgentManifest
    ) -> None:
        """Ihsan below constitutional threshold fails."""
        broken = replace(
            founder_manifest,
            quality={"min_ihsan": 0.50},
        )
        valid, errors = validate_manifest(broken)
        assert valid is False
        assert any("ihsan" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Test: package_agent
# ---------------------------------------------------------------------------


class TestPackageAgent:
    def test_package_agent(self, minimal_manifest: Path, tmp_path: Path) -> None:
        """Create .bizra-agent archive and verify it is a valid ZIP."""
        output_dir = tmp_path / "output"
        archive = package_agent(minimal_manifest, output_dir=output_dir)

        assert archive.exists()
        assert archive.suffix == ARCHIVE_EXTENSION
        assert zipfile.is_zipfile(archive)

        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            assert MANIFEST_FILENAME in names

    def test_package_includes_readme(
        self, minimal_manifest: Path, tmp_path: Path
    ) -> None:
        """Archive contains a README.md with agent metadata."""
        output_dir = tmp_path / "output"
        archive = package_agent(minimal_manifest, output_dir=output_dir)

        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            assert README_FILENAME in names

            readme_text = zf.read(README_FILENAME).decode("utf-8")
            assert "Test Agent" in readme_text
            assert "GO" in readme_text

    def test_package_includes_permit_template(
        self, minimal_manifest: Path, tmp_path: Path
    ) -> None:
        """Archive contains a permit_template.json."""
        output_dir = tmp_path / "output"
        archive = package_agent(minimal_manifest, output_dir=output_dir)

        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            assert PERMIT_TEMPLATE_FILENAME in names

            template = json.loads(zf.read(PERMIT_TEMPLATE_FILENAME).decode("utf-8"))
            assert template["agent_name"] == "test-agent"

    def test_package_agent_no_manifest(self, tmp_path: Path) -> None:
        """Packaging a directory without manifest.yaml raises."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            package_agent(empty_dir)


# ---------------------------------------------------------------------------
# Test: load_agent (roundtrip)
# ---------------------------------------------------------------------------


class TestLoadAgent:
    def test_load_agent(self, minimal_manifest: Path, tmp_path: Path) -> None:
        """Package then load — verify the manifest roundtrips correctly."""
        output_dir = tmp_path / "output"
        archive = package_agent(minimal_manifest, output_dir=output_dir)

        extract_dir = tmp_path / "extracted"
        loaded = load_agent(archive, target_dir=extract_dir)

        assert loaded.name == "test-agent"
        assert loaded.display_name == "Test Agent"
        assert loaded.version == "0.1.0"
        assert "GO" in loaded.capabilities_telescript
        assert "COMPUTE" in loaded.capabilities_telescript
        assert "open_app" in loaded.hda_skills

    def test_load_agent_missing_archive(self, tmp_path: Path) -> None:
        """Loading a non-existent archive raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_agent(tmp_path / "ghost.bizra-agent")


# ---------------------------------------------------------------------------
# Test: create_permit_template
# ---------------------------------------------------------------------------


class TestCreatePermitTemplate:
    def test_create_permit_template(self, founder_manifest: AgentManifest) -> None:
        """Template includes correct capabilities from the manifest."""
        template = create_permit_template(founder_manifest, signing_key="test-key-123")

        assert template["agent_name"] == "founder-ops-agent"
        assert template["agent_version"] == "1.0.0"
        assert "GO" in template["capabilities"]
        assert "COMPUTE" in template["capabilities"]
        assert "STORE" in template["capabilities"]
        assert "NETWORK" in template["capabilities"]
        assert template["ihsan_floor"] == UNIFIED_IHSAN_THRESHOLD
        assert template["ttl_seconds"] == 300
        assert template["max_actions"] == 30
        assert "template_hash" in template
        assert len(template["template_hash"]) == 64  # SHA-256 hex

    def test_permit_template_capability_enums(
        self, founder_manifest: AgentManifest
    ) -> None:
        """Capability enums are integer values from Telescript."""
        template = create_permit_template(founder_manifest, signing_key="test-key")
        enums = template["capability_enums"]
        assert isinstance(enums, list)
        assert all(isinstance(e, int) for e in enums)
        assert len(enums) == len(template["capabilities"])
