"""
Proto-AaaS Agent Packager — Package, validate, and deploy BIZRA agents
=======================================================================
Creates distributable `.bizra-agent` archives from agent manifest directories.
Each archive is a self-contained deployment unit containing the manifest,
Telescript permit templates, and documentation.

Standing on Giants:
- General Magic (Telescript, 1994): Agents as packaged, portable entities
  with capabilities, budgets, and permits — the original mobile code.
- Shannon (1948): Minimal manifest schema = maximal signal; every field
  carries information, no redundancy.
- Al-Ghazali (1095): Permit templates enforce ethical constraints before
  the agent ever executes — excellence by construction.

Created: 2026-02-23 | BIZRA Agent Packager v1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.sovereign.permit import (
    Authority,
    Capability as TelescriptCapability,
    DEFAULT_TTL_SECONDS,
    MAX_ACTIONS_PER_PERMIT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

ARCHIVE_EXTENSION = ".bizra-agent"
MANIFEST_FILENAME = "manifest.yaml"
PERMIT_TEMPLATE_FILENAME = "permit_template.json"
README_FILENAME = "README.md"

# Map string capability names to Telescript Capability enum values
_CAPABILITY_NAME_MAP: dict[str, TelescriptCapability] = {
    "GO": TelescriptCapability.GO,
    "CREATE": TelescriptCapability.CREATE,
    "MEET": TelescriptCapability.MEET,
    "COMPUTE": TelescriptCapability.COMPUTE,
    "STORE": TelescriptCapability.STORE,
    "NETWORK": TelescriptCapability.NETWORK,
}


# ---------------------------------------------------------------------------
# AgentManifest — parsed representation of manifest.yaml
# ---------------------------------------------------------------------------


@dataclass
class AgentManifest:
    """Parsed agent manifest from YAML.

    Represents all configurable aspects of a BIZRA agent:
    identity, capabilities, schedule, persona, and quality gates.
    """

    name: str
    display_name: str
    version: str
    description: str
    capabilities_telescript: list[str] = field(default_factory=list)
    hda_skills: list[str] = field(default_factory=list)
    missions: list[dict[str, Any]] = field(default_factory=list)
    permit_defaults: dict[str, Any] = field(default_factory=dict)
    onboarding_questions: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    persona: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# load_manifest — Parse YAML into AgentManifest
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> AgentManifest:
    """Parse a manifest YAML file into an AgentManifest dataclass.

    Args:
        manifest_path: Path to the manifest.yaml file.

    Returns:
        Populated AgentManifest.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        KeyError: If required top-level keys are missing.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    agent_block: dict[str, Any] = raw.get("agent", {})
    capabilities_block: dict[str, Any] = raw.get("capabilities", {})
    schedule_block: dict[str, Any] = raw.get("schedule", {})
    permits_block: dict[str, Any] = raw.get("permits", {})
    onboarding_block: dict[str, Any] = raw.get("onboarding", {})

    return AgentManifest(
        name=agent_block.get("name", ""),
        display_name=agent_block.get("display_name", ""),
        version=agent_block.get("version", "0.0.0"),
        description=agent_block.get("description", "").strip(),
        capabilities_telescript=capabilities_block.get("telescript", []),
        hda_skills=raw.get("hda_skills", []),
        missions=schedule_block.get("missions", []),
        permit_defaults={
            "ttl_seconds": permits_block.get(
                "default_ttl_seconds", DEFAULT_TTL_SECONDS
            ),
            "max_actions": permits_block.get(
                "default_max_actions", MAX_ACTIONS_PER_PERMIT
            ),
            "max_tokens": permits_block.get("default_max_tokens", 4096),
            "auto_renew": permits_block.get("auto_renew", False),
            "signing_key_env": permits_block.get(
                "signing_key_env", "BIZRA_PERMIT_SIGNING_KEY"
            ),
        },
        onboarding_questions=onboarding_block.get("questions", []),
        models=raw.get("models", {}),
        quality=raw.get("quality", {}),
        persona=raw.get("persona", {}),
    )


# ---------------------------------------------------------------------------
# validate_manifest — field-level validation
# ---------------------------------------------------------------------------


def validate_manifest(manifest: AgentManifest) -> tuple[bool, list[str]]:
    """Validate all required fields in an AgentManifest.

    Returns:
        (valid, errors) where valid is True when no errors are found.
    """
    errors: list[str] = []

    # Identity checks
    if not manifest.name:
        errors.append("Missing required field: name")
    if not manifest.display_name:
        errors.append("Missing required field: display_name")
    if not manifest.version:
        errors.append("Missing required field: version")
    if not manifest.description:
        errors.append("Missing required field: description")

    # Capability checks
    if not manifest.capabilities_telescript:
        errors.append("Agent must declare at least one Telescript capability")

    # Validate capability names are recognized
    for cap_name in manifest.capabilities_telescript:
        if cap_name not in _CAPABILITY_NAME_MAP:
            errors.append(f"Unknown Telescript capability: {cap_name}")

    # Quality gate: Ihsan floor must meet constitutional minimum
    min_ihsan = manifest.quality.get("min_ihsan", UNIFIED_IHSAN_THRESHOLD)
    if min_ihsan < UNIFIED_IHSAN_THRESHOLD:
        errors.append(
            f"Quality min_ihsan ({min_ihsan}) is below constitutional "
            f"threshold ({UNIFIED_IHSAN_THRESHOLD})"
        )

    # Permit defaults: signing key must reference an env var, never a literal
    signing_key_env = manifest.permit_defaults.get("signing_key_env", "")
    if signing_key_env and not signing_key_env.isidentifier():
        errors.append(
            f"signing_key_env must be a valid environment variable name, "
            f"got: {signing_key_env}"
        )

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# create_permit_template — Telescript permit from manifest
# ---------------------------------------------------------------------------


def create_permit_template(
    manifest: AgentManifest, signing_key: str
) -> dict[str, Any]:
    """Generate a Telescript Permit template from manifest capabilities.

    The template captures all capability bindings and budget constraints
    so that a runtime Permit can be minted without re-parsing the manifest.

    Args:
        manifest: The parsed agent manifest.
        signing_key: HMAC signing key for permit creation.

    Returns:
        Dictionary with permit fields suitable for Permit.create().
    """
    # Resolve Telescript capabilities
    resolved_capabilities: list[str] = []
    capability_enums: list[int] = []
    for cap_name in manifest.capabilities_telescript:
        ts_cap = _CAPABILITY_NAME_MAP.get(cap_name)
        if ts_cap is not None:
            resolved_capabilities.append(cap_name)
            capability_enums.append(int(ts_cap))

    # Build template
    ttl = manifest.permit_defaults.get("ttl_seconds", DEFAULT_TTL_SECONDS)
    max_actions = manifest.permit_defaults.get(
        "max_actions", MAX_ACTIONS_PER_PERMIT
    )
    max_tokens = manifest.permit_defaults.get("max_tokens", 4096)

    # Compute a deterministic template hash for integrity verification
    template_content = json.dumps(
        {
            "agent": manifest.name,
            "version": manifest.version,
            "capabilities": sorted(resolved_capabilities),
            "ttl": ttl,
            "max_actions": max_actions,
        },
        sort_keys=True,
    )
    template_hash = hashlib.sha256(template_content.encode()).hexdigest()

    return {
        "agent_name": manifest.name,
        "agent_version": manifest.version,
        "capabilities": resolved_capabilities,
        "capability_enums": capability_enums,
        "ttl_seconds": ttl,
        "max_actions": max_actions,
        "max_tokens": max_tokens,
        "auto_renew": manifest.permit_defaults.get("auto_renew", False),
        "signing_key_env": manifest.permit_defaults.get(
            "signing_key_env", "BIZRA_PERMIT_SIGNING_KEY"
        ),
        "ihsan_floor": UNIFIED_IHSAN_THRESHOLD,
        "template_hash": template_hash,
    }


# ---------------------------------------------------------------------------
# _generate_readme — Auto-generate README for the archive
# ---------------------------------------------------------------------------


def _generate_readme(manifest: AgentManifest) -> str:
    """Generate a README.md for inclusion in the agent archive."""
    caps = ", ".join(manifest.capabilities_telescript) or "(none)"
    skills = ", ".join(manifest.hda_skills) or "(none)"
    mission_names = ", ".join(
        m.get("name", "unnamed") for m in manifest.missions
    ) or "(none)"

    return f"""\
# {manifest.display_name}

> {manifest.description}

**Version:** {manifest.version}
**Ihsan Floor:** {UNIFIED_IHSAN_THRESHOLD}

## Telescript Capabilities

{caps}

## HDA Skills

{skills}

## Scheduled Missions

{mission_names}

---

*Packaged by BIZRA Agent Packager v1.0*
*Standing on Giants: General Magic (1994) + Shannon (1948) + Al-Ghazali (1095)*
"""


# ---------------------------------------------------------------------------
# package_agent — create .bizra-agent archive
# ---------------------------------------------------------------------------


def package_agent(
    agent_dir: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Create a `.bizra-agent` ZIP archive from an agent directory.

    The archive contains:
    - manifest.yaml (the original manifest)
    - permit_template.json (pre-computed Telescript permit template)
    - README.md (auto-generated documentation)

    Args:
        agent_dir: Directory containing manifest.yaml.
        output_dir: Where to write the archive (defaults to agent_dir parent).

    Returns:
        Path to the created archive.

    Raises:
        FileNotFoundError: If agent_dir or manifest.yaml does not exist.
        ValueError: If the manifest is invalid.
    """
    agent_dir = Path(agent_dir)
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    manifest_path = agent_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} in {agent_dir}"
        )

    # Parse and validate
    manifest = load_manifest(manifest_path)
    valid, errors = validate_manifest(manifest)
    if not valid:
        raise ValueError(
            f"Invalid manifest: {'; '.join(errors)}"
        )

    # Generate permit template (use a placeholder key for template generation)
    permit_template = create_permit_template(manifest, signing_key="")

    # Generate README
    readme_content = _generate_readme(manifest)

    # Determine output path
    if output_dir is None:
        output_dir = agent_dir.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"{manifest.name}-{manifest.version}{ARCHIVE_EXTENSION}"
    archive_path = output_dir / archive_name

    # Create ZIP archive
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include original manifest
        zf.write(manifest_path, MANIFEST_FILENAME)

        # Include permit template as JSON
        zf.writestr(
            PERMIT_TEMPLATE_FILENAME,
            json.dumps(permit_template, indent=2, sort_keys=True),
        )

        # Include README
        zf.writestr(README_FILENAME, readme_content)

    logger.info("Packaged agent '%s' -> %s", manifest.name, archive_path)
    return archive_path


# ---------------------------------------------------------------------------
# load_agent — extract archive and return manifest
# ---------------------------------------------------------------------------


def load_agent(
    archive_path: Path,
    target_dir: Optional[Path] = None,
) -> AgentManifest:
    """Extract a `.bizra-agent` archive and return the parsed manifest.

    Args:
        archive_path: Path to the .bizra-agent ZIP archive.
        target_dir: Where to extract (defaults to a temp directory).

    Returns:
        Parsed AgentManifest from the extracted archive.

    Raises:
        FileNotFoundError: If the archive does not exist.
        zipfile.BadZipFile: If the file is not a valid ZIP.
        KeyError: If manifest.yaml is missing from the archive.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="bizra-agent-"))
    else:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if MANIFEST_FILENAME not in names:
            raise KeyError(
                f"{MANIFEST_FILENAME} not found in archive {archive_path}"
            )
        zf.extractall(target_dir)

    manifest_path = target_dir / MANIFEST_FILENAME
    manifest = load_manifest(manifest_path)

    logger.info(
        "Loaded agent '%s' v%s from %s",
        manifest.name,
        manifest.version,
        archive_path,
    )
    return manifest
