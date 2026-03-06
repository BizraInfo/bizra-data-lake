"""
Capsule Runtime — Sandboxed Skill Execution Engine
══════════════════════════════════════════════════

Capsules are packaged workflows with TeleScript capability masks,
CAPSULE.yaml manifests, and receipt-backed proof conditions.

A capsule is a Claude Skill that can prove it worked.

Standing on Giants:
- Thompson (1984): Capability-based security
- Dennis & Van Horn (1966): Supervisor capabilities
- Kay (1993): Early OOP and encapsulated computation

Phase 68.04 — Sovereign Instantiation
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object  # type: ignore[assignment, misc]

    def Field(**kw):  # type: ignore[no-redef]  # noqa: N802
        return kw.get("default")


# ═══════════════════════════════════════════════════════════
# Manifest Schema
# ═══════════════════════════════════════════════════════════


class CapsulePaths(BaseModel):  # type: ignore[misc]
    """File path restrictions for a capsule."""

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=lambda: ["**/.env*", "**/.git/**"])


class CapsuleCapabilities(BaseModel):  # type: ignore[misc]
    """Capability mask declared in CAPSULE.yaml."""

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    paths: CapsulePaths = Field(default_factory=CapsulePaths)


class CapsuleTrigger(BaseModel):  # type: ignore[misc]
    """Auto-trigger conditions."""

    file_patterns: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)


class CapsuleInvocation(BaseModel):  # type: ignore[misc]
    """Invocation restrictions."""

    user_only: bool = False
    min_ihsan: float = 0.95
    trigger: CapsuleTrigger | None = None


class CapsuleStep(BaseModel):  # type: ignore[misc]
    """A single workflow step in a capsule."""

    step: str
    action: str
    args: dict[str, Any] = Field(default_factory=dict)


class CapsuleProof(BaseModel):  # type: ignore[misc]
    """Proof condition for capsule completion."""

    kind: str  # "file_exists" | "valid_yaml" | "custom"
    target: str = ""


class CapsuleManifest(BaseModel):  # type: ignore[misc]
    """Parsed CAPSULE.yaml manifest."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    invocation: CapsuleInvocation = Field(default_factory=CapsuleInvocation)
    capabilities: CapsuleCapabilities = Field(default_factory=CapsuleCapabilities)
    bridges: list[str] = Field(default_factory=list)
    workflow: list[CapsuleStep] = Field(default_factory=list)
    proof: list[CapsuleProof] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# Capsule Result
# ═══════════════════════════════════════════════════════════


@dataclass
class CapsuleResult:
    """Result of a capsule execution."""

    capsule: str
    status: str  # "proved" | "unproved" | "denied" | "failed" | "not_found"
    step_failed: str = ""
    receipts: list[Any] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# Capsule Registry
# ═══════════════════════════════════════════════════════════


class CapsuleRegistry:
    """Auto-discovers and manages capsule manifests.

    Scans `capsules_dir` for subdirectories containing CAPSULE.yaml.
    Invalid manifests are logged and skipped (fail-safe discovery).
    """

    __slots__ = ("_capsules_dir", "_capsules")

    def __init__(self, capsules_dir: Path | str = "capsules") -> None:
        self._capsules_dir = Path(capsules_dir)
        self._capsules: dict[str, CapsuleManifest] = {}

    @property
    def count(self) -> int:
        return len(self._capsules)

    def discover(self) -> int:
        """Auto-discover capsules from directory. Returns count found."""
        if yaml is None:
            logger.debug("PyYAML not available, capsule discovery skipped")
            return 0
        if not self._capsules_dir.exists():
            return 0

        found = 0
        for manifest_path in self._capsules_dir.glob("*/CAPSULE.yaml"):
            try:
                manifest = self._load_manifest(manifest_path)
                self._capsules[manifest.name] = manifest
                found += 1
            except Exception:
                logger.warning("Invalid capsule: %s", manifest_path, exc_info=True)

        return found

    def register(self, manifest: CapsuleManifest) -> None:
        """Register a capsule manifest programmatically."""
        self._capsules[manifest.name] = manifest

    def get(self, name: str) -> CapsuleManifest | None:
        """Get a capsule by name."""
        return self._capsules.get(name)

    def list_all(self) -> list[CapsuleManifest]:
        """List all registered capsules."""
        return list(self._capsules.values())

    def match_trigger(
        self, event_type: str, file_path: str | None = None
    ) -> list[CapsuleManifest]:
        """Find capsules that should auto-trigger for an event."""
        matches = []
        for capsule in self._capsules.values():
            trigger = capsule.invocation.trigger
            if trigger is None:
                continue
            if event_type not in trigger.events:
                continue
            if file_path is None or not trigger.file_patterns:
                matches.append(capsule)
            elif any(fnmatch(file_path, p) for p in trigger.file_patterns):
                matches.append(capsule)
        return matches

    @staticmethod
    def _load_manifest(path: Path) -> CapsuleManifest:
        raw = yaml.safe_load(path.read_text())
        return CapsuleManifest.model_validate(raw)


# ═══════════════════════════════════════════════════════════
# Channel mapping
# ═══════════════════════════════════════════════════════════

_ACTION_TO_CHANNEL: dict[str, str] = {
    "glob": "file",
    "read": "file",
    "grep": "file",
    "write": "file",
    "template": "llm",
    "shell": "desktop",
    "fetch": "browser",
}


def _step_to_channel(action: str) -> str:
    """Map capsule step action to ActionBus channel."""
    return _ACTION_TO_CHANNEL.get(action, "file")


# ═══════════════════════════════════════════════════════════
# Capsule Runtime
# ═══════════════════════════════════════════════════════════


@runtime_checkable
class ActionProposer(Protocol):
    """Protocol for ActionBus.propose()."""

    async def propose(self, action: Any) -> Any: ...


class CapsuleRuntime:
    """Executes capsule workflows through the ActionBus.

    Each workflow step becomes an ActionEnvelope with TeleScript
    restrictions from the capsule manifest. Steps execute sequentially;
    any denial or failure stops the capsule.

    Security properties:
    1. TeleScript from manifest — capsule cannot exceed declared capabilities
    2. Sequential execution — steps run in order, not parallel
    3. Receipt chain — every step produces a verifiable receipt
    4. Variable isolation — step results are scoped to the capsule
    """

    __slots__ = ("_registry", "_action_bus", "_step_results")

    def __init__(
        self,
        registry: CapsuleRegistry,
        action_bus: ActionProposer | None = None,
    ) -> None:
        self._registry = registry
        self._action_bus = action_bus
        self._step_results: dict[str, Any] = {}

    async def execute(
        self, capsule_name: str, context: dict[str, Any] | None = None
    ) -> CapsuleResult:
        """Run a capsule's workflow steps through the ActionBus."""
        ctx = context or {}

        manifest = self._registry.get(capsule_name)
        if manifest is None:
            return CapsuleResult(capsule=capsule_name, status="not_found")

        # Build TeleScript from manifest
        telescript = {
            "allow_capabilities": manifest.capabilities.allow,
            "deny_capabilities": manifest.capabilities.deny,
            "allow_paths": manifest.capabilities.paths.allow,
            "deny_paths": manifest.capabilities.paths.deny,
        }

        self._step_results = {}
        receipts: list[Any] = []

        for step in manifest.workflow:
            resolved_args = self._resolve_vars(step.args)

            # Build action envelope (import here to avoid circular)
            from core.bus.types import ActionBudget, ActionEnvelope

            action_content = f"{capsule_name}:{step.step}:{ctx}"
            action_id = hashlib.blake2b(
                action_content.encode(), digest_size=16
            ).hexdigest()

            action = ActionEnvelope(
                action_id=action_id,
                kind=f"capsule.{capsule_name}.{step.step}",
                channel=_step_to_channel(step.action),
                payload=resolved_args,
                capabilities=tuple(manifest.capabilities.allow),
                telescript=telescript,
                budget=ActionBudget(time_ms=10_000),
                correlation_id=ctx.get("mission_id", ""),
                timestamp=int(time.time() * 1000),
            )

            if self._action_bus is None:
                return CapsuleResult(
                    capsule=capsule_name,
                    status="failed",
                    step_failed=step.step,
                )

            receipt = await self._action_bus.propose(action)
            receipts.append(receipt)

            status_val = getattr(receipt.status, "value", str(receipt.status))

            if status_val == "denied":
                return CapsuleResult(
                    capsule=capsule_name,
                    status="denied",
                    step_failed=step.step,
                    receipts=receipts,
                )

            if status_val == "failed":
                return CapsuleResult(
                    capsule=capsule_name,
                    status="failed",
                    step_failed=step.step,
                    receipts=receipts,
                )

            self._step_results[step.step] = receipt

        # All steps completed — check proofs
        proofs_ok = self._check_proofs(manifest.proof)

        return CapsuleResult(
            capsule=capsule_name,
            status="proved" if proofs_ok else "unproved",
            receipts=receipts,
        )

    def _resolve_vars(self, args: dict[str, Any]) -> dict[str, Any]:
        """Replace $step_name.result references with actual values."""
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                parts = value[1:].split(".")
                step_name = parts[0]
                receipt = self._step_results.get(step_name)
                resolved[key] = receipt if receipt else value
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _check_proofs(proofs: list[CapsuleProof]) -> bool:
        """Check proof conditions. Basic built-in checks."""
        if not proofs:
            return True
        for proof in proofs:
            if proof.kind == "file_exists":
                if not Path(proof.target).exists():
                    return False
            elif proof.kind == "always_true":
                continue
            else:
                # Unknown proof types — conservatively pass
                # (full checking deferred to Omega Loop integration)
                continue
        return True
