"""
Capsule Runtime Tests — Phase 68.04
════════════════════════════════════

TDD anchors for capsule discovery, execution, variable resolution,
denial/failure handling, and proof conditions.

Standing on Giants:
- Beck (2002): TDD by Example
- Thompson (1984): Capability-based security
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.bus.capsule import (
    CapsuleCapabilities,
    CapsuleInvocation,
    CapsuleManifest,
    CapsulePaths,
    CapsuleProof,
    CapsuleRegistry,
    CapsuleResult,
    CapsuleRuntime,
    CapsuleStep,
    CapsuleTrigger,
)

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _manifest(
    name: str = "test-capsule",
    capabilities: list[str] | None = None,
    workflow: list[dict] | None = None,
    proofs: list[dict] | None = None,
    trigger_events: list[str] | None = None,
    trigger_patterns: list[str] | None = None,
) -> CapsuleManifest:
    caps = CapsuleCapabilities(
        allow=capabilities or ["file_read", "grep"],
        deny=[],
        paths=CapsulePaths(allow=["./src/**"], deny=["**/.env*"]),
    )
    steps = [
        CapsuleStep(**s)
        for s in (workflow or [{"step": "read_files", "action": "read", "args": {}}])
    ]
    proof_list = [CapsuleProof(**p) for p in (proofs or [])]
    trigger = None
    if trigger_events:
        trigger = CapsuleTrigger(
            events=trigger_events,
            file_patterns=trigger_patterns or [],
        )
    return CapsuleManifest(
        name=name,
        capabilities=caps,
        workflow=steps,
        proof=proof_list,
        invocation=CapsuleInvocation(trigger=trigger),
    )


def _mock_receipt(status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        status=SimpleNamespace(value=status),
        action_id="a-test",
        receipt_id="r-test",
        outcome_hash="h-test",
    )


def _mock_action_bus(status: str = "completed") -> AsyncMock:
    bus = AsyncMock()
    bus.propose = AsyncMock(return_value=_mock_receipt(status))
    return bus


# ═══════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════


class TestCapsuleDiscovery:
    """Auto-discover capsules from directory."""

    def test_discover_finds_capsules_in_dir(self, tmp_path) -> None:
        pytest.importorskip("yaml")
        import yaml as _yaml

        capsule_dir = tmp_path / "capsules" / "my-cap"
        capsule_dir.mkdir(parents=True)
        manifest = {
            "name": "my-cap",
            "version": "1.0.0",
            "capabilities": {"allow": ["file_read"]},
            "workflow": [{"step": "s1", "action": "read", "args": {}}],
        }
        (capsule_dir / "CAPSULE.yaml").write_text(_yaml.dump(manifest))

        registry = CapsuleRegistry(tmp_path / "capsules")
        found = registry.discover()
        assert found == 1
        assert registry.get("my-cap") is not None

    def test_invalid_manifest_skipped(self, tmp_path) -> None:
        pytest.importorskip("yaml")
        capsule_dir = tmp_path / "capsules" / "bad"
        capsule_dir.mkdir(parents=True)
        (capsule_dir / "CAPSULE.yaml").write_text("not: valid: yaml: [[[")

        registry = CapsuleRegistry(tmp_path / "capsules")
        found = registry.discover()
        assert found == 0

    def test_match_trigger_by_event(self) -> None:
        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                name="triggered",
                trigger_events=["action.receipt"],
            )
        )
        registry.register(_manifest(name="no-trigger"))

        matches = registry.match_trigger("action.receipt")
        assert len(matches) == 1
        assert matches[0].name == "triggered"

    def test_match_trigger_by_file_pattern(self) -> None:
        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                name="route-watcher",
                trigger_events=["action.receipt"],
                trigger_patterns=["*.py"],
            )
        )
        matches = registry.match_trigger("action.receipt", "auth.py")
        assert len(matches) == 1

        matches = registry.match_trigger("action.receipt", "README.md")
        assert len(matches) == 0


# ═══════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════


class TestCapsuleExecution:
    """Workflow step execution."""

    @pytest.mark.asyncio
    async def test_execute_all_steps_succeeds(self) -> None:
        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                workflow=[
                    {"step": "s1", "action": "read", "args": {}},
                    {"step": "s2", "action": "grep", "args": {}},
                ],
                proofs=[{"kind": "always_true"}],
            )
        )
        runtime = CapsuleRuntime(registry, _mock_action_bus())
        result = await runtime.execute("test-capsule")
        assert result.status == "proved"
        assert len(result.receipts) == 2

    @pytest.mark.asyncio
    async def test_denied_step_stops_execution(self) -> None:
        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                workflow=[
                    {"step": "s1", "action": "read", "args": {}},
                    {"step": "s2", "action": "write", "args": {}},
                ],
            )
        )
        runtime = CapsuleRuntime(registry, _mock_action_bus(status="denied"))
        result = await runtime.execute("test-capsule")
        assert result.status == "denied"
        assert result.step_failed == "s1"

    @pytest.mark.asyncio
    async def test_failed_step_stops_execution(self) -> None:
        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                workflow=[{"step": "s1", "action": "shell", "args": {}}],
            )
        )
        runtime = CapsuleRuntime(registry, _mock_action_bus(status="failed"))
        result = await runtime.execute("test-capsule")
        assert result.status == "failed"
        assert result.step_failed == "s1"

    @pytest.mark.asyncio
    async def test_variable_resolution_between_steps(self) -> None:
        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                workflow=[
                    {"step": "discover", "action": "glob", "args": {"pattern": "*.py"}},
                    {
                        "step": "extract",
                        "action": "read",
                        "args": {"files": "$discover.result"},
                    },
                ],
                proofs=[{"kind": "always_true"}],
            )
        )

        bus = _mock_action_bus()
        runtime = CapsuleRuntime(registry, bus)
        result = await runtime.execute("test-capsule")
        assert result.status == "proved"
        # Second call should have resolved $discover.result
        assert bus.propose.call_count == 2

    @pytest.mark.asyncio
    async def test_not_found_capsule(self) -> None:
        registry = CapsuleRegistry()
        runtime = CapsuleRuntime(registry)
        result = await runtime.execute("nonexistent")
        assert result.status == "not_found"


# ═══════════════════════════════════════════════════════════
# Proof conditions
# ═══════════════════════════════════════════════════════════


class TestCapsuleProofs:
    """Proof checking after workflow completion."""

    @pytest.mark.asyncio
    async def test_proof_conditions_checked_after_workflow(self, tmp_path) -> None:
        # Create target file so file_exists proof passes
        target = tmp_path / "output.yaml"
        target.write_text("content: true")

        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                workflow=[{"step": "gen", "action": "write", "args": {}}],
                proofs=[{"kind": "file_exists", "target": str(target)}],
            )
        )
        runtime = CapsuleRuntime(registry, _mock_action_bus())
        result = await runtime.execute("test-capsule")
        assert result.status == "proved"

    @pytest.mark.asyncio
    async def test_unproved_capsule_returns_unproved(self) -> None:
        registry = CapsuleRegistry()
        registry.register(
            _manifest(
                workflow=[{"step": "gen", "action": "write", "args": {}}],
                proofs=[
                    {"kind": "file_exists", "target": "/nonexistent/path/file.xyz"}
                ],
            )
        )
        runtime = CapsuleRuntime(registry, _mock_action_bus())
        result = await runtime.execute("test-capsule")
        assert result.status == "unproved"
