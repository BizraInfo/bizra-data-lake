"""
TeleScript Engine Tests — Phase 68.05
══════════════════════════════════════

TDD anchors for capability verification, path restrictions, policy merge.

Standing on Giants:
- Thompson (1984): Capability-based security
- Beck (2002): TDD by Example
"""

from __future__ import annotations

from core.bus.telescript import (
    EMPTY_POLICY,
    Capability,
    TeleScriptEngine,
    TeleScriptPolicy,
)


def _full_policy(**overrides) -> TeleScriptPolicy:
    """Build a policy with all capabilities allowed."""
    defaults = {
        "allow": frozenset(c.value for c in Capability),
        "deny": frozenset(),
        "allow_paths": (),
        "deny_paths": ("**/.env*", "**/.git/**", "**/.keys/**", "**/credentials*"),
        "require_attestation": (),
    }
    defaults.update(overrides)
    return TeleScriptPolicy(**defaults)


# ═══════════════════════════════════════════════════════════
# Basic capability checks
# ═══════════════════════════════════════════════════════════


class TestTeleScriptBasic:
    """Core capability verification."""

    def test_allowed_capability_passes(self) -> None:
        engine = TeleScriptEngine(_full_policy())
        v = engine.check(requested=("file_read",))
        assert v.allowed is True

    def test_denied_capability_fails(self) -> None:
        policy = _full_policy(deny=frozenset(["shell_execute"]))
        engine = TeleScriptEngine(policy)
        v = engine.check(requested=("shell_execute",))
        assert v.allowed is False
        assert "shell_execute" in v.denied_capabilities

    def test_unknown_capability_denied(self) -> None:
        engine = TeleScriptEngine(_full_policy())
        v = engine.check(requested=("launch_missiles",))
        assert v.allowed is False
        assert "launch_missiles" in v.denied_capabilities

    def test_empty_policy_denies_all(self) -> None:
        engine = TeleScriptEngine(EMPTY_POLICY)
        v = engine.check(requested=("file_read",))
        assert v.allowed is False

    def test_empty_requested_always_allowed(self) -> None:
        engine = TeleScriptEngine(EMPTY_POLICY)
        v = engine.check(requested=())
        assert v.allowed is True


# ═══════════════════════════════════════════════════════════
# Path restrictions
# ═══════════════════════════════════════════════════════════


class TestPathRestrictions:
    """File path glob pattern enforcement."""

    def test_allowed_path_passes(self) -> None:
        policy = _full_policy(allow_paths=("/home/user/projects/**",))
        engine = TeleScriptEngine(policy)
        v = engine.check(
            requested=("file_read",),
            file_path="/home/user/projects/foo.py",
        )
        assert v.allowed is True

    def test_denied_path_blocked(self) -> None:
        policy = _full_policy(
            deny_paths=(
                "**/.env*",
                "**/.git/**",
                "**/.keys/**",
                "**/credentials*",
                "**/secret/**",
            ),
        )
        engine = TeleScriptEngine(policy)
        v = engine.check(
            requested=("file_read",),
            file_path="/app/secret/key.pem",
        )
        assert v.allowed is False
        assert "Path denied" in v.reason

    def test_deny_overrides_allow_path(self) -> None:
        policy = _full_policy(
            allow_paths=("/app/**",),
            deny_paths=(
                "**/.env*",
                "**/.git/**",
                "**/.keys/**",
                "**/credentials*",
                "/app/.env.local",
            ),
        )
        engine = TeleScriptEngine(policy)
        v = engine.check(
            requested=("file_write",),
            file_path="/app/.env.local",
        )
        assert v.allowed is False

    def test_env_files_always_denied(self) -> None:
        engine = TeleScriptEngine(_full_policy())
        v = engine.check(
            requested=("file_read",),
            file_path="/app/.env.production",
        )
        assert v.allowed is False

    def test_git_dir_always_denied(self) -> None:
        engine = TeleScriptEngine(_full_policy())
        v = engine.check(
            requested=("file_write",),
            file_path="/repo/.git/config",
        )
        assert v.allowed is False


# ═══════════════════════════════════════════════════════════
# Policy merge
# ═══════════════════════════════════════════════════════════


class TestPolicyMerge:
    """Action-level telescript merge behavior."""

    def test_action_restricts_default(self) -> None:
        engine = TeleScriptEngine(_full_policy())
        v = engine.check(
            requested=("file_read",),
            action_telescript={"allow_capabilities": ["file_read"]},
        )
        assert v.allowed is True

    def test_action_cannot_expand_permissions(self) -> None:
        policy = _full_policy(allow=frozenset(["file_read"]))
        engine = TeleScriptEngine(policy)
        v = engine.check(
            requested=("shell_execute",),
            action_telescript={"allow_capabilities": ["shell_execute", "file_read"]},
        )
        assert v.allowed is False
        assert "shell_execute" in v.denied_capabilities

    def test_deny_lists_union(self) -> None:
        policy = _full_policy(deny=frozenset(["self_modify"]))
        engine = TeleScriptEngine(policy)
        v = engine.check(
            requested=("shell_execute",),
            action_telescript={"deny_capabilities": ["shell_execute"]},
        )
        assert v.allowed is False


# ═══════════════════════════════════════════════════════════
# Attestation
# ═══════════════════════════════════════════════════════════


class TestAttestation:
    """Capabilities requiring human approval."""

    def test_attestation_required_returns_verdict(self) -> None:
        policy = _full_policy(require_attestation=("self_modify",))
        engine = TeleScriptEngine(policy)
        v = engine.check(requested=("self_modify",))
        assert v.allowed is False
        assert "self_modify" in v.needs_attestation
        assert "Attestation required" in v.reason

    def test_network_requires_attestation(self) -> None:
        policy = _full_policy(require_attestation=("network_http",))
        engine = TeleScriptEngine(policy)
        v = engine.check(requested=("network_http",))
        assert v.allowed is False
        assert "network_http" in v.needs_attestation


# ═══════════════════════════════════════════════════════════
# Capability enum
# ═══════════════════════════════════════════════════════════


class TestCapabilityEnum:
    """Capability taxonomy."""

    def test_capability_count(self) -> None:
        assert len(Capability) == 16

    def test_capability_string_values(self) -> None:
        assert Capability.FILE_READ == "file_read"
        assert Capability.SHELL_EXECUTE == "shell_execute"
        assert Capability.SELF_MODIFY == "self_modify"
