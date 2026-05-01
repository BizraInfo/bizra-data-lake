"""
Tests for BIZRA CLI Command Registry, commands, hooks, and performance.

Standing on Giants:
- Deming (1950): Measure everything
- Thompson & Ritchie (1973): Unix command behavior contracts
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.cli.hooks import CLIHooksManager
from core.cli.registry import CommandEntry, CommandRegistry, CommandResult
from core.cli.shared import VERSION

# ── Fixtures ─────────────────────────────────────────────────────────


class FakeCommand:
    """Minimal command for testing the registry."""

    def __init__(
        self,
        name: str = "fake",
        aliases: tuple = ("f",),
        description: str = "A fake command",
        category: str = "test",
        result: Optional[CommandResult] = None,
        side_effect: Optional[Exception] = None,
    ):
        self.name = name
        self.aliases = aliases
        self.description = description
        self.category = category
        self._result = result or CommandResult.ok("fake ok")
        self._side_effect = side_effect
        self.last_args: Optional[List[str]] = None

    def execute(self, args: List[str]) -> CommandResult:
        self.last_args = args
        if self._side_effect:
            raise self._side_effect
        return self._result


@pytest.fixture
def registry() -> CommandRegistry:
    return CommandRegistry()


@pytest.fixture
def fake_cmd() -> FakeCommand:
    return FakeCommand()


@pytest.fixture
def populated_registry() -> CommandRegistry:
    r = CommandRegistry()
    r.register(FakeCommand(name="doctor", aliases=("doc", "check"), category="system"))
    r.register(FakeCommand(name="start", aliases=("up",), category="lifecycle"))
    r.register(FakeCommand(name="status", aliases=("health", "s"), category="system"))
    r.register(FakeCommand(name="mission", aliases=("m", "do"), category="ops"))
    return r


# ═══════════════════════════════════════════════════════════════════
# § Registry Core
# ═══════════════════════════════════════════════════════════════════


class TestCommandResult:
    def test_ok(self):
        r = CommandResult.ok("done")
        assert r.success is True
        assert r.exit_code == 0
        assert r.message == "done"

    def test_error(self):
        r = CommandResult.error("fail", exit_code=2)
        assert r.success is False
        assert r.exit_code == 2

    def test_info(self):
        r = CommandResult.info("note")
        assert r.success is True
        assert r.message == "note"

    def test_ok_with_data(self):
        r = CommandResult.ok(data={"key": "val"})
        assert r.data == {"key": "val"}


class TestRegistration:
    def test_register_command(self, registry: CommandRegistry, fake_cmd: FakeCommand):
        registry.register(fake_cmd)
        assert "fake" in registry.command_names

    def test_register_creates_category(
        self, registry: CommandRegistry, fake_cmd: FakeCommand
    ):
        registry.register(fake_cmd)
        cats = registry.list_commands()
        assert "test" in cats
        assert "fake" in cats["test"]

    def test_multiple_commands(self, populated_registry: CommandRegistry):
        assert len(populated_registry.command_names) == 4


class TestResolve:
    def test_resolve_by_name(self, registry: CommandRegistry, fake_cmd: FakeCommand):
        registry.register(fake_cmd)
        entry = registry.resolve("fake")
        assert entry is not None
        assert entry.command.name == "fake"

    def test_resolve_by_alias(self, registry: CommandRegistry, fake_cmd: FakeCommand):
        registry.register(fake_cmd)
        entry = registry.resolve("f")
        assert entry is not None
        assert entry.command.name == "fake"

    def test_resolve_strips_dashes(
        self, registry: CommandRegistry, fake_cmd: FakeCommand
    ):
        registry.register(fake_cmd)
        entry = registry.resolve("--fake")
        assert entry is not None

    def test_resolve_case_insensitive(
        self, registry: CommandRegistry, fake_cmd: FakeCommand
    ):
        registry.register(fake_cmd)
        entry = registry.resolve("FAKE")
        assert entry is not None

    def test_resolve_missing(self, registry: CommandRegistry):
        assert registry.resolve("nonexistent") is None


class TestFuzzyMatch:
    def test_suggest_close_match(self, populated_registry: CommandRegistry):
        suggestions = populated_registry.suggest("doctr")
        assert "doctor" in suggestions

    def test_suggest_returns_empty_for_gibberish(
        self, populated_registry: CommandRegistry
    ):
        suggestions = populated_registry.suggest("zzzzz")
        assert suggestions == []


class TestDispatch:
    def test_dispatch_success(self, registry: CommandRegistry, fake_cmd: FakeCommand):
        registry.register(fake_cmd)
        result = registry.dispatch(["fake", "arg1", "arg2"])
        assert result.success is True
        assert fake_cmd.last_args == ["arg1", "arg2"]

    def test_dispatch_alias(self, registry: CommandRegistry, fake_cmd: FakeCommand):
        registry.register(fake_cmd)
        result = registry.dispatch(["f"])
        assert result.success is True

    def test_dispatch_empty_args(self, registry: CommandRegistry):
        result = registry.dispatch([])
        assert result.success is False

    def test_dispatch_unknown_command(self, registry: CommandRegistry):
        result = registry.dispatch(["nonexistent"])
        assert result.success is False
        assert "Unknown command" in result.message

    def test_dispatch_unknown_suggests(self, populated_registry: CommandRegistry):
        result = populated_registry.dispatch(["doctr"])
        assert "doctor" in result.message

    def test_dispatch_exception(self, registry: CommandRegistry):
        cmd = FakeCommand(side_effect=RuntimeError("boom"))
        registry.register(cmd)
        result = registry.dispatch(["fake"])
        assert result.success is False
        assert "boom" in result.message

    def test_dispatch_keyboard_interrupt(self, registry: CommandRegistry):
        cmd = FakeCommand(side_effect=KeyboardInterrupt())
        registry.register(cmd)
        result = registry.dispatch(["fake"])
        assert result.success is False
        assert result.exit_code == 130


# ═══════════════════════════════════════════════════════════════════
# § Performance Metrics
# ═══════════════════════════════════════════════════════════════════


class TestPerformanceTracking:
    def test_metrics_after_dispatch(
        self, registry: CommandRegistry, fake_cmd: FakeCommand
    ):
        registry.register(fake_cmd)
        registry.dispatch(["fake"])
        metrics = registry.get_metrics()
        assert "fake" in metrics
        assert metrics["fake"]["calls"] == 1
        assert metrics["fake"]["success_rate"] == 1.0

    def test_metrics_accumulate(self, registry: CommandRegistry, fake_cmd: FakeCommand):
        registry.register(fake_cmd)
        for _ in range(5):
            registry.dispatch(["fake"])
        metrics = registry.get_metrics()
        assert metrics["fake"]["calls"] == 5

    def test_metrics_track_errors(self, registry: CommandRegistry):
        cmd = FakeCommand(result=CommandResult.error("fail"))
        # The command returns error but doesn't raise — that's still "success" in dispatch
        # because the command executed. Only exceptions count as errors in metrics.
        registry.register(cmd)
        registry.dispatch(["fake"])
        # The command itself returned error result but execution succeeded
        metrics = registry.get_metrics()
        assert metrics["fake"]["calls"] == 1

    def test_p95_latency(self, registry: CommandRegistry, fake_cmd: FakeCommand):
        registry.register(fake_cmd)
        for _ in range(20):
            registry.dispatch(["fake"])
        metrics = registry.get_metrics()
        assert metrics["fake"]["p95_ms"] >= 0

    def test_command_entry_record(self):
        cmd = FakeCommand()
        entry = CommandEntry(command=cmd)
        entry.record(10.0, True)
        entry.record(20.0, True)
        entry.record(30.0, False)
        assert entry.call_count == 3
        assert entry.error_count == 1
        assert entry.avg_ms == 20.0
        assert entry.success_rate == pytest.approx(2 / 3)


# ═══════════════════════════════════════════════════════════════════
# § Hooks
# ═══════════════════════════════════════════════════════════════════


class TestCLIHooksManager:
    def test_pre_command_records_event(self):
        hooks = CLIHooksManager()
        hooks.pre_command("doctor", ["--verbose"])
        assert len(hooks.history) == 1
        assert hooks.history[0].event_type == "cli.command.start"
        assert hooks.history[0].command == "doctor"

    def test_post_command_records_success(self):
        hooks = CLIHooksManager()
        result = CommandResult.ok("done")
        hooks.post_command("doctor", [], result, 42.5)
        assert len(hooks.history) == 1
        assert hooks.history[0].event_type == "cli.command.end"
        assert hooks.history[0].latency_ms == 42.5

    def test_post_command_records_error(self):
        hooks = CLIHooksManager()
        result = CommandResult.error("fail")
        hooks.post_command("doctor", [], result, 100.0)
        assert hooks.history[0].event_type == "cli.command.error"
        assert hooks.history[0].error == "fail"

    def test_total_commands(self):
        hooks = CLIHooksManager()
        hooks.pre_command("a", [])
        hooks.pre_command("b", [])
        assert hooks.total_commands == 2

    def test_total_errors(self):
        hooks = CLIHooksManager()
        hooks.post_command("a", [], CommandResult.error("e1"), 1.0)
        hooks.post_command("b", [], CommandResult.ok(), 1.0)
        assert hooks.total_errors == 1

    def test_history_ring_buffer(self):
        hooks = CLIHooksManager()
        hooks._max_history = 5
        for i in range(10):
            hooks.pre_command(f"cmd{i}", [])
        assert len(hooks.history) == 5

    def test_eventbus_publish(self):
        bus = MagicMock()
        hooks = CLIHooksManager(event_bus=bus)
        hooks.pre_command("doctor", [])
        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "cli.event"

    def test_eventbus_emit_fallback(self):
        bus = MagicMock(spec=["emit"])
        hooks = CLIHooksManager(event_bus=bus)
        hooks.pre_command("doctor", [])
        bus.emit.assert_called_once()

    def test_eventbus_failure_nonfatal(self):
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus down")
        hooks = CLIHooksManager(event_bus=bus)
        # Should not raise
        hooks.pre_command("doctor", [])
        assert hooks.total_commands == 1


class TestHooksIntegration:
    """Verify hooks wire into registry correctly."""

    def test_pre_post_hooks_fire(
        self, registry: CommandRegistry, fake_cmd: FakeCommand
    ):
        hooks = CLIHooksManager()
        registry.register(fake_cmd)
        registry.add_pre_hook(hooks.pre_command)
        registry.add_post_hook(hooks.post_command)

        registry.dispatch(["fake"])

        assert hooks.total_commands == 1
        assert len(hooks.history) == 2  # pre + post

    def test_hooks_survive_command_error(self, registry: CommandRegistry):
        hooks = CLIHooksManager()
        cmd = FakeCommand(side_effect=ValueError("boom"))
        registry.register(cmd)
        registry.add_pre_hook(hooks.pre_command)
        registry.add_post_hook(hooks.post_command)

        result = registry.dispatch(["fake"])
        assert not result.success
        # Both pre and post hooks should have fired
        assert len(hooks.history) == 2


# ═══════════════════════════════════════════════════════════════════
# § Individual Command Modules
# ═══════════════════════════════════════════════════════════════════


class TestCommandModules:
    """Verify each command module satisfies the BaseCommand protocol."""

    def test_all_commands_have_required_attrs(self):
        from core.cli.commands import ALL_COMMANDS

        for cmd_class in ALL_COMMANDS:
            cmd = cmd_class()
            assert hasattr(cmd, "name"), f"{cmd_class} missing name"
            assert hasattr(cmd, "aliases"), f"{cmd_class} missing aliases"
            assert hasattr(cmd, "description"), f"{cmd_class} missing description"
            assert hasattr(cmd, "category"), f"{cmd_class} missing category"
            assert hasattr(cmd, "execute"), f"{cmd_class} missing execute"

    def test_all_commands_register(self):
        from core.cli.commands import ALL_COMMANDS

        r = CommandRegistry()
        for cmd_class in ALL_COMMANDS:
            r.register(cmd_class())
        assert len(r.command_names) == len(ALL_COMMANDS)

    def test_no_alias_collisions(self):
        from core.cli.commands import ALL_COMMANDS

        seen_aliases: Dict[str, str] = {}
        for cmd_class in ALL_COMMANDS:
            cmd = cmd_class()
            for alias in cmd.aliases:
                assert (
                    alias not in seen_aliases
                ), f"Alias '{alias}' used by both {seen_aliases[alias]} and {cmd.name}"
                seen_aliases[alias] = cmd.name

    def test_version_command(self):
        from core.cli.commands.version import VersionCommand

        result = VersionCommand().execute([])
        assert result.success
        assert result.data["version"] == VERSION

    def test_doctor_command_runs(self, capsys):
        from core.cli.commands.doctor import DoctorCommand

        result = DoctorCommand().execute([])
        assert result.success
        assert "issues" in result.data
        captured = capsys.readouterr()
        assert "BIZRA Doctor" in captured.out

    def test_dema_command_status_json(self, capsys):
        from core.cli.commands.dema import DemaCommand

        fake_report = {
            "kind": "node0_dema_status",
            "ready": True,
            "findings": [],
            "dema_service": {"status": "stopped"},
            "lm_studio": {"connected": True},
        }
        with patch(
            "core.cli.commands.dema.read_node0_dema_status",
            return_value=fake_report,
        ):
            result = DemaCommand().execute(["status", "--json", "--root", "/tmp/dema"])

        assert result.success
        assert result.data == fake_report
        captured = capsys.readouterr()
        assert '"kind": "node0_dema_status"' in captured.out

    def test_dema_command_unknown_subcommand(self):
        from core.cli.commands.dema import DemaCommand

        result = DemaCommand().execute(["start"])
        assert not result.success
        assert "Unknown dema command" in result.message

    def test_dema_command_status_read_failure(self):
        from core.cli.commands.dema import DemaCommand

        with patch(
            "core.cli.commands.dema.read_node0_dema_status",
            side_effect=ValueError("bad local state"),
        ):
            result = DemaCommand().execute(["status"])

        assert not result.success
        assert "Failed to read DEMA status" in result.message

    def test_status_command_offline(self, capsys):
        from core.cli.commands.status import StatusCommand

        with patch("core.cli.commands.status.api_health", return_value=None):
            result = StatusCommand().execute([])
        assert result.success
        assert result.data["online"] is False

    def test_identity_command_no_file(self, capsys):
        from core.cli.commands.identity import IdentityCommand

        with patch("core.cli.commands.identity.BIZRA_IDENTITY") as mock_id:
            mock_id.exists.return_value = False
            result = IdentityCommand().execute([])
        assert result.success
        assert result.data["exists"] is False

    def test_mission_command_no_args(self, capsys):
        from core.cli.commands.mission import MissionCommand

        result = MissionCommand().execute([])
        assert not result.success

    def test_mission_command_offline(self, capsys):
        from core.cli.commands.mission import MissionCommand

        with patch("core.cli.commands.mission.api_health", return_value=None):
            result = MissionCommand().execute(["test mission"])
        assert not result.success

    def test_wallet_command_offline(self, capsys):
        from core.cli.commands.wallet import WalletCommand

        with patch("core.cli.commands.wallet.api_health", return_value=None):
            result = WalletCommand().execute([])
        assert not result.success

    def test_briefing_command_offline(self, capsys):
        from core.cli.commands.wallet import BriefingCommand

        with patch("core.cli.commands.wallet.api_health", return_value=None):
            result = BriefingCommand().execute([])
        assert not result.success


# ═══════════════════════════════════════════════════════════════════
# § Full Registry Integration
# ═══════════════════════════════════════════════════════════════════


class TestFullRegistryIntegration:
    """End-to-end: build the production registry and dispatch commands."""

    @pytest.fixture
    def full_registry(self) -> CommandRegistry:
        from core.cli.commands import ALL_COMMANDS

        r = CommandRegistry()
        hooks = CLIHooksManager()
        r.add_pre_hook(hooks.pre_command)
        r.add_post_hook(hooks.post_command)
        for cmd_class in ALL_COMMANDS:
            r.register(cmd_class())
        return r

    def test_dispatch_version(self, full_registry: CommandRegistry):
        result = full_registry.dispatch(["version"])
        assert result.success

    def test_dispatch_alias_v(self, full_registry: CommandRegistry):
        result = full_registry.dispatch(["v"])
        assert result.success

    def test_dispatch_doctor(self, full_registry: CommandRegistry):
        result = full_registry.dispatch(["doctor"])
        assert result.success

    def test_dispatch_doc_alias(self, full_registry: CommandRegistry):
        result = full_registry.dispatch(["doc"])
        assert result.success

    def test_dispatch_unknown_suggests(self, full_registry: CommandRegistry):
        result = full_registry.dispatch(["statsu"])
        assert not result.success
        assert "status" in result.message

    def test_metrics_after_dispatches(self, full_registry: CommandRegistry):
        full_registry.dispatch(["version"])
        full_registry.dispatch(["version"])
        metrics = full_registry.get_metrics()
        assert metrics["version"]["calls"] == 2
        assert metrics["version"]["avg_ms"] > 0

    def test_performance_under_200ms(self, full_registry: CommandRegistry):
        """Target: <200ms average command execution time."""
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            full_registry.dispatch(["version"])
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        assert avg < 200, f"Average command time {avg:.1f}ms exceeds 200ms target"
