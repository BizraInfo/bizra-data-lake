"""
BIZRA CLI Command Registry
===========================

Modular command registration, alias resolution, fuzzy matching,
and middleware execution for all CLI commands.

Standing on Giants:
- Thompson & Ritchie (1973): Unix command dispatch
- Deming (1950): Measure every command, never regress
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence


@dataclass
class CommandResult:
    """Outcome of a CLI command execution."""

    success: bool
    message: str = ""
    exit_code: int = 0
    data: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, message: str = "", data: Optional[Dict[str, Any]] = None) -> "CommandResult":
        return cls(success=True, message=message, exit_code=0, data=data)

    @classmethod
    def error(cls, message: str, exit_code: int = 1) -> "CommandResult":
        return cls(success=False, message=message, exit_code=exit_code)

    @classmethod
    def info(cls, message: str) -> "CommandResult":
        return cls(success=True, message=message, exit_code=0)


class BaseCommand(Protocol):
    """Protocol every command module must satisfy."""

    name: str
    aliases: Sequence[str]
    description: str
    category: str

    def execute(self, args: List[str]) -> CommandResult:
        """Run the command with the given arguments."""
        ...


@dataclass
class CommandEntry:
    """Internal registry entry wrapping a command."""

    command: BaseCommand
    call_count: int = 0
    total_ms: float = 0.0
    error_count: int = 0
    last_execution_ms: float = 0.0
    _latencies: List[float] = field(default_factory=list)

    def record(self, latency_ms: float, success: bool) -> None:
        self.call_count += 1
        self.total_ms += latency_ms
        self.last_execution_ms = latency_ms
        self._latencies.append(latency_ms)
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-500:]
        if not success:
            self.error_count += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count else 0.0

    @property
    def p95_ms(self) -> float:
        if not self._latencies:
            return 0.0
        s = sorted(self._latencies)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 1.0
        return (self.call_count - self.error_count) / self.call_count


class CommandRegistry:
    """
    Central command registry with alias resolution, fuzzy matching,
    performance tracking, and pre/post hooks.

    Usage::

        registry = CommandRegistry()
        registry.register(DoctorCommand())
        result = registry.dispatch(["doctor"])
    """

    def __init__(self) -> None:
        self._commands: Dict[str, CommandEntry] = {}
        self._aliases: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._pre_hooks: List[Callable[[str, List[str]], None]] = []
        self._post_hooks: List[Callable[[str, List[str], CommandResult, float], None]] = []

    def register(self, command: BaseCommand) -> None:
        """Register a command and its aliases."""
        name = command.name
        self._commands[name] = CommandEntry(command=command)

        for alias in command.aliases:
            self._aliases[alias] = name

        cat = getattr(command, "category", "general")
        if cat not in self._categories:
            self._categories[cat] = []
        self._categories[cat].append(name)

    def add_pre_hook(self, hook: Callable[[str, List[str]], None]) -> None:
        """Add a hook that fires before command execution."""
        self._pre_hooks.append(hook)

    def add_post_hook(
        self, hook: Callable[[str, List[str], CommandResult, float], None]
    ) -> None:
        """Add a hook that fires after command execution."""
        self._post_hooks.append(hook)

    def resolve(self, name: str) -> Optional[CommandEntry]:
        """Resolve a command name or alias to its entry."""
        clean = name.lower().strip("-")
        if clean in self._commands:
            return self._commands[clean]
        target = self._aliases.get(clean)
        if target and target in self._commands:
            return self._commands[target]
        return None

    def suggest(self, name: str, n: int = 3) -> List[str]:
        """Fuzzy-match a misspelled command name."""
        all_names = list(self._commands.keys()) + list(self._aliases.keys())
        return get_close_matches(name.lower(), all_names, n=n, cutoff=0.5)

    def dispatch(self, argv: List[str]) -> CommandResult:
        """
        Dispatch a CLI invocation to the matching command.

        Args:
            argv: sys.argv[1:] — the command name plus its arguments.

        Returns:
            CommandResult from the matched command.
        """
        if not argv:
            return CommandResult.error("No command provided. Use 'bizra help' for usage.")

        cmd_name = argv[0]
        cmd_args = argv[1:]
        entry = self.resolve(cmd_name)

        if entry is None:
            suggestions = self.suggest(cmd_name)
            hint = ""
            if suggestions:
                hint = f" Did you mean: {', '.join(suggestions)}?"
            return CommandResult.error(f"Unknown command: {cmd_name}.{hint}")

        # Pre-hooks
        for hook in self._pre_hooks:
            try:
                hook(entry.command.name, cmd_args)
            except Exception:
                pass

        # Execute with timing
        t0 = time.perf_counter()
        try:
            result = entry.command.execute(cmd_args)
        except KeyboardInterrupt:
            result = CommandResult.error("Interrupted.", exit_code=130)
        except Exception as exc:
            result = CommandResult.error(f"Command failed: {exc}", exit_code=1)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        entry.record(elapsed_ms, result.success)

        # Post-hooks
        for hook in self._post_hooks:
            try:
                hook(entry.command.name, cmd_args, result, elapsed_ms)
            except Exception:
                pass

        return result

    def list_commands(self) -> Dict[str, List[str]]:
        """Return commands grouped by category."""
        out: Dict[str, List[str]] = {}
        for cat, names in self._categories.items():
            out[cat] = sorted(names)
        return out

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return performance metrics for all commands."""
        metrics: Dict[str, Dict[str, Any]] = {}
        for name, entry in self._commands.items():
            metrics[name] = {
                "calls": entry.call_count,
                "avg_ms": round(entry.avg_ms, 2),
                "p95_ms": round(entry.p95_ms, 2),
                "success_rate": round(entry.success_rate, 4),
                "last_ms": round(entry.last_execution_ms, 2),
            }
        return metrics

    @property
    def command_names(self) -> List[str]:
        return sorted(self._commands.keys())
