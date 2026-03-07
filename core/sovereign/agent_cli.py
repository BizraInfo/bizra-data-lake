"""Minimal terminal agent kernel for `bizra agent`."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from .mcp_disclosure import MCPProgressiveDisclosure, SkillContext, SkillIndex
from .mcp_disclosure import create_mcp_disclosure
from .runtime_engines.giants_registry import get_giants_registry

_GIANTS_REGISTERED = False


@dataclass(frozen=True)
class HookSpec:
    hook_id: str
    stage: str
    description: str
    enabled_by_default: bool = True


@dataclass(frozen=True)
class PluginSpec:
    plugin_id: str
    category: str
    description: str
    enabled_by_default: bool = True


@dataclass(frozen=True)
class SubagentSpec:
    agent_id: str
    role: str
    description: str
    default_enabled: bool = True


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    category: str
    description: str
    requires_approval: bool = False


@dataclass(frozen=True)
class MCPServerSpec:
    server_id: str
    transport: str
    description: str
    tools: tuple[str, ...]


@dataclass(frozen=True)
class MemoryLayerSpec:
    layer_id: str
    description: str
    default_enabled: bool = True


@dataclass
class AgentTaskPlan:
    task: str
    recommended_skills: list[str]
    recommended_subagents: list[str]
    recommended_tools: list[str]
    active_hooks: list[str]
    active_plugins: list[str]
    active_mcp_servers: list[str]
    rationale: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentKernel:
    """Bootstraps the default built-in component surface for `bizra agent`."""

    def __init__(self) -> None:
        self._skills: MCPProgressiveDisclosure = create_mcp_disclosure()
        self._skill_index: dict[str, SkillIndex] = {}
        self._skill_context: dict[str, SkillContext] = {}
        self._hooks = self._build_hooks()
        self._plugins = self._build_plugins()
        self._subagents = self._build_subagents()
        self._tools = self._build_tools()
        self._mcp_servers = self._build_mcp_servers()
        self._memory_layers = self._build_memory_layers()
        self._register_default_skills()
        self._register_giants()

    def _build_hooks(self) -> list[HookSpec]:
        return [
            HookSpec(
                hook_id="pre_session_brief",
                stage="session",
                description="Hydrate the session with memory-backed briefing before task intake.",
            ),
            HookSpec(
                hook_id="pre_task_intake",
                stage="task",
                description="Normalize user intent before any planning or tool use.",
            ),
            HookSpec(
                hook_id="post_plan_guard",
                stage="plan",
                description="Check planned work against approvals and policy gates.",
            ),
            HookSpec(
                hook_id="pre_tool_call",
                stage="tool",
                description="Validate tool inputs and attach execution receipts.",
            ),
            HookSpec(
                hook_id="post_tool_call",
                stage="tool",
                description="Capture output summaries for session continuity.",
            ),
            HookSpec(
                hook_id="pre_subagent_dispatch",
                stage="subagent",
                description="Constrain delegated scope before a worker is launched.",
            ),
            HookSpec(
                hook_id="post_subagent_dispatch",
                stage="subagent",
                description="Collect delegated results and reintegrate them into the parent task.",
            ),
        ]

    def _build_plugins(self) -> list[PluginSpec]:
        return [
            PluginSpec(
                plugin_id="workspace_adapter",
                category="runtime",
                description="Provides filesystem and workspace context adapters.",
            ),
            PluginSpec(
                plugin_id="approval_guard",
                category="safety",
                description="Applies approval policy before risky commands and edits.",
            ),
            PluginSpec(
                plugin_id="session_memory",
                category="state",
                description="Carries forward local session state and execution receipts.",
            ),
            PluginSpec(
                plugin_id="living_memory_core",
                category="memory",
                description="Mounts semantic, episodic, and procedural memory as default context.",
            ),
            PluginSpec(
                plugin_id="mcp_gateway",
                category="integration",
                description="Exposes built-in and external MCP tool servers through one router.",
            ),
            PluginSpec(
                plugin_id="skill_disclosure",
                category="reasoning",
                description="Uses progressive skill disclosure to keep the agent lightweight by default.",
            ),
        ]

    def _build_subagents(self) -> list[SubagentSpec]:
        return [
            SubagentSpec(
                agent_id="nexus",
                role="integration",
                description="Routes work, aggregates context, and coordinates specialist handoffs.",
            ),
            SubagentSpec(
                agent_id="atlas",
                role="planning",
                description="Builds task structure, sequencing, and dependency-aware execution plans.",
            ),
            SubagentSpec(
                agent_id="oracle",
                role="research",
                description="Finds evidence, source material, and high-signal retrieval paths.",
            ),
            SubagentSpec(
                agent_id="forge",
                role="implementation",
                description="Owns code changes, commands, and targeted verification.",
            ),
            SubagentSpec(
                agent_id="judge",
                role="evaluation",
                description="Reviews changes for correctness, regression risk, and quality gaps.",
            ),
            SubagentSpec(
                agent_id="crown",
                role="governance",
                description="Applies constitutional, policy, and ethics checks to planned actions.",
            ),
            SubagentSpec(
                agent_id="herald",
                role="delivery",
                description="Formats outcomes, summaries, and release-facing communication.",
            ),
        ]

    def _build_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                tool_id="workspace.read",
                category="filesystem",
                description="Read files and inspect local workspace state.",
            ),
            ToolSpec(
                tool_id="workspace.write",
                category="filesystem",
                description="Write or update files inside the working tree.",
                requires_approval=True,
            ),
            ToolSpec(
                tool_id="workspace.apply_patch",
                category="filesystem",
                description="Apply structured patches for precise code edits.",
                requires_approval=True,
            ),
            ToolSpec(
                tool_id="search.ripgrep",
                category="search",
                description="Search code and symbols using ripgrep.",
            ),
            ToolSpec(
                tool_id="git.status",
                category="git",
                description="Inspect current worktree status before making changes.",
            ),
            ToolSpec(
                tool_id="git.diff",
                category="git",
                description="Review staged and unstaged changes with diff-aware context.",
            ),
            ToolSpec(
                tool_id="shell.exec",
                category="execution",
                description="Run terminal commands for builds, checks, and diagnostics.",
                requires_approval=True,
            ),
            ToolSpec(
                tool_id="test.pytest",
                category="verification",
                description="Run targeted test commands and report failures clearly.",
                requires_approval=True,
            ),
            ToolSpec(
                tool_id="mcp.attach_server",
                category="mcp",
                description="Attach an MCP server over stdio or websocket transports.",
            ),
            ToolSpec(
                tool_id="mcp.call_tool",
                category="mcp",
                description="List and invoke tools exposed by registered MCP servers.",
            ),
        ]

    def _build_mcp_servers(self) -> list[MCPServerSpec]:
        return [
            MCPServerSpec(
                server_id="workspace-core",
                transport="inproc",
                description="Default local workspace tool server.",
                tools=(
                    "workspace.read",
                    "workspace.write",
                    "workspace.apply_patch",
                    "search.ripgrep",
                ),
            ),
            MCPServerSpec(
                server_id="exec-core",
                transport="inproc",
                description="Default command, git, and test server.",
                tools=("git.status", "git.diff", "shell.exec", "test.pytest"),
            ),
            MCPServerSpec(
                server_id="mcp-gateway",
                transport="stdio|websocket",
                description="Bridge for external MCP servers and plugins.",
                tools=("mcp.attach_server", "mcp.call_tool"),
            ),
        ]

    def _build_memory_layers(self) -> list[MemoryLayerSpec]:
        return [
            MemoryLayerSpec(
                layer_id="semantic",
                description="Stable user and project knowledge that persists across sessions.",
            ),
            MemoryLayerSpec(
                layer_id="episodic",
                description="Recent events, diffs, and task outcomes for session continuity.",
            ),
            MemoryLayerSpec(
                layer_id="procedural",
                description="Compiled task patterns, hooks, and reflex-like execution routines.",
            ),
        ]

    def _register_default_skills(self) -> None:
        skills = [
            (
                SkillIndex(
                    skill_id="repo_search",
                    name="Repo Search",
                    category="retrieval",
                    relevance_keywords=[
                        "search",
                        "find",
                        "grep",
                        "where",
                        "codebase",
                    ],
                ),
                SkillContext(
                    skill_id="repo_search",
                    description="Locate symbols, files, and high-impact call paths before editing.",
                    parameters={"module_path": "core.sovereign.agent_cli"},
                    examples=["find auth middleware", "search where rate limiter is used"],
                    dependencies=["search.ripgrep", "workspace.read"],
                ),
            ),
            (
                SkillIndex(
                    skill_id="code_review",
                    name="Code Review",
                    category="reasoning",
                    relevance_keywords=[
                        "review",
                        "bug",
                        "security",
                        "regression",
                        "diff",
                    ],
                ),
                SkillContext(
                    skill_id="code_review",
                    description="Prioritize concrete bugs, security issues, and missing tests.",
                    parameters={"module_path": "core.sovereign.agent_cli"},
                    examples=["review unstaged changes", "find security regressions"],
                    dependencies=["git.diff", "workspace.read"],
                ),
            ),
            (
                SkillIndex(
                    skill_id="change_planning",
                    name="Change Planning",
                    category="reasoning",
                    relevance_keywords=[
                        "implement",
                        "refactor",
                        "plan",
                        "architecture",
                        "design",
                    ],
                ),
                SkillContext(
                    skill_id="change_planning",
                    description="Translate vague goals into bounded implementation steps and checks.",
                    parameters={"module_path": "core.sovereign.agent_cli"},
                    examples=["plan a terminal agent", "design route policy enforcement"],
                    dependencies=["workspace.read", "git.status"],
                ),
            ),
            (
                SkillIndex(
                    skill_id="test_selection",
                    name="Test Selection",
                    category="verification",
                    relevance_keywords=[
                        "test",
                        "pytest",
                        "verify",
                        "regression",
                        "coverage",
                    ],
                ),
                SkillContext(
                    skill_id="test_selection",
                    description="Choose the smallest high-signal test set that proves the change.",
                    parameters={"module_path": "core.sovereign.agent_cli"},
                    examples=["run auth tests", "verify the API route policy gate"],
                    dependencies=["test.pytest", "git.diff"],
                ),
            ),
            (
                SkillIndex(
                    skill_id="mcp_orchestration",
                    name="MCP Orchestration",
                    category="integration",
                    relevance_keywords=[
                        "mcp",
                        "plugin",
                        "tool",
                        "server",
                        "connector",
                    ],
                ),
                SkillContext(
                    skill_id="mcp_orchestration",
                    description="Attach external tools through a controlled MCP router.",
                    parameters={"module_path": "core.sovereign.agent_cli"},
                    examples=["attach github MCP", "configure plugin tool routing"],
                    dependencies=["mcp.attach_server", "mcp.call_tool"],
                ),
            ),
        ]

        for index, context in skills:
            self._skills.register_skill(index, context)
            self._skill_index[index.skill_id] = index
            self._skill_context[index.skill_id] = context

    def _register_giants(self) -> None:
        global _GIANTS_REGISTERED
        if _GIANTS_REGISTERED:
            return

        registry = get_giants_registry()
        registry.record_application(
            module="core.sovereign.agent_cli",
            method="AgentKernel.plan",
            giant_names=["Claude Shannon", "Leslie Lamport"],
            explanation="Task planning ranks high-signal components and delegates bounded work across subagents.",
            performance_impact="Keeps terminal agent decisions compact and auditable.",
        )
        registry.record_application(
            module="core.sovereign.agent_cli",
            method="AgentKernel.component_manifest",
            giant_names=["Anthropic"],
            explanation="Progressive skill disclosure keeps the default agent footprint small until deeper context is needed.",
            performance_impact="Reduces default memory pressure in interactive sessions.",
        )
        _GIANTS_REGISTERED = True

    def component_manifest(self) -> dict[str, Any]:
        return {
            "hooks": [asdict(spec) for spec in self._hooks],
            "plugins": [asdict(spec) for spec in self._plugins],
            "subagents": [asdict(spec) for spec in self._subagents],
            "memory_layers": [asdict(spec) for spec in self._memory_layers],
            "skills": [
                {
                    "skill_id": index.skill_id,
                    "name": index.name,
                    "category": index.category,
                    "description": self._skill_context[index.skill_id].description,
                    "relevance_keywords": list(index.relevance_keywords),
                    "dependencies": list(
                        self._skill_context[index.skill_id].dependencies
                    ),
                }
                for index in self._skill_index.values()
            ],
            "mcp_servers": [asdict(spec) for spec in self._mcp_servers],
            "tools": [asdict(spec) for spec in self._tools],
            "component_counts": {
                "hooks": len(self._hooks),
                "plugins": len(self._plugins),
                "subagents": len(self._subagents),
                "memory_layers": len(self._memory_layers),
                "skills": len(self._skill_index),
                "mcp_servers": len(self._mcp_servers),
                "tools": len(self._tools),
            },
            "attribution": [
                "Claude Shannon: signal-first task triage",
                "Leslie Lamport: bounded coordination across subagents",
                "Anthropic: progressive disclosure for skills and tools",
            ],
        }

    def plan(self, task: str) -> AgentTaskPlan:
        normalized = task.lower()
        skill_matches = [
            skill.skill_id for skill in self._skills.discover_skills(task, max_results=4)
        ]

        if not skill_matches:
            skill_matches = ["change_planning", "repo_search"]

        subagents = self._select_subagents(normalized)
        tools = self._select_tools(normalized)
        mcp_servers = self._select_mcp_servers(tools)

        rationale = [
            "Default hooks stay active for every task so tool calls and delegation remain auditable.",
            "Skills are discovered by relevance rather than fully loaded up front.",
            "Semantic, episodic, and procedural memory remain mounted as default context.",
        ]
        if "mcp_orchestration" in skill_matches:
            rationale.append(
                "The MCP gateway is included because the task explicitly mentions plugins, tools, or servers."
            )
        if "forge" in subagents:
            rationale.append(
                "FORGE is selected because the task implies implementation, patching, or command execution."
            )
        if "judge" in subagents:
            rationale.append(
                "JUDGE is selected because the task reads like review, audit, or verification work."
            )

        return AgentTaskPlan(
            task=task,
            recommended_skills=skill_matches,
            recommended_subagents=subagents,
            recommended_tools=tools,
            active_hooks=[spec.hook_id for spec in self._hooks if spec.enabled_by_default],
            active_plugins=[
                spec.plugin_id for spec in self._plugins if spec.enabled_by_default
            ],
            active_mcp_servers=mcp_servers,
            rationale=rationale,
        )

    def _select_subagents(self, task: str) -> list[str]:
        selected = ["nexus"]
        if any(word in task for word in ("plan", "roadmap", "design", "architecture")):
            selected.append("atlas")
        if any(word in task for word in ("research", "search", "find", "source")):
            selected.append("oracle")
        if any(
            word in task
            for word in ("implement", "build", "fix", "edit", "patch", "refactor")
        ):
            selected.append("forge")
        if any(
            word in task for word in ("review", "audit", "security", "verify", "test")
        ):
            selected.append("judge")
        if any(
            word in task
            for word in ("ethic", "policy", "compliance", "constitution", "governance")
        ):
            selected.append("crown")
        if any(
            word in task
            for word in ("publish", "write", "report", "document", "deliver")
        ):
            selected.append("herald")
        return selected

    def _select_tools(self, task: str) -> list[str]:
        selected = ["workspace.read", "search.ripgrep", "git.status"]
        if any(word in task for word in ("review", "diff", "regression", "audit")):
            selected.append("git.diff")
        if any(
            word in task
            for word in ("implement", "fix", "edit", "patch", "write", "refactor")
        ):
            selected.extend(["workspace.write", "workspace.apply_patch", "shell.exec"])
        if any(word in task for word in ("test", "verify", "pytest", "regression")):
            selected.append("test.pytest")
        if any(word in task for word in ("mcp", "plugin", "server", "tool")):
            selected.extend(["mcp.attach_server", "mcp.call_tool"])

        return list(dict.fromkeys(selected))

    def _select_mcp_servers(self, tools: list[str]) -> list[str]:
        selected: list[str] = []
        for spec in self._mcp_servers:
            if any(tool in spec.tools for tool in tools):
                selected.append(spec.server_id)
        return selected


HELP_TEXT = """Commands:
  /help         Show agent shell help
  /components   Show the default built-in component manifest
  /plan TASK    Plan a task against the built-in agent kernel
  /exit         Exit the shell
"""


def build_agent_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the `bizra agent` parser to the main CLI."""
    agent_parser = subparsers.add_parser(
        "agent",
        help="Terminal agent shell with built-in hooks, skills, subagents, plugins, and MCP tools",
    )
    agent_sub = agent_parser.add_subparsers(
        dest="agent_command",
        help="Agent action",
    )
    agent_sub.required = False
    agent_parser.set_defaults(agent_command="chat")

    chat_parser = agent_sub.add_parser(
        "chat",
        help="Start the interactive agent shell",
    )
    chat_parser.add_argument(
        "--task",
        help="Optional task to plan immediately before entering the shell",
    )

    components_parser = agent_sub.add_parser(
        "components",
        help="Show the built-in component manifest",
    )
    components_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text",
    )

    plan_parser = agent_sub.add_parser(
        "plan",
        aliases=["run"],
        help="Plan a task against the built-in component graph",
    )
    plan_parser.add_argument("task", nargs="+", help="Task text")
    plan_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    return agent_parser


def dispatch_agent_command(args: argparse.Namespace) -> int:
    """Route parsed `bizra agent` arguments."""
    kernel = AgentKernel()
    if args.agent_command == "components":
        _print_payload(kernel.component_manifest(), getattr(args, "json", False))
        return 0

    if args.agent_command in {"plan", "run"}:
        plan = kernel.plan(" ".join(args.task))
        _print_payload(plan.to_dict(), getattr(args, "json", False))
        return 0

    return run_agent_shell(kernel, initial_task=getattr(args, "task", None))


def run_agent_shell(kernel: AgentKernel, initial_task: str | None = None) -> int:
    """Run a small interactive planning shell for the built-in agent surface."""
    print("BIZRA Agent Shell")
    print("Default components: hooks, skills, subagents, plugins, MCP tools")
    print("Type /help for commands.")

    if initial_task:
        print()
        print(_render_text_plan(kernel.plan(initial_task)))

    while True:
        try:
            raw = input("bizra-agent> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 130

        if not raw:
            continue
        if raw in {"/exit", "exit", "quit"}:
            return 0
        if raw == "/help":
            print(HELP_TEXT.rstrip())
            continue
        if raw == "/components":
            print(_render_text_manifest(kernel.component_manifest()))
            continue
        if raw.startswith("/plan "):
            print(_render_text_plan(kernel.plan(raw[6:].strip())))
            continue

        print(_render_text_plan(kernel.plan(raw)))


def _print_payload(payload: dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if "component_counts" in payload:
        print(_render_text_manifest(payload))
        return

    print(_render_text_plan(AgentTaskPlan(**payload)))


def _render_text_manifest(payload: dict[str, Any]) -> str:
    lines = ["BIZRA Agent Default Components"]
    counts = payload["component_counts"]
    lines.append(
        "Counts: "
        + ", ".join(f"{name}={value}" for name, value in counts.items())
    )
    lines.append("Hooks: " + ", ".join(item["hook_id"] for item in payload["hooks"]))
    lines.append(
        "Skills: " + ", ".join(item["skill_id"] for item in payload["skills"])
    )
    lines.append(
        "Subagents: " + ", ".join(item["agent_id"] for item in payload["subagents"])
    )
    lines.append(
        "Plugins: " + ", ".join(item["plugin_id"] for item in payload["plugins"])
    )
    lines.append(
        "MCP servers: " + ", ".join(item["server_id"] for item in payload["mcp_servers"])
    )
    lines.append(
        "Memory: " + ", ".join(item["layer_id"] for item in payload["memory_layers"])
    )
    return "\n".join(lines)


def _render_text_plan(plan: AgentTaskPlan) -> str:
    return "\n".join(
        [
            f"Task: {plan.task}",
            "Skills: " + ", ".join(plan.recommended_skills),
            "Subagents: " + ", ".join(plan.recommended_subagents),
            "Tools: " + ", ".join(plan.recommended_tools),
            "Hooks: " + ", ".join(plan.active_hooks),
            "Plugins: " + ", ".join(plan.active_plugins),
            "MCP servers: " + ", ".join(plan.active_mcp_servers),
            "Rationale:",
            *[f"- {item}" for item in plan.rationale],
        ]
    )
