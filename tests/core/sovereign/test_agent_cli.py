"""Tests for the terminal agent CLI scaffold."""

from __future__ import annotations

import argparse
import json

from core.sovereign.agent_cli import AgentKernel, build_agent_parser
from core.sovereign.agent_cli import dispatch_agent_command


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_agent_parser(subparsers)
    return parser.parse_args(argv)


def test_agent_manifest_includes_default_component_families() -> None:
    manifest = AgentKernel().component_manifest()

    assert manifest["component_counts"] == {
        "hooks": 7,
        "plugins": 6,
        "subagents": 7,
        "memory_layers": 3,
        "skills": 5,
        "mcp_servers": 3,
        "tools": 10,
    }
    assert {item["hook_id"] for item in manifest["hooks"]} >= {
        "pre_session_brief",
        "pre_tool_call",
        "post_subagent_dispatch",
    }
    assert {item["plugin_id"] for item in manifest["plugins"]} >= {
        "living_memory_core",
        "mcp_gateway",
        "approval_guard",
    }
    assert {item["agent_id"] for item in manifest["subagents"]} >= {
        "nexus",
        "atlas",
        "oracle",
        "forge",
        "judge",
        "crown",
        "herald",
    }
    assert {item["layer_id"] for item in manifest["memory_layers"]} == {
        "semantic",
        "episodic",
        "procedural",
    }


def test_agent_plan_routes_to_expected_subagents_and_tools() -> None:
    plan = AgentKernel().plan(
        "review the auth diff, fix the regression, run tests, and attach an MCP tool server"
    )

    assert set(plan.recommended_skills) >= {
        "code_review",
        "test_selection",
        "mcp_orchestration",
    }
    assert set(plan.recommended_subagents) >= {"nexus", "forge", "judge"}
    assert set(plan.recommended_tools) >= {
        "git.diff",
        "workspace.apply_patch",
        "shell.exec",
        "test.pytest",
        "mcp.attach_server",
        "mcp.call_tool",
    }
    assert "Semantic, episodic, and procedural memory remain mounted as default context." in plan.rationale


def test_agent_parser_defaults_to_chat_mode() -> None:
    args = _parse(["agent"])

    assert args.command == "agent"
    assert args.agent_command == "chat"


def test_agent_components_command_emits_json(capsys) -> None:
    args = _parse(["agent", "components", "--json"])

    exit_code = dispatch_agent_command(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["component_counts"]["subagents"] == 7
    assert any(item["server_id"] == "mcp-gateway" for item in payload["mcp_servers"])
