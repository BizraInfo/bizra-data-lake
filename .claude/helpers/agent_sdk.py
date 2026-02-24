#!/usr/bin/env python3
"""
BIZRA Agent SDK — Programmatic Claude Code Interface
=====================================================
Wraps the Claude Code CLI for programmatic agent orchestration.

Features:
- Streaming and non-streaming modes
- Tool allowlisting
- Session management
- JSON output parsing

Usage:
    from agent_sdk import ClaudeAgent

    agent = ClaudeAgent(allowed_tools=["Read", "Grep", "Glob"])
    result = agent.run("Find all Python files with FATE in them")
    print(result.output)
"""

import subprocess
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile


@dataclass
class AgentResult:
    """Result from a Claude agent invocation."""
    output: str
    cost_usd: float
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    session_id: str
    is_error: bool = False
    error_message: str = ""

    @classmethod
    def from_json(cls, data: dict) -> "AgentResult":
        """Parse from JSON output."""
        return cls(
            output=data.get("result", ""),
            cost_usd=data.get("cost_usd", 0.0),
            duration_ms=data.get("duration_ms", 0),
            duration_api_ms=data.get("duration_api_ms", 0),
            num_turns=data.get("num_turns", 0),
            session_id=data.get("session_id", ""),
            is_error=data.get("is_error", False),
            error_message=data.get("error_message", ""),
        )


class ClaudeAgent:
    """
    Programmatic interface to Claude Code CLI.

    Examples:
        # Basic usage
        agent = ClaudeAgent()
        result = agent.run("What files are in the core/ directory?")

        # Restricted tools
        agent = ClaudeAgent(allowed_tools=["Read", "Glob"])
        result = agent.run("Read the README.md file")

        # With system prompt
        agent = ClaudeAgent(
            system_prompt="You are a code reviewer. Focus on security issues.",
            allowed_tools=["Read", "Grep"]
        )
        result = agent.run("Review core/pci/envelope.py for security issues")
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        model: str = "sonnet",  # claude-sonnet-4-20250514
    ):
        self.working_dir = working_dir or os.getcwd()
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.model = model

    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        continue_session: bool = False,
    ) -> AgentResult:
        """
        Run a prompt through Claude Code.

        Args:
            prompt: The task to execute
            session_id: Resume a specific session
            continue_session: Continue the most recent session

        Returns:
            AgentResult with output and metadata
        """
        cmd = ["claude"]

        # Add prompt
        cmd.extend(["-p", prompt])

        # Output format
        cmd.extend(["--output-format", "json"])

        # Max turns
        cmd.extend(["--max-turns", str(self.max_turns)])

        # Tool restrictions
        if self.allowed_tools:
            for tool in self.allowed_tools:
                cmd.extend(["--allowedTools", tool])

        if self.disallowed_tools:
            for tool in self.disallowed_tools:
                cmd.extend(["--disallowedTools", tool])

        # System prompt
        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])

        # Session management
        if session_id:
            cmd.extend(["--session-id", session_id])
        elif continue_session:
            cmd.append("--continue")

        # Model selection
        if self.model:
            cmd.extend(["--model", self.model])

        # Run the command
        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                return AgentResult(
                    output="",
                    cost_usd=0.0,
                    duration_ms=0,
                    duration_api_ms=0,
                    num_turns=0,
                    session_id="",
                    is_error=True,
                    error_message=result.stderr,
                )

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                return AgentResult.from_json(data)
            except json.JSONDecodeError:
                return AgentResult(
                    output=result.stdout,
                    cost_usd=0.0,
                    duration_ms=0,
                    duration_api_ms=0,
                    num_turns=0,
                    session_id="",
                    is_error=False,
                )

        except subprocess.TimeoutExpired:
            return AgentResult(
                output="",
                cost_usd=0.0,
                duration_ms=300000,
                duration_api_ms=0,
                num_turns=0,
                session_id="",
                is_error=True,
                error_message="Agent timed out after 5 minutes",
            )
        except Exception as e:
            return AgentResult(
                output="",
                cost_usd=0.0,
                duration_ms=0,
                duration_api_ms=0,
                num_turns=0,
                session_id="",
                is_error=True,
                error_message=str(e),
            )

    def run_streaming(
        self,
        prompt: str,
        callback: callable = None,
    ):
        """
        Run a prompt with streaming output.

        Args:
            prompt: The task to execute
            callback: Function called with each output chunk
        """
        cmd = ["claude", "-p", prompt, "--output-format", "stream-json"]

        if self.allowed_tools:
            for tool in self.allowed_tools:
                cmd.extend(["--allowedTools", tool])

        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])

        process = subprocess.Popen(
            cmd,
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if callback:
                    callback(data)
                else:
                    # Default: print assistant messages
                    if data.get("type") == "assistant":
                        print(data.get("message", ""), end="", flush=True)
            except json.JSONDecodeError:
                pass

        process.wait()


# Specialized agent presets for BIZRA

class ResearchAgent(ClaudeAgent):
    """Agent specialized for research and exploration."""

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__(
            working_dir=working_dir,
            allowed_tools=["Read", "Glob", "Grep", "WebFetch", "WebSearch"],
            system_prompt="You are a research specialist. Focus on finding accurate information and providing citations.",
            max_turns=20,
        )


class CoderAgent(ClaudeAgent):
    """Agent specialized for code generation and modification."""

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__(
            working_dir=working_dir,
            allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            system_prompt="You are a code implementation specialist. Write clean, tested, documented code.",
            max_turns=15,
        )


class ReviewerAgent(ClaudeAgent):
    """Agent specialized for code review."""

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__(
            working_dir=working_dir,
            allowed_tools=["Read", "Glob", "Grep"],
            system_prompt="You are a code reviewer. Focus on security, performance, and maintainability issues.",
            max_turns=10,
        )


class SecurityAgent(ClaudeAgent):
    """Agent specialized for security analysis."""

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__(
            working_dir=working_dir,
            allowed_tools=["Read", "Glob", "Grep"],
            system_prompt="You are a security analyst. Identify vulnerabilities, secrets exposure, and injection risks.",
            max_turns=10,
        )


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Agent SDK CLI")
    parser.add_argument("prompt", help="The prompt to run")
    parser.add_argument("--agent", choices=["research", "coder", "reviewer", "security"], default=None)
    parser.add_argument("--tools", nargs="+", help="Allowed tools")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")

    args = parser.parse_args()

    # Select agent type
    if args.agent == "research":
        agent = ResearchAgent()
    elif args.agent == "coder":
        agent = CoderAgent()
    elif args.agent == "reviewer":
        agent = ReviewerAgent()
    elif args.agent == "security":
        agent = SecurityAgent()
    else:
        agent = ClaudeAgent(allowed_tools=args.tools)

    # Run
    if args.stream:
        agent.run_streaming(args.prompt)
    else:
        result = agent.run(args.prompt)
        if result.is_error:
            print(f"Error: {result.error_message}", file=sys.stderr)
            sys.exit(1)
        print(result.output)
        print(f"\n---\nCost: ${result.cost_usd:.4f} | Turns: {result.num_turns} | Time: {result.duration_ms}ms")
