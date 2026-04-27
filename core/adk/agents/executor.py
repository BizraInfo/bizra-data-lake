"""Executor Agent — runs verified commands and captures results with receipts.

Executes safe, pre-approved operations (tests, builds, health checks)
and produces receipted results. Never runs destructive commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import List

from bizra_config import DATA_LAKE_ROOT
from core.adk.agent import Agent, charter
from core.adk.mission import Mission
from core.adk.tools import tool

OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("BIZRA_EXECUTOR_MODEL", "deepseek-r1:7b")

# Whitelist of safe commands the Executor is allowed to run
SAFE_COMMANDS = {
    "test": [".venv/bin/python", "-m", "pytest", "--co", "-q"],
    "test_run": [".venv/bin/python", "-m", "pytest", "-x", "-q", "--tb=short"],
    "cargo_check": ["cargo", "check", "--workspace"],
    "health": ["curl", "-sf", "http://localhost:8000/docs"],
    "ollama_status": ["curl", "-sf", "http://localhost:11434/api/tags"],
    "git_status": ["git", "status", "--short"],
    "git_log": ["git", "log", "--oneline", "-10"],
    "disk": ["df", "-h", "/data"],
    "smoke": [".venv/bin/python", "deploy/node0/activation_smoke_test.py"],
}


@charter("""
I am the Executor. I run safe, pre-approved operations and capture their
results as receipted evidence. I NEVER run destructive commands (rm, drop,
reset --hard, push --force). I ONLY execute from a whitelist of verified
safe operations. I report results honestly — failures are evidence too.
""")
class ExecutorAgent(Agent):
    name = "Executor"
    governance_class = "PAT"
    model = MODEL

    @tool(max_results=10)
    def run_safe_command(self, command_name: str) -> List[str]:
        """Execute a whitelisted safe command and capture output."""
        refs: list[str] = []

        if command_name not in SAFE_COMMANDS:
            self._evidence_text = f"REFUSED: '{command_name}' not in whitelist. Available: {list(SAFE_COMMANDS.keys())}"
            refs.append(f"executor:refused:{command_name}")
            return refs

        cmd = SAFE_COMMANDS[command_name]
        cwd = str(DATA_LAKE_ROOT)
        if command_name == "cargo_check":
            cwd = str(DATA_LAKE_ROOT / "bizra-omega")

        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd,
            )
            output = r.stdout[-2000:] if r.stdout else ""
            stderr = r.stderr[-500:] if r.stderr else ""
            status = "PASS" if r.returncode == 0 else "FAIL"

            self._evidence_text = (
                f"Command: {command_name}\n"
                f"Status: {status} (exit code {r.returncode})\n"
                f"Output:\n{output}\n"
                f"{'Stderr: ' + stderr if stderr else ''}"
            )
            refs.append(f"executor:{command_name}:{status}")
        except subprocess.TimeoutExpired:
            self._evidence_text = f"Command: {command_name}\nStatus: TIMEOUT (>120s)"
            refs.append(f"executor:{command_name}:TIMEOUT")
        except FileNotFoundError:
            self._evidence_text = f"Command: {command_name}\nStatus: NOT_FOUND"
            refs.append(f"executor:{command_name}:NOT_FOUND")

        return refs

    async def act(self, mission: Mission):
        # Parse which command to run from the question
        question = mission.question.lower()
        matched_cmd = None

        for cmd_name in SAFE_COMMANDS:
            if cmd_name.replace("_", " ") in question or cmd_name in question:
                matched_cmd = cmd_name
                break

        if not matched_cmd:
            # Default: run health check
            if "test" in question:
                matched_cmd = "test_run"
            elif "build" in question or "cargo" in question or "rust" in question:
                matched_cmd = "cargo_check"
            elif "smoke" in question:
                matched_cmd = "smoke"
            elif "status" in question or "health" in question:
                matched_cmd = "health"
            else:
                return self.refuse(
                    reason=f"Cannot determine command from question. Available: {list(SAFE_COMMANDS.keys())}"
                )

        refs = self.run_safe_command(matched_cmd)

        if not refs:
            return self.refuse(reason="Command execution failed with no output")

        # Use LLM to summarize results
        system = (
            "You are a BIZRA Executor agent. Summarize the command execution "
            "results concisely. Report: what ran, what passed/failed, key numbers. "
            "If something failed, explain what the failure means."
        )
        prompt = f"EXECUTION RESULTS:\n{self._evidence_text}\n\nSUMMARY:"

        answer = _call_ollama(prompt, system, self.model)
        if answer.startswith("ERROR:"):
            # Still report the raw results even if LLM is down
            answer = self._evidence_text

        return self.draft(content=answer, evidence=refs)


def _call_ollama(prompt: str, system: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            msg = data.get("message", {})
            return msg.get("content", "") or msg.get("thinking", "")
    except (urllib.error.URLError, TimeoutError) as e:
        return f"ERROR: Ollama unreachable — {e}"
