#!/usr/bin/env python3
"""
FATE Gate Hook for Claude Code
==============================
Constitutional AI validation as a PreToolUse hook.

Dimensions:
- Fidelity: Truth and accuracy
- Accountability: Audit trail
- Transparency: Explainable decisions
- Ethics: Harm prevention

Usage in .claude/settings.json:
{
  "hooks": {
    "PreToolUse": [{
      "matcher": { "tool_name": ".*" },
      "hooks": ["python .claude/hooks/fate_gate.py"]
    }]
  }
}
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# FATE thresholds
IHSAN_THRESHOLD = 0.95
ELITE_THRESHOLD = 0.98

# High-risk tools requiring stricter validation
HIGH_RISK_TOOLS = {
    "Bash": 0.98,      # Command execution
    "Write": 0.96,     # File modification
    "Edit": 0.96,      # Code changes
    "WebFetch": 0.95,  # External access
}

# Blocked patterns (Ethics dimension)
BLOCKED_PATTERNS = [
    "rm -rf /",
    "format c:",
    ":(){:|:&};:",  # Fork bomb
    "dd if=/dev/zero",
    "mkfs.",
    "> /dev/sda",
]


def compute_fate_score(tool_name: str, tool_input: dict) -> dict:
    """Compute FATE dimensions for a tool invocation."""
    scores = {
        "fidelity": 1.0,
        "accountability": 1.0,
        "transparency": 1.0,
        "ethics": 1.0,
    }

    # Fidelity: Check for hardcoded secrets
    input_str = json.dumps(tool_input).lower()
    secret_patterns = ["api_key", "password", "secret", "token", "credential"]
    for pattern in secret_patterns:
        if pattern in input_str and "=" in input_str:
            scores["fidelity"] *= 0.7

    # Accountability: Tool must have traceable input
    if not tool_input:
        scores["accountability"] *= 0.9

    # Transparency: Check for obfuscated commands
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        # Base64 encoded commands reduce transparency
        if "base64" in command and "|" in command:
            scores["transparency"] *= 0.8
        # Piped commands with curl reduce transparency
        if "curl" in command and "|" in command and "sh" in command:
            scores["transparency"] *= 0.7

    # Ethics: Block dangerous patterns
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        for pattern in BLOCKED_PATTERNS:
            if pattern in command:
                scores["ethics"] = 0.0
                break

    # Compute composite score (geometric mean)
    composite = (
        scores["fidelity"] *
        scores["accountability"] *
        scores["transparency"] *
        scores["ethics"]
    ) ** 0.25

    return {
        "dimensions": scores,
        "composite": composite,
        "threshold": HIGH_RISK_TOOLS.get(tool_name, IHSAN_THRESHOLD),
    }


def log_decision(tool_name: str, fate_result: dict, decision: str):
    """Log FATE gate decision for accountability."""
    log_dir = Path(".claude/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": tool_name,
        "fate_scores": fate_result["dimensions"],
        "composite": fate_result["composite"],
        "threshold": fate_result["threshold"],
        "decision": decision,
    }

    log_file = log_dir / "fate_gate.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def main():
    """Main hook entry point."""
    # Read hook input from stdin
    input_data = json.load(sys.stdin)

    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})

    # Compute FATE score
    fate_result = compute_fate_score(tool_name, tool_input)

    # Make decision
    if fate_result["composite"] >= fate_result["threshold"]:
        decision = "allow"
        log_decision(tool_name, fate_result, decision)
        # Output nothing to allow the tool to proceed
        sys.exit(0)
    else:
        decision = "block"
        log_decision(tool_name, fate_result, decision)

        # Output block reason
        reason = f"FATE Gate blocked: composite={fate_result['composite']:.3f} < threshold={fate_result['threshold']}"

        # Find the failing dimension
        for dim, score in fate_result["dimensions"].items():
            if score < 0.9:
                reason += f" ({dim}={score:.2f})"

        output = {
            "decision": "block",
            "reason": reason,
        }
        print(json.dumps(output))
        sys.exit(0)


if __name__ == "__main__":
    main()
