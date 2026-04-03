#!/usr/bin/env python3
"""
BIZRA Bash Post-Execution Hook (PostToolUse Hook)
Analyzes bash command results for issues and provides feedback
"""

import json
import re
import sys
from typing import List, Optional

# Patterns indicating failures that Claude should address
# NOTE: These are intentionally conservative to avoid false positives
ERROR_PATTERNS = [
    # Only match very clear error patterns to avoid noise
    # (r"error:", "Command reported errors"),  # Too broad
    # (r"fatal:", "Fatal error occurred"),  # Too broad
    # (r"cannot\s+find", "Resource not found"),  # Too broad
    (r"permission\s+denied", "Permission denied - may need sudo or file permissions"),
    # (r"No such file or directory", "File or directory doesn't exist"),  # Too common
    (r"command not found", "Command not available - may need installation"),
    # (r"Connection refused", "Connection failed - service may not be running"),
    # (r"FAIL", "Test failures detected"),  # Too broad
]

# Patterns indicating security issues
SECURITY_PATTERNS = [
    (r"(?i)password.*plain.*text", "Password in plaintext detected"),
    (r"(?i)secret.*exposed", "Secret exposure detected"),
    (r"(?i)vulnerability.*found", "Security vulnerability detected"),
]

# BIZRA-specific patterns
BIZRA_VALIDATION_PATTERNS = [
    (r"Ihsan.*score.*<.*0\.95", "Ihsān score below threshold 0.95"),
    (
        r"SAT.*consensus.*failed",
        "SAT validation consensus failed - must reach 3/5 approval",
    ),
    (r"SAPE.*probe.*failed", "SAPE probe validation failed"),
    (r"Receipt.*missing", "Receipt emission missing - violates receipt-first policy"),
    (r"FATE.*escalation.*critical", "FATE critical escalation triggered"),
]


def analyze_output(
    command: str, output: str, exit_code: int
) -> tuple[Optional[str], List[str]]:
    """
    Analyze command output for issues
    Returns: (block_reason, warnings)
    """
    warnings = []
    block_reason = None

    # Check for error patterns
    for pattern, message in ERROR_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            warnings.append(f"⚠️ {message}")

    # Check for security issues (these should block)
    for pattern, message in SECURITY_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            block_reason = f"🔒 SECURITY: {message}"
            break

    # Check for BIZRA-specific validation failures (these should block)
    for pattern, message in BIZRA_VALIDATION_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            block_reason = f"🛑 FAIL-CLOSED: {message}"
            break

    return block_reason, warnings


def format_context(warnings: List[str], command: str) -> str:
    """Format additional context for Claude"""
    context_parts = ["Command execution completed with issues:"]
    context_parts.append(f"Command: {command}")
    context_parts.extend(warnings)
    return "\n".join(context_parts)


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)  # Not a Bash command, allow

    tool_input = input_data.get("tool_input", {})
    tool_response = input_data.get("tool_response", {})

    command = tool_input.get("command", "")
    output = tool_response.get("output", "")
    exit_code = tool_response.get("exit_code", 0)

    # Analyze the output
    block_reason, warnings = analyze_output(command, output, exit_code)

    # If we should block, use decision control
    if block_reason:
        output_json = {
            "decision": "block",
            "reason": f"{block_reason}\n\n"
            "The command executed but produced output that violates BIZRA safety policies. "
            "Please review the output and take corrective action.",
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": format_context(warnings, command),
            },
        }
        print(json.dumps(output_json))
        sys.exit(0)

    # If there are warnings but not blocking, add context
    if warnings:
        output_json = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": format_context(warnings, command),
            }
        }
        print(json.dumps(output_json))
        sys.exit(0)

    # No issues
    sys.exit(0)


if __name__ == "__main__":
    main()
