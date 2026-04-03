#!/usr/bin/env python3
"""
BIZRA Bash Command Validator (PreToolUse Hook)
Enforces fail-closed safety and prevents destructive operations
"""

import json
import re
import sys
from typing import List, Tuple

# Blocklist: Commands that should NEVER run without explicit approval
BLOCKED_COMMANDS = [
    r"\brm\s+-rf\s+/",  # Recursive delete from root
    r"\bdd\s+if=/dev/zero\s+of=/dev/",  # Disk wipe
    r"\b:(){ :\|: & };:",  # Fork bomb
    r"\bmkfs\.",  # Format filesystem
    r"\bchmod\s+000",  # Remove all permissions
    r">\s*/dev/sda",  # Direct disk write
]

# High-risk patterns requiring validation
HIGH_RISK_PATTERNS = [
    (r"\brm\s+-rf", "Recursive force delete - ensure you're not deleting critical files"),
    (r"\bsudo\s+rm", "Sudo delete - verify the target path carefully"),
    (r"\bcurl.*\|\s*bash", "Piping to bash - security risk, prefer downloading first"),
    (r"\bwget.*\|\s*sh", "Piping to shell - security risk, prefer downloading first"),
    (r"--force-with-lease", "Force push - ensure you want to overwrite remote"),
    (r"git\s+push.*--force", "Force push - ensure you want to overwrite remote"),
]

# BIZRA-specific patterns
BIZRA_CRITICAL_PATHS = [
    r"constitution/ihsan_v1\.yaml",  # Never modify without review
    r"src/receipts\.rs",  # Receipt schema guard
    r"core/fate\.py",  # FATE engine
    r"docker-compose\.yml",  # Service configuration
    r"\.env",  # Environment secrets
    r"config/redis/.*\.pem",  # TLS certificates
]


def check_blocked_commands(command: str) -> Tuple[bool, str]:
    """Check if command matches any blocked patterns"""
    for pattern in BLOCKED_COMMANDS:
        if re.search(pattern, command):
            return True, f"BLOCKED: Command matches dangerous pattern: {pattern}"
    return False, ""


def check_high_risk(command: str) -> List[str]:
    """Check for high-risk patterns"""
    warnings = []
    for pattern, message in HIGH_RISK_PATTERNS:
        if re.search(pattern, command):
            warnings.append(f"⚠️ {message}")
    return warnings


def check_bizra_critical_paths(command: str) -> List[str]:
    """Check if command affects BIZRA critical files"""
    warnings = []
    for pattern in BIZRA_CRITICAL_PATHS:
        if re.search(pattern, command):
            warnings.append(
                f"⚠️ CRITICAL: Command affects protected BIZRA file: {pattern}"
            )
            warnings.append(
                "   → Receipt Schema Guard: Update src/receipts.rs, core/fate.py, tests/, and docs/"
            )
    return warnings


def validate_command(command: str) -> Tuple[bool, List[str]]:
    """
    Validate bash command
    Returns: (should_block, warnings)
    """
    # Check blocked commands first
    is_blocked, block_reason = check_blocked_commands(command)
    if is_blocked:
        return True, [block_reason]

    # Collect warnings
    warnings = []
    warnings.extend(check_high_risk(command))
    warnings.extend(check_bizra_critical_paths(command))

    return False, warnings


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    if tool_name != "Bash" or not command:
        sys.exit(0)  # Not a Bash command, allow

    # Validate the command
    should_block, warnings = validate_command(command)

    if should_block:
        # Exit code 2 blocks the command and shows stderr to Claude
        for warning in warnings:
            print(warning, file=sys.stderr)
        print("\n🛑 FAIL-CLOSED: Command blocked for safety", file=sys.stderr)
        print(
            "If this command is necessary, please ask the user for explicit approval.",
            file=sys.stderr,
        )
        sys.exit(2)

    if warnings:
        # Use JSON output to ask for permission
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": "\n".join(warnings)
                + "\n\nDo you want to proceed?",
            }
        }
        print(json.dumps(output))
        sys.exit(0)

    # Command is safe, allow
    sys.exit(0)


if __name__ == "__main__":
    main()
