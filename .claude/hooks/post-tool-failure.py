#!/usr/bin/env python3
"""
BIZRA PostToolUseFailure Hook
Tracks tool failures for diagnostics and fail-closed enforcement.
"""

import json
import os
import sys
from datetime import datetime, timezone


def main():
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})
    error = input_data.get("tool_response", {}).get("error", "")

    # Log failure to audit trail
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if project_dir:
        log_dir = os.path.join(project_dir, "docs", "evidence", "receipts")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "hook_failures.jsonl")

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "tool": tool_name,
            "error_summary": str(error)[:200],
        }

        # Add file path for Write/Edit failures
        if tool_name in ("Write", "Edit") and "file_path" in tool_input:
            entry["file_path"] = tool_input["file_path"]

        # Add command for Bash failures
        if tool_name == "Bash" and "command" in tool_input:
            entry["command"] = tool_input["command"][:200]

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # Provide context to Claude about the failure
    context_parts = [f"Tool failure: {tool_name}"]
    if error:
        context_parts.append(f"Error: {str(error)[:150]}")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUseFailure",
            "additionalContext": "\n".join(context_parts),
        }
    }
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
