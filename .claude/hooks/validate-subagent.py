#!/usr/bin/env python3
"""
BIZRA SubagentStop Hook
Validates sub-agent compliance with BIZRA architectural principles.
"""

import json
import sys
import os
from datetime import datetime, timezone

# BIZRA compliance rules for sub-agents
COMPLIANCE_RULES = {
    "receipt_emission": {
        "description": "Sub-agent must emit receipts for significant actions",
        "critical": True,
    },
    "fail_closed": {
        "description": "Errors must not be silently ignored",
        "critical": True,
    },
    "ihsan_compliance": {
        "description": "Actions must maintain Ihsan threshold",
        "critical": True,
    },
    "evidence_trail": {
        "description": "Actions should produce auditable evidence",
        "critical": False,
    },
}

# Patterns that indicate potential compliance issues
WARNING_PATTERNS = [
    ("silent failure", "Error was not properly escalated"),
    ("skipped validation", "Validation step was bypassed"),
    ("no receipt", "Action did not emit receipt"),
    ("threshold lowered", "Ihsan threshold was modified"),
    ("force push", "Destructive git operation detected"),
]


def analyze_subagent_output(data: dict) -> dict:
    """Analyze sub-agent output for BIZRA compliance."""
    issues = []
    warnings = []

    # Extract relevant fields
    agent_id = data.get("agent_id", "unknown")
    agent_type = data.get("subagent_type", "unknown")
    output = data.get("output", "")
    exit_status = data.get("exit_status", "unknown")

    # Check for error conditions
    if exit_status == "error":
        issues.append({
            "rule": "fail_closed",
            "message": f"Sub-agent {agent_type} exited with error status",
            "severity": "high",
        })

    # Check for warning patterns in output
    output_lower = output.lower() if isinstance(output, str) else ""
    for pattern, message in WARNING_PATTERNS:
        if pattern in output_lower:
            warnings.append({
                "pattern": pattern,
                "message": message,
            })

    # Determine overall compliance
    critical_issues = [i for i in issues if COMPLIANCE_RULES.get(i.get("rule", ""), {}).get("critical", False)]

    return {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "compliant": len(critical_issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "analyzed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main():
    """Main hook entry point."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # If no valid JSON, pass through
        print(json.dumps({"continue": True}))
        return 0
    except Exception:
        # On any error, allow to continue (non-blocking)
        print(json.dumps({"continue": True}))
        return 0

    # Analyze compliance
    result = analyze_subagent_output(input_data)

    # Output analysis summary for logging
    agent_type = result.get("agent_type", "unknown")
    issue_count = len(result.get("issues", []))
    warning_count = len(result.get("warnings", []))

    # Build response
    response = {"continue": True}  # Always continue, hook is informational

    # Add context message if there are issues
    if issue_count > 0 or warning_count > 0:
        summary_parts = []
        if issue_count > 0:
            summary_parts.append(f"{issue_count} compliance issue(s)")
        if warning_count > 0:
            summary_parts.append(f"{warning_count} warning(s)")

        response["message"] = f"[SubagentStop] {agent_type}: {', '.join(summary_parts)}"

        # Log details to stderr for visibility
        print(f"BIZRA SubagentStop Analysis: {json.dumps(result)}", file=sys.stderr)

    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    sys.exit(main())
