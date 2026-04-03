#!/usr/bin/env python3
"""
BIZRA File Operations Validator (PreToolUse Hook)
Validates Write/Edit operations for critical files and schema changes
"""

import json
import os
import re
import sys
from typing import List, Tuple

# Protected files that require special handling
PROTECTED_FILES = {
    "constitution/ihsan_v1.yaml": "Constitution - Single source of truth for ethical weights",
    "src/receipts.rs": "Receipt Schema - Requires updating core/fate.py, tests/, docs/",
    "core/fate.py": "FATE Engine - Receipt schema must stay in sync with src/receipts.rs",
    "docker-compose.yml": "Service Configuration - Affects all infrastructure",
    "model-family-genesis-v1-SEALED.yaml": "Model Configuration - SEALED, requires unsealing",
    ".env": "Environment Secrets - Never commit, verify no secrets in content",
    ".github/workflows/elite-ci-cd.yml": "CI/CD Pipeline - Affects deployment gates",
}

# Path patterns that are always safe
SAFE_PATH_PATTERNS = [
    r"^docs/.*\.md$",  # Documentation
    r"^tests/.*\.py$",  # Test files
    r"^tests/.*\.rs$",  # Rust test files
    r"^scripts/.*\.py$",  # Utility scripts
    r"^\.claude/",  # Claude Code settings
]

# File extensions that require extra validation
CRITICAL_EXTENSIONS = {
    ".rs": "Rust source - Run cargo clippy and cargo test after changes",
    ".py": "Python source - Run pytest after changes",
    ".yaml": "YAML config - Validate syntax with yamllint",
    ".yml": "YAML config - Validate syntax with yamllint",
    ".toml": "TOML config - Validate Cargo.toml syntax",
}


def is_safe_path(file_path: str) -> bool:
    """Check if path matches safe patterns"""
    for pattern in SAFE_PATH_PATTERNS:
        if re.match(pattern, file_path):
            return True
    return False


def check_protected_file(file_path: str) -> Tuple[bool, str]:
    """Check if file is protected"""
    for protected_path, description in PROTECTED_FILES.items():
        if file_path.endswith(protected_path) or protected_path in file_path:
            return True, description
    return False, ""


def check_critical_extension(file_path: str) -> Tuple[bool, str]:
    """Check if file extension requires validation"""
    ext = os.path.splitext(file_path)[1]
    if ext in CRITICAL_EXTENSIONS:
        return True, CRITICAL_EXTENSIONS[ext]
    return False, ""


def check_receipt_schema_impact(file_path: str, content: str = None) -> List[str]:
    """Check if changes might affect receipt schema"""
    warnings = []

    if "receipts.rs" in file_path or "fate.py" in file_path:
        warnings.append("🔐 Receipt Schema Impact Detected")
        warnings.append("   Required updates:")
        warnings.append("   1. src/receipts.rs (Rust struct)")
        warnings.append("   2. core/fate.py (Python equivalent)")
        warnings.append("   3. Tests in tests/")
        warnings.append("   4. Evidence docs in docs/execution/")
        warnings.append("   5. CLAUDE.md documentation")

    if content and ("struct Receipt" in content or "class Receipt" in content):
        warnings.append("⚠️ Receipt struct/class modification detected")
        warnings.append("   Ensure backward compatibility for existing receipts")

    return warnings


def validate_file_operation(
    tool_name: str, file_path: str, content: str = None
) -> Tuple[bool, List[str]]:
    """
    Validate file operation
    Returns: (should_ask_permission, warnings)
    """
    warnings = []

    # Safe paths can proceed
    if is_safe_path(file_path):
        return False, []

    # Check protected files
    is_protected, protection_reason = check_protected_file(file_path)
    if is_protected:
        warnings.append(f"🛡️ Protected File: {protection_reason}")
        warnings.append(f"   Path: {file_path}")

    # Check critical extensions
    is_critical, critical_reason = check_critical_extension(file_path)
    if is_critical:
        warnings.append(f"⚙️ {critical_reason}")

    # Check receipt schema impact
    schema_warnings = check_receipt_schema_impact(file_path, content)
    warnings.extend(schema_warnings)

    # Check for secrets in content
    if content and tool_name == "Write":
        if re.search(r"(?i)(password|secret|key|token)\s*[=:]\s*['\"]?\w+", content):
            warnings.append("⚠️ Potential secrets detected in file content")
            warnings.append("   Verify no sensitive data is being written")

    # Ask for permission if there are any warnings
    should_ask = len(warnings) > 0

    return should_ask, warnings


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    if tool_name not in ["Write", "Edit"]:
        sys.exit(0)  # Not a file operation, allow

    file_path = tool_input.get("file_path", "")
    content = tool_input.get("content") or tool_input.get("new_string")

    # Make path relative to project root for display
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if project_dir and file_path.startswith(project_dir):
        file_path = file_path[len(project_dir) :].lstrip("/")

    # Validate the operation
    should_ask, warnings = validate_file_operation(tool_name, file_path, content)

    if should_ask:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": "\n".join(warnings)
                + "\n\nProceed with this file operation?",
            }
        }
        print(json.dumps(output))
        sys.exit(0)

    # Operation is safe, allow
    sys.exit(0)


if __name__ == "__main__":
    main()
