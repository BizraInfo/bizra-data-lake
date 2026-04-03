#!/bin/bash
# BIZRA PostToolUse Hook for Write/Edit
# Runs linting, validation, and PAT compliance checks after file modifications
# Based on Claude Code best practices - deterministic, non-blocking

# Don't exit on errors - this is a post-processing hook
set +e

# Parse input JSON safely
INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null || echo "")
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null || echo "")

# Exit gracefully if no file path or file doesn't exist
if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
RELATIVE_PATH="${FILE_PATH#$PROJECT_DIR/}"
FILE_EXT="${FILE_PATH##*.}"

# Log file edit for audit trail (if logging enabled)
if [ -n "$BIZRA_HOOK_LOG" ]; then
    mkdir -p "$PROJECT_DIR/docs/evidence/receipts"
    echo "$(date -Iseconds) EDIT: $RELATIVE_PATH" >> "$PROJECT_DIR/docs/evidence/receipts/hook_audit.log"
fi

# File type specific processing
case "$FILE_EXT" in
    rs)
        # Rust files - silent for now (cargo build handles this)
        :
        ;;

    py)
        # Python files - quick syntax validation
        if command -v python3 &>/dev/null; then
            python3 -m py_compile "$FILE_PATH" 2>/dev/null || {
                echo "Python syntax issue in: $RELATIVE_PATH"
            }
        fi
        ;;

    yaml|yml)
        # YAML validation
        if command -v python3 &>/dev/null; then
            python3 -c "import yaml; yaml.safe_load(open('$FILE_PATH'))" 2>/dev/null || {
                echo "YAML syntax issue in: $RELATIVE_PATH"
            }
        fi

        # Constitution modification warnings
        case "$RELATIVE_PATH" in
            constitution/ihsan_v1.yaml)
                echo "Ihsan Constitution modified - verify weights sum to 1.0 and threshold = 0.95"
                ;;
            constitution/pat_enforcement_v1.yaml)
                echo "PAT Constitution modified - verify SNR >= 0.98, Novelty >= 0.75"
                ;;
        esac
        ;;

    json)
        # JSON validation
        if command -v jq &>/dev/null; then
            jq empty "$FILE_PATH" 2>/dev/null || {
                echo "JSON syntax issue in: $RELATIVE_PATH"
            }
        fi
        ;;

    md)
        # Markdown files - detect code blocks without language tags
        if grep -qE '^\s*```\s*$' "$FILE_PATH" 2>/dev/null; then
            echo "Markdown code blocks may need language tags: $RELATIVE_PATH"
        fi
        ;;
esac

# BIZRA Special File Handling
case "$RELATIVE_PATH" in
    src/receipts.rs|core/fate.py)
        echo ""
        echo "Receipt Schema Guard Activated"
        echo "Receipt schema modification detected!"
        echo ""
        echo "Required sync updates:"
        echo "  - src/receipts.rs (Rust struct)"
        echo "  - core/fate.py (Python equivalent)"
        echo "  - tests/ (update receipt tests)"
        echo "  - docs/execution/ (update evidence docs)"
        echo ""
        echo "Ensure backward compatibility for existing receipts!"
        ;;

    bizra_kernel/pat_*.py)
        echo "PAT component modified: $RELATIVE_PATH"
        echo "Run '/pat' to validate PAT constitution compliance"
        ;;

    .claude/*)
        echo "Claude Code config modified: $RELATIVE_PATH"
        ;;
esac

# All checks passed (non-blocking - always exit 0)
exit 0
