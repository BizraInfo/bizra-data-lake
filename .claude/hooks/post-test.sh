#!/bin/bash
# Post-Test Hook - Analyzes test failures and suggests fixes
# Integrates with self-healing system

set -e

PROJECT_ROOT="/mnt/c/BIZRA-DATA-LAKE"
MEMORY_DIR="$PROJECT_ROOT/.claude-flow/memory"

# Source self-healing functions
source "$PROJECT_ROOT/.claude/hooks/self-healing.sh" 2>/dev/null || true

# Analyze pytest output
analyze_pytest() {
    local output="$1"
    local exit_code="$2"

    if [ "$exit_code" -eq 0 ]; then
        # Extract pass count
        passed=$(echo "$output" | grep -oP '\d+(?= passed)' | head -1)
        echo "✅ All tests passed ($passed tests)"
        return 0
    fi

    # Extract failure info
    failed=$(echo "$output" | grep -oP '\d+(?= failed)' | head -1)
    echo "❌ $failed test(s) failed"

    # Check for collection errors (import issues)
    if echo "$output" | grep -q "ERROR collecting"; then
        echo "📦 Collection error detected - likely import issue"
        error_type=$(detect_error_type "$output")
        if [ "$error_type" != "unknown" ]; then
            suggest_fix "$error_type"
        fi
        return 1
    fi

    # Check for assertion failures
    if echo "$output" | grep -q "AssertionError"; then
        echo "🔍 Assertion failure - check test expectations"
        # Extract failed test names
        echo "$output" | grep -oP 'FAILED.*::test_\w+' | head -5
        return 1
    fi

    # Check for timeout issues
    if echo "$output" | grep -q "asyncio.TimeoutError\|TimeoutExpired"; then
        echo "⏱️ Timeout detected - consider increasing wait times in async tests"
        return 1
    fi

    return 1
}

# Main entry point
main() {
    local exit_code="${1:-0}"
    local test_output="${2:-}"

    if [ -n "$test_output" ]; then
        analyze_pytest "$test_output" "$exit_code"
    fi

    # Log test run
    echo "$(date -Iseconds) | exit=$exit_code | tests" >> "$MEMORY_DIR/test-history.log"
}

# Run if called directly with arguments
if [ $# -gt 0 ]; then
    main "$@"
fi
