#!/bin/bash
# Post-Rust Hook - Analyzes Rust compilation and suggests fixes
# Integrates with self-healing system for bizra-omega

set -e

PROJECT_ROOT="/mnt/c/BIZRA-DATA-LAKE"
OMEGA_ROOT="$PROJECT_ROOT/bizra-omega"
MEMORY_DIR="$PROJECT_ROOT/.claude-flow/memory"

# Analyze cargo output
analyze_cargo() {
    local output="$1"
    local exit_code="$2"

    if [ "$exit_code" -eq 0 ]; then
        # Check for warnings
        warning_count=$(echo "$output" | grep -c "warning:" || true)
        if [ "$warning_count" -gt 0 ]; then
            echo "⚠️ Compiled with $warning_count warning(s)"

            # Extract unused import warnings
            if echo "$output" | grep -q "unused import"; then
                echo "  💡 Run: cargo fix --lib --allow-dirty"
            fi

            # Extract dead code warnings
            if echo "$output" | grep -q "never read\|never used"; then
                echo "  💡 Consider removing unused code or prefixing with _"
            fi
        else
            echo "✅ Rust compiled successfully"
        fi
        return 0
    fi

    # Compilation failed
    echo "❌ Rust compilation failed"

    # Check for missing crate
    if echo "$output" | grep -q "can't find crate"; then
        crate=$(echo "$output" | grep -oP "can't find crate for \`\K[^\`]+" | head -1)
        echo "📦 Missing crate: $crate"
        echo "  💡 Add to Cargo.toml: $crate = \"*\""
        return 1
    fi

    # Check for type mismatch
    if echo "$output" | grep -q "mismatched types"; then
        echo "🔍 Type mismatch error"
        echo "$output" | grep -A2 "mismatched types" | head -5
        return 1
    fi

    # Check for borrow checker
    if echo "$output" | grep -q "borrowed\|cannot move\|lifetime"; then
        echo "🔒 Borrow checker error"
        echo "  💡 Consider using .clone(), Arc, or adjusting lifetimes"
        return 1
    fi

    # Check for trait bounds
    if echo "$output" | grep -q "trait bound.*is not satisfied"; then
        echo "🧩 Missing trait implementation"
        echo "$output" | grep "trait bound" | head -3
        return 1
    fi

    return 1
}

# Main entry point
main() {
    local exit_code="${1:-0}"
    local cargo_output="${2:-}"

    if [ -n "$cargo_output" ]; then
        analyze_cargo "$cargo_output" "$exit_code"
    fi

    # Log build
    echo "$(date -Iseconds) | exit=$exit_code | cargo" >> "$MEMORY_DIR/rust-build-history.log"
}

# Run if called directly
if [ $# -gt 0 ]; then
    main "$@"
fi
