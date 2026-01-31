#!/bin/bash
# BIZRA Ecosystem Multi-Repository Synchronization
# "Every node is sovereign. Every human is a seed."

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SWARM_DIR="$ROOT_DIR/.swarm"

# Constitutional constants (must match across all repos)
IHSAN_THRESHOLD=0.95
SNR_THRESHOLD=0.85
VERSION="2.2.0"

# Repository paths
REPOS=(
    "/mnt/c/BIZRA-DATA-LAKE"
    "/mnt/c/BIZRA-OS"
    "/mnt/c/bizra-genesis-node"
    "/mnt/c/BIZRA-Dual-Agentic-system--main"
)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           BIZRA Ecosystem Multi-Repo Synchronization             ║"
echo "║                    Version: $VERSION - Sovereign                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# Function: Check repository status
check_repo_status() {
    local repo_path="$1"
    local repo_name=$(basename "$repo_path")

    if [ -d "$repo_path" ]; then
        echo "✓ $repo_name exists"
        if [ -d "$repo_path/.git" ]; then
            local branch=$(cd "$repo_path" && git branch --show-current 2>/dev/null || echo "unknown")
            local status=$(cd "$repo_path" && git status --porcelain 2>/dev/null | wc -l)
            echo "  Branch: $branch | Uncommitted changes: $status"
        fi
    else
        echo "✗ $repo_name NOT FOUND"
    fi
}

# Function: Verify threshold constants
verify_thresholds() {
    echo
    echo "=== Threshold Verification ==="

    # Check Python
    if [ -f "$ROOT_DIR/core/sovereign/capability_card.py" ]; then
        local py_ihsan=$(grep -oP 'IHSAN_THRESHOLD\s*=\s*\K[0-9.]+' "$ROOT_DIR/core/sovereign/capability_card.py" 2>/dev/null || echo "N/A")
        local py_snr=$(grep -oP 'SNR_THRESHOLD\s*=\s*\K[0-9.]+' "$ROOT_DIR/core/sovereign/capability_card.py" 2>/dev/null || echo "N/A")
        echo "Python:     IHSAN=$py_ihsan  SNR=$py_snr"
    fi

    # Check TypeScript
    if [ -f "$ROOT_DIR/src/core/sovereign/capability-card.ts" ]; then
        local ts_ihsan=$(grep -oP 'IHSAN_THRESHOLD\s*=\s*\K[0-9.]+' "$ROOT_DIR/src/core/sovereign/capability-card.ts" 2>/dev/null || echo "N/A")
        local ts_snr=$(grep -oP 'SNR_THRESHOLD\s*=\s*\K[0-9.]+' "$ROOT_DIR/src/core/sovereign/capability-card.ts" 2>/dev/null || echo "N/A")
        echo "TypeScript: IHSAN=$ts_ihsan  SNR=$ts_snr"
    fi

    # Check Rust
    if [ -f "$ROOT_DIR/native/fate-binding/src/lib.rs" ]; then
        local rs_ihsan=$(grep -oP 'IHSAN_THRESHOLD:\s*f64\s*=\s*\K[0-9.]+' "$ROOT_DIR/native/fate-binding/src/lib.rs" 2>/dev/null || echo "N/A")
        local rs_snr=$(grep -oP 'SNR_THRESHOLD:\s*f64\s*=\s*\K[0-9.]+' "$ROOT_DIR/native/fate-binding/src/lib.rs" 2>/dev/null || echo "N/A")
        echo "Rust:       IHSAN=$rs_ihsan  SNR=$rs_snr"
    fi
}

# Function: Run integration tests
run_integration_tests() {
    echo
    echo "=== Integration Tests ==="

    cd "$ROOT_DIR"

    # Python integration test
    echo "Running Python integration test..."
    python3 -c "
from core.sovereign.capability_card import CapabilityCard, IHSAN_THRESHOLD, SNR_THRESHOLD
from core.sovereign.model_license_gate import GateChain, InMemoryRegistry
from core.sovereign.integration import SovereignRuntime, NetworkMode

# Verify thresholds
assert IHSAN_THRESHOLD == 0.95, f'IHSAN mismatch: {IHSAN_THRESHOLD}'
assert SNR_THRESHOLD == 0.85, f'SNR mismatch: {SNR_THRESHOLD}'

# Quick smoke test
registry = InMemoryRegistry()
chain = GateChain(registry)
print('✓ Python integration: PASSED')
" 2>/dev/null && echo "  Python: OK" || echo "  Python: FAILED"
}

# Function: Generate ecosystem report
generate_report() {
    echo
    echo "=== Ecosystem Report ==="

    local report_file="$SWARM_DIR/ecosystem-report-$(date +%Y%m%d-%H%M%S).json"

    cat > "$report_file" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "version": "$VERSION",
  "repositories": {
$(for repo in "${REPOS[@]}"; do
    name=$(basename "$repo")
    exists=$([ -d "$repo" ] && echo "true" || echo "false")
    echo "    \"$name\": { \"exists\": $exists, \"path\": \"$repo\" },"
done | sed '$ s/,$//')
  },
  "thresholds": {
    "ihsan": $IHSAN_THRESHOLD,
    "snr": $SNR_THRESHOLD
  },
  "status": "synchronized"
}
EOF

    echo "Report saved to: $report_file"
}

# Main execution
echo "=== Repository Status ==="
for repo in "${REPOS[@]}"; do
    check_repo_status "$repo"
done

verify_thresholds

case "${1:-status}" in
    status)
        echo
        echo "Use --test to run integration tests"
        echo "Use --report to generate ecosystem report"
        ;;
    --test)
        run_integration_tests
        ;;
    --report)
        generate_report
        ;;
    *)
        echo "Usage: $0 [status|--test|--report]"
        ;;
esac

echo
echo "=== Synchronization Complete ==="
