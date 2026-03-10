#!/bin/bash
# A+ Implementation Verification Script
set -e

cd /mnt/c/BIZRA-DATA-LAKE
PASS=0; FAIL=0

echo "=== A+ Implementation Verification ==="
echo ""

# Check implementation markers
echo "[1/2] Checking implementation markers..."
markers=(
    "core/inference/gateway.py:CircuitBreaker"
    "core/inference/gateway.py:RateLimiter"
    "core/inference/gateway.py:ConnectionPool"
    "core/pci/crypto.py:timing_safe_compare"
)

for marker in "${markers[@]}"; do
    file="${marker%%:*}"
    pattern="${marker##*:}"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  [PASS] $pattern in $file"
        ((PASS++))
    else
        echo "  [FAIL] $pattern not found in $file"
        ((FAIL++))
    fi
done

# Run tests
echo ""
echo "[2/2] Running test suite..."
tests=(
    "tests/core/pci/test_replay_protection.py"
    "tests/core/pci/test_timing_safe.py"
    "tests/integration/test_full_reasoning_cycle.py"
)

for test in "${tests[@]}"; do
    if [ -f "$test" ]; then
        if python -m pytest "$test" -q --tb=no 2>/dev/null; then
            echo "  [PASS] $test"
            ((PASS++))
        else
            echo "  [FAIL] $test"
            ((FAIL++))
        fi
    else
        echo "  [SKIP] $test (not found)"
    fi
done

# Summary
echo ""
echo "=== Summary ==="
echo "PASS: $PASS | FAIL: $FAIL"
[ $FAIL -eq 0 ] && echo "STATUS: ALL CHECKS PASSED" || echo "STATUS: SOME CHECKS FAILED"
exit $FAIL
