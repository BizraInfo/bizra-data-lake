#!/bin/bash
# BIZRA CI/CD Gate Runner
# Runs Claude Code validation gates in CI/CD pipelines
#
# Usage: ./ci-claude-gate.sh [gate] [options]
#
# Gates:
#   build       Build validation
#   test        Test validation
#   ihsan       Ihsān constitution validation
#   sape        SAPE probe validation
#   receipts    Receipt evidence validation
#   full        All gates (default)
#
# Options:
#   --json          Output JSON results
#   --max-turns N   Max agentic turns (default: 10)
#   --max-budget N  Max USD budget (default: 2.00)
#   --timeout N     Timeout in seconds (default: 300)

set -e

# Defaults
GATE="full"
JSON_OUTPUT=false
MAX_TURNS=10
MAX_BUDGET="2.00"
TIMEOUT=300

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    build|test|ihsan|sape|receipts|full)
      GATE="$1"
      shift
      ;;
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    --max-turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --max-budget)
      MAX_BUDGET="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Base Claude command for CI
CLAUDE_BASE="claude -p --no-session-persistence --max-turns $MAX_TURNS --max-budget-usd $MAX_BUDGET"

if [ "$JSON_OUTPUT" = true ]; then
  CLAUDE_BASE="$CLAUDE_BASE --output-format json"
fi

# Gate functions
run_build_gate() {
  echo "Running Build Gate..."
  $CLAUDE_BASE --allowedTools "Bash(cargo:*)" "Read" \
    "Build Rust in release mode. Run cargo build --release. Report success or failure with any error messages."
}

run_test_gate() {
  echo "Running Test Gate..."
  $CLAUDE_BASE --allowedTools "Bash(cargo test:*)" "Bash(pytest:*)" "Read" \
    "Run all tests: 'cargo test' for Rust, 'pytest tests/' for Python. Report total tests, passed, failed. Exit 1 on any failure."
}

run_ihsan_gate() {
  echo "Running Ihsān Gate..."
  $CLAUDE_BASE --allowedTools "Read" "Bash(python3:*)" \
    "Validate constitution/ihsan_v1.yaml:
1. Verify 8 dimensions: correctness, safety, user_benefit, efficiency, auditability, anti_centralization, robustness, adl_fairness
2. Verify weights sum to exactly 1.0
3. Verify production threshold is 0.99
Report status: PASS or FAIL with details."
}

run_sape_gate() {
  echo "Running SAPE Gate..."
  $CLAUDE_BASE --allowedTools "Read" "Grep" "Bash(python3:*)" \
    "Validate SAPE probe system:
1. Check src/sape.rs exists and defines 9 probes: threat_scan, compliance, bias, user_benefit, correctness, safety, groundedness, relevance, fluency
2. Verify core/sape.py implements same probes
3. Check probe threshold handling
Report status: PASS or FAIL with details."
}

run_receipts_gate() {
  echo "Running Receipts Gate..."
  $CLAUDE_BASE --allowedTools "Read" "Glob" "Bash(python3:*)" \
    "Validate receipt evidence in docs/evidence/receipts/:
1. Find all .json files
2. Validate each has required fields: receipt_id, timestamp, task_summary, rejection_codes, escalation_level, integrity_hash
3. Report total receipts, valid count, invalid count with details
Report status: PASS or FAIL."
}

run_full_gate() {
  echo "Running Full Gate Suite..."

  local exit_code=0

  echo "=== Gate 1/5: Build ==="
  if ! run_build_gate; then
    echo "Build gate FAILED"
    exit_code=1
  fi

  echo ""
  echo "=== Gate 2/5: Test ==="
  if ! run_test_gate; then
    echo "Test gate FAILED"
    exit_code=1
  fi

  echo ""
  echo "=== Gate 3/5: Ihsān ==="
  if ! run_ihsan_gate; then
    echo "Ihsān gate FAILED"
    exit_code=1
  fi

  echo ""
  echo "=== Gate 4/5: SAPE ==="
  if ! run_sape_gate; then
    echo "SAPE gate FAILED"
    exit_code=1
  fi

  echo ""
  echo "=== Gate 5/5: Receipts ==="
  if ! run_receipts_gate; then
    echo "Receipts gate FAILED"
    exit_code=1
  fi

  echo ""
  if [ $exit_code -eq 0 ]; then
    echo "════════════════════════════════════════"
    echo "  ALL GATES PASSED"
    echo "════════════════════════════════════════"
  else
    echo "════════════════════════════════════════"
    echo "  SOME GATES FAILED"
    echo "════════════════════════════════════════"
  fi

  return $exit_code
}

# Run gate with timeout
run_with_timeout() {
  timeout "$TIMEOUT" "$@"
  local exit_code=$?
  if [ $exit_code -eq 124 ]; then
    echo "Gate timed out after ${TIMEOUT}s"
    return 1
  fi
  return $exit_code
}

# Main execution
case $GATE in
  build)
    run_with_timeout run_build_gate
    ;;
  test)
    run_with_timeout run_test_gate
    ;;
  ihsan)
    run_with_timeout run_ihsan_gate
    ;;
  sape)
    run_with_timeout run_sape_gate
    ;;
  receipts)
    run_with_timeout run_receipts_gate
    ;;
  full)
    run_with_timeout run_full_gate
    ;;
esac
