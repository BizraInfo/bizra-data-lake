#!/bin/bash
# BIZRA Full Validation Script
# Runs all validation gates before deployment
#
# Usage: ./bizra-validate.sh [--strict] [--json]
#
# Options:
#   --strict    Exit on first failure (default: run all)
#   --json      Output JSON results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Parse arguments
STRICT_MODE=false
JSON_OUTPUT=false
for arg in "$@"; do
  case $arg in
    --strict) STRICT_MODE=true ;;
    --json) JSON_OUTPUT=true ;;
  esac
done

# Colors (skip if JSON output)
if [ "$JSON_OUTPUT" = false ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  NC='\033[0m' # No Color
else
  RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

# Results tracking
declare -A RESULTS
FAILED=0

log() {
  if [ "$JSON_OUTPUT" = false ]; then
    echo -e "$1"
  fi
}

run_gate() {
  local gate_name="$1"
  local gate_cmd="$2"

  log "${BLUE}[GATE]${NC} Running: $gate_name"

  if eval "$gate_cmd" > /tmp/bizra_gate_output.txt 2>&1; then
    RESULTS["$gate_name"]="PASS"
    log "${GREEN}[PASS]${NC} $gate_name"
    return 0
  else
    RESULTS["$gate_name"]="FAIL"
    log "${RED}[FAIL]${NC} $gate_name"
    cat /tmp/bizra_gate_output.txt
    FAILED=$((FAILED + 1))
    if [ "$STRICT_MODE" = true ]; then
      exit 1
    fi
    return 1
  fi
}

# Header
log "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
log "${BLUE}║${NC}           BIZRA Full Validation Pipeline                   ${BLUE}║${NC}"
log "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
log ""

# Gate 1: Rust Build
run_gate "rust_build" "cd '$PROJECT_DIR' && cargo build --release 2>&1"

# Gate 2: Rust Clippy
run_gate "rust_clippy" "cd '$PROJECT_DIR' && cargo clippy --all-targets -- -D warnings 2>&1"

# Gate 3: Rust Tests
run_gate "rust_tests" "cd '$PROJECT_DIR' && cargo test 2>&1"

# Gate 4: Python Imports
run_gate "python_imports" "cd '$PROJECT_DIR' && python3 -c 'from core import main; print(\"OK\")' 2>&1"

# Gate 5: Python Tests
run_gate "python_tests" "cd '$PROJECT_DIR' && pytest tests/ --tb=short 2>&1" || true

# Gate 6: OpenClaw / PAT Build (if available)
if command -v pnpm >/dev/null 2>&1 && [ -f "$PROJECT_DIR/BIZRA-PAT/package.json" ]; then
  run_gate "pat_build" "cd '$PROJECT_DIR' && pnpm --dir BIZRA-PAT build 2>&1"
  if [ -f "$PROJECT_DIR/BIZRA-PAT/ui/package.json" ]; then
    run_gate "pat_ui_build" "cd '$PROJECT_DIR' && pnpm --dir BIZRA-PAT/ui build 2>&1"
  fi
fi

# Gate 7: Ihsān Constitution
run_gate "ihsan_validation" "cd '$PROJECT_DIR' && python3 << 'EOF'
import yaml
import sys

with open('constitution/ihsan_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

dimensions = const.get('dimensions', {})
required = ['correctness', 'safety', 'user_benefit', 'efficiency',
            'auditability', 'anti_centralization', 'robustness', 'adl_fairness']

missing = [d for d in required if d not in dimensions]
if missing:
    print(f'Missing dimensions: {missing}')
    sys.exit(1)

weights = [dimensions[d].get('weight', 0) for d in required]
total = sum(weights)
if abs(total - 1.0) > 0.01:
    print(f'Weights sum to {total}, expected 1.0')
    sys.exit(1)

threshold = (
    const.get('threshold_policy', {})
    .get('thresholds_by_env', {})
    .get('production', const.get('units', {}).get('threshold', 0))
)
if threshold < 0.95:
    print(f'Production threshold {threshold} < 0.95')
    sys.exit(1)

print('Ihsan constitution valid')
EOF
"

# Gate 8: Receipt Schema
run_gate "receipt_schema" "cd '$PROJECT_DIR' && python3 << 'EOF'
import json
import glob
import sys

receipt_dir = 'docs/evidence/receipts'
json_receipts = glob.glob(f'{receipt_dir}/**/*.json', recursive=True)
jsonl_receipts = glob.glob(f'{receipt_dir}/**/*.jsonl', recursive=True)

if not json_receipts and not jsonl_receipts:
    print('No receipts found (OK for fresh project)')
    sys.exit(0)

invalid = 0
for path in json_receipts:
    with open(path) as f:
        try:
            json.load(f)
        except json.JSONDecodeError as e:
            print(f'{path}: invalid JSON - {e}')
            invalid += 1

for path in jsonl_receipts:
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f'{path}:{line_no}: invalid JSONL - {e}')
                invalid += 1

if invalid > 0:
    print(f'{invalid} invalid receipts')
    sys.exit(1)

print(f'{len(json_receipts)} JSON receipts and {len(jsonl_receipts)} JSONL receipts valid')
EOF
"

# Summary
log ""
log "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
log "${BLUE}║${NC}                     Validation Summary                      ${BLUE}║${NC}"
log "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
log ""

if [ "$JSON_OUTPUT" = true ]; then
  # Output JSON
  echo "{"
  echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"total_gates\": ${#RESULTS[@]},"
  echo "  \"passed\": $((${#RESULTS[@]} - FAILED)),"
  echo "  \"failed\": $FAILED,"
  echo "  \"results\": {"
  first=true
  for gate in "${!RESULTS[@]}"; do
    if [ "$first" = true ]; then
      first=false
    else
      echo ","
    fi
    echo -n "    \"$gate\": \"${RESULTS[$gate]}\""
  done
  echo ""
  echo "  },"
  echo "  \"status\": \"$([ $FAILED -eq 0 ] && echo "PASS" || echo "FAIL")\""
  echo "}"
else
  for gate in "${!RESULTS[@]}"; do
    if [ "${RESULTS[$gate]}" = "PASS" ]; then
      log "${GREEN}✓${NC} $gate"
    else
      log "${RED}✗${NC} $gate"
    fi
  done

  log ""
  log "Total: ${#RESULTS[@]} gates, $((${#RESULTS[@]} - FAILED)) passed, $FAILED failed"

  if [ $FAILED -eq 0 ]; then
    log ""
    log "${GREEN}════════════════════════════════════════════════════════════${NC}"
    log "${GREEN}  All BIZRA validation gates passed! Ready for deployment.  ${NC}"
    log "${GREEN}════════════════════════════════════════════════════════════${NC}"
  else
    log ""
    log "${RED}════════════════════════════════════════════════════════════${NC}"
    log "${RED}  $FAILED validation gate(s) failed. Fix before deployment.  ${NC}"
    log "${RED}════════════════════════════════════════════════════════════${NC}"
    exit 1
  fi
fi
