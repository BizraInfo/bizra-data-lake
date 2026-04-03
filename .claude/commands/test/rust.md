---
allowed-tools: Bash(cargo:*)
description: Run Rust test suite with validation
argument-hint: [test-name] [--nocapture]
---

# Rust Test Suite Execution

## Test Status

- Current branch: !`git branch --show-current`
- Uncommitted changes: !`git status --short | wc -l`
- Last test run: !`ls -t target/*/deps/*.d 2>/dev/null | head -1 | xargs stat -c %y 2>/dev/null || echo "Never"`

## Configuration

Test filter: **${1:-all}**
Output mode: **${2:-captured}**

## Your Task

### 1. Pre-test Validation
```bash
# Ensure code compiles
cargo check --tests

# Verify no uncommitted changes to critical files
git diff --name-only | grep -E "(receipts\.rs|fate\.py|ihsan_v1\.yaml)" && \
  echo "⚠️ WARNING: Uncommitted changes to critical files" || true
```

### 2. Run Tests
```bash
# Run all tests or specific test
if [ -n "$1" ]; then
    cargo test $1 $2 -- --test-threads=1
else
    cargo test --all-targets -- --test-threads=1
fi
```

### 3. Critical Test Categories

**MUST PASS** (Fail-Closed):
- `cargo test --test pat_sat_runtime_tests` - PAT/SAT integration
- `cargo test ihsan` - Ihsān gate tests
- `cargo test sape` - SAPE probe tests
- `cargo test receipts` - Receipt generation tests

**SHOULD PASS** (Non-blocking):
- `cargo test mcp` - MCP tool tests
- `cargo test a2a` - A2A protocol tests
- `cargo test reasoning` - Multi-method reasoning tests

### 4. Integration Tests
```bash
# Run integration tests
cargo test --test '*' --no-fail-fast

# Check for ignored tests
cargo test -- --ignored
```

### 5. Coverage Analysis (if available)
```bash
if command -v cargo-tarpaulin &> /dev/null; then
    cargo tarpaulin --out Html --output-dir target/coverage
    echo "Coverage report: target/coverage/index.html"
fi
```

## Test Result Analysis

After tests complete, analyze:

1. **Pass Rate**: Calculate percentage of passing tests
2. **Failed Tests**: List any failing tests with failure reasons
3. **Ignored Tests**: Note any tests that were skipped
4. **Performance**: Report slow tests (>1s execution time)

## BIZRA-Specific Validation

### Receipt Tests
- Verify receipt emission for all operations
- Check receipt integrity (SHA-256 hashes)
- Validate receipt schema compatibility

### Ihsān Tests
- Confirm 0.99 threshold enforcement
- Test all 8 ethical dimensions
- Verify constitution loading from YAML

### SAPE Tests
- Test all 9 probes (threat_scan, compliance, bias, etc.)
- Verify pattern elevation (>3 repetitions)
- Check Neo4j integration for high-stakes probes

### FATE Tests
- Test escalation levels (Low→Medium→High→Critical)
- Verify Redis persistence
- Check quarantine handling

## Fail-Closed Requirements

**BLOCK** if any of these occur:
- Ihsān gate tests fail
- Receipt emission tests fail
- SAPE probe tests fail
- PAT/SAT consensus tests fail

**WARN** but allow:
- Performance tests fail (investigate but don't block)
- Integration tests with external dependencies fail

## Evidence Generation

Create test receipt:
```json
{
  "receipt_id": "test-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "test_summary": {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "ignored": 0
  },
  "critical_tests": {
    "ihsan": "pass|fail",
    "sape": "pass|fail",
    "receipts": "pass|fail",
    "pat_sat": "pass|fail"
  },
  "execution_time": "0s"
}
```

Save to: `docs/evidence/receipts/test-$(date +%Y%m%d-%H%M%S).json`

## Report Format

```
## Rust Test Results

**Status**: ✅ PASS | ❌ FAIL
**Total Tests**: X passed, Y failed, Z ignored
**Execution Time**: Xs

### Critical Tests (MUST PASS)
- [ ] PAT/SAT Runtime: PASS
- [ ] Ihsān Gate: PASS
- [ ] SAPE Probes: PASS
- [ ] Receipt Generation: PASS

### Failed Tests (if any)
- test_name: failure_reason

### Performance
- Slowest test: test_name (Xs)

### Receipt Location
- docs/evidence/receipts/test-YYYYMMDD-HHMMSS.json
```
