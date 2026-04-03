---
allowed-tools: Bash(pytest:*), Bash(python*:*)
description: Run Python test suite with coverage
argument-hint: [test-path] [-v|-vv]
---

# Python Test Suite Execution

## Test Environment

- Python version: !`python3 --version`
- Pytest version: !`pytest --version`
- Coverage plugin: !`pip show pytest-cov 2>/dev/null | grep Version || echo "Not installed"`

## Configuration

Test path: **${1:-tests/}**
Verbosity: **${2:--v}**

## Your Task

### 1. Pre-test Validation
```bash
# Validate imports
python3 -m compileall core/

# Check critical modules
python3 -c "from core import main, sape, fate; print('✓ Imports OK')"
```

### 2. Run Tests with Coverage
```bash
# Run pytest with coverage
pytest ${1:-tests/} ${2:--v} \
  --cov=core \
  --cov=bizra_kernel \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --tb=short

# Save coverage data
coverage report > coverage-report.txt
```

### 3. Critical Test Categories

**MUST PASS** (Fail-Closed):
- `pytest tests/test_agent_factory.py` - Agent spawning
- `pytest tests/test_sape.py` - SAPE probe logic
- `pytest tests/test_fate.py` - FATE escalation
- `pytest tests/test_synapse_security.py` - TLS security

**SHOULD PASS** (Non-blocking):
- `pytest tests/test_warm_pools.py` - Warm pool optimization
- `pytest tests/test_kg_receipts.py` - Knowledge graph receipts

### 4. Integration Tests
```bash
# Test full stack if services running
if docker compose ps | grep -q "Up"; then
    pytest tests/integration/ -v
else
    echo "⚠️ Skipping integration tests (services not running)"
fi
```

### 5. Type Checking (if mypy available)
```bash
if command -v mypy &> /dev/null; then
    mypy core/ --ignore-missing-imports --pretty
fi
```

## Coverage Requirements

**Minimum coverage thresholds**:
- core/: 70%
- bizra_kernel/: 60%
- Overall: 65%

**Critical modules** (must be >80%):
- core/sape.py
- core/fate.py
- core/agent_factory.py
- core/synapse.py

## BIZRA-Specific Validation

### Agent Factory Tests
- Verify PAT agent spawning (all 7 agents)
- Verify SAT agent spawning (all 5 agents)
- Test warm pool optimization (90% time reduction)
- Check resource allocation (URP integration)

### SAPE Tests
- Test 9-probe system
- Verify pattern elevation (>3 repetitions)
- Check Neo4j graph evidence integration

### FATE Tests
- Test escalation levels
- Verify Redis persistence (rediss:// TLS)
- Check quarantine and rejection receipts

### Synapse Security Tests
- Verify TLS URL detection (rediss://)
- Check certificate validation
- Test password authentication

## Fail-Closed Requirements

**BLOCK** if:
- Import errors in core modules
- Critical tests fail (agent factory, SAPE, FATE)
- Coverage <60% overall
- Type errors in critical modules

**WARN** but allow:
- Integration tests fail (services may not be running)
- Coverage slightly below threshold (but >50%)
- Non-critical test failures

## Evidence Generation

Create test receipt:
```json
{
  "receipt_id": "pytest-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "test_summary": {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "skipped": 0
  },
  "coverage": {
    "overall": 0.0,
    "core": 0.0,
    "bizra_kernel": 0.0
  },
  "critical_tests": {
    "agent_factory": "pass|fail",
    "sape": "pass|fail",
    "fate": "pass|fail",
    "synapse_security": "pass|fail"
  },
  "execution_time": "0s"
}
```

Save to: `docs/evidence/receipts/pytest-$(date +%Y%m%d-%H%M%S).json`

## Report Format

```
## Python Test Results

**Status**: ✅ PASS | ❌ FAIL
**Total Tests**: X passed, Y failed, Z skipped
**Coverage**: X.X% (threshold: 65%)
**Execution Time**: Xs

### Coverage by Module
- core/: X%
- bizra_kernel/: X%

### Critical Tests (MUST PASS)
- [ ] Agent Factory: PASS
- [ ] SAPE Logic: PASS
- [ ] FATE Engine: PASS
- [ ] Synapse Security: PASS

### Failed Tests (if any)
- test_name: failure_reason

### Type Checking
- mypy: X errors, Y warnings

### Reports
- HTML Coverage: htmlcov/index.html
- Receipt: docs/evidence/receipts/pytest-YYYYMMDD-HHMMSS.json
```
